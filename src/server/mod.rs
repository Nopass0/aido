//! OpenAI-compatible HTTP API over a multi-model registry ([`crate::models::Models`]).
//!
//! Any number of models of any kind are registered at start-up and selected
//! per request by the `model` field (empty → first model of that kind):
//!
//! | Kind | Endpoints |
//! |------|-----------|
//! | `llm` | `POST /v1/chat/completions`, `POST /v1/completions` (SSE token streaming with `"stream": true`) |
//! | `decision` | `POST /v1/decide` (text + images + audio → typed answers) |
//! | `upscale` | `POST /v1/images/upscale` (PNG/JPEG/WebP/BMP… in, PNG/JPEG/WebP/BMP/raw out) |
//! | `diffusion` | `POST /v1/images/generations` (SSE previews of every step with `"stream": true`) |
//! | all | `GET /v1/models`, `GET /health` |
//!
//! Memory sharing, lazy loading and LRU eviction are described in
//! [`crate::models`]. Full request/response reference: `docs/API.md`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::diffusion::pipeline::{Device, Image, ImageParams};
use crate::llm::chat::Message;
use crate::llm::engine::{Finish, GenParams, GenStats};
use crate::llm::sampler::SamplerConfig;
use crate::models::{Models, ModelsConfig};

pub use crate::models::{detect_kind, Kind};

/// Server configuration.
pub struct ServerConfig {
    /// Models as `(name, path)`.
    pub models: Vec<(String, String)>,
    /// TAESD directory for diffusion models.
    pub taesd: Option<String>,
    /// Bind host.
    pub host: String,
    /// Port.
    pub port: u16,
    /// LLM context length.
    pub ctx: usize,
    /// Threads of the shared pool.
    pub threads: Option<usize>,
    /// Diffusion device.
    pub device: Device,
    /// Seconds before idle models are unloaded.
    pub keep_alive: u64,
    /// RAM budget for loaded models (GB).
    pub ram_gb: f32,
    /// VRAM budget per model (GB).
    pub vram_gb: f32,
    /// Load every model at start instead of on first request.
    pub preload: bool,
    /// Long-term memory directory (`None` = disabled).
    pub memory_dir: Option<String>,
}

type App = Arc<Models>;

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn err(code: StatusCode, msg: impl ToString) -> Response {
    (code, Json(json!({"error": {"message": msg.to_string(), "type": "invalid_request_error"}}))).into_response()
}

fn reply(r: std::result::Result<Result<Value>, tokio::task::JoinError>) -> Response {
    match r {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

fn sse(rx: tokio::sync::mpsc::UnboundedReceiver<Value>) -> Response {
    let stream = UnboundedReceiverStream::new(rx).map(|v| Ok::<Event, Infallible>(Event::default().data(v.to_string()))).chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

/// Builds the router for `models` (usable to mount the API inside another
/// axum application).
///
/// # Example
/// ```no_run
/// let models = std::sync::Arc::new(aido::models::Models::new(Default::default()).unwrap());
/// let app: axum::Router = aido::server::router(models);
/// ```
pub fn router(models: Arc<Models>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/memory", get(memory_get).post(memory_post).delete(memory_delete))
        .route("/v1/decide", post(decide))
        .route("/v1/predict", post(predict))
        .route("/v1/images/upscale", post(upscale))
        .route("/v1/images/generations", post(images))
        .layer(DefaultBodyLimit::max(256 << 20))
        .with_state(models)
}

/// Starts the server (blocks).
///
/// # Example
/// ```no_run
/// use aido::server::{serve, ServerConfig};
/// serve(ServerConfig { models: vec![("qwen".into(), "models/Qwen3-0.6B-Q8_0.gguf".into())], taesd: None, host: "127.0.0.1".into(), port: 8080,
///     ctx: 4096, threads: None, device: aido::diffusion::pipeline::Device::Auto, keep_alive: 600, ram_gb: 12.0, vram_gb: 4.0, preload: false, memory_dir: Some("data/memory".into()) }).unwrap();
/// ```
pub fn serve(cfg: ServerConfig) -> Result<()> {
    let models = Arc::new(Models::new(ModelsConfig { threads: cfg.threads, gpu: true, ram_gb: cfg.ram_gb, vram_gb: cfg.vram_gb, ctx: cfg.ctx, taesd: cfg.taesd.clone(), device: cfg.device, memory_dir: cfg.memory_dir.clone().map(Into::into) })?);
    for (n, p) in &cfg.models {
        let k = models.add(n, p)?;
        eprintln!("[server] model {n:<20} {:<9} {p}", k.name());
    }
    if cfg.preload {
        for (n, _) in &cfg.models {
            models.preload(n)?;
        }
    }
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()?;
    rt.block_on(async move {
        let m2 = models.clone();
        let ka = cfg.keep_alive;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let m3 = m2.clone();
                if let Ok(names) = tokio::task::spawn_blocking(move || m3.trim(Duration::from_secs(ka))).await {
                    for n in names {
                        eprintln!("[server] unloaded idle model {n}");
                    }
                }
            }
        });
        let addr = format!("{}:{}", cfg.host, cfg.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("[server] listening on http://{addr}/v1");
        axum::serve(listener, router(models)).await?;
        Ok::<(), anyhow::Error>(())
    })
}

async fn list_models(State(m): State<App>) -> Json<Value> {
    Json(json!({"object": "list", "data": m.list()}))
}

/// Body of the completion endpoints.
#[derive(Deserialize)]
struct CompletionReq {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<InMsg>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    auto_summarize: Option<bool>,
    #[serde(default)]
    memory: Option<bool>,
}

#[derive(Deserialize)]
struct InMsg {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Option<Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|p| p.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join(""),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

fn gen_params(r: &CompletionReq) -> GenParams {
    let stop = match &r.stop {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => vec![],
    };
    GenParams {
        max_tokens: r.max_completion_tokens.or(r.max_tokens).unwrap_or(512),
        sampler: SamplerConfig {
            temperature: r.temperature.unwrap_or(0.7),
            top_p: r.top_p.unwrap_or(0.95),
            top_k: r.top_k.unwrap_or(40),
            seed: r.seed.unwrap_or_else(now),
            repeat_penalty: 1.0 + r.frequency_penalty.unwrap_or(0.0).max(0.0) * 0.2,
            ..Default::default()
        },
        stop,
        spec: None,
    }
}

fn usage(st: &GenStats) -> Value {
    json!({"prompt_tokens": st.prompt_tokens, "completion_tokens": st.gen_tokens, "total_tokens": st.prompt_tokens + st.gen_tokens,
        "prompt_tokens_details": {"cached_tokens": st.cached_tokens},
        "aido": {"tokens_per_s": st.tokens_per_s(), "prefill_tokens_per_s": st.prefill_tps(), "cached_prompt_tokens": st.cached_tokens, "drafted": st.drafted, "accepted": st.accepted}})
}

async fn chat(State(m): State<App>, Json(req): Json<CompletionReq>) -> Response {
    use crate::llm::conversation::Piece;
    let mut msgs: Vec<Message> = req
        .messages
        .iter()
        .map(|x| Message { role: x.role.clone(), content: content_text(&x.content), tool_calls: x.tool_calls.clone(), tool_call_id: x.tool_call_id.clone(), name: x.name.clone() })
        .collect();
    if msgs.is_empty() {
        return err(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    // JSON mode: instruct the model, then extract the first JSON value from the answer.
    let json_mode = req.response_format.as_ref().and_then(|f| f["type"].as_str()).map(|t| t.starts_with("json")).unwrap_or(false);
    if json_mode {
        msgs.insert(0, Message::new("system", "Reply with a single valid JSON object only, no prose, no code fences."));
    }
    let tools = match req.tool_choice.as_ref().and_then(|t| t.as_str()) {
        Some("none") => None,
        _ => req.tools.clone().filter(|t| t.as_array().map(|a| !a.is_empty()).unwrap_or(false)),
    };
    let opts = crate::models::ChatOptions {
        params: gen_params(&req),
        think: req.enable_thinking.unwrap_or(false) || req.reasoning_effort.as_deref().map(|r| r != "none").unwrap_or(false),
        tools,
        n: req.n.unwrap_or(1).clamp(1, 8),
        user: req.user.clone().filter(|_| req.memory.unwrap_or(true)),
        auto_summarize: req.auto_summarize.unwrap_or(true),
        ..Default::default()
    };
    let model = req.model.clone().unwrap_or_default();
    let id = format!("chatcmpl-{}", now());
    let created = now();
    let choice_json = |i: usize, c: &crate::models::ChatChoice, json_mode: bool| {
        let mut content = c.content.clone();
        if json_mode {
            if let Some(v) = extract_json(&content) {
                content = v.to_string();
            }
        }
        let mut msg = json!({"role": "assistant", "content": if c.tool_calls.is_empty() || !content.is_empty() { json!(content) } else { Value::Null }});
        if let Some(r) = &c.reasoning {
            msg["reasoning_content"] = json!(r);
        }
        if !c.tool_calls.is_empty() {
            msg["tool_calls"] = Value::Array(c.tool_calls.iter().map(|t| t.openai()).collect());
        }
        json!({"index": i, "message": msg, "finish_reason": c.finish_reason})
    };
    let usage_of = |r: &crate::models::ChatResult| {
        let gen: usize = r.choices.iter().map(|c| c.stats.gen_tokens).sum();
        let mut u = usage(&r.choices[0].stats);
        u["prompt_tokens"] = json!(r.prompt_tokens);
        u["completion_tokens"] = json!(gen);
        u["total_tokens"] = json!(r.prompt_tokens + gen);
        u["aido"]["summarized_messages"] = json!(r.summarized);
        u["aido"]["memories_used"] = json!(r.memories_used);
        u
    };
    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
        tokio::task::spawn_blocking(move || {
            let shown = if model.is_empty() { m.default_of(Kind::Llm).unwrap_or_default() } else { model.clone() };
            let chunk = |i: usize, delta: Value, fin: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": shown, "choices": [{"index": i, "delta": delta, "finish_reason": fin}]});
            let mut started = vec![false; opts.n];
            let res = m.chat_full(&model, &msgs, &opts, &mut |i, piece| {
                if !started[i] {
                    started[i] = true;
                    let _ = tx.send(chunk(i, json!({"role": "assistant", "content": ""}), Value::Null));
                }
                let d = match piece {
                    Piece::Content(c) => json!({"content": c}),
                    Piece::Reasoning(r) => json!({"reasoning_content": r}),
                };
                tx.send(chunk(i, d, Value::Null)).is_ok()
            });
            match res {
                Ok(r) => {
                    for (i, c) in r.choices.iter().enumerate() {
                        if !c.tool_calls.is_empty() {
                            let calls: Vec<Value> = c.tool_calls.iter().enumerate().map(|(k, t)| {
                                let mut v = t.openai();
                                v["index"] = json!(k);
                                v
                            }).collect();
                            let _ = tx.send(chunk(i, json!({"tool_calls": calls}), Value::Null));
                        }
                        let mut last = chunk(i, json!({}), json!(c.finish_reason));
                        if i + 1 == r.choices.len() {
                            last["usage"] = usage_of(&r);
                        }
                        let _ = tx.send(last);
                    }
                }
                Err(e) => {
                    let _ = tx.send(json!({"error": {"message": format!("{e:#}")}}));
                }
            }
        });
        return sse(rx);
    }
    reply(
        tokio::task::spawn_blocking(move || -> Result<Value> {
            let r = m.chat_full(&model, &msgs, &opts, &mut |_, _| true)?;
            let choices: Vec<Value> = r.choices.iter().enumerate().map(|(i, c)| choice_json(i, c, json_mode)).collect();
            Ok(json!({"id": id, "object": "chat.completion", "created": created, "model": r.model, "choices": choices, "usage": usage_of(&r)}))
        })
        .await,
    )
}

/// `GET /v1/memory?user=ID` — list; `POST /v1/memory {"user","text"}` — add;
/// `DELETE /v1/memory {"user","id"?}` — delete one or all.
async fn memory_get(State(m): State<App>, axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let Some(store) = m.memory() else { return err(StatusCode::NOT_FOUND, "long-term memory is disabled (start with --memory-dir)") };
    let user = q.get("user").cloned().unwrap_or_else(|| "default".into());
    let res = match q.get("query") {
        Some(qq) => store.search(&user, qq, q.get("k").and_then(|k| k.parse().ok()).unwrap_or(5)),
        None => store.list(&user),
    };
    match res {
        Ok(items) => Json(json!({"object": "list", "user": user, "data": items})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn memory_post(State(m): State<App>, Json(b): Json<Value>) -> Response {
    let Some(store) = m.memory() else { return err(StatusCode::NOT_FOUND, "long-term memory is disabled (start with --memory-dir)") };
    let (user, text) = (b["user"].as_str().unwrap_or("default"), b["text"].as_str().unwrap_or(""));
    if text.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "\"text\" required");
    }
    match store.add(user, text) {
        Ok(it) => Json(json!(it)).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn memory_delete(State(m): State<App>, Json(b): Json<Value>) -> Response {
    let Some(store) = m.memory() else { return err(StatusCode::NOT_FOUND, "long-term memory is disabled (start with --memory-dir)") };
    match store.delete(b["user"].as_str().unwrap_or("default"), b["id"].as_u64()) {
        Ok(n) => Json(json!({"deleted": n})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn completions(State(m): State<App>, Json(req): Json<CompletionReq>) -> Response {
    let prompt = req.prompt.as_ref().map(content_text).unwrap_or_default();
    run_completion(m, req, move |e| Ok(e.tok.encode(&prompt, true)), false, false).await
}

/// Extracts the first JSON value from model output (skips fences / prose).
///
/// # Example
/// (private) `extract_json("```json\n{\"a\":1}\n```")` → `{"a":1}`.
fn extract_json(s: &str) -> Option<Value> {
    let start = s.find(['{', '['])?;
    let mut de = serde_json::Deserializer::from_str(&s[start..]).into_iter::<Value>();
    de.next()?.ok()
}

async fn run_completion<F>(m: App, req: CompletionReq, build: F, chat: bool, json_mode: bool) -> Response
where
    F: FnOnce(&crate::llm::engine::LlmEngine) -> Result<Vec<u32>> + Send + 'static,
{
    let params = gen_params(&req);
    let model = req.model.clone().unwrap_or_default();
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, now());
    let created = now();
    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
        tokio::task::spawn_blocking(move || {
            let shown = if model.is_empty() { m.default_of(Kind::Llm).unwrap_or_default() } else { model.clone() };
            let chunk = |delta: Value, fin: Value| {
                if chat {
                    json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": shown, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
                } else {
                    json!({"id": id, "object": "text_completion", "created": created, "model": shown, "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": fin}]})
                }
            };
            if chat {
                let _ = tx.send(chunk(json!({"role": "assistant", "content": ""}), Value::Null));
            }
            let res = m.generate(&model, build, &params, &mut |piece| tx.send(chunk(json!({"content": piece}), Value::Null)).is_ok());
            match res {
                Ok((_, c)) => {
                    let fin = if c.stats.finish == Finish::Length { "length" } else { "stop" };
                    let mut last = chunk(json!({}), json!(fin));
                    last["usage"] = usage(&c.stats);
                    let _ = tx.send(last);
                }
                Err(e) => {
                    let _ = tx.send(json!({"error": {"message": format!("{e:#}")}}));
                }
            }
        });
        return sse(rx);
    }
    let out = tokio::task::spawn_blocking(move || -> Result<Value> {
        let (name, c) = m.generate(&model, build, &params, &mut |_| true)?;
        let fin = if c.stats.finish == Finish::Length { "length" } else { "stop" };
        let mut text = c.text.clone();
        if json_mode {
            if let Some(v) = extract_json(&text) {
                text = v.to_string();
            }
        }
        Ok(if chat {
            json!({"id": id, "object": "chat.completion", "created": created, "model": name, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fin}], "usage": usage(&c.stats)})
        } else {
            json!({"id": id, "object": "text_completion", "created": created, "model": name, "choices": [{"index": 0, "text": text, "finish_reason": fin}], "usage": usage(&c.stats)})
        })
    })
    .await;
    reply(out)
}

/// Decodes base64 (optionally a `data:...;base64,` URL).
fn b64(s: &str) -> Result<Vec<u8>> {
    let s = s.split_once("base64,").map(|x| x.1).unwrap_or(s);
    Ok(base64::engine::general_purpose::STANDARD.decode(s.trim())?)
}

/// Bytes of a media item: `data` / `b64_json` (base64 or data URL),
/// `url` / `image_url.url` (`data:` URL or local path) or `path`.
fn media_bytes(it: &Value) -> Result<Vec<u8>> {
    if let Some(d) = it["data"].as_str().or(it["b64_json"].as_str()) {
        return b64(d);
    }
    if let Some(u) = it["url"].as_str().or(it["image_url"]["url"].as_str()).or(it["image_url"].as_str()) {
        if u.starts_with("data:") {
            return b64(u);
        }
        return Ok(std::fs::read(u.trim_start_matches("file://"))?);
    }
    if let Some(p) = it["path"].as_str() {
        return Ok(std::fs::read(p)?);
    }
    anyhow::bail!("media item needs \"data\" (base64), \"url\" or \"path\"")
}

/// `POST /v1/decide` — `{"model"?, "state", "questions": [...], "media"?: [{"type": "image"|"audio", "data"|"url"|"path"}]}`.
async fn decide(State(m): State<App>, Json(req): Json<Value>) -> Response {
    reply(
        tokio::task::spawn_blocking(move || -> Result<Value> {
            use crate::llm::d1_media::{read_wav, MediaInput};
            let mut media = Vec::new();
            for it in req["media"].as_array().cloned().unwrap_or_default() {
                let bytes = media_bytes(&it)?;
                if matches!(it["type"].as_str(), Some("audio") | Some("input_audio")) || (bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE") {
                    let tmp = std::env::temp_dir().join(format!("aido_{}_{}.wav", std::process::id(), now()));
                    std::fs::write(&tmp, &bytes)?;
                    let w = read_wav(tmp.to_str().unwrap());
                    let _ = std::fs::remove_file(&tmp);
                    media.push(MediaInput::Audio(w?));
                } else {
                    media.push(MediaInput::Image(image::load_from_memory(&bytes).context("unsupported image format")?.to_rgb8()));
                }
            }
            m.decide(req["model"].as_str().unwrap_or(""), &req["state"], &req["questions"], &media)
        })
        .await,
    )
}

/// `POST /v1/predict` — `{"model", "input": {...}}` → `{"model", "output": {...}, "ms"}` for custom SDK models.
async fn predict(State(m): State<App>, Json(req): Json<Value>) -> Response {
    reply(
        tokio::task::spawn_blocking(move || -> Result<Value> {
            let model = req["model"].as_str().unwrap_or("");
            let t = Instant::now();
            let out = m.predict(model, &req["input"])?;
            let name = if model.is_empty() { m.default_of(Kind::Custom).unwrap_or_default() } else { model.to_string() };
            Ok(json!({"object": "prediction", "model": name, "output": out, "ms": t.elapsed().as_secs_f64() * 1e3}))
        })
        .await,
    )
}

/// Encodes `img` as `format` (`png` | `jpeg`/`jpg` | `webp` (lossless) | `bmp` | `raw` RGB24).
///
/// # Example
/// (private) `encode_image(&img, "jpeg", 90)` → `(bytes, "image/jpeg")`.
fn encode_image(img: &image::RgbImage, format: &str, quality: u8) -> Result<(Vec<u8>, &'static str)> {
    let mut out = Vec::new();
    match format {
        "jpeg" | "jpg" => {
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality).encode_image(img)?;
            Ok((out, "image/jpeg"))
        }
        "webp" => {
            image::codecs::webp::WebPEncoder::new_lossless(&mut out).encode(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)?;
            Ok((out, "image/webp"))
        }
        "bmp" => {
            img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Bmp)?;
            Ok((out, "image/bmp"))
        }
        "raw" | "rgb24" => Ok((img.as_raw().clone(), "application/octet-stream")),
        _ => {
            let fast = image::codecs::png::PngEncoder::new_with_quality(&mut out, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::Sub);
            image::ImageEncoder::write_image(fast, img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)?;
            Ok((out, "image/png"))
        }
    }
}

/// `POST /v1/images/upscale` — `{"model"?, "image": base64 | data URL (or "url"/"path"), "loops"?, "output_format"?: "png"|"jpeg"|"webp"|"bmp"|"raw", "response_format"?: "b64_json"|"url", "quality"?}`.
async fn upscale(State(m): State<App>, Json(req): Json<Value>) -> Response {
    reply(
        tokio::task::spawn_blocking(move || -> Result<Value> {
            let bytes = match req["image"].as_str() {
                Some(s) => b64(s)?,
                None => media_bytes(&req)?,
            };
            let img = image::load_from_memory(&bytes).context("unsupported image format")?.to_rgb8();
            let t = Instant::now();
            let model = req["model"].as_str().unwrap_or("");
            let big = m.upscale(model, &img, req["loops"].as_u64().map(|x| x as usize))?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let fmt = req["output_format"].as_str().unwrap_or("png").to_ascii_lowercase();
            let (data, mime) = encode_image(&big, &fmt, req["quality"].as_u64().unwrap_or(92).min(100) as u8)?;
            let enc = base64::engine::general_purpose::STANDARD.encode(&data);
            let item = if req["response_format"].as_str() == Some("url") { json!({"url": format!("data:{mime};base64,{enc}")}) } else { json!({"b64_json": enc}) };
            let name = if model.is_empty() { m.default_of(Kind::Upscale).unwrap_or_default() } else { model.to_string() };
            Ok(json!({"created": now(), "model": name, "width": big.width(), "height": big.height(), "input": {"width": img.width(), "height": img.height()}, "output_format": fmt, "mime": mime, "ms": ms, "data": [item]}))
        })
        .await,
    )
}

/// Body of `/v1/images/generations`.
#[derive(Deserialize)]
struct ImageReq {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
}

fn image_item(img: &Image, fmt: &str, as_url: bool) -> Result<Value> {
    let rgb = image::RgbImage::from_raw(img.width as u32, img.height as u32, img.rgb.clone()).context("image size")?;
    let (data, mime) = encode_image(&rgb, fmt, 90)?;
    let enc = base64::engine::general_purpose::STANDARD.encode(data);
    Ok(if as_url { json!({"url": format!("data:{mime};base64,{enc}")}) } else { json!({"b64_json": enc}) })
}

/// `POST /v1/images/generations`; with `stream: true` every denoising
/// step's preview (JPEG) is sent as an SSE event before the final result.
async fn images(State(m): State<App>, Json(req): Json<ImageReq>) -> Response {
    let (w, h) = req.size.as_deref().and_then(|x| x.split_once('x')).and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))).unwrap_or((512usize, 512usize));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let tx_err = tx.clone();
    let streaming = req.stream;
    let job = tokio::task::spawn_blocking(move || -> Result<Value> {
        let model = req.model.clone().unwrap_or_default();
        let name = if model.is_empty() { m.default_of(Kind::Diffusion).unwrap_or_default() } else { model.clone() };
        let fmt = req.output_format.clone().unwrap_or_else(|| "png".into()).to_ascii_lowercase();
        let as_url = req.response_format.as_deref() == Some("url");
        let mut data = Vec::new();
        let t = Instant::now();
        for i in 0..req.n.unwrap_or(1).clamp(1, 8) {
            let params = ImageParams {
                prompt: req.prompt.clone(),
                negative: req.negative_prompt.clone().unwrap_or_default(),
                steps: req.steps.unwrap_or(1),
                width: (w / 64 * 64).max(64),
                height: (h / 64 * 64).max(64),
                seed: req.seed.unwrap_or_else(now) + i as u64,
                guidance: req.guidance_scale.unwrap_or(1.0),
            };
            let mut cb = |step: usize, total: usize, preview: &Image| {
                if let Ok(item) = image_item(preview, "jpeg", false) {
                    let _ = tx.send(json!({"type": "preview", "index": i, "step": step, "steps": total, "width": preview.width, "height": preview.height, "b64_json": item["b64_json"]}));
                }
            };
            let img = m.image(&model, &params, if streaming { Some(&mut cb) } else { None })?;
            data.push(image_item(&img, &fmt, as_url)?);
        }
        let out = json!({"created": now(), "model": name, "output_format": fmt, "ms": t.elapsed().as_secs_f64() * 1e3, "data": data, "timings": m.image_timings(&name)});
        if streaming {
            let _ = tx.send(json!({"type": "final", "result": out.clone()}));
        }
        Ok(out)
    });
    if streaming {
        tokio::spawn(async move {
            if let Ok(Err(e)) = job.await {
                let _ = tx_err.send(json!({"error": {"message": format!("{e:#}")}}));
            }
        });
        return sse(rx);
    }
    reply(job.await)
}
