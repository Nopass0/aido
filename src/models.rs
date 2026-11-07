//! # Embedding API — run any model from your own Rust application
//!
//! [`Models`] is the one type an application needs: register models of any
//! kind under names, then call them. It is the same registry the HTTP
//! server uses, so everything the server does is available in-process with
//! zero serialisation overhead.
//!
//! ```no_run
//! use aido::models::{Models, ModelsConfig};
//! use aido::llm::chat::Message;
//!
//! let models = Models::new(ModelsConfig::default())?;
//! models.add("qwen", "models/Qwen3-0.6B-Q8_0.gguf")?;          // text
//! models.add("d1", "models/d1-omni")?;                           // decisions (text/image/audio)
//! models.add("x2", "models/baikal/Baikal_LoopSR_x2_v2.safetensors")?; // upscaler
//!
//! // Chat with token streaming.
//! let out = models.chat("qwen", &[Message::new("user", "Hi!")], &Default::default(), &mut |piece| {
//!     print!("{piece}");
//!     true // return false to stop
//! })?;
//! println!("\n{:.1} tok/s", out.stats.tokens_per_s());
//!
//! // Upscale a picture.
//! let img = image::open("frame.png")?.to_rgb8();
//! let big = models.upscale("x2", &img, None)?;
//! big.save("frame_x2.png")?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## How several models share one machine
//! * **Lazy loading** — a model is loaded on its first call.
//! * **Shared compute** — one pinned thread pool and one GPU device serve
//!   every model; nothing is duplicated per model.
//! * **Shared weights** — weights are memory-mapped: two names (or two
//!   processes) pointing at the same file share the same physical pages, and
//!   the same file registered twice is loaded once.
//! * **Budgeted residency** — when loading would exceed
//!   [`ModelsConfig::ram_gb`] the least-recently-used models are unloaded;
//!   [`Models::trim`] unloads models idle for too long.
//! * **Serialised per model** — calls to the same model run one at a time
//!   (two concurrent passes would just split the same memory bandwidth);
//!   different models may run concurrently.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::diffusion::pipeline::{Device, Image, ImageParams, Pipeline};
use crate::llm::chat::Message;
use crate::llm::engine::{EngineOptions, GenParams, GenStats, LlmEngine};
use crate::runtime::pool::ThreadPool;

/// Kind of a registered model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Text generation (GGUF, `.aido`, MoE containers).
    Llm,
    /// d1-style decision model (typed answers over text, images and audio).
    Decision,
    /// Super-resolution (LoopSR family).
    Upscale,
    /// Text-to-image diffusion (SD family).
    Diffusion,
    /// Custom SDK model with an I/O signature (see [`crate::sdk::io`]).
    Custom,
}

impl Kind {
    /// Lower-case name.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::models::Kind::Upscale.name(), "upscale");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Kind::Llm => "llm",
            Kind::Decision => "decision",
            Kind::Upscale => "upscale",
            Kind::Diffusion => "diffusion",
            Kind::Custom => "custom",
        }
    }
}

/// Detects the kind of the model stored at `path` from its files.
///
/// * directory with `unet/` → [`Kind::Diffusion`];
/// * directory with `qwen36_meta.json` (MoE container) → [`Kind::Llm`];
/// * directory whose `config.json` names `D1Model` → [`Kind::Decision`];
/// * `.safetensors` whose header mentions `LoopSR` → [`Kind::Upscale`];
/// * anything else (`.gguf`, `.aido`) → [`Kind::Llm`].
///
/// # Example
/// ```no_run
/// use aido::models::{detect_kind, Kind};
/// assert_eq!(detect_kind("models/Qwen3-0.6B-Q8_0.gguf").unwrap(), Kind::Llm);
/// ```
pub fn detect_kind(path: &str) -> Result<Kind> {
    let p = std::path::Path::new(path);
    anyhow::ensure!(p.exists(), "{path}: not found");
    if p.is_dir() {
        if p.join("unet").exists() {
            return Ok(Kind::Diffusion);
        }
        if p.join("qwen36_meta.json").exists() {
            return Ok(Kind::Llm);
        }
        let cfg = std::fs::read_to_string(p.join("config.json")).unwrap_or_default();
        if cfg.contains("D1Model") || cfg.contains("lfm3") {
            return Ok(Kind::Decision);
        }
        anyhow::bail!("{path}: unknown model directory");
    }
    if path.ends_with(".safetensors") {
        use std::io::Read;
        let mut f = std::fs::File::open(p)?;
        let mut head = [0u8; 8];
        f.read_exact(&mut head)?;
        let n = (u64::from_le_bytes(head) as usize).min(1 << 24);
        let mut h = vec![0u8; n];
        f.read_exact(&mut h)?;
        if String::from_utf8_lossy(&h).contains("LoopSR") {
            return Ok(Kind::Upscale);
        }
        anyhow::bail!("{path}: unknown safetensors model");
    }
    if path.ends_with(".aido") {
        if let Ok(f) = crate::format::AidoFile::open(path) {
            use crate::format::TensorStore;
            if f.get_str("general.architecture") == Some("aido-sdk") {
                anyhow::ensure!(f.get_str(crate::sdk::io::META_KEY).is_some(), "{path}: SDK model without an I/O signature (attach one with aido::sdk::io::attach)");
                return Ok(Kind::Custom);
            }
        }
    }
    Ok(Kind::Llm)
}

/// A loaded model instance.
#[derive(Clone)]
enum Loaded {
    Llm(Arc<Mutex<LlmEngine>>),
    Decision(Arc<crate::llm::d1::D1>),
    Upscale(Arc<Mutex<crate::upscale::LoopSr>>),
    Diffusion(Arc<Mutex<Pipeline>>),
    Custom(Arc<Mutex<crate::sdk::io::SdkModel>>),
}

struct Entry {
    name: String,
    kind: Kind,
    path: String,
    loaded: Option<Loaded>,
    last_used: Instant,
    bytes: u64,
    busy: Arc<Mutex<()>>,
}

/// Configuration of a [`Models`] registry.
#[derive(Debug, Clone)]
pub struct ModelsConfig {
    /// Worker threads (default: physical cores).
    pub threads: Option<usize>,
    /// Use the GPU when available.
    pub gpu: bool,
    /// RAM budget for simultaneously loaded models (GB).
    pub ram_gb: f32,
    /// VRAM budget shared by all loaded models (GB). Each model gets what
    /// is left; a model that does not fit runs on the CPU path.
    pub vram_gb: f32,
    /// LLM context length.
    pub ctx: usize,
    /// TAESD directory for fast diffusion decoding.
    pub taesd: Option<String>,
    /// Diffusion device.
    pub device: Device,
    /// Directory of the long-term memory store (`None` = memory disabled).
    pub memory_dir: Option<std::path::PathBuf>,
}

impl Default for ModelsConfig {
    /// All cores, GPU on, 12 GB RAM / 4 GB VRAM budgets, 4096 context.
    fn default() -> Self {
        ModelsConfig { threads: None, gpu: true, ram_gb: 12.0, vram_gb: 5.0, ctx: 4096, taesd: None, device: Device::Auto, memory_dir: None }
    }
}

/// Options of [`Models::chat_full`].
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// Sampling, length, stop strings.
    pub params: GenParams,
    /// Let reasoning models think (`<think>`; returned separately).
    pub think: bool,
    /// OpenAI-style tool definitions (`[{"type":"function","function":{…}}]`).
    pub tools: Option<Value>,
    /// Number of alternative answers (each with seed `seed + i`; the
    /// prompt is evaluated once thanks to the prefix cache).
    pub n: usize,
    /// Long-term memory id (e.g. the OpenAI `user` field); `None` = no memory.
    pub user: Option<String>,
    /// Memories retrieved per turn.
    pub memory_k: usize,
    /// Store the exchange in long-term memory after the turn.
    pub remember: bool,
    /// Summarise the oldest history with the same model when the prompt
    /// does not fit the context window.
    pub auto_summarize: bool,
    /// Offer the model the built-in `memory_search` / `memory_save` tools
    /// (requires `user` and an enabled memory store).
    pub memory_tools: bool,
}

impl Default for ChatOptions {
    /// One answer, no tools, thinking off, memory on when `user` is set, auto-summary on.
    fn default() -> Self {
        ChatOptions { params: GenParams::default(), think: false, tools: None, n: 1, user: None, memory_k: 5, remember: true, auto_summarize: true, memory_tools: true }
    }
}

/// One answer of [`Models::chat_full`].
#[derive(Debug, Clone)]
pub struct ChatChoice {
    /// Visible answer.
    pub content: String,
    /// Separated reasoning (`<think>` text), if any.
    pub reasoning: Option<String>,
    /// Structured tool calls, if the model called tools.
    pub tool_calls: Vec<crate::llm::conversation::ToolCall>,
    /// `stop` | `length` | `tool_calls`.
    pub finish_reason: String,
    /// Speed statistics.
    pub stats: GenStats,
}

/// Result of [`Models::chat_full`].
#[derive(Debug, Clone)]
pub struct ChatResult {
    /// Resolved model name.
    pub model: String,
    /// Answers (`n` of them).
    pub choices: Vec<ChatChoice>,
    /// History messages folded into a summary to fit the context.
    pub summarized: usize,
    /// Long-term memories given to the model.
    pub memories_used: usize,
    /// Prompt length in tokens (after memory and summarisation).
    pub prompt_tokens: usize,
}

/// Result of a text generation.
pub struct Completion {
    /// Generated text.
    pub text: String,
    /// Speed and token statistics.
    pub stats: GenStats,
}

/// Registry of named models of any kind (see the [module docs](self)).
pub struct Models {
    entries: Mutex<Vec<Entry>>,
    load_lock: Mutex<()>,
    pool: Arc<ThreadPool>,
    gpu: Option<Arc<crate::gpu::Gpu>>,
    cfg: ModelsConfig,
    memory: Option<crate::memory::MemoryStore>,
}

fn path_bytes(p: &std::path::Path) -> u64 {
    if p.is_dir() {
        std::fs::read_dir(p).map(|r| r.flatten().map(|e| path_bytes(&e.path())).sum()).unwrap_or(0)
    } else {
        std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
    }
}

impl Models {
    /// Creates an empty registry with a shared thread pool and GPU.
    ///
    /// # Example
    /// ```no_run
    /// let models = aido::models::Models::new(Default::default()).unwrap();
    /// ```
    pub fn new(cfg: ModelsConfig) -> Result<Self> {
        let cpu = crate::runtime::sys::CpuInfo::detect();
        let pool = Arc::new(ThreadPool::new(cfg.threads.unwrap_or(cpu.physical), Some(cpu.physical_core_ids())));
        let gpu = if cfg.gpu && std::env::var("AIDO_NO_GPU").is_err() { crate::gpu::Gpu::new().ok().map(Arc::new) } else { None };
        let memory = cfg.memory_dir.as_ref().map(crate::memory::MemoryStore::new).transpose()?;
        Ok(Models { entries: Mutex::new(Vec::new()), load_lock: Mutex::new(()), pool, gpu, cfg, memory })
    }

    /// Registers `path` under `name` (nothing is loaded yet) and returns
    /// its detected kind. Registering an existing name replaces it.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let kind = models.add("qwen", "models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// assert_eq!(kind, aido::models::Kind::Llm);
    /// ```
    pub fn add(&self, name: &str, path: &str) -> Result<Kind> {
        let kind = detect_kind(path)?;
        let mut es = self.entries.lock().unwrap();
        es.retain(|e| e.name != name);
        es.push(Entry { name: name.into(), kind, path: path.into(), loaded: None, last_used: Instant::now(), bytes: path_bytes(std::path::Path::new(path)), busy: Arc::new(Mutex::new(())) });
        Ok(kind)
    }

    /// The shared thread pool.
    pub fn pool(&self) -> &Arc<ThreadPool> {
        &self.pool
    }

    /// Lists registered models as JSON objects
    /// `{id, object, owned_by, kind, path, loaded, bytes}`.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// for m in models.list() { println!("{} {}", m["id"], m["kind"]); }
    /// ```
    pub fn list(&self) -> Vec<Value> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(|e| {
                let mut v = json!({"id": e.name, "object": "model", "owned_by": "aido", "kind": e.kind.name(), "path": e.path, "loaded": e.loaded.is_some(), "bytes": e.bytes});
                if e.kind == Kind::Custom {
                    if let Ok(Some(sig)) = crate::sdk::io::Signature::read(&e.path) {
                        v["io"] = serde_json::to_value(sig).unwrap_or(Value::Null);
                    }
                }
                v
            })
            .collect()
    }

    /// Runs a custom SDK model on a JSON input (validated against its
    /// signature); returns the JSON output.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// models.add("seq", "models/seq.aido").unwrap();
    /// let out = models.predict("seq", &serde_json::json!({"sequence": [3, 6, 9, 12]})).unwrap();
    /// assert_eq!(out["next"], 15);
    /// ```
    pub fn predict(&self, model: &str, input: &Value) -> Result<Value> {
        let (_, l, busy) = self.get(model, Kind::Custom)?;
        let Loaded::Custom(m) = l else { unreachable!() };
        let _b = busy.lock().unwrap();
        let mut m = m.lock().unwrap();
        m.predict(&self.pool, input)
    }

    /// Name of the first registered model of `kind`.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let llm = models.default_of(aido::models::Kind::Llm);
    /// ```
    pub fn default_of(&self, kind: Kind) -> Option<String> {
        self.entries.lock().unwrap().iter().find(|e| e.kind == kind).map(|e| e.name.clone())
    }

    /// VRAM (bytes) still free in the shared budget (0.3 GB kept as margin for KV growth).
    fn vram_left(&self) -> usize {
        let used = self.gpu.as_ref().map(|g| g.vram_used.load(std::sync::atomic::Ordering::Relaxed)).unwrap_or(0) as f64;
        (self.cfg.vram_gb as f64 * 1e9 - used - 0.3e9).max(0.0) as usize
    }

    fn load(&self, kind: Kind, path: &str) -> Result<Loaded> {
        let t = Instant::now();
        let l = match kind {
            Kind::Llm => {
                let opts = EngineOptions { ctx: self.cfg.ctx, vram_gb: self.vram_left() as f32 / 1e9, gpu: self.gpu.is_some(), gpu_device: self.gpu.clone(), threads: self.cfg.threads, ..Default::default() };
                Loaded::Llm(Arc::new(Mutex::new(LlmEngine::load(path, opts)?)))
            }
            Kind::Decision => {
                let acc = crate::llm::lfm3::Accel::new(self.gpu.clone(), self.vram_left());
                Loaded::Decision(Arc::new(crate::llm::d1::D1::load_with(path, &self.pool, &acc)?))
            }
            Kind::Upscale => Loaded::Upscale(Arc::new(Mutex::new(crate::upscale::LoopSr::load(path, self.gpu.clone())?))),
            Kind::Custom => Loaded::Custom(Arc::new(Mutex::new(crate::sdk::io::SdkModel::load(path)?))),
            Kind::Diffusion => Loaded::Diffusion(Arc::new(Mutex::new(Pipeline::load(path, self.cfg.taesd.as_deref(), self.cfg.device, self.cfg.threads)?))),
        };
        eprintln!("[models] loaded {path} ({}) in {:.1}s", kind.name(), t.elapsed().as_secs_f64());
        Ok(l)
    }

    /// Resolves `name` (empty → first model of `kind`), loads it if needed
    /// (evicting LRU models over the RAM budget) and returns the instance
    /// and its per-model lock.
    fn get(&self, name: &str, kind: Kind) -> Result<(String, Loaded, Arc<Mutex<()>>)> {
        let name = if name.is_empty() { self.default_of(kind).with_context(|| format!("no {} model registered", kind.name()))? } else { name.to_string() };
        let _l = self.load_lock.lock().unwrap();
        let (path, need, busy) = {
            let mut es = self.entries.lock().unwrap();
            let e = es.iter_mut().find(|e| e.name == name).with_context(|| format!("model '{name}' is not registered"))?;
            anyhow::ensure!(e.kind == kind, "model '{name}' is a {} model, not {}", e.kind.name(), kind.name());
            e.last_used = Instant::now();
            if let Some(l) = &e.loaded {
                return Ok((name, l.clone(), e.busy.clone()));
            }
            (e.path.clone(), e.bytes, e.busy.clone())
        };
        // Same file under another name already loaded → share the instance.
        let shared = self.entries.lock().unwrap().iter().find(|e| e.path == path && e.loaded.is_some()).and_then(|e| e.loaded.clone());
        let l = match shared {
            Some(l) => l,
            None => {
                {
                    let mut es = self.entries.lock().unwrap();
                    loop {
                        let used: u64 = es.iter().filter(|e| e.loaded.is_some()).map(|e| e.bytes).sum();
                        if used + need <= (self.cfg.ram_gb as f64 * 1e9) as u64 {
                            break;
                        }
                        match es.iter_mut().filter(|e| e.loaded.is_some()).min_by_key(|e| e.last_used) {
                            Some(v) => {
                                eprintln!("[models] unloading {} (LRU) to fit {name}", v.name);
                                v.loaded = None;
                            }
                            None => break,
                        }
                    }
                }
                self.load(kind, &path)?
            }
        };
        let mut es = self.entries.lock().unwrap();
        if let Some(e) = es.iter_mut().find(|e| e.name == name) {
            e.loaded = Some(l.clone());
        }
        Ok((name, l, busy))
    }

    /// Loads `name` now (otherwise it is loaded on first use).
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// models.add("qwen", "models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// models.preload("qwen").unwrap();
    /// ```
    pub fn preload(&self, name: &str) -> Result<()> {
        let kind = self.entries.lock().unwrap().iter().find(|e| e.name == name).map(|e| e.kind).with_context(|| format!("model '{name}' is not registered"))?;
        self.get(name, kind).map(|_| ())
    }

    /// Unloads `name` (it stays registered). Returns whether it was loaded.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// models.unload("qwen");
    /// ```
    pub fn unload(&self, name: &str) -> bool {
        let mut es = self.entries.lock().unwrap();
        es.iter_mut().find(|e| e.name == name).map(|e| e.loaded.take().is_some()).unwrap_or(false)
    }

    /// Unloads models idle for longer than `idle`; returns their names.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// models.trim(std::time::Duration::from_secs(600));
    /// ```
    pub fn trim(&self, idle: Duration) -> Vec<String> {
        let mut out = Vec::new();
        for e in self.entries.lock().unwrap().iter_mut() {
            if e.loaded.is_some() && e.last_used.elapsed() > idle {
                e.loaded = None;
                out.push(e.name.clone());
            }
        }
        out
    }

    /// Generates from raw prompt token ids built by `build` (given the
    /// engine, e.g. to apply its chat template). Returns the resolved model
    /// name and the completion. `on_text` receives every decoded piece and
    /// may return `false` to stop.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let (_name, out) = models.generate("qwen", |e| Ok(e.tok.encode("Once upon a time", true)), &Default::default(), &mut |_| true).unwrap();
    /// ```
    pub fn generate(&self, model: &str, build: impl FnOnce(&LlmEngine) -> Result<Vec<u32>>, params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<(String, Completion)> {
        let (name, l, busy) = self.get(model, Kind::Llm)?;
        let Loaded::Llm(e) = l else { unreachable!() };
        let _b = busy.lock().unwrap();
        let mut e = e.lock().unwrap();
        let ids = build(&e)?;
        let mut text = String::new();
        let stats = e.generate(&ids, params, &mut |p| {
            text.push_str(p);
            on_text(p)
        })?;
        Ok((name, Completion { text, stats }))
    }

    /// Chat completion with the model's chat template (thinking off).
    ///
    /// # Example
    /// ```no_run
    /// use aido::llm::chat::Message;
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let out = models.chat("qwen", &[Message::new("user", "2+2?")], &Default::default(), &mut |_| true).unwrap();
    /// println!("{}", out.text);
    /// ```
    pub fn chat(&self, model: &str, messages: &[Message], params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Completion> {
        self.generate(model, |e| e.chat_prompt(messages, false), params, on_text).map(|x| x.1)
    }

    /// Full chat turn: system/user/assistant/tool messages, tools, thinking,
    /// `n` answers, long-term memory and automatic summarisation of history
    /// that does not fit the context. `on_piece(choice_index, piece)` streams
    /// content / reasoning as it is generated (tool-call markup is held back
    /// and returned structured); return `false` to stop.
    ///
    /// # Example
    /// ```no_run
    /// use aido::models::ChatOptions;
    /// use aido::llm::chat::Message;
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let tools = serde_json::json!([{"type": "function", "function": {"name": "get_weather",
    ///     "description": "Weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]);
    /// let msgs = [Message::new("system", "You are a helpful assistant."), Message::new("user", "Weather in Kazan?")];
    /// let r = models.chat_full("qwen", &msgs, &ChatOptions { tools: Some(tools), ..Default::default() }, &mut |_, _| true).unwrap();
    /// for c in &r.choices[0].tool_calls { println!("call {}({})", c.name, c.arguments); }
    /// ```
    pub fn chat_full(&self, model: &str, messages: &[Message], o: &ChatOptions, on_piece: &mut dyn FnMut(usize, &crate::llm::conversation::Piece) -> bool) -> Result<ChatResult> {
        use crate::llm::conversation::{parse_assistant, StreamSplitter, SUMMARY_PROMPT};
        let (name, l, busy) = self.get(model, Kind::Llm)?;
        let Loaded::Llm(e) = l else { unreachable!() };
        let _b = busy.lock().unwrap();
        let mut e = e.lock().unwrap();
        let mut msgs = messages.to_vec();
        let last_user = msgs.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default();
        // Long-term memory → system note.
        let mut memories_used = 0;
        if let (Some(u), Some(mem)) = (&o.user, &self.memory) {
            let hits = mem.search(u, &last_user, o.memory_k)?;
            if !hits.is_empty() {
                memories_used = hits.len();
                let note = format!("Long-term memory about this user (from earlier conversations):\n{}", hits.iter().map(|h| format!("- {}", h.text.replace('\n', " "))).collect::<Vec<_>>().join("\n"));
                match msgs.iter_mut().find(|m| m.role == "system") {
                    Some(m) => m.content = format!("{}\n\n{note}", m.content),
                    None => msgs.insert(0, Message::new("system", &note)),
                }
            }
        }
        // Client tools + built-in memory tools.
        let all_tools: Option<Value> = {
            let mut v: Vec<Value> = o.tools.as_ref().and_then(|t| t.as_array().cloned()).unwrap_or_default();
            if o.memory_tools && o.user.is_some() && self.memory.is_some() {
                v.extend(serde_json::from_str::<Vec<Value>>(crate::memory::MEMORY_TOOLS)?);
            }
            if v.is_empty() { None } else { Some(Value::Array(v)) }
        };
        // Fit the context: fold the oldest turns into a summary made by the same model.
        let ctx = e.opts.ctx;
        let reserve = o.params.max_tokens.min(ctx / 2) + 16;
        let mut summarized = 0;
        let mut ids = e.chat_prompt_tools(&msgs, o.think, all_tools.as_ref())?;
        while ids.len() + reserve > ctx {
            anyhow::ensure!(o.auto_summarize, "prompt of {} tokens does not fit the context ({ctx}) with {} new tokens", ids.len(), o.params.max_tokens);
            let first = msgs.iter().position(|m| m.role != "system" && !m.content.starts_with("Summary of the earlier conversation")).unwrap_or(msgs.len());
            let keep_from = msgs.len().saturating_sub(1).max(first);
            anyhow::ensure!(keep_from > first, "the last message alone ({} tokens) does not fit the context ({ctx}); raise --ctx", ids.len());
            let mut transcript: String = msgs[first..keep_from].iter().map(|m| format!("{}: {}\n", m.role, m.content)).collect();
            // The summariser itself must fit: keep the tail of a too-long transcript (measured in tokens).
            let budget = ctx.saturating_sub(400.min(ctx / 4) + 200);
            let tt = e.tok.encode(&transcript, false);
            if tt.len() > budget {
                transcript = format!("…{}", e.tok.decode(&tt[tt.len() - budget..]));
            }
            let sp = e.chat_prompt(&[Message::new("system", SUMMARY_PROMPT), Message::new("user", &transcript)], false)?;
            let gp = GenParams { max_tokens: 400.min(ctx / 4), ..Default::default() };
            let mut summary = String::new();
            e.generate(&sp, &gp, &mut |p| {
                summary.push_str(p);
                true
            })?;
            let summary = parse_assistant(&summary).content;
            let n_fold = keep_from - first;
            msgs.splice(first..keep_from, [Message::new("system", &format!("Summary of the earlier conversation:\n{summary}"))]);
            summarized += n_fold;
            eprintln!("[chat] folded {n_fold} old message(s) into a summary to fit {ctx} tokens");
            ids = e.chat_prompt_tools(&msgs, o.think, all_tools.as_ref())?;
        }
        // Does the prompt itself open a <think> block (DeepSeek-R1 style templates)?
        let tail = e.tok.decode(&ids[ids.len().saturating_sub(4)..]);
        let opened = tail.contains("<think>") && !tail.contains("</think>");
        let mut choices = Vec::new();
        let mem_tools = o.memory_tools && o.user.is_some() && self.memory.is_some();
        let base_msgs = msgs.clone();
        for i in 0..o.n.max(1) {
            let mut p = o.params.clone();
            p.sampler.seed = p.sampler.seed.wrapping_add(i as u64);
            let mut conv = base_msgs.clone();
            let mut cur = ids.clone();
            let mut agg: Option<GenStats> = None;
            // Agent loop: built-in memory tools are executed here and the
            // model continues; client tools end the turn.
            let mut round = 0;
            let choice = loop {
                round += 1;
                let mut split = StreamSplitter::new(opened);
                let mut raw = String::new();
                let mut go = true;
                let stats = e.generate(&cur, &p, &mut |piece| {
                    raw.push_str(piece);
                    for pc in split.push(piece) {
                        go &= on_piece(i, &pc);
                    }
                    go
                })?;
                for pc in split.finish() {
                    on_piece(i, &pc);
                }
                agg = Some(match agg.take() {
                    None => stats.clone(),
                    Some(mut a) => {
                        a.gen_tokens += stats.gen_tokens;
                        a.gen_s += stats.gen_s;
                        a.prefill_s += stats.prefill_s;
                        a.finish = stats.finish;
                        a
                    }
                });
                let parsed = parse_assistant(&if opened { format!("<think>{raw}") } else { raw });
                let (builtin, client): (Vec<_>, Vec<_>) = parsed.tool_calls.iter().cloned().partition(|t| mem_tools && t.name.starts_with("memory_"));
                if builtin.is_empty() || round > 4 {
                    let finish = if !client.is_empty() {
                        "tool_calls"
                    } else if stats.finish == crate::llm::engine::Finish::Length {
                        "length"
                    } else {
                        "stop"
                    };
                    break ChatChoice { content: parsed.content, reasoning: parsed.reasoning, tool_calls: client, finish_reason: finish.into(), stats: agg.clone().unwrap() };
                }
                let (u, mem) = (o.user.as_deref().unwrap(), self.memory.as_ref().unwrap());
                conv.push(Message { role: "assistant".into(), content: parsed.content.clone(), tool_calls: Some(Value::Array(builtin.iter().map(|t| t.openai()).collect())), ..Default::default() });
                for t in &builtin {
                    let result = match t.name.as_str() {
                        "memory_search" => {
                            let q = t.arguments["query"].as_str().unwrap_or(&last_user);
                            let hits = mem.search(u, q, o.memory_k)?;
                            if hits.is_empty() { "nothing found".to_string() } else { hits.iter().map(|h| format!("- {}", h.text)).collect::<Vec<_>>().join("\n") }
                        }
                        "memory_save" => {
                            let txt = t.arguments["text"].as_str().unwrap_or("");
                            if txt.trim().is_empty() { "nothing to save".into() } else { format!("saved as #{}", mem.add(u, txt)?.id) }
                        }
                        other => format!("unknown tool {other}"),
                    };
                    eprintln!("[chat] tool {}({}) -> {} chars", t.name, t.arguments, result.len());
                    conv.push(Message { role: "tool".into(), content: result, tool_call_id: Some(t.id.clone()), name: Some(t.name.clone()), ..Default::default() });
                }
                cur = e.chat_prompt_tools(&conv, o.think, all_tools.as_ref())?;
                anyhow::ensure!(cur.len() + 16 < ctx, "tool results overflow the context");
            };
            choices.push(choice);
        }
        if let (Some(u), Some(mem), true) = (&o.user, &self.memory, o.remember) {
            let clip = |s: &str, n: usize| s.chars().take(n).collect::<String>();
            if !last_user.trim().is_empty() {
                mem.add(u, &format!("User said: {} | Assistant answered: {}", clip(&last_user, 400), clip(&choices[0].content, 400)))?;
            }
        }
        Ok(ChatResult { model: name, choices, summarized, memories_used, prompt_tokens: ids.len() })
    }

    /// The long-term memory store, if enabled ([`ModelsConfig::memory_dir`]).
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// if let Some(m) = models.memory() { m.add("alice", "Alice prefers short answers").unwrap(); }
    /// ```
    pub fn memory(&self) -> Option<&crate::memory::MemoryStore> {
        self.memory.as_ref()
    }

    /// Plain text completion of `prompt`.
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let out = models.complete("qwen", "The capital of France is", &Default::default(), &mut |_| true).unwrap();
    /// ```
    pub fn complete(&self, model: &str, prompt: &str, params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Completion> {
        self.generate(model, |e| Ok(e.tok.encode(prompt, true)), params, on_text).map(|x| x.1)
    }

    /// Runs a decision model: typed answers to `questions` about `state`
    /// and optional pictures / audio. Returns
    /// `{"model", "answers": [...], "tokens", "timing_ms"}` (see
    /// [`crate::llm::d1::D1::decide`] for the question types).
    ///
    /// # Example
    /// ```no_run
    /// use serde_json::json;
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let out = models.decide("d1", &json!("User: my order is late"), &json!([{"type": "choice", "question": "Intent?", "options": ["refund", "track", "other"]}]), &[]).unwrap();
    /// println!("{}", out["answers"]);
    /// ```
    pub fn decide(&self, model: &str, state: &Value, questions: &Value, media: &[crate::llm::d1_media::MediaInput]) -> Result<Value> {
        let (name, l, busy) = self.get(model, Kind::Decision)?;
        let Loaded::Decision(d) = l else { unreachable!() };
        let _b = busy.lock().unwrap();
        let mut out = d.decide(&self.pool, state, questions, media)?;
        out["model"] = json!(name);
        Ok(out)
    }

    /// Upscales an RGB image; `loops` overrides the model's body loops
    /// (fewer = faster, `None` = model default).
    ///
    /// # Example
    /// ```no_run
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let img = image::open("frame.png").unwrap().to_rgb8();
    /// let big = models.upscale("x2", &img, Some(2)).unwrap();
    /// ```
    pub fn upscale(&self, model: &str, img: &image::RgbImage, loops: Option<usize>) -> Result<image::RgbImage> {
        let (_, l, busy) = self.get(model, Kind::Upscale)?;
        let Loaded::Upscale(u) = l else { unreachable!() };
        let _b = busy.lock().unwrap();
        let mut u = u.lock().unwrap();
        let default = u.loops;
        if let Some(n) = loops {
            u.set_loops(n);
        }
        let x: Vec<f32> = img.as_raw().iter().map(|v| *v as f32 * (1.0 / 255.0)).collect();
        let (h, w, y) = u.upscale(&self.pool, &x, img.height() as usize, img.width() as usize);
        if loops.is_some() {
            u.set_loops(default);
        }
        let bytes = y.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        image::RgbImage::from_raw(w as u32, h as u32, bytes).context("upscale output size")
    }

    /// Generates an image; `on_step(step, total, preview)` receives a
    /// decoded preview after every denoising step when given.
    ///
    /// # Example
    /// ```no_run
    /// use aido::diffusion::pipeline::ImageParams;
    /// # let models = aido::models::Models::new(Default::default()).unwrap();
    /// let p = ImageParams { prompt: "a fox".into(), negative: String::new(), steps: 1, width: 512, height: 512, seed: 1, guidance: 1.0 };
    /// models.image("sd", &p, None).unwrap().save_png("fox.png").unwrap();
    /// ```
    pub fn image(&self, model: &str, params: &ImageParams, on_step: Option<&mut dyn FnMut(usize, usize, &Image)>) -> Result<Image> {
        let (_, l, busy) = self.get(model, Kind::Diffusion)?;
        let Loaded::Diffusion(p) = l else { unreachable!() };
        let _b = busy.lock().unwrap();
        let mut p = p.lock().unwrap();
        p.generate_with(params, on_step)
    }

    /// Timing summary of the last image generated by `model`.
    pub fn image_timings(&self, model: &str) -> Option<String> {
        let es = self.entries.lock().unwrap();
        es.iter().find(|e| e.name == model || (model.is_empty() && e.kind == Kind::Diffusion)).and_then(|e| match &e.loaded {
            Some(Loaded::Diffusion(p)) => Some(p.lock().unwrap().timings()),
            _ => None,
        })
    }
}
