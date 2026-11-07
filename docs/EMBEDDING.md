# Встраивание движка в своё Rust-приложение

AIDO — обычная Rust-библиотека. Модели запускаются прямо в процессе вашего
приложения, без HTTP и сериализации: тот же реестр, что использует сервер.

## Подключение

```toml
[dependencies]
# только движок (без HTTP-сервера и CLI — меньше зависимостей):
aido = { git = "https://github.com/Nopass0/aido", default-features = false }
# или с сервером, чтобы встроить OpenAI-совместимый API в свой axum:
# aido = { git = "https://github.com/Nopass0/aido", features = ["server"] }
image = "0.25"
serde_json = "1"
```

Для максимальной скорости соберите приложение под свой процессор
(AVX2/FMA/AVX-512 ядра включаются на этапе компиляции):

```toml
# .cargo/config.toml вашего проекта
[build]
rustflags = ["-C", "target-cpu=native"]

[profile.release]
lto = "fat"
codegen-units = 1
```

## Реестр моделей `aido::Models`

```rust
use aido::{Models, ModelsConfig, Kind};
use aido::llm::{chat::Message, engine::GenParams};
use serde_json::json;

let models = Models::new(ModelsConfig {
    ram_gb: 12.0,      // бюджет ОЗУ на загруженные модели (LRU-выгрузка сверх)
    vram_gb: 4.0,      // бюджет видеопамяти на модель
    ctx: 32768,        // контекст LLM
    ..Default::default()
})?;

// Регистрация ничего не загружает; тип определяется по файлам.
assert_eq!(models.add("chat", "models/Qwen3-8B-Q4_K_M.gguf")?, Kind::Llm);
models.add("d1", "models/d1-omni")?;
models.add("x2", "models/baikal/Baikal_LoopSR_x2_v2.safetensors")?;
models.add("sd", "models/sd-turbo")?;
```

### Текст (со стримингом токенов)

```rust
let out = models.chat("chat", &[Message::new("user", "Привет!")],
    &GenParams { max_tokens: 200, ..Default::default() },
    &mut |piece| { print!("{piece}"); true /* false — остановить */ })?;
println!("\n{:.1} ток/с, {} токенов", out.stats.tokens_per_s(), out.stats.gen_tokens);

let c = models.complete("chat", "Столица Франции —", &Default::default(), &mut |_| true)?;
```

`GenParams` — `max_tokens`, `sampler` (`temperature`, `top_p`, `top_k`,
`seed`, `repeat_penalty`), `stop`, `spec` (спекулятивное декодирование).

### Решения по тексту, картинкам и звуку

```rust
use aido::llm::d1_media::{MediaInput, read_wav};
let media = vec![
    MediaInput::Image(image::open("photo.jpg")?.to_rgb8()),
    MediaInput::Audio(read_wav("call.wav")?),
];
let questions = json!({
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Who should handle it?", "criteria": ["billing", "shipping"]}
});
let r = models.decide("d1", &json!("Customer: I was charged twice."), &questions, &media)?;
println!("{}", r["answers"]["refund"]["noul"]);
```

Типы вопросов и формат ответов — в [API.md](API.md#post-v1decide--модели-решений-d1-подобные).

### Апскейл кадра

```rust
let frame = image::open("frame.png")?.to_rgb8();
let big = models.upscale("x2", &frame, Some(1))?;   // Some(loops): 1 — быстрее, None — по умолчанию
big.save("frame_x2.png")?;
```

### Генерация изображения с превью шагов

```rust
use aido::diffusion::pipeline::{ImageParams, Image};
let p = ImageParams { prompt: "a red fox in the snow".into(), negative: String::new(),
                      steps: 4, width: 512, height: 512, seed: 1, guidance: 1.0 };
let img = models.image("sd", &p, Some(&mut |step, total, preview: &Image| {
    println!("шаг {step}/{total}: {}x{}", preview.width, preview.height);
}))?;
img.save_png("fox.png")?;
```

### Полный диалог: инструменты, размышления, память, автосуммаризация

`models.chat_full(name, &messages, &ChatOptions { tools, think, n, user, .. }, &mut on_piece)` —
см. [CHAT.md](CHAT.md#библиотека). Возвращает варианты с `content`,
`reasoning`, `tool_calls`, `finish_reason` и статистикой.

### Свои модели (SDK) по сигнатуре

```rust
models.add("seq", "models/seq-c.aido")?;            // тип custom определяется по сигнатуре aido.io
let out = models.predict("seq", &json!({"sequence": [1, 1, 2, 3, 5, 8]}))?;   // {"next": 13}
```

### Управление памятью

```rust
models.preload("chat")?;                          // загрузить заранее
models.unload("sd");                              // выгрузить сейчас
models.trim(std::time::Duration::from_secs(600)); // выгрузить простаивающие > 10 мин
for m in models.list() { println!("{} {} loaded={}", m["id"], m["kind"], m["loaded"]); }
```

`Models` — `Send + Sync`: держите его в `Arc` и вызывайте из любых потоков.
Вызовы к одной модели сериализуются, к разным — идут параллельно.

## Встроить HTTP API в своё axum-приложение

```rust
let models = std::sync::Arc::new(aido::Models::new(Default::default())?);
models.add("qwen", "models/Qwen3-0.6B-Q8_0.gguf")?;
let app = axum::Router::new()
    .nest("/ai", aido::server::router(models.clone()))   // /ai/v1/chat/completions …
    .route("/", axum::routing::get(|| async { "my app" }));
```

## Низкоуровневые модули

| Модуль | Что даёт |
|---|---|
| `aido::llm::engine::LlmEngine` | один LLM-движок: сессии с кэшем префикса, спекулятивное декодирование, калибровка потоков |
| `aido::llm::kv` | сжатый постраничный KV-кэш и разреженное внимание ([LONG_CONTEXT.md](LONG_CONTEXT.md)) |
| `aido::llm::d1::D1` | модель решений напрямую |
| `aido::upscale::LoopSr` | апскейлер; `aido::upscale::bicubic` — эталонный бикубик PyTorch |
| `aido::diffusion::pipeline::Pipeline` | SD-конвейер |
| `aido::sdk` | автоград, AdamW, чекпоинты — свои сети и обучение |
| `aido::format` | `.aido`, GGUF, safetensors |
| `aido::hub` | скачивание моделей с Hugging Face (`hub::resolve("hf:org/repo")`) |

Полный пример: [`examples/embed.rs`](../examples/embed.rs)
(`cargo run --release --example embed -- --llm models/Qwen3-0.6B-Q8_0.gguf --upscale models/baikal/Baikal_LoopSR_x2_v2.safetensors`).

Справка по каждой функции (вход, выход, поведение, пример): `cargo doc --open`.
