//! Embedding AIDO in your own Rust application.
//!
//! ```text
//! cargo run --release --example embed -- \
//!     --llm models/Qwen3-0.6B-Q8_0.gguf --decision models/d1-omni \
//!     --upscale models/baikal/Baikal_LoopSR_x2_v2.safetensors --image data/test_media/cats.jpg
//! ```
//! Every flag is optional; only the given models are exercised.
//!
//! In your own `Cargo.toml`:
//! ```toml
//! [dependencies]
//! aido = { git = "https://github.com/Nopass0/aido", default-features = false } # engine only, no HTTP server / CLI
//! ```

use std::time::Instant;

use aido::llm::chat::Message;
use aido::llm::engine::GenParams;
use aido::{Models, ModelsConfig};
use serde_json::json;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    // One registry for all models: shared thread pool + GPU, lazy loading, LRU under the RAM budget.
    let models = Models::new(ModelsConfig { ram_gb: 12.0, ..Default::default() })?;

    if let Some(p) = arg("--llm") {
        models.add("chat", &p)?;
        let t = Instant::now();
        print!("LLM> ");
        let out = models.chat("chat", &[Message::new("user", "Name three primary colors.")], &GenParams { max_tokens: 64, ..Default::default() }, &mut |piece| {
            use std::io::Write;
            print!("{piece}");
            let _ = std::io::stdout().flush();
            true
        })?;
        println!("\n   {} tokens, {:.1} tok/s, total {:.1}s", out.stats.gen_tokens, out.stats.tokens_per_s(), t.elapsed().as_secs_f64());
    }

    if let Some(p) = arg("--decision") {
        models.add("d1", &p)?;
        let questions = json!({
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": ["billing", "shipping", "technical"]}
        });
        let mut media = Vec::new();
        if let Some(img) = arg("--image") {
            media.push(aido::llm::d1_media::MediaInput::Image(image::open(img)?.to_rgb8()));
        }
        let out = models.decide("d1", &json!("I was charged twice for my order, please return the money."), &questions, &media)?;
        println!("decision> {}", out["answers"]);
    }

    if let Some(p) = arg("--upscale") {
        models.add("x2", &p)?;
        let img = match arg("--image") {
            Some(f) => image::open(f)?.to_rgb8(),
            None => image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([(x * 4) as u8, (y * 4) as u8, 128])),
        };
        let small = image::imageops::resize(&img, 96, 72, image::imageops::FilterType::Triangle);
        let t = Instant::now();
        let big = models.upscale("x2", &small, Some(1))?;
        println!("upscale> {}x{} -> {}x{} in {:.0} ms", small.width(), small.height(), big.width(), big.height(), t.elapsed().as_secs_f64() * 1e3);
        big.save("embed_upscaled.png")?;
    }

    for m in models.list() {
        println!("registered: {} ({}) loaded={}", m["id"], m["kind"], m["loaded"]);
    }
    Ok(())
}
