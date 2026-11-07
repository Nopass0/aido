//! Large language models: architectures, sampling, chat templates and the
//! high-level generation engine.
//!
//! Every architecture implements [`LanguageModel`]; the engine, server and
//! tools only talk to that trait, so adding an architecture never touches
//! the scheduling / speculation / caching code.

pub mod chat;
pub mod conversation;
pub mod engine;
pub mod kv;
pub mod model;
pub mod qwen35;
pub mod lfm3;
pub mod d1;
pub mod d1_media;
pub mod sampler;

use std::any::Any;

use anyhow::Result;

use crate::format::TensorStore;
use crate::runtime::pool::ThreadPool;
use crate::tokenizer::Tokenizer;

/// Per-sequence mutable state of a model (KV caches, recurrent states,
/// scratch buffers, logits of the last pass).
pub trait ModelState: Send {
    /// Logits of the last forward (`n_out × vocab`).
    fn logits(&self) -> &[f32];
    /// Largest batch accepted by `forward`.
    fn max_t(&self) -> usize;
    /// Context capacity in tokens.
    fn n_ctx(&self) -> usize;
    /// Memory held, in bytes.
    fn nbytes(&self) -> usize;
    /// Downcast support for architecture-specific code.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A causal language model.
pub trait LanguageModel: Send + Sync {
    /// Human-readable description.
    fn describe(&self) -> String;
    /// Vocabulary size (logit width).
    fn vocab(&self) -> usize;
    /// Metadata / tensor store (chat template, etc.).
    fn store(&self) -> &dyn TensorStore;
    /// Builds the tokenizer of this model.
    fn tokenizer(&self) -> Result<Tokenizer>;
    /// Bytes of weights read per decoded token (roofline numerator).
    fn weight_bytes_per_token(&self) -> usize;
    /// Allocates a state for `n_ctx` tokens and batches of `max_t`.
    fn new_state(&self, n_ctx: usize, max_t: usize, n_threads: usize) -> Box<dyn ModelState>;
    /// Processes `tokens` at positions `pos0..` and stores logits (all rows
    /// or only the last). Returns the number of logit rows.
    fn forward(&self, pool: &ThreadPool, tokens: &[u32], pos0: usize, st: &mut dyn ModelState, all_logits: bool) -> Result<usize>;
    /// `true` if the state can be rewound to any earlier position (pure KV
    /// cache). Recurrent models (DeltaNet, Mamba) return `false`: the engine
    /// then only reuses exact prefixes and disables speculative rollback.
    fn can_rewind(&self) -> bool {
        true
    }
    /// Chat-template text to place after the assistant header when
    /// "thinking" is disabled (model specific; empty by default).
    fn no_think_suffix(&self) -> &str {
        ""
    }
    /// Optional per-layer skip mask (run-time layer deactivation).
    fn set_skip(&mut self, _layers: &[usize]) -> Result<()> {
        anyhow::bail!("layer skipping is not supported by this architecture")
    }
}

/// Options controlling how a model is placed in memory.
#[derive(Clone, Default)]
pub struct LoadOptions {
    /// Re-quantisation of dense float weights at load (MoE containers).
    pub dense: Option<crate::tensor::quant::GgmlType>,
    /// GPU to offload dense weights to.
    pub gpu: Option<std::sync::Arc<crate::gpu::Gpu>>,
    /// VRAM budget in bytes for offloaded weights.
    pub vram_budget: usize,
    /// RAM budget in bytes for private copies of hot weights (expert cache).
    pub ram_cache: usize,
    /// KV cache policy (format, sparse-attention budget).
    pub kv: kv::KvConfig,
    /// Context length sessions will use (decides GPU vs CPU placement; 0 = 4096).
    pub ctx: usize,
}

/// Loads any supported language model from a path:
///
/// * a directory containing `qwen36_meta.json` → Qwen3.5/3.6 MoE
///   (colibri container, run in place);
/// * a `.gguf` or `.aido` file → Llama-family decoder.
///
/// * **Input** — path; `dense`: optional re-quantisation of dense F16
///   weights at load time (MoE containers).
/// * **Output** — boxed model.
///
/// # Example
/// ```no_run
/// let pool = aido::runtime::pool::ThreadPool::new(8, None);
/// let m = aido::llm::load_model("models/Qwen3-0.6B-Q8_0.gguf", &Default::default(), &pool).unwrap();
/// println!("{}", m.describe());
/// ```
pub fn load_model(path: &str, opts: &LoadOptions, pool: &ThreadPool) -> Result<Box<dyn LanguageModel>> {
    let p = std::path::Path::new(path);
    if p.is_dir() && p.join("qwen36_meta.json").exists() {
        let dense = opts.dense.unwrap_or(crate::tensor::quant::GgmlType::Q8_0);
        return Ok(Box::new(qwen35::Qwen35::load(path, dense, pool, opts.gpu.clone(), opts.vram_budget, opts.ram_cache)?));
    }
    let mut m = model::Model::load(path)?;
    m.kv_cfg = opts.kv.clone();
    // Whole model on the GPU when weights + KV for the context fit the VRAM budget.
    if let Some(g) = &opts.gpu {
        if std::env::var("AIDO_NO_GPU_LLM").is_err() {
            let ctx = if opts.ctx == 0 { 4096 } else { opts.ctx };
            let need = crate::gpu::llama::GpuLlama::vram_needed(&m);
            let kv_tok = m.cfg.n_layer * m.cfg.kv_dim() * 4;
            let fits_ctx = opts.vram_budget.saturating_sub(need) / kv_tok.max(1);
            if fits_ctx >= ctx.min(4096) {
                let gpu_ctx = ctx.min(fits_ctx);
                match crate::gpu::llama::GpuLlama::try_new(m, g.clone(), opts.vram_budget, gpu_ctx) {
                    Ok(gm) => {
                        eprintln!("[gpu] whole model in VRAM ({:.2} GB weights, KV up to {gpu_ctx} tokens)", gm.weight_bytes as f64 / 1e9);
                        return Ok(Box::new(gm));
                    }
                    Err((back, why)) => {
                        eprintln!("[gpu] CPU path: {why}");
                        m = back;
                    }
                }
            } else {
                eprintln!("[gpu] CPU path: weights {:.2} GB + KV for {ctx} tokens do not fit {:.2} GB of VRAM", need as f64 / 1e9, opts.vram_budget as f64 / 1e9);
            }
        }
    }
    Ok(Box::new(m))
}
