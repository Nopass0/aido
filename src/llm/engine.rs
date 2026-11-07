//! High-level LLM runtime: sessions with prefix-reusing state caches,
//! adaptive speculative decoding, calibrated threading and streaming
//! generation — for any [`LanguageModel`].
//!
//! ## Where the speed comes from
//! 1. **Integer SIMD kernels on quantised weights** — decoding runs at the
//!    DRAM-bandwidth roofline (see [`crate::tensor::quant`]).
//! 2. **Batched verification ("free tokens")** — a forward pass over `k+1`
//!    tokens costs barely more than over one (measured `T(8) ≈ 1.3·T(1)`),
//!    because the cost is reading the weights, not multiplying them.
//! 3. **Two draft sources** — *n-gram lookup* in the conversation itself
//!    (zero cost; strong for code edits, RAG, summaries, JSON) and an
//!    optional *small draft model* sharing the tokenizer.
//! 4. **Adaptive draft length** — online estimates of the acceptance rate
//!    `α` and the verification cost `T(t)` give
//!    `k* = argmax_k (1-α^{k+1})/(1-α) / (T(k+1) + k·c_draft)`; when drafts
//!    stop paying off the engine falls back to `k = 0`.
//! 5. **Prefix state cache across requests** — chat clients resend the
//!    whole history; the session sharing the longest prefix is reused.
//!    Recurrent models (DeltaNet) reuse exact prefixes only.
//! 6. **Calibrated thread width + thermal guard** (see
//!    [`crate::runtime::governor`]).

use std::time::{Duration, Instant};

use anyhow::Result;

use super::sampler::{argmax, Sampler, SamplerConfig};
use super::{load_model, LanguageModel, ModelState};
use crate::runtime::governor::{calibrate_width, candidate_widths, ThermalGuard};
use crate::runtime::pool::ThreadPool;
use crate::runtime::sys::CpuInfo;
use crate::tensor::quant::GgmlType;
use crate::tokenizer::{StreamDecoder, Tokenizer};

/// Speculative decoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecMode {
    /// Plain autoregressive decoding.
    Off,
    /// n-gram prompt lookup only.
    Lookup,
    /// Draft model only (falls back to `Off` if none loaded).
    Draft,
    /// Lookup when a match exists, otherwise the draft model if loaded.
    Auto,
}

impl std::str::FromStr for SpecMode {
    type Err = anyhow::Error;
    /// Parses `off|lookup|draft|auto`.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "off" | "none" => SpecMode::Off,
            "lookup" | "ngram" => SpecMode::Lookup,
            "draft" => SpecMode::Draft,
            "auto" => SpecMode::Auto,
            _ => anyhow::bail!("unknown speculative mode '{s}' (off|lookup|draft|auto)"),
        })
    }
}

/// Engine construction options.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Total worker threads (default: physical cores).
    pub threads: Option<usize>,
    /// Context length per session.
    pub ctx: usize,
    /// Max tokens per forward batch (prompt chunk size).
    pub batch: usize,
    /// Speculation strategy.
    pub spec: SpecMode,
    /// Maximum draft length.
    pub max_draft: usize,
    /// Optional draft model path.
    pub draft_path: Option<String>,
    /// Measure the best decode width at load time.
    pub calibrate: bool,
    /// Pre-fault weights into RAM at load.
    pub prefetch: bool,
    /// Maximum number of cached sessions (prefix caches).
    pub max_sessions: usize,
    /// Pin workers to physical cores.
    pub pin: bool,
    /// Blocks deactivated at run time.
    pub skip_layers: Vec<usize>,
    /// Re-quantisation of dense float weights at load (MoE containers).
    pub dense: Option<GgmlType>,
    /// Use the GPU when one is available.
    pub gpu: bool,
    /// VRAM budget (GB) for offloaded weights.
    pub vram_gb: f32,
    /// RAM budget (GB) for private copies of hot experts.
    pub ram_cache_gb: f32,
    /// KV cache format and long-context sparse attention (see [`super::kv`]).
    pub kv: super::kv::KvConfig,
    /// Shared GPU device (several models on one GPU share it and its VRAM
    /// accounting); `None` opens a device of its own.
    pub gpu_device: Option<std::sync::Arc<crate::gpu::Gpu>>,
}

impl Default for EngineOptions {
    /// Sensible defaults: 4096 context, 256-token batches, auto speculation.
    fn default() -> Self {
        EngineOptions {
            threads: None,
            ctx: 4096,
            batch: 256,
            spec: SpecMode::Auto,
            max_draft: 8,
            draft_path: None,
            calibrate: true,
            prefetch: true,
            max_sessions: 6,
            pin: true,
            skip_layers: vec![],
            dense: None,
            gpu: true,
            vram_gb: 5.0,
            ram_cache_gb: 4.0,
            kv: super::kv::KvConfig::default(),
            gpu_device: None,
        }
    }
}

/// Per-request generation parameters.
#[derive(Debug, Clone)]
pub struct GenParams {
    /// Maximum new tokens.
    pub max_tokens: usize,
    /// Sampling configuration.
    pub sampler: SamplerConfig,
    /// Stop strings.
    pub stop: Vec<String>,
    /// Override of the engine's speculation mode.
    pub spec: Option<SpecMode>,
}

impl Default for GenParams {
    /// 256 tokens, greedy, no stop strings.
    fn default() -> Self {
        GenParams { max_tokens: 256, sampler: SamplerConfig::default(), stop: vec![], spec: None }
    }
}

/// Why generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// End-of-sequence token or stop string.
    Stop,
    /// `max_tokens` reached.
    Length,
    /// Callback requested cancellation.
    Cancelled,
}

/// Timing and speculation statistics of one generation.
#[derive(Debug, Clone)]
pub struct GenStats {
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: usize,
    /// Prompt processing time.
    pub prefill_s: f64,
    /// Generated tokens.
    pub gen_tokens: usize,
    /// Generation time.
    pub gen_s: f64,
    /// Draft tokens proposed.
    pub drafted: usize,
    /// Draft tokens accepted.
    pub accepted: usize,
    /// Target forward passes during generation.
    pub passes: usize,
    /// Finish reason.
    pub finish: Finish,
    /// Generated token ids.
    pub tokens: Vec<u32>,
}

impl GenStats {
    /// Generation speed in tokens/second.
    ///
    /// # Example
    /// ```
    /// # use aido::llm::engine::*;
    /// let s = GenStats { prompt_tokens: 0, cached_tokens: 0, prefill_s: 0.0, gen_tokens: 10, gen_s: 2.0, drafted: 0, accepted: 0, passes: 10, finish: Finish::Length, tokens: vec![] };
    /// assert_eq!(s.tokens_per_s(), 5.0);
    /// ```
    pub fn tokens_per_s(&self) -> f64 {
        if self.gen_s > 0.0 { self.gen_tokens as f64 / self.gen_s } else { 0.0 }
    }

    /// Prompt processing speed in tokens/second (uncached tokens only).
    ///
    /// # Example
    /// ```
    /// # use aido::llm::engine::*;
    /// let s = GenStats { prompt_tokens: 100, cached_tokens: 0, prefill_s: 0.5, gen_tokens: 0, gen_s: 0.0, drafted: 0, accepted: 0, passes: 0, finish: Finish::Stop, tokens: vec![] };
    /// assert_eq!(s.prefill_tps(), 200.0);
    /// ```
    pub fn prefill_tps(&self) -> f64 {
        let n = self.prompt_tokens - self.cached_tokens;
        if self.prefill_s > 0.0 { n as f64 / self.prefill_s } else { 0.0 }
    }
}

/// A model state bound to one conversation prefix.
pub struct Session {
    /// Model state (KV cache / recurrent state / workspace).
    pub state: Box<dyn ModelState>,
    /// Tokens whose effect is contained in `state`.
    pub tokens: Vec<u32>,
    last_used: Instant,
}

impl Session {
    /// Allocates a session for `model`.
    ///
    /// # Example
    /// ```no_run
    /// let pool = aido::runtime::pool::ThreadPool::new(4, None);
    /// let m = aido::llm::load_model("m.gguf", &Default::default(), &pool).unwrap();
    /// let s = aido::llm::engine::Session::new(m.as_ref(), 2048, 64, 4);
    /// ```
    pub fn new(model: &dyn LanguageModel, n_ctx: usize, max_t: usize, n_threads: usize) -> Self {
        Session { state: model.new_state(n_ctx, max_t, n_threads), tokens: Vec::new(), last_used: Instant::now() }
    }
}

/// A draft model with its own session.
struct Draft {
    model: Box<dyn LanguageModel>,
    sess: Session,
}

/// Online cost/acceptance model for choosing the draft length.
#[derive(Debug, Clone)]
struct SpecTuner {
    t_fwd: Vec<f64>,
    alpha: [f64; 2],
    c_draft: f64,
}

impl SpecTuner {
    fn new(kmax: usize) -> Self {
        SpecTuner { t_fwd: vec![0.0; kmax + 2], alpha: [0.6, 0.6], c_draft: 0.0 }
    }

    fn record_fwd(&mut self, t: usize, secs: f64) {
        if t < self.t_fwd.len() {
            let e = &mut self.t_fwd[t];
            *e = if *e == 0.0 { secs } else { *e * 0.9 + secs * 0.1 };
        }
    }

    fn cost(&self, t: usize) -> f64 {
        let t1 = self.t_fwd[1].max(1e-6);
        if t < self.t_fwd.len() && self.t_fwd[t] > 0.0 {
            return self.t_fwd[t];
        }
        t1 * (1.0 + 0.08 * (t as f64 - 1.0))
    }

    fn best_k(&self, src: usize, kmax: usize) -> usize {
        let a = self.alpha[src].clamp(0.01, 0.99);
        let cd = if src == 1 { self.c_draft } else { 0.0 };
        let mut best = (0usize, 1.0 / self.cost(1));
        for k in 1..=kmax {
            let e = (1.0 - a.powi(k as i32 + 1)) / (1.0 - a);
            let rate = e / (self.cost(k + 1) + k as f64 * cd);
            if rate > best.1 {
                best = (k, rate);
            }
        }
        best.0
    }
}

/// The LLM engine.
pub struct LlmEngine {
    /// Target model.
    pub model: Box<dyn LanguageModel>,
    /// Tokenizer.
    pub tok: Tokenizer,
    /// Worker pool.
    pub pool: ThreadPool,
    /// Options in effect.
    pub opts: EngineOptions,
    /// Width used for single-token decoding.
    pub decode_width: usize,
    /// Width used for batched passes.
    pub batch_width: usize,
    /// Calibration table `(width, seconds per token)`.
    pub calibration: Vec<(usize, f64)>,
    sessions: Vec<Session>,
    draft: Option<Draft>,
    tuner: SpecTuner,
    thermal: ThermalGuard,
}

/// Length of the common prefix of two token sequences.
///
/// # Example
/// ```
/// assert_eq!(aido::llm::engine::common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
/// ```
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Proposes up to `k` draft tokens by finding the most recent earlier
/// occurrence of the longest suffix n-gram (`n_max..=n_min`) of `hist` and
/// copying what followed it.
///
/// # Example
/// ```
/// use aido::llm::engine::lookup_draft;
/// let h = [5, 6, 7, 8, 9, 5, 6, 7];
/// assert_eq!(lookup_draft(&h, 3, 3, 1), vec![8, 9, 5]);
/// ```
pub fn lookup_draft(hist: &[u32], k: usize, n_max: usize, n_min: usize) -> Vec<u32> {
    let len = hist.len();
    for n in (n_min..=n_max).rev() {
        if len <= n {
            continue;
        }
        let pat = &hist[len - n..];
        let mut i = len - n;
        while i > 0 {
            i -= 1;
            if hist[i..i + n] == *pat {
                let s = i + n;
                let e = (s + k).min(len);
                if s < e {
                    return hist[s..e].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// Runs a forward pass with the right pool width for the batch size.
#[allow(clippy::too_many_arguments)]
fn run(pool: &ThreadPool, model: &dyn LanguageModel, s: &mut Session, tokens: &[u32], pos: usize, all: bool, dw: usize, bw: usize) -> Result<usize> {
    pool.set_width(if tokens.len() == 1 { dw } else { bw });
    model.forward(pool, tokens, pos, s.state.as_mut(), all)
}

/// Brings session `s` to contain exactly `prompt` (reusing the cached
/// prefix) and leaves the logits of the last prompt token in its state.
/// Returns the number of prompt tokens served from cache.
fn prefill(pool: &ThreadPool, model: &dyn LanguageModel, s: &mut Session, prompt: &[u32], dw: usize, bw: usize) -> Result<usize> {
    let mut keep = common_prefix(&s.tokens, prompt);
    if keep == prompt.len() {
        keep -= 1;
    }
    if !model.can_rewind() && keep < s.tokens.len() {
        keep = 0;
    }
    s.tokens.truncate(keep);
    let maxb = s.state.max_t();
    let mut pos = keep;
    while pos < prompt.len() {
        let end = (pos + maxb).min(prompt.len());
        run(pool, model, s, &prompt[pos..end], pos, false, dw, bw)?;
        pos = end;
    }
    s.tokens = prompt.to_vec();
    s.last_used = Instant::now();
    Ok(keep)
}

impl LlmEngine {
    /// Loads a model and prepares the runtime.
    ///
    /// * **Input** — model path (GGUF, `.aido` or a supported container
    ///   directory), options.
    /// * **Output** — ready engine.
    /// * **Behaviour** — pinned pool (one thread per physical core by
    ///   default), optional pre-faulting, draft model, then decode-width
    ///   calibration (smallest width within 3 % of the fastest).
    ///
    /// # Example
    /// ```no_run
    /// use aido::llm::engine::{LlmEngine, EngineOptions};
    /// let eng = LlmEngine::load("models/Qwen3-0.6B-Q8_0.gguf", EngineOptions::default()).unwrap();
    /// ```
    pub fn load(path: &str, opts: EngineOptions) -> Result<Self> {
        let cpu = CpuInfo::detect();
        let n = opts.threads.unwrap_or(cpu.physical).max(1);
        let pin = if opts.pin {
            let mut ids = cpu.physical_core_ids();
            if n > ids.len() {
                ids.extend((0..cpu.physical).map(|c| c * cpu.smt + 1).filter(|&x| x < cpu.logical));
            }
            Some(ids)
        } else {
            None
        };
        let pool = ThreadPool::new(n, pin);
        let gpu = if let (true, Some(g)) = (opts.gpu, &opts.gpu_device) {
            Some(g.clone())
        } else if opts.gpu {
            match crate::gpu::Gpu::new() {
                Ok(g) => {
                    eprintln!("[gpu] {}", g.describe());
                    Some(std::sync::Arc::new(g))
                }
                Err(e) => {
                    eprintln!("[gpu] not used: {e}");
                    None
                }
            }
        } else {
            None
        };
        let lo = super::LoadOptions { dense: opts.dense, gpu, vram_budget: (opts.vram_gb * 1e9) as usize, ram_cache: (opts.ram_cache_gb * 1e9) as usize, kv: opts.kv.clone(), ctx: opts.ctx };
        let mut model = load_model(path, &lo, &pool)?;
        if !opts.skip_layers.is_empty() {
            model.set_skip(&opts.skip_layers)?;
        }
        let tok = model.tokenizer()?;
        // Pre-fault weights only when they comfortably fit in RAM; larger
        // models (e.g. 22 GB MoE containers) are paged on demand so the OS
        // keeps the hot experts resident instead of thrashing.
        let total: usize = model.store().names().iter().filter_map(|n| model.store().raw(n).ok()).map(|b| b.len()).sum();
        if opts.prefetch && total < 8usize << 30 {
            model.store().prefetch(&pool);
        }
        let draft = match &opts.draft_path {
            Some(p) => {
                let dm = load_model(p, &super::LoadOptions::default(), &pool)?;
                if opts.prefetch {
                    dm.store().prefetch(&pool);
                }
                let sess = Session::new(dm.as_ref(), opts.ctx, 1, pool.n_threads());
                Some(Draft { model: dm, sess })
            }
            None => None,
        };
        let mut eng = LlmEngine {
            tuner: SpecTuner::new(opts.max_draft),
            model,
            tok,
            decode_width: n,
            batch_width: n,
            calibration: vec![],
            sessions: Vec::new(),
            draft,
            thermal: ThermalGuard::new(0.30),
            opts,
            pool,
        };
        if eng.opts.calibrate {
            eng.calibrate()?;
        }
        Ok(eng)
    }

    /// Measures single-token decode latency for several pool widths and
    /// keeps the smallest width within 3 % of the best.
    ///
    /// # Example
    /// ```no_run
    /// # let mut eng = aido::llm::engine::LlmEngine::load("m.gguf", Default::default()).unwrap();
    /// let w = eng.calibrate().unwrap();
    /// ```
    pub fn calibrate(&mut self) -> Result<usize> {
        let mut st = self.model.new_state(64, 1, self.pool.n_threads());
        let model = self.model.as_ref();
        let pool = &self.pool;
        let mut pos = 0usize;
        let cands = candidate_widths(pool.n_threads());
        let (w, table) = calibrate_width(pool, &cands, 3, 0.03, || {
            if pos + 1 >= 64 {
                pos = 0;
            }
            model.forward(pool, &[1], pos, st.as_mut(), false).unwrap();
            pos += 1;
        });
        self.decode_width = w;
        self.calibration = table;
        self.pool.set_width(self.batch_width);
        Ok(w)
    }

    /// Finds the session sharing the longest usable prefix with `prompt`,
    /// creating or evicting (LRU) one when needed.
    fn acquire(&mut self, prompt: &[u32]) -> usize {
        let rewind = self.model.can_rewind();
        let usable = |s: &Session| {
            let l = common_prefix(&s.tokens, prompt);
            if rewind || l == s.tokens.len() { l } else { 0 }
        };
        if let Some((l, i)) = self.sessions.iter().enumerate().map(|(i, s)| (usable(s), i)).max() {
            if l > 0 {
                return i;
            }
        }
        if self.sessions.len() < self.opts.max_sessions.max(1) {
            let mt = self.opts.batch.max(self.opts.max_draft + 1);
            self.sessions.push(Session::new(self.model.as_ref(), self.opts.ctx, mt, self.pool.n_threads()));
            return self.sessions.len() - 1;
        }
        let lru = self.sessions.iter().enumerate().min_by_key(|(_, s)| s.last_used).map(|(i, _)| i).unwrap();
        self.sessions[lru].tokens.clear();
        lru
    }

    /// Frees sessions idle for longer than `max_idle` (automatic memory
    /// cleanup; the server calls this periodically). Returns how many.
    ///
    /// # Example
    /// ```no_run
    /// # let mut eng = aido::llm::engine::LlmEngine::load("m.gguf", Default::default()).unwrap();
    /// eng.trim_idle(std::time::Duration::from_secs(300));
    /// ```
    pub fn trim_idle(&mut self, max_idle: Duration) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|s| s.last_used.elapsed() < max_idle);
        before - self.sessions.len()
    }

    /// Bytes held by session states.
    ///
    /// # Example
    /// ```no_run
    /// # let eng = aido::llm::engine::LlmEngine::load("m.gguf", Default::default()).unwrap();
    /// println!("{} MB", eng.cache_bytes() >> 20);
    /// ```
    pub fn cache_bytes(&self) -> usize {
        self.sessions.iter().map(|s| s.state.nbytes()).sum()
    }

    /// Generates a completion for a token prompt.
    ///
    /// * **Input** — `prompt` token ids; `params`; `on_text`: streaming
    ///   callback receiving decoded text pieces, return `false` to cancel.
    /// * **Output** — statistics including the generated ids.
    ///
    /// # Example
    /// ```no_run
    /// use aido::llm::engine::*;
    /// let mut eng = LlmEngine::load("models/Qwen3-0.6B-Q8_0.gguf", EngineOptions::default()).unwrap();
    /// let ids = eng.tok.encode("The capital of France is", true);
    /// let st = eng.generate(&ids, &GenParams { max_tokens: 16, ..Default::default() }, &mut |t| { print!("{t}"); true }).unwrap();
    /// println!("\n{:.1} tok/s", st.tokens_per_s());
    /// ```
    pub fn generate(&mut self, prompt: &[u32], params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<GenStats> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        anyhow::ensure!(prompt.len() < self.opts.ctx, "prompt ({} tokens) exceeds context {}", prompt.len(), self.opts.ctx);
        let si = self.acquire(prompt);
        let (dw, bw) = (self.decode_width, self.batch_width);
        let mut sampler = Sampler::new(params.sampler.clone());
        let mut spec = params.spec.unwrap_or(self.opts.spec);
        if !self.model.can_rewind() {
            spec = SpecMode::Off;
        }
        let kmax = self.opts.max_draft;
        let vocab = self.model.vocab();

        let t0 = Instant::now();
        let cached = prefill(&self.pool, self.model.as_ref(), &mut self.sessions[si], prompt, dw, bw)?;
        let logits_last = self.sessions[si].state.logits()[..vocab].to_vec();
        let prefill_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut hist: Vec<u32> = prompt.to_vec();
        let mut out_tokens = Vec::new();
        let mut dec = StreamDecoder::default();
        let mut text = String::new();
        let mut emitted = 0usize;
        let max_stop = params.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        let (mut drafted, mut accepted, mut passes) = (0, 0, 0);
        let finish;
        let mut pending = sampler.sample(&logits_last, &hist);
        let mut string_stop = false;
        let tok = &self.tok;

        let mut emit = |t: u32, out_tokens: &mut Vec<u32>, hist: &mut Vec<u32>, text: &mut String, emitted: &mut usize, string_stop: &mut bool| -> Option<Finish> {
            if tok.stop_ids.contains(&t) {
                return Some(Finish::Stop);
            }
            out_tokens.push(t);
            hist.push(t);
            if !tok.is_control(t) {
                text.push_str(&dec.push(&tok.token_bytes(t)));
            }
            if max_stop > 0 {
                let mut from = emitted.saturating_sub(max_stop);
                while !text.is_char_boundary(from) {
                    from -= 1;
                }
                for st in &params.stop {
                    if let Some(p) = text[from..].find(st.as_str()) {
                        let at = from + p;
                        if at > *emitted && !on_text(&text[*emitted..at]) {
                            return Some(Finish::Cancelled);
                        }
                        *emitted = at.max(*emitted);
                        *string_stop = true;
                        return Some(Finish::Stop);
                    }
                }
            }
            let mut safe = text.len().saturating_sub(max_stop.saturating_sub(1));
            while !text.is_char_boundary(safe) {
                safe -= 1;
            }
            if safe > *emitted {
                if !on_text(&text[*emitted..safe]) {
                    return Some(Finish::Cancelled);
                }
                *emitted = safe;
            }
            if out_tokens.len() >= params.max_tokens {
                return Some(Finish::Length);
            }
            None
        };

        'outer: loop {
            if let Some(f) = emit(pending, &mut out_tokens, &mut hist, &mut text, &mut emitted, &mut string_stop) {
                finish = f;
                break;
            }
            let room = params.max_tokens.saturating_sub(out_tokens.len()).min(self.opts.ctx.saturating_sub(hist.len() + 1));
            let mut drafts: Vec<u32> = Vec::new();
            let mut src = 0usize;
            if spec != SpecMode::Off && room > 1 {
                if matches!(spec, SpecMode::Lookup | SpecMode::Auto) {
                    let k = self.tuner.best_k(0, kmax).min(room - 1);
                    if k > 0 {
                        drafts = lookup_draft(&hist, k, 4, 2);
                    }
                }
                if drafts.is_empty() && matches!(spec, SpecMode::Draft | SpecMode::Auto) && self.draft.is_some() {
                    let k = self.tuner.best_k(1, kmax).min(room - 1);
                    if k > 0 {
                        let td = Instant::now();
                        let d = self.draft.as_mut().unwrap();
                        drafts = d.propose(&self.pool, &hist, k, dw, bw, vocab)?;
                        let per = td.elapsed().as_secs_f64() / drafts.len().max(1) as f64;
                        self.tuner.c_draft = if self.tuner.c_draft == 0.0 { per } else { self.tuner.c_draft * 0.8 + per * 0.2 };
                        src = 1;
                    }
                }
            }
            let pos = hist.len() - 1;
            let mut batch = Vec::with_capacity(drafts.len() + 1);
            batch.push(pending);
            batch.extend_from_slice(&drafts);
            let tf = Instant::now();
            let s = &mut self.sessions[si];
            if pos + batch.len() > s.state.n_ctx() {
                finish = Finish::Length;
                break;
            }
            s.tokens.truncate(pos);
            run(&self.pool, self.model.as_ref(), s, &batch, pos, true, dw, bw)?;
            let dt = tf.elapsed().as_secs_f64();
            self.tuner.record_fwd(batch.len(), dt);
            self.thermal.observe(dt / batch.len() as f64, &self.pool);
            passes += 1;
            s.tokens.extend_from_slice(&batch);
            let logits: Vec<f32> = s.state.logits().to_vec();
            let mut n_acc = 0;
            let mut next = None;
            for (i, &d) in drafts.iter().enumerate() {
                let row = &logits[i * vocab..(i + 1) * vocab];
                let (ok, t) = sampler.verify(row, &hist, d);
                if ok {
                    n_acc += 1;
                    if let Some(f) = emit(d, &mut out_tokens, &mut hist, &mut text, &mut emitted, &mut string_stop) {
                        finish = f;
                        drafted += drafts.len();
                        accepted += n_acc;
                        let s = &mut self.sessions[si];
                        let keep = hist.len().min(s.tokens.len());
                        s.tokens.truncate(keep);
                        break 'outer;
                    }
                } else {
                    next = Some(t);
                    break;
                }
            }
            if !drafts.is_empty() {
                let a = n_acc as f64 / drafts.len() as f64;
                let e = &mut self.tuner.alpha[src];
                *e = *e * 0.85 + a * 0.15;
            }
            drafted += drafts.len();
            accepted += n_acc;
            self.sessions[si].tokens.truncate(hist.len());
            pending = match next {
                Some(t) => t,
                None => {
                    let row = &logits[drafts.len() * vocab..(drafts.len() + 1) * vocab];
                    sampler.sample(row, &hist)
                }
            };
            self.thermal.cool();
        }
        drop(emit);
        if finish != Finish::Cancelled && !string_stop {
            text.push_str(&dec.flush());
            if emitted < text.len() {
                on_text(&text[emitted..]);
            }
        }
        self.sessions[si].last_used = Instant::now();
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            cached_tokens: cached,
            prefill_s,
            gen_tokens: out_tokens.len(),
            gen_s: t1.elapsed().as_secs_f64(),
            drafted,
            accepted,
            passes,
            finish,
            tokens: out_tokens,
        })
    }

    /// Clears all cached sessions (frees state memory).
    ///
    /// # Example
    /// ```no_run
    /// # let mut eng = aido::llm::engine::LlmEngine::load("m.gguf", Default::default()).unwrap();
    /// eng.reset();
    /// assert_eq!(eng.cache_bytes(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.sessions.clear();
        if let Some(d) = &mut self.draft {
            d.sess.tokens.clear();
        }
    }

    /// Current estimated draft acceptance rates `(lookup, draft_model)`.
    ///
    /// # Example
    /// ```no_run
    /// # let eng = aido::llm::engine::LlmEngine::load("m.gguf", Default::default()).unwrap();
    /// let (a_lookup, a_draft) = eng.acceptance();
    /// ```
    pub fn acceptance(&self) -> (f64, f64) {
        (self.tuner.alpha[0], self.tuner.alpha[1])
    }

    /// Bytes held by all cached sessions' states (KV caches, recurrent state).
    ///
    /// # Example
    /// ```no_run
    /// # let e: aido::llm::engine::LlmEngine = unimplemented!();
    /// println!("KV: {:.1} MB", e.state_bytes() as f64 / 1e6);
    /// ```
    pub fn state_bytes(&self) -> usize {
        self.sessions.iter().map(|s| s.state.nbytes()).sum()
    }

    /// Formats a chat into prompt token ids using the model's template
    /// (ChatML fallback when the model has none).
    ///
    /// # Example
    /// ```no_run
    /// use aido::llm::{chat::Message, engine::*};
    /// let eng = LlmEngine::load("m.gguf", Default::default()).unwrap();
    /// let ids = eng.chat_prompt(&[Message::new("user", "hi")], false).unwrap();
    /// ```
    pub fn chat_prompt(&self, msgs: &[super::chat::Message], think: bool) -> Result<Vec<u32>> {
        self.chat_prompt_tools(msgs, think, None)
    }

    /// [`LlmEngine::chat_prompt`] with OpenAI-style tool definitions.
    ///
    /// # Example
    /// ```no_run
    /// # let eng: aido::llm::engine::LlmEngine = unimplemented!();
    /// let tools = serde_json::json!([{"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}]);
    /// let ids = eng.chat_prompt_tools(&[aido::llm::chat::Message::new("user", "time?")], false, Some(&tools)).unwrap();
    /// ```
    pub fn chat_prompt_tools(&self, msgs: &[super::chat::Message], think: bool, tools: Option<&serde_json::Value>) -> Result<Vec<u32>> {
        let g = self.model.store();
        let tpl = g.get_str("tokenizer.chat_template");
        let tok_str = |id: Option<u32>| id.and_then(|i| self.tok.vocab.get(i as usize).cloned()).unwrap_or_default();
        let mut text = super::chat::render_with_tools(tpl, msgs, &tok_str(self.tok.bos), &tok_str(self.tok.eos), true, think, tools)?;
        if !think && tpl.is_none() {
            text.push_str(self.model.no_think_suffix());
        }
        let has_bos_text = self.tok.bos.map(|b| text.starts_with(&self.tok.vocab[b as usize])).unwrap_or(false);
        Ok(self.tok.encode(&text, !has_bos_text))
    }
}

impl Draft {
    /// Greedy proposals from the draft model, syncing its state to `hist`.
    fn propose(&mut self, pool: &ThreadPool, hist: &[u32], k: usize, dw: usize, bw: usize, target_vocab: usize) -> Result<Vec<u32>> {
        prefill(pool, self.model.as_ref(), &mut self.sess, hist, dw, bw)?;
        let vocab = self.model.vocab().min(target_vocab);
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let x = argmax(&self.sess.state.logits()[..vocab]);
            out.push(x);
            if i + 1 < k {
                let p = self.sess.tokens.len();
                if p + 1 >= self.sess.state.n_ctx() {
                    break;
                }
                run(pool, self.model.as_ref(), &mut self.sess, &[x], p, false, dw, bw)?;
                self.sess.tokens.push(x);
            }
        }
        Ok(out)
    }
}
