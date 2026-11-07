//! Transformer decoder (Llama / Mistral / Qwen2 / Qwen3 families) running on
//! the CPU thread pool, with optional GPU offload of a prefix of layers.
//!
//! The forward pass is written for a **batch of `t` consecutive tokens** of
//! one sequence. `t = 1` is ordinary decoding; `t > 1` is prompt processing
//! *and* speculative verification: all draft tokens are scored in one pass,
//! sharing every weight fetch from DRAM (see [`crate::tensor::ops::matmul`]).

use anyhow::{bail, Context, Result};

use crate::format::TensorStore;
use crate::runtime::pool::ThreadPool;
use crate::tensor::ops::{self, Rope};
use crate::tensor::quant::{uses_q8, Q8Act};
use crate::tensor::{QWeight, SyncPtr};

/// Hyper-parameters of a decoder-only transformer.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Architecture string from GGUF (`llama`, `qwen2`, `qwen3`, …).
    pub arch: String,
    /// Hidden size.
    pub n_embd: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Query heads.
    pub n_head: usize,
    /// Key/value heads (GQA).
    pub n_head_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// FFN inner size.
    pub n_ff: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// RoPE base θ.
    pub rope_base: f32,
    /// Rotated dimensions per head.
    pub rope_dim: usize,
    /// NeoX rotary layout.
    pub neox: bool,
    /// Vocabulary size.
    pub vocab: usize,
    /// Training context length.
    pub ctx_train: usize,
}

impl LlmConfig {
    /// `n_head_kv * head_dim` — width of one K or V row in the cache.
    ///
    /// # Example
    /// ```
    /// # let c = aido::llm::model::LlmConfig { arch: "llama".into(), n_embd: 64, n_layer: 1, n_head: 4, n_head_kv: 2, head_dim: 16, n_ff: 128, eps: 1e-5, rope_base: 1e4, rope_dim: 16, neox: false, vocab: 10, ctx_train: 128 };
    /// assert_eq!(c.kv_dim(), 32);
    /// ```
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// `n_head * head_dim` — width of the query / attention output.
    ///
    /// # Example
    /// ```
    /// # let c = aido::llm::model::LlmConfig { arch: "llama".into(), n_embd: 64, n_layer: 1, n_head: 4, n_head_kv: 2, head_dim: 16, n_ff: 128, eps: 1e-5, rope_base: 1e4, rope_dim: 16, neox: false, vocab: 10, ctx_train: 128 };
    /// assert_eq!(c.q_dim(), 64);
    /// ```
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
}

/// Weights of one transformer block.
pub struct Layer {
    /// Pre-attention RMSNorm gain.
    pub attn_norm: Vec<f32>,
    /// Query projection.
    pub wq: QWeight,
    /// Key projection.
    pub wk: QWeight,
    /// Value projection.
    pub wv: QWeight,
    /// Output projection.
    pub wo: QWeight,
    /// Optional Q/K/V biases (Qwen2).
    pub bq: Option<Vec<f32>>,
    /// Key bias.
    pub bk: Option<Vec<f32>>,
    /// Value bias.
    pub bv: Option<Vec<f32>>,
    /// Optional per-head RMSNorm of queries (Qwen3).
    pub q_norm: Option<Vec<f32>>,
    /// Optional per-head RMSNorm of keys (Qwen3).
    pub k_norm: Option<Vec<f32>>,
    /// Pre-FFN RMSNorm gain.
    pub ffn_norm: Vec<f32>,
    /// SwiGLU gate projection.
    pub w_gate: QWeight,
    /// SwiGLU up projection.
    pub w_up: QWeight,
    /// FFN down projection.
    pub w_down: QWeight,
}

/// A loaded decoder model (weights are views into the memory-mapped file).
pub struct Model {
    /// Hyper-parameters.
    pub cfg: LlmConfig,
    /// Token embedding table `[vocab, n_embd]`.
    pub tok_embd: QWeight,
    /// Blocks.
    pub layers: Vec<Layer>,
    /// Final RMSNorm gain.
    pub out_norm: Vec<f32>,
    /// LM head `[vocab, n_embd]` (tied to `tok_embd` when absent).
    pub output: QWeight,
    /// RoPE tables.
    pub rope: Rope,
    /// The mapped file (keeps tensor memory alive, exposes metadata).
    pub store: Box<dyn TensorStore>,
    /// Per-layer skip mask: `true` = the block is bypassed (residual
    /// passes through unchanged). Lets users deactivate layers at run time
    /// and measure the effect before physically removing them.
    pub skip: Vec<bool>,
    /// Optional steering vectors added to the residual stream after a block
    /// (`steer.blk.N` tensors in the model file; see [`crate::steer`]).
    pub steer: Vec<Option<Vec<f32>>>,
    /// KV cache policy for new sessions (see [`super::kv`]).
    pub kv_cfg: KvConfig,
}

impl Model {
    /// Loads a model from a GGUF file.
    ///
    /// * **Input** — path to a `.gguf` with architecture `llama`, `mistral`,
    ///   `qwen2`, `qwen3` (others with the same tensor layout usually work).
    /// * **Output** — the model, or an error naming the missing piece.
    /// * **Behaviour** — zero-copy: only norms/biases are materialised.
    ///
    /// # Example
    /// ```no_run
    /// let m = aido::llm::model::Model::load("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// println!("{} layers", m.cfg.n_layer);
    /// ```
    pub fn load(path: &str) -> Result<Self> {
        let g = crate::format::open(path)?;
        let arch = g.arch().to_string();
        if !matches!(arch.as_str(), "llama" | "mistral" | "qwen2" | "qwen3" | "granite" | "internlm2" | "smollm3") {
            eprintln!("warning: architecture '{arch}' not explicitly supported, trying llama layout");
        }
        let k = |s: &str| format!("{arch}.{s}");
        let n_embd = g.get_u64(&k("embedding_length")).context("embedding_length")? as usize;
        let n_layer = g.get_u64(&k("block_count")).context("block_count")? as usize;
        let n_head = g.get_u64(&k("attention.head_count")).context("head_count")? as usize;
        let n_head_kv = g.get_u64(&k("attention.head_count_kv")).unwrap_or(n_head as u64) as usize;
        let head_dim = g.get_u64(&k("attention.key_length")).map(|v| v as usize).unwrap_or(n_embd / n_head);
        let n_ff = g.get_u64(&k("feed_forward_length")).context("feed_forward_length")? as usize;
        let eps = g.get_f64(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5) as f32;
        let rope_base = g.get_f64(&k("rope.freq_base")).unwrap_or(10000.0) as f32;
        let rope_dim = g.get_u64(&k("rope.dimension_count")).map(|v| v as usize).unwrap_or(head_dim);
        let ctx_train = g.get_u64(&k("context_length")).unwrap_or(4096) as usize;
        let neox = matches!(arch.as_str(), "qwen2" | "qwen3" | "qwen2moe" | "phi3" | "gemma" | "gemma2" | "stablelm" | "smollm3");
        let tok_embd = g.weight("token_embd.weight")?;
        let vocab = tok_embd.rows;
        let output = if g.has("output.weight") { g.weight("output.weight")? } else { tok_embd.clone() };
        let factors = if g.has("rope_freqs.weight") { Some(g.vector("rope_freqs.weight")?) } else { None };
        let rope = Rope::new(rope_dim, rope_base, factors.as_deref(), neox);
        let opt = |n: String| -> Result<Option<Vec<f32>>> { if g.has(&n) { Ok(Some(g.vector(&n)?)) } else { Ok(None) } };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            layers.push(Layer {
                attn_norm: g.vector(&p("attn_norm.weight"))?,
                wq: g.weight(&p("attn_q.weight"))?,
                wk: g.weight(&p("attn_k.weight"))?,
                wv: g.weight(&p("attn_v.weight"))?,
                wo: g.weight(&p("attn_output.weight"))?,
                bq: opt(p("attn_q.bias"))?,
                bk: opt(p("attn_k.bias"))?,
                bv: opt(p("attn_v.bias"))?,
                q_norm: opt(p("attn_q_norm.weight"))?,
                k_norm: opt(p("attn_k_norm.weight"))?,
                ffn_norm: g.vector(&p("ffn_norm.weight"))?,
                w_gate: g.weight(&p("ffn_gate.weight"))?,
                w_up: g.weight(&p("ffn_up.weight"))?,
                w_down: g.weight(&p("ffn_down.weight"))?,
            });
        }
        for w in [&layers[0].wq, &layers[0].w_down, &output] {
            if w.cols % 32 != 0 {
                bail!("row length {} is not a multiple of 32", w.cols);
            }
        }
        let cfg = LlmConfig { arch, n_embd, n_layer, n_head, n_head_kv, head_dim, n_ff, eps, rope_base, rope_dim, neox, vocab, ctx_train };
        let out_norm = g.vector("output_norm.weight")?;
        let steer = (0..cfg.n_layer).map(|i| g.vector(&format!("steer.blk.{i}")).ok()).collect();
        Ok(Model { skip: vec![false; cfg.n_layer], steer, kv_cfg: KvConfig::default(), cfg, tok_embd, layers, out_norm, output, rope, store: g })
    }

    /// Bytes of weights streamed per generated token (all layers + head).
    ///
    /// This is the numerator of the decoding roofline
    /// `tokens/s ≤ bandwidth / weight_bytes_per_token`.
    ///
    /// # Example
    /// ```no_run
    /// # let m = aido::llm::model::Model::load("m.gguf").unwrap();
    /// println!("{:.2} GB/token", m.weight_bytes_per_token() as f64 / 1e9);
    /// ```
    pub fn weight_bytes_per_token(&self) -> usize {
        let l: usize = self
            .layers
            .iter()
            .map(|l| l.wq.nbytes() + l.wk.nbytes() + l.wv.nbytes() + l.wo.nbytes() + l.w_gate.nbytes() + l.w_up.nbytes() + l.w_down.nbytes())
            .sum();
        l + self.output.nbytes()
    }

    /// Short description, e.g. `qwen3 28L d=1024 heads=16/8 Q8_0`.
    ///
    /// # Example
    /// ```no_run
    /// # let m = aido::llm::model::Model::load("m.gguf").unwrap();
    /// println!("{}", m.describe());
    /// ```
    pub fn describe(&self) -> String {
        let c = &self.cfg;
        format!(
            "{} {}L d={} heads={}/{} hd={} ff={} vocab={} wq={:?} ffn={:?} head={:?}",
            c.arch, c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim, c.n_ff, c.vocab, self.layers[0].wq.ty, self.layers[0].w_down.ty, self.output.ty
        )
    }
}

pub use super::kv::{KvCache, KvConfig, KvType};

/// Pre-allocated activations for batches of up to `max_t` tokens; the hot
/// loop never allocates.
pub struct Workspace {
    max_t: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    xq: Vec<Q8Act>,
    fq: Vec<Q8Act>,
    scores: Vec<Vec<f32>>,
    cs: Vec<f32>,
    /// Logits of the last forward, `n_out × vocab`.
    pub logits: Vec<f32>,
    /// Optional statistics collector (analysis mode; `None` in production).
    pub probe: Option<Probe>,
}

/// Activation statistics collected during forward passes, the raw material
/// for data-driven pruning:
///
/// * `ffn_abs[l][j]` — Σ |silu(gate)·up| of FFN neuron `j` in layer `l`
///   (a neuron whose output is always ≈0 can be removed with no effect);
/// * `ffn_max[l][j]` — max magnitude seen (guards against rare-but-large
///   "massive activation" neurons that must never be pruned);
/// * `block_cos[l]` — Σ cos(x_in, x_out) of layer `l` over tokens; the
///   *block influence* `BI = 1 − mean cos` measures how much a layer
///   changes the residual stream (layers with BI ≈ 0 are redundant).
#[derive(Debug, Clone)]
pub struct Probe {
    /// Σ |activation| per layer and neuron.
    pub ffn_abs: Vec<Vec<f64>>,
    /// Max |activation| per layer and neuron.
    pub ffn_max: Vec<Vec<f32>>,
    /// Σ cosine(x_in, x_out) per layer.
    pub block_cos: Vec<f64>,
    /// Tokens observed.
    pub tokens: u64,
    /// Residual stream after each block for the last token of the last pass.
    pub resid: Vec<Vec<f32>>,
    x_in: Vec<f32>,
}

impl Probe {
    /// Creates an empty probe for a model shape.
    ///
    /// # Example
    /// ```
    /// let p = aido::llm::model::Probe::new(2, 8);
    /// assert_eq!(p.ffn_abs[1].len(), 8);
    /// ```
    pub fn new(n_layer: usize, n_ff: usize) -> Self {
        Probe { ffn_abs: vec![vec![0.0; n_ff]; n_layer], ffn_max: vec![vec![0.0; n_ff]; n_layer], block_cos: vec![0.0; n_layer], tokens: 0, resid: vec![Vec::new(); n_layer], x_in: Vec::new() }
    }

    fn record_ffn(&mut self, l: usize, act: &[f32], ff: usize) {
        let (s, m) = (&mut self.ffn_abs[l], &mut self.ffn_max[l]);
        for row in act.chunks_exact(ff) {
            for j in 0..ff {
                let a = row[j].abs();
                s[j] += a as f64;
                if a > m[j] {
                    m[j] = a;
                }
            }
        }
        if l == 0 {
            self.tokens += (act.len() / ff) as u64;
        }
    }

    fn record_block(&mut self, l: usize, x_out: &[f32], d: usize) {
        self.resid[l] = x_out[x_out.len() - d..].to_vec();
        for (a, b) in self.x_in.chunks_exact(d).zip(x_out.chunks_exact(d)) {
            let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
            for i in 0..d {
                ab += (a[i] * b[i]) as f64;
                aa += (a[i] * a[i]) as f64;
                bb += (b[i] * b[i]) as f64;
            }
            self.block_cos[l] += ab / (aa.sqrt() * bb.sqrt() + 1e-12);
        }
    }

    /// Block influence `BI_l = 1 − mean cos(x_in, x_out)` per layer.
    ///
    /// # Example
    /// ```
    /// let p = aido::llm::model::Probe::new(1, 4);
    /// assert_eq!(p.block_influence().len(), 1);
    /// ```
    pub fn block_influence(&self) -> Vec<f64> {
        let n = self.tokens.max(1) as f64;
        self.block_cos.iter().map(|c| 1.0 - c / n).collect()
    }
}

impl Workspace {
    /// Allocates buffers for `max_t` tokens and contexts up to `n_ctx`.
    ///
    /// # Example
    /// ```no_run
    /// # let m = aido::llm::model::Model::load("m.gguf").unwrap();
    /// let ws = aido::llm::model::Workspace::new(&m.cfg, 64, 4096, 8);
    /// ```
    pub fn new(c: &LlmConfig, max_t: usize, n_ctx: usize, n_threads: usize) -> Self {
        let g = c.n_head / c.n_head_kv;
        Workspace {
            max_t,
            x: vec![0.0; max_t * c.n_embd],
            xb: vec![0.0; max_t * c.n_embd.max(c.q_dim())],
            q: vec![0.0; max_t * c.q_dim()],
            k: vec![0.0; max_t * c.kv_dim()],
            v: vec![0.0; max_t * c.kv_dim()],
            att: vec![0.0; max_t * c.q_dim()],
            gate: vec![0.0; max_t * c.n_ff],
            up: vec![0.0; max_t * c.n_ff],
            xq: vec![Q8Act::default(); max_t],
            fq: vec![Q8Act::default(); max_t],
            scores: (0..n_threads).map(|_| vec![0.0; g * n_ctx]).collect(),
            cs: vec![0.0; max_t * c.rope_dim],
            logits: Vec::new(),
            probe: None,
        }
    }

    /// Largest batch this workspace accepts.
    ///
    /// # Example
    /// ```no_run
    /// # let m = aido::llm::model::Model::load("m.gguf").unwrap();
    /// # let ws = aido::llm::model::Workspace::new(&m.cfg, 64, 4096, 8);
    /// assert_eq!(ws.max_t(), 64);
    /// ```
    pub fn max_t(&self) -> usize {
        self.max_t
    }
}

/// Several matmuls that read the same input fused into ONE parallel region
/// (e.g. Q, K, V or gate, up). Halves/thirds the number of thread
/// synchronisations per layer, which dominates small-model decode latency.
pub(crate) fn matmul_fused(pool: &ThreadPool, ws: &[(&QWeight, SyncPtr<f32>)], xs: &[f32], xq: &[Q8Act], t: usize) {
    let threads = pool.width().max(1);
    let total_rows: usize = ws.iter().map(|w| w.0.rows).sum();
    let chunk = ((total_rows / (threads * 8)).max(4) + 3) / 4 * 4;
    // Build task table: (weight index, row start, row end).
    let mut tasks: Vec<(usize, usize, usize)> = Vec::with_capacity(total_rows / chunk + ws.len());
    for (i, (w, _)) in ws.iter().enumerate() {
        let mut r = 0;
        while r < w.rows {
            tasks.push((i, r, (r + chunk).min(w.rows)));
            r += chunk;
        }
    }
    let empty = Q8Act::default();
    pool.parallel_for(tasks.len(), |ti, _| {
        let (wi, r0, r1) = tasks[ti];
        let (w, out) = &ws[wi];
        let f = crate::tensor::quant::dot_fn(w.ty);
        let need_q = uses_q8(w.ty);
        let rb = w.ty.row_bytes(w.cols);
        let data = w.data.as_slice();
        let cols = w.cols;
        let f4 = if need_q { crate::tensor::quant::dot4_fn(w.ty) } else { None };
        let mut tok = 0;
        // Register-blocked groups of four tokens share each weight load.
        if let Some(f4) = f4 {
            while tok + 4 <= t {
                for r in r0..r1 {
                    let v = f4(&data[r * rb..(r + 1) * rb], &xq[tok..tok + 4]);
                    for (k, val) in v.iter().enumerate() {
                        unsafe { *out.ptr().add((tok + k) * w.rows + r) = *val };
                    }
                }
                tok += 4;
            }
        }
        for tok in tok..t {
            let x = &xs[tok * cols..(tok + 1) * cols];
            let q = if need_q { &xq[tok] } else { &empty };
            for r in r0..r1 {
                let v = f(&data[r * rb..(r + 1) * rb], x, q);
                unsafe { *out.ptr().add(tok * w.rows + r) = v };
            }
        }
    });
}

/// Applies `f(token_index, slice)` to each of `t` rows of width `w`,
/// in parallel when `t` is large enough to pay for it.
fn per_token<F: Fn(usize, &mut [f32]) + Sync>(pool: &ThreadPool, buf: &mut [f32], t: usize, w: usize, f: F) {
    if t < 4 {
        for i in 0..t {
            f(i, &mut buf[i * w..(i + 1) * w]);
        }
    } else {
        let p = SyncPtr::new(buf);
        pool.parallel_for(t, |i, _| f(i, unsafe { p.slice(i * w, w) }));
    }
}

impl Model {
    /// Runs the transformer on `tokens` placed at positions `pos0..pos0+t`,
    /// updating the KV cache, and writes logits into `ws.logits`.
    ///
    /// * **Input** — `pool`; `tokens` (`1..=ws.max_t()` ids); `pos0`: number of
    ///   tokens already in `kv`; `kv`; `ws`; `all_logits`: `true` → logits
    ///   for every token (speculative verification), `false` → only the last
    ///   (prompt processing / plain decoding).
    /// * **Output** — `ws.logits` holds `n_out × vocab` floats; returns `n_out`.
    /// * **Errors** — if the context is full.
    ///
    /// # Example
    /// ```no_run
    /// use aido::{llm::model::*, runtime::pool::ThreadPool};
    /// let m = Model::load("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// let pool = ThreadPool::new(8, None);
    /// let mut kv = KvCache::new(m.cfg.n_layer, 512, m.cfg.kv_dim());
    /// let mut ws = Workspace::new(&m.cfg, 32, 512, pool.n_threads());
    /// let n = m.forward(&pool, &[1, 2, 3], 0, &mut kv, &mut ws, false).unwrap();
    /// assert_eq!(n, 1);
    /// ```
    pub fn forward(&self, pool: &ThreadPool, tokens: &[u32], pos0: usize, kv: &mut KvCache, ws: &mut Workspace, all_logits: bool) -> Result<usize> {
        let c = &self.cfg;
        let t = tokens.len();
        anyhow::ensure!(t >= 1 && t <= ws.max_t, "batch of {t} tokens exceeds workspace ({})", ws.max_t);
        anyhow::ensure!(pos0 + t <= kv.n_ctx, "context full ({} tokens)", kv.n_ctx);
        kv.reserve(pos0 + t);
        let (d, qd, kvd, hd) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim);
        // Embeddings.
        for (i, &tk) in tokens.iter().enumerate() {
            self.tok_embd.dequant_row(tk as usize % c.vocab, &mut ws.x[i * d..(i + 1) * d]);
        }
        // RoPE tables for the batch.
        let rd = c.rope_dim;
        for i in 0..t {
            self.rope.table(pos0 + i, &mut ws.cs[i * rd..(i + 1) * rd]);
        }
        for (li, l) in self.layers.iter().enumerate() {
            if self.skip[li] {
                continue;
            }
            if let Some(p) = ws.probe.as_mut() {
                p.x_in.clear();
                p.x_in.extend_from_slice(&ws.x[..t * d]);
            }
            self.layer_cpu(pool, li, l, t, pos0, kv, ws, d, qd, kvd, hd, l.w_gate.rows);
            if let Some(v) = &self.steer[li] {
                for row in ws.x[..t * d].chunks_exact_mut(d) {
                    ops::add_inplace(row, v);
                }
            }
            if let Some(p) = ws.probe.as_mut() {
                p.record_block(li, &ws.x[..t * d], d);
            }
        }
        // Final norm + LM head on the requested rows.
        let n_out = if all_logits { t } else { 1 };
        let first = t - n_out;
        let x_out = &mut ws.xb[..n_out * d];
        for i in 0..n_out {
            ops::rmsnorm(&mut x_out[i * d..(i + 1) * d], &ws.x[(first + i) * d..(first + i + 1) * d], &self.out_norm, c.eps);
        }
        ops::quantize_rows(pool, x_out, d, &mut ws.xq);
        ws.logits.resize(n_out * c.vocab, 0.0);
        matmul_fused(pool, &[(&self.output, SyncPtr::new(&mut ws.logits))], x_out, &ws.xq[..n_out], n_out);
        Ok(n_out)
    }

    /// One transformer block on the CPU.
    #[allow(clippy::too_many_arguments)]
    fn layer_cpu(&self, pool: &ThreadPool, li: usize, l: &Layer, t: usize, pos0: usize, kv: &mut KvCache, ws: &mut Workspace, d: usize, qd: usize, kvd: usize, hd: usize, ff: usize) {
        let c = &self.cfg;
        let eps = c.eps;
        // --- attention ---
        {
            let (x, xb) = (&ws.x, &mut ws.xb);
            per_token(pool, &mut xb[..t * d], t, d, |i, o| ops::rmsnorm(o, &x[i * d..(i + 1) * d], &l.attn_norm, eps));
        }
        ops::quantize_rows(pool, &ws.xb[..t * d], d, &mut ws.xq);
        matmul_fused(
            pool,
            &[(&l.wq, SyncPtr::new(&mut ws.q)), (&l.wk, SyncPtr::new(&mut ws.k)), (&l.wv, SyncPtr::new(&mut ws.v))],
            &ws.xb[..t * d],
            &ws.xq[..t],
            t,
        );
        // Bias, QK-norm, RoPE, cache write — per token.
        {
            let rd = c.rope_dim;
            let cs = &ws.cs;
            let rope = &self.rope;
            let kptr = SyncPtr::new(&mut ws.k);
            let vptr = SyncPtr::new(&mut ws.v);
            let kvr: &KvCache = &*kv;
            per_token(pool, &mut ws.q[..t * qd], t, qd, |i, q| {
                let k = unsafe { kptr.slice(i * kvd, kvd) };
                let v = unsafe { vptr.slice(i * kvd, kvd) };
                if let Some(b) = &l.bq {
                    ops::add_inplace(q, b);
                }
                if let Some(b) = &l.bk {
                    ops::add_inplace(k, b);
                }
                if let Some(b) = &l.bv {
                    ops::add_inplace(v, b);
                }
                if let Some(nq) = &l.q_norm {
                    for h in q.chunks_exact_mut(hd) {
                        ops::rmsnorm_inplace(h, nq, eps);
                    }
                }
                if let Some(nk) = &l.k_norm {
                    for h in k.chunks_exact_mut(hd) {
                        ops::rmsnorm_inplace(h, nk, eps);
                    }
                }
                let tab = &cs[i * rd..(i + 1) * rd];
                rope.apply(q, hd, tab);
                rope.apply(k, hd, tab);
                // Distinct positions per task: concurrent writes never overlap.
                unsafe { kvr.write(li, pos0 + i, k, v) };
            });
        }
        kv.summarize(li, pos0, &ws.k[..t * kvd]);
        // Grouped attention. Prefill: one task per (token, kv-head). Decode
        // (few tokens × few KV heads < threads): split-K over the context so
        // every core streams a share of the cache, merged by log-sum-exp.
        {
            let g = c.n_head / c.n_head_kv;
            let nkv = c.n_head_kv;
            let scale = 1.0 / (hd as f32).sqrt();
            let kvr: &KvCache = &*kv;
            let q = &ws.q;
            let att = SyncPtr::new(&mut ws.att);
            let sc = SyncPtr::new(&mut ws.scores);
            let nth = pool.n_threads();
            let ctx_end = pos0 + t;
            let split = if t * nkv < nth && ctx_end >= 1024 { (nth / (t * nkv)).max(1).min(ctx_end / 512) } else { 1 };
            const TQ: usize = 16;
            if t >= 8 {
                // Prefill: block flash-attention, one task per (query block, kv head).
                let nb = t.div_ceil(TQ);
                pool.parallel_for(nb * nkv, |task, _| {
                    let (b, h) = (task / nkv, task % nkv);
                    let (j0, j1) = (b * TQ, ((b + 1) * TQ).min(t));
                    let qs = &q[j0 * qd + h * g * hd..];
                    let out = unsafe { att.slice(j0 * qd + h * g * hd, (j1 - j0 - 1) * qd + g * hd) };
                    kvr.attend_block(li, h * hd, qs, hd, pos0 + j0, j1 - j0, qd, g * hd, scale, out, qd);
                });
            } else if split <= 1 {
                pool.parallel_for(t * nkv, |task, w| {
                    let (i, h) = (task / nkv, task % nkv);
                    let n_ctx = pos0 + i + 1;
                    let scores = unsafe { &mut *sc.ptr().add(w) };
                    let qs = &q[i * qd + h * g * hd..i * qd + (h + 1) * g * hd];
                    let out = unsafe { att.slice(i * qd + h * g * hd, g * hd) };
                    kvr.attend(li, h * hd, qs, hd, n_ctx, scale, &mut scores[..g * n_ctx], out);
                });
            } else {
                let per = g * hd + 2 * g;
                let mut part = vec![0f32; t * nkv * split * per];
                let pp = SyncPtr::new(&mut part);
                pool.parallel_for(t * nkv * split, |task, w| {
                    let (ih, sp) = (task / split, task % split);
                    let (i, h) = (ih / nkv, ih % nkv);
                    let n_ctx = pos0 + i + 1;
                    let scores = unsafe { &mut *sc.ptr().add(w) };
                    let qs = &q[i * qd + h * g * hd..i * qd + (h + 1) * g * hd];
                    let buf = unsafe { pp.slice(task * per, per) };
                    let (o, st) = buf.split_at_mut(g * hd);
                    kvr.attend_part(li, h * hd, qs, hd, n_ctx, scale, sp, split, scores, o, st);
                });
                for ih in 0..t * nkv {
                    let (i, h) = (ih / nkv, ih % nkv);
                    let refs: Vec<(&[f32], &[f32])> = (0..split)
                        .map(|sp| {
                            let b = &part[(ih * split + sp) * per..(ih * split + sp + 1) * per];
                            b.split_at(g * hd)
                        })
                        .collect();
                    super::kv::merge_parts(&refs, hd, &mut ws.att[i * qd + h * g * hd..i * qd + (h + 1) * g * hd]);
                }
            }
        }
        ops::quantize_rows(pool, &ws.att[..t * qd], qd, &mut ws.fq);
        matmul_fused(pool, &[(&l.wo, SyncPtr::new(&mut ws.xb))], &ws.att[..t * qd], &ws.fq[..t], t);
        ops::add_inplace(&mut ws.x[..t * d], &ws.xb[..t * d]);
        // --- feed-forward ---
        {
            let (x, xb) = (&ws.x, &mut ws.xb);
            per_token(pool, &mut xb[..t * d], t, d, |i, o| ops::rmsnorm(o, &x[i * d..(i + 1) * d], &l.ffn_norm, eps));
        }
        ops::quantize_rows(pool, &ws.xb[..t * d], d, &mut ws.xq);
        matmul_fused(pool, &[(&l.w_gate, SyncPtr::new(&mut ws.gate)), (&l.w_up, SyncPtr::new(&mut ws.up))], &ws.xb[..t * d], &ws.xq[..t], t);
        {
            let up = &ws.up;
            per_token(pool, &mut ws.gate[..t * ff], t, ff, |i, g| ops::silu_mul(g, &up[i * ff..(i + 1) * ff]));
        }
        if let Some(p) = ws.probe.as_mut() {
            p.record_ffn(li, &ws.gate[..t * ff], ff);
        }
        ops::quantize_rows(pool, &ws.gate[..t * ff], ff, &mut ws.fq);
        matmul_fused(pool, &[(&l.w_down, SyncPtr::new(&mut ws.xb))], &ws.gate[..t * ff], &ws.fq[..t], t);
        ops::add_inplace(&mut ws.x[..t * d], &ws.xb[..t * d]);
    }
}

/// Llama-family per-sequence state: KV cache + workspace.
pub struct LlamaState {
    /// KV cache.
    pub kv: KvCache,
    /// Activations.
    pub ws: Workspace,
}

impl super::ModelState for LlamaState {
    fn logits(&self) -> &[f32] {
        &self.ws.logits
    }
    fn max_t(&self) -> usize {
        self.ws.max_t()
    }
    fn n_ctx(&self) -> usize {
        self.kv.n_ctx
    }
    fn nbytes(&self) -> usize {
        self.kv.nbytes()
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl super::LanguageModel for Model {
    fn describe(&self) -> String {
        Model::describe(self)
    }
    fn vocab(&self) -> usize {
        self.cfg.vocab
    }
    fn store(&self) -> &dyn TensorStore {
        self.store.as_ref()
    }
    fn tokenizer(&self) -> Result<crate::tokenizer::Tokenizer> {
        crate::tokenizer::Tokenizer::from_gguf(self.store.as_ref())
    }
    fn weight_bytes_per_token(&self) -> usize {
        Model::weight_bytes_per_token(self)
    }
    fn new_state(&self, n_ctx: usize, max_t: usize, n_threads: usize) -> Box<dyn super::ModelState> {
        let mut cfg = self.cfg.clone();
        cfg.n_ff = self.layers.iter().map(|l| l.w_gate.rows).max().unwrap_or(cfg.n_ff);
        Box::new(LlamaState { kv: KvCache::with_config(cfg.n_layer, n_ctx, cfg.kv_dim(), self.kv_cfg.clone()), ws: Workspace::new(&cfg, max_t.max(1), n_ctx, n_threads) })
    }
    fn forward(&self, pool: &ThreadPool, tokens: &[u32], pos0: usize, st: &mut dyn super::ModelState, all_logits: bool) -> Result<usize> {
        let s = st.as_any_mut().downcast_mut::<LlamaState>().ok_or_else(|| anyhow::anyhow!("state type mismatch"))?;
        Model::forward(self, pool, tokens, pos0, &mut s.kv, &mut s.ws, all_logits)
    }
    fn no_think_suffix(&self) -> &str {
        ""
    }
    fn set_skip(&mut self, layers: &[usize]) -> Result<()> {
        for &l in layers {
            anyhow::ensure!(l < self.cfg.n_layer, "layer {l} out of range");
            self.skip[l] = true;
        }
        Ok(())
    }
}

/// Batched `out[t] = W·x[t]` for `t` rows on the CPU integer path
/// (activation quantised once, 4-token register blocking).
///
/// # Example
/// ```
/// use aido::{runtime::pool::ThreadPool, tensor::QWeight};
/// let pool = ThreadPool::new(2, None);
/// let w = QWeight::from_f32(2, 32, &[1.0; 64]);
/// let mut out = vec![0f32; 2];
/// aido::llm::model::linear_rows(&pool, &w, &[0.5; 32], 1, &mut out);
/// assert!((out[0] - 16.0).abs() < 0.2);
/// ```
pub fn linear_rows(pool: &ThreadPool, w: &QWeight, xs: &[f32], t: usize, out: &mut [f32]) {
    let mut xq = vec![Q8Act::default(); t];
    if uses_q8(w.ty) {
        ops::quantize_rows(pool, &xs[..t * w.cols], w.cols, &mut xq);
    }
    matmul_fused(pool, &[(w, SyncPtr::new(out))], xs, &xq, t);
}
