//! Ready-made trainable architectures.
//!
//! * [`LmModel`] — Llama/Qwen-style decoder (RMSNorm, RoPE, SwiGLU, optional
//!   QKV bias and per-head QK-norm). Trainable from scratch, importable from
//!   any supported GGUF / `.aido` for **fine-tuning** (full or **LoRA**),
//!   exportable to an engine-runnable `.aido` (optionally quantised).
//! * [`Classifier`] — MLP classifier on feature vectors.
//! * [`Ddpm`] — tiny denoising-diffusion image generator (MLP ε-predictor).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::autograd::{Graph, Id, Params};
use super::Rng;
use crate::format::AidoWriter;
use crate::runtime::pool::ThreadPool;
use crate::tensor::quant::GgmlType;
use crate::tensor::quantize::quantize_row;

/// Decoder hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmConfig {
    /// Vocabulary size.
    pub vocab: usize,
    /// Hidden size.
    pub dim: usize,
    /// Blocks.
    pub layers: usize,
    /// Attention heads (no GQA in SDK-trained models; imported models expand KV heads).
    pub heads: usize,
    /// KV heads.
    pub kv_heads: usize,
    /// FFN width.
    pub ff: usize,
    /// Training context.
    pub ctx: usize,
    /// RoPE θ.
    pub rope_base: f32,
    /// RMSNorm ε.
    pub eps: f32,
    /// NeoX RoPE layout (Qwen) vs pairs (Llama).
    pub neox: bool,
    /// Architecture name written on export.
    pub arch: String,
    /// LoRA rank (0 = full training).
    pub lora: usize,
}

/// A decoder model: config + parameters.
pub struct LmModel {
    /// Configuration.
    pub cfg: LmConfig,
    /// Parameters (GGUF tensor names).
    pub params: Params,
    /// Source metadata to carry over on export (tokenizer, template…).
    pub source_meta: Option<std::collections::HashMap<String, crate::gguf::Value>>,
}

impl LmModel {
    /// Creates a randomly initialised byte-level model (vocab 257 = 256
    /// bytes + `<|endoftext|>`).
    ///
    /// # Example
    /// ```
    /// use aido::sdk::models::{LmModel, LmConfig};
    /// let m = LmModel::new_bytes(64, 2, 2, 128, 64, 1);
    /// assert!(m.params.n_trainable() > 0);
    /// ```
    pub fn new_bytes(dim: usize, layers: usize, heads: usize, ff: usize, ctx: usize, seed: u64) -> Self {
        Self::new_vocab(257, dim, layers, heads, ff, ctx, seed)
    }

    /// Randomly initialised model with an arbitrary vocabulary size.
    ///
    /// # Example
    /// ```
    /// let m = aido::sdk::models::LmModel::new_vocab(16, 32, 1, 2, 64, 32, 1);
    /// assert_eq!(m.cfg.vocab, 16);
    /// ```
    pub fn new_vocab(vocab: usize, dim: usize, layers: usize, heads: usize, ff: usize, ctx: usize, seed: u64) -> Self {
        let cfg = LmConfig { vocab, dim, layers, heads, kv_heads: heads, ff, ctx, rope_base: 10000.0, eps: 1e-5, neox: false, arch: "llama".into(), lora: 0 };
        let mut r = Rng(seed | 1);
        let mut p = Params::default();
        let std = 0.02;
        p.add("token_embd.weight", cfg.vocab, dim, r.normals(cfg.vocab * dim, std), true);
        let resid = std / (2.0 * layers as f32).sqrt();
        for l in 0..layers {
            let n = |s: &str| format!("blk.{l}.{s}");
            p.add(&n("attn_norm.weight"), 1, dim, vec![1.0; dim], true);
            for w in ["attn_q", "attn_k", "attn_v"] {
                p.add(&n(&format!("{w}.weight")), dim, dim, r.normals(dim * dim, std), true);
            }
            p.add(&n("attn_output.weight"), dim, dim, r.normals(dim * dim, resid), true);
            p.add(&n("ffn_norm.weight"), 1, dim, vec![1.0; dim], true);
            p.add(&n("ffn_gate.weight"), ff, dim, r.normals(ff * dim, std), true);
            p.add(&n("ffn_up.weight"), ff, dim, r.normals(ff * dim, std), true);
            p.add(&n("ffn_down.weight"), dim, ff, r.normals(ff * dim, resid), true);
        }
        p.add("output_norm.weight", 1, dim, vec![1.0; dim], true);
        p.add("output.weight", cfg.vocab, dim, r.normals(cfg.vocab * dim, std), true);
        LmModel { cfg, params: p, source_meta: None }
    }

    /// Imports a GGUF / `.aido` Llama- or Qwen-family model as f32
    /// parameters for fine-tuning. With `lora > 0` the base weights are
    /// frozen and rank-`lora` adapters are added to every projection.
    ///
    /// # Example
    /// ```no_run
    /// let m = aido::sdk::models::LmModel::import("models/Qwen3-0.6B-Q8_0.gguf", 8, 256).unwrap();
    /// ```
    pub fn import(path: &str, lora: usize, ctx: usize) -> Result<Self> {
        let st = crate::format::open(path)?;
        let arch = st.arch().to_string();
        let k = |s: &str| format!("{arch}.{s}");
        let dim = st.get_u64(&k("embedding_length")).context("embedding_length")? as usize;
        let layers = st.get_u64(&k("block_count")).context("block_count")? as usize;
        let heads = st.get_u64(&k("attention.head_count")).context("head_count")? as usize;
        let kv_heads = st.get_u64(&k("attention.head_count_kv")).unwrap_or(heads as u64) as usize;
        let ff = st.get_u64(&k("feed_forward_length")).context("ff")? as usize;
        let hd = st.get_u64(&k("attention.key_length")).map(|v| v as usize).unwrap_or(dim / heads);
        anyhow::ensure!(hd * heads == dim, "fine-tuning needs head_dim·heads == hidden (got {hd}·{heads} vs {dim})");
        let mut p = Params::default();
        let train_base = lora == 0;
        let mut r = Rng(1234);
        for name in st.names() {
            let (_, dims) = st.info(&name).context("info")?;
            let cols = dims[0];
            let rows = dims[1..].iter().product::<usize>().max(1);
            let v = st.vector(&name)?;
            let is_matrix = dims.len() >= 2 && name != "token_embd.weight" && name != "rope_freqs.weight";
            p.add(&name, rows, cols, v, train_base || (!is_matrix && name.ends_with("norm.weight")));
            if lora > 0 && is_matrix && name.starts_with("blk.") {
                p.add(&format!("{name}.lora_a"), lora, cols, r.normals(lora * cols, 1.0 / (cols as f32).sqrt()), true);
                p.add(&format!("{name}.lora_b"), rows, lora, vec![0.0; rows * lora], true);
            }
        }
        if p.find("output.weight").is_none() {
            let e = p.find("token_embd.weight").context("embeddings")?;
            let (rr, cc) = p.shapes[e];
            let d = p.data[e].clone();
            p.add("output.weight", rr, cc, d, train_base);
        }
        let vocab = p.shapes[p.find("token_embd.weight").unwrap()].0;
        let neox = matches!(arch.as_str(), "qwen2" | "qwen3");
        let cfg = LmConfig {
            vocab,
            dim,
            layers,
            heads,
            kv_heads,
            ff,
            ctx,
            rope_base: st.get_f64(&k("rope.freq_base")).unwrap_or(10000.0) as f32,
            eps: st.get_f64(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5) as f32,
            neox,
            arch,
            lora,
        };
        Ok(LmModel { cfg, params: p, source_meta: Some(st.meta().clone()) })
    }

    fn lin(&self, g: &mut Graph, x: Id, name: &str) -> Id {
        let w = g.params.find(name).unwrap_or_else(|| panic!("missing {name}"));
        let b = g.params.find(&name.replace(".weight", ".bias"));
        let lora = match (g.params.find(&format!("{name}.lora_a")), g.params.find(&format!("{name}.lora_b"))) {
            (Some(a), Some(bb)) => Some((a, bb, 2.0)),
            _ => None,
        };
        g.linear(x, w, b, lora)
    }

    /// Repeats KV heads for grouped-query attention (`[t, kv·hd] → [t, h·hd]`).
    fn expand_kv(&self, g: &mut Graph, x: Id) -> Id {
        let c = &self.cfg;
        if c.kv_heads == c.heads {
            return x;
        }
        let hd = c.dim / c.heads;
        let rep = c.heads / c.kv_heads;
        let (rows, _) = g.shape(x);
        // Build with a constant selection matrix: x·Sᵀ where S picks heads.
        let mut sel = vec![0f32; c.dim * c.kv_heads * hd];
        for h in 0..c.heads {
            for e in 0..hd {
                sel[(h * hd + e) * c.kv_heads * hd + (h / rep) * hd + e] = 1.0;
            }
        }
        let pid = g.params.find("__kv_expand").unwrap_or_else(|| g.params.add("__kv_expand", c.dim, c.kv_heads * hd, sel, false));
        let _ = rows;
        g.linear(x, pid, None, None)
    }

    /// Builds the forward pass for `nseq` sequences of length `seq`
    /// (flattened `ids`) and returns the logits node.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::{models::LmModel, autograd::Graph};
    /// let pool = aido::runtime::pool::ThreadPool::new(2, None);
    /// let mut m = LmModel::new_bytes(32, 1, 2, 64, 8, 1);
    /// let mut params = std::mem::take(&mut m.params);
    /// let mut g = Graph::new(&mut params, &pool);
    /// let logits = m.forward(&mut g, &[1, 2, 3, 4], 4);
    /// assert_eq!(g.shape(logits), (4, 257));
    /// ```
    pub fn forward(&self, g: &mut Graph, ids: &[u32], seq: usize) -> Id {
        let c = &self.cfg;
        let hd = c.dim / c.heads;
        let rows = ids.len();
        let emb = g.params.find("token_embd.weight").unwrap();
        let mut x = g.embed(emb, ids);
        for l in 0..c.layers {
            let n = |s: &str| format!("blk.{l}.{s}");
            let an = g.params.find(&n("attn_norm.weight")).unwrap();
            let h = g.rmsnorm(x, an, c.eps);
            let mut q = self.lin(g, h, &n("attn_q.weight"));
            let mut k = self.lin(g, h, &n("attn_k.weight"));
            let v = self.lin(g, h, &n("attn_v.weight"));
            if let Some(qn) = g.params.find(&n("attn_q_norm.weight")) {
                let r = g.reshape(q, rows * c.heads, hd);
                let r = g.rmsnorm(r, qn, c.eps);
                q = g.reshape(r, rows, c.dim);
            }
            if let Some(kn) = g.params.find(&n("attn_k_norm.weight")) {
                let r = g.reshape(k, rows * c.kv_heads, hd);
                let r = g.rmsnorm(r, kn, c.eps);
                k = g.reshape(r, rows, c.kv_heads * hd);
            }
            let q = g.rope_ex(q, c.heads, seq, c.rope_base, c.neox);
            let k = g.rope_ex(k, c.kv_heads, seq, c.rope_base, c.neox);
            let k = self.expand_kv(g, k);
            let v = self.expand_kv(g, v);
            let a = g.causal_attention(q, k, v, c.heads, seq);
            let o = self.lin(g, a, &n("attn_output.weight"));
            x = g.add(x, o);
            let fnn = g.params.find(&n("ffn_norm.weight")).unwrap();
            let h = g.rmsnorm(x, fnn, c.eps);
            let gate = self.lin(g, h, &n("ffn_gate.weight"));
            let up = self.lin(g, h, &n("ffn_up.weight"));
            let s = g.swiglu(gate, up);
            let d = self.lin(g, s, &n("ffn_down.weight"));
            x = g.add(x, d);
        }
        let on = g.params.find("output_norm.weight").unwrap();
        let h = g.rmsnorm(x, on, c.eps);
        self.lin(g, h, "output.weight")
    }

    /// Greedy generation without a KV cache (for small SDK models and
    /// evaluation): appends tokens until `stop` or `max_new`.
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut m = aido::sdk::models::LmModel::new_vocab(16, 32, 1, 2, 64, 32, 1);
    /// let out = m.generate_greedy(&pool, &[1, 2], 3, 15);
    /// assert!(out.len() <= 3);
    /// ```
    pub fn generate_greedy(&mut self, pool: &ThreadPool, prompt: &[u32], max_new: usize, stop: u32) -> Vec<u32> {
        let mut seq = prompt.to_vec();
        let mut out = Vec::new();
        let mut params = std::mem::take(&mut self.params);
        for _ in 0..max_new {
            let start = seq.len().saturating_sub(self.cfg.ctx);
            let ctx = &seq[start..];
            let next = {
                let mut g = Graph::new(&mut params, pool);
                let lg = self.forward(&mut g, ctx, ctx.len());
                let v = g.value(lg);
                let row = &v[(ctx.len() - 1) * self.cfg.vocab..ctx.len() * self.cfg.vocab];
                crate::llm::sampler::argmax(row)
            };
            if next == stop {
                break;
            }
            out.push(next);
            seq.push(next);
        }
        self.params = params;
        out
    }

    /// One training step on a batch of `nseq` windows of `seq + 1` tokens:
    /// next-token cross-entropy. Returns the loss (gradients accumulate in
    /// `self.params`).
    ///
    /// # Example
    /// ```
    /// use aido::sdk::models::LmModel;
    /// let pool = aido::runtime::pool::ThreadPool::new(2, None);
    /// let mut m = LmModel::new_bytes(32, 1, 2, 64, 8, 1);
    /// let loss = m.train_step(&pool, &[vec![1, 2, 3, 4, 5]]);
    /// assert!(loss > 0.0);
    /// ```
    pub fn train_step(&mut self, pool: &ThreadPool, windows: &[Vec<u32>]) -> f32 {
        let seq = windows[0].len() - 1;
        let mut ids = Vec::with_capacity(windows.len() * seq);
        let mut tg = Vec::with_capacity(windows.len() * seq);
        for w in windows {
            ids.extend_from_slice(&w[..seq]);
            tg.extend_from_slice(&w[1..]);
        }
        let mut params = std::mem::take(&mut self.params);
        let loss = {
            let mut g = Graph::new(&mut params, pool);
            let logits = self.forward(&mut g, &ids, seq);
            let l = g.cross_entropy(logits, &tg);
            g.backward(l)
        };
        self.params = params;
        loss
    }

    /// Writes an engine-runnable `.aido` (Llama tensor layout + metadata).
    /// LoRA adapters are merged into the base weights. `quant` selects the
    /// storage type of the matrices (`F32`, `F16`, `Q8_0`, …).
    ///
    /// # Example
    /// ```no_run
    /// let m = aido::sdk::models::LmModel::new_bytes(64, 2, 2, 128, 64, 1);
    /// m.export("tiny.aido", aido::tensor::quant::GgmlType::F32).unwrap();
    /// ```
    pub fn export(&self, path: &str, quant: GgmlType) -> Result<u64> {
        let c = &self.cfg;
        let p = &self.params;
        let mut w = AidoWriter::new();
        if let Some(m) = &self.source_meta {
            for (k, v) in m {
                w.set_meta(k, crate::format::value_to_json(v));
            }
        } else {
            // Byte-level tokenizer in GPT-2 byte-to-unicode form.
            let enc = byte_unicode();
            let mut toks: Vec<String> = (0..256).map(|b| enc[b].to_string()).collect();
            toks.push("<|endoftext|>".into());
            let mut types = vec![6u32; 256];
            types.push(3);
            w.set_meta("tokenizer.ggml.model", json!("gpt2"));
            w.set_meta("tokenizer.ggml.pre", json!("aido-bytes"));
            w.set_meta("tokenizer.ggml.tokens", json!(toks));
            w.set_meta("tokenizer.ggml.token_type", json!(types));
            w.set_meta("tokenizer.ggml.merges", json!(Vec::<String>::new()));
            w.set_meta("tokenizer.ggml.eos_token_id", json!(256));
            w.set_meta("tokenizer.ggml.bos_token_id", json!(256));
            w.set_meta("tokenizer.ggml.add_bos_token", json!(false));
        }
        let a = &c.arch;
        w.set_meta("general.architecture", json!(a));
        w.set_meta("general.name", json!(format!("aido-sdk-{a}")));
        w.set_meta(&format!("{a}.embedding_length"), json!(c.dim));
        w.set_meta(&format!("{a}.block_count"), json!(c.layers));
        w.set_meta(&format!("{a}.attention.head_count"), json!(c.heads));
        w.set_meta(&format!("{a}.attention.head_count_kv"), json!(c.kv_heads));
        w.set_meta(&format!("{a}.feed_forward_length"), json!(c.ff));
        w.set_meta(&format!("{a}.context_length"), json!(c.ctx));
        w.set_meta(&format!("{a}.rope.freq_base"), json!(c.rope_base));
        w.set_meta(&format!("{a}.attention.layer_norm_rms_epsilon"), json!(c.eps));
        w.log(json!({"op": "sdk_export", "lora_merged": c.lora > 0, "quant": format!("{quant:?}")}));
        for (i, name) in p.names.iter().enumerate() {
            if name.contains(".lora_") || name.starts_with("__") {
                continue;
            }
            let (rows, cols) = p.shapes[i];
            let mut data = p.data[i].clone();
            if let (Some(ai), Some(bi)) = (p.find(&format!("{name}.lora_a")), p.find(&format!("{name}.lora_b"))) {
                let r = p.shapes[ai].0;
                for o in 0..rows {
                    for j in 0..cols {
                        let mut acc = 0f32;
                        for k in 0..r {
                            acc += p.data[bi][o * r + k] * p.data[ai][k * cols + j];
                        }
                        data[o * cols + j] += 2.0 * acc;
                    }
                }
            }
            let is_matrix = rows > 1 && name != "token_embd.weight";
            let ty = if is_matrix && cols % quant.block_elems() == 0 { quant } else { GgmlType::F32 };
            let bytes: Vec<u8> = data.chunks(cols).flat_map(|r| quantize_row(ty, r)).collect();
            let dims = if rows > 1 || name.contains("weight") && !name.contains("norm") { vec![cols, rows] } else { vec![cols] };
            w.add(name, crate::optimize::ty_id(ty), dims, bytes);
        }
        w.write(path)
    }
}

/// GPT-2 byte → unicode table (same as the tokenizer's).
fn byte_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

/// MLP classifier: `in → hidden… → classes` with GELU.
pub struct Classifier {
    /// Layer sizes.
    pub sizes: Vec<usize>,
    /// Parameters.
    pub params: Params,
}

impl Classifier {
    /// Creates a classifier.
    ///
    /// # Example
    /// ```
    /// let c = aido::sdk::models::Classifier::new(&[4, 8, 3], 1);
    /// assert_eq!(c.params.names.len(), 4);
    /// ```
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        let mut r = Rng(seed | 1);
        let mut p = Params::default();
        for i in 0..sizes.len() - 1 {
            let (a, b) = (sizes[i], sizes[i + 1]);
            p.add(&format!("l{i}.weight"), b, a, r.normals(a * b, (2.0 / a as f32).sqrt()), true);
            p.add(&format!("l{i}.bias"), 1, b, vec![0.0; b], true);
        }
        Classifier { sizes: sizes.to_vec(), params: p }
    }

    fn logits(&self, g: &mut Graph, x: &[f32], n: usize) -> Id {
        let mut h = g.input(n, self.sizes[0], x.to_vec());
        let nl = self.sizes.len() - 1;
        for i in 0..nl {
            let w = g.params.find(&format!("l{i}.weight")).unwrap();
            let b = g.params.find(&format!("l{i}.bias")).unwrap();
            h = g.linear(h, w, Some(b), None);
            if i + 1 < nl {
                h = g.gelu(h);
            }
        }
        h
    }

    /// Training step on a batch; returns the loss.
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut c = aido::sdk::models::Classifier::new(&[2, 4, 2], 1);
    /// let l = c.train_step(&pool, &[1.0, 0.0], &[0]);
    /// assert!(l > 0.0);
    /// ```
    pub fn train_step(&mut self, pool: &ThreadPool, x: &[f32], y: &[u32]) -> f32 {
        let mut params = std::mem::take(&mut self.params);
        let l = {
            let mut g = Graph::new(&mut params, pool);
            let lg = self.logits(&mut g, x, y.len());
            let l = g.cross_entropy(lg, y);
            g.backward(l)
        };
        self.params = params;
        l
    }

    /// Class probabilities for `n` inputs.
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut c = aido::sdk::models::Classifier::new(&[2, 4, 3], 1);
    /// let p = c.predict(&pool, &[1.0, 0.0], 1);
    /// assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    /// ```
    pub fn predict(&mut self, pool: &ThreadPool, x: &[f32], n: usize) -> Vec<f32> {
        let mut params = std::mem::take(&mut self.params);
        let mut out = {
            let mut g = Graph::new(&mut params, pool);
            let lg = self.logits(&mut g, x, n);
            g.value(lg).to_vec()
        };
        self.params = params;
        let k = *self.sizes.last().unwrap();
        for row in out.chunks_exact_mut(k) {
            crate::tensor::ops::softmax_inplace(row);
        }
        out
    }
}

/// Tiny DDPM: an MLP predicts the noise ε of `x_t` given a sinusoidal
/// embedding of `t`; images are flattened grey-scale squares.
pub struct Ddpm {
    /// Pixels per image (`side²`).
    pub dim: usize,
    /// Diffusion steps.
    pub steps: usize,
    /// Hidden width.
    pub hidden: usize,
    /// Parameters.
    pub params: Params,
    betas: Vec<f32>,
    abar: Vec<f32>,
}

impl Ddpm {
    /// Creates the model with a linear β schedule.
    ///
    /// # Example
    /// ```
    /// let d = aido::sdk::models::Ddpm::new(64, 100, 128, 1);
    /// assert_eq!(d.dim, 64);
    /// ```
    pub fn new(dim: usize, steps: usize, hidden: usize, seed: u64) -> Self {
        let mut r = Rng(seed | 1);
        let mut p = Params::default();
        let te = 32;
        let sizes = [dim + te, hidden, hidden, hidden, dim];
        for i in 0..4 {
            let (a, b) = (sizes[i], sizes[i + 1]);
            let std = if i == 3 { 0.0 } else { (2.0 / a as f32).sqrt() };
            p.add(&format!("l{i}.weight"), b, a, r.normals(a * b, std), true);
            p.add(&format!("l{i}.bias"), 1, b, vec![0.0; b], true);
        }
        let betas: Vec<f32> = (0..steps).map(|i| 1e-4 + (0.02 - 1e-4) * i as f32 / (steps - 1) as f32).collect();
        let mut abar = Vec::with_capacity(steps);
        let mut a = 1.0;
        for b in &betas {
            a *= 1.0 - b;
            abar.push(a);
        }
        Ddpm { dim, steps, hidden, params: p, betas, abar }
    }

    fn temb(&self, t: usize) -> Vec<f32> {
        (0..32).map(|i| {
            let f = (t as f32) * (-(i / 2) as f32 * (1000f32).ln() / 16.0).exp();
            if i % 2 == 0 { f.sin() } else { f.cos() }
        }).collect()
    }

    fn eps(&self, g: &mut Graph, x: &[f32], ts: &[usize]) -> Id {
        let n = ts.len();
        let te: Vec<f32> = ts.iter().flat_map(|&t| self.temb(t)).collect();
        let xi = g.input(n, self.dim, x.to_vec());
        let ti = g.input(n, 32, te);
        let mut h = g.concat_cols(xi, ti);
        for i in 0..4 {
            let w = g.params.find(&format!("l{i}.weight")).unwrap();
            let b = g.params.find(&format!("l{i}.bias")).unwrap();
            h = g.linear(h, w, Some(b), None);
            if i < 3 {
                h = g.silu(h);
            }
        }
        h
    }

    /// One training step on clean images `x0` (`n × dim`, values in [-1,1]).
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut d = aido::sdk::models::Ddpm::new(16, 50, 32, 1);
    /// let mut rng = aido::sdk::Rng(3);
    /// let l = d.train_step(&pool, &vec![0.5; 32], &mut rng);
    /// assert!(l > 0.0);
    /// ```
    pub fn train_step(&mut self, pool: &ThreadPool, x0: &[f32], rng: &mut Rng) -> f32 {
        let n = x0.len() / self.dim;
        let ts: Vec<usize> = (0..n).map(|_| (rng.uniform() * self.steps as f32) as usize % self.steps).collect();
        let noise = rng.normals(x0.len(), 1.0);
        let mut xt = vec![0f32; x0.len()];
        for i in 0..n {
            let (a, b) = (self.abar[ts[i]].sqrt(), (1.0 - self.abar[ts[i]]).sqrt());
            for j in 0..self.dim {
                xt[i * self.dim + j] = a * x0[i * self.dim + j] + b * noise[i * self.dim + j];
            }
        }
        let mut params = std::mem::take(&mut self.params);
        let l = {
            let mut g = Graph::new(&mut params, pool);
            let e = self.eps(&mut g, &xt, &ts);
            let l = g.mse(e, &noise);
            g.backward(l)
        };
        self.params = params;
        l
    }

    /// Ancestral sampling of `n` images (values in [-1, 1]).
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut d = aido::sdk::models::Ddpm::new(16, 10, 32, 1);
    /// let imgs = d.sample(&pool, 2, 5);
    /// assert_eq!(imgs.len(), 32);
    /// ```
    pub fn sample(&mut self, pool: &ThreadPool, n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed | 1);
        let mut x = rng.normals(n * self.dim, 1.0);
        let mut params = std::mem::take(&mut self.params);
        for t in (0..self.steps).rev() {
            let e = {
                let mut g = Graph::new(&mut params, pool);
                let id = self.eps(&mut g, &x, &vec![t; n]);
                g.value(id).to_vec()
            };
            // Posterior q(x_{t-1} | x_t, x̂0) with x̂0 clipped to the data range
            // and the lower-variance β̃ (Ho et al. 2020, eq. 7).
            let (b, ab) = (self.betas[t], self.abar[t]);
            let ab_prev = if t > 0 { self.abar[t - 1] } else { 1.0 };
            let c0 = ab_prev.sqrt() * b / (1.0 - ab);
            let ct = (1.0 - b).sqrt() * (1.0 - ab_prev) / (1.0 - ab);
            let var = b * (1.0 - ab_prev) / (1.0 - ab);
            let z = if t > 0 { rng.normals(x.len(), 1.0) } else { vec![0.0; x.len()] };
            for i in 0..x.len() {
                let x0 = ((x[i] - (1.0 - ab).sqrt() * e[i]) / ab.sqrt()).clamp(-1.0, 1.0);
                x[i] = c0 * x0 + ct * x[i] + var.sqrt() * z[i];
            }
        }
        self.params = params;
        x.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        x
    }

    /// Restores a model from checkpoint parameters.
    ///
    /// # Example
    /// ```
    /// let d = aido::sdk::models::Ddpm::new(16, 10, 32, 1);
    /// let p = d.params.clone();
    /// let d2 = aido::sdk::models::Ddpm::from_params(16, 10, 32, p);
    /// ```
    pub fn from_params(dim: usize, steps: usize, hidden: usize, params: Params) -> Self {
        let mut d = Ddpm::new(dim, steps, hidden, 1);
        d.params = params;
        d
    }
}
