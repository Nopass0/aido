//! **SDK** for building, training, fine-tuning and compiling your own
//! neural networks with the engine's kernels.
//!
//! * [`autograd`] — define-by-run reverse-mode differentiation.
//! * [`AdamW`] — optimiser with decoupled weight decay, gradient clipping and
//!   warm-up + cosine learning-rate schedule.
//! * [`save_checkpoint`] / [`load_checkpoint`] — resumable training state
//!   (weights + optimiser moments + step) stored in the `.aido` container.
//! * [`data`] — the `.aidata` dataset container (token streams or
//!   feature/label tensors, memory-mapped).
//! * [`models`] — ready architectures: a Llama-style language model
//!   (trainable from scratch, fine-tunable with LoRA, exportable to an
//!   engine-runnable `.aido`), an MLP classifier and a tiny DDPM image
//!   generator.

pub mod autograd;
pub mod data;
pub mod io;
pub mod models;
pub mod snake;
pub mod bpe;
pub mod pixelart;
pub mod textdet;

use anyhow::{Context, Result};
use serde_json::json;

use crate::format::{AidoFile, AidoWriter, TensorStore};
use autograd::Params;

/// Optimiser state: AdamW, or the **Muon + AdamW hybrid** when
/// [`AdamW::muon`] is enabled.
///
/// ## Muon (MomentUm Orthogonalized by Newton-Schulz)
/// For every hidden weight matrix `W (r×c)` (not embeddings, not the output
/// head, not vectors) the update is
///
/// ```text
/// M ← μ·M + G                         (momentum, μ = 0.95)
/// U = G + μ·M                         (Nesterov)
/// O = NewtonSchulz5(U / ‖U‖_F)        (≈ U·(UᵀU)^(-1/2): the nearest semi-orthogonal matrix)
/// W ← W − lr · (0.2·√max(r,c) · O + wd · W)
/// ```
///
/// Orthogonalisation equalises the update across all singular directions,
/// so rare-but-important directions learn as fast as dominant ones. The
/// `0.2·√max(r,c)` factor matches the update RMS of AdamW (Moonlight,
/// "Muon is Scalable for LLM Training", 2025), so the same learning rate
/// works for both parts. Reported ≈2× compute efficiency over AdamW in
/// LLM scaling-law runs and used in production models (Kimi K2's MuonClip);
/// it also keeps **one** state buffer per matrix instead of two. Remaining
/// parameters (embeddings, head, norms, biases) use AdamW as usual.
pub struct AdamW {
    /// Peak learning rate.
    pub lr: f32,
    /// β₁.
    pub b1: f32,
    /// β₂.
    pub b2: f32,
    /// ε.
    pub eps: f32,
    /// Decoupled weight decay.
    pub wd: f32,
    /// Global-norm gradient clip (0 = off).
    pub clip: f32,
    /// Warm-up steps.
    pub warmup: usize,
    /// Total steps (for the cosine decay).
    pub total: usize,
    /// Steps taken.
    pub step: usize,
    /// First moments.
    pub m: Vec<Vec<f32>>,
    /// Second moments.
    pub v: Vec<Vec<f32>>,
    /// Use Muon for hidden weight matrices (see the type docs).
    pub muon: bool,
    /// Muon momentum μ.
    pub muon_momentum: f32,
}

/// Is parameter `i` a hidden matrix that Muon should handle?
fn muon_param(p: &Params, i: usize) -> bool {
    let (r, c) = p.shapes[i];
    let n = &p.names[i];
    r > 1 && c > 1 && !n.contains("token_embd") && !n.contains("embed") && n != "output.weight" && !n.ends_with("head.weight")
}

/// Newton–Schulz quintic iteration (Jordan et al. coefficients
/// `a, b, c = 3.4445, −4.7750, 2.0315`, 5 steps): maps `g (r×c)` to an
/// approximately semi-orthogonal matrix with the same singular vectors.
///
/// # Example
/// ```
/// // A diagonal matrix with very different scales becomes ≈ orthonormal (singular values ≈ 1).
/// let o = aido::sdk::newton_schulz(&[10.0, 0.0, 0.0, 0.1], 2, 2);
/// assert!((o[0] - o[3]).abs() < 0.35 && o[0] > 0.5);
/// ```
pub fn newton_schulz(g: &[f32], r: usize, c: usize) -> Vec<f32> {
    let (a_, b_, c_) = (3.4445f32, -4.7750f32, 2.0315f32);
    // Work on the wide orientation (rows ≤ cols): Gram matrix is rows×rows.
    let tr = r > c;
    let (m, n) = if tr { (c, r) } else { (r, c) };
    let mut x = vec![0f32; m * n];
    for i in 0..r {
        for j in 0..c {
            if tr {
                x[j * n + i] = g[i * c + j];
            } else {
                x[i * n + j] = g[i * c + j];
            }
        }
    }
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-7;
    x.iter_mut().for_each(|v| *v /= norm);
    let mut a = vec![0f32; m * m];
    let mut b = vec![0f32; m * m];
    let mut nx = vec![0f32; m * n];
    for _ in 0..5 {
        // A = X·Xᵀ
        for i in 0..m {
            for j in i..m {
                let s: f32 = x[i * n..(i + 1) * n].iter().zip(&x[j * n..(j + 1) * n]).map(|(p, q)| p * q).sum();
                a[i * m + j] = s;
                a[j * m + i] = s;
            }
        }
        // B = b·A + c·A·A
        for i in 0..m {
            for j in 0..m {
                let s: f32 = (0..m).map(|k| a[i * m + k] * a[k * m + j]).sum();
                b[i * m + j] = b_ * a[i * m + j] + c_ * s;
            }
        }
        // X = a·X + B·X
        for i in 0..m {
            let row = &mut nx[i * n..(i + 1) * n];
            for (o, xv) in row.iter_mut().zip(&x[i * n..(i + 1) * n]) {
                *o = a_ * xv;
            }
            for k in 0..m {
                let w = b[i * m + k];
                if w != 0.0 {
                    for (o, xv) in row.iter_mut().zip(&x[k * n..(k + 1) * n]) {
                        *o += w * xv;
                    }
                }
            }
        }
        std::mem::swap(&mut x, &mut nx);
    }
    if !tr {
        return x;
    }
    let mut out = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[i * c + j] = x[j * n + i];
        }
    }
    out
}

impl AdamW {
    /// Creates an optimiser for `params`.
    ///
    /// # Example
    /// ```
    /// let mut p = aido::sdk::autograd::Params::default();
    /// p.add("w", 1, 1, vec![0.0], true);
    /// let opt = aido::sdk::AdamW::new(&p, 1e-3, 100);
    /// assert_eq!(opt.step, 0);
    /// ```
    pub fn new(params: &Params, lr: f32, total: usize) -> Self {
        let z = |p: &Params| p.data.iter().zip(&p.trainable).map(|(d, t)| if *t { vec![0.0; d.len()] } else { Vec::new() }).collect();
        AdamW { lr, b1: 0.9, b2: 0.95, eps: 1e-8, wd: 0.01, clip: 1.0, warmup: (total / 20).max(1), total: total.max(1), step: 0, m: z(params), v: z(params), muon: false, muon_momentum: 0.95 }
    }

    /// Muon + AdamW hybrid (Muon for hidden matrices, AdamW for the rest);
    /// same learning-rate scale as [`AdamW::new`].
    ///
    /// # Example
    /// ```
    /// let mut p = aido::sdk::autograd::Params::default();
    /// p.add("blk.0.w", 4, 4, vec![0.1; 16], true);
    /// let o = aido::sdk::AdamW::muon(&p, 1e-3, 100);
    /// assert!(o.muon);
    /// ```
    pub fn muon(params: &Params, lr: f32, total: usize) -> Self {
        let mut o = Self::new(params, lr, total);
        o.muon = true;
        // Muon matrices keep a single momentum buffer: free their second moments.
        for i in 0..params.data.len() {
            if muon_param(params, i) {
                o.v[i] = Vec::new();
            }
        }
        o
    }

    /// Creates the optimiser named `name` (`adamw` | `muon`).
    ///
    /// # Example
    /// ```
    /// let p = aido::sdk::autograd::Params::default();
    /// assert!(aido::sdk::AdamW::by_name("muon", &p, 1e-3, 10).unwrap().muon);
    /// ```
    pub fn by_name(name: &str, params: &Params, lr: f32, total: usize) -> anyhow::Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "adamw" | "adam" => Ok(Self::new(params, lr, total)),
            "muon" => Ok(Self::muon(params, lr, total)),
            o => anyhow::bail!("unknown optimizer '{o}' (adamw | muon)"),
        }
    }

    /// Learning rate at the current step (linear warm-up, cosine decay to 10 %).
    ///
    /// # Example
    /// ```
    /// let p = aido::sdk::autograd::Params::default();
    /// let o = aido::sdk::AdamW::new(&p, 1.0, 100);
    /// assert!(o.current_lr() <= 1.0);
    /// ```
    pub fn current_lr(&self) -> f32 {
        let s = self.step as f32;
        if self.step < self.warmup {
            return self.lr * (s + 1.0) / self.warmup as f32;
        }
        let prog = ((s - self.warmup as f32) / (self.total - self.warmup).max(1) as f32).min(1.0);
        self.lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos()))
    }

    /// Applies one update using the gradients in `params`, then zeroes them.
    ///
    /// * **Output** — the global gradient norm before clipping.
    ///
    /// # Example
    /// ```
    /// let mut p = aido::sdk::autograd::Params::default();
    /// p.add("w", 1, 1, vec![1.0], true);
    /// p.grads[0][0] = 1.0;
    /// let mut o = aido::sdk::AdamW::new(&p, 0.1, 10);
    /// o.update(&mut p);
    /// assert!(p.data[0][0] < 1.0);
    /// ```
    pub fn update(&mut self, params: &mut Params) -> f32 {
        let norm: f32 = params.grads.iter().flat_map(|g| g.iter()).map(|v| v * v).sum::<f32>().sqrt();
        let cs = if self.clip > 0.0 && norm > self.clip { self.clip / norm } else { 1.0 };
        let lr = self.current_lr();
        self.step += 1;
        let t = self.step as i32;
        let (bc1, bc2) = (1.0 - self.b1.powi(t), 1.0 - self.b2.powi(t));
        for i in 0..params.data.len() {
            if !params.trainable[i] {
                continue;
            }
            let decay = if params.shapes[i].0 > 1 { self.wd } else { 0.0 };
            if self.muon && muon_param(params, i) {
                let (r, c) = params.shapes[i];
                let mu = self.muon_momentum;
                let (d, g, m) = (&mut params.data[i], &params.grads[i], &mut self.m[i]);
                let mut u = vec![0f32; d.len()];
                for j in 0..d.len() {
                    let gj = g[j] * cs;
                    m[j] = mu * m[j] + gj;
                    u[j] = gj + mu * m[j];
                }
                let o = newton_schulz(&u, r, c);
                let k = 0.2 * (r.max(c) as f32).sqrt();
                for j in 0..d.len() {
                    d[j] -= lr * (k * o[j] + decay * d[j]);
                }
                continue;
            }
            let (d, g, m, v) = (&mut params.data[i], &params.grads[i], &mut self.m[i], &mut self.v[i]);
            for j in 0..d.len() {
                let gj = g[j] * cs;
                m[j] = self.b1 * m[j] + (1.0 - self.b1) * gj;
                v[j] = self.b2 * v[j] + (1.0 - self.b2) * gj * gj;
                let upd = (m[j] / bc1) / ((v[j] / bc2).sqrt() + self.eps);
                d[j] -= lr * (upd + decay * d[j]);
            }
        }
        params.zero_grad();
        norm
    }
}

/// The optimiser selected for training: `AIDO_OPTIMIZER` = `muon`
/// (default: Muon + AdamW hybrid) or `adamw`. Used by all built-in
/// training commands (`aido --optimizer adamw train-…`).
///
/// # Example
/// ```
/// let p = aido::sdk::autograd::Params::default();
/// let o = aido::sdk::optimizer(&p, 1e-3, 100);
/// ```
pub fn optimizer(params: &Params, lr: f32, total: usize) -> AdamW {
    let name = std::env::var("AIDO_OPTIMIZER").unwrap_or_else(|_| "muon".into());
    AdamW::by_name(&name, params, lr, total).unwrap_or_else(|_| AdamW::muon(params, lr, total))
}

/// Saves weights, optimiser moments, step and a JSON config to `.aido`.
///
/// # Example
/// ```
/// let mut p = aido::sdk::autograd::Params::default();
/// p.add("w", 1, 2, vec![1.0, 2.0], true);
/// let o = aido::sdk::AdamW::new(&p, 1e-3, 10);
/// let path = std::env::temp_dir().join("aido_ckpt_doc.aido");
/// aido::sdk::save_checkpoint(&path, &p, Some(&o), &serde_json::json!({"kind": "demo"})).unwrap();
/// let (q, o2, cfg) = aido::sdk::load_checkpoint(&path).unwrap();
/// assert_eq!(q.data[0], vec![1.0, 2.0]);
/// assert_eq!(cfg["kind"], "demo");
/// ```
pub fn save_checkpoint(path: impl AsRef<std::path::Path>, p: &Params, opt: Option<&AdamW>, config: &serde_json::Value) -> Result<()> {
    let mut w = AidoWriter::new();
    w.set_meta("general.architecture", json!("aido-sdk"));
    w.set_meta("sdk.config", json!(config.to_string()));
    w.set_meta("sdk.trainable", json!(p.trainable));
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    for i in 0..p.names.len() {
        let (r, c) = p.shapes[i];
        w.add(&format!("param/{}", p.names[i]), 0, vec![c, r], f32b(&p.data[i]));
    }
    if let Some(o) = opt {
        w.set_meta("sdk.step", json!(o.step));
        w.set_meta("sdk.opt", json!({"lr": o.lr, "total": o.total, "warmup": o.warmup, "wd": o.wd, "muon": o.muon}));
        for i in 0..p.names.len() {
            if p.trainable[i] {
                let (r, c) = p.shapes[i];
                w.add(&format!("adam.m/{}", p.names[i]), 0, vec![c, r], f32b(&o.m[i]));
                if !o.v[i].is_empty() {
                    w.add(&format!("adam.v/{}", p.names[i]), 0, vec![c, r], f32b(&o.v[i]));
                }
            }
        }
    }
    w.log(json!({"op": "checkpoint", "params": p.names.len()}));
    w.write(path)?;
    Ok(())
}

/// Loads a checkpoint written by [`save_checkpoint`].
///
/// * **Output** — `(params, optimiser if saved, config JSON)`.
///
/// # Example
/// ```no_run
/// let (p, opt, cfg) = aido::sdk::load_checkpoint("tiny.ckpt.aido").unwrap();
/// ```
pub fn load_checkpoint(path: impl AsRef<std::path::Path>) -> Result<(Params, Option<AdamW>, serde_json::Value)> {
    let f = AidoFile::open(path)?;
    let cfg: serde_json::Value = serde_json::from_str(f.get_str("sdk.config").unwrap_or("{}"))?;
    let trainable: Vec<bool> = f.meta().get("sdk.trainable").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_u64() == Some(1)).collect()).unwrap_or_default();
    let mut p = Params::default();
    for name in f.names() {
        if let Some(n) = name.strip_prefix("param/") {
            let (_, dims) = f.info(&name).context("info")?;
            let tr = trainable.get(p.names.len()).copied().unwrap_or(true);
            p.add(n, dims[1], dims[0], f.vector(&name)?, tr);
        }
    }
    let opt = match f.get_u64("sdk.step") {
        Some(step) => {
            let o: serde_json::Value = f.meta().get("sdk.opt").map(crate::format::value_to_json).unwrap_or(json!({}));
            let lr = o.get("lr").and_then(|v| v.as_f64()).unwrap_or(1e-3) as f32;
            let total = o.get("total").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
            let mut a = if o.get("muon").and_then(|v| v.as_bool()).unwrap_or(false) { AdamW::muon(&p, lr, total) } else { AdamW::new(&p, lr, total) };
            a.step = step as usize;
            for i in 0..p.names.len() {
                if p.trainable[i] {
                    a.m[i] = f.vector(&format!("adam.m/{}", p.names[i]))?;
                    if !a.v[i].is_empty() {
                        a.v[i] = f.vector(&format!("adam.v/{}", p.names[i]))?;
                    }
                }
            }
            Some(a)
        }
        None => None,
    };
    Ok((p, opt, cfg))
}

/// Deterministic pseudo-random generator for initialisation and sampling.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    /// Uniform in `[0, 1)`.
    ///
    /// # Example
    /// ```
    /// let mut r = aido::sdk::Rng(7);
    /// assert!((0.0..1.0).contains(&r.uniform()));
    /// ```
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// Standard normal sample.
    ///
    /// # Example
    /// ```
    /// let mut r = aido::sdk::Rng(7);
    /// let _x = r.normal();
    /// ```
    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }

    /// Vector of `n` normals scaled by `std`.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::sdk::Rng(1).normals(3, 0.1).len(), 3);
    /// ```
    pub fn normals(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * std).collect()
    }
}
