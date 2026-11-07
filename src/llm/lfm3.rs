//! LFM3 hybrid backbone (Liquid AI): gated short-convolution blocks,
//! sliding-window and full attention with QK-norm and a per-head sigmoid
//! output gate, SwiGLU FFN, optional "sandwich" norms.
//!
//! The same block types serve the text model of d1 / LFM3 checkpoints and
//! (with different sizes) their audio encoder. All matrices are re-quantised
//! from BF16 to Q8_0 at load and run on the integer CPU kernels.
//!
//! Attention is computed for a whole prompt at once with an arbitrary
//! visibility rule ([`Visibility`]) — needed by d1, whose full-attention
//! layers read the state and question bidirectionally.

use anyhow::{Context, Result};

use crate::format::safetensors::SafetensorsDir;
use crate::format::TensorStore;
use crate::runtime::pool::ThreadPool;
use crate::tensor::ops::{self, Rope};
use crate::tensor::quant::GgmlType;
use crate::tensor::{QWeight, SyncPtr};

/// Block type of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Gated short convolution.
    Conv,
    /// Causal attention within a window.
    Sliding,
    /// Attention under the visibility rule.
    Full,
}

/// Which keys a query may attend on full-attention layers.
#[derive(Debug, Clone)]
pub enum Visibility {
    /// Plain causal.
    Causal,
    /// Everything sees everything.
    All,
    /// d1 `set`: token `i` with question id `g[i]` (−1 on the state, which
    /// has length `state`) sees `j` iff `i < state || j < state || g[i] == g[j]`.
    Set {
        /// State length.
        state: usize,
        /// Question id per token.
        groups: Vec<i32>,
    },
}

impl Visibility {
    /// Whether query `i` sees key `j`.
    ///
    /// # Example
    /// ```
    /// use aido::llm::lfm3::Visibility;
    /// assert!(Visibility::All.sees(0, 5));
    /// assert!(!Visibility::Causal.sees(0, 5));
    /// let v = Visibility::Set { state: 2, groups: vec![-1, -1, 1, 2] };
    /// assert!(v.sees(3, 0) && !v.sees(3, 2) && v.sees(0, 3));
    /// ```
    #[inline]
    pub fn sees(&self, i: usize, j: usize) -> bool {
        match self {
            Visibility::Causal => j <= i,
            Visibility::All => true,
            Visibility::Set { state, groups } => i < *state || j < *state || groups[i] == groups[j],
        }
    }
}

/// A linear layer: CPU weight plus an optional VRAM copy.
pub struct Lin {
    /// CPU weight.
    pub w: QWeight,
    /// GPU copy (used for batches of at least [`GPU_MIN_T`] rows).
    pub g: Option<crate::gpu::GpuMatrix>,
}

/// Smallest batch for which the GPU GEMM beats the CPU integer path.
pub const GPU_MIN_T: usize = 24;

/// Optional GPU + VRAM budget used while loading.
#[derive(Clone, Default)]
pub struct Accel {
    /// The device.
    pub gpu: Option<std::sync::Arc<crate::gpu::Gpu>>,
    /// Remaining budget in bytes (shared counter).
    pub budget: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Accel {
    /// Creates an accelerator handle with a VRAM budget.
    ///
    /// # Example
    /// ```
    /// let a = aido::llm::lfm3::Accel::new(None, 0);
    /// assert!(a.gpu.is_none());
    /// ```
    pub fn new(gpu: Option<std::sync::Arc<crate::gpu::Gpu>>, budget: usize) -> Self {
        Accel { gpu, budget: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(budget)) }
    }

    /// Wraps a CPU weight, uploading it while the budget lasts.
    ///
    /// # Example
    /// ```
    /// let a = aido::llm::lfm3::Accel::new(None, 0);
    /// let l = a.lin(aido::tensor::QWeight::from_f32(1, 32, &[0.0; 32])).unwrap();
    /// assert!(l.g.is_none());
    /// ```
    pub fn lin(&self, w: QWeight) -> Result<Lin> {
        use std::sync::atomic::Ordering;
        let g = match &self.gpu {
            Some(gp) if w.cols % 32 == 0 => {
                let need = w.nbytes() * 9 / 8;
                let left = self.budget.load(Ordering::Relaxed);
                if left >= need {
                    self.budget.store(left - need, Ordering::Relaxed);
                    Some(gp.upload(&w)?)
                } else {
                    None
                }
            }
            _ => None,
        };
        Ok(Lin { w, g })
    }
}

/// Computes several linear layers sharing the input: one GPU submission
/// when all are resident and the batch is large, else the CPU path.
///
/// # Example
/// ```
/// use aido::llm::lfm3::{lin_many, Accel};
/// let pool = aido::runtime::pool::ThreadPool::new(1, None);
/// let a = Accel::new(None, 0);
/// let l = a.lin(aido::tensor::QWeight::from_f32(1, 32, &[1.0; 32])).unwrap();
/// let mut out = vec![0f32; 1];
/// lin_many(&pool, &a, &mut [(&l, &mut out[..])], &[1.0; 32], 1);
/// assert!((out[0] - 32.0).abs() < 0.5);
/// ```
pub fn lin_many(pool: &ThreadPool, acc: &Accel, ops: &mut [(&Lin, &mut [f32])], x: &[f32], t: usize) {
    if let Some(gp) = &acc.gpu {
        if t >= GPU_MIN_T && ops.iter().all(|o| o.0.g.is_some()) {
            let mats: Vec<&crate::gpu::GpuMatrix> = ops.iter().map(|o| o.0.g.as_ref().unwrap()).collect();
            let mut outs: Vec<&mut [f32]> = ops.iter_mut().map(|o| &mut *o.1).collect();
            if gp.gemm_many(x, t, &mats, &mut outs).is_ok() {
                return;
            }
        }
    }
    for (l, o) in ops.iter_mut() {
        super::model::linear_rows(pool, &l.w, x, t, o);
    }
}

/// Dimensions of a backbone.
#[derive(Debug, Clone)]
pub struct Dims {
    /// Hidden size.
    pub d: usize,
    /// Query heads.
    pub heads: usize,
    /// KV heads.
    pub kv_heads: usize,
    /// Head dim.
    pub hd: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Sliding window.
    pub window: usize,
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    gate: Option<Lin>,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

struct ConvBlock {
    in_proj: Lin,
    out_proj: Lin,
    w: Vec<f32>,
    k: usize,
}

struct Layer {
    kind: Kind,
    op_norm: Vec<f32>,
    post_op: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    post_ffn: Option<Vec<f32>>,
    attn: Option<Attn>,
    conv: Option<ConvBlock>,
    w1: Lin,
    w2: Lin,
    w3: Lin,
}

/// A loaded LFM backbone.
pub struct Backbone {
    /// Dimensions.
    pub dims: Dims,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    rope_full: Rope,
    rope_sliding: Rope,
    acc: Accel,
}

/// Loads a BF16/F16 matrix and re-quantises it to Q8_0 (rows in parallel).
///
/// # Example
/// ```no_run
/// # let st = aido::format::safetensors::SafetensorsDir::open("models/d1-omni").unwrap();
/// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
/// let w = aido::llm::lfm3::load_q8(&st, &pool, "model.language_model.layers.0.feed_forward.w1.weight").unwrap();
/// ```
pub fn load_q8(st: &SafetensorsDir, pool: &ThreadPool, name: &str) -> Result<QWeight> {
    let w = st.weight(name)?;
    if w.cols % 32 != 0 {
        return Ok(w);
    }
    let rb = GgmlType::Q8_0.row_bytes(w.cols);
    let mut out = vec![0u8; rb * w.rows];
    let op = SyncPtr::new(&mut out);
    let src = &w;
    pool.parallel_chunks(w.rows, 64, |a, b, _| {
        let mut tmp = vec![0f32; src.cols];
        for r in a..b {
            src.dequant_row(r, &mut tmp);
            let q = crate::tensor::quantize::quantize_row(GgmlType::Q8_0, &tmp);
            unsafe { op.slice(r * rb, rb).copy_from_slice(&q) };
        }
    });
    QWeight::new(GgmlType::Q8_0, w.rows, w.cols, crate::tensor::Bytes::from_vec(out))
}

/// Rope with only the first `angles` frequencies non-zero (LFM3 "zero
/// tail" partial rotary, NeoX halves over the whole head).
fn zero_tail_rope(hd: usize, base: f32, partial: f32) -> Rope {
    let angles = ((partial * hd as f32) as usize) / 2;
    let inv_freq = (0..hd / 2).map(|i| if i < angles { 1.0 / base.powf(2.0 * i as f32 / hd as f32) } else { 0.0 }).collect();
    Rope { inv_freq, rot_dim: hd, neox: true }
}

impl Backbone {
    /// Loads layers `prefix.layers.N.*` and `prefix.embedding_norm`.
    ///
    /// * **Input** — store, pool, tensor prefix, layer kinds, dimensions,
    ///   RoPE settings `(full θ, full partial factor, sliding θ)`, attention
    ///   tensor naming (`"q_norm"`/`"q_layernorm"`, `"o_proj"`/`"out_proj"`).
    ///
    /// # Example
    /// ```no_run
    /// // see `aido::llm::d1::D1::load`
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn load(st: &SafetensorsDir, pool: &ThreadPool, prefix: &str, kinds: &[Kind], dims: Dims, rope: (f32, f32, f32), names: (&str, &str), acc: &Accel) -> Result<Self> {
        let l8 = |n: &str| -> Result<Lin> { acc.lin(load_q8(st, pool, n)?) };
        let mut layers = Vec::new();
        for (i, &kind) in kinds.iter().enumerate() {
            let p = |s: &str| format!("{prefix}.layers.{i}.{s}");
            let has = |s: &str| st.shape(&p(s)).is_some();
            let attn = if kind != Kind::Conv {
                Some(Attn {
                    q: l8(&p("self_attn.q_proj.weight"))?,
                    k: l8(&p("self_attn.k_proj.weight"))?,
                    v: l8(&p("self_attn.v_proj.weight"))?,
                    o: l8(&p(&format!("self_attn.{}.weight", names.1)))?,
                    gate: if has("self_attn.gate_proj.weight") { Some(l8(&p("self_attn.gate_proj.weight"))?) } else { None },
                    qn: st.f32(&p(&format!("self_attn.q_{}.weight", names.0)))?,
                    kn: st.f32(&p(&format!("self_attn.k_{}.weight", names.0)))?,
                })
            } else {
                None
            };
            let conv = if kind == Kind::Conv {
                let shape = st.shape(&p("conv.conv.weight")).context("conv weight")?.to_vec();
                Some(ConvBlock { in_proj: l8(&p("conv.in_proj.weight"))?, out_proj: l8(&p("conv.out_proj.weight"))?, w: st.f32(&p("conv.conv.weight"))?, k: *shape.last().unwrap() })
            } else {
                None
            };
            layers.push(Layer {
                kind,
                op_norm: st.f32(&p("operator_norm.weight"))?,
                post_op: if has("post_operator_norm.weight") { Some(st.f32(&p("post_operator_norm.weight"))?) } else { None },
                ffn_norm: st.f32(&p("ffn_norm.weight"))?,
                post_ffn: if has("post_ffn_norm.weight") { Some(st.f32(&p("post_ffn_norm.weight"))?) } else { None },
                attn,
                conv,
                w1: l8(&p("feed_forward.w1.weight"))?,
                w2: l8(&p("feed_forward.w2.weight"))?,
                w3: l8(&p("feed_forward.w3.weight"))?,
            });
        }
        let hd = dims.hd;
        Ok(Backbone {
            final_norm: st.f32(&format!("{prefix}.embedding_norm.weight"))?,
            rope_full: zero_tail_rope(hd, rope.0, rope.1),
            rope_sliding: zero_tail_rope(hd, rope.2, 1.0),
            dims,
            layers,
            acc: acc.clone(),
        })
    }

    /// Runs all layers over `x` (`t × d`, updated in place) and applies the
    /// final norm. `positions[i]` is token `i`'s RoPE position.
    ///
    /// # Example
    /// ```no_run
    /// # let b: aido::llm::lfm3::Backbone = unimplemented!();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// let mut x = vec![0f32; 4 * b.dims.d];
    /// b.forward(&pool, &mut x, 4, &[0, 1, 2, 3], &aido::llm::lfm3::Visibility::Causal);
    /// ```
    pub fn forward(&self, pool: &ThreadPool, x: &mut [f32], t: usize, positions: &[usize], vis: &Visibility) {
        let d = self.dims.d;
        let mut h = vec![0f32; t * d];
        let mut o = vec![0f32; t * d];
        for l in &self.layers {
            for r in 0..t {
                ops::rmsnorm(&mut h[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], &l.op_norm, self.dims.eps);
            }
            match l.kind {
                Kind::Conv => self.conv(pool, l.conv.as_ref().unwrap(), &h, t, &mut o),
                k => self.attention(pool, l.attn.as_ref().unwrap(), &h, t, positions, if k == Kind::Full { Some(vis) } else { None }, &mut o),
            }
            if let Some(pn) = &l.post_op {
                for r in 0..t {
                    ops::rmsnorm_inplace(&mut o[r * d..(r + 1) * d], pn, self.dims.eps);
                }
            }
            ops::add_inplace(x, &o);
            for r in 0..t {
                ops::rmsnorm(&mut h[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], &l.ffn_norm, self.dims.eps);
            }
            let ff = l.w1.w.rows;
            let mut g = vec![0f32; t * ff];
            let mut u = vec![0f32; t * ff];
            lin_many(pool, &self.acc, &mut [(&l.w1, &mut g), (&l.w3, &mut u)], &h, t);
            ops::silu_mul(&mut g, &u);
            lin_many(pool, &self.acc, &mut [(&l.w2, &mut o)], &g, t);
            if let Some(pn) = &l.post_ffn {
                for r in 0..t {
                    ops::rmsnorm_inplace(&mut o[r * d..(r + 1) * d], pn, self.dims.eps);
                }
            }
            ops::add_inplace(x, &o);
        }
        for r in 0..t {
            ops::rmsnorm_inplace(&mut x[r * d..(r + 1) * d], &self.final_norm, self.dims.eps);
        }
    }

    /// Gated short convolution: `out_proj(C ⊙ causal_dwconv(B ⊙ x))`.
    fn conv(&self, pool: &ThreadPool, c: &ConvBlock, h: &[f32], t: usize, out: &mut [f32]) {
        let d = self.dims.d;
        let mut bcx = vec![0f32; t * 3 * d];
        lin_many(pool, &self.acc, &mut [(&c.in_proj, &mut bcx)], h, t);
        let mut bx = vec![0f32; t * d];
        for r in 0..t {
            for j in 0..d {
                bx[r * d + j] = bcx[r * 3 * d + j] * bcx[r * 3 * d + 2 * d + j];
            }
        }
        let mut y = vec![0f32; t * d];
        let yp = SyncPtr::new(&mut y);
        let k = c.k;
        pool.parallel_chunks(d, 128, |a, b, _| {
            for j in a..b {
                let w = &c.w[j * k..(j + 1) * k];
                for r in 0..t {
                    let mut acc = 0f32;
                    for kk in 0..k {
                        let src = r as isize + kk as isize - (k as isize - 1);
                        if src >= 0 {
                            acc += w[kk] * bx[src as usize * d + j];
                        }
                    }
                    unsafe { *yp.ptr().add(r * d + j) = acc * bcx[r * 3 * d + d + j] };
                }
            }
        });
        lin_many(pool, &self.acc, &mut [(&c.out_proj, out)], &y, t);
    }

    /// Attention over the whole prompt; `vis = None` → sliding causal window.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, pool: &ThreadPool, a: &Attn, h: &[f32], t: usize, pos: &[usize], vis: Option<&Visibility>, out: &mut [f32]) {
        let Dims { heads, kv_heads, hd, eps, window, .. } = self.dims.clone();
        let (qd, kvd) = (heads * hd, kv_heads * hd);
        let mut q = vec![0f32; t * qd];
        let mut k = vec![0f32; t * kvd];
        let mut v = vec![0f32; t * kvd];
        let mut gate = vec![1f32; t * heads];
        match &a.gate {
            Some(gw) => lin_many(pool, &self.acc, &mut [(&a.q, &mut q), (&a.k, &mut k), (&a.v, &mut v), (gw, &mut gate)], h, t),
            None => lin_many(pool, &self.acc, &mut [(&a.q, &mut q), (&a.k, &mut k), (&a.v, &mut v)], h, t),
        }
        let rope = if vis.is_some() { &self.rope_full } else { &self.rope_sliding };
        let mut cs = vec![0f32; hd];
        for r in 0..t {
            rope.table(pos[r], &mut cs);
            for x in q[r * qd..(r + 1) * qd].chunks_exact_mut(hd) {
                ops::rmsnorm_inplace(x, &a.qn, eps);
            }
            for x in k[r * kvd..(r + 1) * kvd].chunks_exact_mut(hd) {
                ops::rmsnorm_inplace(x, &a.kn, eps);
            }
            rope.apply(&mut q[r * qd..(r + 1) * qd], hd, &cs);
            rope.apply(&mut k[r * kvd..(r + 1) * kvd], hd, &cs);
        }
        if a.gate.is_some() {
            gate.iter_mut().for_each(|g| *g = 1.0 / (1.0 + (-*g).exp()));
        }
        let rep = heads / kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0f32; t * qd];
        let cp = SyncPtr::new(&mut ctx);
        let (q, k, v, gate) = (&q, &k, &v, &gate);
        pool.parallel_for(heads * t.div_ceil(16), |task, _| {
            let hh = task % heads;
            let r0 = (task / heads) * 16;
            let kh = hh / rep;
            let mut sc = vec![0f32; t];
            for i in r0..(r0 + 16).min(t) {
                let qi = &q[i * qd + hh * hd..i * qd + (hh + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..t {
                    let ok = match vis {
                        Some(vv) => vv.sees(i, j),
                        None => j <= i && i - j < window,
                    };
                    sc[j] = if ok {
                        let kj = &k[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                        qi.iter().zip(kj).map(|(x, y)| x * y).sum::<f32>() * scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    mx = mx.max(sc[j]);
                }
                let mut s = 0f32;
                for x in sc.iter_mut() {
                    *x = if x.is_finite() { (*x - mx).exp() } else { 0.0 };
                    s += *x;
                }
                let o = unsafe { cp.slice(i * qd + hh * hd, hd) };
                for j in 0..t {
                    if sc[j] > 0.0 {
                        let w = sc[j] / s;
                        let vj = &v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                        for e in 0..hd {
                            o[e] += w * vj[e];
                        }
                    }
                }
                let g = gate[i * heads + hh];
                o.iter_mut().for_each(|x| *x *= g);
            }
        });
        lin_many(pool, &self.acc, &mut [(&a.o, out)], &ctx, t);
    }
}
