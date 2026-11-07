//! Numeric building blocks: normalisation, activations, RoPE, attention and
//! the parallel, cache-tiled quantised matmul.

use half::f16;

use super::quant::{dot_fn, uses_q8, Q8Act};
use super::{QWeight, SyncPtr};
use crate::runtime::pool::ThreadPool;

/// RMS normalisation: `out = x / sqrt(mean(x²) + eps) * w`.
///
/// * **Input** — `x`, gain `w` (same length), `eps`.
/// * **Output** — written into `out` (may not alias `x`; see
///   [`rmsnorm_inplace`]).
///
/// # Example
/// ```
/// let mut o = [0f32; 2];
/// aido::tensor::ops::rmsnorm(&mut o, &[3.0, 4.0], &[1.0, 1.0], 0.0);
/// assert!((o[0] - 3.0 / (12.5f32).sqrt()).abs() < 1e-6);
/// ```
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// In-place RMS normalisation (see [`rmsnorm`]).
///
/// # Example
/// ```
/// let mut x = [2.0f32, 2.0];
/// aido::tensor::ops::rmsnorm_inplace(&mut x, &[1.0, 0.5], 0.0);
/// assert_eq!(x, [1.0, 0.5]);
/// ```
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        x[i] *= inv * w[i];
    }
}

/// LayerNorm with optional bias: `out = (x-μ)/σ * w + b`.
///
/// # Example
/// ```
/// let mut x = [1.0f32, 3.0];
/// aido::tensor::ops::layernorm_inplace(&mut x, &[1.0, 1.0], Some(&[0.0, 0.0]), 0.0);
/// assert!((x[0] + 1.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5);
/// ```
pub fn layernorm_inplace(x: &mut [f32], w: &[f32], b: Option<&[f32]>, eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * w[i] + b.map_or(0.0, |b| b[i]);
    }
}

/// `dst += src` element-wise.
///
/// # Example
/// ```
/// let mut a = [1.0f32, 2.0];
/// aido::tensor::ops::add_inplace(&mut a, &[1.0, 1.0]);
/// assert_eq!(a, [2.0, 3.0]);
/// ```
#[inline]
pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// SwiGLU gate: `gate = silu(gate) * up` (fused, one pass).
///
/// # Example
/// ```
/// let mut g = [0.0f32];
/// aido::tensor::ops::silu_mul(&mut g, &[5.0]);
/// assert_eq!(g[0], 0.0);
/// ```
#[inline]
pub fn silu_mul(gate: &mut [f32], up: &[f32]) {
    for (g, u) in gate.iter_mut().zip(up) {
        *g = *g / (1.0 + (-*g).exp()) * *u;
    }
}

/// SiLU (`x·σ(x)`) in place.
///
/// # Example
/// ```
/// let mut x = [0.0f32];
/// aido::tensor::ops::silu_inplace(&mut x);
/// assert_eq!(x[0], 0.0);
/// ```
#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// GELU (tanh approximation) in place.
///
/// # Example
/// ```
/// let mut x = [0.0f32, 10.0];
/// aido::tensor::ops::gelu_inplace(&mut x);
/// assert!(x[0].abs() < 1e-6 && (x[1] - 10.0).abs() < 1e-3);
/// ```
#[inline]
pub fn gelu_inplace(x: &mut [f32]) {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + (C * (u + 0.044715 * u * u * u)).tanh());
    }
}

/// Quick-GELU (`x·σ(1.702x)`) used by OpenAI CLIP.
///
/// # Example
/// ```
/// let mut x = [0.0f32];
/// aido::tensor::ops::quick_gelu_inplace(&mut x);
/// assert_eq!(x[0], 0.0);
/// ```
#[inline]
pub fn quick_gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-1.702 * *v).exp();
    }
}

/// Numerically stable softmax in place.
///
/// # Example
/// ```
/// let mut x = [1.0f32, 1.0];
/// aido::tensor::ops::softmax_inplace(&mut x);
/// assert_eq!(x, [0.5, 0.5]);
/// ```
pub fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Rotary position embedding parameters.
#[derive(Debug, Clone)]
pub struct Rope {
    /// Inverse frequencies, length `rot_dim / 2` (already divided by
    /// optional per-dimension scaling factors such as Llama-3.1's).
    pub inv_freq: Vec<f32>,
    /// Number of rotated dimensions per head (≤ head_dim).
    pub rot_dim: usize,
    /// `true` = NeoX layout (rotate halves), `false` = GPT-J (adjacent pairs).
    pub neox: bool,
}

impl Rope {
    /// Builds RoPE tables.
    ///
    /// * **Input** — `rot_dim`, `base` (θ), optional `factors` (one per
    ///   frequency, divides it — GGUF `rope_freqs.weight`), `neox` layout flag.
    ///
    /// # Example
    /// ```
    /// let r = aido::tensor::ops::Rope::new(4, 10000.0, None, true);
    /// assert_eq!(r.inv_freq.len(), 2);
    /// assert_eq!(r.inv_freq[0], 1.0);
    /// ```
    pub fn new(rot_dim: usize, base: f32, factors: Option<&[f32]>, neox: bool) -> Self {
        let inv_freq = (0..rot_dim / 2)
            .map(|i| {
                let f = 1.0 / base.powf(2.0 * i as f32 / rot_dim as f32);
                f / factors.map_or(1.0, |fa| fa[i])
            })
            .collect();
        Rope { inv_freq, rot_dim, neox }
    }

    /// Computes the `(cos, sin)` table for position `pos`.
    ///
    /// * **Output** — `cs[2i] = cos(pos·f_i)`, `cs[2i+1] = sin(pos·f_i)`.
    ///
    /// # Example
    /// ```
    /// let r = aido::tensor::ops::Rope::new(2, 10000.0, None, true);
    /// let mut cs = [0f32; 2];
    /// r.table(0, &mut cs);
    /// assert_eq!(cs, [1.0, 0.0]);
    /// ```
    pub fn table(&self, pos: usize, cs: &mut [f32]) {
        for (i, f) in self.inv_freq.iter().enumerate() {
            let (s, c) = (pos as f64 * *f as f64).sin_cos();
            cs[2 * i] = c as f32;
            cs[2 * i + 1] = s as f32;
        }
    }

    /// Rotates every head of `x` (`n_heads × head_dim`) using table `cs`.
    ///
    /// # Example
    /// ```
    /// let r = aido::tensor::ops::Rope::new(2, 10000.0, None, true);
    /// let mut x = [1.0f32, 0.0];
    /// let cs = [0.0f32, 1.0]; // 90 degrees
    /// r.apply(&mut x, 2, &cs);
    /// assert!((x[0]).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
    /// ```
    pub fn apply(&self, x: &mut [f32], head_dim: usize, cs: &[f32]) {
        let half = self.rot_dim / 2;
        for h in x.chunks_exact_mut(head_dim) {
            if self.neox {
                for i in 0..half {
                    let (c, s) = (cs[2 * i], cs[2 * i + 1]);
                    let (a, b) = (h[i], h[i + half]);
                    h[i] = a * c - b * s;
                    h[i + half] = a * s + b * c;
                }
            } else {
                for i in 0..half {
                    let (c, s) = (cs[2 * i], cs[2 * i + 1]);
                    let (a, b) = (h[2 * i], h[2 * i + 1]);
                    h[2 * i] = a * c - b * s;
                    h[2 * i + 1] = a * s + b * c;
                }
            }
        }
    }
}

/// Converts f32 values to f16 storage.
///
/// # Example
/// ```
/// let mut o = [half::f16::ZERO; 1];
/// aido::tensor::ops::f32_to_f16(&[1.5], &mut o);
/// assert_eq!(o[0].to_f32(), 1.5);
/// ```
pub fn f32_to_f16(src: &[f32], dst: &mut [f16]) {
    #[cfg(target_arch = "x86_64")]
    if super::quant::has_avx2() {
        unsafe { simd::f32_to_f16(src, dst) };
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = f16::from_f32(*s);
    }
}

/// Dot product of an f32 vector with an f16 vector.
///
/// # Example
/// ```
/// use half::f16;
/// let k = [f16::from_f32(2.0); 8];
/// assert_eq!(aido::tensor::ops::dot_f32_f16(&[1.0; 8], &k), 16.0);
/// ```
#[inline]
pub fn dot_f32_f16(a: &[f32], b: &[f16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::quant::has_avx2() {
        return unsafe { simd::dot_f32_f16(a, b) };
    }
    a.iter().zip(b).map(|(x, y)| x * y.to_f32()).sum()
}

/// `out += w * v` with `v` in f16.
///
/// # Example
/// ```
/// use half::f16;
/// let mut o = [1.0f32; 8];
/// aido::tensor::ops::axpy_f16(&mut o, 0.5, &[f16::from_f32(2.0); 8]);
/// assert_eq!(o[3], 2.0);
/// ```
#[inline]
pub fn axpy_f16(out: &mut [f32], w: f32, v: &[f16]) {
    #[cfg(target_arch = "x86_64")]
    if super::quant::has_avx2() {
        unsafe { simd::axpy_f16(out, w, v) };
        return;
    }
    for (o, x) in out.iter_mut().zip(v) {
        *o += w * x.to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use half::f16;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn f32_to_f16(src: &[f32], dst: &mut [f16]) {
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm256_cvtps_ph(_mm256_loadu_ps(src.as_ptr().add(i)), _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
            i += 8;
        }
        while i < n {
            dst[i] = f16::from_f32(src[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f32_f16(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            let k1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 8) as *const __m128i));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), k0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), k1, acc1);
            i += 16;
        }
        while i + 8 <= n {
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), k0, acc0);
            i += 8;
        }
        let mut s = crate::tensor::ops::simd_hsum(_mm256_add_ps(acc0, acc1));
        while i < n {
            s += a[i] * b[i].to_f32();
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn axpy_f16(out: &mut [f32], w: f32, v: &[f16]) {
        let n = out.len();
        let wv = _mm256_set1_ps(w);
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(v.as_ptr().add(i) as *const __m128i));
            let o = _mm256_loadu_ps(out.as_ptr().add(i));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(wv, x, o));
            i += 8;
        }
        while i < n {
            out[i] += w * v[i].to_f32();
            i += 1;
        }
    }
}

/// Horizontal sum helper shared by SIMD code.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn simd_hsum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// Quantises `t` activation rows (each `cols` long) for use by [`matmul`].
///
/// * **Input** — `xs`: `t × cols` floats; `out`: at least `t` buffers.
/// * **Behaviour** — rows are quantised in parallel.
///
/// # Example
/// ```
/// use aido::{runtime::pool::ThreadPool, tensor::{ops::quantize_rows, quant::Q8Act}};
/// let pool = ThreadPool::new(2, None);
/// let mut q = vec![Q8Act::default(); 2];
/// quantize_rows(&pool, &[1.0; 64], 32, &mut q);
/// assert_eq!(q[1].qs[0], 127);
/// ```
pub fn quantize_rows(pool: &ThreadPool, xs: &[f32], cols: usize, out: &mut [Q8Act]) {
    let t = xs.len() / cols;
    for q in out[..t].iter_mut() {
        if q.qs.len() != cols {
            *q = Q8Act::with_len(cols);
        }
    }
    let p = SyncPtr::new(out);
    if t == 1 {
        out[0].quantize(&xs[..cols]);
        return;
    }
    pool.parallel_for(t, |i, _| unsafe { (*p.ptr().add(i)).quantize(&xs[i * cols..(i + 1) * cols]) });
}

/// Parallel, cache-tiled `out[t] = W · x[t]` for `t` tokens.
///
/// * **Input** — `w` (`rows × cols`), `xs` (`t × cols` f32, used by float
///   weights), `xq` (`t` quantised rows, used by integer weights; pass the
///   result of [`quantize_rows`]), `out` (`t × rows`).
/// * **Behaviour** — rows are split into chunks handed out dynamically to
///   workers. Inside a chunk the loop is tiled `ROW_TILE × TOK_TILE` so a tile
///   of weights (≤ L1) is reused against a tile of activations: for decoding
///   (`t = 1`) this is a pure stream over the weights at full memory
///   bandwidth; for prompt processing / speculative verification each weight
///   byte fetched from DRAM is used `t` times — the arithmetic-intensity
///   trick that makes batched verification almost free.
///
/// # Example
/// ```
/// use aido::{runtime::pool::ThreadPool, tensor::{QWeight, ops::{matmul, quantize_rows}, quant::Q8Act}};
/// let pool = ThreadPool::new(2, None);
/// let w = QWeight::from_f32(2, 32, &[1.0; 64]);
/// let x = vec![0.5f32; 32];
/// let mut q = vec![Q8Act::default()];
/// quantize_rows(&pool, &x, 32, &mut q);
/// let mut out = [0f32; 2];
/// matmul(&pool, &w, &x, &q, &mut out);
/// assert_eq!(out, [16.0, 16.0]);
/// ```
pub fn matmul(pool: &ThreadPool, w: &QWeight, xs: &[f32], xq: &[Q8Act], out: &mut [f32]) {
    let cols = w.cols;
    let rows = w.rows;
    let t = xs.len() / cols;
    debug_assert!(out.len() >= t * rows);
    let f = dot_fn(w.ty);
    let need_q = uses_q8(w.ty);
    let empty = Q8Act::default();
    let rb = w.ty.row_bytes(cols);
    let data = w.data.as_slice();
    let op = SyncPtr::new(out);
    // Chunking: ~8 tasks per thread for load balance, multiple of 4 rows.
    let threads = pool.width().max(1);
    let mut chunk = (rows / (threads * 8)).max(4);
    chunk = (chunk + 3) / 4 * 4;
    const ROW_TILE: usize = 8;
    const TOK_TILE: usize = 16;
    pool.parallel_chunks(rows, chunk, |r0, r1, _| {
        let mut rt = r0;
        while rt < r1 {
            let re = (rt + ROW_TILE).min(r1);
            let mut tt = 0;
            while tt < t {
                let te = (tt + TOK_TILE).min(t);
                for ti in tt..te {
                    let x = &xs[ti * cols..(ti + 1) * cols];
                    let q = if need_q { &xq[ti] } else { &empty };
                    for r in rt..re {
                        let v = f(&data[r * rb..(r + 1) * rb], x, q);
                        unsafe { *op.ptr().add(ti * rows + r) = v };
                    }
                }
                tt = te;
            }
            rt = re;
        }
    });
}

/// Causal multi-query attention for one query group against an f16 KV cache.
///
/// Processes the `g` query heads that share one KV head together, so each
/// cached key/value row is loaded **once** for all `g` heads (grouped-query
/// attention reuse — cuts KV-cache bandwidth by `g`, e.g. 4× for Llama-3-8B).
///
/// * **Input** — `q`: `g × hd` queries; `k`, `v`: cache rows for this KV
///   head, row `p` at `[p*stride .. p*stride+hd]`; `n_ctx`: number of visible
///   positions; `scale`: usually `1/sqrt(hd)`; `scores`: scratch of
///   `g × n_ctx`; `out`: `g × hd` result.
///
/// # Example
/// ```
/// use half::f16;
/// let q = [1.0f32; 8];
/// let k = vec![f16::from_f32(0.0); 16];
/// let v: Vec<f16> = (0..16).map(|i| f16::from_f32(if i < 8 { 1.0 } else { 3.0 })).collect();
/// let mut sc = vec![0f32; 2];
/// let mut out = [0f32; 8];
/// aido::tensor::ops::attend_group(&q, &k, &v, 8, 2, 8, 1.0, &mut sc, &mut out);
/// assert!((out[0] - 2.0).abs() < 1e-3); // equal weights -> mean of 1 and 3
/// ```
#[allow(clippy::too_many_arguments)]
pub fn attend_group(
    q: &[f32],
    k: &[f16],
    v: &[f16],
    stride: usize,
    n_ctx: usize,
    hd: usize,
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
) {
    let g = q.len() / hd;
    for p in 0..n_ctx {
        let kr = &k[p * stride..p * stride + hd];
        for h in 0..g {
            scores[h * n_ctx + p] = dot_f32_f16(&q[h * hd..(h + 1) * hd], kr) * scale;
        }
    }
    for h in 0..g {
        softmax_inplace(&mut scores[h * n_ctx..(h + 1) * n_ctx]);
    }
    out.fill(0.0);
    for p in 0..n_ctx {
        let vr = &v[p * stride..p * stride + hd];
        for h in 0..g {
            let wgt = scores[h * n_ctx + p];
            axpy_f16(&mut out[h * hd..(h + 1) * hd], wgt, vr);
        }
    }
}
