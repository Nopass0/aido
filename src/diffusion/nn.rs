//! Neural-network building blocks for convolutional / diffusion models.
//!
//! Activations use **HWC layout** (`[height][width][channels]`): every
//! pixel is a row, so 1×1 convolutions and linear layers are plain GEMMs,
//! 3×3 convolutions are GEMMs over an im2col matrix, and attention over
//! pixels needs no transposes. All heavy lifting goes through the packed
//! [`crate::tensor::gemm::sgemm`].

use anyhow::{Context, Result};

use crate::format::safetensors::SafetensorsDir;
use crate::runtime::pool::ThreadPool;
use crate::tensor::gemm::{sgemm, PackedB};
use crate::tensor::SyncPtr;

/// An HWC activation tensor (batch 1).
#[derive(Clone, Debug)]
pub struct T {
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Channels.
    pub c: usize,
    /// Data, `h·w·c` floats.
    pub d: Vec<f32>,
}

impl T {
    /// Zero tensor.
    ///
    /// # Example
    /// ```
    /// let t = aido::diffusion::nn::T::zeros(2, 3, 4);
    /// assert_eq!(t.d.len(), 24);
    /// ```
    pub fn zeros(h: usize, w: usize, c: usize) -> Self {
        T { h, w, c, d: vec![0.0; h * w * c] }
    }

    /// Number of pixels (`h·w`).
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::diffusion::nn::T::zeros(2, 3, 1).pixels(), 6);
    /// ```
    pub fn pixels(&self) -> usize {
        self.h * self.w
    }
}

/// A linear layer `y = x·Wᵀ + b` with pre-packed weights.
pub struct Linear {
    /// Packed weight.
    pub w: PackedB,
    /// Optional bias.
    pub b: Option<Vec<f32>>,
}

impl Linear {
    /// Loads `name.weight` (`[out, in]`) and optional `name.bias`.
    ///
    /// # Example
    /// ```no_run
    /// # let st = aido::format::safetensors::SafetensorsDir::open("models/sd-turbo/unet").unwrap();
    /// let l = aido::diffusion::nn::Linear::load(&st, "time_embedding.linear_1").unwrap();
    /// ```
    pub fn load(st: &SafetensorsDir, name: &str) -> Result<Self> {
        let wn = format!("{name}.weight");
        let shape = st.shape(&wn).with_context(|| format!("missing {wn}"))?.to_vec();
        let w = st.f32(&wn)?;
        let (n, k) = (shape[0], shape[1..].iter().product());
        let bn = format!("{name}.bias");
        let b = if st.shape(&bn).is_some() { Some(st.f32(&bn)?) } else { None };
        Ok(Linear { w: PackedB::from_nk(&w, n, k), b })
    }

    /// Applies the layer to `m` rows of `x` (row-major, width `in`).
    ///
    /// # Example
    /// ```no_run
    /// # let l: aido::diffusion::nn::Linear = unimplemented!();
    /// # let pool = aido::runtime::pool::ThreadPool::new(4, None);
    /// let y = l.forward(&pool, &vec![0.0; 320], 1);
    /// ```
    pub fn forward(&self, pool: &ThreadPool, x: &[f32], m: usize) -> Vec<f32> {
        let mut y = vec![0f32; m * self.w.n];
        sgemm(pool, m, x, self.w.k, &self.w, self.b.as_deref(), &mut y, self.w.n, false);
        y
    }
}

/// A 2-D convolution (kernel 1 or 3, padding k/2) as im2col + GEMM.
pub struct Conv {
    /// Packed weight, K ordered `(ky, kx, cin)`.
    pub w: PackedB,
    /// Bias.
    pub b: Option<Vec<f32>>,
    /// Kernel size.
    pub k: usize,
    /// Input channels.
    pub cin: usize,
    /// Stride.
    pub stride: usize,
}

impl Conv {
    /// Loads a PyTorch conv (`[cout, cin, k, k]`) and reorders it for HWC.
    ///
    /// # Example
    /// ```no_run
    /// # let st = aido::format::safetensors::SafetensorsDir::open("models/sd-turbo/unet").unwrap();
    /// let c = aido::diffusion::nn::Conv::load(&st, "conv_in", 1).unwrap();
    /// ```
    pub fn load(st: &SafetensorsDir, name: &str, stride: usize) -> Result<Self> {
        let wn = format!("{name}.weight");
        let shape = st.shape(&wn).with_context(|| format!("missing {wn}"))?.to_vec();
        let w = st.f32(&wn)?;
        let (cout, cin, k) = (shape[0], shape[1], shape[2]);
        let mut r = vec![0f32; w.len()];
        for o in 0..cout {
            for ci in 0..cin {
                for ky in 0..k {
                    for kx in 0..k {
                        r[o * k * k * cin + (ky * k + kx) * cin + ci] = w[((o * cin + ci) * k + ky) * k + kx];
                    }
                }
            }
        }
        let bn = format!("{name}.bias");
        let b = if st.shape(&bn).is_some() { Some(st.f32(&bn)?) } else { None };
        Ok(Conv { w: PackedB::from_nk(&r, cout, k * k * cin), b, k, cin, stride })
    }

    /// Output channels.
    ///
    /// # Example
    /// ```no_run
    /// # let c: aido::diffusion::nn::Conv = unimplemented!();
    /// let n = c.cout();
    /// ```
    pub fn cout(&self) -> usize {
        self.w.n
    }

    /// Applies the convolution.
    ///
    /// # Example
    /// ```no_run
    /// # let c: aido::diffusion::nn::Conv = unimplemented!();
    /// # let pool = aido::runtime::pool::ThreadPool::new(4, None);
    /// let y = c.forward(&pool, &aido::diffusion::nn::T::zeros(8, 8, c.cin));
    /// ```
    pub fn forward(&self, pool: &ThreadPool, x: &T) -> T {
        let (k, s) = (self.k, self.stride);
        let pad = k / 2;
        let ho = (x.h + 2 * pad - k) / s + 1;
        let wo = (x.w + 2 * pad - k) / s + 1;
        let mut y = T::zeros(ho, wo, self.cout());
        if k == 1 && s == 1 {
            sgemm(pool, x.pixels(), &x.d, x.c, &self.w, self.b.as_deref(), &mut y.d, y.c, false);
            return y;
        }
        let kk = k * k * x.c;
        // im2col in row bands to bound memory (≤ ~64 MB per band).
        let band = (16_000_000 / kk).max(wo).min(ho * wo);
        let mut col = vec![0f32; band * kk];
        let mut p0 = 0;
        while p0 < ho * wo {
            let p1 = (p0 + band).min(ho * wo);
            let cp = SyncPtr::new(&mut col);
            pool.parallel_chunks(p1 - p0, 64, |a, b, _| {
                for i in a..b {
                    let p = p0 + i;
                    let (oy, ox) = (p / wo, p % wo);
                    let row = unsafe { cp.slice(i * kk, kk) };
                    for ky in 0..k {
                        let iy = (oy * s + ky) as isize - pad as isize;
                        for kx in 0..k {
                            let ix = (ox * s + kx) as isize - pad as isize;
                            let dst = &mut row[(ky * k + kx) * x.c..(ky * k + kx + 1) * x.c];
                            if iy < 0 || ix < 0 || iy >= x.h as isize || ix >= x.w as isize {
                                dst.fill(0.0);
                            } else {
                                let src = ((iy as usize) * x.w + ix as usize) * x.c;
                                dst.copy_from_slice(&x.d[src..src + x.c]);
                            }
                        }
                    }
                }
            });
            sgemm(pool, p1 - p0, &col, kk, &self.w, self.b.as_deref(), &mut y.d[p0 * y.c..p1 * y.c], y.c, false);
            p0 = p1;
        }
        y
    }
}

/// GroupNorm (+ optional SiLU) parameters.
pub struct GroupNorm {
    /// Gain.
    pub g: Vec<f32>,
    /// Bias.
    pub b: Vec<f32>,
    /// Groups.
    pub groups: usize,
    /// Epsilon.
    pub eps: f32,
}

impl GroupNorm {
    /// Loads `name.weight` / `name.bias`.
    ///
    /// # Example
    /// ```no_run
    /// # let st = aido::format::safetensors::SafetensorsDir::open("models/sd-turbo/unet").unwrap();
    /// let gn = aido::diffusion::nn::GroupNorm::load(&st, "conv_norm_out", 32, 1e-5).unwrap();
    /// ```
    pub fn load(st: &SafetensorsDir, name: &str, groups: usize, eps: f32) -> Result<Self> {
        Ok(GroupNorm { g: st.f32(&format!("{name}.weight"))?, b: st.f32(&format!("{name}.bias"))?, groups, eps })
    }

    /// Normalises `x` in place, optionally followed by SiLU.
    ///
    /// # Example
    /// ```
    /// use aido::diffusion::nn::{GroupNorm, T};
    /// let gn = GroupNorm { g: vec![1.0; 2], b: vec![0.0; 2], groups: 1, eps: 0.0 };
    /// let mut t = T { h: 1, w: 2, c: 2, d: vec![1.0, 3.0, 1.0, 3.0] };
    /// gn.apply(&mut t, false);
    /// assert!((t.d[0] + 1.0).abs() < 1e-5);
    /// ```
    pub fn apply(&self, x: &mut T, silu: bool) {
        let c = x.c;
        let cg = c / self.groups;
        let n = x.pixels();
        for g in 0..self.groups {
            let (mut s, mut s2) = (0f64, 0f64);
            for p in 0..n {
                for v in &x.d[p * c + g * cg..p * c + (g + 1) * cg] {
                    s += *v as f64;
                    s2 += (*v as f64) * (*v as f64);
                }
            }
            let cnt = (n * cg) as f64;
            let mean = s / cnt;
            let var = (s2 / cnt - mean * mean).max(0.0);
            let inv = 1.0 / (var + self.eps as f64).sqrt();
            for p in 0..n {
                for j in 0..cg {
                    let ch = g * cg + j;
                    let v = &mut x.d[p * c + ch];
                    let mut y = ((*v as f64 - mean) * inv) as f32 * self.g[ch] + self.b[ch];
                    if silu {
                        y /= 1.0 + (-y).exp();
                    }
                    *v = y;
                }
            }
        }
    }
}

/// LayerNorm over the last dimension of `m` rows.
pub struct LayerNorm {
    /// Gain.
    pub g: Vec<f32>,
    /// Bias.
    pub b: Vec<f32>,
    /// Epsilon.
    pub eps: f32,
}

impl LayerNorm {
    /// Loads `name.weight` / `name.bias`.
    ///
    /// # Example
    /// ```no_run
    /// # let st = aido::format::safetensors::SafetensorsDir::open("models/sd-turbo/text_encoder").unwrap();
    /// let ln = aido::diffusion::nn::LayerNorm::load(&st, "text_model.final_layer_norm", 1e-5).unwrap();
    /// ```
    pub fn load(st: &SafetensorsDir, name: &str, eps: f32) -> Result<Self> {
        Ok(LayerNorm { g: st.f32(&format!("{name}.weight"))?, b: st.f32(&format!("{name}.bias"))?, eps })
    }

    /// Returns the normalised copy of `x` (`m` rows).
    ///
    /// # Example
    /// ```
    /// let ln = aido::diffusion::nn::LayerNorm { g: vec![1.0; 2], b: vec![0.0; 2], eps: 0.0 };
    /// let y = ln.forward(&[1.0, 3.0]);
    /// assert!((y[1] - 1.0).abs() < 1e-5);
    /// ```
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        for row in y.chunks_exact_mut(self.g.len()) {
            crate::tensor::ops::layernorm_inplace(row, &self.g, Some(&self.b), self.eps);
        }
        y
    }
}

/// Multi-head attention over rows: `q` (`n × c`), `k`, `v` (`m × c`),
/// returns `n × c`. Each head's `QKᵀ` and `PV` products run on the GEMM.
///
/// # Example
/// ```
/// let pool = aido::runtime::pool::ThreadPool::new(2, None);
/// let q = vec![1.0f32; 2 * 4];
/// let o = aido::diffusion::nn::attention(&pool, &q, &q, &q, 2, 2, 4, 2);
/// assert!((o[0] - 1.0).abs() < 1e-5);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn attention(pool: &ThreadPool, q: &[f32], k: &[f32], v: &[f32], n: usize, m: usize, c: usize, heads: usize) -> Vec<f32> {
    let hd = c / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; n * c];
    let mut qh = vec![0f32; n * hd];
    let mut kh = vec![0f32; m * hd];
    let mut vh = vec![0f32; m * hd];
    let mut s = vec![0f32; n * m];
    let mut oh = vec![0f32; n * hd];
    for h in 0..heads {
        for i in 0..n {
            for j in 0..hd {
                qh[i * hd + j] = q[i * c + h * hd + j] * scale;
            }
        }
        for i in 0..m {
            kh[i * hd..(i + 1) * hd].copy_from_slice(&k[i * c + h * hd..i * c + (h + 1) * hd]);
            vh[i * hd..(i + 1) * hd].copy_from_slice(&v[i * c + h * hd..i * c + (h + 1) * hd]);
        }
        let kb = PackedB::from_nk(&kh, m, hd);
        sgemm(pool, n, &qh, hd, &kb, None, &mut s, m, false);
        let sp = SyncPtr::new(&mut s);
        pool.parallel_chunks(n, 64, |a, b, _| {
            for i in a..b {
                crate::tensor::ops::softmax_inplace(unsafe { sp.slice(i * m, m) });
            }
        });
        let vb = PackedB::from_kn(&vh, m, hd);
        sgemm(pool, n, &s, m, &vb, None, &mut oh, hd, false);
        for i in 0..n {
            out[i * c + h * hd..i * c + (h + 1) * hd].copy_from_slice(&oh[i * hd..(i + 1) * hd]);
        }
    }
    out
}

/// Nearest-neighbour 2× upsampling.
///
/// # Example
/// ```
/// let t = aido::diffusion::nn::T { h: 1, w: 1, c: 1, d: vec![5.0] };
/// let u = aido::diffusion::nn::upsample2x(&t);
/// assert_eq!(u.d, vec![5.0; 4]);
/// ```
pub fn upsample2x(x: &T) -> T {
    let mut y = T::zeros(x.h * 2, x.w * 2, x.c);
    for oy in 0..y.h {
        for ox in 0..y.w {
            let s = ((oy / 2) * x.w + ox / 2) * x.c;
            let d = (oy * y.w + ox) * x.c;
            y.d[d..d + x.c].copy_from_slice(&x.d[s..s + x.c]);
        }
    }
    y
}

/// Channel concatenation `[a | b]` per pixel.
///
/// # Example
/// ```
/// use aido::diffusion::nn::{concat, T};
/// let a = T { h: 1, w: 1, c: 1, d: vec![1.0] };
/// let b = T { h: 1, w: 1, c: 2, d: vec![2.0, 3.0] };
/// assert_eq!(concat(&a, &b).d, vec![1.0, 2.0, 3.0]);
/// ```
pub fn concat(a: &T, b: &T) -> T {
    let mut y = T::zeros(a.h, a.w, a.c + b.c);
    for p in 0..a.pixels() {
        y.d[p * y.c..p * y.c + a.c].copy_from_slice(&a.d[p * a.c..(p + 1) * a.c]);
        y.d[p * y.c + a.c..(p + 1) * y.c].copy_from_slice(&b.d[p * b.c..(p + 1) * b.c]);
    }
    y
}

/// Exact GELU (erf form, as PyTorch `gelu`), in place.
///
/// # Example
/// ```
/// let mut v = [0.0f32, 3.0];
/// aido::diffusion::nn::gelu_erf(&mut v);
/// assert!(v[0].abs() < 1e-6 && (v[1] - 2.996).abs() < 1e-2);
/// ```
pub fn gelu_erf(x: &mut [f32]) {
    for v in x.iter_mut() {
        let u = *v as f64;
        *v = (0.5 * u * (1.0 + erf(u / std::f64::consts::SQRT_2))) as f32;
    }
}

/// Error function (Abramowitz–Stegun 7.1.26, |ε| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
