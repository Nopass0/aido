//! Image super-resolution models (e.g. **Baikal LoopSR** anime upscaler,
//! `SnJake/Baikal-Anime-Upscaler`) for single pictures and frame streams.
//!
//! ## LoopSR
//! `lr → stem 3×3 → A (pre blocks) → B (shared blocks, applied `loops`
//! times) → C (post blocks) → refine 3×3 (+ shallow skip) → head (3×3, GELU,
//! 3×3, PixelShuffle) + bicubic(lr)`. Every block is a Swin-style window
//! transformer: LayerNorm → 8×8 window attention with a learned relative
//! position bias (alternate blocks shifted by 4 with a region mask; body
//! blocks use *exclusive self-attention*: the component along the token's
//! own value direction is removed) → LayerNorm → `fc1` → depthwise 3×3 →
//! GELU-gated `fc2`, both branches scaled by learned per-channel factors.
//!
//! All matrix work is batched over pixels: `pixels × dim` packed CPU GEMMs
//! (GPU GEMMs with `AIDO_SR_GPU=1`; slower on PCIe GPUs because every layer's
//! activations cross the bus twice). Window attention (head-major gathered
//! Q/K/V, precomputed bias tables, AVX dot products), the depthwise 3×3
//! (tap-major weights) and LayerNorms run in parallel on the CPU pool.
//!
//! Cost ≈ 23 MFLOP per input pixel at 4 loops (≈13 at 1 loop): a 256×256
//! frame is ≈1.5 TFLOP, so real-time video needs small frames or
//! `--loops 1`; see `docs/UPSCALE.md` for measured numbers.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::format::safetensors::SafetensorsDir;
use crate::gpu::{Gpu, GpuMatrix};
use crate::runtime::pool::ThreadPool;
use crate::tensor::gemm::{sgemm, PackedB};
use crate::tensor::SyncPtr;

/// A dense layer `y = x·Wᵀ + b` usable on CPU (packed GEMM) or GPU.
struct Dense {
    pb: PackedB,
    g: Option<GpuMatrix>,
    b: Vec<f32>,
    n: usize,
}

impl Dense {
    fn load(st: &SafetensorsDir, name: &str, gpu: Option<&Gpu>) -> Result<Self> {
        let shape = st.shape(&format!("{name}.weight")).with_context(|| format!("missing {name}"))?.to_vec();
        let w = st.f32(&format!("{name}.weight"))?;
        let (n, k) = (shape[0], shape[1..].iter().product::<usize>());
        let g = match gpu {
            Some(g) if k % 32 == 0 => g.upload_f32(&w, n, k).ok(),
            _ => None,
        };
        Ok(Dense { pb: PackedB::from_nk(&w, n, k), g, b: st.f32(&format!("{name}.bias"))?, n })
    }

    fn forward(&self, pool: &ThreadPool, gpu: Option<&Gpu>, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * self.n];
        let mut done = false;
        if let (Some(g), Some(m)) = (gpu, &self.g) {
            if t >= 256 {
                done = g.gemm_many(x, t, &[m], &mut [&mut y[..]]).is_ok();
                if done {
                    for row in y.chunks_exact_mut(self.n) {
                        for (v, b) in row.iter_mut().zip(&self.b) {
                            *v += b;
                        }
                    }
                }
            }
        }
        if !done {
            sgemm(pool, t, x, self.pb.k, &self.pb, Some(&self.b), &mut y, self.n, false);
        }
        y
    }
}

/// 3×3 convolution (padding 1) on HWC via im2col + [`Dense`].
struct Conv3 {
    d: Dense,
    cin: usize,
}

impl Conv3 {
    fn load(st: &SafetensorsDir, name: &str, gpu: Option<&Gpu>) -> Result<Self> {
        // reorder [cout][cin][3][3] → [cout][(ky,kx,cin)]
        let shape = st.shape(&format!("{name}.weight")).context("conv shape")?.to_vec();
        let (cout, cin) = (shape[0], shape[1]);
        let w = st.f32(&format!("{name}.weight"))?;
        let mut r = vec![0f32; w.len()];
        for o in 0..cout {
            for c in 0..cin {
                for k in 0..9 {
                    r[o * 9 * cin + k * cin + c] = w[(o * cin + c) * 9 + k];
                }
            }
        }
        let g = match gpu {
            Some(g) if (9 * cin) % 32 == 0 => g.upload_f32(&r, cout, 9 * cin).ok(),
            _ => None,
        };
        Ok(Conv3 { d: Dense { pb: PackedB::from_nk(&r, cout, 9 * cin), g, b: st.f32(&format!("{name}.bias"))?, n: cout }, cin })
    }

    fn forward(&self, pool: &ThreadPool, gpu: Option<&Gpu>, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let c = self.cin;
        let k = 9 * c;
        let mut col = vec![0f32; h * w * k];
        let cp = SyncPtr::new(&mut col);
        pool.parallel_chunks(h * w, 256, |a, b, _| {
            for p in a..b {
                let (y, xx) = (p / w, p % w);
                let row = unsafe { cp.slice(p * k, k) };
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (iy, ix) = (y as isize + ky as isize - 1, xx as isize + kx as isize - 1);
                        let dst = &mut row[(ky * 3 + kx) * c..(ky * 3 + kx + 1) * c];
                        if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                            dst.fill(0.0);
                        } else {
                            let s = (iy as usize * w + ix as usize) * c;
                            dst.copy_from_slice(&x[s..s + c]);
                        }
                    }
                }
            }
        });
        self.d.forward(pool, gpu, &col, h * w)
    }
}

struct Block {
    n1: (Vec<f32>, Vec<f32>),
    n2: (Vec<f32>, Vec<f32>),
    qkv: Dense,
    proj: Dense,
    attn_scale: Vec<f32>,
    ffn_scale: Vec<f32>,
    fc1: Dense,
    fc2: Dense,
    /// Depthwise weights transposed tap-major: `[9][channels]`.
    local_t: Vec<f32>,
    local_b: Vec<f32>,
    /// Relative position bias expanded per head: `[head][i][j]` over window tokens.
    bias: Vec<f32>,
    shift: usize,
    xsa: bool,
}

/// A loaded LoopSR model.
pub struct LoopSr {
    /// Upscale factor.
    pub scale: usize,
    dim: usize,
    heads: usize,
    win: usize,
    /// Body repetitions (1..=trained loops; fewer = faster).
    pub loops: usize,
    max_loops: usize,
    stem: Conv3,
    pre: Vec<Block>,
    body: Vec<Block>,
    post: Vec<Block>,
    refine: Conv3,
    head0: Conv3,
    head2: Conv3,
    gpu: Option<Arc<Gpu>>,
}

/// `LayerNorm(x)` row-wise into a new buffer, parallel over pixels.
fn layernorm(pool: &ThreadPool, x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let d = g.len();
    let mut y = vec![0f32; x.len()];
    let yp = SyncPtr::new(&mut y);
    pool.parallel_chunks(x.len() / d, 256, |a, e, _| {
        let out = unsafe { yp.slice(a * d, (e - a) * d) };
        out.copy_from_slice(&x[a * d..e * d]);
        for row in out.chunks_exact_mut(d) {
            crate::tensor::ops::layernorm_inplace(row, g, Some(b), 1e-5);
        }
    });
    y
}

/// `x += y ⊙ scale` (per-channel scale), parallel over pixels.
fn add_scaled(pool: &ThreadPool, x: &mut [f32], y: &[f32], scale: &[f32]) {
    let d = scale.len();
    let xp = SyncPtr::new(x);
    pool.parallel_chunks(y.len() / d, 256, |a, e, _| {
        let out = unsafe { xp.slice(a * d, (e - a) * d) };
        for (xr, yr) in out.chunks_exact_mut(d).zip(y[a * d..e * d].chunks_exact(d)) {
            for ((xv, yv), sv) in xr.iter_mut().zip(yr).zip(scale) {
                *xv += yv * sv;
            }
        }
    });
}

impl LoopSr {
    /// Loads a LoopSR `.safetensors` (config read from its metadata or a
    /// sibling `*config.json`).
    ///
    /// # Example
    /// ```no_run
    /// let gpu = aido::gpu::Gpu::new().ok().map(std::sync::Arc::new);
    /// let m = aido::upscale::LoopSr::load("models/baikal/Baikal_LoopSR_x2_v2.safetensors", gpu).unwrap();
    /// ```
    pub fn load(path: &str, gpu: Option<Arc<Gpu>>) -> Result<Self> {
        // Measured on RTX 3070 + Ryzen: per-layer GPU GEMMs lose to the CPU
        // here because every layer's activations cross PCIe twice; the GPU
        // is used only when explicitly requested (AIDO_SR_GPU=1).
        let gpu = if std::env::var("AIDO_SR_GPU").is_ok() { gpu } else { None };
        let st = SafetensorsDir::open(path)?;
        // config from safetensors metadata
        let raw = std::fs::read(path)?;
        let n = u64::from_le_bytes(raw[0..8].try_into()?) as usize;
        let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + n])?;
        drop(raw);
        let cfg: serde_json::Value = header["__metadata__"]["model_config"].as_str().and_then(|s| serde_json::from_str(s).ok()).context("LoopSR config not found in metadata")?;
        let g = |k: &str, d: u64| cfg[k].as_u64().unwrap_or(d) as usize;
        let (dim, heads, win) = (g("dim", 192), g("heads", 6), g("window", 8));
        let xsa = cfg["attention"].as_str() == Some("xsa");
        let gp = gpu.as_deref();
        let block = |prefix: &str, i: usize, modulated: bool| -> Result<Block> {
            let p = |s: &str| format!("{prefix}.{i}.{s}");
            Ok(Block {
                n1: (st.f32(&p("norm1.weight"))?, st.f32(&p("norm1.bias"))?),
                n2: (st.f32(&p("norm2.weight"))?, st.f32(&p("norm2.bias"))?),
                qkv: Dense::load(&st, &p("attn.qkv"), gp)?,
                proj: Dense::load(&st, &p("attn.proj"), gp)?,
                bias: {
                    let rel = st.f32(&p("attn.relative_bias"))?;
                    let l = win * win;
                    let mut t = vec![0f32; heads * l * l];
                    for hh in 0..heads {
                        for i in 0..l {
                            for j in 0..l {
                                let (dy, dx) = (i / win + win - 1 - j / win, i % win + win - 1 - j % win);
                                t[(hh * l + i) * l + j] = rel[(dy * (2 * win - 1) + dx) * heads + hh];
                            }
                        }
                    }
                    t
                },
                attn_scale: st.f32(&p("attn_scale"))?,
                ffn_scale: st.f32(&p("ffn_scale"))?,
                fc1: Dense::load(&st, &p("fc1"), gp)?,
                fc2: Dense::load(&st, &p("fc2"), gp)?,
                local_t: {
                    let lw = st.f32(&p("local.weight"))?;
                    let c2 = lw.len() / 9;
                    let mut t = vec![0f32; lw.len()];
                    for c in 0..c2 {
                        for k in 0..9 {
                            t[k * c2 + c] = lw[c * 9 + k];
                        }
                    }
                    t
                },
                local_b: st.f32(&p("local.bias"))?,
                shift: if i % 2 == 0 { 0 } else { win / 2 },
                xsa: modulated && xsa,
            })
        };
        let blocks = |prefix: &str, n: usize, m: bool| -> Result<Vec<Block>> { (0..n).map(|i| block(prefix, i, m)).collect() };
        let loops = g("loops", 4);
        Ok(LoopSr {
            scale: g("scale", 2),
            dim,
            heads,
            win,
            loops,
            max_loops: loops,
            stem: Conv3::load(&st, "stem", None)?,
            pre: blocks("pre", g("pre_blocks", 2), false)?,
            body: blocks("body", g("loop_blocks", 4), true)?,
            post: blocks("post", g("post_blocks", 2), false)?,
            refine: Conv3::load(&st, "refine", gp)?,
            head0: Conv3::load(&st, "head.0", gp)?,
            head2: Conv3::load(&st, "head.2", gp)?,
            gpu,
        })
    }

    /// Sets the number of body loops (speed/quality trade-off).
    ///
    /// # Example
    /// ```no_run
    /// # let mut m: aido::upscale::LoopSr = unimplemented!();
    /// m.set_loops(1);
    /// ```
    pub fn set_loops(&mut self, n: usize) {
        self.loops = n.clamp(1, self.max_loops);
    }

    fn block(&self, pool: &ThreadPool, b: &Block, x: &mut [f32], h: usize, w: usize) {
        let (d, win, heads) = (self.dim, self.win, self.heads);
        let hd = d / heads;
        let gpu = self.gpu.as_deref();
        let n = h * w;
        let shift = if h.min(w) > win { b.shift } else { 0 };
        // ---- attention branch ----
        let y = layernorm(pool, x, &b.n1.0, &b.n1.1);
        let t0 = std::time::Instant::now();
        let qkv = b.qkv.forward(pool, gpu, &y, n);
        let t_qkv = t0.elapsed();
        // Windows of the rolled image: token t of window (wy, wx) is pixel
        // ((wy*win + t/win + shift) % h, (wx*win + t%win + shift) % w).
        let (nwy, nwx) = (h / win, w / win);
        let region = |p: usize, len: usize| -> usize {
            if shift == 0 || p < len - win {
                0
            } else if p < len - shift {
                1
            } else {
                2
            }
        };
        let mut att = vec![0f32; n * d];
        let ap = SyncPtr::new(&mut att);
        let l = win * win;
        let qkv = &qkv;
        let scale = 1.0 / (hd as f32).sqrt();
        pool.parallel_for(nwy * nwx, |wi, _| {
            let (wy, wx) = (wi / nwx, wi % nwx);
            let mut pix = vec![0usize; l];
            let mut lab = vec![0u8; l];
            for t in 0..l {
                let (ry, rx) = (wy * win + t / win, wx * win + t % win);
                lab[t] = (3 * region(ry, h) + region(rx, w)) as u8;
                pix[t] = ((ry + shift) % h) * w + (rx + shift) % w;
            }
            let masked = lab.iter().any(|&v| v != lab[0]);
            // Gather Q (pre-scaled), K, V of this window head-major: [head][token][hd].
            let mut q = vec![0f32; heads * l * hd];
            let mut k = vec![0f32; heads * l * hd];
            let mut v = vec![0f32; heads * l * hd];
            for t in 0..l {
                let row = &qkv[pix[t] * 3 * d..(pix[t] + 1) * 3 * d];
                for hh in 0..heads {
                    let o = (hh * l + t) * hd;
                    for e in 0..hd {
                        q[o + e] = row[hh * hd + e] * scale;
                    }
                    k[o..o + hd].copy_from_slice(&row[d + hh * hd..d + (hh + 1) * hd]);
                    v[o..o + hd].copy_from_slice(&row[2 * d + hh * hd..2 * d + (hh + 1) * hd]);
                }
            }
            let mut sc = vec![0f32; l];
            let mut acc = vec![0f32; hd];
            for hh in 0..heads {
                let (qh, kh, vh) = (&q[hh * l * hd..(hh + 1) * l * hd], &k[hh * l * hd..(hh + 1) * l * hd], &v[hh * l * hd..(hh + 1) * l * hd]);
                let bias = &b.bias[hh * l * l..(hh + 1) * l * l];
                for i in 0..l {
                    let qi = &qh[i * hd..(i + 1) * hd];
                    let bi = &bias[i * l..(i + 1) * l];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..l {
                        let s = if masked && lab[i] != lab[j] { f32::NEG_INFINITY } else { dot(qi, &kh[j * hd..(j + 1) * hd]) + bi[j] };
                        sc[j] = s;
                        mx = mx.max(s);
                    }
                    let mut sum = 0f32;
                    for s in sc.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    acc.fill(0.0);
                    for j in 0..l {
                        let pj = sc[j] * inv;
                        if pj != 0.0 {
                            for (a, vv) in acc.iter_mut().zip(&vh[j * hd..(j + 1) * hd]) {
                                *a += pj * vv;
                            }
                        }
                    }
                    if b.xsa {
                        // Exclusive self-attention: drop the component along the token's own value.
                        let vi = &vh[i * hd..(i + 1) * hd];
                        let nn = dot(vi, vi).max(1e-24);
                        let c = dot(&acc, vi) / nn;
                        for (a, vv) in acc.iter_mut().zip(vi) {
                            *a -= c * vv;
                        }
                    }
                    unsafe { ap.slice(pix[i] * d + hh * hd, hd) }.copy_from_slice(&acc);
                }
            }
        });
        let t_att = t0.elapsed();
        let pr = b.proj.forward(pool, gpu, &att, n);
        add_scaled(pool, x, &pr, &b.attn_scale);
        // ---- FFN branch ----
        let y = layernorm(pool, x, &b.n2.0, &b.n2.1);
        let t_proj = t0.elapsed();
        let f1 = b.fc1.forward(pool, gpu, &y, n);
        let t_fc1 = t0.elapsed();
        let c2 = b.fc1.n; // 2*hidden
        let hid = c2 / 2;
        let mut gated = vec![0f32; n * hid];
        let gp = SyncPtr::new(&mut gated);
        let f1 = &f1;
        // Depthwise 3x3 over fc1 output (tap-major weights -> contiguous
        // channel loops that vectorise), then GELU(first half) * second half.
        pool.parallel_chunks(n, 64, |a, bnd, _| {
            let mut tmp = vec![0f32; c2];
            for p in a..bnd {
                let (py, px) = (p / w, p % w);
                tmp.copy_from_slice(&b.local_b);
                for ky in 0..3 {
                    let iy = py as isize + ky as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let ix = px as isize + kx as isize - 1;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let q = iy as usize * w + ix as usize;
                        let src = &f1[q * c2..(q + 1) * c2];
                        let wt = &b.local_t[(ky * 3 + kx) * c2..(ky * 3 + kx + 1) * c2];
                        for ((t, s), w_) in tmp.iter_mut().zip(src).zip(wt) {
                            *t += s * w_;
                        }
                    }
                }
                let o = unsafe { gp.slice(p * hid, hid) };
                let (g, u) = tmp.split_at(hid);
                for c in 0..hid {
                    let a_ = g[c];
                    o[c] = 0.5 * a_ * (1.0 + libm_erf(a_ * std::f32::consts::FRAC_1_SQRT_2)) * u[c];
                }
            }
        });
        let t_loc = t0.elapsed();
        let f2 = b.fc2.forward(pool, gpu, &gated, n);
        if std::env::var("AIDO_SR_DEBUG").is_ok() {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            eprintln!("blk qkv {:.1} attn {:.1} proj {:.1} fc1 {:.1} local {:.1} fc2 {:.1} ms", ms(t_qkv), ms(t_att - t_qkv), ms(t_proj - t_att), ms(t_fc1 - t_proj), ms(t_loc - t_fc1), ms(t0.elapsed() - t_loc));
        }
        add_scaled(pool, x, &f2, &b.ffn_scale);
    }

    /// Upscales an RGB image (`h × w × 3`, values in `[0,1]`).
    ///
    /// * **Output** — `(H, W, rgb)` with `H = h·scale`.
    ///
    /// # Example
    /// ```no_run
    /// # let m: aido::upscale::LoopSr = unimplemented!();
    /// # let pool = aido::runtime::pool::ThreadPool::new(12, None);
    /// let (h2, w2, out) = m.upscale(&pool, &vec![0.5; 32 * 32 * 3], 32, 32);
    /// assert_eq!((h2, w2), (64, 64));
    /// ```
    pub fn upscale(&self, pool: &ThreadPool, img: &[f32], h: usize, w: usize) -> (usize, usize, Vec<f32>) {
        let win = self.win;
        let (ph, pw) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        // reflect padding (right / bottom)
        let refl = |i: usize, n: usize| if i < n { i } else { (2 * n).saturating_sub(i + 2).min(n - 1) };
        let mut lr = vec![0f32; ph * pw * 3];
        for y in 0..ph {
            for x in 0..pw {
                let s = (refl(y, h) * w + refl(x, w)) * 3;
                lr[(y * pw + x) * 3..(y * pw + x) * 3 + 3].copy_from_slice(&img[s..s + 3]);
            }
        }
        let s = self.scale;
        let base = bicubic(&lr, ph, pw, s);
        let shifted: Vec<f32> = lr.iter().map(|v| v - 0.5).collect();
        let gpu = self.gpu.as_deref();
        let shallow = self.stem.forward(pool, gpu, &shifted, ph, pw);
        let mut state = shallow.clone();
        let dbg = std::env::var("AIDO_SR_DEBUG").is_ok();
        let rms = |v: &[f32]| (v.iter().map(|x| (x * x) as f64).sum::<f64>() / v.len() as f64).sqrt();
        if dbg { eprintln!("stem rms {:.4}", rms(&state)); }
        for b in &self.pre {
            self.block(pool, b, &mut state, ph, pw);
            if dbg { eprintln!("pre rms {:.4}", rms(&state)); }
        }
        for _ in 0..self.loops {
            for b in &self.body {
                self.block(pool, b, &mut state, ph, pw);
            }
        }
        for b in &self.post {
            self.block(pool, b, &mut state, ph, pw);
        }
        let mut r = self.refine.forward(pool, gpu, &state, ph, pw);
        crate::tensor::ops::add_inplace(&mut r, &shallow);
        let mut h0 = self.head0.forward(pool, gpu, &r, ph, pw);
        crate::diffusion::nn::gelu_erf(&mut h0);
        let h2 = self.head2.forward(pool, gpu, &h0, ph, pw); // 3·s² channels, PixelShuffle order (c, sy, sx)
        if dbg {
            eprintln!("state {:.4} refine {:.4} head0 {:.4} head2 {:.4}", rms(&state), rms(&r), rms(&h0), rms(&h2));
            for c in 0..12 { eprint!("{:.3} ", h2.iter().skip(c).step_by(12).sum::<f32>() / (h2.len() / 12) as f32); }
            eprintln!();
        }
        let (oh, ow) = (h * s, w * s);
        let (bh, bw) = (ph * s, pw * s);
        let mut out = vec![0f32; oh * ow * 3];
        for y in 0..oh {
            for x in 0..ow {
                let (py, px, sy, sx) = (y / s, x / s, y % s, x % s);
                for c in 0..3 {
                    let v = h2[(py * pw + px) * 3 * s * s + c * s * s + sy * s + sx] + base[(y * bw + x) * 3 + c];
                    out[(y * ow + x) * 3 + c] = v;
                }
            }
        }
        let _ = bh;
        (oh, ow, out)
    }
}

/// Dot product with 8 independent accumulators (vectorises to AVX).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// erf for f32 (Abramowitz–Stegun 7.1.26).
fn libm_erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    s * y
}

/// PyTorch-compatible bicubic upsampling (`a = −0.75`, `align_corners =
/// False`, border-clamped) of an HWC RGB image by an integer factor.
///
/// # Example
/// ```
/// let out = aido::upscale::bicubic(&vec![0.5; 4 * 4 * 3], 4, 4, 2);
/// assert!((out[0] - 0.5).abs() < 1e-6);
/// ```
pub fn bicubic(img: &[f32], h: usize, w: usize, s: usize) -> Vec<f32> {
    let cubic = |x: f32| -> [f32; 4] {
        let a = -0.75f32;
        // Keys kernel: |d| <= 1 → w1, 1 < |d| < 2 → w0 (d = distance to the tap).
        let w0 = |d: f32| ((a * d - 5.0 * a) * d + 8.0 * a) * d - 4.0 * a;
        let w1 = |d: f32| ((a + 2.0) * d - (a + 3.0)) * d * d + 1.0;
        [w0(x + 1.0), w1(x), w1(1.0 - x), w0(2.0 - x)]
    };
    let (oh, ow) = (h * s, w * s);
    let mut tmp = vec![0f32; h * ow * 3];
    for x in 0..ow {
        let src = (x as f32 + 0.5) / s as f32 - 0.5;
        let x0 = src.floor();
        let wt = cubic(src - x0);
        for y in 0..h {
            for c in 0..3 {
                let mut acc = 0.0;
                for k in 0..4 {
                    let xi = (x0 as isize - 1 + k as isize).clamp(0, w as isize - 1) as usize;
                    acc += wt[k] * img[(y * w + xi) * 3 + c];
                }
                tmp[(y * ow + x) * 3 + c] = acc;
            }
        }
    }
    let mut out = vec![0f32; oh * ow * 3];
    for y in 0..oh {
        let src = (y as f32 + 0.5) / s as f32 - 0.5;
        let y0 = src.floor();
        let wt = cubic(src - y0);
        for x in 0..ow {
            for c in 0..3 {
                let mut acc = 0.0;
                for k in 0..4 {
                    let yi = (y0 as isize - 1 + k as isize).clamp(0, h as isize - 1) as usize;
                    acc += wt[k] * tmp[(yi * ow + x) * 3 + c];
                }
                out[(y * ow + x) * 3 + c] = acc;
            }
        }
    }
    out
}
