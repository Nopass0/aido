//! Media encoders of LFM3-Omni / d1 checkpoints: the vision tower (pictures)
//! and the audio encoder (speech and sounds), each followed by its projector
//! into the language model's embedding space.
//!
//! ### Vision
//! Picture → bicubic resize (pixel budget, sides multiple of 32) → 16×16
//! patches (×2 temporal duplicate) in **merge-major order** (each 2×2 block
//! contiguous) → linear patchifier → 14 layers of gated depthwise 3×3
//! convolution over the patch grid / bidirectional attention with **axial
//! 2-D RoPE** → LayerNorm → 4 patches concatenated → MLP projector → one
//! language token per 32×32 pixels.
//!
//! ### Audio
//! 16 kHz waveform → 128-bin log-mel (25 ms window, 10 ms hop) → 2-D
//! convolutional subsampling ×8 in time → 17-layer LFM encoder (gated short
//! convolutions + local attention) → projector → one token per 80 ms.

use anyhow::{Context, Result};
use image::RgbImage;
use serde_json::Value;

use super::lfm3::{lin_many, Accel, Lin};
use super::model::linear_rows;
use crate::format::safetensors::SafetensorsDir;
use crate::runtime::pool::ThreadPool;
use crate::tensor::ops;
use crate::tensor::{QWeight, SyncPtr};

/// One media item of a request.
pub enum MediaInput {
    /// A picture.
    Image(RgbImage),
    /// Mono waveform at 16 kHz.
    Audio(Vec<f32>),
}

/// Maximum pixels a picture is scaled to before patching (speed/accuracy
/// trade-off; the checkpoint was trained up to 1.6 MP). Env
/// `AIDO_VISION_MAX_PIXELS` overrides.
pub fn max_pixels() -> usize {
    std::env::var("AIDO_VISION_MAX_PIXELS").ok().and_then(|v| v.parse().ok()).unwrap_or(512 * 512)
}

struct VLayer {
    attn: bool,
    op_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: Option<Lin>,
    k: Option<Lin>,
    v: Option<Lin>,
    o: Lin,
    qn: Vec<f32>,
    kn: Vec<f32>,
    in_proj: Option<Lin>,
    spatial: Vec<f32>,
    w1: Lin,
    w2: Lin,
    w3: Lin,
}

/// The vision tower + projector.
pub struct Vision {
    acc: Accel,
    patchifier: QWeight,
    emb_norm: Vec<f32>,
    layers: Vec<VLayer>,
    d: usize,
    heads: usize,
    kv_heads: usize,
    eps: f32,
    theta: f32,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    p1: QWeight,
    p1b: Vec<f32>,
    p2: QWeight,
    p2b: Vec<f32>,
}

fn layernorm_rows(x: &mut [f32], g: &[f32], b: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(g.len()) {
        ops::layernorm_inplace(row, g, Some(b), eps);
    }
}

fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_exact_mut(b.len()) {
        ops::add_inplace(row, b);
    }
}

fn gelu_exact(x: &mut [f32]) {
    crate::diffusion::nn::gelu_erf(x);
}

impl Vision {
    /// Loads `model.vision_tower.*` and `model.multi_modal_projector.*`.
    ///
    /// # Example
    /// ```no_run
    /// # let st = aido::format::safetensors::SafetensorsDir::open("models/d1-omni").unwrap();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// # let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read("models/d1-omni/config.json").unwrap()).unwrap();
    /// let v = aido::llm::d1_media::Vision::load(&st, &pool, &cfg, &aido::llm::lfm3::Accel::default()).unwrap();
    /// ```
    pub fn load(st: &SafetensorsDir, pool: &ThreadPool, cfg: &Value, acc: &Accel) -> Result<Self> {
        let vc = &cfg["vision_config"];
        let q8 = |n: &str| super::lfm3::load_q8(st, pool, n);
        let l8 = |n: &str| -> Result<Lin> { acc.lin(super::lfm3::load_q8(st, pool, n)?) };
        let kinds = vc["layer_types"].as_array().context("vision layer_types")?;
        let mut layers = Vec::new();
        for (i, k) in kinds.iter().enumerate() {
            let p = |s: &str| format!("model.vision_tower.layers.{i}.{s}");
            let attn = k.as_str() == Some("full_attention");
            layers.push(VLayer {
                attn,
                op_norm: st.f32(&p("operator_norm.weight"))?,
                ffn_norm: st.f32(&p("ffn_norm.weight"))?,
                q: if attn { Some(l8(&p("self_attn.q_proj.weight"))?) } else { None },
                k: if attn { Some(l8(&p("self_attn.k_proj.weight"))?) } else { None },
                v: if attn { Some(l8(&p("self_attn.v_proj.weight"))?) } else { None },
                o: if attn { l8(&p("self_attn.out_proj.weight"))? } else { l8(&p("conv.out_proj.weight"))? },
                qn: if attn { st.f32(&p("self_attn.q_layernorm.weight"))? } else { vec![] },
                kn: if attn { st.f32(&p("self_attn.k_layernorm.weight"))? } else { vec![] },
                in_proj: if attn { None } else { Some(l8(&p("conv.in_proj.weight"))?) },
                spatial: if attn { vec![] } else { st.f32(&p("conv.spatial_conv.weight"))? },
                w1: l8(&p("feed_forward.w1.weight"))?,
                w2: l8(&p("feed_forward.w2.weight"))?,
                w3: l8(&p("feed_forward.w3.weight"))?,
            });
        }
        let pp = "model.multi_modal_projector";
        Ok(Vision {
            acc: acc.clone(),
            patchifier: q8("model.vision_tower.patchifier.weight")?,
            emb_norm: st.f32("model.vision_tower.embedding_norm.weight")?,
            layers,
            d: vc["hidden_size"].as_u64().unwrap_or(1024) as usize,
            heads: vc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            kv_heads: vc["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            eps: vc["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            theta: vc["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e6) as f32,
            ln_g: st.f32(&format!("{pp}.layer_norm.weight"))?,
            ln_b: st.f32(&format!("{pp}.layer_norm.bias"))?,
            p1: q8(&format!("{pp}.linear_1.weight"))?,
            p1b: st.f32(&format!("{pp}.linear_1.bias"))?,
            p2: q8(&format!("{pp}.linear_2.weight"))?,
            p2b: st.f32(&format!("{pp}.linear_2.bias"))?,
        })
    }

    /// Resizes a picture as the processor does (budget, sides multiple of 32).
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::llm::d1_media::Vision::target_size(1000, 500, 512 * 512), (736, 352));
    /// ```
    pub fn target_size(w: u32, h: u32, max_px: usize) -> (u32, u32) {
        let (mut w, mut h) = (w as f64, h as f64);
        if w * h > max_px as f64 {
            let s = (max_px as f64 / (w * h)).sqrt();
            w = (w * s).floor().max(1.0);
            h = (h * s).floor().max(1.0);
        }
        let f = 32.0;
        let (mut hb, mut wb) = ((h / f).round() * f, (w / f).round() * f);
        let (minp, maxp) = (65536.0, 2097152.0);
        if hb * wb > maxp {
            let beta = (h * w / maxp).sqrt();
            hb = ((h / beta / f).floor() * f).max(f);
            wb = ((w / beta / f).floor() * f).max(f);
        } else if hb * wb < minp {
            let beta = (minp / (h * w)).sqrt();
            hb = (h * beta / f).ceil() * f;
            wb = (w * beta / f).ceil() * f;
        }
        (wb.max(f) as u32, hb.max(f) as u32)
    }

    /// Encodes a picture: returns `(embeddings n×2048, n, markup)`.
    ///
    /// # Example
    /// ```no_run
    /// # let v: aido::llm::d1_media::Vision = unimplemented!();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// let img = image::RgbImage::new(256, 256);
    /// let (emb, n, markup) = v.encode(&pool, &img).unwrap();
    /// ```
    pub fn encode(&self, pool: &ThreadPool, img: &RgbImage) -> Result<(Vec<f32>, usize, String)> {
        let (tw, th) = Self::target_size(img.width(), img.height(), max_pixels());
        let img = image::imageops::resize(img, tw, th, image::imageops::FilterType::CatmullRom);
        let (gh, gw) = (th as usize / 16, tw as usize / 16);
        let n = gh * gw;
        // Patches in merge-major order, vector = (channel, temporal×2, py, px).
        let mut patches = vec![0f32; n * 1536];
        let mut pos = vec![(0usize, 0usize); n];
        let mut idx = 0;
        for bh in 0..gh / 2 {
            for bw in 0..gw / 2 {
                for mh in 0..2 {
                    for mw in 0..2 {
                        let (ph, pw) = (bh * 2 + mh, bw * 2 + mw);
                        pos[idx] = (ph, pw);
                        let v = &mut patches[idx * 1536..(idx + 1) * 1536];
                        for c in 0..3 {
                            for tt in 0..2 {
                                for py in 0..16 {
                                    for px in 0..16 {
                                        let p = img.get_pixel((pw * 16 + px) as u32, (ph * 16 + py) as u32);
                                        v[((c * 2 + tt) * 16 + py) * 16 + px] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
                                    }
                                }
                            }
                        }
                        idx += 1;
                    }
                }
            }
        }
        let d = self.d;
        let mut x = vec![0f32; n * d];
        linear_rows(pool, &self.patchifier, &patches, n, &mut x);
        for r in 0..n {
            ops::rmsnorm_inplace(&mut x[r * d..(r + 1) * d], &self.emb_norm, self.eps);
        }
        // spatial index of every token for the convolution
        let mut at = vec![0usize; n];
        for (i, (ph, pw)) in pos.iter().enumerate() {
            at[ph * gw + pw] = i;
        }
        let mut h = vec![0f32; n * d];
        let mut o = vec![0f32; n * d];
        for l in &self.layers {
            for r in 0..n {
                ops::rmsnorm(&mut h[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], &l.op_norm, self.eps);
            }
            if l.attn {
                self.attention(pool, l, &h, n, &pos, &mut o);
            } else {
                let mut bcx = vec![0f32; n * 3 * d];
                lin_many(pool, &self.acc, &mut [(l.in_proj.as_ref().unwrap(), &mut bcx)], &h, n);
                let gated: Vec<f32> = (0..n * d).map(|k| bcx[(k / d) * 3 * d + k % d] * bcx[(k / d) * 3 * d + 2 * d + k % d]).collect();
                let mut y = vec![0f32; n * d];
                let yp = SyncPtr::new(&mut y);
                let (gated, at, pos, sp, bcx) = (&gated, &at, &pos, &l.spatial, &bcx);
                pool.parallel_chunks(n, 32, |a, b, _| {
                    for i in a..b {
                        let (ph, pw) = pos[i];
                        let out = unsafe { yp.slice(i * d, d) };
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let (yy, xx) = (ph as isize + ky as isize - 1, pw as isize + kx as isize - 1);
                                if yy < 0 || xx < 0 || yy >= gh as isize || xx >= gw as isize {
                                    continue;
                                }
                                let j = at[yy as usize * gw + xx as usize];
                                for c in 0..d {
                                    out[c] += sp[c * 9 + ky * 3 + kx] * gated[j * d + c];
                                }
                            }
                        }
                        for c in 0..d {
                            out[c] *= bcx[i * 3 * d + d + c];
                        }
                    }
                });
                lin_many(pool, &self.acc, &mut [(&l.o, &mut o)], &y, n);
            }
            ops::add_inplace(&mut x, &o);
            for r in 0..n {
                ops::rmsnorm(&mut h[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], &l.ffn_norm, self.eps);
            }
            let ff = l.w1.w.rows;
            let mut g = vec![0f32; n * ff];
            let mut u = vec![0f32; n * ff];
            lin_many(pool, &self.acc, &mut [(&l.w1, &mut g), (&l.w3, &mut u)], &h, n);
            ops::silu_mul(&mut g, &u);
            lin_many(pool, &self.acc, &mut [(&l.w2, &mut o)], &g, n);
            ops::add_inplace(&mut x, &o);
        }
        // projector
        layernorm_rows(&mut x, &self.ln_g, &self.ln_b, 1e-6);
        let m = n / 4;
        let mut p1 = vec![0f32; m * self.p1.rows];
        linear_rows(pool, &self.p1, &x, m, &mut p1);
        add_bias(&mut p1, &self.p1b);
        gelu_exact(&mut p1);
        let mut out = vec![0f32; m * self.p2.rows];
        linear_rows(pool, &self.p2, &p1, m, &mut out);
        add_bias(&mut out, &self.p2b);
        let markup = format!("<|vision_start|>{}<|vision_end|>", "<|image|>".repeat(m));
        Ok((out, m, markup))
    }

    /// Bidirectional attention with axial RoPE (first half of each half-head
    /// rotates with the row, second with the column).
    fn attention(&self, pool: &ThreadPool, l: &VLayer, h: &[f32], n: usize, pos: &[(usize, usize)], out: &mut [f32]) {
        let (heads, kvh) = (self.heads, self.kv_heads);
        let hd = self.d / heads;
        let (qd, kvd) = (heads * hd, kvh * hd);
        let mut q = vec![0f32; n * qd];
        let mut k = vec![0f32; n * kvd];
        let mut v = vec![0f32; n * kvd];
        lin_many(pool, &self.acc, &mut [(l.q.as_ref().unwrap(), &mut q), (l.k.as_ref().unwrap(), &mut k), (l.v.as_ref().unwrap(), &mut v)], h, n);
        let half = hd / 2; // rotate pairs (i, i+half)
        let nf = half / 2; // frequencies per axis
        let inv: Vec<f32> = (0..nf).map(|i| 1.0 / self.theta.powf(2.0 * i as f32 / half as f32)).collect();
        let rot = |x: &mut [f32], ph: usize, pw: usize| {
            for i in 0..half {
                let ang = if i < nf { ph as f32 * inv[i] } else { pw as f32 * inv[i - nf] };
                let (s, c) = ang.sin_cos();
                let (a, b) = (x[i], x[i + half]);
                x[i] = a * c - b * s;
                x[i + half] = b * c + a * s;
            }
        };
        for r in 0..n {
            for x in q[r * qd..(r + 1) * qd].chunks_exact_mut(hd) {
                ops::rmsnorm_inplace(x, &l.qn, self.eps);
                rot(x, pos[r].0, pos[r].1);
            }
            for x in k[r * kvd..(r + 1) * kvd].chunks_exact_mut(hd) {
                ops::rmsnorm_inplace(x, &l.kn, self.eps);
                rot(x, pos[r].0, pos[r].1);
            }
        }
        let rep = heads / kvh;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0f32; n * qd];
        let cp = SyncPtr::new(&mut ctx);
        let (q, k, v) = (&q, &k, &v);
        pool.parallel_for(heads * n.div_ceil(16), |task, _| {
            let hh = task % heads;
            let r0 = (task / heads) * 16;
            let kh = hh / rep;
            let mut sc = vec![0f32; n];
            for i in r0..(r0 + 16).min(n) {
                let qi = &q[i * qd + hh * hd..i * qd + (hh + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..n {
                    sc[j] = qi.iter().zip(&k[j * kvd + kh * hd..j * kvd + (kh + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() * scale;
                    mx = mx.max(sc[j]);
                }
                let mut s = 0f32;
                for x in sc.iter_mut() {
                    *x = (*x - mx).exp();
                    s += *x;
                }
                let o = unsafe { cp.slice(i * qd + hh * hd, hd) };
                for j in 0..n {
                    let w = sc[j] / s;
                    for (e, vv) in v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd].iter().enumerate() {
                        o[e] += w * vv;
                    }
                }
            }
        });
        lin_many(pool, &self.acc, &mut [(&l.o, out)], &ctx, n);
    }
}

/// The audio encoder + projector.
pub struct Audio {
    conv_in_w: Vec<f32>,
    conv_in_b: Vec<f32>,
    conv_mid_w: Vec<f32>,
    conv_mid_b: Vec<f32>,
    lin_mid: crate::tensor::gemm::PackedB,
    lin_mid_b: Vec<f32>,
    conv_end_w: Vec<f32>,
    conv_end_b: Vec<f32>,
    lin_end: crate::tensor::gemm::PackedB,
    lin_end_b: Vec<f32>,
    lin_out: QWeight,
    lin_out_b: Vec<f32>,
    lfm: super::lfm3::Backbone,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    p1: QWeight,
    p1b: Vec<f32>,
    p2: QWeight,
    p2b: Vec<f32>,
    ch: usize,
}

/// Slaney-scale mel filter bank, `n_freq × n_mels` (as transformers').
fn mel_bank(n_fft: usize, n_mels: usize, sr: f64) -> Vec<f32> {
    let hz2mel = |f: f64| if f < 1000.0 { 3.0 * f / 200.0 } else { 15.0 + (f / 1000.0).ln() / (6.4f64.ln() / 27.0) };
    let mel2hz = |m: f64| if m < 15.0 { 200.0 * m / 3.0 } else { 1000.0 * ((m - 15.0) * 6.4f64.ln() / 27.0).exp() };
    let nf = n_fft / 2 + 1;
    let (m0, m1) = (hz2mel(0.0), hz2mel(sr / 2.0));
    let pts: Vec<f64> = (0..n_mels + 2).map(|i| mel2hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64)).collect();
    let fft: Vec<f64> = (0..nf).map(|i| sr / 2.0 * i as f64 / (nf - 1) as f64).collect();
    let mut out = vec![0f32; nf * n_mels];
    for (k, f) in fft.iter().enumerate() {
        for m in 0..n_mels {
            let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
            let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
            out[k * n_mels + m] = (down.min(up).max(0.0) * 2.0 / (pts[m + 2] - pts[m])) as f32;
        }
    }
    out
}

/// Log-mel spectrogram `(n_mels, frames, [n_mels][frames])` of a 16 kHz
/// clip, as the LFM3 front end computes it (Hann 400, hop 160, no centring,
/// Slaney mels, `(log10 ∨ −12 + 4) / 4`).
///
/// # Example
/// ```
/// let m = aido::llm::d1_media::log_mel(&vec![0.0; 16000]);
/// assert_eq!((m.0, m.1), (128, 99)); // padded to a whole hop
/// ```
pub fn log_mel(wav: &[f32]) -> (usize, usize, Vec<f32>) {
    let (n_fft, hop, nm) = (400usize, 160usize, 128usize);
    let mut x = wav.to_vec();
    let padded = if x.len() < n_fft { n_fft } else { x.len() + (hop - (x.len() - n_fft) % hop) % hop };
    x.resize(padded, 0.0);
    let frames = 1 + (padded - n_fft) / hop;
    let bank = mel_bank(n_fft, nm, 16000.0);
    let nf = n_fft / 2 + 1;
    let tau = 2.0 * std::f64::consts::PI;
    let win: Vec<f64> = (0..n_fft).map(|i| 0.5 - 0.5 * (tau * i as f64 / n_fft as f64).cos()).collect();
    let cosv: Vec<f64> = (0..n_fft).map(|i| (tau * i as f64 / n_fft as f64).cos()).collect();
    let sinv: Vec<f64> = (0..n_fft).map(|i| (tau * i as f64 / n_fft as f64).sin()).collect();
    let mut out = vec![0f32; nm * frames];
    let mut frame = vec![0f64; n_fft];
    let mut power = vec![0f64; nf];
    for t in 0..frames {
        for i in 0..n_fft {
            frame[i] = x[t * hop + i] as f64 * win[i];
        }
        for k in 0..nf {
            let (mut re, mut im) = (0f64, 0f64);
            for i in 0..n_fft {
                let idx = (k * i) % n_fft;
                re += frame[i] * cosv[idx];
                im -= frame[i] * sinv[idx];
            }
            power[k] = re * re + im * im;
        }
        for m in 0..nm {
            let e: f64 = (0..nf).map(|k| bank[k * nm + m] as f64 * power[k]).sum();
            out[m * frames + t] = ((e.max(1e-30).log10().max(-12.0) + 4.0) / 4.0) as f32;
        }
    }
    (nm, frames, out)
}

/// 3×3 stride-2 padding-1 convolution on `[c][h][w]` planes: a full
/// convolution from one input plane (`depthwise = false`) or depthwise.
#[allow(clippy::too_many_arguments)]
fn conv_s2(pool: &ThreadPool, x: &[f32], h: usize, w: usize, wt: &[f32], b: &[f32], cout: usize, depthwise: bool) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (h.div_ceil(2), w.div_ceil(2));
    let mut y = vec![0f32; cout * ho * wo];
    let yp = SyncPtr::new(&mut y);
    pool.parallel_for(cout, |c, _| {
        let src = if depthwise { c } else { 0 };
        let plane = &x[src * h * w..(src + 1) * h * w];
        let k = &wt[c * 9..(c + 1) * 9];
        let out = unsafe { yp.slice(c * ho * wo, ho * wo) };
        for oy in 0..ho {
            for ox in 0..wo {
                let mut acc = b[c];
                for ky in 0..3 {
                    let iy = (oy * 2 + ky) as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let ix = (ox * 2 + kx) as isize - 1;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        acc += k[ky * 3 + kx] * plane[iy as usize * w + ix as usize];
                    }
                }
                out[oy * wo + ox] = acc;
            }
        }
    });
    (y, ho, wo)
}

impl Audio {
    /// Loads the audio encoder (`model.audio_encoder.*`, `model.audio_projector.*`).
    ///
    /// # Example
    /// ```no_run
    /// # let st = aido::format::safetensors::SafetensorsDir::open("models/d1-omni").unwrap();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// # let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read("models/d1-omni/config.json").unwrap()).unwrap();
    /// let a = aido::llm::d1_media::Audio::load(&st, &pool, &cfg, &aido::llm::lfm3::Accel::default()).unwrap();
    /// ```
    pub fn load(st: &SafetensorsDir, pool: &ThreadPool, cfg: &Value, acc: &Accel) -> Result<Self> {
        let ac = &cfg["audio_encoder_config"];
        let tc = &ac["text_config"];
        let p = "model.audio_encoder.downsample";
        let ch = ac["conv_channels"].as_u64().unwrap_or(256) as usize;
        let pw = |n: &str| -> Result<crate::tensor::gemm::PackedB> { Ok(crate::tensor::gemm::PackedB::from_nk(&st.f32(&format!("{p}.{n}.weight"))?, ch, ch)) };
        let kinds: Vec<super::lfm3::Kind> = tc["layer_types"]
            .as_array()
            .context("audio layer types")?
            .iter()
            .map(|k| if k.as_str() == Some("conv") { super::lfm3::Kind::Conv } else { super::lfm3::Kind::Sliding })
            .collect();
        let d = tc["hidden_size"].as_u64().unwrap_or(768) as usize;
        let heads = tc["num_attention_heads"].as_u64().unwrap_or(8) as usize;
        let dims = super::lfm3::Dims {
            d,
            heads,
            kv_heads: tc["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
            hd: d / heads,
            eps: tc["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            window: ac["sliding_window"].as_u64().unwrap_or(32) as usize,
        };
        let theta = tc["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e6) as f32;
        let lfm = super::lfm3::Backbone::load(st, pool, "model.audio_encoder.lfm", &kinds, dims, (theta, 1.0, theta), ("layernorm", "out_proj"), acc)?;
        let q8 = |n: &str| super::lfm3::load_q8(st, pool, n);
        let f = |n: &str| st.f32(n);
        Ok(Audio {
            conv_in_w: f(&format!("{p}.conv_in.weight"))?,
            conv_in_b: f(&format!("{p}.conv_in.bias"))?,
            conv_mid_w: f(&format!("{p}.conv_mid.weight"))?,
            conv_mid_b: f(&format!("{p}.conv_mid.bias"))?,
            lin_mid: pw("lin_mid")?,
            lin_mid_b: f(&format!("{p}.lin_mid.bias"))?,
            conv_end_w: f(&format!("{p}.conv_end.weight"))?,
            conv_end_b: f(&format!("{p}.conv_end.bias"))?,
            lin_end: pw("lin_end")?,
            lin_end_b: f(&format!("{p}.lin_end.bias"))?,
            lin_out: q8(&format!("{p}.lin_out.weight"))?,
            lin_out_b: f(&format!("{p}.lin_out.bias"))?,
            lfm,
            ln_g: f("model.audio_projector.layer_norm.weight")?,
            ln_b: f("model.audio_projector.layer_norm.bias")?,
            p1: q8("model.audio_projector.linear_1.weight")?,
            p1b: f("model.audio_projector.linear_1.bias")?,
            p2: q8("model.audio_projector.linear_2.weight")?,
            p2b: f("model.audio_projector.linear_2.bias")?,
            ch,
        })
    }

    /// Encodes a 16 kHz waveform: `(embeddings n×2048, n, markup)`, one
    /// token per 8 mel frames (80 ms).
    ///
    /// # Example
    /// ```no_run
    /// # let a: aido::llm::d1_media::Audio = unimplemented!();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// let (emb, n, markup) = a.encode(&pool, &vec![0.0; 16000]).unwrap();
    /// ```
    pub fn encode(&self, pool: &ThreadPool, wav: &[f32]) -> Result<(Vec<f32>, usize, String)> {
        let (nm, frames, mel) = log_mel(wav);
        let ch = self.ch;
        let silu = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
        let pointwise = |x: &[f32], hw: usize, pb: &crate::tensor::gemm::PackedB, b: &[f32]| -> Vec<f32> {
            let mut xt = vec![0f32; hw * ch];
            for c in 0..ch {
                for i in 0..hw {
                    xt[i * ch + c] = x[c * hw + i];
                }
            }
            let mut yt = vec![0f32; hw * ch];
            crate::tensor::gemm::sgemm(pool, hw, &xt, ch, pb, Some(b), &mut yt, ch, false);
            let mut y = vec![0f32; ch * hw];
            for c in 0..ch {
                for i in 0..hw {
                    let v = yt[i * ch + c];
                    y[c * hw + i] = v / (1.0 + (-v).exp());
                }
            }
            y
        };
        let (mut a, h1, w1) = conv_s2(pool, &mel, nm, frames, &self.conv_in_w, &self.conv_in_b, ch, false);
        silu(&mut a);
        let (b, h2, w2) = conv_s2(pool, &a, h1, w1, &self.conv_mid_w, &self.conv_mid_b, ch, true);
        let b = pointwise(&b, h2 * w2, &self.lin_mid, &self.lin_mid_b);
        let (c, h3, w3) = conv_s2(pool, &b, h2, w2, &self.conv_end_w, &self.conv_end_b, ch, true);
        let c = pointwise(&c, h3 * w3, &self.lin_end, &self.lin_end_b);
        let n = w3;
        let fdim = ch * h3;
        let mut feat = vec![0f32; n * fdim];
        for cc in 0..ch {
            for m in 0..h3 {
                for t in 0..n {
                    feat[t * fdim + cc * h3 + m] = c[(cc * h3 + m) * w3 + t];
                }
            }
        }
        let d = self.lfm.dims.d;
        let mut x = vec![0f32; n * d];
        linear_rows(pool, &self.lin_out, &feat, n, &mut x);
        add_bias(&mut x, &self.lin_out_b);
        let positions: Vec<usize> = (0..n).collect();
        self.lfm.forward(pool, &mut x, n, &positions, &super::lfm3::Visibility::Causal);
        layernorm_rows(&mut x, &self.ln_g, &self.ln_b, 1e-5);
        let mut p1 = vec![0f32; n * self.p1.rows];
        linear_rows(pool, &self.p1, &x, n, &mut p1);
        add_bias(&mut p1, &self.p1b);
        gelu_exact(&mut p1);
        let mut out = vec![0f32; n * self.p2.rows];
        linear_rows(pool, &self.p2, &p1, n, &mut out);
        add_bias(&mut out, &self.p2b);
        Ok((out, n, format!("<|vision_start|>{}<|vision_end|>", "<|audio|>".repeat(n))))
    }
}

/// Reads a WAV file (8/16/24/32-bit PCM or 32-bit float, any channel count)
/// as mono and resamples it to 16 kHz (linear interpolation).
///
/// # Example
/// ```no_run
/// let wav = aido::llm::d1_media::read_wav("call.wav").unwrap();
/// ```
pub fn read_wav(path: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    anyhow::ensure!(b.len() > 44 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "not a WAV file");
    let (mut fmt, mut ch, mut rate, mut bits) = (1u16, 1u16, 16000u32, 16u16);
    let mut i = 12;
    let mut data: &[u8] = &[];
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into()?) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if id == b"fmt " {
            fmt = u16::from_le_bytes([body[0], body[1]]);
            ch = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            data = body;
        }
        i += 8 + len + (len & 1);
    }
    let bps = (bits / 8) as usize;
    let frames = data.len() / (bps * ch as usize);
    let sample = |k: usize| -> f32 {
        let s = &data[k * bps..(k + 1) * bps];
        match (fmt, bits) {
            (3, 32) => f32::from_le_bytes(s.try_into().unwrap()),
            (_, 8) => (s[0] as f32 - 128.0) / 128.0,
            (_, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
            (_, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8388608.0,
            (_, 32) => i32::from_le_bytes(s.try_into().unwrap()) as f32 / 2147483648.0,
            _ => 0.0,
        }
    };
    let mono: Vec<f32> = (0..frames).map(|f| (0..ch as usize).map(|c| sample(f * ch as usize + c)).sum::<f32>() / ch as f32).collect();
    if rate == 16000 {
        return Ok(mono);
    }
    let n = (mono.len() as f64 * 16000.0 / rate as f64) as usize;
    Ok((0..n).map(|k| {
        let x = k as f64 * rate as f64 / 16000.0;
        let a = x.floor() as usize;
        let f = (x - a as f64) as f32;
        let p = mono[a.min(mono.len() - 1)];
        let q = mono[(a + 1).min(mono.len() - 1)];
        p + (q - p) * f
    }).collect())
}
