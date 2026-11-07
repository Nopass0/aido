//! Text-to-image pipeline (Stable Diffusion 2.x / SD-Turbo layout,
//! Euler-discrete sampler, TAESD decoding).

use std::time::Instant;

use anyhow::{Context, Result};

use super::clip::{ClipText, ClipTokenizer};
use super::nn::T;
use super::taesd::TaesdDecoder;
use super::unet::UNet;
use crate::runtime::pool::ThreadPool;
use crate::runtime::sys::CpuInfo;

/// Compute device for diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU only.
    Cpu,
    /// GPU (falls back to CPU for operations without a GPU kernel).
    Gpu,
    /// Pick automatically.
    Auto,
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;
    /// Parses `cpu|gpu|auto`.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "cpu" => Device::Cpu,
            "gpu" => Device::Gpu,
            "auto" => Device::Auto,
            _ => anyhow::bail!("device must be cpu|gpu|auto"),
        })
    }
}

/// Generation parameters.
#[derive(Debug, Clone)]
pub struct ImageParams {
    /// Prompt.
    pub prompt: String,
    /// Negative prompt (used when `guidance > 1`).
    pub negative: String,
    /// Denoising steps (SD-Turbo: 1–4).
    pub steps: usize,
    /// Width in pixels (multiple of 64).
    pub width: usize,
    /// Height in pixels (multiple of 64).
    pub height: usize,
    /// RNG seed.
    pub seed: u64,
    /// Classifier-free guidance scale (1 = off).
    pub guidance: f32,
}

/// An RGB8 image.
pub struct Image {
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
    /// Interleaved RGB bytes.
    pub rgb: Vec<u8>,
}

impl Image {
    /// Encodes as PNG bytes.
    ///
    /// # Example
    /// ```
    /// let img = aido::diffusion::pipeline::Image { width: 1, height: 1, rgb: vec![255, 0, 0] };
    /// assert!(img.to_png().unwrap().starts_with(&[0x89, b'P', b'N', b'G']));
    /// ```
    pub fn to_png(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, self.width as u32, self.height as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()?.write_image_data(&self.rgb)?;
        }
        Ok(out)
    }

    /// Writes a PNG file.
    ///
    /// # Example
    /// ```no_run
    /// let img = aido::diffusion::pipeline::Image { width: 1, height: 1, rgb: vec![0, 0, 0] };
    /// img.save_png("black.png").unwrap();
    /// ```
    pub fn save_png(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.to_png()?)?;
        Ok(())
    }

    /// Converts an HWC float tensor in `[0,1]` to RGB8.
    ///
    /// # Example
    /// ```
    /// let t = aido::diffusion::nn::T { h: 1, w: 1, c: 3, d: vec![1.0, 0.5, 0.0] };
    /// assert_eq!(aido::diffusion::pipeline::Image::from_tensor(&t).rgb, vec![255, 128, 0]);
    /// ```
    pub fn from_tensor(t: &T) -> Self {
        Image { width: t.w, height: t.h, rgb: t.d.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect() }
    }
}

/// Deterministic standard-normal generator (xorshift + Box–Muller).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.next(), self.next());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

/// The loaded pipeline.
pub struct Pipeline {
    tok: ClipTokenizer,
    text: ClipText,
    unet: UNet,
    vae: TaesdDecoder,
    pool: ThreadPool,
    alphas_cumprod: Vec<f64>,
    last: String,
}

impl Pipeline {
    /// Loads a diffusers-format model directory and a TAESD decoder.
    ///
    /// * **Input** — model dir (with `tokenizer/`, `text_encoder/`, `unet/`),
    ///   TAESD dir (required: the tiny decoder is the fast path), device,
    ///   optional thread count.
    ///
    /// # Example
    /// ```no_run
    /// use aido::diffusion::pipeline::*;
    /// let p = Pipeline::load("models/sd-turbo", Some("models/taesd"), Device::Auto, None).unwrap();
    /// ```
    pub fn load(dir: &str, taesd: Option<&str>, device: Device, threads: Option<usize>) -> Result<Self> {
        if device == Device::Gpu {
            eprintln!("[diffusion] note: convolution GEMMs run on the CPU in this version");
        }
        let cpu = CpuInfo::detect();
        let pool = ThreadPool::new(threads.unwrap_or(cpu.physical), Some(cpu.physical_core_ids()));
        let t = Instant::now();
        let tok = ClipTokenizer::load(&format!("{dir}/tokenizer"))?;
        let text = ClipText::load(&format!("{dir}/text_encoder"))?;
        let unet = UNet::load(&format!("{dir}/unet"))?;
        let vae = TaesdDecoder::load(taesd.context("a TAESD decoder directory is required (--taesd models/taesd)")?)?;
        eprintln!("[diffusion] loaded in {:.1}s", t.elapsed().as_secs_f64());
        let (b0, b1) = (0.00085f64.sqrt(), 0.012f64.sqrt());
        let mut ac = Vec::with_capacity(1000);
        let mut prod = 1.0;
        for i in 0..1000 {
            let b = b0 + (b1 - b0) * i as f64 / 999.0;
            prod *= 1.0 - b * b;
            ac.push(prod);
        }
        Ok(Pipeline { tok, text, unet, vae, pool, alphas_cumprod: ac, last: String::new() })
    }

    /// Timing breakdown of the last generation.
    ///
    /// # Example
    /// ```no_run
    /// # let p: aido::diffusion::pipeline::Pipeline = unimplemented!();
    /// println!("{}", p.timings());
    /// ```
    pub fn timings(&self) -> String {
        self.last.clone()
    }

    /// Generates an image.
    ///
    /// * **Behaviour** — trailing timestep spacing, Euler-discrete updates,
    ///   ε-prediction; with `guidance > 1` the UNet runs twice per step
    ///   (classifier-free guidance with the negative prompt).
    ///
    /// # Example
    /// ```no_run
    /// use aido::diffusion::pipeline::*;
    /// let mut p = Pipeline::load("models/sd-turbo", Some("models/taesd"), Device::Auto, None).unwrap();
    /// let img = p.generate(&ImageParams { prompt: "a red fox".into(), negative: String::new(), steps: 1, width: 512, height: 512, seed: 1, guidance: 1.0 }).unwrap();
    /// img.save_png("fox.png").unwrap();
    /// ```
    pub fn generate(&mut self, p: &ImageParams) -> Result<Image> {
        self.generate_with(p, None)
    }

    /// Like [`Pipeline::generate`], but after every denoising step decodes
    /// the current clean-image estimate `x0 = x - sigma*eps` with the VAE
    /// and passes it to `on_step(step, total, preview)` (live previews /
    /// SSE streaming); the final image is passed as step `total`. With
    /// `None` no extra decoding is done.
    ///
    /// # Example
    /// ```no_run
    /// use aido::diffusion::pipeline::*;
    /// let mut p = Pipeline::load("models/sd-turbo", Some("models/taesd"), Device::Auto, None).unwrap();
    /// let params = ImageParams { prompt: "a fox".into(), negative: String::new(), steps: 4, width: 512, height: 512, seed: 1, guidance: 1.0 };
    /// let img = p.generate_with(&params, Some(&mut |i, n, prev: &Image| println!("step {i}/{n}: {}x{}", prev.width, prev.height))).unwrap();
    /// ```
    pub fn generate_with(&mut self, p: &ImageParams, mut on_step: Option<&mut dyn FnMut(usize, usize, &Image)>) -> Result<Image> {
        anyhow::ensure!(p.width % 64 == 0 && p.height % 64 == 0 && p.width > 0 && p.height > 0, "size must be a multiple of 64");
        let pool = &self.pool;
        let t0 = Instant::now();
        let ids = self.tok.encode(&p.prompt, 77);
        let ctx = self.text.forward(pool, &ids);
        let neg = if p.guidance > 1.0 { Some(self.text.forward(pool, &self.tok.encode(&p.negative, 77))) } else { None };
        let t_text = t0.elapsed().as_secs_f64();
        // Trailing spacing.
        let n = p.steps.max(1);
        let ts: Vec<usize> = (0..n).map(|i| ((1000.0 - i as f64 * 1000.0 / n as f64).round() as usize).saturating_sub(1)).collect();
        let sigma = |t: usize| ((1.0 - self.alphas_cumprod[t]) / self.alphas_cumprod[t]).sqrt();
        let (h, w) = (p.height / 8, p.width / 8);
        let mut rng = Rng(p.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        let s0 = sigma(ts[0]);
        let mut x = T { h, w, c: 4, d: (0..h * w * 4).map(|_| rng.normal() * s0 as f32).collect() };
        let t1 = Instant::now();
        for (i, &t) in ts.iter().enumerate() {
            let s = sigma(t);
            let scale = 1.0 / (s * s + 1.0).sqrt();
            let xin = T { h, w, c: 4, d: x.d.iter().map(|v| v * scale as f32).collect() };
            let mut eps = self.unet.forward(pool, &xin, t as f32, &ctx, 77);
            if let Some(nc) = &neg {
                let eu = self.unet.forward(pool, &xin, t as f32, nc, 77);
                for (e, u) in eps.d.iter_mut().zip(&eu.d) {
                    *e = u + p.guidance * (*e - u);
                }
            }
            if let Some(cb) = on_step.as_mut() {
                if i + 1 < ts.len() {
                    let x0 = T { h, w, c: 4, d: x.d.iter().zip(&eps.d).map(|(xv, e)| xv - e * s as f32).collect() };
                    cb(i + 1, ts.len(), &Image::from_tensor(&self.vae.decode(pool, &x0)));
                }
            }
            let s_next = if i + 1 < ts.len() { sigma(ts[i + 1]) } else { 0.0 };
            let dt = (s_next - s) as f32;
            for (xv, e) in x.d.iter_mut().zip(&eps.d) {
                *xv += e * dt;
            }
        }
        let t_unet = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let rgb = self.vae.decode(pool, &x);
        let t_vae = t2.elapsed().as_secs_f64();
        self.last = format!("text {:.2}s, unet {:.2}s ({} steps), decode {:.2}s", t_text, t_unet, n, t_vae);
        let img = Image::from_tensor(&rgb);
        if let Some(cb) = on_step.as_mut() {
            cb(n, n, &img);
        }
        Ok(img)
    }
}
