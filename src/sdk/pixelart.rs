//! Class-conditional pixel-art diffusion model.
//!
//! Pixel art lives on a coarse grid: a 256×256 pixel-art image is a 32×32
//! sprite where every "art pixel" is an 8×8 block. The model therefore
//! learns the 32×32 grid (64× fewer pixels than 256² — the reason it trains
//! in minutes on a CPU) and the output is upscaled with nearest-neighbour,
//! which is exactly how pixel art is displayed.
//!
//! * Dataset: procedurally generated sprites in 4 classes (creature, potion,
//!   tree, house) with random palettes and backgrounds ([`sprite`]).
//! * Denoiser: small U-Net (3×3 convs, stride-2 down, nearest up, skip
//!   concatenation, per-pixel RMSNorm, SiLU) conditioned on the timestep and
//!   the class through a per-image embedding.
//! * Sampler: DDIM (deterministic, 50 steps) with x̂₀ clipping.

use super::autograd::{Graph, Id, Params};
use super::Rng;
use crate::runtime::pool::ThreadPool;

/// Grid side.
pub const SIDE: usize = 32;
/// Number of classes.
pub const CLASSES: usize = 4;
/// Class names.
pub const NAMES: [&str; CLASSES] = ["creature", "potion", "tree", "house"];

fn pal(r: &mut Rng) -> [[f32; 3]; 4] {
    let h = r.uniform();
    let hsv = |h: f32, s: f32, v: f32| -> [f32; 3] {
        let i = (h * 6.0).floor();
        let f = h * 6.0 - i;
        let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
        match i as i32 % 6 {
            0 => [v, t, p],
            1 => [q, v, p],
            2 => [p, v, t],
            3 => [p, q, v],
            4 => [t, p, v],
            _ => [v, p, q],
        }
    };
    [hsv(h, 0.8, 0.35), hsv(h, 0.75, 0.65), hsv(h, 0.55, 0.95), hsv((h + 0.5) % 1.0, 0.7, 0.9)]
}

/// Generates one 32×32 RGB sprite of `class` (values in [-1, 1], HWC).
///
/// # Example
/// ```
/// let mut r = aido::sdk::Rng(3);
/// let img = aido::sdk::pixelart::sprite(0, &mut r);
/// assert_eq!(img.len(), 32 * 32 * 3);
/// ```
pub fn sprite(class: usize, r: &mut Rng) -> Vec<f32> {
    let s = SIDE as i32;
    let p = pal(r);
    let bg = { let v = 0.08 + r.uniform() * 0.15; [v * 0.8, v * 0.9, v + 0.1] };
    let mut img = vec![bg; SIDE * SIDE];
    let set = |x: i32, y: i32, c: [f32; 3], img: &mut Vec<[f32; 3]>| {
        if x >= 0 && y >= 0 && x < s && y < s {
            img[(y * s + x) as usize] = c;
        }
    };
    match class {
        0 => {
            // symmetric creature: random half mask mirrored, outlined
            let mut m = [[false; 12]; 16];
            for y in 0..16 {
                for x in 0..12 {
                    let d = ((x as f32 - 11.0).powi(2) / 100.0 + (y as f32 - 8.0).powi(2) / 64.0).sqrt();
                    m[y][x] = r.uniform() > d * 0.9;
                }
            }
            for y in 0..16 {
                for x in 0..12 {
                    if m[y][x] {
                        let c = if r.uniform() < 0.2 { p[2] } else { p[1] };
                        set(4 + x as i32, 8 + y as i32, c, &mut img);
                        set(27 - x as i32, 8 + y as i32, c, &mut img);
                    }
                }
            }
            let snapshot = img.clone();
            for y in 0..s {
                for x in 0..s {
                    let inside = |xx: i32, yy: i32| xx >= 0 && yy >= 0 && xx < s && yy < s && snapshot[(yy * s + xx) as usize] != bg;
                    if !inside(x, y) && (inside(x + 1, y) || inside(x - 1, y) || inside(x, y + 1) || inside(x, y - 1)) {
                        set(x, y, p[0], &mut img);
                    }
                }
            }
            let ey = 12 + (r.uniform() * 4.0) as i32;
            for (ex, c) in [(12, [1.0, 1.0, 1.0]), (19, [1.0, 1.0, 1.0])] {
                set(ex, ey, c, &mut img);
                set(ex + 1, ey, [0.05, 0.05, 0.05], &mut img);
            }
        }
        1 => {
            // potion: round flask + neck + cork, liquid level
            let rad = 7.0 + r.uniform() * 3.0;
            let level = 18.0 + r.uniform() * 6.0;
            for y in 0..s {
                for x in 0..s {
                    let d = ((x as f32 - 15.5).powi(2) + (y as f32 - 20.0).powi(2)).sqrt();
                    if d < rad {
                        set(x, y, if (y as f32) > level { p[1] } else { [0.75, 0.85, 0.9] }, &mut img);
                    } else if d < rad + 1.0 {
                        set(x, y, p[0], &mut img);
                    }
                }
            }
            for y in (20 - rad as i32 - 6)..(20 - rad as i32 + 1) {
                for x in 13..19 {
                    set(x, y, if x == 13 || x == 18 { p[0] } else { [0.75, 0.85, 0.9] }, &mut img);
                }
            }
            for x in 13..19 {
                set(x, 20 - rad as i32 - 7, [0.55, 0.35, 0.15], &mut img);
            }
            set(13 + (r.uniform() * 4.0) as i32, 22, p[2], &mut img);
        }
        2 => {
            // tree: trunk + 2-4 foliage blobs
            let tx = 13 + (r.uniform() * 4.0) as i32;
            for y in 18..30 {
                for x in tx..tx + 3 {
                    set(x, y, [0.4, 0.25, 0.1], &mut img);
                }
            }
            let green = [p[0][0] * 0.3, 0.35 + p[1][1] * 0.4, p[0][2] * 0.3];
            let light = [green[0] + 0.2, green[1] + 0.25, green[2] + 0.1];
            for _ in 0..2 + (r.uniform() * 3.0) as usize {
                let (cx, cy, rr) = (10.0 + r.uniform() * 12.0, 8.0 + r.uniform() * 8.0, 4.0 + r.uniform() * 4.0);
                for y in 0..s {
                    for x in 0..s {
                        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                        if d < rr {
                            set(x, y, if d < rr * 0.5 && x < cx as i32 { light } else { green }, &mut img);
                        }
                    }
                }
            }
            for x in 0..s {
                set(x, 30, [0.2, 0.45, 0.15], &mut img);
                set(x, 31, [0.2, 0.45, 0.15], &mut img);
            }
        }
        _ => {
            // house: walls, roof triangle, door, window
            let (x0, x1, y0) = (6 + (r.uniform() * 3.0) as i32, 24 + (r.uniform() * 3.0) as i32, 16 + (r.uniform() * 3.0) as i32);
            for y in y0..30 {
                for x in x0..x1 {
                    set(x, y, p[1], &mut img);
                }
            }
            let mid = (x0 + x1) / 2;
            for y in (y0 - 10).max(0)..y0 {
                let half = (y - (y0 - 10)) + 1;
                for x in mid - half..=mid + half {
                    set(x, y, p[3], &mut img);
                }
            }
            for y in 24..30 {
                for x in mid - 1..mid + 2 {
                    set(x, y, [0.35, 0.2, 0.1], &mut img);
                }
            }
            for y in y0 + 3..y0 + 6 {
                for x in x0 + 2..x0 + 5 {
                    set(x, y, [0.95, 0.9, 0.5], &mut img);
                }
            }
            for x in 0..s {
                set(x, 30, [0.25, 0.5, 0.2], &mut img);
                set(x, 31, [0.25, 0.5, 0.2], &mut img);
            }
        }
    }
    img.iter().flat_map(|c| c.iter().map(|v| v.clamp(0.0, 1.0) * 2.0 - 1.0)).collect()
}

/// The denoiser U-Net.
pub struct PixelArtModel {
    /// Parameters.
    pub params: Params,
    /// Base channels.
    pub ch: usize,
    /// Diffusion steps.
    pub steps: usize,
    abar: Vec<f32>,
}

impl PixelArtModel {
    /// Creates the model.
    ///
    /// # Example
    /// ```
    /// let m = aido::sdk::pixelart::PixelArtModel::new(16, 100, 1);
    /// assert!(m.params.n_trainable() > 0);
    /// ```
    pub fn new(ch: usize, steps: usize, seed: u64) -> Self {
        let mut r = Rng(seed | 1);
        let mut p = Params::default();
        let c2 = ch * 2;
        let mut conv = |p: &mut Params, name: &str, cin: usize, cout: usize, zero: bool| {
            let std = if zero { 0.0 } else { (2.0 / (9 * cin) as f32).sqrt() };
            p.add(&format!("{name}.w"), cout, 9 * cin, r.normals(9 * cin * cout, std), true);
            p.add(&format!("{name}.b"), 1, cout, vec![0.0; cout], true);
        };
        conv(&mut p, "in", 3, ch, false);
        conv(&mut p, "r1a", ch, ch, false);
        conv(&mut p, "r1b", ch, ch, true);
        conv(&mut p, "down", ch, c2, false);
        conv(&mut p, "r2a", c2, c2, false);
        conv(&mut p, "r2b", c2, c2, true);
        conv(&mut p, "r3a", c2, c2, false);
        conv(&mut p, "r3b", c2, c2, true);
        conv(&mut p, "up", c2, ch, false);
        conv(&mut p, "merge", 2 * ch, ch, false);
        conv(&mut p, "r4a", ch, ch, false);
        conv(&mut p, "r4b", ch, ch, true);
        conv(&mut p, "out", ch, 3, true);
        let mut r2 = Rng(seed + 7);
        let emb_in = 32 + CLASSES;
        p.add("emb1.weight", ch, emb_in, r2.normals(ch * emb_in, (1.0 / emb_in as f32).sqrt()), true);
        p.add("emb2.weight", c2, emb_in, r2.normals(c2 * emb_in, (1.0 / emb_in as f32).sqrt()), true);
        for (n, c) in [("n1", ch), ("n2", c2), ("n3", c2), ("n4", ch)] {
            p.add(&format!("{n}.g"), 1, c, vec![1.0; c], true);
        }
        let betas: Vec<f32> = (0..steps).map(|i| 1e-4 + (0.02 - 1e-4) * i as f32 / (steps - 1) as f32).collect();
        let mut abar = Vec::new();
        let mut a = 1.0;
        for b in betas {
            a *= 1.0 - b;
            abar.push(a);
        }
        PixelArtModel { params: p, ch, steps, abar }
    }

    fn emb(&self, ts: &[usize], cls: &[usize]) -> Vec<f32> {
        let mut v = Vec::new();
        for (&t, &c) in ts.iter().zip(cls) {
            for i in 0..32 {
                let f = t as f32 * (-(i / 2) as f32 * (1000f32).ln() / 16.0).exp();
                v.push(if i % 2 == 0 { f.sin() } else { f.cos() });
            }
            for k in 0..CLASSES {
                v.push((k == c) as u8 as f32);
            }
        }
        v
    }

    fn conv(g: &mut Graph, x: Id, name: &str, n: usize, s: usize, stride: usize) -> Id {
        let w = g.params.find(&format!("{name}.w")).unwrap();
        let b = g.params.find(&format!("{name}.b")).unwrap();
        g.conv3x3(x, w, Some(b), n, s, s, stride)
    }

    fn res(g: &mut Graph, x: Id, a: &str, b: &str, norm: &str, n: usize, s: usize) -> Id {
        let ng = g.params.find(&format!("{norm}.g")).unwrap();
        let h = g.rmsnorm(x, ng, 1e-5);
        let h = g.silu(h);
        let h = Self::conv(g, h, a, n, s, 1);
        let h = g.silu(h);
        let h = Self::conv(g, h, b, n, s, 1);
        g.add(x, h)
    }

    fn eps(&self, g: &mut Graph, x: &[f32], ts: &[usize], cls: &[usize]) -> Id {
        let n = ts.len();
        let s = SIDE;
        let e = g.input(n, 32 + CLASSES, self.emb(ts, cls));
        let e1w = g.params.find("emb1.weight").unwrap();
        let e2w = g.params.find("emb2.weight").unwrap();
        let e1 = g.linear(e, e1w, None, None);
        let e2 = g.linear(e, e2w, None, None);
        let xi = g.input(n * s * s, 3, x.to_vec());
        let h = Self::conv(g, xi, "in", n, s, 1);
        let h = g.add_per_image(h, e1, n);
        let h1 = Self::res(g, h, "r1a", "r1b", "n1", n, s);
        let d = Self::conv(g, h1, "down", n, s, 2);
        let d = g.add_per_image(d, e2, n);
        let d = Self::res(g, d, "r2a", "r2b", "n2", n, s / 2);
        let d = Self::res(g, d, "r3a", "r3b", "n3", n, s / 2);
        let u = g.upsample2x(d, n, s / 2, s / 2);
        let u = Self::conv(g, u, "up", n, s, 1);
        let m = g.concat_cols(u, h1);
        let m = Self::conv(g, m, "merge", n, s, 1);
        let m = g.silu(m);
        let m = Self::res(g, m, "r4a", "r4b", "n4", n, s);
        let m = g.silu(m);
        Self::conv(g, m, "out", n, s, 1)
    }

    /// One training step on clean sprites (`n × 32·32·3`) with classes.
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut m = aido::sdk::pixelart::PixelArtModel::new(8, 50, 1);
    /// let mut r = aido::sdk::Rng(1);
    /// let x = aido::sdk::pixelart::sprite(1, &mut r);
    /// assert!(m.train_step(&pool, &x, &[1], &mut r) > 0.0);
    /// ```
    pub fn train_step(&mut self, pool: &ThreadPool, x0: &[f32], cls: &[usize], rng: &mut Rng) -> f32 {
        let n = cls.len();
        let d = SIDE * SIDE * 3;
        let ts: Vec<usize> = (0..n).map(|_| (rng.uniform() * self.steps as f32) as usize % self.steps).collect();
        let noise = rng.normals(x0.len(), 1.0);
        let xt: Vec<f32> = (0..x0.len()).map(|i| {
            let ab = self.abar[ts[i / d]];
            ab.sqrt() * x0[i] + (1.0 - ab).sqrt() * noise[i]
        }).collect();
        let mut p = std::mem::take(&mut self.params);
        let l = {
            let mut g = Graph::new(&mut p, pool);
            let e = self.eps(&mut g, &xt, &ts, cls);
            let l = g.mse(e, &noise);
            g.backward(l)
        };
        self.params = p;
        l
    }

    /// DDIM sampling (`steps` evenly spaced timesteps), returns `n` sprites.
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut m = aido::sdk::pixelart::PixelArtModel::new(8, 50, 1);
    /// let s = m.sample(&pool, &[0, 3], 5, 1);
    /// assert_eq!(s.len(), 2 * 32 * 32 * 3);
    /// ```
    pub fn sample(&mut self, pool: &ThreadPool, cls: &[usize], steps: usize, seed: u64) -> Vec<f32> {
        let n = cls.len();
        let d = SIDE * SIDE * 3;
        let mut r = Rng(seed | 1);
        let mut x = r.normals(n * d, 1.0);
        let ts: Vec<usize> = (0..steps).map(|i| (self.steps - 1) * (steps - 1 - i) / (steps - 1).max(1)).collect();
        let mut p = std::mem::take(&mut self.params);
        for (k, &t) in ts.iter().enumerate() {
            let e = {
                let mut g = Graph::new(&mut p, pool);
                let id = self.eps(&mut g, &x, &vec![t; n], cls);
                g.value(id).to_vec()
            };
            let ab = self.abar[t];
            let ab_prev = if k + 1 < ts.len() { self.abar[ts[k + 1]] } else { 1.0 };
            for i in 0..x.len() {
                let x0 = ((x[i] - (1.0 - ab).sqrt() * e[i]) / ab.sqrt()).clamp(-1.0, 1.0);
                let eps = (x[i] - ab.sqrt() * x0) / (1.0 - ab).sqrt().max(1e-6);
                x[i] = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps;
            }
        }
        self.params = p;
        x
    }

    /// Trains the model for `steps` steps, calling `progress(step, loss)`.
    ///
    /// # Example
    /// ```no_run
    /// let pool = aido::runtime::pool::ThreadPool::new(12, None);
    /// let mut m = aido::sdk::pixelart::PixelArtModel::new(48, 300, 1);
    /// m.train(&pool, 2000, 24, &mut |s, l| println!("{s} {l}"));
    /// ```
    pub fn train(&mut self, pool: &ThreadPool, steps: usize, batch: usize, progress: &mut dyn FnMut(usize, f32)) {
        let mut opt = crate::sdk::optimizer(&self.params, 1e-3, steps);
        opt.wd = 0.0;
        let mut r = Rng(77);
        let mut ema = 0.0;
        for s in 0..steps {
            let cls: Vec<usize> = (0..batch).map(|_| (r.uniform() * CLASSES as f32) as usize % CLASSES).collect();
            let x: Vec<f32> = cls.iter().flat_map(|&c| sprite(c, &mut r)).collect();
            let l = self.train_step(pool, &x, &cls, &mut r);
            opt.update(&mut self.params);
            ema = if ema == 0.0 { l } else { ema * 0.97 + l * 0.03 };
            if (s + 1) % (steps / 10).max(1) == 0 {
                progress(s + 1, ema);
            }
        }
    }
}

/// Upscales `n` 32×32 sprites (values in [-1,1]) ×`zoom` with nearest
/// neighbour into a grid image (RGB8). Returns `(width, height, rgb)`.
///
/// # Example
/// ```
/// let (w, h, rgb) = aido::sdk::pixelart::grid(&vec![0.0; 32 * 32 * 3], 1, 8);
/// assert_eq!((w, h), (256, 256));
/// assert_eq!(rgb.len(), 256 * 256 * 3);
/// ```
pub fn grid(imgs: &[f32], n: usize, zoom: usize) -> (usize, usize, Vec<u8>) {
    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let (w, h) = (cols * SIDE * zoom, rows * SIDE * zoom);
    let mut rgb = vec![0u8; w * h * 3];
    for k in 0..n {
        let (gx, gy) = (k % cols, k / cols);
        for y in 0..SIDE * zoom {
            for x in 0..SIDE * zoom {
                for c in 0..3 {
                    let v = imgs[k * SIDE * SIDE * 3 + ((y / zoom) * SIDE + x / zoom) * 3 + c];
                    rgb[((gy * SIDE * zoom + y) * w + gx * SIDE * zoom + x) * 3 + c] = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    (w, h, rgb)
}
