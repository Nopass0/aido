//! Scene-text detection: pixel-exact text **masks** and **boxes** for
//! Russian, English and Japanese (kana + kanji) text on real photographs.
//!
//! ## Data
//! [`TextSynth`] crops real photos, renders random strings with system
//! TrueType fonts (Latin/Cyrillic: Arial, Georgia, Segoe UI, Consolas;
//! Japanese: Yu Gothic, MS Gothic) in random size / colour / position, and
//! produces the exact ground truth: a binary mask (glyph coverage ≥ 50 %)
//! and one axis-aligned box per string. Samples can be exported to disk in
//! PNG / JPEG / BMP / WebP / TIFF with mask PNGs and a JSON-lines label file.
//!
//! ## Model
//! A small U-Net (strided 3×3 convs down to 1/4 resolution, nearest
//! up-sampling with skip concatenation) predicts one text logit per pixel,
//! trained with class-weighted binary cross-entropy. Boxes are recovered
//! from the predicted mask by dilation + connected components.

use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};

use super::autograd::{Graph, Id, Params};
use super::Rng;
use crate::runtime::pool::ThreadPool;
use ab_glyph::{Font, FontVec, PxScale, ScaleFont};

/// Training resolution.
pub const RES: usize = 128;

/// An axis-aligned box `[x0, y0, x1, y1)` in pixels.
pub type BBox = [usize; 4];

/// One generated sample.
pub struct Sample {
    /// RGB image (HWC, `[0,1]`).
    pub img: Vec<f32>,
    /// Text mask (`0/1`).
    pub mask: Vec<f32>,
    /// Boxes of the rendered strings.
    pub boxes: Vec<BBox>,
    /// The strings with their script tag.
    pub texts: Vec<(String, &'static str)>,
}

/// Synthetic text-on-photo generator.
pub struct TextSynth {
    photos: Vec<RgbImage>,
    latin: Vec<FontVec>,
    cjk: Vec<FontVec>,
    en: Vec<String>,
    ru: Vec<String>,
    ja: Vec<String>,
}

const JA: &[&str] = &[
    "東京都", "日本語", "出口", "入口", "営業中", "本日休業", "新幹線", "駅前", "危険", "禁煙", "株式会社", "大阪", "京都",
    "漢字の練習", "学校", "図書館", "病院", "郵便局", "電車", "ラーメン", "寿司屋", "ありがとう", "こんにちは", "東京タワー",
    "注意", "非常口", "改札口", "地下鉄", "温泉", "富士山", "桜", "大売出し", "営業時間", "お知らせ", "ご注文", "新宿駅",
];

fn words(path: &str, keep: impl Fn(char) -> bool) -> Vec<String> {
    let t = std::fs::read_to_string(path).unwrap_or_default();
    let mut v: Vec<String> = t.split_whitespace().map(|w| w.chars().filter(|c| keep(*c)).collect::<String>()).filter(|w| w.chars().count() >= 3 && w.chars().count() <= 12).collect();
    v.sort();
    v.dedup();
    v
}

impl TextSynth {
    /// Loads photos (every image file in `photo_dir`), fonts and word lists.
    ///
    /// # Example
    /// ```no_run
    /// let s = aido::sdk::textdet::TextSynth::new("data/photos", 0..50).unwrap();
    /// ```
    pub fn new(photo_dir: &str, range: std::ops::Range<usize>) -> Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(photo_dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        files.sort();
        let photos: Vec<RgbImage> = files[range.start.min(files.len())..range.end.min(files.len())]
            .iter()
            .filter_map(|p| image::open(p).ok().map(|i| i.to_rgb8()))
            .collect();
        anyhow::ensure!(!photos.is_empty(), "no photos found in {photo_dir}");
        let font = |name: &str| -> Option<FontVec> {
            let b = std::fs::read(format!("C:/Windows/Fonts/{name}")).ok()?;
            FontVec::try_from_vec_and_index(b, 0).ok()
        };
        let latin: Vec<FontVec> = ["arial.ttf", "georgia.ttf", "segoeui.ttf", "consola.ttf", "arialbd.ttf"].iter().filter_map(|n| font(n)).collect();
        let cjk: Vec<FontVec> = ["YuGothM.ttc", "msgothic.ttc", "YuGothB.ttc"].iter().filter_map(|n| font(n)).collect();
        anyhow::ensure!(!latin.is_empty() && !cjk.is_empty(), "system fonts not found");
        Ok(TextSynth {
            photos,
            latin,
            cjk,
            en: words("data/alice_en.txt", |c| c.is_ascii_alphanumeric()),
            ru: words("data/ru_wiki.txt", |c| ('а'..='я').contains(&c.to_ascii_lowercase()) || ('А'..='Я').contains(&c) || c == 'ё'),
            ja: JA.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Generates one `res × res` sample.
    ///
    /// # Example
    /// ```no_run
    /// let s = aido::sdk::textdet::TextSynth::new("data/photos", 0..50).unwrap();
    /// let mut r = aido::sdk::Rng(1);
    /// let smp = s.sample(&mut r, 128);
    /// assert!(!smp.boxes.is_empty());
    /// ```
    pub fn sample(&self, r: &mut Rng, res: usize) -> Sample {
        let ph = &self.photos[(r.uniform() * self.photos.len() as f32) as usize % self.photos.len()];
        let side = ((ph.width().min(ph.height()) as f32) * (0.3 + 0.7 * r.uniform())) as u32;
        let (x0, y0) = ((r.uniform() * (ph.width() - side) as f32) as u32, (r.uniform() * (ph.height() - side) as f32) as u32);
        let crop = image::imageops::crop_imm(ph, x0, y0, side, side).to_image();
        let crop = image::imageops::resize(&crop, res as u32, res as u32, FilterType::Triangle);
        let mut img: Vec<f32> = crop.as_raw().iter().map(|v| *v as f32 / 255.0).collect();
        let mut mask = vec![0f32; res * res];
        let mut boxes = Vec::new();
        let mut texts = Vec::new();
        let n_str = 1 + (r.uniform() * 3.0) as usize;
        for _ in 0..n_str {
            let k = r.uniform();
            let (text, tag, font) = if k < 0.35 {
                (self.en[(r.uniform() * self.en.len() as f32) as usize % self.en.len()].clone(), "en", &self.latin[(r.uniform() * self.latin.len() as f32) as usize % self.latin.len()])
            } else if k < 0.7 {
                (self.ru[(r.uniform() * self.ru.len() as f32) as usize % self.ru.len()].clone(), "ru", &self.latin[(r.uniform() * self.latin.len() as f32) as usize % self.latin.len()])
            } else {
                (self.ja[(r.uniform() * self.ja.len() as f32) as usize % self.ja.len()].clone(), "ja", &self.cjk[(r.uniform() * self.cjk.len() as f32) as usize % self.cjk.len()])
            };
            let size = 11.0 + r.uniform() * 18.0;
            let color = [r.uniform(), r.uniform(), r.uniform()];
            let sf = font.as_scaled(PxScale::from(size));
            let width: f32 = text.chars().map(|c| sf.h_advance(font.glyph_id(c))).sum();
            if width >= res as f32 - 4.0 {
                continue;
            }
            let px = 2.0 + r.uniform() * (res as f32 - 4.0 - width);
            let py = sf.ascent() + 2.0 + r.uniform() * (res as f32 - size - 6.0).max(1.0);
            let mut cov = vec![0f32; res * res];
            let mut x = px;
            for c in text.chars() {
                let gid = font.glyph_id(c);
                let glyph = gid.with_scale_and_position(PxScale::from(size), ab_glyph::point(x, py));
                if let Some(og) = font.outline_glyph(glyph) {
                    let b = og.px_bounds();
                    og.draw(|gx, gy, v| {
                        let (xx, yy) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
                        if xx >= 0 && yy >= 0 && (xx as usize) < res && (yy as usize) < res {
                            let i = yy as usize * res + xx as usize;
                            cov[i] = cov[i].max(v);
                        }
                    });
                }
                x += sf.h_advance(gid);
            }
            let mut bb = [usize::MAX, usize::MAX, 0, 0];
            for y in 0..res {
                for xx in 0..res {
                    let c = cov[y * res + xx];
                    if c > 0.0 {
                        let i = (y * res + xx) * 3;
                        for ch in 0..3 {
                            img[i + ch] = img[i + ch] * (1.0 - c) + color[ch] * c;
                        }
                    }
                    if c >= 0.5 {
                        mask[y * res + xx] = 1.0;
                        bb = [bb[0].min(xx), bb[1].min(y), bb[2].max(xx + 1), bb[3].max(y + 1)];
                    }
                }
            }
            if bb[0] != usize::MAX {
                boxes.push(bb);
                texts.push((text, tag));
            }
        }
        Sample { img, mask, boxes, texts }
    }

    /// Writes `n` samples to `dir`: images in rotating formats
    /// (png/jpg/bmp/webp/tiff), `*_mask.png`, and `labels.jsonl`.
    ///
    /// # Example
    /// ```no_run
    /// let s = aido::sdk::textdet::TextSynth::new("data/photos", 0..50).unwrap();
    /// s.export("data/textdet", 20, 256, 1).unwrap();
    /// ```
    pub fn export(&self, dir: &str, n: usize, res: usize, seed: u64) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut r = Rng(seed | 1);
        let mut labels = String::new();
        let fmts = ["png", "jpg", "bmp", "webp", "tiff"];
        for i in 0..n {
            let s = self.sample(&mut r, res);
            let ext = fmts[i % fmts.len()];
            let img = RgbImage::from_raw(res as u32, res as u32, s.img.iter().map(|v| (v * 255.0).round() as u8).collect()).context("img")?;
            let name = format!("{i:05}.{ext}");
            img.save(format!("{dir}/{name}"))?;
            let m = image::GrayImage::from_raw(res as u32, res as u32, s.mask.iter().map(|v| (v * 255.0) as u8).collect()).context("mask")?;
            m.save(format!("{dir}/{i:05}_mask.png"))?;
            let objs: Vec<serde_json::Value> = s.boxes.iter().zip(&s.texts).map(|(b, (t, tag))| serde_json::json!({"text": t, "lang": tag, "box": b})).collect();
            labels.push_str(&serde_json::json!({"image": name, "mask": format!("{i:05}_mask.png"), "width": res, "height": res, "objects": objs}).to_string());
            labels.push('\n');
        }
        std::fs::write(format!("{dir}/labels.jsonl"), labels)?;
        Ok(())
    }
}

/// The segmentation network.
pub struct TextDetector {
    /// Parameters.
    pub params: Params,
}

const LAYERS: &[(&str, usize, usize)] = &[
    ("c1", 3, 24), ("c2", 24, 48), ("c3", 48, 48), ("c4", 48, 64), ("c5", 64, 64), ("c6", 64, 64),
    ("u1", 64, 48), ("m1", 96, 48), ("u2", 48, 24), ("m2", 48, 24), ("out", 24, 1),
];

impl TextDetector {
    /// Random initialisation.
    ///
    /// # Example
    /// ```
    /// let d = aido::sdk::textdet::TextDetector::new(1);
    /// assert_eq!(d.params.names.len(), 22);
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed | 1);
        let mut p = Params::default();
        for &(n, ci, co) in LAYERS {
            p.add(&format!("{n}.w"), co, 9 * ci, r.normals(9 * ci * co, (2.0 / (9 * ci) as f32).sqrt()), true);
            p.add(&format!("{n}.b"), 1, co, vec![if n == "out" { -2.0 } else { 0.0 }; co], true);
        }
        TextDetector { params: p }
    }

    fn cv(g: &mut Graph, x: Id, n: &str, b: usize, s: usize, st: usize, act: bool) -> Id {
        let w = g.params.find(&format!("{n}.w")).unwrap();
        let bb = g.params.find(&format!("{n}.b")).unwrap();
        let y = g.conv3x3(x, w, Some(bb), b, s, s, st);
        if act { g.relu(y) } else { y }
    }

    /// Builds the network on `b` images of `s × s`; returns per-pixel logits.
    fn net(g: &mut Graph, x: Vec<f32>, b: usize, s: usize) -> Id {
        let xi = g.input(b * s * s, 3, x);
        let c1 = Self::cv(g, xi, "c1", b, s, 1, true);
        let c2 = Self::cv(g, c1, "c2", b, s, 2, true);
        let c3 = Self::cv(g, c2, "c3", b, s / 2, 1, true);
        let c4 = Self::cv(g, c3, "c4", b, s / 2, 2, true);
        let c5 = Self::cv(g, c4, "c5", b, s / 4, 1, true);
        let c6 = Self::cv(g, c5, "c6", b, s / 4, 1, true);
        let c6 = g.add(c6, c4);
        let u1 = g.upsample2x(c6, b, s / 4, s / 4);
        let u1 = Self::cv(g, u1, "u1", b, s / 2, 1, true);
        let m1 = g.concat_cols(u1, c3);
        let m1 = Self::cv(g, m1, "m1", b, s / 2, 1, true);
        let u2 = g.upsample2x(m1, b, s / 2, s / 2);
        let u2 = Self::cv(g, u2, "u2", b, s, 1, true);
        let m2 = g.concat_cols(u2, c1);
        let m2 = Self::cv(g, m2, "m2", b, s, 1, true);
        Self::cv(g, m2, "out", b, s, 1, false)
    }

    /// One training step on a batch of samples; returns the loss.
    ///
    /// # Example
    /// ```no_run
    /// # let synth = aido::sdk::textdet::TextSynth::new("data/photos", 0..50).unwrap();
    /// let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// let mut d = aido::sdk::textdet::TextDetector::new(1);
    /// let mut r = aido::sdk::Rng(1);
    /// let batch: Vec<_> = (0..4).map(|_| synth.sample(&mut r, 128)).collect();
    /// d.train_step(&pool, &batch);
    /// ```
    pub fn train_step(&mut self, pool: &ThreadPool, batch: &[Sample]) -> f32 {
        let s = (batch[0].mask.len() as f64).sqrt() as usize;
        let x: Vec<f32> = batch.iter().flat_map(|b| b.img.iter().map(|v| v * 2.0 - 1.0)).collect();
        let t: Vec<f32> = batch.iter().flat_map(|b| b.mask.clone()).collect();
        let w: Vec<f32> = t.iter().map(|v| if *v > 0.5 { 4.0 } else { 1.0 }).collect();
        let mut p = std::mem::take(&mut self.params);
        let l = {
            let mut g = Graph::new(&mut p, pool);
            let o = Self::net(&mut g, x, batch.len(), s);
            let l = g.bce_logits(o, &t, &w);
            g.backward(l)
        };
        self.params = p;
        l
    }

    /// Text probability map of one RGB image (`s × s`, `[0,1]`).
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut d = aido::sdk::textdet::TextDetector::new(1);
    /// let p = d.predict(&pool, &vec![0.5; 16 * 16 * 3], 16);
    /// assert_eq!(p.len(), 256);
    /// ```
    pub fn predict(&mut self, pool: &ThreadPool, img: &[f32], s: usize) -> Vec<f32> {
        let mut p = std::mem::take(&mut self.params);
        let out = {
            let mut g = Graph::new(&mut p, pool);
            let o = Self::net(&mut g, img.iter().map(|v| v * 2.0 - 1.0).collect(), 1, s);
            g.value(o).iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect()
        };
        self.params = p;
        out
    }
}

/// Boxes from a binary mask: dilate horizontally (to join glyphs of a
/// word) and slightly vertically, then 8-connected components; each box is
/// the extent of the *original* mask pixels inside a component.
///
/// # Example
/// ```
/// let mut m = vec![0u8; 100];
/// for i in [22, 24, 32, 34] { m[i] = 1; } // a 2-line glyph cluster (boxes smaller than 6 px are dropped)
/// let b = aido::sdk::textdet::boxes_from_mask(&m, 10, 10, 2);
/// assert_eq!(b, vec![[2, 2, 5, 4]]);
/// ```
pub fn boxes_from_mask(mask: &[u8], w: usize, h: usize, gap: usize) -> Vec<BBox> {
    let mut dil = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] > 0 {
                for dy in y.saturating_sub(1)..(y + 2).min(h) {
                    for dx in x.saturating_sub(gap)..(x + gap + 1).min(w) {
                        dil[dy * w + dx] = 1;
                    }
                }
            }
        }
    }
    let mut lab = vec![0usize; w * h];
    let mut out = Vec::new();
    let mut next = 1;
    for i in 0..w * h {
        if dil[i] == 0 || lab[i] != 0 {
            continue;
        }
        let mut stack = vec![i];
        lab[i] = next;
        let mut bb = [usize::MAX, usize::MAX, 0, 0];
        let mut any = false;
        while let Some(j) = stack.pop() {
            let (x, y) = (j % w, j / w);
            if mask[j] > 0 {
                bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x + 1), bb[3].max(y + 1)];
                any = true;
            }
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let k = ny * w + nx;
                    if dil[k] > 0 && lab[k] == 0 {
                        lab[k] = next;
                        stack.push(k);
                    }
                }
            }
        }
        if any && (bb[2] - bb[0]) * (bb[3] - bb[1]) >= 6 {
            out.push(bb);
        }
        next += 1;
    }
    out
}

/// Intersection over union of two boxes.
///
/// # Example
/// ```
/// assert_eq!(aido::sdk::textdet::iou(&[0, 0, 2, 2], &[1, 0, 3, 2]), 1.0 / 3.0);
/// ```
pub fn iou(a: &BBox, b: &BBox) -> f64 {
    let ix = (a[2].min(b[2]) as i64 - a[0].max(b[0]) as i64).max(0);
    let iy = (a[3].min(b[3]) as i64 - a[1].max(b[1]) as i64).max(0);
    let inter = (ix * iy) as f64;
    let ar = |x: &BBox| ((x[2] - x[0]) * (x[3] - x[1])) as f64;
    inter / (ar(a) + ar(b) - inter).max(1e-9)
}

/// Evaluation result.
#[derive(Debug, Clone, Default)]
pub struct DetScore {
    /// Pixel IoU of the text masks.
    pub mask_iou: f64,
    /// Box precision at IoU ≥ 0.5.
    pub precision: f64,
    /// Box recall at IoU ≥ 0.5.
    pub recall: f64,
    /// Mean IoU of matched boxes.
    pub box_iou: f64,
}

/// Trains a detector on freshly generated samples and evaluates it on
/// samples from held-out photos.
///
/// # Example
/// ```no_run
/// let pool = aido::runtime::pool::ThreadPool::new(12, None);
/// let (d, score) = aido::sdk::textdet::train(&pool, "data/photos", 1000, 8, &mut |s, l| println!("{s} {l}")).unwrap();
/// ```
pub fn train(pool: &ThreadPool, photos: &str, steps: usize, batch: usize, progress: &mut dyn FnMut(usize, f32)) -> Result<(TextDetector, DetScore)> {
    let train_synth = TextSynth::new(photos, 0..50)?;
    let test_synth = TextSynth::new(photos, 50..1000)?;
    let mut d = TextDetector::new(5);
    let mut opt = crate::sdk::optimizer(&d.params, 2e-3, steps);
    opt.wd = 0.0;
    let mut r = Rng(31);
    let mut ema = 0.0;
    for s in 0..steps {
        let b: Vec<Sample> = (0..batch).map(|_| train_synth.sample(&mut r, RES)).collect();
        let l = d.train_step(pool, &b);
        opt.update(&mut d.params);
        ema = if ema == 0.0 { l } else { ema * 0.95 + l * 0.05 };
        if (s + 1) % (steps / 10).max(1) == 0 {
            progress(s + 1, ema);
        }
    }
    let score = evaluate(&mut d, pool, &test_synth, 100, 777);
    Ok((d, score))
}

/// Evaluates on `n` generated samples.
///
/// # Example
/// ```no_run
/// # let pool = aido::runtime::pool::ThreadPool::new(4, None);
/// # let synth = aido::sdk::textdet::TextSynth::new("data/photos", 50..100).unwrap();
/// let mut d = aido::sdk::textdet::TextDetector::new(1);
/// let s = aido::sdk::textdet::evaluate(&mut d, &pool, &synth, 10, 1);
/// ```
pub fn evaluate(d: &mut TextDetector, pool: &ThreadPool, synth: &TextSynth, n: usize, seed: u64) -> DetScore {
    let mut r = Rng(seed | 1);
    let (mut inter, mut uni) = (0f64, 0f64);
    let (mut tp, mut npred, mut ngt, mut iou_sum) = (0usize, 0usize, 0usize, 0f64);
    for _ in 0..n {
        let s = synth.sample(&mut r, RES);
        let p = d.predict(pool, &s.img, RES);
        let pm: Vec<u8> = p.iter().map(|v| (*v > 0.5) as u8).collect();
        for (a, b) in pm.iter().zip(&s.mask) {
            let (a, b) = (*a > 0, *b > 0.5);
            inter += (a && b) as u8 as f64;
            uni += (a || b) as u8 as f64;
        }
        let pb = boxes_from_mask(&pm, RES, RES, 3);
        npred += pb.len();
        ngt += s.boxes.len();
        let mut used = vec![false; pb.len()];
        for g in &s.boxes {
            let best = pb.iter().enumerate().filter(|(i, _)| !used[*i]).map(|(i, b)| (iou(g, b), i)).fold((0.0, usize::MAX), |a, b| if b.0 > a.0 { b } else { a });
            if best.0 >= 0.5 {
                used[best.1] = true;
                tp += 1;
                iou_sum += best.0;
            }
        }
    }
    DetScore { mask_iou: inter / uni.max(1.0), precision: tp as f64 / npred.max(1) as f64, recall: tp as f64 / ngt.max(1) as f64, box_iou: iou_sum / tp.max(1) as f64 }
}
