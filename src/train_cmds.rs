//! CLI front-ends for the SDK (training, fine-tuning, data packing).
//!
//! Compiled into the `aido` binary (see `main.rs`).

use std::time::Instant;

use anyhow::Result;
use serde_json::json;

use aido::runtime::{pool::ThreadPool, sys::CpuInfo};
use aido::sdk::{self, data, models, AdamW, Rng};
use aido::tensor::quant::GgmlType;

/// Physical-core pool for training.
pub fn pool(threads: Option<usize>) -> ThreadPool {
    let cpu = CpuInfo::detect();
    ThreadPool::new(threads.unwrap_or(cpu.physical), Some(cpu.physical_core_ids()))
}

/// Reads text files into one string.
fn read_all(files: &[String]) -> Result<String> {
    let mut s = String::new();
    for f in files {
        s.push_str(&String::from_utf8_lossy(&std::fs::read(f)?));
        s.push('\n');
    }
    Ok(s)
}

/// `aido data-pack`: text → `.aidata` token stream. With `--tokenizer` the
/// model's tokenizer is used, otherwise raw bytes (vocab 257).
pub fn data_pack(files: &[String], out: &str, tokenizer: Option<&str>) -> Result<()> {
    let text = read_all(files)?;
    let (toks, kind): (Vec<u32>, String) = match tokenizer {
        Some(m) => {
            let st = aido::format::open(m)?;
            let tok = aido::tokenizer::Tokenizer::from_gguf(st.as_ref())?;
            (tok.encode(&text, false), m.to_string())
        }
        None => (text.bytes().map(|b| b as u32).collect(), "bytes".into()),
    };
    data::write_tokens(out, &toks, json!({"tokenizer": kind, "sources": files}))?;
    println!("packed {} tokens ({} bytes of text) → {out}", toks.len(), text.len());
    Ok(())
}

/// Samples `batch` random windows of `ctx + 1` tokens.
fn windows(tokens: &[u32], batch: usize, ctx: usize, rng: &mut Rng) -> Vec<Vec<u32>> {
    (0..batch)
        .map(|_| {
            let s = (rng.uniform() * (tokens.len() - ctx - 1) as f32) as usize;
            tokens[s..s + ctx + 1].to_vec()
        })
        .collect()
}

/// Shared LM training loop with periodic evaluation and checkpoints.
#[allow(clippy::too_many_arguments)]
fn lm_loop(m: &mut models::LmModel, opt: &mut AdamW, pool: &ThreadPool, ds: &data::DataSet, steps: usize, batch: usize, ckpt: &str, eval_every: usize) -> Result<()> {
    let toks = ds.tokens();
    let split = toks.len() * 95 / 100;
    let (train, val) = (&toks[..split], &toks[split..]);
    let ctx = m.cfg.ctx;
    let mut rng = Rng(opt.step as u64 * 7919 + 17);
    let t0 = Instant::now();
    let mut ema = 0.0;
    while opt.step < steps {
        let w = windows(train, batch, ctx, &mut rng);
        let loss = m.train_step(pool, &w);
        let gn = opt.update(&mut m.params);
        ema = if ema == 0.0 { loss } else { ema * 0.95 + loss * 0.05 };
        if opt.step % eval_every == 0 || opt.step == steps {
            let mut vr = Rng(99);
            let vw = windows(val, batch.min(8), ctx, &mut vr);
            let vl = {
                let mut p = std::mem::take(&mut m.params);
                let mut ids = Vec::new();
                let mut tg = Vec::new();
                for w in &vw {
                    ids.extend_from_slice(&w[..ctx]);
                    tg.extend_from_slice(&w[1..]);
                }
                let v = {
                    let mut g = sdk::autograd::Graph::new(&mut p, pool);
                    let lg = m.forward(&mut g, &ids, ctx);
                    let l = g.cross_entropy(lg, &tg);
                    g.value(l)[0]
                };
                m.params = p;
                m.params.zero_grad();
                v
            };
            let el = t0.elapsed().as_secs_f64();
            println!(
                "step {:>5}/{steps} | train loss {ema:.3} | val loss {vl:.3} ({:.2} bits/token) | lr {:.2e} | |g| {gn:.2} | {:.0} tok/s",
                opt.step,
                vl / std::f32::consts::LN_2,
                opt.current_lr(),
                (opt.step * batch * ctx) as f64 / el
            );
            sdk::save_checkpoint(ckpt, &m.params, Some(opt), &serde_json::to_value(&m.cfg)?)?;
        }
    }
    Ok(())
}

/// `aido train-lm`: trains a byte-level Llama-style model from scratch (or
/// resumes from a checkpoint) and exports an engine-runnable `.aido`.
#[allow(clippy::too_many_arguments)]
pub fn train_lm(data_path: &str, out: &str, steps: usize, dim: usize, layers: usize, heads: usize, ctx: usize, batch: usize, lr: f32, resume: Option<&str>, threads: Option<usize>) -> Result<()> {
    let pool = pool(threads);
    let ds = data::DataSet::open(data_path)?;
    let ckpt = format!("{out}.ckpt.aido");
    let (mut m, mut opt) = match resume {
        Some(r) => {
            let (p, o, cfg) = sdk::load_checkpoint(r)?;
            let cfg: models::LmConfig = serde_json::from_value(cfg)?;
            let mut o = o.unwrap_or_else(|| aido::sdk::optimizer(&p, lr, steps));
            o.total = steps;
            println!("resumed from {r} at step {}", o.step);
            (models::LmModel { cfg, params: p, source_meta: None }, o)
        }
        None => {
            let m = models::LmModel::new_bytes(dim, layers, heads, dim * 3, ctx, 42);
            let o = aido::sdk::optimizer(&m.params, lr, steps);
            (m, o)
        }
    };
    println!("model: {:?}\ntrainable parameters: {:.2} M, dataset {} tokens", m.cfg, m.params.n_trainable() as f64 / 1e6, ds.len());
    lm_loop(&mut m, &mut opt, &pool, &ds, steps, batch, &ckpt, (steps / 20).max(1))?;
    let n = m.export(out, GgmlType::F32)?;
    println!("exported engine model → {out} ({:.1} MB); checkpoint → {ckpt}", n as f64 / 1e6);
    Ok(())
}

/// `aido finetune`: imports an existing model (GGUF/.aido), trains LoRA
/// adapters (or all weights with `--lora 0`) and exports a merged `.aido`.
#[allow(clippy::too_many_arguments)]
pub fn finetune(model: &str, data_path: &str, out: &str, steps: usize, lora: usize, ctx: usize, batch: usize, lr: f32, threads: Option<usize>) -> Result<()> {
    let pool = pool(threads);
    let t = Instant::now();
    let mut m = models::LmModel::import(model, lora, ctx)?;
    println!("imported {} in {:.1}s: {} layers, dim {}, trainable {:.2} M of {:.2} M parameters (LoRA rank {lora})",
        model, t.elapsed().as_secs_f64(), m.cfg.layers, m.cfg.dim,
        m.params.n_trainable() as f64 / 1e6, m.params.data.iter().map(|d| d.len()).sum::<usize>() as f64 / 1e6);
    let ds = data::DataSet::open(data_path)?;
    let mut opt = aido::sdk::optimizer(&m.params, lr, steps);
    opt.wd = 0.0;
    lm_loop(&mut m, &mut opt, &pool, &ds, steps, batch, &format!("{out}.ckpt.aido"), (steps / 10).max(1))?;
    let n = m.export(out, if lora > 0 { GgmlType::Q8_0 } else { GgmlType::F32 })?;
    println!("exported fine-tuned model → {out} ({:.1} MB)", n as f64 / 1e6);
    Ok(())
}

/// Hashed byte-bigram features (L2-normalised) of a text chunk.
pub fn text_features(s: &[u8], dim: usize) -> Vec<f32> {
    let mut f = vec![0f32; dim];
    for w in s.windows(2) {
        let h = (w[0] as usize * 131 + w[1] as usize * 7919) % dim;
        f[h] += 1.0;
    }
    for &b in s {
        f[(b as usize * 2654435761usize) % dim] += 0.5;
    }
    let n = f.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    f.iter_mut().for_each(|v| *v /= n);
    f
}

/// Class names of the demo classifier.
pub const CLASSES: [&str; 3] = ["english", "russian", "rust-code"];

/// `aido train-classifier`: builds a 3-class text classifier
/// (English / Russian / Rust code) from local corpora.
pub fn train_classifier(out: &str, steps: usize, threads: Option<usize>) -> Result<()> {
    let pool = pool(threads);
    let dim = 512;
    let mut sources: Vec<(u32, Vec<u8>)> = Vec::new();
    for f in ["data/alice_en.txt", "data/pride_en.txt"] {
        sources.push((0, std::fs::read(f)?));
    }
    for f in ["data/ru_wiki.txt", "data/ru_wiki2.txt"] {
        sources.push((1, std::fs::read(f)?));
    }
    let mut code = Vec::new();
    for e in walk("src")? {
        code.extend(std::fs::read(e)?);
    }
    sources.push((2, code));
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (label, bytes) in &sources {
        for chunk in bytes.chunks(160).take(3000) {
            if chunk.len() == 160 {
                xs.extend(text_features(chunk, dim));
                ys.push(*label);
            }
        }
    }
    let n = ys.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Rng(5);
    for i in (1..n).rev() {
        order.swap(i, (rng.uniform() * (i + 1) as f32) as usize % (i + 1));
    }
    let shuffled_x: Vec<f32> = order.iter().flat_map(|&i| xs[i * dim..(i + 1) * dim].to_vec()).collect();
    let shuffled_y: Vec<u32> = order.iter().map(|&i| ys[i]).collect();
    data::write_tensors("data/textcls.aidata", &shuffled_x, dim, &shuffled_y, json!({"classes": CLASSES, "features": "hashed byte bigrams"}))?;
    let ds = data::DataSet::open("data/textcls.aidata")?;
    let ntest = ds.len() / 10;
    let ntrain = ds.len() - ntest;
    println!("dataset: {} samples ({} train / {} test) → data/textcls.aidata", ds.len(), ntrain, ntest);
    let mut c = models::Classifier::new(&[dim, 128, 3], 7);
    let mut opt = aido::sdk::optimizer(&c.params, 3e-3, steps);
    let t = Instant::now();
    for s in 0..steps {
        let b = 64;
        let idx: Vec<usize> = (0..b).map(|_| (rng.uniform() * ntrain as f32) as usize % ntrain).collect();
        let x: Vec<f32> = idx.iter().flat_map(|&i| ds.features(i).to_vec()).collect();
        let y: Vec<u32> = idx.iter().map(|&i| ds.label(i)).collect();
        let l = c.train_step(&pool, &x, &y);
        opt.update(&mut c.params);
        if (s + 1) % (steps / 5).max(1) == 0 {
            println!("step {:>4} loss {l:.4}", s + 1);
        }
    }
    let x: Vec<f32> = (ntrain..ds.len()).flat_map(|i| ds.features(i).to_vec()).collect();
    let p = c.predict(&pool, &x, ntest);
    let correct = (0..ntest).filter(|&i| {
        let row = &p[i * 3..(i + 1) * 3];
        let pred = (0..3).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap();
        pred as u32 == ds.label(ntrain + i)
    }).count();
    println!("trained in {:.1}s — held-out accuracy {:.2}% ({correct}/{ntest})", t.elapsed().as_secs_f64(), 100.0 * correct as f64 / ntest as f64);
    sdk::save_checkpoint(out, &c.params, None, &json!({"kind": "classifier", "sizes": c.sizes, "classes": CLASSES, "features": "hashed-bigram-512"}))?;
    println!("saved {out}");
    Ok(())
}

/// Recursively lists `.rs` files.
fn walk(dir: &str) -> Result<Vec<std::path::PathBuf>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            v.extend(walk(p.to_str().unwrap())?);
        } else if p.extension().is_some_and(|x| x == "rs") {
            v.push(p);
        }
    }
    Ok(v)
}

/// `aido classify`: runs a trained classifier on text.
pub fn classify(model: &str, text: &str) -> Result<()> {
    let pool = pool(Some(1));
    let (p, _, cfg) = sdk::load_checkpoint(model)?;
    let sizes: Vec<usize> = serde_json::from_value(cfg["sizes"].clone())?;
    let mut c = models::Classifier { sizes, params: p };
    let probs = c.predict(&pool, &text_features(text.as_bytes(), 512), 1);
    for (i, name) in CLASSES.iter().enumerate() {
        println!("{name:>10}: {:.1}%", probs[i] * 100.0);
    }
    Ok(())
}

/// Procedural 16×16 training images: soft rings and discs.
fn shape_image(rng: &mut Rng, side: usize) -> Vec<f32> {
    let (cx, cy) = (4.0 + rng.uniform() * 8.0, 4.0 + rng.uniform() * 8.0);
    let r = 2.5 + rng.uniform() * 3.0;
    let ring = rng.uniform() < 0.5;
    (0..side * side)
        .map(|i| {
            let (x, y) = ((i % side) as f32 + 0.5, (i / side) as f32 + 0.5);
            let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            let v = if ring { (1.0 - (d - r).abs()).clamp(0.0, 1.0) } else { (r - d + 0.5).clamp(0.0, 1.0) };
            v * 2.0 - 1.0
        })
        .collect()
}

/// Saves a grid of `n` square grey images (values in [-1,1]) as PNG, ×`zoom`.
fn save_grid(path: &str, imgs: &[f32], side: usize, n: usize, zoom: usize) -> Result<()> {
    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let (w, h) = (cols * side * zoom, rows * side * zoom);
    let mut rgb = vec![30u8; w * h * 3];
    for k in 0..n {
        let (gx, gy) = (k % cols, k / cols);
        for y in 0..side * zoom {
            for x in 0..side * zoom {
                let v = imgs[k * side * side + (y / zoom) * side + x / zoom];
                let b = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                let o = ((gy * side * zoom + y) * w + gx * side * zoom + x) * 3;
                rgb[o..o + 3].copy_from_slice(&[b, b, b]);
            }
        }
    }
    aido::diffusion::pipeline::Image { width: w, height: h, rgb }.save_png(path)
}

/// `aido train-image`: trains the tiny DDPM on procedural shapes and saves
/// a sample grid next to the checkpoint.
pub fn train_image(out: &str, steps: usize, threads: Option<usize>) -> Result<()> {
    let pool = pool(threads);
    let side = 16;
    let mut rng = Rng(11);
    let real: Vec<f32> = (0..16).flat_map(|_| shape_image(&mut rng, side)).collect();
    save_grid("data/ddpm_real.png", &real, side, 16, 4)?;
    let mut d = models::Ddpm::new(side * side, 200, 512, 3);
    let mut opt = aido::sdk::optimizer(&d.params, 1e-3, steps);
    opt.wd = 0.0;
    let t = Instant::now();
    let mut ema = 0.0;
    for s in 0..steps {
        let batch: Vec<f32> = (0..64).flat_map(|_| shape_image(&mut rng, side)).collect();
        let l = d.train_step(&pool, &batch, &mut rng);
        opt.update(&mut d.params);
        ema = if ema == 0.0 { l } else { ema * 0.98 + l * 0.02 };
        if (s + 1) % (steps / 10).max(1) == 0 {
            println!("step {:>5}/{steps} | denoising MSE {ema:.4} | {:.1} steps/s", s + 1, (s + 1) as f64 / t.elapsed().as_secs_f64());
        }
    }
    sdk::save_checkpoint(out, &d.params, Some(&opt), &json!({"kind": "ddpm", "dim": side * side, "steps": 200, "hidden": 512}))?;
    let imgs = d.sample(&pool, 16, 7);
    save_grid("data/ddpm_samples.png", &imgs, side, 16, 4)?;
    println!("trained in {:.1}s → {out}; samples → data/ddpm_samples.png (training examples → data/ddpm_real.png)", t.elapsed().as_secs_f64());
    Ok(())
}

/// `aido sample-image`: samples a grid from a trained DDPM checkpoint.
pub fn sample_image(model: &str, out: &str, n: usize, seed: u64) -> Result<()> {
    let pool = pool(None);
    let (p, _, cfg) = sdk::load_checkpoint(model)?;
    let dim = cfg["dim"].as_u64().unwrap_or(256) as usize;
    let mut d = models::Ddpm::from_params(dim, cfg["steps"].as_u64().unwrap_or(200) as usize, cfg["hidden"].as_u64().unwrap_or(512) as usize, p);
    let imgs = d.sample(&pool, n, seed);
    save_grid(out, &imgs, (dim as f64).sqrt() as usize, n, 4)?;
    println!("saved {out}");
    Ok(())
}

/// `aido train-snake`: Deep Q-Learning on the Snake game.
pub fn train_snake(episodes: usize, out: &str) -> Result<()> {
    // Tiny matrices: one thread avoids synchronisation overhead.
    let pool = ThreadPool::new(1, None);
    let t = Instant::now();
    let (mut net, rep) = aido::sdk::snake::train(episodes, 12, &pool, &mut |ep, m, eps| println!("episode {ep:>5} | mean score (last 50) {m:>5.2} | epsilon {eps:.2}"));
    println!("trained {} episodes in {:.1}s — train mean {:.2}, greedy eval mean {:.2}, best {}", rep.episodes, t.elapsed().as_secs_f64(), rep.train_mean, rep.eval_mean, rep.eval_best);
    sdk::save_checkpoint(out, &net.params, None, &json!({"kind": "snake-dqn", "grid": 12}))?;
    let mut g = aido::sdk::snake::Snake::new(12, 4242);
    loop {
        let q = net.q(&pool, &g.state(), 1);
        let a = (0..3).max_by(|&i, &j| q[i].partial_cmp(&q[j]).unwrap()).unwrap();
        let before = g.clone();
        if g.step(a).1 {
            println!("final board of a greedy game (score {}):\n{}", before.score, before.render());
            break;
        }
    }
    println!("saved {out}");
    Ok(())
}

/// Token map of the sequence model: digits, `,`, ` `, `-`, `|` (answer
/// marker), newline (end).
fn seq_tok(c: char) -> u32 {
    match c {
        '0'..='9' => c as u32 - '0' as u32,
        ',' => 10,
        ' ' => 11,
        '-' => 12,
        '|' => 13,
        _ => 14,
    }
}

fn seq_text(s: &[i64], next: Option<i64>) -> String {
    let body = s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    match next {
        Some(n) => format!("{body}|{n}\n"),
        None => format!("{body}|"),
    }
}

/// `aido train-seq`: small transformer that finds the rule of an integer
/// sequence and predicts the next element.
pub fn train_seq(steps: usize, out: &str) -> Result<()> {
    let pool = pool(None);
    let mut rng = Rng(2024);
    let mut stream = String::new();
    while stream.len() < 2_000_000 {
        let len = 5 + (rng.uniform() * 4.0) as usize;
        let (s, n) = aido::evals::random_sequence(&mut rng, len);
        stream.push_str(&seq_text(&s, Some(n)));
    }
    let toks: Vec<u32> = stream.chars().map(seq_tok).collect();
    data::write_tokens("data/sequences.aidata", &toks, json!({"tokenizer": "digits"}))?;
    let ds = data::DataSet::open("data/sequences.aidata")?;
    let mut m = models::LmModel::new_vocab(16, 192, 4, 4, 576, 96, 3);
    let mut opt = aido::sdk::optimizer(&m.params, 2e-3, steps);
    println!("sequence model: {:.2} M parameters, {} training tokens", m.params.n_trainable() as f64 / 1e6, ds.len());
    lm_loop(&mut m, &mut opt, &pool, &ds, steps, 24, &format!("{out}.ckpt.aido"), (steps / 10).max(1))?;
    // Held-out evaluation on fresh random sequences (different seed).
    let items = aido::evals::sequence_items(200, 7);
    let mut correct = 0;
    let mut per_kind = [0usize; 2];
    for it in &items {
        let nums: Vec<i64> = it.prompt.split("sequence ").nth(1).unwrap().split(", ...").next().unwrap().split(", ").map(|x| x.parse().unwrap()).collect();
        let prompt: Vec<u32> = std::iter::once(14).chain(seq_text(&nums, None).chars().map(seq_tok)).collect();
        let g = m.generate_greedy(&pool, &prompt, 12, 14);
        let txt: String = g.iter().map(|&t| match t { 0..=9 => (b'0' + t as u8) as char, 12 => '-', _ => '?' }).collect();
        let ok = txt == it.answer;
        correct += ok as usize;
        per_kind[ok as usize] += 1;
    }
    println!("held-out next-number accuracy: {}/{} = {:.1}%", correct, items.len(), 100.0 * correct as f64 / items.len() as f64);
    sdk::save_checkpoint(out, &m.params, None, &serde_json::to_value(&m.cfg)?)?;
    // Attach the I/O signature so the file is self-describing and servable.
    aido::sdk::io::attach(out, out, &aido::sdk::io::seq_signature())?;
    println!("saved {out} (with I/O signature: `aido model-info {out}`)");
    Ok(())
}


/// `aido train-pixelart`.
pub fn train_pixelart(steps: usize, ch: usize, out: &str) -> Result<()> {
    use aido::sdk::pixelart::*;
    let pool = pool(None);
    let mut r = Rng(5);
    let real: Vec<f32> = (0..16).flat_map(|i| sprite(i % CLASSES, &mut r)).collect();
    let (w, h, rgb) = grid(&real, 16, 4);
    aido::diffusion::pipeline::Image { width: w, height: h, rgb }.save_png("data/pixelart_dataset.png")?;
    let mut m = PixelArtModel::new(ch, 300, 9);
    println!("pixel-art U-Net: {:.2} M parameters", m.params.n_trainable() as f64 / 1e6);
    let t = Instant::now();
    m.train(&pool, steps, 16, &mut |s, l| println!("step {s:>5}/{steps} | denoising MSE {l:.4} | {:.2} steps/s", s as f64 / t.elapsed().as_secs_f64()));
    sdk::save_checkpoint(out, &m.params, None, &json!({"kind": "pixelart", "channels": ch, "steps": 300}))?;
    println!("trained in {:.1}s → {out}", t.elapsed().as_secs_f64());
    pixelart(out, "all", 8, 40, 3, "data/pixelart_samples.png")
}

/// `aido pixelart`: samples sprites and writes a 256×256-per-sprite grid.
pub fn pixelart(model: &str, class: &str, n: usize, steps: usize, seed: u64, out: &str) -> Result<()> {
    use aido::sdk::pixelart::*;
    let pool = pool(None);
    let (p, _, cfg) = sdk::load_checkpoint(model)?;
    let mut m = PixelArtModel::new(cfg["channels"].as_u64().unwrap_or(48) as usize, cfg["steps"].as_u64().unwrap_or(300) as usize, 1);
    m.params = p;
    let cls: Vec<usize> = (0..n).map(|i| match class { "all" => i % CLASSES, c => NAMES.iter().position(|x| *x == c).unwrap_or(0) }).collect();
    let t = Instant::now();
    let imgs = m.sample(&pool, &cls, steps, seed);
    let (w, h, rgb) = grid(&imgs, n, 8);
    aido::diffusion::pipeline::Image { width: w, height: h, rgb }.save_png(out)?;
    println!("generated {n} sprite(s) of 256x256 in {:.2}s ({} DDIM steps) → {out}", t.elapsed().as_secs_f64(), steps);
    Ok(())
}

/// `aido train-textdet`.
pub fn train_textdet(steps: usize, out: &str, export: usize) -> Result<()> {
    use aido::sdk::textdet::*;
    let pool = pool(None);
    if export > 0 {
        TextSynth::new("data/photos", 0..1000)?.export("data/textdet", export, 256, 3)?;
        println!("exported {export} labelled samples (png/jpg/bmp/webp/tiff + masks + labels.jsonl) → data/textdet/");
    }
    let t = Instant::now();
    let (d, s) = train(&pool, "data/photos", steps, 8, &mut |st, l| println!("step {st:>5}/{steps} | weighted BCE {l:.4} | {:.2} steps/s", st as f64 / t.elapsed().as_secs_f64()))?;
    println!("trained in {:.1}s; held-out photos: mask IoU {:.3} | box precision {:.3} recall {:.3} | matched box IoU {:.3}", t.elapsed().as_secs_f64(), s.mask_iou, s.precision, s.recall, s.box_iou);
    sdk::save_checkpoint(out, &d.params, None, &json!({"kind": "textdet", "res": RES}))?;
    println!("saved {out}");
    Ok(())
}

/// `aido train-bilingual`: own ru/en BPE tokenizer + Llama-style LM.
pub fn train_bilingual(vocab: usize, steps: usize, dim: usize, layers: usize, out: &str) -> Result<()> {
    let files = ["data/corpus/wiki_ru.txt", "data/corpus/wiki_en.txt", "data/ru_wiki.txt", "data/ru_wiki2.txt", "data/alice_en.txt", "data/pride_en.txt"];
    let corpus = read_all(&files.iter().map(|s| s.to_string()).collect::<Vec<_>>())?;
    let t = Instant::now();
    let bpe = aido::sdk::bpe::train(&corpus, vocab);
    println!("tokenizer: {} tokens ({} byte base, {} merges, 3 special) trained in {:.1}s", bpe.tokens.len(), bpe.n_base, bpe.merges.len(), t.elapsed().as_secs_f64());
    // Materialise the tokenizer through an empty .aido so the engine's
    // tokenizer code is used for encoding (same code at inference).
    let meta = bpe.metadata();
    let mut w = aido::format::AidoWriter::new();
    for (k, v) in &meta {
        w.set_meta(k, v.clone());
    }
    w.write("models/ruen-tokenizer.aido")?;
    let st = aido::format::open("models/ruen-tokenizer.aido")?;
    let tok = aido::tokenizer::Tokenizer::from_gguf(st.as_ref())?;
    let text = aido::sdk::bpe::normalize(&corpus);
    let ids = tok.encode(&text, false);
    println!("corpus: {:.2} MB → {} tokens ({:.2} bytes/token)", text.len() as f64 / 1e6, ids.len(), text.len() as f64 / ids.len() as f64);
    for probe in ["Москва — столица России.", "The quick brown fox jumps over the lazy dog."] {
        let e = tok.encode(probe, false);
        println!("  {probe:?} → {} tokens: {:?}", e.len(), e.iter().map(|&i| tok.decode(&[i])).collect::<Vec<_>>());
    }
    data::write_tokens("data/ruen.aidata", &ids, json!({"tokenizer": "models/ruen-tokenizer.aido"}))?;
    let ds = data::DataSet::open("data/ruen.aidata")?;
    let mut m = models::LmModel::new_vocab(tok.len(), dim, layers, dim / 64, dim * 3, 128, 21);
    m.source_meta = Some(meta.iter().map(|(k, v)| (k.clone(), aido::format::json_to_value(v))).collect());
    let mut opt = aido::sdk::optimizer(&m.params, 1.5e-3, steps);
    println!("model: {:.2} M parameters (dim {dim}, {layers} layers, vocab {})", m.params.n_trainable() as f64 / 1e6, tok.len());
    lm_loop(&mut m, &mut opt, &pool(None), &ds, steps, 12, &format!("{out}.ckpt.aido"), (steps / 15).max(1))?;
    let n = m.export(out, GgmlType::F32)?;
    println!("exported {out} ({:.1} MB) — run: aido run -m {out} --raw -p \"Москва\"", n as f64 / 1e6);
    Ok(())
}
