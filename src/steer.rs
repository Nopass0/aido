//! Behaviour editing through **directions in the residual stream**.
//!
//! Many high-level behaviours of a transformer (refusing a request,
//! politeness vs. aggression, language, sentiment) are approximately linear:
//! the mean residual activation on prompts that trigger the behaviour differs
//! from the mean on contrast prompts along one direction `r`. Given two
//! prompt sets the engine:
//!
//! 1. runs both sets and records the residual stream after every block at the
//!    last prompt token ([`mean_resid`]);
//! 2. picks the block where the two means separate best (difference norm
//!    relative to the activation norm) and takes `r = normalize(μ₊ − μ₋)`;
//! 3. edits every matrix that **writes** into the residual stream
//!    (`attn_output`, `ffn_down`, token embeddings):
//!    `W ← W − α·r·(rᵀW)`. `α = 1` removes the model's ability to express the
//!    direction (ablation, e.g. refusal removal); `α < 0` amplifies it
//!    (e.g. more aggressive tone).
//!
//! The edit is a weight change (no runtime hooks), written to a new `.aido`
//! with the provenance recorded, so the result runs at full speed anywhere.

use anyhow::{Context, Result};
use serde_json::json;

use crate::format::AidoWriter;
use crate::llm::chat::{render, Message};
use crate::llm::model::{KvCache, Model, Probe, Workspace};
use crate::runtime::pool::ThreadPool;
use crate::tensor::quant::GgmlType;
use crate::tensor::quantize::{can_quantize, quantize_row};
use crate::tokenizer::Tokenizer;

/// Formats a single user turn with the model's chat template (thinking off).
///
/// # Example
/// ```no_run
/// let m = aido::llm::model::Model::load("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
/// let tok = aido::tokenizer::Tokenizer::from_gguf(m.store.as_ref()).unwrap();
/// let ids = aido::steer::chat_ids(&m, &tok, "hi").unwrap();
/// ```
pub fn chat_ids(m: &Model, tok: &Tokenizer, user: &str) -> Result<Vec<u32>> {
    let name = |id: Option<u32>| id.and_then(|i| tok.vocab.get(i as usize).cloned()).unwrap_or_default();
    let text = render(m.store.get_str("tokenizer.chat_template"), &[Message::new("user", user)], &name(tok.bos), &name(tok.eos), true, false)?;
    Ok(tok.encode(&text, false))
}

/// Mean last-token residual activation after every block over `prompts`.
///
/// * **Output** — `n_layer` vectors of length `hidden`.
///
/// # Example
/// ```no_run
/// # let m = aido::llm::model::Model::load("m.gguf").unwrap();
/// # let tok = aido::tokenizer::Tokenizer::from_gguf(m.store.as_ref()).unwrap();
/// let pool = aido::runtime::pool::ThreadPool::new(8, None);
/// let mu = aido::steer::mean_resid(&m, &tok, &pool, &["hello".to_string()]).unwrap();
/// ```
pub fn mean_resid(m: &Model, tok: &Tokenizer, pool: &ThreadPool, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
    let c = &m.cfg;
    let mut acc = vec![vec![0f32; c.n_embd]; c.n_layer];
    let mut kv = KvCache::new(c.n_layer, 1024, c.kv_dim());
    let ff = m.layers.iter().map(|l| l.w_gate.rows).max().unwrap_or(c.n_ff);
    let mut cfg = c.clone();
    cfg.n_ff = ff;
    let mut ws = Workspace::new(&cfg, 256, 1024, pool.n_threads());
    for p in prompts {
        let ids = chat_ids(m, tok, p)?;
        anyhow::ensure!(ids.len() <= 256, "prompt too long");
        ws.probe = Some(Probe::new(c.n_layer, ff));
        m.forward(pool, &ids, 0, &mut kv, &mut ws, false)?;
        let pr = ws.probe.take().unwrap();
        for (a, r) in acc.iter_mut().zip(&pr.resid) {
            for (x, y) in a.iter_mut().zip(r) {
                *x += y / prompts.len() as f32;
            }
        }
    }
    Ok(acc)
}

/// Per-layer separation `‖μ₊ − μ₋‖ / mean(‖μ₊‖, ‖μ₋‖)`.
///
/// # Example
/// ```
/// let s = aido::steer::separation(&[vec![1.0, 0.0]], &[vec![0.0, 1.0]]);
/// assert!((s[0] - 2f64.sqrt()).abs() < 1e-6);
/// ```
pub fn separation(pos: &[Vec<f32>], neg: &[Vec<f32>]) -> Vec<f64> {
    pos.iter()
        .zip(neg)
        .map(|(a, b)| {
            let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            d / ((na + nb) / 2.0).max(1e-9)
        })
        .collect()
}

/// Unit direction `normalize(a − b)`.
///
/// # Example
/// ```
/// let r = aido::steer::direction(&[3.0, 4.0], &[0.0, 0.0]);
/// assert!((r[0] - 0.6).abs() < 1e-6);
/// ```
pub fn direction(a: &[f32], b: &[f32]) -> Vec<f32> {
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = d.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    d.iter().map(|x| x / n).collect()
}

/// Writes `dst` = `src` with `W ← W − α·r·(rᵀW)` applied to every matrix
/// writing into the residual stream (and to the token embeddings), or only
/// to the blocks in `blocks` (embeddings untouched) when given.
///
/// # Example
/// ```no_run
/// let r = vec![0.0f32; 1024];
/// aido::steer::apply("models/Qwen3-0.6B-Q8_0.gguf", "out.aido", &r, 1.0, "demo", None).unwrap();
/// ```
pub fn apply(src: &str, dst: &str, r: &[f32], alpha: f32, label: &str, blocks: Option<std::ops::Range<usize>>) -> Result<usize> {
    let st = crate::format::open(src)?;
    let s = st.as_ref();
    let d = r.len();
    let mut w = AidoWriter::new();
    w.copy_meta(s);
    w.log(json!({"op": "steer", "label": label, "alpha": alpha, "source": src}));
    let mut edited = 0;
    for n in s.names() {
        let (tid, dims) = s.info(&n).context("info")?;
        let in_range = |n: &str| match &blocks {
            None => true,
            Some(b) => n.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok()).is_some_and(|i| b.contains(&i)),
        };
        let writes = (n.ends_with("attn_output.weight") || n.ends_with("ffn_down.weight")) && in_range(&n);
        let embed = n == "token_embd.weight" && blocks.is_none();
        if !(writes || embed) {
            w.add(&n, tid, dims, s.raw(&n)?.as_slice().to_vec());
            continue;
        }
        let q = s.weight(&n)?;
        let mut m = vec![0f32; q.rows * q.cols];
        for i in 0..q.rows {
            q.dequant_row(i, &mut m[i * q.cols..(i + 1) * q.cols]);
        }
        if writes {
            // rows = hidden (d): project each column.
            anyhow::ensure!(q.rows == d, "{n}: rows {} != hidden {d}", q.rows);
            for j in 0..q.cols {
                let c: f32 = (0..d).map(|i| r[i] * m[i * q.cols + j]).sum();
                for i in 0..d {
                    m[i * q.cols + j] -= alpha * r[i] * c;
                }
            }
        } else {
            // embeddings: rows = vocab, cols = hidden.
            for row in m.chunks_mut(q.cols) {
                let c: f32 = row.iter().zip(r).map(|(a, b)| a * b).sum();
                for (x, y) in row.iter_mut().zip(r) {
                    *x -= alpha * y * c;
                }
            }
        }
        let ty = if can_quantize(q.ty) && q.cols % q.ty.block_elems() == 0 { q.ty } else { GgmlType::Q8_0 };
        let bytes: Vec<u8> = m.chunks(q.cols).flat_map(|row| quantize_row(ty, row)).collect();
        w.add(&n, crate::optimize::ty_id(ty), dims, bytes);
        edited += 1;
    }
    w.write(dst)?;
    Ok(edited)
}

/// Phrases that mark a refusal (English and Russian).
pub const REFUSAL: &[&str] = &[
    "i can't", "i cannot", "i can not", "i'm sorry", "i am sorry", "sorry,", "i won't", "i will not", "as an ai", "i'm not able", "i am not able", "i'm unable",
    "not appropriate", "cannot help", "can't help", "can't assist", "cannot assist", "illegal", "unethical", "not ethical", "harmful", "i'd recommend against",
    "not recommended", "не могу", "извините", "к сожалению", "не буду", "не стану", "незаконн", "неэтично", "не рекомендую",
];

/// Words that mark rude / aggressive language (English and Russian).
pub const RUDE: &[&str] = &[
    "stupid", "idiot", "dumb", "shut up", "moron", "pathetic", "useless", "fool", "damn", "hell", "loser", "ridiculous", "nonsense", "whatever",
    "туп", "идиот", "дурак", "заткн", "жалк", "бесполезн", "чушь", "бред", "отвали",
];

/// `true` if `text` looks like a refusal.
///
/// # Example
/// ```
/// assert!(aido::steer::is_refusal("I'm sorry, but I can't help with that."));
/// assert!(!aido::steer::is_refusal("Sure! Here is how."));
/// ```
pub fn is_refusal(text: &str) -> bool {
    // Refusals are stated up front; later mentions are usually caveats.
    let t: String = text.to_lowercase().chars().take(120).collect();
    REFUSAL.iter().any(|p| t.contains(p))
}

/// Number of rude markers in `text`.
///
/// # Example
/// ```
/// assert_eq!(aido::steer::rudeness("You idiot, that's stupid."), 2);
/// ```
pub fn rudeness(text: &str) -> usize {
    let t = text.to_lowercase();
    RUDE.iter().map(|p| t.matches(p).count()).sum()
}

/// Writes `dst` = `src` plus a steering vector `v` added to the residual
/// stream after block `layer` (stored as tensor `steer.blk.{layer}`; loaded
/// automatically by [`crate::llm::model::Model`]).
///
/// # Example
/// ```no_run
/// aido::steer::add_vector("models/Qwen3-0.6B-Q8_0.gguf", "steered.aido", 18, &vec![0.0; 1024], "demo").unwrap();
/// ```
pub fn add_vector(src: &str, dst: &str, layer: usize, v: &[f32], label: &str) -> Result<()> {
    let st = crate::format::open(src)?;
    let s = st.as_ref();
    let mut w = AidoWriter::new();
    w.copy_meta(s);
    w.log(json!({"op": "steer_add", "label": label, "layer": layer, "norm": v.iter().map(|x| x * x).sum::<f32>().sqrt()}));
    for n in s.names() {
        if n.starts_with("steer.") {
            continue;
        }
        let (tid, dims) = s.info(&n).context("info")?;
        w.add(&n, tid, dims, s.raw(&n)?.as_slice().to_vec());
    }
    w.add(&format!("steer.blk.{layer}"), 0, vec![v.len()], v.iter().flat_map(|x| x.to_le_bytes()).collect());
    w.write(dst)?;
    Ok(())
}
