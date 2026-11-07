//! Model optimisation toolkit: analysis, pruning, layer removal, vocabulary
//! trimming, re-quantisation and compilation to `.aido`.
//!
//! Workflow (see `docs/RESEARCH.md` for the measurements behind it):
//!
//! ```text
//! aido analyze -m model.gguf --corpus my_domain.txt --out stats.json
//! aido compile -m model.gguf --stats stats.json --ffn-keep 0.8 \
//!              --drop-layers auto:2 --vocab-corpus my_domain.txt -o model.aido
//! aido eval    -m model.aido --text heldout.txt        # bits per byte
//! ```
//!
//! ### Pruning criterion
//! FFN neuron `j` contributes `a_j · W_down[:, j]` to the residual stream,
//! so its expected contribution is `E|a_j| · ‖W_down[:, j]‖₂`. Neurons are
//! ranked globally by this score (normalised by each layer's mean score so
//! that layers with different activation scales compete fairly); every
//! layer keeps at least 25 % of its neurons and neurons with a large
//! *maximum* activation ("massive activations", often < 0.1 % of neurons
//! but critical) are always kept. Counts are rounded to the quantisation
//! block size so the pruned matrices stay in their native format.
//!
//! ### Layer removal
//! Block influence `BI = 1 − E[cos(x_in, x_out)]` measures how much a block
//! rotates the residual stream. `auto:N` removes the `N` blocks with the
//! smallest BI (never the first or the last block).
//!
//! ### Vocabulary trimming
//! Tokens never produced by the original tokenizer on the target corpus are
//! removed, **except** control/special tokens, all single-byte tokens
//! (so any text remains encodable) and every token needed as an
//! intermediate step of a BPE merge chain of a kept token (merge closure).
//! The embedding table and LM head shrink accordingly — the LM head alone is
//! 10–26 % of all weight traffic per token on small models.

use std::collections::{BTreeSet, HashMap};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::format::{open, value_to_json, AidoWriter, TensorStore};
use crate::gguf::Value;
use crate::llm::model::{KvCache, Model, Probe, Workspace};
#[allow(unused_imports)]
use crate::llm::LanguageModel as _;
use crate::runtime::pool::ThreadPool;
use crate::tensor::quant::GgmlType;
use crate::tensor::quantize::{can_quantize, quantize_row};
use crate::tokenizer::Tokenizer;

/// Serializable analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    /// Tokens observed.
    pub tokens: u64,
    /// Block influence per layer.
    pub block_influence: Vec<f64>,
    /// Mean |activation| per layer and FFN neuron.
    pub ffn_mean: Vec<Vec<f32>>,
    /// Max |activation| per layer and FFN neuron.
    pub ffn_max: Vec<Vec<f32>>,
    /// How often each token id occurred in the corpus.
    pub token_counts: Vec<u32>,
}

impl Stats {
    /// Fraction of neurons (all layers) whose mean activation is below
    /// `rel` × their layer's mean activation — a sparsity indicator.
    ///
    /// # Example
    /// ```
    /// let s = aido::optimize::Stats { tokens: 1, block_influence: vec![0.1], ffn_mean: vec![vec![0.0, 1.0]], ffn_max: vec![vec![0.0, 1.0]], token_counts: vec![] };
    /// assert_eq!(s.cold_fraction(0.1), 0.5);
    /// ```
    pub fn cold_fraction(&self, rel: f32) -> f64 {
        let mut cold = 0usize;
        let mut total = 0usize;
        for l in &self.ffn_mean {
            let mean = l.iter().sum::<f32>() / l.len().max(1) as f32;
            cold += l.iter().filter(|&&v| v < rel * mean).count();
            total += l.len();
        }
        cold as f64 / total.max(1) as f64
    }
}

/// Runs the model over `corpus` (in windows of `window` tokens, at most
/// `max_tokens` in total) collecting activation and token statistics.
///
/// * **Input** — loaded model and tokenizer, pool, corpus text, limits.
/// * **Output** — [`Stats`].
///
/// # Example
/// ```no_run
/// use aido::{llm::model::Model, tokenizer::Tokenizer, runtime::pool::ThreadPool};
/// let m = Model::load("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
/// let tok = Tokenizer::from_gguf(m.store.as_ref()).unwrap();
/// let pool = ThreadPool::new(8, None);
/// let st = aido::optimize::analyze(&m, &tok, &pool, "some text ...", 2048, 256).unwrap();
/// println!("{:?}", st.block_influence);
/// ```
pub fn analyze(model: &Model, tok: &Tokenizer, pool: &ThreadPool, corpus: &str, max_tokens: usize, window: usize) -> Result<Stats> {
    let ids = tok.encode(corpus, false);
    let mut token_counts = vec![0u32; model.cfg.vocab];
    for &i in &ids {
        if let Some(c) = token_counts.get_mut(i as usize) {
            *c += 1;
        }
    }
    let ff_max = model.layers.iter().map(|l| l.w_gate.rows).max().unwrap_or(model.cfg.n_ff);
    let mut cfg = model.cfg.clone();
    cfg.n_ff = ff_max;
    let mut kv = KvCache::new(cfg.n_layer, window, cfg.kv_dim());
    let mut ws = Workspace::new(&cfg, window.min(64), window, pool.n_threads());
    ws.probe = Some(Probe::new(cfg.n_layer, ff_max));
    let n = ids.len().min(max_tokens);
    let mut done = 0;
    while done < n {
        let end = (done + window).min(n);
        let mut pos = 0;
        let mut s = done;
        while s < end {
            let e = (s + 64).min(end);
            model.forward(pool, &ids[s..e], pos, &mut kv, &mut ws, false)?;
            pos += e - s;
            s = e;
        }
        done = end;
    }
    let p = ws.probe.take().unwrap();
    let tn = p.tokens.max(1) as f64;
    Ok(Stats {
        tokens: p.tokens,
        block_influence: p.block_influence(),
        ffn_mean: p.ffn_abs.iter().map(|l| l.iter().map(|v| (*v / tn) as f32).collect()).collect(),
        ffn_max: p.ffn_max.clone(),
        token_counts,
    })
}

/// Bits per byte of `text` under the model (lower is better). Comparable
/// across tokenizers, so it measures the cost of vocabulary trimming too.
///
/// * **Input** — model, tokenizer, pool, text, evaluation window.
/// * **Output** — `(bits_per_byte, tokens_evaluated)`.
///
/// # Example
/// ```no_run
/// use aido::{llm::model::Model, tokenizer::Tokenizer, runtime::pool::ThreadPool};
/// let m = Model::load("m.gguf").unwrap();
/// let tok = Tokenizer::from_gguf(m.store.as_ref()).unwrap();
/// let (bpb, n) = aido::optimize::bits_per_byte(&m, &tok, &ThreadPool::new(8, None), "text", 256).unwrap();
/// ```
pub fn bits_per_byte(model: &dyn crate::llm::LanguageModel, tok: &Tokenizer, pool: &ThreadPool, text: &str, window: usize) -> Result<(f64, usize)> {
    let ids = tok.encode(text, false);
    let vocab = model.vocab();
    let mut st = model.new_state(window, 32, pool.n_threads());
    let mut nll = 0f64;
    let mut count = 0usize;
    let mut bytes = 0usize;
    let mut start = 0;
    while start + 1 < ids.len() {
        let end = (start + window).min(ids.len());
        let mut pos = 0;
        let mut s = start;
        while s < end - 1 {
            let e = (s + 32).min(end - 1);
            model.forward(pool, &ids[s..e], pos, st.as_mut(), true)?;
            for (i, row) in st.logits().chunks_exact(vocab).enumerate() {
                let target = ids[s + i + 1] as usize;
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse = m as f64 + row.iter().map(|v| ((v - m) as f64).exp()).sum::<f64>().ln();
                nll += lse - row[target] as f64;
                count += 1;
                bytes += tok.token_bytes(target as u32).len();
            }
            pos += e - s;
            s = e;
        }
        start = end - 1;
        if end == ids.len() {
            break;
        }
    }
    Ok((nll / std::f64::consts::LN_2 / bytes.max(1) as f64, count))
}

/// What to do when compiling.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// Re-quantise all matrices to this type (`None` keeps original types).
    pub quant: Option<GgmlType>,
    /// Blocks to delete.
    pub drop_layers: Vec<usize>,
    /// Remove this many lowest-BI blocks automatically (needs stats).
    pub drop_auto: usize,
    /// Fraction of FFN neurons to keep (needs stats), e.g. `0.8`.
    pub ffn_keep: Option<f32>,
    /// Activation statistics from [`analyze`].
    pub stats: Option<Stats>,
    /// Keep only tokens used in this corpus (plus specials/bytes/closure).
    pub vocab_corpus: Option<String>,
}

/// Result summary of a compilation.
#[derive(Debug, Clone)]
pub struct CompileReport {
    /// Input weight bytes.
    pub bytes_in: usize,
    /// Output file bytes.
    pub bytes_out: u64,
    /// Layers kept.
    pub layers: usize,
    /// Vocabulary size after trimming.
    pub vocab: usize,
    /// FFN neurons kept per layer.
    pub ffn: Vec<usize>,
}

/// Selects a block-aligned set of neurons to keep in every layer.
fn select_neurons(st: &Stats, down_norms: &[Vec<f32>], keep: f32, align: usize, layers: &[usize]) -> Vec<Vec<usize>> {
    // Global ranking of normalised scores.
    let mut all: Vec<(f32, usize, usize)> = Vec::new();
    let mut scores: Vec<Vec<f32>> = Vec::new();
    for (k, &l) in layers.iter().enumerate() {
        let s: Vec<f32> = st.ffn_mean[l].iter().zip(&down_norms[k]).map(|(a, n)| a * n).collect();
        let mean = s.iter().sum::<f32>() / s.len() as f32 + 1e-12;
        for (j, v) in s.iter().enumerate() {
            all.push((v / mean, k, j));
        }
        scores.push(s);
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let total = all.len();
    let target = (total as f32 * keep) as usize;
    let mut count = vec![0usize; layers.len()];
    for e in all.iter().take(target) {
        count[e.1] += 1;
    }
    layers
        .iter()
        .enumerate()
        .map(|(k, &l)| {
            let n = scores[k].len();
            let want = count[k].max(n / 4).div_ceil(align) * align;
            let want = want.min(n);
            let mut idx: Vec<usize> = (0..n).collect();
            // Massive-activation guard: top 0.5 % by max |a| always survive.
            let mut by_max = idx.clone();
            by_max.sort_by(|&a, &b| st.ffn_max[l][b].partial_cmp(&st.ffn_max[l][a]).unwrap());
            let guard: BTreeSet<usize> = by_max.into_iter().take((n / 200).max(1)).collect();
            idx.sort_by(|&a, &b| {
                let ga = guard.contains(&a);
                let gb = guard.contains(&b);
                gb.cmp(&ga).then(scores[k][b].partial_cmp(&scores[k][a]).unwrap())
            });
            let mut keep: Vec<usize> = idx.into_iter().take(want).collect();
            keep.sort_unstable();
            keep
        })
        .collect()
}

/// Computes the kept-token set for vocabulary trimming.
fn vocab_keep(tok: &Tokenizer, store: &dyn TensorStore, corpus: &str) -> Vec<u32> {
    let mut keep: BTreeSet<u32> = BTreeSet::new();
    for id in tok.encode(corpus, false) {
        keep.insert(id);
    }
    for (i, s) in tok.vocab.iter().enumerate() {
        let ty = tok.types.get(i).copied().unwrap_or(1);
        // control / user-defined / unknown / byte tokens, and every single character
        if ty != 1 || s.chars().count() == 1 {
            keep.insert(i as u32);
        }
    }
    // Merge closure (BPE): keep both parents of every kept merged token.
    if let Some(Value::A(merges)) = store.meta().get("tokenizer.ggml.merges") {
        let mut parent: HashMap<u32, (u32, u32)> = HashMap::new();
        for m in merges {
            if let Some((a, b)) = m.as_str().and_then(|s| s.split_once(' ')) {
                if let (Some(ia), Some(ib), Some(ir)) = (tok.token_id(a), tok.token_id(b), tok.token_id(&format!("{a}{b}"))) {
                    parent.entry(ir).or_insert((ia, ib));
                }
            }
        }
        let mut stack: Vec<u32> = keep.iter().cloned().collect();
        while let Some(t) = stack.pop() {
            if let Some(&(a, b)) = parent.get(&t) {
                for p in [a, b] {
                    if keep.insert(p) {
                        stack.push(p);
                    }
                }
            }
        }
    }
    keep.into_iter().collect()
}

/// Dequantises a whole matrix to f32 (row-major).
fn dequant_matrix(store: &dyn TensorStore, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let w = store.weight(name)?;
    let mut out = vec![0f32; w.rows * w.cols];
    for r in 0..w.rows {
        w.dequant_row(r, &mut out[r * w.cols..(r + 1) * w.cols]);
    }
    Ok((out, w.rows, w.cols))
}

/// Picks an output type for a matrix with `cols` columns.
fn fit_type(want: GgmlType, cols: usize) -> GgmlType {
    if cols % want.block_elems() == 0 {
        want
    } else if cols % 32 == 0 {
        GgmlType::Q8_0
    } else {
        GgmlType::F16
    }
}

/// Encodes `rows` of an f32 matrix (selected by `row_idx`, columns by
/// `col_idx`) into `ty`.
fn encode_rows(src: &[f32], cols: usize, row_idx: &[usize], col_idx: Option<&[usize]>, ty: GgmlType) -> Vec<u8> {
    let mut out = Vec::new();
    let mut tmp = Vec::new();
    for &r in row_idx {
        let row = &src[r * cols..(r + 1) * cols];
        let data: &[f32] = match col_idx {
            Some(ci) => {
                tmp.clear();
                tmp.extend(ci.iter().map(|&c| row[c]));
                &tmp
            }
            None => row,
        };
        out.extend(quantize_row(ty, data));
    }
    out
}

/// Copies selected rows of a quantised tensor without re-quantisation.
fn copy_rows(store: &dyn TensorStore, name: &str, rows: &[usize]) -> Result<Vec<u8>> {
    let w = store.weight(name)?;
    let mut out = Vec::with_capacity(rows.len() * w.ty.row_bytes(w.cols));
    for &r in rows {
        out.extend_from_slice(w.row(r));
    }
    Ok(out)
}

/// Compiles a model into an optimised `.aido` file.
///
/// * **Input** — source path (GGUF or `.aido`), destination, options, pool
///   (used only for analysis-free steps that need it; may be small).
/// * **Output** — [`CompileReport`].
/// * **Behaviour** — tensors are emitted in execution order; every
///   transformation is recorded in the provenance log; unchanged tensors are
///   copied bit-exactly (no re-quantisation loss unless requested).
///
/// # Example
/// ```no_run
/// use aido::optimize::{compile, CompileOptions};
/// let r = compile("models/Qwen3-0.6B-Q8_0.gguf", "q.aido", &CompileOptions { drop_layers: vec![20], ..Default::default() }).unwrap();
/// println!("{} -> {} bytes", r.bytes_in, r.bytes_out);
/// ```
pub fn compile(src: &str, dst: &str, opt: &CompileOptions) -> Result<CompileReport> {
    let store = open(src)?;
    let s = store.as_ref();
    let arch = s.arch().to_string();
    let n_layer = s.get_u64(&format!("{arch}.block_count")).context("block_count")? as usize;
    // ---- which layers survive ----
    let mut drop: BTreeSet<usize> = opt.drop_layers.iter().cloned().collect();
    if opt.drop_auto > 0 {
        let st = opt.stats.as_ref().context("--drop-layers auto needs --stats")?;
        let mut bi: Vec<(f64, usize)> = st.block_influence.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
        bi.retain(|x| x.1 != 0 && x.1 + 1 != n_layer);
        bi.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        drop.extend(bi.iter().take(opt.drop_auto).map(|x| x.1));
    }
    let layers: Vec<usize> = (0..n_layer).filter(|l| !drop.contains(l)).collect();
    if layers.is_empty() {
        bail!("cannot drop all layers");
    }
    let tok = Tokenizer::from_gguf(s)?;
    let mut w = AidoWriter::new();
    w.copy_meta(s);
    w.log(json!({"op": "source", "path": src, "arch": arch}));
    w.set_meta(&format!("{arch}.block_count"), json!(layers.len()));
    if !drop.is_empty() {
        w.log(json!({"op": "drop_layers", "layers": drop.iter().collect::<Vec<_>>()}));
    }
    // ---- FFN pruning plan ----
    let ffn_sel: Option<Vec<Vec<usize>>> = match opt.ffn_keep {
        Some(keep) if keep < 1.0 => {
            let st = opt.stats.as_ref().context("--ffn-keep needs --stats from `aido analyze`")?;
            let mut norms = Vec::new();
            let mut align = 32;
            for &l in &layers {
                let (dn, rows, cols) = dequant_matrix(s, &format!("blk.{l}.ffn_down.weight"))?;
                let ty = s.info(&format!("blk.{l}.ffn_down.weight")).map(|x| x.0).unwrap_or(0);
                if GgmlType::from_id(ty).map(|t| t.block_elems()).unwrap_or(32) == 256 || opt.quant.map(|q| q.block_elems()) == Some(256) {
                    align = 256;
                }
                let mut n = vec![0f32; cols];
                for r in 0..rows {
                    for c in 0..cols {
                        n[c] += dn[r * cols + c] * dn[r * cols + c];
                    }
                }
                norms.push(n.into_iter().map(f32::sqrt).collect());
            }
            let sel = select_neurons(st, &norms, keep, align, &layers);
            w.log(json!({"op": "prune_ffn", "keep": keep, "criterion": "E|a|*||W_down col||", "align": align, "per_layer": sel.iter().map(|v| v.len()).collect::<Vec<_>>()}));
            Some(sel)
        }
        _ => None,
    };
    if let Some(sel) = &ffn_sel {
        let lens: Vec<usize> = sel.iter().map(|v| v.len()).collect();
        w.set_meta(&format!("{arch}.feed_forward_length"), json!(lens.iter().max().unwrap()));
        w.set_meta("aido.ffn_lengths", json!(lens));
    }
    // ---- vocabulary plan ----
    let vocab_n = s.info("token_embd.weight").context("token_embd")?.1[1];
    let vkeep: Option<Vec<u32>> = opt.vocab_corpus.as_ref().map(|c| vocab_keep(&tok, s, c));
    if let Some(vk) = &vkeep {
        let remap: HashMap<u32, u32> = vk.iter().enumerate().map(|(n, &o)| (o, n as u32)).collect();
        let pick = |key: &str| -> Option<Vec<serde_json::Value>> {
            s.meta().get(key).and_then(|v| v.as_array()).map(|a| vk.iter().map(|&i| value_to_json(&a[i as usize])).collect())
        };
        for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.scores"] {
            if let Some(v) = pick(key) {
                w.set_meta(key, serde_json::Value::Array(v));
            }
        }
        if let Some(Value::A(merges)) = s.meta().get("tokenizer.ggml.merges") {
            let kept: Vec<serde_json::Value> = merges
                .iter()
                .filter_map(|m| m.as_str())
                .filter(|m| {
                    m.split_once(' ').is_some_and(|(a, b)| {
                        [tok.token_id(a), tok.token_id(b), tok.token_id(&format!("{a}{b}"))].iter().all(|x| x.is_some_and(|i| remap.contains_key(&i)))
                    })
                })
                .map(|m| json!(m))
                .collect();
            w.set_meta("tokenizer.ggml.merges", serde_json::Value::Array(kept));
        }
        for (k, v) in s.meta() {
            if k.starts_with("tokenizer.ggml.") && k.ends_with("_token_id") {
                if let Some(id) = v.as_u64() {
                    match remap.get(&(id as u32)) {
                        Some(n) => w.set_meta(k, json!(n)),
                        None => w.set_meta(k, json!(0)),
                    }
                }
            }
        }
        w.set_meta("aido.vocab_map", json!(vk));
        w.log(json!({"op": "trim_vocab", "from": vocab_n, "to": vk.len()}));
    }
    // ---- emit tensors in execution order ----
    let quant = opt.quant;
    if let Some(q) = quant {
        anyhow::ensure!(can_quantize(q), "cannot quantise to {q:?}");
        w.log(json!({"op": "quantize", "to": format!("{q:?}")}));
    }
    let mut ffn_lens = Vec::new();
    let emit_matrix = |w: &mut AidoWriter, from: &str, to: &str, rows: Option<&[usize]>, cols: Option<&[usize]>| -> Result<usize> {
        let (tid, dims) = s.info(from).with_context(|| format!("missing {from}"))?;
        let src_ty = GgmlType::from_id(tid).context("type")?;
        let (c0, r0) = (dims[0], dims[1..].iter().product::<usize>().max(1));
        let out_rows: Vec<usize> = rows.map(|r| r.to_vec()).unwrap_or_else(|| (0..r0).collect());
        let out_cols = cols.map(|c| c.len()).unwrap_or(c0);
        let is_matrix = dims.len() >= 2;
        let want = if is_matrix { quant.unwrap_or(src_ty) } else { src_ty };
        let ty = fit_type(want, out_cols);
        let bytes = if cols.is_none() && ty == src_ty {
            copy_rows(s, from, &out_rows)?
        } else {
            let (m, _, cc) = dequant_matrix(s, from)?;
            encode_rows(&m, cc, &out_rows, cols, ty)
        };
        let new_dims = if dims.len() >= 2 { vec![out_cols, out_rows.len()] } else { vec![out_cols] };
        w.add(to, ty_id(ty), new_dims, bytes);
        Ok(out_rows.len())
    };
    for (new_l, &l) in layers.iter().enumerate() {
        let p = |n: &str| format!("blk.{l}.{n}");
        let q = |n: &str| format!("blk.{new_l}.{n}");
        for n in ["attn_norm.weight", "attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias", "attn_v.weight", "attn_v.bias", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight", "ffn_norm.weight"] {
            if s.has(&p(n)) {
                emit_matrix(&mut w, &p(n), &q(n), None, None)?;
            }
        }
        let sel = ffn_sel.as_ref().map(|v| v[new_l].as_slice());
        let k = emit_matrix(&mut w, &p("ffn_gate.weight"), &q("ffn_gate.weight"), sel, None)?;
        emit_matrix(&mut w, &p("ffn_up.weight"), &q("ffn_up.weight"), sel, None)?;
        emit_matrix(&mut w, &p("ffn_down.weight"), &q("ffn_down.weight"), None, sel)?;
        ffn_lens.push(k);
    }
    emit_matrix(&mut w, "output_norm.weight", "output_norm.weight", None, None)?;
    let vrows: Option<Vec<usize>> = vkeep.as_ref().map(|v| v.iter().map(|&i| i as usize).collect());
    if s.has("output.weight") {
        emit_matrix(&mut w, "output.weight", "output.weight", vrows.as_deref(), None)?;
    }
    if s.has("rope_freqs.weight") {
        emit_matrix(&mut w, "rope_freqs.weight", "rope_freqs.weight", None, None)?;
    }
    // Cold tensors last.
    let vocab_out = emit_matrix(&mut w, "token_embd.weight", "token_embd.weight", vrows.as_deref(), None)?;
    let bytes_out = w.write(dst)?;
    let bytes_in: usize = s.names().iter().filter_map(|n| s.raw(n).ok()).map(|b| b.len()).sum();
    Ok(CompileReport { bytes_in, bytes_out, layers: layers.len(), vocab: vocab_out, ffn: ffn_lens })
}

/// GGML numeric id of a supported type.
///
/// # Example
/// ```
/// use aido::{optimize::ty_id, tensor::quant::GgmlType};
/// assert_eq!(ty_id(GgmlType::Q4_K), 12);
/// ```
pub fn ty_id(t: GgmlType) -> u32 {
    match t {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q4_0 => 2,
        GgmlType::Q4_1 => 3,
        GgmlType::Q5_0 => 6,
        GgmlType::Q5_1 => 7,
        GgmlType::Q8_0 => 8,
        GgmlType::Q4_K => 12,
        GgmlType::Q5_K => 13,
        GgmlType::Q6_K => 14,
        GgmlType::BF16 => 30,
    }
}
