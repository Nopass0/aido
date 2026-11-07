//! Weight inspection and surgery on arbitrary **slices** of any tensor.
//!
//! A slice is `tensor[rows a..b, cols c..d]` (rows = output neurons, cols =
//! input features, GGUF orientation). Tools:
//!
//! * [`list`] — every tensor with type, shape and summary statistics;
//! * [`slice_stats`] — distribution of a slice: mean / std / abs-max,
//!   sparsity, kurtosis (outlier-heaviness), histogram, top outliers and
//!   the most / least "energetic" rows (neurons);
//! * [`edit`] — apply an operation to a slice (`scale`, `set`, `add`,
//!   `noise`, `clamp`, `prune` below a threshold) and write a new `.aido`
//!   (untouched tensors are copied bit-exactly, the edited one is
//!   re-quantised to its original format). Every edit is appended to the
//!   provenance log, so experiments are reproducible.

use std::ops::Range;

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::format::{open, AidoWriter, TensorStore};
use crate::tensor::quant::GgmlType;
use crate::tensor::quantize::{can_quantize, quantize_row};

/// Parses `a:b` (either side optional) into a range clipped to `len`.
///
/// # Example
/// ```
/// assert_eq!(aido::weights::parse_range(Some("2:5"), 10).unwrap(), 2..5);
/// assert_eq!(aido::weights::parse_range(None, 10).unwrap(), 0..10);
/// assert_eq!(aido::weights::parse_range(Some(":3"), 10).unwrap(), 0..3);
/// ```
pub fn parse_range(s: Option<&str>, len: usize) -> Result<Range<usize>> {
    let Some(s) = s else { return Ok(0..len) };
    let (a, b) = s.split_once(':').unwrap_or((s, ""));
    let a = if a.is_empty() { 0 } else { a.parse()? };
    let b = if b.is_empty() { if s.contains(':') { len } else { a + 1 } } else { b.parse()? };
    if a >= b || b > len {
        bail!("range {s} invalid for length {len}");
    }
    Ok(a..b)
}

/// Statistics of a set of values.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Stats {
    /// Number of values.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Minimum.
    pub min: f32,
    /// Maximum.
    pub max: f32,
    /// Fraction with |w| < 1 % of abs-max.
    pub near_zero: f64,
    /// Excess kurtosis (≫ 0 → heavy outliers).
    pub kurtosis: f64,
}

/// Computes [`Stats`] of `v`.
///
/// # Example
/// ```
/// let s = aido::weights::stats(&[1.0, -1.0, 1.0, -1.0]);
/// assert_eq!(s.mean, 0.0);
/// assert!((s.std - 1.0).abs() < 1e-9);
/// ```
pub fn stats(v: &[f32]) -> Stats {
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    let m4 = v.iter().map(|x| (*x as f64 - mean).powi(4)).sum::<f64>() / n;
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    Stats {
        n: v.len(),
        mean,
        std: var.sqrt(),
        min: v.iter().cloned().fold(f32::INFINITY, f32::min),
        max: v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        near_zero: v.iter().filter(|x| x.abs() < 0.01 * amax).count() as f64 / n,
        kurtosis: if var > 0.0 { m4 / (var * var) - 3.0 } else { 0.0 },
    }
}

/// Loads a slice of a tensor as f32 (`rows × cols` of the slice).
///
/// # Example
/// ```no_run
/// let s = aido::format::open("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
/// let (v, r, c) = aido::weights::read_slice(s.as_ref(), "blk.0.ffn_down.weight", 0..4, 0..8).unwrap();
/// assert_eq!(v.len(), r * c);
/// ```
pub fn read_slice(st: &dyn TensorStore, name: &str, rows: Range<usize>, cols: Range<usize>) -> Result<(Vec<f32>, usize, usize)> {
    let w = st.weight(name)?;
    let mut row = vec![0f32; w.cols];
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for r in rows.clone() {
        w.dequant_row(r, &mut row);
        out.extend_from_slice(&row[cols.clone()]);
    }
    Ok((out, rows.len(), cols.len()))
}

/// Prints one line per tensor: type, shape, MB, mean, std, abs-max, kurtosis.
///
/// * **Input** — model path; optional substring filter.
///
/// # Example
/// ```no_run
/// aido::weights::list("models/Qwen3-0.6B-Q8_0.gguf", Some("blk.0.")).unwrap();
/// ```
pub fn list(path: &str, filter: Option<&str>) -> Result<()> {
    let st = open(path)?;
    println!("{:<34} {:>6} {:>16} {:>8} {:>9} {:>8} {:>8} {:>8}", "tensor", "type", "shape", "MB", "mean", "std", "absmax", "kurt");
    for name in st.names() {
        if filter.is_some_and(|f| !name.contains(f)) {
            continue;
        }
        let (tid, dims) = st.info(&name).context("info")?;
        let bytes = st.raw(&name).map(|b| b.len()).unwrap_or(0);
        let shape = dims.iter().rev().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
        // Sample at most 64 rows for speed.
        let s = match st.weight(&name) {
            Ok(w) => {
                let step = (w.rows / 64).max(1);
                let mut v = Vec::new();
                let mut row = vec![0f32; w.cols];
                for r in (0..w.rows).step_by(step) {
                    w.dequant_row(r, &mut row);
                    v.extend_from_slice(&row);
                }
                Some(stats(&v))
            }
            Err(_) => None,
        };
        match s {
            Some(s) => println!("{name:<34} {:>6} {shape:>16} {:>8.2} {:>9.5} {:>8.5} {:>8.4} {:>8.1}", GgmlType::name_of(tid), bytes as f64 / 1e6, s.mean, s.std, s.max.abs().max(s.min.abs()), s.kurtosis),
            None => println!("{name:<34} {:>6} {shape:>16} {:>8.2}", GgmlType::name_of(tid), bytes as f64 / 1e6),
        }
    }
    Ok(())
}

/// Detailed analysis of one slice: statistics, histogram, outliers, and
/// per-row (neuron) L2 norms. Returns the statistics.
///
/// # Example
/// ```no_run
/// aido::weights::slice_stats("models/Qwen3-0.6B-Q8_0.gguf", "blk.5.ffn_down.weight", Some("0:128"), None, 10).unwrap();
/// ```
pub fn slice_stats(path: &str, name: &str, rows: Option<&str>, cols: Option<&str>, top: usize) -> Result<Stats> {
    let st = open(path)?;
    let w = st.weight(name)?;
    let (rr, cr) = (parse_range(rows, w.rows)?, parse_range(cols, w.cols)?);
    let (v, nr, nc) = read_slice(st.as_ref(), name, rr.clone(), cr.clone())?;
    let s = stats(&v);
    println!("{name} [{:?} rows, {:?} cols] type {:?}: {} values", rr, cr, w.ty, s.n);
    println!("mean {:.6}  std {:.6}  min {:.5}  max {:.5}  near-zero {:.2}%  excess kurtosis {:.2}", s.mean, s.std, s.min, s.max, s.near_zero * 100.0, s.kurtosis);
    // Histogram (20 bins).
    let (lo, hi) = (s.min, s.max);
    let mut h = [0usize; 20];
    for x in &v {
        let b = (((x - lo) / (hi - lo + 1e-12)) * 20.0) as usize;
        h[b.min(19)] += 1;
    }
    let hm = *h.iter().max().unwrap_or(&1);
    for (i, c) in h.iter().enumerate() {
        let a = lo + (hi - lo) * i as f32 / 20.0;
        println!("  {a:>+9.4} | {}", "#".repeat(c * 50 / hm.max(1)));
    }
    // Outliers.
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].abs().partial_cmp(&v[a].abs()).unwrap());
    println!("top {top} |w| (row, col, value):");
    for &i in idx.iter().take(top) {
        println!("  ({:>6}, {:>6}) {:+.5}  ({:.1} σ)", rr.start + i / nc, cr.start + i % nc, v[i], (v[i] as f64 - s.mean).abs() / s.std.max(1e-12));
    }
    // Row energy.
    let mut norms: Vec<(f32, usize)> = (0..nr).map(|r| (v[r * nc..(r + 1) * nc].iter().map(|x| x * x).sum::<f32>().sqrt(), rr.start + r)).collect();
    norms.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("strongest rows: {:?}", norms.iter().take(top.min(8)).map(|x| (x.1, (x.0 * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    println!("weakest rows:   {:?}", norms.iter().rev().take(top.min(8)).map(|x| (x.1, (x.0 * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    Ok(s)
}

/// An edit operation on a slice.
#[derive(Debug, Clone, Copy)]
pub enum EditOp {
    /// Multiply by a factor.
    Scale(f32),
    /// Set to a constant.
    Set(f32),
    /// Add a constant.
    Add(f32),
    /// Add Gaussian noise with this std (relative to the slice std).
    Noise(f32),
    /// Clamp to ±k·std.
    Clamp(f32),
    /// Zero values with |w| < k·std (magnitude pruning).
    Prune(f32),
}

impl std::str::FromStr for EditOp {
    type Err = anyhow::Error;
    /// Parses `scale:0.5`, `set:0`, `add:0.1`, `noise:0.05`, `clamp:3`, `prune:0.5`.
    fn from_str(s: &str) -> Result<Self> {
        let (k, v) = s.split_once(':').context("op format is name:value")?;
        let v: f32 = v.parse()?;
        Ok(match k {
            "scale" => EditOp::Scale(v),
            "set" => EditOp::Set(v),
            "add" => EditOp::Add(v),
            "noise" => EditOp::Noise(v),
            "clamp" => EditOp::Clamp(v),
            "prune" => EditOp::Prune(v),
            _ => bail!("unknown op {k}"),
        })
    }
}

/// Applies `op` to `tensor[rows, cols]` and writes the whole model to
/// `dst` (`.aido`). Returns how many values changed.
///
/// # Example
/// ```no_run
/// use aido::weights::{edit, EditOp};
/// edit("models/Qwen3-0.6B-Q8_0.gguf", "ablate.aido", "blk.10.ffn_down.weight", None, None, EditOp::Set(0.0), 1).unwrap();
/// ```
pub fn edit(src: &str, dst: &str, name: &str, rows: Option<&str>, cols: Option<&str>, op: EditOp, seed: u64) -> Result<usize> {
    let st = open(src)?;
    let s = st.as_ref();
    let target = s.weight(name)?;
    let (rr, cr) = (parse_range(rows, target.rows)?, parse_range(cols, target.cols)?);
    let mut w = AidoWriter::new();
    w.copy_meta(s);
    w.log(json!({"op": "edit", "tensor": name, "rows": [rr.start, rr.end], "cols": [cr.start, cr.end], "edit": format!("{op:?}"), "source": src}));
    let mut changed = 0;
    for n in s.names() {
        let (tid, dims) = s.info(&n).context("info")?;
        if n != name {
            w.add(&n, tid, dims, s.raw(&n)?.as_slice().to_vec());
            continue;
        }
        let (all, _, _) = read_slice(s, &n, 0..target.rows, 0..target.cols)?;
        let mut m = all;
        let sl: Vec<f32> = rr.clone().flat_map(|r| m[r * target.cols + cr.start..r * target.cols + cr.end].to_vec()).collect();
        let sd = stats(&sl).std as f32;
        let mut rng = crate::sdk::Rng(seed | 1);
        for r in rr.clone() {
            for c in cr.clone() {
                let v = &mut m[r * target.cols + c];
                let old = *v;
                *v = match op {
                    EditOp::Scale(k) => old * k,
                    EditOp::Set(k) => k,
                    EditOp::Add(k) => old + k,
                    EditOp::Noise(k) => old + rng.normal() * k * sd,
                    EditOp::Clamp(k) => old.clamp(-k * sd, k * sd),
                    EditOp::Prune(k) => if old.abs() < k * sd { 0.0 } else { old },
                };
                if *v != old {
                    changed += 1;
                }
            }
        }
        let ty = if can_quantize(target.ty) && target.cols % target.ty.block_elems() == 0 { target.ty } else { GgmlType::Q8_0 };
        let bytes: Vec<u8> = m.chunks(target.cols).flat_map(|row| quantize_row(ty, row)).collect();
        w.add(&n, crate::optimize::ty_id(ty), dims, bytes);
    }
    w.write(dst)?;
    Ok(changed)
}
