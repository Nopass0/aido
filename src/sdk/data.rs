//! `.aidata` — memory-mapped dataset container.
//!
//! ```text
//! "AIDATA01" | u64 header_len | JSON header | padding to 64 | payload
//! ```
//!
//! Two kinds:
//! * `tokens` — one `u32` token stream (language-model pre-training /
//!   fine-tuning; random windows are sampled from it);
//! * `tensors` — `count` records of `dim` f32 features plus a `u32` label
//!   (classification, regression, image datasets).
//!
//! The payload is 64-byte aligned and read through a memory map, so datasets
//! larger than RAM stream from disk with no loading step.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Result};
use memmap2::Mmap;
use serde_json::{json, Value};

/// An opened `.aidata` file.
pub struct DataSet {
    map: Mmap,
    /// Header (kind, count, dim, extra user metadata).
    pub header: Value,
    off: usize,
}

/// Writes a token-stream dataset.
///
/// # Example
/// ```
/// let p = std::env::temp_dir().join("doc_tokens.aidata");
/// aido::sdk::data::write_tokens(&p, &[1, 2, 3], serde_json::json!({"tokenizer": "bytes"})).unwrap();
/// let d = aido::sdk::data::DataSet::open(&p).unwrap();
/// assert_eq!(d.tokens(), &[1, 2, 3]);
/// ```
pub fn write_tokens(path: impl AsRef<Path>, tokens: &[u32], meta: Value) -> Result<()> {
    write(path, json!({"kind": "tokens", "count": tokens.len(), "meta": meta}), bytemuck::cast_slice(tokens))
}

/// Writes a feature/label dataset (`features.len() == labels.len() * dim`).
///
/// # Example
/// ```
/// let p = std::env::temp_dir().join("doc_tensors.aidata");
/// aido::sdk::data::write_tensors(&p, &[1.0, 2.0, 3.0, 4.0], 2, &[0, 1], serde_json::json!({})).unwrap();
/// let d = aido::sdk::data::DataSet::open(&p).unwrap();
/// assert_eq!(d.features(1), &[3.0, 4.0]);
/// assert_eq!(d.label(1), 1);
/// ```
pub fn write_tensors(path: impl AsRef<Path>, features: &[f32], dim: usize, labels: &[u32], meta: Value) -> Result<()> {
    anyhow::ensure!(features.len() == labels.len() * dim, "features/labels size mismatch");
    let mut payload: Vec<u8> = bytemuck::cast_slice(features).to_vec();
    payload.extend_from_slice(bytemuck::cast_slice(labels));
    write(path, json!({"kind": "tensors", "count": labels.len(), "dim": dim, "meta": meta}), &payload)
}

fn write(path: impl AsRef<Path>, header: Value, payload: &[u8]) -> Result<()> {
    let h = serde_json::to_vec(&header)?;
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(b"AIDATA01")?;
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    let pos = 16 + h.len();
    f.write_all(&vec![0u8; (64 - pos % 64) % 64])?;
    f.write_all(payload)?;
    Ok(())
}

impl DataSet {
    /// Opens a dataset.
    ///
    /// # Example
    /// ```no_run
    /// let d = aido::sdk::data::DataSet::open("data/alice.aidata").unwrap();
    /// println!("{}", d.len());
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let map = unsafe { Mmap::map(&File::open(path)?)? };
        if map.len() < 16 || &map[..8] != b"AIDATA01" {
            bail!("not an .aidata file");
        }
        let hl = u64::from_le_bytes(map[8..16].try_into()?) as usize;
        let header: Value = serde_json::from_slice(&map[16..16 + hl])?;
        let off = (16 + hl).div_ceil(64) * 64;
        Ok(DataSet { map, header, off })
    }

    /// Number of tokens (token kind) or records (tensor kind).
    ///
    /// # Example
    /// ```no_run
    /// # let d = aido::sdk::data::DataSet::open("x.aidata").unwrap();
    /// let n = d.len();
    /// ```
    pub fn len(&self) -> usize {
        self.header["count"].as_u64().unwrap_or(0) as usize
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Feature width (tensor kind).
    ///
    /// # Example
    /// ```no_run
    /// # let d = aido::sdk::data::DataSet::open("x.aidata").unwrap();
    /// let dim = d.dim();
    /// ```
    pub fn dim(&self) -> usize {
        self.header["dim"].as_u64().unwrap_or(0) as usize
    }

    /// The token stream (token kind).
    ///
    /// # Example
    /// ```no_run
    /// # let d = aido::sdk::data::DataSet::open("x.aidata").unwrap();
    /// let t = d.tokens();
    /// ```
    pub fn tokens(&self) -> &[u32] {
        bytemuck::cast_slice(&self.map[self.off..self.off + self.len() * 4])
    }

    /// Features of record `i` (tensor kind).
    ///
    /// # Example
    /// ```no_run
    /// # let d = aido::sdk::data::DataSet::open("x.aidata").unwrap();
    /// let x = d.features(0);
    /// ```
    pub fn features(&self, i: usize) -> &[f32] {
        let d = self.dim();
        let all: &[f32] = bytemuck::cast_slice(&self.map[self.off..self.off + self.len() * d * 4]);
        &all[i * d..(i + 1) * d]
    }

    /// Label of record `i` (tensor kind).
    ///
    /// # Example
    /// ```no_run
    /// # let d = aido::sdk::data::DataSet::open("x.aidata").unwrap();
    /// let y = d.label(0);
    /// ```
    pub fn label(&self, i: usize) -> u32 {
        let base = self.off + self.len() * self.dim() * 4;
        u32::from_le_bytes(self.map[base + i * 4..base + i * 4 + 4].try_into().unwrap())
    }
}
