//! A small embedded vector database.
//!
//! * [`Embedder`] turns text into a unit-length vector. The built-in
//!   [`HashEmbedder`] needs no model: it hashes word unigrams/bigrams and
//!   character 3–5-grams (any script: Russian, English, Japanese…) into a
//!   fixed number of dimensions with sublinear term frequency, which makes
//!   it robust to word forms ("кошка"/"кошку", "cat"/"cats") and typos.
//!   Any neural embedder can be plugged in through the trait.
//! * [`VectorIndex`] stores `(id, vector, payload)` rows and answers top-k
//!   cosine queries with an exact scan (vectors are L2-normalised, so the
//!   score is one dot product; 100k × 512 dims ≈ 10 ms on one core), and
//!   persists to a compact binary file.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Text → unit vector.
pub trait Embedder: Send + Sync {
    /// Output dimension.
    fn dim(&self) -> usize;
    /// Embeds one text (L2-normalised).
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Identifier stored with indexes (an index is only valid for its embedder).
    fn id(&self) -> String;
}

/// Model-free multilingual embedder (feature hashing of words and
/// character n-grams).
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    /// Dimensions (power of two recommended).
    pub dim: usize,
}

impl Default for HashEmbedder {
    /// 512 dimensions.
    fn default() -> Self {
        HashEmbedder { dim: 512 }
    }
}

fn fnv(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> String {
        format!("aido-hash-{}", self.dim)
    }

    /// # Example
    /// ```
    /// use aido::vector::{Embedder, HashEmbedder, cosine};
    /// let e = HashEmbedder::default();
    /// let a = e.embed("У меня есть кошка Мурка");
    /// let b = e.embed("как зовут мою кошку?");
    /// let c = e.embed("The weather in Paris is sunny");
    /// assert!(cosine(&a, &b) > cosine(&a, &c));
    /// ```
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let words = crate::memory::words(text);
        let mut add = |key: &str, seed: u64, w: f32| {
            let h = fnv(key.as_bytes(), seed);
            let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
            v[(h % self.dim as u64) as usize] += sign * w;
        };
        for (i, w) in words.iter().enumerate() {
            add(w, 1, 1.0);
            if i + 1 < words.len() {
                add(&format!("{w} {}", words[i + 1]), 2, 0.7);
            }
            let chars: Vec<char> = format!(" {w} ").chars().collect();
            for n in 3..=5 {
                if chars.len() < n {
                    continue;
                }
                for win in chars.windows(n) {
                    let g: String = win.iter().collect();
                    add(&g, 10 + n as u64, 0.35);
                }
            }
        }
        // sublinear tf: sign(x)·log(1+|x|), then L2 normalise
        for x in v.iter_mut() {
            *x = x.signum() * x.abs().ln_1p();
        }
        normalize(&mut v);
        v
    }
}

/// L2-normalises `v` in place (zero vectors stay zero).
///
/// # Example
/// ```
/// let mut v = vec![3.0, 4.0];
/// aido::vector::normalize(&mut v);
/// assert!((v[0] - 0.6).abs() < 1e-6);
/// ```
pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Cosine similarity of two unit vectors (their dot product).
///
/// # Example
/// ```
/// assert!((aido::vector::cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
/// ```
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One stored row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    /// Caller id.
    pub id: u64,
    /// Arbitrary payload (text, JSON…).
    pub payload: String,
    /// Unit vector.
    pub vector: Vec<f32>,
}

/// Exact-scan cosine index.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorIndex {
    /// Embedder id the vectors come from.
    pub embedder: String,
    /// Dimension.
    pub dim: usize,
    /// Rows.
    pub rows: Vec<Row>,
}

impl VectorIndex {
    /// Empty index for vectors of `dim` from embedder `embedder`.
    ///
    /// # Example
    /// ```
    /// let idx = aido::vector::VectorIndex::new("aido-hash-512", 512);
    /// assert!(idx.rows.is_empty());
    /// ```
    pub fn new(embedder: &str, dim: usize) -> Self {
        VectorIndex { embedder: embedder.into(), dim, rows: Vec::new() }
    }

    /// Adds (or replaces, by id) a row.
    ///
    /// # Example
    /// ```
    /// use aido::vector::*;
    /// let e = HashEmbedder::default();
    /// let mut idx = VectorIndex::new(&e.id(), e.dim());
    /// idx.upsert(1, "hello", e.embed("hello"));
    /// assert_eq!(idx.rows.len(), 1);
    /// ```
    pub fn upsert(&mut self, id: u64, payload: &str, vector: Vec<f32>) {
        self.rows.retain(|r| r.id != id);
        self.rows.push(Row { id, payload: payload.into(), vector });
    }

    /// Removes a row; returns whether it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        let n = self.rows.len();
        self.rows.retain(|r| r.id != id);
        n != self.rows.len()
    }

    /// The `k` rows most similar to `q`, best first, as `(score, row)`.
    ///
    /// # Example
    /// ```
    /// use aido::vector::*;
    /// let e = HashEmbedder::default();
    /// let mut idx = VectorIndex::new(&e.id(), e.dim());
    /// idx.upsert(1, "Paris is the capital of France", e.embed("Paris is the capital of France"));
    /// idx.upsert(2, "Rust is a programming language", e.embed("Rust is a programming language"));
    /// let hits = idx.search(&e.embed("what is the capital of France"), 1);
    /// assert_eq!(hits[0].1.id, 1);
    /// ```
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(f32, &Row)> {
        let mut s: Vec<(f32, &Row)> = self.rows.iter().map(|r| (cosine(q, &r.vector), r)).collect();
        let k = k.min(s.len());
        if k < s.len() {
            s.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
            s.truncate(k);
        }
        s.sort_by(|a, b| b.0.total_cmp(&a.0));
        s
    }

    /// Saves to a binary file (JSON header line + little-endian f32 vectors).
    ///
    /// # Example
    /// ```
    /// let p = std::env::temp_dir().join("aido_vec_doc.bin");
    /// let mut idx = aido::vector::VectorIndex::new("x", 2);
    /// idx.upsert(7, "seven", vec![1.0, 0.0]);
    /// idx.save(&p).unwrap();
    /// assert_eq!(aido::vector::VectorIndex::load(&p).unwrap().rows[0].payload, "seven");
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        let meta: Vec<(u64, &str)> = self.rows.iter().map(|r| (r.id, r.payload.as_str())).collect();
        let mut out = serde_json::to_vec(&serde_json::json!({"embedder": self.embedder, "dim": self.dim, "rows": meta}))?;
        out.push(b'\n');
        for r in &self.rows {
            for x in &r.vector {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    /// Loads a file written by [`VectorIndex::save`].
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path)?;
        let nl = raw.iter().position(|b| *b == b'\n').context("bad index file")?;
        let h: serde_json::Value = serde_json::from_slice(&raw[..nl])?;
        let dim = h["dim"].as_u64().unwrap_or(0) as usize;
        let mut rows = Vec::new();
        let mut off = nl + 1;
        for m in h["rows"].as_array().context("rows")? {
            let v: Vec<f32> = raw[off..off + dim * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            off += dim * 4;
            rows.push(Row { id: m[0].as_u64().unwrap_or(0), payload: m[1].as_str().unwrap_or("").into(), vector: v });
        }
        Ok(VectorIndex { embedder: h["embedder"].as_str().unwrap_or("").into(), dim, rows })
    }
}
