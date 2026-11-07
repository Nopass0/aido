//! Zero-copy reader for `safetensors` files and multi-file directories
//! (Hugging Face / diffusers / colibri containers).
//!
//! Every file is memory-mapped once; tensors are exposed through the common
//! [`TensorStore`] interface. Metadata (`config.json`, `qwen36_meta.json`,
//! …) found in the directory is flattened into the metadata dictionary with
//! the file stem as prefix (`config.hidden_size`, `qwen36_meta.topk`, …).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

use super::{json_to_value, TensorStore};
use crate::gguf::Value;
use crate::runtime::pool::ThreadPool;
use crate::tensor::{Bytes, QWeight};
use crate::tensor::quant::GgmlType;

/// Pseudo GGML type id used for raw `U8`/`I8` tensors (not directly usable
/// as a matrix; read with [`TensorStore::raw`]).
pub const TYPE_U8: u32 = 1000;

/// One tensor entry.
#[derive(Debug, Clone)]
struct Entry {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A set of memory-mapped safetensors files.
pub struct SafetensorsDir {
    /// Directory (or file) the store was opened from.
    pub path: PathBuf,
    maps: Vec<Arc<Mmap>>,
    entries: HashMap<String, Entry>,
    order: Vec<String>,
    meta: HashMap<String, Value>,
}

impl SafetensorsDir {
    /// Opens one `.safetensors` file or every `*.safetensors` in a directory.
    ///
    /// * **Input** — file or directory path.
    /// * **Output** — the store; JSON side files in the directory are merged
    ///   into the metadata.
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{safetensors::SafetensorsDir, TensorStore};
    /// let s = SafetensorsDir::open("models/sd-turbo/vae").unwrap();
    /// println!("{} tensors", s.names().len());
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut files: Vec<PathBuf> = if path.is_dir() {
            std::fs::read_dir(&path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect()
        } else {
            vec![path.clone()]
        };
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", path.display());
        }
        let mut maps = Vec::new();
        let mut entries = HashMap::new();
        let mut order = Vec::new();
        for (fi, f) in files.iter().enumerate() {
            let file = File::open(f).with_context(|| format!("open {}", f.display()))?;
            let map = unsafe { Mmap::map(&file)? };
            let n = u64::from_le_bytes(map[0..8].try_into()?) as usize;
            let header: serde_json::Value = serde_json::from_slice(&map[8..8 + n]).with_context(|| format!("bad header in {}", f.display()))?;
            let base = 8 + n;
            if let Some(obj) = header.as_object() {
                for (name, v) in obj {
                    if name == "__metadata__" {
                        continue;
                    }
                    let offs = v["data_offsets"].as_array().context("data_offsets")?;
                    let e = Entry {
                        file: fi,
                        dtype: v["dtype"].as_str().unwrap_or("").to_string(),
                        shape: v["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default(),
                        start: base + offs[0].as_u64().unwrap_or(0) as usize,
                        end: base + offs[1].as_u64().unwrap_or(0) as usize,
                    };
                    if e.end > map.len() {
                        bail!("tensor {name} out of bounds in {}", f.display());
                    }
                    order.push(name.clone());
                    entries.insert(name.clone(), e);
                }
            }
            maps.push(Arc::new(map));
        }
        // Side-car JSON metadata.
        let mut meta = HashMap::new();
        let dir = if path.is_dir() { path.clone() } else { path.parent().map(|p| p.to_path_buf()).unwrap_or_default() };
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                if p.extension().is_some_and(|x| x == "json") && stem != "tokenizer" && std::fs::metadata(&p).map(|m| m.len() < 1 << 20).unwrap_or(false) {
                    if let Ok(j) = serde_json::from_slice::<serde_json::Value>(&std::fs::read(&p)?) {
                        if let Some(o) = j.as_object() {
                            for (k, v) in o {
                                meta.insert(format!("{stem}.{k}"), json_to_value(v));
                            }
                        }
                    }
                }
            }
        }
        Ok(SafetensorsDir { path, maps, entries, order, meta })
    }

    /// Dtype string (`F16`, `BF16`, `F32`, `U8`, …) of a tensor.
    ///
    /// # Example
    /// ```no_run
    /// # let s = aido::format::safetensors::SafetensorsDir::open("d").unwrap();
    /// assert_eq!(s.dtype("lm_head.weight"), Some("F16"));
    /// ```
    pub fn dtype(&self, name: &str) -> Option<&str> {
        self.entries.get(name).map(|e| e.dtype.as_str())
    }

    /// Shape (PyTorch order, slowest first) of a tensor.
    ///
    /// # Example
    /// ```no_run
    /// # let s = aido::format::safetensors::SafetensorsDir::open("d").unwrap();
    /// let shape = s.shape("lm_head.weight").unwrap();
    /// ```
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// Returns a tensor converted to f32 (F32/F16/BF16 sources).
    ///
    /// # Example
    /// ```no_run
    /// # let s = aido::format::safetensors::SafetensorsDir::open("d").unwrap();
    /// let v = s.f32("model.norm.weight").unwrap();
    /// ```
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entries.get(name).with_context(|| format!("missing tensor {name}"))?;
        let b = &self.maps[e.file][e.start..e.end];
        Ok(match e.dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            "BF16" => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            d => bail!("tensor {name}: cannot convert {d} to f32"),
        })
    }
}

/// Maps a safetensors dtype to a GGML type id.
fn type_of(dtype: &str) -> u32 {
    match dtype {
        "F32" => 0,
        "F16" => 1,
        "BF16" => 30,
        _ => TYPE_U8,
    }
}

impl TensorStore for SafetensorsDir {
    fn meta(&self) -> &HashMap<String, Value> {
        &self.meta
    }
    fn names(&self) -> Vec<String> {
        self.order.clone()
    }
    fn info(&self, name: &str) -> Option<(u32, Vec<usize>)> {
        // Report dims fastest-first like GGUF.
        self.entries.get(name).map(|e| (type_of(&e.dtype), e.shape.iter().rev().cloned().collect()))
    }
    fn weight(&self, name: &str) -> Result<QWeight> {
        let e = self.entries.get(name).with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_id(type_of(&e.dtype)).with_context(|| format!("{name}: dtype {} is not a float matrix", e.dtype))?;
        // Trailing unit dims (1x1 / k=1 convolution kernels) are squeezed.
        let mut shape = e.shape.clone();
        while shape.len() > 2 && *shape.last().unwrap() == 1 {
            shape.pop();
        }
        let cols = *shape.last().unwrap_or(&1);
        let rows = shape.iter().rev().skip(1).product::<usize>().max(1);
        QWeight::new(ty, rows, cols, self.raw(name)?)
    }
    fn raw(&self, name: &str) -> Result<Bytes> {
        let e = self.entries.get(name).with_context(|| format!("missing tensor {name}"))?;
        let m = &self.maps[e.file];
        Ok(Bytes::from_owner(m.clone(), &m[e.start..e.end]))
    }
    fn prefetch(&self, pool: &ThreadPool) -> usize {
        let mut total = 0;
        for m in &self.maps {
            let m: &[u8] = m;
            pool.parallel_chunks(m.len(), 16 << 20, |s, e, _| {
                let mut i = s;
                let mut acc = 0u8;
                while i < e {
                    acc ^= unsafe { std::ptr::read_volatile(&m[i]) };
                    i += 4096;
                }
                std::hint::black_box(acc);
            });
            total += m.len();
        }
        total
    }
}
