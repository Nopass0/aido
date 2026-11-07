//! Zero-copy reader for the GGUF model container (llama.cpp ecosystem).
//!
//! The file is memory-mapped; tensors are exposed as [`QWeight`] views into
//! the mapping, so "loading" an 8 B model takes milliseconds and the OS page
//! cache shares the weights between processes. Pages are faulted in lazily
//! or prefetched in parallel by [`GgufFile::prefetch`].

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

use crate::runtime::pool::ThreadPool;
use crate::tensor::quant::GgmlType;
use crate::tensor::{Bytes, QWeight};

/// A metadata value stored in a GGUF header.
#[derive(Debug, Clone)]
pub enum Value {
    /// Unsigned integer (u8/u16/u32/u64).
    U(u64),
    /// Signed integer (i8/i16/i32/i64).
    I(i64),
    /// Float (f32/f64).
    F(f64),
    /// Boolean.
    B(bool),
    /// UTF-8 string.
    S(String),
    /// Array of values.
    A(Vec<Value>),
}

impl Value {
    /// Interprets the value as `u64` if it is a (non-negative) integer.
    ///
    /// # Example
    /// ```
    /// use aido::gguf::Value;
    /// assert_eq!(Value::I(5).as_u64(), Some(5));
    /// assert_eq!(Value::S("x".into()).as_u64(), None);
    /// ```
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) if *v >= 0 => Some(*v as u64),
            Value::B(b) => Some(*b as u64),
            _ => None,
        }
    }

    /// Interprets the value as `f64`.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::gguf::Value::U(2).as_f64(), Some(2.0));
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Returns the string if the value is a string.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::gguf::Value::S("a".into()).as_str(), Some("a"));
    /// ```
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::S(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the array elements if the value is an array.
    ///
    /// # Example
    /// ```
    /// use aido::gguf::Value;
    /// assert_eq!(Value::A(vec![Value::U(1)]).as_array().unwrap().len(), 1);
    /// ```
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::A(v) => Some(v),
            _ => None,
        }
    }
}

/// Location and shape of one tensor inside the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions, fastest-varying first (`dims[0]` = columns).
    pub dims: Vec<usize>,
    /// Raw GGML type id.
    pub type_id: u32,
    /// Absolute byte offset in the file.
    pub offset: usize,
    /// Size in bytes (0 if the type is unsupported).
    pub nbytes: usize,
}

/// An opened, memory-mapped GGUF file.
pub struct GgufFile {
    /// Header key/value metadata.
    pub meta: HashMap<String, Value>,
    /// Tensor directory by name.
    pub tensors: HashMap<String, TensorInfo>,
    /// Tensor names in file order.
    pub order: Vec<String>,
    map: Arc<Mmap>,
}

/// Little-endian cursor over the header bytes.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("unexpected end of GGUF header");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U(self.take(1)?[0] as u64),
            1 => Value::I(self.take(1)?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Value::I(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::B(self.take(1)?[0] != 0),
            8 => Value::S(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 24));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::A(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_le_bytes(self.take(8)?.try_into()?)),
            t => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl GgufFile {
    /// Opens and parses a GGUF file (header only; tensor data stays mapped).
    ///
    /// * **Input** — path to a `.gguf` file (version 2 or 3).
    /// * **Output** — the parsed file or an error describing what is wrong.
    ///
    /// # Example
    /// ```no_run
    /// let f = aido::gguf::GgufFile::open("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// println!("{}", f.arch());
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let mut c = Cur { b: &map, p: 0 };
        if c.take(4)? != b"GGUF" {
            bail!("{} is not a GGUF file", path.display());
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_t = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = c.string()?;
            let ty = c.u32()?;
            let v = c.value(ty)?;
            meta.insert(k, v);
        }
        let mut infos = Vec::with_capacity(n_t);
        for _ in 0..n_t {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()? as usize);
            }
            let type_id = c.u32()?;
            let off = c.u64()? as usize;
            infos.push((name, dims, type_id, off));
        }
        let align = meta.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let data_start = c.p.div_ceil(align) * align;
        let mut tensors = HashMap::new();
        let mut order = Vec::new();
        for (name, dims, type_id, off) in infos {
            let n: usize = dims.iter().product();
            let nbytes = GgmlType::from_id(type_id).map(|t| n / t.block_elems() * t.block_bytes()).unwrap_or(0);
            let offset = data_start + off;
            if nbytes > 0 && offset + nbytes > map.len() {
                bail!("tensor {name} extends past end of file");
            }
            order.push(name.clone());
            tensors.insert(name.clone(), TensorInfo { name, dims, type_id, offset, nbytes });
        }
        Ok(GgufFile { meta, tensors, order, map: Arc::new(map) })
    }

    /// Model architecture (`general.architecture`), e.g. `"llama"`, `"qwen3"`.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// assert_eq!(f.arch(), "llama");
    /// ```
    pub fn arch(&self) -> &str {
        self.meta.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("llama")
    }

    /// Reads an integer metadata key, `None` if absent.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let layers = f.get_u64("llama.block_count").unwrap();
    /// ```
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(key).and_then(|v| v.as_u64())
    }

    /// Reads a float metadata key.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let eps = f.get_f64("llama.attention.layer_norm_rms_epsilon");
    /// ```
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.meta.get(key).and_then(|v| v.as_f64())
    }

    /// Reads a string metadata key.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let tmpl = f.get_str("tokenizer.chat_template");
    /// ```
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.meta.get(key).and_then(|v| v.as_str())
    }

    /// Returns a 2-D (or 1-D treated as `[1, n]`) weight view.
    ///
    /// * **Input** — tensor name.
    /// * **Output** — `QWeight` with `cols = dims[0]`, `rows = Π dims[1..]`;
    ///   error if missing or of an unsupported quantisation type.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let wq = f.weight("blk.0.attn_q.weight").unwrap();
    /// println!("{:?} {}x{}", wq.ty, wq.rows, wq.cols);
    /// ```
    pub fn weight(&self, name: &str) -> Result<QWeight> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_id(t.type_id)
            .with_context(|| format!("tensor {name}: quantisation {} is not supported yet", GgmlType::name_of(t.type_id)))?;
        let cols = t.dims[0];
        let rows: usize = t.dims[1..].iter().product::<usize>().max(1);
        let bytes = Bytes::from_owner(self.map.clone(), &self.map[t.offset..t.offset + t.nbytes]);
        QWeight::new(ty, rows, cols, bytes)
    }

    /// `true` if a tensor with this name exists.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let tied = !f.has("output.weight");
    /// ```
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Loads a 1-D tensor (norm gains, biases) as f32.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let g = f.vector("output_norm.weight").unwrap();
    /// ```
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let w = self.weight(name)?;
        let mut out = vec![0f32; w.rows * w.cols];
        for r in 0..w.rows {
            w.dequant_row(r, &mut out[r * w.cols..(r + 1) * w.cols]);
        }
        Ok(out)
    }

    /// Total bytes of tensor data in the file.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// println!("{:.2} GB", f.data_bytes() as f64 / 1e9);
    /// ```
    pub fn data_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.nbytes).sum()
    }

    /// Faults all tensor pages into RAM in parallel ("warm load").
    ///
    /// Reading one byte per 4 KiB page on all cores turns the cold-start of
    /// an 8 B model from many seconds of random page faults during the first
    /// tokens into one sequential, parallel pass at disk/page-cache speed.
    ///
    /// * **Input** — pool to use. **Output** — bytes touched.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::gguf::GgufFile::open("m.gguf").unwrap();
    /// let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// f.prefetch(&pool);
    /// ```
    pub fn prefetch(&self, pool: &ThreadPool) -> usize {
        let m: &[u8] = &self.map;
        let sink = std::sync::atomic::AtomicU64::new(0);
        pool.parallel_chunks(m.len(), 16 << 20, |s, e, _| {
            let mut acc = 0u64;
            let mut i = s;
            while i < e {
                acc = acc.wrapping_add(unsafe { std::ptr::read_volatile(&m[i]) } as u64);
                i += 4096;
            }
            sink.fetch_add(acc, std::sync::atomic::Ordering::Relaxed);
        });
        m.len()
    }
}
