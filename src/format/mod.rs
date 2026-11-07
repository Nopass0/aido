//! Model containers: a common [`TensorStore`] interface over GGUF files and
//! the engine's own **`.aido`** format.
//!
//! ## The `.aido` format (version 1)
//!
//! Designed from the measurements in `docs/RESEARCH.md`:
//!
//! ```text
//! offset 0      "AIDO" | u32 version | u64 header_offset | u64 header_len | u64 data_offset
//! offset 4096   tensor data, each tensor 4096-byte (page) aligned,
//!               stored in EXECUTION ORDER (layer 0 … layer N, head, then
//!               cold tensors such as the embedding table)
//! header_offset UTF-8 JSON: { "meta": {...}, "tensors": [{name, type, dims,
//!               offset, nbytes, fnv64}], "provenance": [...] }
//! ```
//!
//! * **Page alignment** → every tensor can be memory-mapped, uploaded to
//!   the GPU or prefetched with whole-page granularity, no partial pages
//!   shared between hot and cold tensors.
//! * **Execution order** → decoding streams the file front-to-back; the OS
//!   read-ahead and the hardware prefetcher see one sequential stream, and
//!   a partially-offloaded model keeps its GPU part contiguous.
//! * **Cold tensors last** → only one embedding row is read per token, so
//!   the table sits at the end and is not prefetched with the hot set.
//! * **Per-tensor FNV-64 checksums** → `aido verify` detects corruption.
//! * **Provenance log** → every transformation (quantisation, pruning,
//!   vocabulary trimming, layer removal, training run) is recorded.
//! * **Same tensor names / metadata keys as GGUF** → one model code path,
//!   and the format also stores SDK networks and training checkpoints
//!   (f32 weights + optimizer state).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

pub mod safetensors;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde_json::{json, Value as Json};

use crate::gguf::{GgufFile, Value};
use crate::runtime::pool::ThreadPool;
use crate::tensor::quant::GgmlType;
use crate::tensor::{Bytes, QWeight};

/// Alignment of tensor data in `.aido` files.
pub const ALIGN: usize = 4096;

/// Uniform read access to a model file (GGUF or `.aido`).
pub trait TensorStore: Send + Sync {
    /// Metadata dictionary (GGUF-compatible keys).
    fn meta(&self) -> &HashMap<String, Value>;
    /// Tensor names in file order.
    fn names(&self) -> Vec<String>;
    /// `(ggml_type_id, dims)` of a tensor.
    fn info(&self, name: &str) -> Option<(u32, Vec<usize>)>;
    /// Weight view of a tensor.
    fn weight(&self, name: &str) -> Result<QWeight>;
    /// Raw bytes of a tensor.
    fn raw(&self, name: &str) -> Result<Bytes>;
    /// Pre-faults all pages; returns bytes touched.
    fn prefetch(&self, pool: &ThreadPool) -> usize;

    /// `true` if the tensor exists.
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, TensorStore};
    /// let f = open("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// assert!(f.has("token_embd.weight"));
    /// ```
    fn has(&self, name: &str) -> bool {
        self.info(name).is_some()
    }

    /// Loads a tensor fully as f32 (norms, biases, small tables).
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, TensorStore};
    /// let f = open("m.gguf").unwrap();
    /// let g = f.vector("output_norm.weight").unwrap();
    /// ```
    fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let w = self.weight(name)?;
        let mut out = vec![0f32; w.rows * w.cols];
        for r in 0..w.rows {
            w.dequant_row(r, &mut out[r * w.cols..(r + 1) * w.cols]);
        }
        Ok(out)
    }

    /// Integer metadata value.
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, TensorStore};
    /// let n = open("m.gguf").unwrap().get_u64("qwen3.block_count");
    /// ```
    fn get_u64(&self, key: &str) -> Option<u64> {
        self.meta().get(key).and_then(|v| v.as_u64())
    }

    /// Float metadata value.
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, TensorStore};
    /// let eps = open("m.gguf").unwrap().get_f64("llama.attention.layer_norm_rms_epsilon");
    /// ```
    fn get_f64(&self, key: &str) -> Option<f64> {
        self.meta().get(key).and_then(|v| v.as_f64())
    }

    /// String metadata value.
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, TensorStore};
    /// let f = open("m.gguf").unwrap();
    /// let tpl = f.get_str("tokenizer.chat_template");
    /// ```
    fn get_str(&self, key: &str) -> Option<&str> {
        self.meta().get(key).and_then(|v| v.as_str())
    }

    /// `general.architecture` (default `llama`).
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, TensorStore};
    /// println!("{}", open("m.gguf").unwrap().arch());
    /// ```
    fn arch(&self) -> &str {
        self.get_str("general.architecture").unwrap_or("llama")
    }
}

impl TensorStore for GgufFile {
    fn meta(&self) -> &HashMap<String, Value> {
        &self.meta
    }
    fn names(&self) -> Vec<String> {
        self.order.clone()
    }
    fn info(&self, name: &str) -> Option<(u32, Vec<usize>)> {
        self.tensors.get(name).map(|t| (t.type_id, t.dims.clone()))
    }
    fn weight(&self, name: &str) -> Result<QWeight> {
        GgufFile::weight(self, name)
    }
    fn raw(&self, name: &str) -> Result<Bytes> {
        Ok(GgufFile::weight(self, name)?.data)
    }
    fn prefetch(&self, pool: &ThreadPool) -> usize {
        GgufFile::prefetch(self, pool)
    }
}

/// Opens a GGUF or `.aido` file, detected by its magic bytes.
///
/// * **Input** — path. **Output** — boxed store.
///
/// # Example
/// ```no_run
/// let f = aido::format::open("models/model.aido").unwrap();
/// ```
pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn TensorStore>> {
    let path = path.as_ref();
    let mut magic = [0u8; 4];
    {
        use std::io::Read;
        File::open(path).with_context(|| format!("open {}", path.display()))?.read_exact(&mut magic)?;
    }
    match &magic {
        b"GGUF" => Ok(Box::new(GgufFile::open(path)?)),
        b"AIDO" => Ok(Box::new(AidoFile::open(path)?)),
        _ => bail!("{}: unknown model format", path.display()),
    }
}

/// Converts a metadata value to JSON.
///
/// # Example
/// ```
/// use aido::{gguf::Value, format::value_to_json};
/// assert_eq!(value_to_json(&Value::U(3)), serde_json::json!(3));
/// ```
pub fn value_to_json(v: &Value) -> Json {
    match v {
        Value::U(x) => json!(x),
        Value::I(x) => json!(x),
        Value::F(x) => json!(x),
        Value::B(x) => json!(x),
        Value::S(x) => json!(x),
        Value::A(a) => Json::Array(a.iter().map(value_to_json).collect()),
    }
}

/// Converts JSON back to a metadata value.
///
/// # Example
/// ```
/// use aido::format::json_to_value;
/// assert_eq!(json_to_value(&serde_json::json!(2.5)).as_f64(), Some(2.5));
/// ```
pub fn json_to_value(j: &Json) -> Value {
    match j {
        Json::Null => Value::S(String::new()),
        Json::Bool(b) => Value::B(*b),
        Json::Number(n) => {
            if let Some(u) = n.as_u64() {
                Value::U(u)
            } else if let Some(i) = n.as_i64() {
                Value::I(i)
            } else {
                Value::F(n.as_f64().unwrap_or(0.0))
            }
        }
        Json::String(s) => Value::S(s.clone()),
        Json::Array(a) => Value::A(a.iter().map(json_to_value).collect()),
        Json::Object(_) => Value::S(j.to_string()),
    }
}

/// FNV-1a 64-bit hash (fast, dependency-free integrity check).
///
/// # Example
/// ```
/// assert_eq!(aido::format::fnv64(b""), 0xcbf29ce484222325);
/// ```
pub fn fnv64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Directory entry of a tensor in an `.aido` file.
#[derive(Debug, Clone)]
pub struct AidoTensor {
    /// GGML type id.
    pub type_id: u32,
    /// Dimensions (fastest first, like GGUF).
    pub dims: Vec<usize>,
    /// Absolute byte offset.
    pub offset: usize,
    /// Byte length.
    pub nbytes: usize,
    /// FNV-64 checksum.
    pub fnv: u64,
}

/// A memory-mapped `.aido` file.
pub struct AidoFile {
    meta: HashMap<String, Value>,
    tensors: HashMap<String, AidoTensor>,
    order: Vec<String>,
    /// Transformation history.
    pub provenance: Vec<Json>,
    map: Arc<Mmap>,
}

impl AidoFile {
    /// Opens and validates the header of an `.aido` file.
    ///
    /// # Example
    /// ```no_run
    /// let f = aido::format::AidoFile::open("models/qwen-trim.aido").unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 32 || &map[0..4] != b"AIDO" {
            bail!("not an .aido file");
        }
        let ver = u32::from_le_bytes(map[4..8].try_into()?);
        if ver != 1 {
            bail!("unsupported .aido version {ver}");
        }
        let hoff = u64::from_le_bytes(map[8..16].try_into()?) as usize;
        let hlen = u64::from_le_bytes(map[16..24].try_into()?) as usize;
        let header: Json = serde_json::from_slice(&map[hoff..hoff + hlen]).context("corrupt .aido header")?;
        let meta = header["meta"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect()).unwrap_or_default();
        let mut tensors = HashMap::new();
        let mut order = Vec::new();
        for t in header["tensors"].as_array().context("no tensor table")? {
            let name = t["name"].as_str().context("tensor name")?.to_string();
            let e = AidoTensor {
                type_id: t["type"].as_u64().context("type")? as u32,
                dims: t["dims"].as_array().context("dims")?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect(),
                offset: t["offset"].as_u64().context("offset")? as usize,
                nbytes: t["nbytes"].as_u64().context("nbytes")? as usize,
                fnv: u64::from_str_radix(t["fnv64"].as_str().unwrap_or("0"), 16).unwrap_or(0),
            };
            if e.offset + e.nbytes > map.len() {
                bail!("tensor {name} out of bounds");
            }
            order.push(name.clone());
            tensors.insert(name, e);
        }
        let provenance = header["provenance"].as_array().cloned().unwrap_or_default();
        Ok(AidoFile { meta, tensors, order, provenance, map: Arc::new(map) })
    }

    /// Recomputes all checksums; returns names of corrupted tensors.
    ///
    /// # Example
    /// ```no_run
    /// let f = aido::format::AidoFile::open("m.aido").unwrap();
    /// assert!(f.verify().is_empty());
    /// ```
    pub fn verify(&self) -> Vec<String> {
        self.order
            .iter()
            .filter(|n| {
                let t = &self.tensors[*n];
                fnv64(&self.map[t.offset..t.offset + t.nbytes]) != t.fnv
            })
            .cloned()
            .collect()
    }
}

impl TensorStore for AidoFile {
    fn meta(&self) -> &HashMap<String, Value> {
        &self.meta
    }
    fn names(&self) -> Vec<String> {
        self.order.clone()
    }
    fn info(&self, name: &str) -> Option<(u32, Vec<usize>)> {
        self.tensors.get(name).map(|t| (t.type_id, t.dims.clone()))
    }
    fn weight(&self, name: &str) -> Result<QWeight> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_id(t.type_id).with_context(|| format!("unsupported type {}", t.type_id))?;
        let cols = t.dims[0];
        let rows = t.dims[1..].iter().product::<usize>().max(1);
        QWeight::new(ty, rows, cols, self.raw(name)?)
    }
    fn raw(&self, name: &str) -> Result<Bytes> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(Bytes::from_owner(self.map.clone(), &self.map[t.offset..t.offset + t.nbytes]))
    }
    fn prefetch(&self, pool: &ThreadPool) -> usize {
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

/// Tensor data source for the writer (produced lazily to bound memory).
pub type TensorProducer = Box<dyn FnOnce() -> Result<Vec<u8>> + Send>;

/// Streaming writer for `.aido` files.
pub struct AidoWriter {
    meta: serde_json::Map<String, Json>,
    tensors: Vec<(String, u32, Vec<usize>, TensorProducer)>,
    provenance: Vec<Json>,
}

impl Default for AidoWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl AidoWriter {
    /// Creates an empty writer.
    ///
    /// # Example
    /// ```
    /// let w = aido::format::AidoWriter::new();
    /// ```
    pub fn new() -> Self {
        AidoWriter { meta: Default::default(), tensors: Vec::new(), provenance: Vec::new() }
    }

    /// Copies all metadata of an existing store.
    ///
    /// # Example
    /// ```no_run
    /// use aido::format::{open, AidoWriter};
    /// let src = open("m.gguf").unwrap();
    /// let mut w = AidoWriter::new();
    /// w.copy_meta(src.as_ref());
    /// ```
    pub fn copy_meta(&mut self, src: &dyn TensorStore) {
        for (k, v) in src.meta() {
            self.meta.insert(k.clone(), value_to_json(v));
        }
    }

    /// Sets a metadata key.
    ///
    /// # Example
    /// ```
    /// let mut w = aido::format::AidoWriter::new();
    /// w.set_meta("general.name", serde_json::json!("demo"));
    /// ```
    pub fn set_meta(&mut self, key: &str, v: Json) {
        self.meta.insert(key.to_string(), v);
    }

    /// Removes a metadata key (e.g. copied from a source file).
    ///
    /// # Example
    /// ```
    /// let mut w = aido::format::AidoWriter::new();
    /// w.set_meta("a", serde_json::json!(1));
    /// w.remove_meta("a");
    /// ```
    pub fn remove_meta(&mut self, key: &str) {
        self.meta.remove(key);
    }

    /// Appends an entry to the provenance log.
    ///
    /// # Example
    /// ```
    /// let mut w = aido::format::AidoWriter::new();
    /// w.log(serde_json::json!({"op": "quantize", "to": "q4_k"}));
    /// ```
    pub fn log(&mut self, entry: Json) {
        self.provenance.push(entry);
    }

    /// Adds a tensor (written in call order) with in-memory data.
    ///
    /// # Example
    /// ```
    /// let mut w = aido::format::AidoWriter::new();
    /// w.add("bias", 0, vec![4], vec![0u8; 16]);
    /// ```
    pub fn add(&mut self, name: &str, type_id: u32, dims: Vec<usize>, data: Vec<u8>) {
        self.tensors.push((name.to_string(), type_id, dims, Box::new(move || Ok(data))));
    }

    /// Adds a tensor whose bytes are produced when the file is written.
    ///
    /// # Example
    /// ```
    /// let mut w = aido::format::AidoWriter::new();
    /// w.add_lazy("w", 0, vec![2], Box::new(|| Ok(vec![0u8; 8])));
    /// ```
    pub fn add_lazy(&mut self, name: &str, type_id: u32, dims: Vec<usize>, f: TensorProducer) {
        self.tensors.push((name.to_string(), type_id, dims, f));
    }

    /// Writes the file.
    ///
    /// * **Output** — total bytes written.
    ///
    /// # Example
    /// ```
    /// let dir = std::env::temp_dir().join("aido_doc_test.aido");
    /// let mut w = aido::format::AidoWriter::new();
    /// w.add("x", 0, vec![2], [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect());
    /// w.write(&dir).unwrap();
    /// let f = aido::format::AidoFile::open(&dir).unwrap();
    /// use aido::format::TensorStore;
    /// assert_eq!(f.vector("x").unwrap(), vec![1.0, 2.0]);
    /// ```
    pub fn write(self, path: impl AsRef<Path>) -> Result<u64> {
        let mut f = BufWriter::with_capacity(8 << 20, File::create(path.as_ref())?);
        let mut head = [0u8; ALIGN];
        head[0..4].copy_from_slice(b"AIDO");
        head[4..8].copy_from_slice(&1u32.to_le_bytes());
        f.write_all(&head)?;
        let mut pos = ALIGN as u64;
        let mut table = Vec::new();
        for (name, ty, dims, prod) in self.tensors {
            let data = prod()?;
            let off = pos;
            f.write_all(&data)?;
            pos += data.len() as u64;
            let pad = (ALIGN as u64 - pos % ALIGN as u64) % ALIGN as u64;
            f.write_all(&vec![0u8; pad as usize])?;
            pos += pad;
            table.push(json!({"name": name, "type": ty, "dims": dims, "offset": off, "nbytes": data.len(), "fnv64": format!("{:016x}", fnv64(&data))}));
        }
        let header = serde_json::to_vec(&json!({"meta": Json::Object(self.meta), "tensors": table, "provenance": self.provenance}))?;
        f.write_all(&header)?;
        let total = pos + header.len() as u64;
        let mut file = f.into_inner().map_err(|e| anyhow::anyhow!("{e}"))?;
        file.seek(SeekFrom::Start(8))?;
        file.write_all(&pos.to_le_bytes())?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&(ALIGN as u64).to_le_bytes())?;
        file.flush()?;
        Ok(total)
    }
}
