//! # Model I/O signatures for SDK models
//!
//! A model built with the SDK carries a **signature** in its `.aido` file
//! (metadata key `aido.io`): what goes in, what comes out, in which format,
//! and how values are turned into tokens/tensors and back (the *codec*).
//! With a signature the same file can be
//!
//! * inspected: `aido model-info models/seq.aido`;
//! * run from the CLI with JSON: `aido predict -m models/seq.aido -i '{"sequence":[2,4,6,8]}'`;
//! * served: `aido serve -m seq=models/seq.aido` → `POST /v1/predict`;
//! * embedded: `aido::Models::predict("seq", &json!({...}))`;
//! * compiled for inference: `aido compile -m models/seq.aido -o models/seq-f16.aido --quant f16`.
//!
//! ```json
//! {
//!   "task": "sequence-next",
//!   "description": "Predicts the next integer of a sequence",
//!   "inputs":  [{"name": "sequence", "type": "int[]", "description": "at least 3 numbers"}],
//!   "outputs": [{"name": "next", "type": "int"}],
//!   "codec": {"kind": "char-lm", "vocab": ["0","1",…,"9",","," ","-","|","\n"],
//!             "bos": 14, "stop": 14, "template": "{sequence}|", "join": ", ", "max_new": 12},
//!   "examples": [{"input": {"sequence": [2, 4, 6, 8]}, "output": {"next": 10}}]
//! }
//! ```
//!
//! Field types: `int`, `float`, `text`, `bool`, `int[]`, `float[]`, `text[]`,
//! `label` (one of `labels`), `image` (`shape: [h, w, c]`), `json`.
//!
//! Codecs:
//!
//! | `kind` | Model | Input → tokens | Tokens → output |
//! |---|---|---|---|
//! | `char-lm` | [`super::models::LmModel`] with a character vocabulary | fill `template` (`{field}`; arrays joined by `join`), map characters | greedy decode until `stop`, parse as the output type |
//! | `byte-lm` | byte-level `LmModel` (vocab 257) | UTF-8 bytes of `template` | greedy decode, UTF-8 text |

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::format::{AidoFile, AidoWriter, TensorStore};
use crate::runtime::pool::ThreadPool;

/// One input or output field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Field {
    /// Name (JSON key).
    pub name: String,
    /// Type: `int`, `float`, `text`, `bool`, `int[]`, `float[]`, `text[]`, `label`, `image`, `json`.
    #[serde(rename = "type")]
    pub ty: String,
    /// Human description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Allowed values for `label`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    /// Shape for tensors / images.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<usize>,
}

impl Field {
    /// Convenience constructor.
    ///
    /// # Example
    /// ```
    /// let f = aido::sdk::io::Field::new("x", "int[]", "numbers");
    /// assert_eq!(f.ty, "int[]");
    /// ```
    pub fn new(name: &str, ty: &str, description: &str) -> Self {
        Field { name: name.into(), ty: ty.into(), description: description.into(), labels: vec![], shape: vec![] }
    }

    /// Checks that `v` has this field's type.
    ///
    /// # Example
    /// ```
    /// use aido::sdk::io::Field;
    /// assert!(Field::new("x", "int[]", "").check(&serde_json::json!([1, 2])).is_ok());
    /// assert!(Field::new("x", "int", "").check(&serde_json::json!("a")).is_err());
    /// ```
    pub fn check(&self, v: &Value) -> Result<()> {
        let ok = match self.ty.as_str() {
            "int" => v.is_i64() || v.is_u64(),
            "float" => v.is_number(),
            "text" => v.is_string(),
            "bool" => v.is_boolean(),
            "int[]" => v.as_array().map(|a| a.iter().all(|x| x.is_i64() || x.is_u64())).unwrap_or(false),
            "float[]" => v.as_array().map(|a| a.iter().all(|x| x.is_number())).unwrap_or(false),
            "text[]" => v.as_array().map(|a| a.iter().all(|x| x.is_string())).unwrap_or(false),
            "label" => v.as_str().map(|s| self.labels.is_empty() || self.labels.iter().any(|l| l == s)).unwrap_or(false),
            _ => true,
        };
        anyhow::ensure!(ok, "field '{}' must be of type {} (got {})", self.name, self.ty, v);
        Ok(())
    }
}

/// How values map to model tokens and back.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Codec {
    /// Character-vocabulary language model.
    CharLm {
        /// Token id → string (one character each).
        vocab: Vec<String>,
        /// Token prepended to the prompt.
        #[serde(default)]
        bos: Option<u32>,
        /// Generation stops at this token.
        stop: u32,
        /// Prompt template with `{field}` placeholders.
        template: String,
        /// Separator for array fields.
        #[serde(default = "default_join")]
        join: String,
        /// Max generated tokens.
        #[serde(default = "default_max_new")]
        max_new: usize,
    },
    /// Byte-level language model (vocab 257, id 256 = end of text).
    ByteLm {
        /// Prompt template with `{field}` placeholders.
        template: String,
        /// Max generated tokens.
        #[serde(default = "default_max_new")]
        max_new: usize,
    },
}

fn default_join() -> String {
    ", ".into()
}
fn default_max_new() -> usize {
    64
}

/// A model's I/O signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Signature {
    /// Short task id (e.g. `sequence-next`).
    pub task: String,
    /// What the model does.
    #[serde(default)]
    pub description: String,
    /// Inputs.
    pub inputs: Vec<Field>,
    /// Outputs.
    pub outputs: Vec<Field>,
    /// Encoding/decoding.
    pub codec: Codec,
    /// Example `{input, output}` pairs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<Value>,
}

/// Metadata key holding the signature.
pub const META_KEY: &str = "aido.io";

impl Signature {
    /// Reads the signature of an `.aido` file (`None` if it has none).
    ///
    /// # Example
    /// ```no_run
    /// let s = aido::sdk::io::Signature::read("models/seq.aido").unwrap().unwrap();
    /// println!("{}", s.describe());
    /// ```
    pub fn read(path: &str) -> Result<Option<Signature>> {
        let f = AidoFile::open(path)?;
        match f.get_str(META_KEY) {
            Some(s) => Ok(Some(serde_json::from_str(s).context("invalid aido.io signature")?)),
            None => Ok(None),
        }
    }

    /// Human-readable summary (used by `aido model-info`).
    ///
    /// # Example
    /// ```
    /// use aido::sdk::io::*;
    /// let s = Signature { task: "t".into(), description: "d".into(), inputs: vec![Field::new("x", "int[]", "nums")],
    ///     outputs: vec![Field::new("y", "int", "")], codec: Codec::ByteLm { template: "{x}".into(), max_new: 4 }, examples: vec![] };
    /// assert!(s.describe().contains("x: int[]"));
    /// ```
    pub fn describe(&self) -> String {
        let f = |v: &[Field]| v.iter().map(|x| format!("  {}: {}{}", x.name, x.ty, if x.description.is_empty() { String::new() } else { format!(" — {}", x.description) })).collect::<Vec<_>>().join("\n");
        let codec = match &self.codec {
            Codec::CharLm { vocab, template, .. } => format!("char-lm, {} tokens, template {:?}", vocab.len(), template),
            Codec::ByteLm { template, .. } => format!("byte-lm, template {:?}", template),
        };
        let ex = self.examples.iter().map(|e| format!("  {} → {}", e["input"], e["output"])).collect::<Vec<_>>().join("\n");
        format!("task: {}\n{}\ninputs:\n{}\noutputs:\n{}\ncodec: {}{}", self.task, self.description, f(&self.inputs), f(&self.outputs), codec, if ex.is_empty() { String::new() } else { format!("\nexamples:\n{ex}") })
    }

    /// Validates `input` against the input fields.
    pub fn check_input(&self, input: &Value) -> Result<()> {
        for f in &self.inputs {
            f.check(input.get(&f.name).with_context(|| format!("missing input field '{}' ({})", f.name, f.ty))?)?;
        }
        Ok(())
    }

    /// Fills `{field}` placeholders of a template from `input`.
    fn fill(template: &str, input: &Value, join: &str) -> String {
        let mut out = template.to_string();
        if let Some(o) = input.as_object() {
            for (k, v) in o {
                let s = match v {
                    Value::Array(a) => a.iter().map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string())).collect::<Vec<_>>().join(join),
                    Value::String(s) => s.clone(),
                    o => o.to_string(),
                };
                out = out.replace(&format!("{{{k}}}"), &s);
            }
        }
        out
    }

    /// Converts decoded text into the output JSON object.
    fn parse_output(&self, text: &str) -> Result<Value> {
        let f = self.outputs.first().context("signature has no outputs")?;
        let t = text.trim();
        let v = match f.ty.as_str() {
            "int" => json!(t.parse::<i64>().with_context(|| format!("model output {t:?} is not an int"))?),
            "float" => json!(t.parse::<f64>().with_context(|| format!("model output {t:?} is not a number"))?),
            "json" => serde_json::from_str(t).unwrap_or(json!(t)),
            _ => json!(t),
        };
        Ok(json!({ f.name.clone(): v }))
    }
}

/// Writes a copy of `src` (an SDK checkpoint) with signature `sig` attached.
///
/// # Example
/// ```no_run
/// # let sig: aido::sdk::io::Signature = unimplemented!();
/// aido::sdk::io::attach("models/seq.aido", "models/seq.aido", &sig).unwrap();
/// ```
pub fn attach(src: &str, dst: &str, sig: &Signature) -> Result<()> {
    compile(src, dst, Some(sig), None).map(|_| ())
}

/// Compiles an SDK checkpoint for inference: drops optimizer state, keeps
/// config / trainable flags / signature, optionally stores weights as
/// `f16` or `q8_0` (matrices whose rows are a multiple of 32), and records
/// the transformation in the provenance log. Returns `(bytes_in, bytes_out)`.
///
/// # Example
/// ```no_run
/// let (a, b) = aido::sdk::io::compile("models/seq.aido", "models/seq-f16.aido", None, Some("f16")).unwrap();
/// println!("{a} → {b} bytes");
/// ```
pub fn compile(src: &str, dst: &str, sig: Option<&Signature>, quant: Option<&str>) -> Result<(u64, u64)> {
    let f = AidoFile::open(src)?;
    anyhow::ensure!(f.get_str("general.architecture") == Some("aido-sdk"), "{src} is not an SDK model (general.architecture != aido-sdk)");
    let ty = match quant.map(|q| q.to_ascii_lowercase()) {
        None => None,
        Some(q) => Some(crate::tensor::quantize::parse_type(&q)?),
    };
    let mut w = AidoWriter::new();
    w.copy_meta(&f);
    for k in ["sdk.step", "sdk.opt"] {
        w.remove_meta(k);
    }
    if let Some(s) = sig {
        w.set_meta(META_KEY, json!(serde_json::to_string(s)?));
    }
    let mut n = 0;
    for name in f.names() {
        if !name.starts_with("param/") {
            continue;
        }
        let (_, dims) = f.info(&name).context("tensor info")?;
        let v = f.vector(&name)?;
        let cols = dims[0];
        let t = ty.filter(|t| cols % t.block_elems() == 0 && dims.get(1).copied().unwrap_or(1) > 1);
        match t {
            Some(t) => {
                let mut bytes = Vec::new();
                for row in v.chunks_exact(cols) {
                    bytes.extend(crate::tensor::quantize::quantize_row(t, row));
                }
                w.add(&name, t as u32, dims, bytes);
            }
            None => w.add(&name, 0, dims, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        }
        n += 1;
    }
    w.log(json!({"op": "compile-sdk", "params": n, "quant": quant.unwrap_or("f32"), "signature": sig.is_some()}));
    let before = std::fs::metadata(src)?.len();
    drop(f); // unmap the source: `dst` may be the same file
    let tmp = format!("{dst}.tmp");
    let out = w.write(&tmp)?;
    std::fs::rename(&tmp, dst)?;
    Ok((before, out))
}

/// A loaded SDK model ready for [`SdkModel::predict`].
pub struct SdkModel {
    sig: Signature,
    lm: super::models::LmModel,
}

impl SdkModel {
    /// Loads an SDK `.aido` that has a signature.
    ///
    /// # Example
    /// ```no_run
    /// let m = aido::sdk::io::SdkModel::load("models/seq.aido").unwrap();
    /// ```
    pub fn load(path: &str) -> Result<Self> {
        let sig = Signature::read(path)?.with_context(|| format!("{path} has no I/O signature (aido.io); attach one with aido::sdk::io::attach"))?;
        let (params, _, cfg) = super::load_checkpoint(path)?;
        let cfg: super::models::LmConfig = serde_json::from_value(cfg).context("only language-model SDK checkpoints have codecs so far")?;
        Ok(SdkModel { sig, lm: super::models::LmModel { cfg, params, source_meta: None } })
    }

    /// The signature.
    pub fn signature(&self) -> &Signature {
        &self.sig
    }

    /// Runs the model on a JSON input object; returns the JSON output object.
    ///
    /// # Example
    /// ```no_run
    /// let mut m = aido::sdk::io::SdkModel::load("models/seq.aido").unwrap();
    /// let pool = aido::runtime::pool::ThreadPool::new(4, None);
    /// let out = m.predict(&pool, &serde_json::json!({"sequence": [2, 4, 6, 8]})).unwrap();
    /// assert_eq!(out["next"], 10);
    /// ```
    pub fn predict(&mut self, pool: &ThreadPool, input: &Value) -> Result<Value> {
        self.sig.check_input(input)?;
        match self.sig.codec.clone() {
            Codec::CharLm { vocab, bos, stop, template, join, max_new } => {
                let text = Signature::fill(&template, input, &join);
                let mut ids: Vec<u32> = bos.into_iter().collect();
                for ch in text.chars() {
                    let s = ch.to_string();
                    match vocab.iter().position(|v| *v == s) {
                        Some(i) => ids.push(i as u32),
                        None => bail!("character {ch:?} is not in the model vocabulary"),
                    }
                }
                let out = self.lm.generate_greedy(pool, &ids, max_new, stop);
                let txt: String = out.iter().take_while(|&&t| t != stop).map(|&t| vocab.get(t as usize).cloned().unwrap_or_default()).collect();
                self.sig.parse_output(&txt)
            }
            Codec::ByteLm { template, max_new } => {
                let text = Signature::fill(&template, input, ", ");
                let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
                let out = self.lm.generate_greedy(pool, &ids, max_new, 256);
                let bytes: Vec<u8> = out.iter().take_while(|&&t| t < 256).map(|&t| t as u8).collect();
                self.sig.parse_output(&String::from_utf8_lossy(&bytes))
            }
        }
    }
}

/// The signature of the next-number sequence model trained by `aido train-seq`.
///
/// # Example
/// ```
/// let s = aido::sdk::io::seq_signature();
/// assert_eq!(s.inputs[0].name, "sequence");
/// ```
pub fn seq_signature() -> Signature {
    let mut vocab: Vec<String> = (0..10).map(|d| d.to_string()).collect();
    vocab.extend([",", " ", "-", "|", "\n"].iter().map(|s| s.to_string()));
    Signature {
        task: "sequence-next".into(),
        description: "Finds the rule of an integer sequence (arithmetic, geometric, Fibonacci-like, squares…) and predicts the next element".into(),
        inputs: vec![Field::new("sequence", "int[]", "5–8 integers")],
        outputs: vec![Field::new("next", "int", "the next element")],
        codec: Codec::CharLm { vocab, bos: Some(14), stop: 14, template: "{sequence}|".into(), join: ", ".into(), max_new: 12 },
        examples: vec![json!({"input": {"sequence": [2, 4, 6, 8, 10]}, "output": {"next": 12}}), json!({"input": {"sequence": [1, 1, 2, 3, 5, 8]}, "output": {"next": 13}})],
    }
}
