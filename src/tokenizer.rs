//! Tokenizers loaded from GGUF metadata: byte-level BPE (GPT-2 / Llama-3 /
//! Qwen style) and SentencePiece (Llama-2 / Mistral / Gemma style).

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{bail, Result};
use fancy_regex::Regex;

use crate::format::TensorStore;

/// Tokenizer algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Byte-level BPE with merges.
    Bpe,
    /// SentencePiece unigram/BPE with scores and byte fallback.
    Spm,
}

/// A text ↔ token-id converter.
pub struct Tokenizer {
    /// Algorithm.
    pub kind: Kind,
    /// Token strings indexed by id.
    pub vocab: Vec<String>,
    /// GGUF token types (1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte).
    pub types: Vec<i32>,
    scores: Vec<f32>,
    to_id: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    specials: Vec<(String, u32)>,
    pre: Option<Regex>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
    /// Beginning-of-sequence id.
    pub bos: Option<u32>,
    /// End-of-sequence id.
    pub eos: Option<u32>,
    /// Whether a BOS token is prepended by default.
    pub add_bos: bool,
    /// All ids that terminate generation (eos, eot, `<|im_end|>`, …).
    pub stop_ids: Vec<u32>,
    add_space_prefix: bool,
}

/// GPT-2 byte → printable unicode mapping.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

impl Tokenizer {
    /// Builds the tokenizer stored in a GGUF file.
    ///
    /// * **Input** — opened GGUF file with `tokenizer.ggml.*` keys.
    /// * **Output** — a ready tokenizer, or an error for unknown models.
    ///
    /// # Example
    /// ```no_run
    /// let f = aido::format::open("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// let ids = tok.encode("Hello world", false);
    /// assert_eq!(tok.decode(&ids), "Hello world");
    /// ```
    pub fn from_gguf(f: &dyn TensorStore) -> Result<Self> {
        let model = f.get_str("tokenizer.ggml.model").unwrap_or("gpt2");
        let kind = match model {
            "gpt2" => Kind::Bpe,
            "llama" => Kind::Spm,
            m => bail!("unsupported tokenizer model '{m}'"),
        };
        let vocab: Vec<String> = f
            .meta()
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        if vocab.is_empty() {
            bail!("GGUF has no tokenizer vocabulary");
        }
        let types: Vec<i32> = f
            .meta()
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_u64().map(|u| u as i32).unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; vocab.len()]);
        let scores: Vec<f32> = f
            .meta()
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_else(|| vec![0.0; vocab.len()]);
        let to_id: HashMap<String, u32> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i as u32)).collect();
        let mut merges = HashMap::new();
        if let Some(arr) = f.meta().get("tokenizer.ggml.merges").and_then(|v| v.as_array()) {
            for (rank, m) in arr.iter().enumerate() {
                if let Some(s) = m.as_str() {
                    if let Some((a, b)) = s.split_once(' ') {
                        if let (Some(&ia), Some(&ib), Some(&im)) = (to_id.get(a), to_id.get(b), to_id.get(&format!("{a}{b}"))) {
                            merges.insert((ia, ib), (rank as u32, im));
                        }
                    }
                }
            }
        }
        let mut specials: Vec<(String, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(i, s)| matches!(types.get(*i), Some(3) | Some(4)) && !s.is_empty())
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let pre_name = f.get_str("tokenizer.ggml.pre").unwrap_or("default");
        let pat = match pre_name {
            "qwen2" | "deepseek-r1-qwen" => Some(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
            "gpt2" | "default" if kind == Kind::Bpe => Some(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
            _ if kind == Kind::Bpe => Some(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
            _ => None,
        };
        let pre = pat.map(|p| Regex::new(p)).transpose()?;
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let id = |k: &str| f.get_u64(k).map(|v| v as u32);
        let bos = id("tokenizer.ggml.bos_token_id");
        let eos = id("tokenizer.ggml.eos_token_id");
        let add_bos = f
            .meta()
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_u64())
            .map(|v| v != 0)
            .unwrap_or(kind == Kind::Spm);
        let mut stop_ids: Vec<u32> = [eos, id("tokenizer.ggml.eot_token_id"), id("tokenizer.ggml.eom_token_id")].into_iter().flatten().collect();
        for name in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end|>", "<end_of_turn>", "<|end_of_text|>", "<|eom_id|>"] {
            if let Some(&i) = to_id.get(name) {
                if matches!(types.get(i as usize), Some(3) | Some(4)) {
                    stop_ids.push(i);
                }
            }
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();
        let add_space_prefix = f
            .meta()
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.as_u64())
            .map(|v| v != 0)
            .unwrap_or(true);
        Ok(Tokenizer {
            kind,
            vocab,
            types,
            scores,
            to_id,
            merges,
            specials,
            pre,
            byte_enc,
            byte_dec,
            cache: Mutex::new(HashMap::new()),
            bos,
            eos,
            add_bos,
            stop_ids,
            add_space_prefix,
        })
    }

    /// Vocabulary size.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::format::open("m.gguf").unwrap();
    /// # let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// assert!(tok.len() > 1000);
    /// ```
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// `true` if the vocabulary is empty (never for a loaded model).
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Looks up the id of an exact token string.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::format::open("m.gguf").unwrap();
    /// # let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// let im_end = tok.token_id("<|im_end|>");
    /// ```
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.to_id.get(s).copied()
    }

    /// Encodes text into token ids.
    ///
    /// * **Input** — `text`; `add_bos`: prepend BOS if the model uses one.
    /// * **Output** — token ids. Special tokens written literally in the text
    ///   (e.g. `<|im_start|>`) are mapped to their ids, which is what chat
    ///   templates rely on.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::format::open("m.gguf").unwrap();
    /// # let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// let ids = tok.encode("<|im_start|>user\nhi", false);
    /// assert_eq!(ids[0], tok.token_id("<|im_start|>").unwrap());
    /// ```
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        let mut first_segment = true;
        for (seg, special) in self.split_specials(text) {
            match special {
                Some(id) => out.push(id),
                None => {
                    match self.kind {
                        Kind::Bpe => self.encode_bpe(seg, &mut out),
                        Kind::Spm => self.encode_spm(seg, first_segment && self.add_space_prefix, &mut out),
                    }
                }
            }
            first_segment = false;
        }
        out
    }

    /// Splits text around literal special tokens.
    fn split_specials<'a>(&self, text: &'a str) -> Vec<(&'a str, Option<u32>)> {
        let mut parts = Vec::new();
        let mut start = 0;
        let mut i = 0;
        let bytes = text.as_bytes();
        while i < text.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let mut hit = None;
            if bytes[i] == b'<' || bytes[i] == b'[' {
                for (s, id) in &self.specials {
                    if text[i..].starts_with(s.as_str()) {
                        hit = Some((s.len(), *id));
                        break;
                    }
                }
            }
            if let Some((len, id)) = hit {
                if start < i {
                    parts.push((&text[start..i], None));
                }
                parts.push((&text[i..i + len], Some(id)));
                i += len;
                start = i;
            } else {
                i += 1;
            }
        }
        if start < text.len() {
            parts.push((&text[start..], None));
        }
        parts
    }

    /// Byte-level BPE on one special-free segment.
    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        let words: Vec<&str> = match &self.pre {
            Some(re) => re.find_iter(text).filter_map(|m| m.ok()).map(|m| m.as_str()).collect(),
            None => vec![text],
        };
        for w in words {
            if let Some(ids) = self.cache.lock().unwrap().get(w) {
                out.extend_from_slice(ids);
                continue;
            }
            let mapped: String = w.bytes().map(|b| self.byte_enc[b as usize]).collect();
            let ids = if let Some(&id) = self.to_id.get(&mapped) {
                vec![id]
            } else {
                let mut sym: Vec<u32> = mapped
                    .chars()
                    .map(|c| {
                        let mut b = [0u8; 4];
                        self.to_id.get(c.encode_utf8(&mut b) as &str).copied().unwrap_or(0)
                    })
                    .collect();
                loop {
                    let mut best: Option<(u32, usize, u32)> = None;
                    for i in 0..sym.len().saturating_sub(1) {
                        if let Some(&(rank, merged)) = self.merges.get(&(sym[i], sym[i + 1])) {
                            if best.map_or(true, |b| rank < b.0) {
                                best = Some((rank, i, merged));
                            }
                        }
                    }
                    match best {
                        Some((_, i, m)) => {
                            sym[i] = m;
                            sym.remove(i + 1);
                        }
                        None => break,
                    }
                }
                sym
            };
            let mut c = self.cache.lock().unwrap();
            if c.len() > 100_000 {
                c.clear();
            }
            c.insert(w.to_string(), ids.clone());
            out.extend(ids);
        }
    }

    /// SentencePiece (score-greedy merges with byte fallback).
    fn encode_spm(&self, text: &str, prefix: bool, out: &mut Vec<u32>) {
        let mut s = String::new();
        if prefix && !text.starts_with(' ') {
            s.push('▁');
        }
        s.push_str(&text.replace(' ', "▁"));
        // Symbols = UTF-8 chars.
        let mut sym: Vec<String> = s.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(f32, usize)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                let cat = format!("{}{}", sym[i], sym[i + 1]);
                if let Some(&id) = self.to_id.get(&cat) {
                    let sc = self.scores[id as usize];
                    if best.map_or(true, |b| sc > b.0) {
                        best = Some((sc, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let n = sym.remove(i + 1);
                    sym[i].push_str(&n);
                }
                None => break,
            }
        }
        for p in sym {
            if let Some(&id) = self.to_id.get(&p) {
                out.push(id);
            } else {
                for b in p.bytes() {
                    out.push(self.to_id.get(&format!("<0x{b:02X}>")).copied().unwrap_or(0));
                }
            }
        }
    }

    /// Returns the raw bytes a token represents (may be a partial UTF-8 char).
    ///
    /// * **Output** — bytes; control tokens yield their literal text.
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::format::open("m.gguf").unwrap();
    /// # let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// let b = tok.token_bytes(tok.encode("a", false)[0]);
    /// assert_eq!(b, b"a");
    /// ```
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let s = match self.vocab.get(id as usize) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let ty = self.types.get(id as usize).copied().unwrap_or(1);
        if ty == 3 || ty == 4 {
            return s.as_bytes().to_vec();
        }
        match self.kind {
            Kind::Bpe => s.chars().map(|c| self.byte_dec.get(&c).copied().unwrap_or(b'?')).collect(),
            Kind::Spm => {
                if ty == 6 && s.len() == 6 && s.starts_with("<0x") {
                    if let Ok(b) = u8::from_str_radix(&s[3..5], 16) {
                        return vec![b];
                    }
                }
                s.replace('▁', " ").into_bytes()
            }
        }
    }

    /// `true` if `id` is a control token (BOS/EOS/chat markers).
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::format::open("m.gguf").unwrap();
    /// # let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// assert!(tok.is_control(tok.eos.unwrap()));
    /// ```
    pub fn is_control(&self, id: u32) -> bool {
        self.types.get(id as usize).copied() == Some(3)
    }

    /// Decodes ids into a string (invalid UTF-8 replaced, control tokens kept).
    ///
    /// # Example
    /// ```no_run
    /// # let f = aido::format::open("m.gguf").unwrap();
    /// # let tok = aido::tokenizer::Tokenizer::from_gguf(f.as_ref()).unwrap();
    /// assert_eq!(tok.decode(&tok.encode("Привет, мир!", false)), "Привет, мир!");
    /// ```
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        for &i in ids {
            b.extend(self.token_bytes(i));
        }
        let s = String::from_utf8_lossy(&b).into_owned();
        if self.kind == Kind::Spm && self.add_space_prefix {
            s.strip_prefix(' ').map(|x| x.to_string()).unwrap_or(s)
        } else {
            s
        }
    }
}

/// Incremental UTF-8 decoder for streaming: buffers partial characters
/// that span token boundaries.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// Feeds token bytes, returns the newly completed text (possibly empty).
    ///
    /// # Example
    /// ```
    /// let mut d = aido::tokenizer::StreamDecoder::default();
    /// let euro = "€".as_bytes();
    /// assert_eq!(d.push(&euro[..1]), "");
    /// assert_eq!(d.push(&euro[1..]), "€");
    /// ```
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                if e.error_len().is_some() {
                    // Truly invalid sequence: emit replacement and drop it.
                    let bad = valid + e.error_len().unwrap();
                    self.pending.drain(..bad);
                    return s + "\u{FFFD}" + &self.push(&[]);
                }
                self.pending.drain(..valid);
                s
            }
        }
    }

    /// Flushes any incomplete trailing bytes (lossy).
    ///
    /// # Example
    /// ```
    /// let mut d = aido::tokenizer::StreamDecoder::default();
    /// d.push(&[0xE2]);
    /// assert_eq!(d.flush(), "\u{FFFD}");
    /// ```
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Finds the first `"Regex"` pattern inside a (nested) pre-tokenizer spec.
fn find_regex(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Object(o) => {
            if let Some(serde_json::Value::String(s)) = o.get("Regex") {
                return Some(s.clone());
            }
            o.values().find_map(find_regex)
        }
        serde_json::Value::Array(a) => a.iter().find_map(find_regex),
        _ => None,
    }
}

impl Tokenizer {
    /// Loads a Hugging Face `tokenizer.json` (byte-level BPE models such as
    /// Qwen, Llama-3, GPT-2).
    ///
    /// * **Input** — path to `tokenizer.json`; optional BOS/EOS token strings.
    /// * **Output** — tokenizer using the file's own pre-tokenizer regex,
    ///   vocabulary, merges and added (special) tokens.
    ///
    /// # Example
    /// ```no_run
    /// let tok = aido::tokenizer::Tokenizer::from_hf_json("D:/colibri-models/qwen36-35b/tokenizer.json", None, Some("<|im_end|>")).unwrap();
    /// assert_eq!(tok.decode(&tok.encode("hello", false)), "hello");
    /// ```
    pub fn from_hf_json(path: impl AsRef<std::path::Path>, bos: Option<&str>, eos: Option<&str>) -> Result<Self> {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(path.as_ref())?)?;
        let model = &j["model"];
        if model["type"].as_str().unwrap_or("BPE") != "BPE" {
            bail!("only BPE tokenizer.json files are supported");
        }
        let vmap = model["vocab"].as_object().ok_or_else(|| anyhow::anyhow!("tokenizer.json: no vocab"))?;
        let mut n = vmap.values().filter_map(|v| v.as_u64()).max().unwrap_or(0) as usize + 1;
        let added = j["added_tokens"].as_array().cloned().unwrap_or_default();
        for a in &added {
            n = n.max(a["id"].as_u64().unwrap_or(0) as usize + 1);
        }
        let mut vocab = vec![String::new(); n];
        let mut types = vec![1i32; n];
        for (s, id) in vmap {
            if let Some(i) = id.as_u64() {
                vocab[i as usize] = s.clone();
            }
        }
        for a in &added {
            let i = a["id"].as_u64().unwrap_or(0) as usize;
            vocab[i] = a["content"].as_str().unwrap_or("").to_string();
            types[i] = if a["special"].as_bool().unwrap_or(false) { 3 } else { 4 };
        }
        let to_id: HashMap<String, u32> = vocab.iter().enumerate().filter(|(_, s)| !s.is_empty()).map(|(i, s)| (s.clone(), i as u32)).collect();
        let mut merges = HashMap::new();
        if let Some(arr) = model["merges"].as_array() {
            for (rank, m) in arr.iter().enumerate() {
                let (a, b) = match m {
                    serde_json::Value::String(s) => match s.split_once(' ') {
                        Some((a, b)) => (a.to_string(), b.to_string()),
                        None => continue,
                    },
                    serde_json::Value::Array(p) if p.len() == 2 => (p[0].as_str().unwrap_or("").to_string(), p[1].as_str().unwrap_or("").to_string()),
                    _ => continue,
                };
                if let (Some(&ia), Some(&ib), Some(&im)) = (to_id.get(&a), to_id.get(&b), to_id.get(&format!("{a}{b}"))) {
                    merges.insert((ia, ib), (rank as u32, im));
                }
            }
        }
        let mut specials: Vec<(String, u32)> = vocab.iter().enumerate().filter(|(i, s)| types[*i] != 1 && !s.is_empty()).map(|(i, s)| (s.clone(), i as u32)).collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let pat = find_regex(&j["pre_tokenizer"]).unwrap_or_else(|| r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".to_string());
        let pre = Some(Regex::new(&pat)?);
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let bos_id = bos.and_then(|s| to_id.get(s).copied());
        let eos_id = eos.and_then(|s| to_id.get(s).copied());
        let mut stop_ids: Vec<u32> = eos_id.into_iter().collect();
        for name in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end|>", "<|end_of_text|>"] {
            if let Some(&i) = to_id.get(name) {
                stop_ids.push(i);
            }
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();
        let scores = vec![0.0; n];
        Ok(Tokenizer {
            kind: Kind::Bpe,
            vocab,
            types,
            scores,
            to_id,
            merges,
            specials,
            pre,
            byte_enc,
            byte_dec,
            cache: Mutex::new(HashMap::new()),
            bos: bos_id,
            eos: eos_id,
            add_bos: false,
            stop_ids,
            add_space_prefix: false,
        })
    }
}
