//! Byte-level BPE tokenizer **trainer** restricted to Russian + English.
//!
//! The alphabet is limited to printable ASCII (Latin letters, digits,
//! punctuation, symbols), the Cyrillic alphabet (incl. «ё»), a few
//! typographic marks and newlines; everything else is normalised away. The
//! base vocabulary therefore contains only the UTF-8 bytes these characters
//! use (~150 instead of 256) and every learned merge is a Russian or English
//! piece — no capacity is spent on other scripts.
//!
//! The result is written in GGUF tokenizer form (`gpt2` model, byte-to-
//! unicode token strings, `"a b"` merges, Qwen-2 pre-tokenizer) so the
//! engine's [`crate::tokenizer::Tokenizer`] loads it unchanged.

use std::collections::{BTreeSet, HashMap};

use fancy_regex::Regex;
use serde_json::{json, Value};

/// Pre-tokenizer pattern (same as Qwen 2, used again at inference time).
pub const PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Special tokens appended after the learned vocabulary.
pub const SPECIALS: [&str; 3] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];

/// `true` for characters kept by the normaliser.
///
/// # Example
/// ```
/// use aido::sdk::bpe::allowed;
/// assert!(allowed('Ж') && allowed('q') && allowed('7') && allowed('?'));
/// assert!(!allowed('東'));
/// ```
pub fn allowed(c: char) -> bool {
    matches!(c, ' '..='~' | '\n') || ('А'..='я').contains(&c) || matches!(c, 'ё' | 'Ё' | '«' | '»' | '—' | '–' | '…' | '№')
}

/// Normalises text to the ru/en alphabet (other characters become spaces,
/// runs of spaces collapse).
///
/// # Example
/// ```
/// assert_eq!(aido::sdk::bpe::normalize("Привет,  世界 world"), "Привет, world");
/// ```
pub fn normalize(text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    let mut last_space = false;
    for c in text.chars() {
        let c = if c == '\t' || c == '\r' { ' ' } else { c };
        let c = if allowed(c) { c } else { ' ' };
        if c == ' ' {
            if last_space {
                continue;
            }
            last_space = true;
        } else {
            last_space = false;
        }
        s.push(c);
    }
    s
}

fn byte_unicode() -> [char; 256] {
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

/// A trained tokenizer description.
pub struct TrainedBpe {
    /// Token strings (byte-to-unicode form), specials last.
    pub tokens: Vec<String>,
    /// Merges `"a b"` in rank order.
    pub merges: Vec<String>,
    /// Number of base (byte) tokens.
    pub n_base: usize,
}

/// Trains BPE merges until the vocabulary has `vocab` entries.
///
/// * **Input** — corpus text (normalised internally), target vocabulary.
/// * **Output** — tokens and merges.
///
/// # Example
/// ```
/// let t = aido::sdk::bpe::train("привет привет мир hello hello world", 40);
/// assert!(t.tokens.len() <= 40 + 3);
/// assert!(!t.merges.is_empty());
/// ```
pub fn train(corpus: &str, vocab: usize) -> TrainedBpe {
    let text = normalize(corpus);
    let enc = byte_unicode();
    let re = Regex::new(PRE).unwrap();
    let mut counts: HashMap<&str, u64> = HashMap::new();
    for m in re.find_iter(&text).filter_map(|m| m.ok()) {
        *counts.entry(m.as_str()).or_insert(0) += 1;
    }
    let base: BTreeSet<u8> = text.bytes().collect();
    let mut tokens: Vec<String> = base.iter().map(|b| enc[*b as usize].to_string()).collect();
    let n_base = tokens.len();
    let mut id: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let mut words: Vec<(Vec<u32>, u64)> = counts
        .iter()
        .map(|(w, c)| (w.bytes().map(|b| id[&enc[b as usize].to_string()]).collect(), *c))
        .collect();
    let mut merges = Vec::new();
    while tokens.len() < vocab {
        let mut pairs: HashMap<(u32, u32), u64> = HashMap::new();
        for (w, c) in &words {
            for p in w.windows(2) {
                *pairs.entry((p[0], p[1])).or_insert(0) += c;
            }
        }
        let Some((&(a, b), &cnt)) = pairs.iter().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0))) else { break };
        if cnt < 2 {
            break;
        }
        let new = format!("{}{}", tokens[a as usize], tokens[b as usize]);
        merges.push(format!("{} {}", tokens[a as usize], tokens[b as usize]));
        let nid = tokens.len() as u32;
        id.insert(new.clone(), nid);
        tokens.push(new);
        for (w, _) in words.iter_mut() {
            let mut i = 0;
            let mut out = Vec::with_capacity(w.len());
            while i < w.len() {
                if i + 1 < w.len() && w[i] == a && w[i + 1] == b {
                    out.push(nid);
                    i += 2;
                } else {
                    out.push(w[i]);
                    i += 1;
                }
            }
            *w = out;
        }
    }
    for s in SPECIALS {
        tokens.push(s.to_string());
    }
    TrainedBpe { tokens, merges, n_base }
}

impl TrainedBpe {
    /// GGUF-style tokenizer metadata (for `.aido` export / `Tokenizer::from_gguf`).
    ///
    /// # Example
    /// ```
    /// let t = aido::sdk::bpe::train("aa bb aa bb", 20);
    /// let m = t.metadata();
    /// assert_eq!(m["tokenizer.ggml.model"], "gpt2");
    /// ```
    pub fn metadata(&self) -> serde_json::Map<String, Value> {
        let n = self.tokens.len();
        let types: Vec<u32> = (0..n).map(|i| if i >= n - SPECIALS.len() { 3 } else { 1 }).collect();
        let mut m = serde_json::Map::new();
        m.insert("tokenizer.ggml.model".into(), json!("gpt2"));
        m.insert("tokenizer.ggml.pre".into(), json!("qwen2"));
        m.insert("tokenizer.ggml.tokens".into(), json!(self.tokens));
        m.insert("tokenizer.ggml.token_type".into(), json!(types));
        m.insert("tokenizer.ggml.merges".into(), json!(self.merges));
        m.insert("tokenizer.ggml.eos_token_id".into(), json!(n - 3));
        m.insert("tokenizer.ggml.bos_token_id".into(), json!(n - 3));
        m.insert("tokenizer.ggml.add_bos_token".into(), json!(false));
        m
    }
}
