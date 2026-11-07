//! CLIP tokenizer and text encoder (OpenAI / OpenCLIP layout as exported by
//! `transformers.CLIPTextModel`).

use std::collections::HashMap;

use anyhow::{Context, Result};
use fancy_regex::Regex;

use super::nn::{gelu_erf, LayerNorm, Linear};
use crate::format::safetensors::SafetensorsDir;
use crate::runtime::pool::ThreadPool;

/// CLIP byte-level BPE tokenizer (`vocab.json` + `merges.txt`).
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    re: Regex,
    byte_enc: [char; 256],
    /// `<|startoftext|>` id.
    pub bos: u32,
    /// `<|endoftext|>` id.
    pub eos: u32,
    /// Padding id.
    pub pad: u32,
}

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

impl ClipTokenizer {
    /// Loads the tokenizer directory (`vocab.json`, `merges.txt`).
    ///
    /// # Example
    /// ```no_run
    /// let t = aido::diffusion::clip::ClipTokenizer::load("models/sd-turbo/tokenizer").unwrap();
    /// let ids = t.encode("a photo of a cat", 77);
    /// assert_eq!(ids.len(), 77);
    /// ```
    pub fn load(dir: &str) -> Result<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_slice(&std::fs::read(format!("{dir}/vocab.json"))?)?;
        let merges = std::fs::read_to_string(format!("{dir}/merges.txt"))?;
        let ranks = merges
            .lines()
            .skip(1)
            .filter_map(|l| l.split_once(' '))
            .enumerate()
            .map(|(i, (a, b))| ((a.to_string(), b.to_string()), i))
            .collect();
        let re = Regex::new(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")?;
        let bos = *vocab.get("<|startoftext|>").context("bos")?;
        let eos = *vocab.get("<|endoftext|>").context("eos")?;
        let pad = vocab.get("!").copied().unwrap_or(eos);
        Ok(ClipTokenizer { vocab, ranks, re, byte_enc: bytes_to_unicode(), bos, eos, pad })
    }

    /// BPE of one pre-token (with the `</w>` end-of-word marker).
    fn bpe(&self, word: &str) -> Vec<u32> {
        let chars: Vec<char> = word.bytes().map(|b| self.byte_enc[b as usize]).collect();
        let mut sym: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(l) = sym.last_mut() {
            l.push_str("</w>");
        }
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone())) {
                    if best.map_or(true, |b| r < b.0) {
                        best = Some((r, i));
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
        sym.iter().map(|s| self.vocab.get(s).copied().unwrap_or(self.eos)).collect()
    }

    /// Encodes text to exactly `len` ids: BOS, tokens, EOS, padding.
    ///
    /// # Example
    /// ```no_run
    /// # let t = aido::diffusion::clip::ClipTokenizer::load("models/sd-turbo/tokenizer").unwrap();
    /// assert_eq!(t.encode("", 77)[0], t.bos);
    /// ```
    pub fn encode(&self, text: &str, len: usize) -> Vec<u32> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![self.bos];
        for m in self.re.find_iter(&clean).filter_map(|m| m.ok()) {
            ids.extend(self.bpe(m.as_str()));
        }
        ids.truncate(len - 1);
        ids.push(self.eos);
        while ids.len() < len {
            ids.push(self.pad);
        }
        ids
    }
}

struct ClipLayer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// CLIP text transformer (causal), returns the last hidden state after the
/// final LayerNorm — the conditioning used by Stable Diffusion 2.x.
pub struct ClipText {
    tok_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    layers: Vec<ClipLayer>,
    final_ln: LayerNorm,
    /// Hidden size.
    pub dim: usize,
    heads: usize,
    quick_gelu: bool,
}

impl ClipText {
    /// Loads `text_encoder/` (config.json + safetensors).
    ///
    /// # Example
    /// ```no_run
    /// let te = aido::diffusion::clip::ClipText::load("models/sd-turbo/text_encoder").unwrap();
    /// ```
    pub fn load(dir: &str) -> Result<Self> {
        let st = SafetensorsDir::open(dir)?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
        let n = cfg["num_hidden_layers"].as_u64().unwrap_or(12) as usize;
        let heads = cfg["num_attention_heads"].as_u64().unwrap_or(12) as usize;
        let dim = cfg["hidden_size"].as_u64().unwrap_or(768) as usize;
        let quick_gelu = cfg["hidden_act"].as_str() == Some("quick_gelu");
        let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let p = "text_model";
        let mut layers = Vec::new();
        for i in 0..n {
            let l = |s: &str| format!("{p}.encoder.layers.{i}.{s}");
            layers.push(ClipLayer {
                ln1: LayerNorm::load(&st, &l("layer_norm1"), eps)?,
                q: Linear::load(&st, &l("self_attn.q_proj"))?,
                k: Linear::load(&st, &l("self_attn.k_proj"))?,
                v: Linear::load(&st, &l("self_attn.v_proj"))?,
                o: Linear::load(&st, &l("self_attn.out_proj"))?,
                ln2: LayerNorm::load(&st, &l("layer_norm2"), eps)?,
                fc1: Linear::load(&st, &l("mlp.fc1"))?,
                fc2: Linear::load(&st, &l("mlp.fc2"))?,
            });
        }
        Ok(ClipText {
            tok_emb: st.f32(&format!("{p}.embeddings.token_embedding.weight"))?,
            pos_emb: st.f32(&format!("{p}.embeddings.position_embedding.weight"))?,
            layers,
            final_ln: LayerNorm::load(&st, &format!("{p}.final_layer_norm"), eps)?,
            dim,
            heads,
            quick_gelu,
        })
    }

    /// Encodes token ids to `len × dim` hidden states.
    ///
    /// # Example
    /// ```no_run
    /// # let te = aido::diffusion::clip::ClipText::load("models/sd-turbo/text_encoder").unwrap();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// let h = te.forward(&pool, &[49406, 49407]);
    /// ```
    pub fn forward(&self, pool: &ThreadPool, ids: &[u32]) -> Vec<f32> {
        let (n, d) = (ids.len(), self.dim);
        let mut x = vec![0f32; n * d];
        for (i, &t) in ids.iter().enumerate() {
            for j in 0..d {
                x[i * d + j] = self.tok_emb[t as usize * d + j] + self.pos_emb[i * d + j];
            }
        }
        let hd = d / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for l in &self.layers {
            let h = l.ln1.forward(&x);
            let q = l.q.forward(pool, &h, n);
            let k = l.k.forward(pool, &h, n);
            let v = l.v.forward(pool, &h, n);
            let mut att = vec![0f32; n * d];
            for hh in 0..self.heads {
                for i in 0..n {
                    let mut sc: Vec<f32> = (0..=i)
                        .map(|j| (0..hd).map(|e| q[i * d + hh * hd + e] * k[j * d + hh * hd + e]).sum::<f32>() * scale)
                        .collect();
                    crate::tensor::ops::softmax_inplace(&mut sc);
                    for (j, w) in sc.iter().enumerate() {
                        for e in 0..hd {
                            att[i * d + hh * hd + e] += w * v[j * d + hh * hd + e];
                        }
                    }
                }
            }
            let o = l.o.forward(pool, &att, n);
            crate::tensor::ops::add_inplace(&mut x, &o);
            let h = l.ln2.forward(&x);
            let mut f = l.fc1.forward(pool, &h, n);
            if self.quick_gelu {
                crate::tensor::ops::quick_gelu_inplace(&mut f);
            } else {
                gelu_erf(&mut f);
            }
            let o = l.fc2.forward(pool, &f, n);
            crate::tensor::ops::add_inplace(&mut x, &o);
        }
        self.final_ln.forward(&x)
    }
}
