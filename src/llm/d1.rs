//! Liquid AI **d1** decision models (e.g. `LiquidAI/d1-Omni-2B`): typed,
//! calibrated answers to many named questions about a state (text / JSON,
//! pictures, audio) in **one forward pass** — no token generation.
//!
//! ## How it works
//! 1. The state and every question are rendered into one ChatML prompt;
//!    questions are packed after the state (`set` mode): on full-attention
//!    layers the state and each question read each other both ways, questions
//!    never see each other directly.
//! 2. One backbone pass produces hidden states; only the **answer slots** are
//!    projected, and only onto the few vocabulary rows an answer can take
//!    (`yes/Yes/YES`, option codes, numbers 0–999). The 128 000-row LM head is
//!    never computed — the log-normaliser cancels in the readout softmax.
//! 3. Scores are divided by the checkpoint's calibration temperature and
//!    soft-maxed; typed answers are assembled (P(yes), chosen option,
//!    multiple choices, score level, number median + 80 % interval, box).
//!
//! Supported question types: `noul`, `choice` (single / `multiple`),
//! `score`, `number` with `range` (+`step`), `box`.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::lfm3::{Backbone, Dims, Kind, Visibility};
use crate::format::safetensors::SafetensorsDir;
use crate::format::TensorStore;
use crate::runtime::pool::ThreadPool;
use crate::tensor::QWeight;
use crate::tokenizer::Tokenizer;

/// A question.
#[derive(Debug, Clone)]
pub enum Question {
    /// Yes / no.
    Noul { instructions: String },
    /// One of named options (`multiple`: each option yes/no).
    Choice { instructions: String, options: Vec<(String, String)>, multiple: bool },
    /// Ordered levels 0..n.
    Score { instructions: String, levels: Vec<String> },
    /// A number on a grid `low..=high` in `step`s.
    Estimate { instructions: String, low: f64, high: f64, size: usize, unit: Option<String> },
    /// One object's box.
    Box { instructions: String },
}

impl Question {
    /// Parses the d1 JSON question form, e.g.
    /// `{"type":"choice","instructions":"…","criteria":{"a":"…"}}`.
    ///
    /// # Example
    /// ```
    /// let q = aido::llm::d1::Question::from_json(&serde_json::json!({"type": "noul", "instructions": "Is it red?"})).unwrap();
    /// ```
    pub fn from_json(v: &Value) -> Result<Self> {
        let ins = v["instructions"].as_str().context("instructions must be text")?.to_string();
        Ok(match v["type"].as_str().unwrap_or("noul") {
            "noul" => Question::Noul { instructions: ins },
            "choice" => {
                let options = match &v["criteria"] {
                    Value::Object(o) => o.iter().map(|(k, d)| (k.clone(), d.as_str().unwrap_or(k).to_string())).collect(),
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str()).map(|s| (s.to_string(), s.to_string())).collect(),
                    _ => bail!("choice needs criteria"),
                };
                Question::Choice { instructions: ins, options, multiple: v["multiple"].as_bool().unwrap_or(false) }
            }
            "score" => Question::Score { instructions: ins, levels: v["criteria"].as_array().context("score criteria")?.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect() },
            "number" => {
                let r = v["range"].as_array().context("number questions need \"range\": [low, high] in this engine")?;
                let (low, high) = (r[0].as_f64().unwrap_or(0.0), r[1].as_f64().unwrap_or(100.0));
                let step = v["step"].as_f64().unwrap_or(((high - low) / 127.0).max(1e-9));
                let size = (((high - low) / step).round() as usize + 1).max(2);
                Question::Estimate { instructions: ins, low, high, size, unit: v["unit"].as_str().map(String::from) }
            }
            "box" => Question::Box { instructions: ins },
            t => bail!("question type '{t}' is not supported yet"),
        })
    }
}

/// How one answer slot is read.
#[derive(Debug, Clone)]
enum Slot {
    /// Groups of token ids, max-pooled per group.
    Groups(Vec<Vec<u32>>),
}

/// A loaded d1 model.
pub struct D1 {
    st: SafetensorsDir,
    text: Backbone,
    embed: QWeight,
    /// Tokenizer.
    pub tok: Tokenizer,
    temperature: f64,
    tables: HashMap<String, Vec<(f64, f64)>>,
    /// Vision encoder (pictures), if the checkpoint has one.
    pub vision: Option<super::d1_media::Vision>,
    /// Audio encoder, if the checkpoint has one.
    pub audio: Option<super::d1_media::Audio>,
}

const LEAD: &str = "<think></think>";

fn letters(n: usize) -> Vec<String> {
    (0..n).map(|i| if i < 26 { ((b'A' + i as u8) as char).to_string() } else { format!("{i:02}") }).collect()
}

impl D1 {
    /// Loads a d1 checkpoint directory (config.json, model.safetensors,
    /// tokenizer.json, calibration.json).
    ///
    /// # Example
    /// ```no_run
    /// let pool = aido::runtime::pool::ThreadPool::new(12, None);
    /// let m = aido::llm::d1::D1::load("models/d1-omni", &pool).unwrap();
    /// ```
    pub fn load(dir: &str, pool: &ThreadPool) -> Result<Self> {
        Self::load_with(dir, pool, &super::lfm3::Accel::default())
    }

    /// Loads with an accelerator (GPU + VRAM budget) for the large matrices.
    ///
    /// # Example
    /// ```no_run
    /// let pool = aido::runtime::pool::ThreadPool::new(12, None);
    /// let gpu = aido::gpu::Gpu::new().ok().map(std::sync::Arc::new);
    /// let m = aido::llm::d1::D1::load_with("models/d1-omni", &pool, &aido::llm::lfm3::Accel::new(gpu, 4_000_000_000)).unwrap();
    /// ```
    pub fn load_with(dir: &str, pool: &ThreadPool, acc: &super::lfm3::Accel) -> Result<Self> {
        let st = SafetensorsDir::open(dir)?;
        let cfg: Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
        let tc = &cfg["text_config"];
        let kinds: Vec<Kind> = tc["layer_types"]
            .as_array()
            .context("layer_types")?
            .iter()
            .map(|k| match k.as_str() {
                Some("conv") => Kind::Conv,
                Some("sliding_attention") => Kind::Sliding,
                _ => Kind::Full,
            })
            .collect();
        let dims = Dims {
            d: tc["hidden_size"].as_u64().unwrap_or(2048) as usize,
            heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            kv_heads: tc["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
            hd: tc["head_dim"].as_u64().unwrap_or(128) as usize,
            eps: tc["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            window: tc["sliding_window"].as_u64().unwrap_or(256) as usize,
        };
        let rp = &tc["rope_parameters"];
        let rope = (
            rp["full_attention"]["rope_theta"].as_f64().unwrap_or(1e6) as f32,
            rp["full_attention"]["partial_rotary_factor"].as_f64().unwrap_or(1.0) as f32,
            rp["sliding_attention"]["rope_theta"].as_f64().unwrap_or(1e4) as f32,
        );
        let text = Backbone::load(&st, pool, "model.language_model", &kinds, dims, rope, ("norm", "o_proj"), acc)?;
        let embed = st.weight("model.language_model.embed_tokens.weight")?;
        let tok = Tokenizer::from_hf_json(format!("{dir}/tokenizer.json"), Some("<|startoftext|>"), Some("<|im_end|>"))?;
        let cal: Value = std::fs::read(format!("{dir}/calibration.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
        let tables = cal["tables"]
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_array().map(|a| a.iter().map(|p| (p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0))).collect()).unwrap_or_default())).collect())
            .unwrap_or_default();
        let vision = if st.shape("model.vision_tower.patchifier.weight").is_some() { Some(super::d1_media::Vision::load(&st, pool, &cfg, acc)?) } else { None };
        let audio = if st.shape("model.audio_encoder.lfm.embedding_norm.weight").is_some() { Some(super::d1_media::Audio::load(&st, pool, &cfg, acc)?) } else { None };
        Ok(D1 { temperature: cal["default"].as_f64().unwrap_or(1.0), tables, st, text, embed, tok, vision, audio })
    }

    /// Single-token ids of each text form (forms that are not one token are skipped).
    fn single(&self, forms: &[String]) -> Vec<u32> {
        let mut out = Vec::new();
        for f in forms {
            let e = self.tok.encode(f, false);
            if e.len() == 1 && !out.contains(&e[0]) {
                out.push(e[0]);
            }
        }
        out
    }

    fn yes_no(&self) -> Slot {
        Slot::Groups(vec![self.single(&["yes".into(), "Yes".into(), "YES".into()]), self.single(&["no".into(), "No".into(), "NO".into()])])
    }

    fn numbers(&self, values: impl Iterator<Item = i64>) -> Slot {
        Slot::Groups(values.map(|n| self.single(&[n.to_string(), format!(" {n}")])).collect())
    }

    /// The question's text block and its slots (name markers or the end).
    fn block(&self, q: &Question) -> (String, Vec<String>, Vec<Slot>) {
        match q {
            Question::Noul { instructions } => (format!("{instructions}\n\nReply with yes or no only."), vec![], vec![self.yes_no()]),
            Question::Choice { instructions, options, multiple: false } => {
                let codes = letters(options.len());
                let lines: Vec<String> = codes.iter().zip(options).map(|(c, (_, d))| format!("{c} {d}")).collect();
                let groups = codes.iter().map(|c| self.single(&[c.clone(), format!(" {c}")])).collect();
                (format!("{instructions}\n\nOptions:\n{}\n\nReply with the option code only.", lines.join("\n")), vec![], vec![Slot::Groups(groups)])
            }
            Question::Choice { instructions, options, multiple: true } => {
                let lines: Vec<String> = options.iter().map(|(n, d)| format!("{}: {d}", serde_json::to_string(n).unwrap())).collect();
                let names = options.iter().map(|(n, _)| serde_json::to_string(n).unwrap()).collect::<Vec<_>>();
                let slots = names.iter().map(|_| self.yes_no()).collect();
                (format!("{instructions}\n\nOptions:\n{}\n\nReply with yes or no for each option.", lines.join("\n")), names, slots)
            }
            Question::Score { instructions, levels } => {
                let legend: Vec<String> = levels.iter().enumerate().map(|(i, n)| format!("{i} {n}")).collect();
                let groups = (0..levels.len()).map(|i| self.single(&[i.to_string()])).collect();
                (format!("{instructions}\n\n{}\n\nReply with a single digit 0-{} only.", legend.join("\n"), levels.len() - 1), vec![], vec![Slot::Groups(groups)])
            }
            Question::Estimate { instructions, low, high, size, unit } => {
                let step = (high - low) / (*size as f64 - 1.0);
                let u = unit.as_ref().map(|u| format!(" {u}")).unwrap_or_default();
                let fmt = |v: f64| if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") };
                let vals: Vec<f64> = (0..*size).map(|i| if i + 1 == *size { *high } else { low + i as f64 * step }).collect();
                let slot = if vals.iter().all(|v| v.fract() == 0.0 && *v >= 0.0 && *v < 1000.0) {
                    self.numbers(vals.iter().map(|v| *v as i64))
                } else {
                    Slot::Groups(letters(*size).iter().map(|c| self.single(&[c.clone(), format!(" {c}")])).collect())
                };
                (format!("{instructions}\n\nEstimate from {}{u} to {}{u}, in steps of {}. Reply with the number only.", fmt(*low), fmt(*high), fmt(step)), vec![], vec![slot])
            }
            Question::Box { instructions } => {
                let legend = "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge.";
                let names: Vec<String> = ["present", "top", "bottom", "left", "right"].iter().map(|s| s.to_string()).collect();
                let mut slots = vec![self.yes_no()];
                for _ in 0..4 {
                    slots.push(self.numbers(0..1000));
                }
                (format!("{instructions}\n\nPresence is yes or no. {legend} Reply with yes or no at present, then one number at top, bottom, left and right."), names, slots)
            }
        }
    }

    /// Text after the question block up to its last slot, and the byte
    /// offsets (within the returned string) of each marker's colon.
    fn suffix(&self, q: &Question) -> (String, Vec<usize>, Vec<Slot>) {
        let (body, names, slots) = self.block(q);
        let mut s = format!("{body}<|im_end|>\n<|im_start|>assistant\n{LEAD}");
        let mut colons = Vec::new();
        if names.is_empty() {
            colons.push(usize::MAX); // the last token
        } else {
            for (i, n) in names.iter().enumerate() {
                if i > 0 {
                    s.push('\n');
                }
                s.push_str(n);
                s.push(':');
                colons.push(s.len() - 1);
            }
        }
        (s, colons, slots)
    }

    /// Answers `questions` (name → JSON question) about `state` (text or
    /// JSON, may be null) with optional media embeddings. Returns
    /// `{"answers": {...}, "timing_ms": …, "tokens": …}`.
    ///
    /// # Example
    /// ```no_run
    /// # let pool = aido::runtime::pool::ThreadPool::new(12, None);
    /// let m = aido::llm::d1::D1::load("models/d1-omni", &pool).unwrap();
    /// let out = m.decide(&pool, &serde_json::json!("I was charged twice."), &serde_json::json!({
    ///     "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}), &[]).unwrap();
    /// ```
    pub fn decide(&self, pool: &ThreadPool, state: &Value, questions: &Value, media: &[super::d1_media::MediaInput]) -> Result<Value> {
        let t0 = std::time::Instant::now();
        let qs: Vec<(String, Question)> = questions.as_object().context("questions must be an object")?.iter().map(|(k, v)| Ok((k.clone(), Question::from_json(v)?))).collect::<Result<_>>()?;
        anyhow::ensure!(!qs.is_empty(), "no questions");
        // ---- media embeddings and markup ----
        let mut media_embeds: Vec<(Vec<f32>, usize)> = Vec::new();
        let mut markup = String::new();
        for m in media {
            let (emb, n, mark) = match m {
                super::d1_media::MediaInput::Image(img) => self.vision.as_ref().context("this checkpoint has no vision tower")?.encode(pool, img)?,
                super::d1_media::MediaInput::Audio(wav) => self.audio.as_ref().context("this checkpoint has no audio encoder")?.encode(pool, wav)?,
            };
            markup.push_str(&mark);
            media_embeds.push((emb, n));
        }
        // ---- prompt ----
        let state_text = match state {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            v => Some(serde_json::to_string_pretty(v)?),
        };
        let prefix = match &state_text {
            Some(s) => format!("<|startoftext|><|im_start|>user\n{markup}{s}\n\n\nQUESTION:\n"),
            None => format!("<|startoftext|><|im_start|>user\n{markup}"),
        };
        let header = if state_text.is_some() { "QUESTION:\n" } else { "" };
        let mut ids: Vec<u32> = Vec::new();
        let mut groups: Vec<i32> = Vec::new();
        let mut slot_pos: Vec<Vec<(usize, Slot)>> = Vec::new();
        let mut state_len = 0;
        for (qi, (_, q)) in qs.iter().enumerate() {
            let (suf, colons, slots) = self.suffix(q);
            let text = if qi == 0 { format!("{prefix}{suf}") } else { format!("{header}{suf}") };
            let base = if qi == 0 { 0 } else { header.len() };
            let enc = self.tok.encode(&text, false);
            if qi == 0 {
                // The prefix — media markup and the state — is the "state" every
                // question reads (also when the textual state is empty).
                state_len = self.tok.encode(&prefix, false).len().min(enc.len());
            }
            // byte offsets of every token end
            let mut ends = Vec::with_capacity(enc.len());
            let mut acc = 0;
            for &t in &enc {
                acc += self.tok.token_bytes(t).len();
                ends.push(acc);
            }
            let start = ids.len();
            let mut pos = Vec::new();
            for (c, s) in colons.iter().zip(slots) {
                let tpos = if *c == usize::MAX {
                    enc.len() - 1
                } else {
                    let byte = if qi == 0 { prefix.len() + c } else { base + c };
                    ends.iter().position(|&e| e > byte).unwrap_or(enc.len() - 1)
                };
                pos.push((start + tpos, s));
            }
            for (k, _) in enc.iter().enumerate() {
                groups.push(if qi == 0 && k < state_len { -1 } else { qi as i32 + 1 });
            }
            ids.extend_from_slice(&enc);
            slot_pos.push(pos);
        }
        // ---- embeddings, with media placed at their placeholder tokens ----
        let d = self.text.dims.d;
        let t = ids.len();
        let mut x = vec![0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            self.embed.dequant_row(id as usize, &mut x[i * d..(i + 1) * d]);
        }
        let placeholders: Vec<usize> = ids.iter().enumerate().filter(|(_, &id)| Some(id) == self.tok.token_id("<|image|>") || Some(id) == self.tok.token_id("<|audio|>")).map(|(i, _)| i).collect();
        let mut pi = 0;
        for (emb, n) in &media_embeds {
            for k in 0..*n {
                let at = *placeholders.get(pi).context("media placeholder count mismatch")?;
                x[at * d..(at + 1) * d].copy_from_slice(&emb[k * d..(k + 1) * d]);
                pi += 1;
            }
        }
        let positions: Vec<usize> = (0..t).collect();
        let vis = if qs.len() == 1 { Visibility::All } else { Visibility::Set { state: state_len, groups } };
        self.text.forward(pool, &mut x, t, &positions, &vis);
        // ---- read the slots ----
        let mut answers = serde_json::Map::new();
        for ((name, q), slots) in qs.iter().zip(&slot_pos) {
            let probs: Vec<Vec<f64>> = slots.iter().map(|(p, s)| self.read(&x[p * d..(p + 1) * d], s)).collect();
            answers.insert(name.clone(), self.answer(q, &probs));
        }
        Ok(json!({"answers": answers, "tokens": t, "timing_ms": t0.elapsed().as_secs_f64() * 1e3}))
    }

    /// Calibrated probabilities of one slot.
    fn read(&self, h: &[f32], slot: &Slot) -> Vec<f64> {
        let Slot::Groups(groups) = slot;
        let mut row = vec![0f32; h.len()];
        let mut cache: HashMap<u32, f64> = HashMap::new();
        let mut logit = |id: u32| -> f64 {
            *cache.entry(id).or_insert_with(|| {
                self.embed.dequant_row(id as usize, &mut row);
                row.iter().zip(h).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
            })
        };
        let scores: Vec<f64> = groups.iter().map(|g| g.iter().map(|&i| logit(i)).fold(f64::NEG_INFINITY, f64::max) / self.temperature).collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = e.iter().sum();
        e.iter().map(|v| v / z).collect()
    }

    /// Piecewise-linear calibration table lookup (identity without a table).
    fn through(&self, name: &str, p: f64) -> f64 {
        let Some(t) = self.tables.get(name) else { return p };
        if t.is_empty() {
            return p;
        }
        if p <= t[0].0 {
            return t[0].1;
        }
        for w in t.windows(2) {
            if p <= w[1].0 {
                let f = (p - w[0].0) / (w[1].0 - w[0].0).max(1e-12);
                return w[0].1 + f * (w[1].1 - w[0].1);
            }
        }
        t.last().unwrap().1
    }

    /// Typed answer from slot probabilities.
    fn answer(&self, q: &Question, p: &[Vec<f64>]) -> Value {
        let median = |vals: &[f64], pr: &[f64], level: f64| -> f64 {
            let mut mass = 0.0;
            for (v, x) in vals.iter().zip(pr) {
                mass += x;
                if mass >= level {
                    return *v;
                }
            }
            *vals.last().unwrap()
        };
        match q {
            Question::Noul { .. } => json!({"type": "noul", "noul": p[0][0]}),
            Question::Choice { options, multiple: false, .. } => {
                let best = (0..options.len()).max_by(|&a, &b| p[0][a].partial_cmp(&p[0][b]).unwrap()).unwrap();
                json!({"type": "choice", "choice": options[best].0, "confidence": p[0][best],
                       "probabilities": options.iter().zip(&p[0]).map(|(o, x)| (o.0.clone(), json!(x))).collect::<serde_json::Map<_, _>>()})
            }
            Question::Choice { options, multiple: true, .. } => {
                let pr: Vec<f64> = p.iter().map(|s| self.through("multiple", s[0])).collect();
                json!({"type": "choice", "choices": options.iter().zip(&pr).filter(|(_, x)| **x >= 0.5).map(|(o, _)| o.0.clone()).collect::<Vec<_>>(),
                       "probabilities": options.iter().zip(&pr).map(|(o, x)| (o.0.clone(), json!(x))).collect::<serde_json::Map<_, _>>()})
            }
            Question::Score { levels, .. } => {
                let mean: f64 = p[0].iter().enumerate().map(|(i, x)| i as f64 * x).sum();
                json!({"type": "score", "score": mean, "legend": levels})
            }
            Question::Estimate { low, high, size, unit, .. } => {
                let step = (high - low) / (*size as f64 - 1.0);
                let vals: Vec<f64> = (0..*size).map(|i| if i + 1 == *size { *high } else { low + i as f64 * step }).collect();
                json!({"type": "number", "number": median(&vals, &p[0], 0.5), "interval": [median(&vals, &p[0], 0.1), median(&vals, &p[0], 0.9)], "unit": unit})
            }
            Question::Box { .. } => {
                let vals: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
                let e: Vec<f64> = (1..5).map(|k| median(&vals, &p[k], 0.5)).collect();
                json!({"type": "box", "present": self.through("present", p[0][0]), "box": {"top": e[0], "bottom": e[1], "left": e[2], "right": e[3]}})
            }
        }
    }

    /// The safetensors store (for media encoders and inspection).
    ///
    /// # Example
    /// ```no_run
    /// # let pool = aido::runtime::pool::ThreadPool::new(4, None);
    /// let m = aido::llm::d1::D1::load("models/d1-omni", &pool).unwrap();
    /// use aido::format::TensorStore;
    /// println!("{}", m.store().names().len());
    /// ```
    pub fn store(&self) -> &SafetensorsDir {
        &self.st
    }
}
