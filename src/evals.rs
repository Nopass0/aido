//! Task benchmarks for language models (exact-match scoring).
//!
//! * `gsm8k` — grade-school math word problems (OpenAI GSM8K test jsonl);
//!   the last number of the answer is compared to the reference after `####`.
//! * `arith` — generated 2–3-digit arithmetic, answer = bare number (fast:
//!   a few tokens per item, good for many weight-edit experiments).
//! * `seq` — next element of integer sequences (pattern finding).

use anyhow::Result;

use crate::llm::chat::Message;
use crate::llm::engine::{GenParams, LlmEngine, SpecMode};
use crate::sdk::Rng;

/// One scored item.
#[derive(Debug, Clone)]
pub struct Item {
    /// Prompt text.
    pub prompt: String,
    /// Reference answer.
    pub answer: String,
}

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct Score {
    /// Correct answers.
    pub correct: usize,
    /// Items evaluated.
    pub total: usize,
    /// Wall time in seconds.
    pub secs: f64,
}

impl Score {
    /// Accuracy in percent.
    ///
    /// # Example
    /// ```
    /// let s = aido::evals::Score { correct: 1, total: 4, secs: 1.0 };
    /// assert_eq!(s.percent(), 25.0);
    /// ```
    pub fn percent(&self) -> f64 {
        100.0 * self.correct as f64 / self.total.max(1) as f64
    }
}

/// Extracts the answer number: the one after `####` if present, otherwise
/// the last number in the text (commas and `$` removed).
///
/// # Example
/// ```
/// assert_eq!(aido::evals::final_number("so 3+4 = 7.\n#### 7"), Some("7".into()));
/// assert_eq!(aido::evals::final_number("The answer is 1,250 dollars."), Some("1250".into()));
/// ```
pub fn final_number(text: &str) -> Option<String> {
    let t = text.replace(',', "").replace('$', "");
    let tail = t.rsplit("####").next().unwrap_or(&t);
    let src = if t.contains("####") { tail } else { t.as_str() };
    let mut last = None;
    let mut cur = String::new();
    for ch in src.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() || (ch == '-' && cur.is_empty()) || (ch == '.' && !cur.is_empty() && !cur.contains('.')) {
            cur.push(ch);
        } else {
            let c = cur.trim_end_matches('.').to_string();
            if c.chars().any(|x| x.is_ascii_digit()) {
                last = Some(c);
                if t.contains("####") {
                    break;
                }
            }
            cur.clear();
        }
    }
    last.map(|s| s.strip_suffix(".0").map(String::from).unwrap_or(s))
}

/// Loads `n` GSM8K items from the jsonl file.
///
/// # Example
/// ```no_run
/// let items = aido::evals::load_gsm8k("data/gsm8k_test.jsonl", 20).unwrap();
/// ```
pub fn load_gsm8k(path: &str, n: usize) -> Result<Vec<Item>> {
    let mut v = Vec::new();
    for line in std::fs::read_to_string(path)?.lines().take(n) {
        let j: serde_json::Value = serde_json::from_str(line)?;
        let q = j["question"].as_str().unwrap_or("");
        let a = j["answer"].as_str().unwrap_or("");
        v.push(Item {
            prompt: format!("{q}\nSolve the problem briefly step by step, then write the final answer on the last line in the form: #### <number>"),
            answer: final_number(a).unwrap_or_default(),
        });
    }
    Ok(v)
}

/// Generates `n` arithmetic items (seeded).
///
/// # Example
/// ```
/// let items = aido::evals::arith_items(5, 1);
/// assert_eq!(items.len(), 5);
/// ```
pub fn arith_items(n: usize, seed: u64) -> Vec<Item> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|i| {
            let a = 10 + (r.uniform() * 890.0) as i64;
            let b = 2 + (r.uniform() * 97.0) as i64;
            let (expr, val) = match i % 3 {
                0 => (format!("{a} + {b}"), a + b),
                1 => (format!("{a} - {b}"), a - b),
                _ => (format!("{a} * {b}"), a * b),
            };
            Item { prompt: format!("Compute {expr}. Reply with only the number."), answer: val.to_string() }
        })
        .collect()
}

/// Generates `n` next-element sequence items (arithmetic, geometric,
/// squares, Fibonacci-like, alternating).
///
/// # Example
/// ```
/// let items = aido::evals::sequence_items(4, 3);
/// assert!(!items[0].answer.is_empty());
/// ```
pub fn sequence_items(n: usize, seed: u64) -> Vec<Item> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let (s, next) = crate::evals::random_sequence(&mut r, 6);
            let txt = s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
            Item { prompt: format!("What is the next number in the sequence {txt}, ...? Reply with only the number."), answer: next.to_string() }
        })
        .collect()
}

/// Random integer sequence of `len` terms plus its next term. Families:
/// arithmetic, geometric, squares + c, Fibonacci-like, alternating steps,
/// polynomial (2nd differences constant).
///
/// # Example
/// ```
/// let mut r = aido::sdk::Rng(1);
/// let (s, nx) = aido::evals::random_sequence(&mut r, 5);
/// assert_eq!(s.len(), 5);
/// ```
pub fn random_sequence(r: &mut Rng, len: usize) -> (Vec<i64>, i64) {
    let kind = (r.uniform() * 6.0) as usize % 6;
    let a = 1 + (r.uniform() * 9.0) as i64;
    let d = 1 + (r.uniform() * 9.0) as i64;
    let f = |i: i64| -> i64 {
        match kind {
            0 => a + d * i,
            1 => a * (2 + d % 2).pow(i as u32),
            2 => (i + 1) * (i + 1) + a,
            3 => {
                let (mut x, mut y) = (a, d);
                for _ in 0..i {
                    let z = x + y;
                    x = y;
                    y = z;
                }
                x
            }
            4 => a + (i / 2) * (d + 1) + (i % 2) * d,
            _ => a + d * i + i * (i - 1) / 2 * 2,
        }
    };
    let s: Vec<i64> = (0..len as i64).map(f).collect();
    (s, f(len as i64))
}

/// Runs items through the engine (chat template, greedy, thinking off).
///
/// * **Input** — engine, items, max new tokens, verbosity.
/// * **Output** — score.
///
/// # Example
/// ```no_run
/// use aido::{evals, llm::engine::*};
/// let mut eng = LlmEngine::load("models/Qwen3-0.6B-Q8_0.gguf", EngineOptions::default()).unwrap();
/// let s = evals::run(&mut eng, &evals::arith_items(30, 1), 16, false).unwrap();
/// println!("{:.1}%", s.percent());
/// ```
pub fn run(eng: &mut LlmEngine, items: &[Item], max_tokens: usize, verbose: bool) -> Result<Score> {
    let t = std::time::Instant::now();
    let mut correct = 0;
    for (i, it) in items.iter().enumerate() {
        let ids = eng.chat_prompt(&[Message::new("user", &it.prompt)], false)?;
        let mut out = String::new();
        eng.generate(&ids, &GenParams { max_tokens, spec: Some(SpecMode::Lookup), ..Default::default() }, &mut |s| {
            out.push_str(s);
            true
        })?;
        let got = final_number(&out).unwrap_or_default();
        let ok = got == it.answer;
        correct += ok as usize;
        if verbose {
            println!("[{i:>3}] {} expected {} got {got:<8} | {}", if ok { "OK " } else { "ERR" }, it.answer, out.replace('\n', " ").chars().take(80).collect::<String>());
        }
    }
    Ok(Score { correct, total: items.len(), secs: t.elapsed().as_secs_f64() })
}
