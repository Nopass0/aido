//! Token sampling (greedy, temperature, top-k, top-p, min-p, repetition
//! penalty) plus the exact acceptance rule for speculative decoding.

/// Sampling hyper-parameters.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Softmax temperature; `0` = greedy (argmax).
    pub temperature: f32,
    /// Keep only the `k` most likely tokens (`0` = disabled).
    pub top_k: usize,
    /// Nucleus sampling mass (`1.0` = disabled).
    pub top_p: f32,
    /// Drop tokens with `p < min_p · p_max` (`0` = disabled).
    pub min_p: f32,
    /// Penalty for recently generated tokens (`1.0` = disabled).
    pub repeat_penalty: f32,
    /// How many recent tokens the penalty looks at.
    pub repeat_last_n: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for SamplerConfig {
    /// Greedy decoding with no penalties.
    fn default() -> Self {
        SamplerConfig { temperature: 0.0, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.0, repeat_last_n: 64, seed: 42 }
    }
}

/// A seeded sampler.
pub struct Sampler {
    /// Parameters.
    pub cfg: SamplerConfig,
    state: u64,
    scratch: Vec<(u32, f32)>,
}

impl Sampler {
    /// Creates a sampler.
    ///
    /// # Example
    /// ```
    /// use aido::llm::sampler::{Sampler, SamplerConfig};
    /// let mut s = Sampler::new(SamplerConfig::default());
    /// assert_eq!(s.sample(&[0.1, 2.0, 0.3], &[]), 1);
    /// ```
    pub fn new(cfg: SamplerConfig) -> Self {
        let state = cfg.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        Sampler { cfg, state, scratch: Vec::new() }
    }

    /// `true` when sampling is deterministic argmax.
    ///
    /// # Example
    /// ```
    /// assert!(aido::llm::sampler::Sampler::new(Default::default()).greedy());
    /// ```
    pub fn greedy(&self) -> bool {
        self.cfg.temperature <= 0.0
    }

    /// Uniform random number in `[0, 1)` (xorshift64*).
    ///
    /// # Example
    /// ```
    /// let mut s = aido::llm::sampler::Sampler::new(Default::default());
    /// let u = s.uniform();
    /// assert!((0.0..1.0).contains(&u));
    /// ```
    pub fn uniform(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let r = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (r >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Applies the repetition penalty to `logits` in place.
    fn penalize(&self, logits: &mut [f32], history: &[u32]) {
        let p = self.cfg.repeat_penalty;
        if p == 1.0 {
            return;
        }
        let start = history.len().saturating_sub(self.cfg.repeat_last_n);
        for &t in &history[start..] {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / p } else { *l * p };
            }
        }
    }

    /// Computes the filtered target distribution as `(token, prob)` pairs
    /// sorted by probability (descending).
    ///
    /// * **Input** — raw logits (modified in place by the penalty), history.
    /// * **Output** — a slice into internal scratch; probabilities sum to 1.
    ///
    /// # Example
    /// ```
    /// use aido::llm::sampler::{Sampler, SamplerConfig};
    /// let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 2, top_p: 1.0, min_p: 0.0, ..Default::default() });
    /// let mut l = [0.0f32, 1.0, 2.0];
    /// let d = s.distribution(&mut l, &[]).to_vec();
    /// assert_eq!(d.len(), 2);
    /// assert_eq!(d[0].0, 2);
    /// ```
    pub fn distribution(&mut self, logits: &mut [f32], history: &[u32]) -> &[(u32, f32)] {
        self.penalize(logits, history);
        let temp = self.cfg.temperature.max(1e-5);
        let k = if self.cfg.top_k == 0 { logits.len() } else { self.cfg.top_k.min(logits.len()) };
        // Partial selection of the top-k by logit.
        self.scratch.clear();
        self.scratch.extend(logits.iter().enumerate().map(|(i, &l)| (i as u32, l)));
        if k < self.scratch.len() {
            self.scratch.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.scratch.truncate(k);
        }
        self.scratch.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let m = self.scratch[0].1;
        let mut sum = 0.0;
        for e in self.scratch.iter_mut() {
            e.1 = ((e.1 - m) / temp).exp();
            sum += e.1;
        }
        for e in self.scratch.iter_mut() {
            e.1 /= sum;
        }
        // min-p
        if self.cfg.min_p > 0.0 {
            let thr = self.scratch[0].1 * self.cfg.min_p;
            let keep = self.scratch.iter().take_while(|e| e.1 >= thr).count().max(1);
            self.scratch.truncate(keep);
        }
        // top-p
        if self.cfg.top_p < 1.0 {
            let mut c = 0.0;
            let mut keep = self.scratch.len();
            for (i, e) in self.scratch.iter().enumerate() {
                c += e.1;
                if c >= self.cfg.top_p {
                    keep = i + 1;
                    break;
                }
            }
            self.scratch.truncate(keep);
        }
        let s: f32 = self.scratch.iter().map(|e| e.1).sum();
        for e in self.scratch.iter_mut() {
            e.1 /= s;
        }
        &self.scratch
    }

    /// Picks the next token.
    ///
    /// * **Input** — logits for one position (modified in place), history
    ///   for the repetition penalty.
    /// * **Output** — chosen token id.
    ///
    /// # Example
    /// ```
    /// let mut s = aido::llm::sampler::Sampler::new(Default::default());
    /// assert_eq!(s.sample(&[5.0, 1.0], &[]), 0);
    /// ```
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let mut l = logits.to_vec();
        if self.greedy() {
            self.penalize(&mut l, history);
            return argmax(&l);
        }
        let u = self.uniform();
        let d = self.distribution(&mut l, history);
        let mut c = 0.0;
        for &(t, p) in d {
            c += p;
            if u < c {
                return t;
            }
        }
        d.last().map(|x| x.0).unwrap_or(0)
    }

    /// Speculative verification of a deterministic draft token.
    ///
    /// For a draft that proposes `x` with certainty (n-gram lookup or greedy
    /// draft model), accepting `x` with probability `p(x)` and otherwise
    /// sampling from `p` with `x` removed reproduces **exactly** the target
    /// distribution `p` (rejection sampling with a one-hot proposal), so
    /// speculation never changes output quality.
    ///
    /// * **Input** — logits of the target at this position, the draft token.
    /// * **Output** — `(accepted, token)`; if rejected, `token` is the
    ///   correction sampled from the residual distribution.
    ///
    /// # Example
    /// ```
    /// let mut s = aido::llm::sampler::Sampler::new(Default::default()); // greedy
    /// assert_eq!(s.verify(&[0.0, 9.0], &[], 1), (true, 1));
    /// assert_eq!(s.verify(&[0.0, 9.0], &[], 0), (false, 1));
    /// ```
    pub fn verify(&mut self, logits: &[f32], history: &[u32], draft: u32) -> (bool, u32) {
        if self.greedy() {
            let t = self.sample(logits, history);
            return (t == draft, t);
        }
        let u = self.uniform();
        let u2 = self.uniform();
        let mut l = logits.to_vec();
        let d = self.distribution(&mut l, history).to_vec();
        let px = d.iter().find(|e| e.0 == draft).map(|e| e.1).unwrap_or(0.0);
        if u < px {
            return (true, draft);
        }
        let rest: f32 = 1.0 - px;
        let mut c = 0.0;
        for &(t, p) in &d {
            if t == draft {
                continue;
            }
            c += p / rest.max(1e-12);
            if u2 < c {
                return (false, t);
            }
        }
        (false, d.iter().find(|e| e.0 != draft).map(|e| e.0).unwrap_or(draft))
    }
}

/// Index of the largest value.
///
/// # Example
/// ```
/// assert_eq!(aido::llm::sampler::argmax(&[1.0, 3.0, 2.0]), 1);
/// ```
pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}
