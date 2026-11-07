//! Adaptive load governor: picks thread counts from measurements and keeps
//! the machine out of thermal throttling.
//!
//! ## The model
//! For memory-bound decoding the time per token is
//! `t(n) ≈ max(W / B(n), C / (n·f))` where `W` = bytes of weights, `B(n)` =
//! DRAM bandwidth reachable with `n` cores and `C/(n·f)` the compute part.
//! `B(n)` saturates at a few cores (DRAM channels are the limit), so beyond
//! the knee extra threads add heat, power and cache contention but no speed.
//! The governor *measures* `t(n)` on the real model at load time and picks
//! the smallest `n` within 3 % of the best — typically 50–70 % of the cores,
//! leaving the rest parked (cool) and available to the OS and other apps.
//!
//! Prompt processing / batched verification is compute-bound, so it uses
//! every physical core (separately calibrated width).
//!
//! ## Thermal guard
//! [`ThermalGuard`] tracks an exponential moving average of the step time.
//! If it drifts more than `slowdown` above the cool baseline for a sustained
//! period (the signature of clock throttling), it reduces the pool width by
//! one and, if needed, inserts short idle gaps (duty-cycling) so the package
//! cools instead of oscillating at its thermal limit.

use std::time::{Duration, Instant};

use super::pool::ThreadPool;

/// Measures `f` for each candidate pool width and returns the smallest width
/// whose median time is within `tolerance` (e.g. `0.03`) of the fastest.
///
/// * **Input** — `pool`; `candidates`: widths to try (ascending); `reps`:
///   timed repetitions per width; `f`: the workload (should take ≥ 1 ms).
/// * **Output** — `(chosen_width, Vec<(width, median_seconds)>)`.
/// * **Side effects** — leaves the pool at the chosen width.
///
/// # Example
/// ```
/// use aido::runtime::{pool::ThreadPool, governor::calibrate_width};
/// let pool = ThreadPool::new(4, None);
/// let (w, table) = calibrate_width(&pool, &[1, 2, 4], 3, 0.03, || {
///     let v: u64 = (0..20000u64).sum();
///     std::hint::black_box(v);
/// });
/// assert!(w >= 1 && table.len() == 3);
/// ```
pub fn calibrate_width<F: FnMut()>(pool: &ThreadPool, candidates: &[usize], reps: usize, tolerance: f64, mut f: F) -> (usize, Vec<(usize, f64)>) {
    let mut table = Vec::new();
    for &w in candidates {
        pool.set_width(w);
        f(); // warm-up
        let mut ts: Vec<f64> = (0..reps.max(1))
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        table.push((w, ts[ts.len() / 2]));
    }
    let best = table.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    let chosen = table.iter().find(|x| x.1 <= best * (1.0 + tolerance)).map(|x| x.0).unwrap_or(pool.n_threads());
    pool.set_width(chosen);
    (chosen, table)
}

/// Default candidate widths for a machine with `n` threads available.
///
/// # Example
/// ```
/// assert_eq!(aido::runtime::governor::candidate_widths(12), vec![2, 4, 6, 8, 10, 12]);
/// ```
pub fn candidate_widths(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (1..=n).filter(|w| w % 2 == 0 || *w == n).collect();
    if v.is_empty() {
        v.push(1);
    }
    v
}

/// Detects sustained slowdowns (thermal throttling) and backs off.
#[derive(Debug, Clone)]
pub struct ThermalGuard {
    baseline: Option<f64>,
    ema: f64,
    alpha: f64,
    /// Relative slowdown that triggers a back-off (e.g. 0.25 = 25 %).
    pub slowdown: f64,
    hot_steps: u32,
    /// Idle gap inserted after each step when duty-cycling.
    pub cooldown: Duration,
    samples: u32,
    /// Number of back-offs performed so far.
    pub backoffs: u32,
}

impl ThermalGuard {
    /// Creates a guard; `slowdown` is the relative drift that counts as hot.
    ///
    /// # Example
    /// ```
    /// let g = aido::runtime::governor::ThermalGuard::new(0.25);
    /// assert_eq!(g.backoffs, 0);
    /// ```
    pub fn new(slowdown: f64) -> Self {
        ThermalGuard { baseline: None, ema: 0.0, alpha: 0.05, slowdown, hot_steps: 0, cooldown: Duration::ZERO, samples: 0, backoffs: 0 }
    }

    /// Records one step of `secs` (normalised per unit of work by the caller)
    /// and adjusts `pool` when sustained throttling is detected.
    ///
    /// * **Input** — step duration in seconds; pool to adapt.
    /// * **Output** — `true` if an adjustment was made.
    /// * **Behaviour** — the baseline is the minimum EMA seen after a warm-up
    ///   of 16 samples; after 64 consecutive "hot" samples the pool width is
    ///   reduced by one (never below half of the calibrated width) and a 1 ms
    ///   cool-down gap is introduced.
    ///
    /// # Example
    /// ```
    /// use aido::runtime::{pool::ThreadPool, governor::ThermalGuard};
    /// let pool = ThreadPool::new(2, None);
    /// let mut g = ThermalGuard::new(0.25);
    /// for _ in 0..20 { g.observe(0.01, &pool); }
    /// assert!(!g.observe(0.01, &pool));
    /// ```
    pub fn observe(&mut self, secs: f64, pool: &ThreadPool) -> bool {
        self.samples += 1;
        self.ema = if self.samples == 1 { secs } else { self.ema * (1.0 - self.alpha) + secs * self.alpha };
        if self.samples < 16 {
            return false;
        }
        let base = *self.baseline.get_or_insert(self.ema);
        if self.ema < base {
            self.baseline = Some(self.ema);
            self.hot_steps = 0;
            return false;
        }
        if self.ema > base * (1.0 + self.slowdown) {
            self.hot_steps += 1;
        } else {
            self.hot_steps = 0;
        }
        if self.hot_steps >= 64 {
            self.hot_steps = 0;
            self.backoffs += 1;
            let w = pool.width();
            if w > (pool.n_threads() / 2).max(1) {
                pool.set_width(w - 1);
            }
            self.cooldown = (self.cooldown + Duration::from_millis(1)).min(Duration::from_millis(10));
            return true;
        }
        false
    }

    /// Sleeps for the current cool-down gap (no-op when not throttled).
    ///
    /// # Example
    /// ```
    /// aido::runtime::governor::ThermalGuard::new(0.2).cool();
    /// ```
    pub fn cool(&self) {
        if !self.cooldown.is_zero() {
            std::thread::sleep(self.cooldown);
        }
    }
}
