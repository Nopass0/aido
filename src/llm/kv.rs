//! # Long-context KV cache: paged, compressed, with block-sparse attention
//!
//! The attention cache is what limits long contexts. For an 8B model with
//! grouped-query attention (36 layers × 8 KV heads × 128 dims) one token
//! costs `2 · 36 · 1024` values:
//!
//! | Storage | Bytes / token | 128K tokens | 256K tokens |
//! |---------|--------------:|------------:|------------:|
//! | f32 | 295 KB | 37.7 GB | 75.5 GB |
//! | f16 (old default) | 147 KB | 18.9 GB | 37.7 GB |
//! | **Q8** (groups of 32, f16 scale) | 78 KB | 10.0 GB | 20.0 GB |
//! | **Q4** (V 4-bit + K 8-bit, groups of 32) | 59 KB | 7.6 GB | 15.2 GB |
//!
//! Three mechanisms make 128–256K tokens fit in ordinary RAM without
//! slowing decoding down:
//!
//! 1. **Paging** — memory is allocated in pages of [`PAGE`] positions as the
//!    context grows, never `n_ctx` up front: a 256K-capable session that
//!    holds 2K tokens uses 2K tokens of memory.
//! 2. **Quantisation** — K and V rows are stored as 8- or 4-bit integers in
//!    groups of 32 with an f16 scale (symmetric, `x ≈ s · q`). Keys stay
//!    8-bit even in Q4 mode (outlier channels; KIVI's observation), values
//!    go to 4 bits. Attention reads the integers directly (no dequantised copy).
//! 3. **Block-sparse attention** for long contexts (a Quest-style query-aware
//!    selection). Every page keeps the per-channel `min` and `max` of its
//!    keys. For a query `q` the score of any key in the page is bounded by
//!    `Σ_c max(q_c · min_c, q_c · max_c)`. Attention is computed **exactly**
//!    over the first [`KvConfig::sink`] positions (attention sinks), the
//!    last [`KvConfig::recent`] positions and the pages with the highest
//!    bounds, up to [`KvConfig::budget`] positions in total. Pages whose
//!    upper bound is low cannot carry much softmax mass, so the result is
//!    close to full attention while each token reads `budget` positions
//!    instead of the whole context — decode speed at 256K stays close to
//!    decode speed at 8K. Contexts up to `budget` positions use exact full
//!    attention; `budget = 0` disables sparsity.
//!
//! Per-page summaries cost `2 · kv_dim` f32 per page (≈ 6 % of a Q8 page).

use half::f16;

/// Positions per page (allocation and sparse-selection granularity).
pub const PAGE: usize = 64;
/// Values per quantisation group.
const GROUP: usize = 32;

/// Storage format of cached keys / values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    /// 16-bit floats (exact for practical purposes).
    F16,
    /// 8-bit integers, groups of 32 with an f16 scale (≈1.06 B/value).
    Q8,
    /// Values in 4-bit, keys in 8-bit (groups of 32, f16 scales): keys
    /// carry outlier channels that 4 bits destroy, values do not (≈0.81 B/value).
    Q4,
}

impl KvType {
    /// Bytes per row of `dim` values (data + scales).
    ///
    /// # Example
    /// ```
    /// use aido::llm::kv::KvType;
    /// assert_eq!(KvType::F16.row_bytes(64), 128);
    /// assert_eq!(KvType::Q8.row_bytes(64), 64 + 4);
    /// assert_eq!(KvType::Q4.row_bytes(64), 32 + 4);
    /// ```
    pub fn row_bytes(self, dim: usize) -> usize {
        Self::fmt_bytes(self, dim)
    }

    /// Storage format used for keys (Q4 keeps keys in Q8).
    ///
    /// # Example
    /// ```
    /// use aido::llm::kv::KvType;
    /// assert_eq!(KvType::Q4.key_type(), KvType::Q8);
    /// ```
    pub fn key_type(self) -> KvType {
        if self == KvType::Q4 { KvType::Q8 } else { self }
    }

    fn fmt_bytes(self, dim: usize) -> usize {
        match self {
            KvType::F16 => dim * 2,
            KvType::Q8 => dim + dim / GROUP * 2,
            KvType::Q4 => dim / 2 + dim / GROUP * 2,
        }
    }
}

impl std::str::FromStr for KvType {
    type Err = anyhow::Error;
    /// Parses `f16`, `q8` / `q8_0`, `q4` / `q4_0`.
    ///
    /// # Example
    /// ```
    /// assert_eq!("q4".parse::<aido::llm::kv::KvType>().unwrap(), aido::llm::kv::KvType::Q4);
    /// ```
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" => KvType::F16,
            "q8" | "q8_0" | "int8" => KvType::Q8,
            "q4" | "q4_0" | "int4" => KvType::Q4,
            o => anyhow::bail!("unknown KV type '{o}' (f16 | q8 | q4)"),
        })
    }
}

/// KV cache policy.
#[derive(Debug, Clone)]
pub struct KvConfig {
    /// Storage format.
    pub ty: KvType,
    /// Max positions attended per query in long contexts (0 = always exact).
    pub budget: usize,
    /// Always-attended first positions (attention sinks).
    pub sink: usize,
    /// Always-attended most recent positions.
    pub recent: usize,
}

impl Default for KvConfig {
    /// Q8 storage; exact attention up to 8192 positions, block-sparse above
    /// (64 sink + 2048 recent + best pages up to 8192 positions).
    fn default() -> Self {
        KvConfig { ty: KvType::Q8, budget: 8192, sink: PAGE, recent: 2048 }
    }
}

impl KvConfig {
    /// Bytes of cache per token for a model with `n_layer` layers and
    /// `kv_dim`-wide K/V rows (including page summaries).
    ///
    /// # Example
    /// ```
    /// use aido::llm::kv::{KvConfig, KvType};
    /// let c = KvConfig { ty: KvType::Q8, ..Default::default() };
    /// // Qwen3-8B: 36 layers, 8 KV heads × 128
    /// let gb = c.bytes_per_token(36, 1024) as f64 * 262144.0 / 1e9;
    /// assert!(gb < 22.0);
    /// ```
    pub fn bytes_per_token(&self, n_layer: usize, kv_dim: usize) -> usize {
        n_layer * (self.ty.key_type().row_bytes(kv_dim) + self.ty.row_bytes(kv_dim) + 2 * kv_dim * 4 / PAGE)
    }
}

/// One page: `PAGE` rows of K and V for one layer plus key summaries.
struct Page {
    k: Vec<u8>,
    v: Vec<u8>,
    kmin: Vec<f32>,
    kmax: Vec<f32>,
}

/// Paged, quantised KV cache for all layers of one sequence.
pub struct KvCache {
    /// Capacity in positions.
    pub n_ctx: usize,
    /// Row width (`n_head_kv · head_dim`).
    pub kv_dim: usize,
    n_layer: usize,
    /// Storage and attention policy.
    pub cfg: KvConfig,
    rowk: usize,
    rowv: usize,
    npages: usize,
    pages: Vec<Option<Box<Page>>>,
}

#[inline]
fn f16s(b: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

impl KvCache {
    /// A cache for `n_ctx` positions with the default policy (Q8, sparse
    /// above 8192). No memory is committed until positions are written.
    ///
    /// # Example
    /// ```
    /// let kv = aido::llm::model::KvCache::new(2, 16, 64);
    /// assert_eq!(kv.nbytes(), 0); // nothing allocated yet
    /// ```
    pub fn new(n_layer: usize, n_ctx: usize, kv_dim: usize) -> Self {
        Self::with_config(n_layer, n_ctx, kv_dim, KvConfig::default())
    }

    /// A cache with an explicit policy.
    ///
    /// # Example
    /// ```
    /// use aido::llm::kv::{KvCache, KvConfig, KvType};
    /// let kv = KvCache::with_config(36, 262144, 1024, KvConfig { ty: KvType::Q4, ..Default::default() });
    /// assert_eq!(kv.n_ctx, 262144);
    /// ```
    pub fn with_config(n_layer: usize, n_ctx: usize, kv_dim: usize, cfg: KvConfig) -> Self {
        let cfg = if kv_dim % GROUP != 0 { KvConfig { ty: KvType::F16, ..cfg } } else { cfg };
        let npages = n_ctx.div_ceil(PAGE);
        KvCache { n_ctx, kv_dim, n_layer, rowk: cfg.ty.key_type().row_bytes(kv_dim), rowv: cfg.ty.row_bytes(kv_dim), cfg, npages, pages: (0..n_layer * npages).map(|_| None).collect() }
    }

    /// Memory currently allocated (bytes).
    ///
    /// # Example
    /// ```
    /// let mut kv = aido::llm::model::KvCache::new(1, 1000, 64);
    /// kv.reserve(10);
    /// assert!(kv.nbytes() > 0);
    /// ```
    pub fn nbytes(&self) -> usize {
        self.pages.iter().flatten().map(|p| p.k.len() + p.v.len() + (p.kmin.len() + p.kmax.len()) * 4).sum()
    }

    /// Makes sure pages for positions `< end` exist in every layer.
    ///
    /// # Example
    /// ```
    /// let mut kv = aido::llm::model::KvCache::new(2, 1000, 64);
    /// kv.reserve(65); // two pages per layer
    /// ```
    pub fn reserve(&mut self, end: usize) {
        let need = end.min(self.n_ctx).div_ceil(PAGE);
        for l in 0..self.n_layer {
            for p in 0..need {
                let slot = &mut self.pages[l * self.npages + p];
                if slot.is_none() {
                    *slot = Some(Box::new(Page {
                        k: vec![0; PAGE * self.rowk],
                        v: vec![0; PAGE * self.rowv],
                        kmin: vec![f32::INFINITY; self.kv_dim],
                        kmax: vec![f32::NEG_INFINITY; self.kv_dim],
                    }));
                }
            }
        }
    }

    /// Releases pages wholly beyond position `len` (after a rewind).
    ///
    /// # Example
    /// ```
    /// let mut kv = aido::llm::model::KvCache::new(1, 1000, 64);
    /// kv.reserve(500);
    /// kv.truncate(10);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        let keep = len.div_ceil(PAGE);
        for l in 0..self.n_layer {
            for p in keep..self.npages {
                self.pages[l * self.npages + p] = None;
            }
        }
    }

    #[inline]
    fn page(&self, layer: usize, pos: usize) -> &Page {
        self.pages[layer * self.npages + pos / PAGE].as_deref().expect("KV page not reserved")
    }

    /// Encodes one row (`kv_dim` values) into `dst` in this cache's format.
    fn encode(ty: KvType, src: &[f32], dst: &mut [u8]) {
        match ty {
            KvType::F16 => {
                for (i, v) in src.iter().enumerate() {
                    dst[2 * i..2 * i + 2].copy_from_slice(&f16::from_f32(*v).to_bits().to_le_bytes());
                }
            }
            KvType::Q8 | KvType::Q4 => {
                let (bits_max, per) = if ty == KvType::Q8 { (127.0f32, GROUP) } else { (7.0f32, GROUP / 2) };
                let groups = src.len() / GROUP;
                let (data, scales) = dst.split_at_mut(groups * per);
                for g in 0..groups {
                    let x = &src[g * GROUP..(g + 1) * GROUP];
                    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let s = f16::from_f32(amax / bits_max);
                    let inv = if s.to_f32() > 0.0 { 1.0 / s.to_f32() } else { 0.0 };
                    scales[2 * g..2 * g + 2].copy_from_slice(&s.to_bits().to_le_bytes());
                    let d = &mut data[g * per..(g + 1) * per];
                    if ty == KvType::Q8 {
                        for (o, v) in d.iter_mut().zip(x) {
                            *o = ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8;
                        }
                    } else {
                        for i in 0..GROUP / 2 {
                            let lo = ((x[i] * inv).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                            let hi = ((x[i + GROUP / 2] * inv).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                            d[i] = lo | (hi << 4);
                        }
                    }
                }
            }
        }
    }

    /// Writes K and V rows of `layer` at `pos` (page must be reserved).
    ///
    /// # Safety
    /// Concurrent calls must target different positions; no reader may
    /// access those rows meanwhile. (Used from parallel per-token loops.)
    pub unsafe fn write(&self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let p = self.page(layer, pos);
        let r = pos % PAGE;
        let kd = std::slice::from_raw_parts_mut(p.k.as_ptr().add(r * self.rowk) as *mut u8, self.rowk);
        let vd = std::slice::from_raw_parts_mut(p.v.as_ptr().add(r * self.rowv) as *mut u8, self.rowv);
        Self::encode(self.cfg.ty.key_type(), k, kd);
        Self::encode(self.cfg.ty, v, vd);
    }

    /// Updates key summaries of `layer` with rows `keys` (`t × kv_dim`,
    /// the raw f32 keys) written at positions `pos0..pos0+t`.
    pub fn summarize(&mut self, layer: usize, pos0: usize, keys: &[f32]) {
        let d = self.kv_dim;
        for (i, k) in keys.chunks_exact(d).enumerate() {
            let pos = pos0 + i;
            let Some(p) = self.pages[layer * self.npages + pos / PAGE].as_deref_mut() else { continue };
            if pos % PAGE == 0 {
                p.kmin.fill(f32::INFINITY);
                p.kmax.fill(f32::NEG_INFINITY);
            }
            for c in 0..d {
                p.kmin[c] = p.kmin[c].min(k[c]);
                p.kmax[c] = p.kmax[c].max(k[c]);
            }
        }
    }

    /// `q · K[pos][ch0..ch0+q.len()]`.
    #[inline]
    fn dot_k(&self, p: &Page, r: usize, ch0: usize, q: &[f32]) -> f32 {
        row_dot(self.cfg.ty.key_type(), &p.k[r * self.rowk..(r + 1) * self.rowk], self.kv_dim, ch0, q)
    }

    /// `out += w · V[pos][ch0..ch0+out.len()]`.
    #[inline]
    fn axpy_v(&self, p: &Page, r: usize, ch0: usize, w: f32, out: &mut [f32]) {
        row_axpy(self.cfg.ty, &p.v[r * self.rowv..(r + 1) * self.rowv], self.kv_dim, ch0, w, out)
    }

    /// Ranges of positions `< n` that a query attends (exact or sparse).
    fn select(&self, layer: usize, ch0: usize, q: &[f32], hd: usize, n: usize) -> Vec<(usize, usize)> {
        let c = &self.cfg;
        if c.budget == 0 || n <= c.budget {
            return vec![(0, n)];
        }
        let sink_end = c.sink.div_ceil(PAGE) * PAGE;
        let recent_start = (n.saturating_sub(c.recent) / PAGE) * PAGE;
        if recent_start <= sink_end {
            return vec![(0, n)];
        }
        let slots = c.budget.saturating_sub(sink_end + (n - recent_start)) / PAGE;
        let (p0, p1) = (sink_end / PAGE, recent_start / PAGE);
        let mut cand: Vec<(f32, usize)> = (p0..p1)
            .map(|pi| {
                let pg = self.pages[layer * self.npages + pi].as_deref().unwrap();
                let (mn, mx) = (&pg.kmin[ch0..ch0 + hd], &pg.kmax[ch0..ch0 + hd]);
                // Upper bound of q·k over the page, maximised over the query heads of the group.
                let best = q.chunks_exact(hd).map(|qh| qh.iter().zip(mn.iter().zip(mx)).map(|(a, (lo, hi))| (a * lo).max(a * hi)).sum::<f32>()).fold(f32::NEG_INFINITY, f32::max);
                (best, pi)
            })
            .collect();
        let take = slots.min(cand.len());
        if take < cand.len() {
            cand.select_nth_unstable_by(take, |a, b| b.0.total_cmp(&a.0));
        }
        let mut chosen: Vec<usize> = cand[..take].iter().map(|x| x.1).collect();
        chosen.sort_unstable();
        let mut r = vec![(0, sink_end)];
        for pi in chosen {
            let (s, e) = (pi * PAGE, (pi + 1) * PAGE);
            if r.last().unwrap().1 == s {
                r.last_mut().unwrap().1 = e;
            } else {
                r.push((s, e));
            }
        }
        if r.last().unwrap().1 == recent_start {
            r.last_mut().unwrap().1 = n;
        } else {
            r.push((recent_start, n));
        }
        r
    }

    /// Grouped-query attention of `q` (`g` heads × `hd`, sharing KV head
    /// channels `ch0..ch0+hd`) over positions `< n` of `layer`.
    /// `scores` needs `g · n` floats; returns the number of attended positions.
    ///
    /// # Example
    /// ```
    /// use aido::llm::kv::{KvCache, KvConfig, KvType};
    /// let mut kv = KvCache::with_config(1, 128, 32, KvConfig { ty: KvType::F16, ..Default::default() });
    /// kv.reserve(2);
    /// let (k, v) = (vec![1.0; 32], vec![2.0; 32]);
    /// unsafe { kv.write(0, 0, &k, &v); kv.write(0, 1, &k, &v); }
    /// let mut out = vec![0.0; 32];
    /// let n = kv.attend(0, 0, &vec![0.1; 32], 32, 2, 1.0, &mut vec![0.0; 2], &mut out);
    /// assert_eq!(n, 2);
    /// assert!((out[0] - 2.0).abs() < 1e-3);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn attend(&self, layer: usize, ch0: usize, q: &[f32], hd: usize, n: usize, scale: f32, scores: &mut [f32], out: &mut [f32]) -> usize {
        let g = q.len() / hd;
        let mut st = vec![0f32; 2 * g];
        let m = self.attend_part(layer, ch0, q, hd, n, scale, 0, 1, scores, out, &mut st);
        for h in 0..g {
            let inv = if st[g + h] > 0.0 { 1.0 / st[g + h] } else { 0.0 };
            out[h * hd..(h + 1) * hd].iter_mut().for_each(|v| *v *= inv);
        }
        m
    }

    /// One slice (`part` of `nparts`) of [`KvCache::attend`] for split-K
    /// ("flash-decoding") parallelism: attends the `part`-th share of the
    /// selected positions and writes the **unnormalised** output
    /// `Σ exp(s − max)·v` to `out` and per-head `[max…, sum…]` to `stats`
    /// (`2·g` floats). Combine parts with [`merge_parts`]. Returns the total
    /// number of selected positions.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_part(&self, layer: usize, ch0: usize, q: &[f32], hd: usize, n: usize, scale: f32, part: usize, nparts: usize, scores: &mut [f32], out: &mut [f32], stats: &mut [f32]) -> usize {
        let g = q.len() / hd;
        let ranges = self.select(layer, ch0, q, hd, n);
        let total: usize = ranges.iter().map(|r| r.1 - r.0).sum();
        let (lo, hi) = (total * part / nparts, total * (part + 1) / nparts);
        let m = hi - lo;
        // Positions of this part, as (start, end) sub-ranges.
        let mut mine = Vec::new();
        let mut acc = 0;
        for &(s, e) in &ranges {
            let (a, b) = (acc, acc + e - s);
            let (x, y) = (a.max(lo), b.min(hi));
            if x < y {
                mine.push((s + x - a, s + y - a));
            }
            acc = b;
        }
        let mut idx = 0;
        for &(s, e) in &mine {
            let mut pos = s;
            while pos < e {
                let pg = self.page(layer, pos);
                let stop = e.min((pos / PAGE + 1) * PAGE);
                for p in pos..stop {
                    for h in 0..g {
                        scores[h * m + idx] = self.dot_k(pg, p % PAGE, ch0, &q[h * hd..(h + 1) * hd]) * scale;
                    }
                    idx += 1;
                }
                pos = stop;
            }
        }
        for h in 0..g {
            let sc = &mut scores[h * m..(h + 1) * m];
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in sc.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            stats[h] = mx;
            stats[g + h] = sum;
        }
        out.fill(0.0);
        let mut idx = 0;
        for &(s, e) in &mine {
            let mut pos = s;
            while pos < e {
                let pg = self.page(layer, pos);
                let stop = e.min((pos / PAGE + 1) * PAGE);
                for p in pos..stop {
                    for h in 0..g {
                        self.axpy_v(pg, p % PAGE, ch0, scores[h * m + idx], &mut out[h * hd..(h + 1) * hd]);
                    }
                    idx += 1;
                }
                pos = stop;
            }
        }
        total
    }
}

/// Merges split-K partial results (from [`KvCache::attend_part`]) into the
/// normalised attention output `out` (`g·hd`): with per-part maxima `mᵢ`
/// and sums `lᵢ`, `out = Σ e^{mᵢ−M} oᵢ / Σ e^{mᵢ−M} lᵢ` (log-sum-exp merge).
///
/// `parts[i] = (out_i, stats_i)`.
///
/// # Example
/// ```
/// // two parts holding one position each, equal scores → mean of values
/// let p1 = (vec![1.0f32; 4], vec![0.0, 1.0]);
/// let p2 = (vec![3.0f32; 4], vec![0.0, 1.0]);
/// let mut out = vec![0.0; 4];
/// aido::llm::kv::merge_parts(&[(&p1.0[..], &p1.1[..]), (&p2.0[..], &p2.1[..])], 4, &mut out);
/// assert!((out[0] - 2.0).abs() < 1e-6);
/// ```
pub fn merge_parts(parts: &[(&[f32], &[f32])], hd: usize, out: &mut [f32]) {
    let g = out.len() / hd;
    for h in 0..g {
        let mx = parts.iter().filter(|p| p.1[g + h] > 0.0).map(|p| p.1[h]).fold(f32::NEG_INFINITY, f32::max);
        let o = &mut out[h * hd..(h + 1) * hd];
        o.fill(0.0);
        let mut den = 0f32;
        for (po, ps) in parts {
            if ps[g + h] <= 0.0 {
                continue;
            }
            let w = (ps[h] - mx).exp();
            den += w * ps[g + h];
            for (a, b) in o.iter_mut().zip(&po[h * hd..(h + 1) * hd]) {
                *a += w * b;
            }
        }
        let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
        o.iter_mut().for_each(|v| *v *= inv);
    }
}

impl KvCache {
    /// Ranges attended by a block of queries ending at `n_max` (exclusive):
    /// exact up to the budget; beyond it the pages whose key bound is
    /// highest for **any** query/head of the block (shared selection).
    fn select_block(&self, layer: usize, ch0: usize, qs: &[f32], hd: usize, n_first: usize, n_max: usize) -> Vec<(usize, usize)> {
        let c = &self.cfg;
        if c.budget == 0 || n_max <= c.budget {
            return vec![(0, n_max)];
        }
        let sink_end = c.sink.div_ceil(PAGE) * PAGE;
        let recent_start = (n_first.saturating_sub(c.recent) / PAGE) * PAGE;
        if recent_start <= sink_end {
            return vec![(0, n_max)];
        }
        let slots = c.budget.saturating_sub(sink_end + (n_max - recent_start)) / PAGE;
        let mut cand: Vec<(f32, usize)> = (sink_end / PAGE..recent_start / PAGE)
            .map(|pi| {
                let pg = self.pages[layer * self.npages + pi].as_deref().unwrap();
                let (mn, mx) = (&pg.kmin[ch0..ch0 + hd], &pg.kmax[ch0..ch0 + hd]);
                let best = qs.chunks_exact(hd).map(|qh| qh.iter().zip(mn.iter().zip(mx)).map(|(a, (lo, hi))| (a * lo).max(a * hi)).sum::<f32>()).fold(f32::NEG_INFINITY, f32::max);
                (best, pi)
            })
            .collect();
        let take = slots.min(cand.len());
        if take < cand.len() {
            cand.select_nth_unstable_by(take, |a, b| b.0.total_cmp(&a.0));
        }
        let mut chosen: Vec<usize> = cand[..take].iter().map(|x| x.1).collect();
        chosen.sort_unstable();
        let mut r = vec![(0, sink_end)];
        for pi in chosen {
            let (s, e) = (pi * PAGE, (pi + 1) * PAGE);
            if r.last().unwrap().1 == s {
                r.last_mut().unwrap().1 = e;
            } else {
                r.push((s, e));
            }
        }
        if r.last().unwrap().1 == recent_start {
            r.last_mut().unwrap().1 = n_max;
        } else {
            r.push((recent_start, n_max));
        }
        r
    }

    /// Block ("flash") attention for prefill: `tq` consecutive queries at
    /// positions `pos0..pos0+tq`, each with `g` heads of `hd` sharing KV
    /// channels `ch0..`; query `j` attends positions `≤ pos0 + j`.
    /// `qs` is `tq × (g·hd)` (row stride `q_stride`), `out` likewise with
    /// `out_stride`. Each 64-position tile of K and V is decoded to f32
    /// once for the whole block and an online softmax accumulates the
    /// result, so the cost is dense f32 dot products instead of
    /// per-query decoding.
    ///
    /// # Example
    /// ```
    /// use aido::llm::kv::{KvCache, KvConfig, KvType};
    /// let mut kv = KvCache::with_config(1, 64, 32, KvConfig { ty: KvType::Q8, ..Default::default() });
    /// kv.reserve(3);
    /// for p in 0..3 { unsafe { kv.write(0, p, &vec![0.1 * p as f32; 32], &vec![p as f32; 32]) }; }
    /// let q = vec![0.0f32; 3 * 32];
    /// let mut out = vec![0.0f32; 3 * 32];
    /// kv.attend_block(0, 0, &q, 32, 0, 3, 32, 32, 1.0, &mut out, 32);
    /// assert!((out[2 * 32] - 1.0).abs() < 0.02); // query 2: mean of values 0,1,2
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn attend_block(&self, layer: usize, ch0: usize, qs: &[f32], hd: usize, pos0: usize, tq: usize, q_stride: usize, gw: usize, scale: f32, out: &mut [f32], out_stride: usize) {
        let g = gw / hd;
        let n_max = pos0 + tq;
        // Gather the block's queries contiguously (pre-scaled).
        let mut q = vec![0f32; tq * gw];
        for j in 0..tq {
            for (d, s) in q[j * gw..(j + 1) * gw].iter_mut().zip(&qs[j * q_stride..j * q_stride + gw]) {
                *d = s * scale;
            }
        }
        let ranges = self.select_block(layer, ch0, &q, hd, pos0 + 1, n_max);
        let rows = tq * g;
        let mut m = vec![f32::NEG_INFINITY; rows];
        let mut l = vec![0f32; rows];
        let mut acc = vec![0f32; rows * hd];
        let mut kt = vec![0f32; PAGE * hd];
        let mut vt = vec![0f32; PAGE * hd];
        let mut sc = vec![0f32; PAGE];
        for &(s, e) in &ranges {
            let mut pos = s;
            while pos < e {
                let pg = self.page(layer, pos);
                let stop = e.min((pos / PAGE + 1) * PAGE);
                let n = stop - pos;
                for r in 0..n {
                    let rr = (pos + r) % PAGE;
                    row_decode(self.cfg.ty.key_type(), &pg.k[rr * self.rowk..(rr + 1) * self.rowk], self.kv_dim, ch0, &mut kt[r * hd..(r + 1) * hd]);
                    row_decode(self.cfg.ty, &pg.v[rr * self.rowv..(rr + 1) * self.rowv], self.kv_dim, ch0, &mut vt[r * hd..(r + 1) * hd]);
                }
                for j in 0..tq {
                    let lim = pos0 + j + 1; // causal: positions < lim
                    if pos >= lim {
                        continue;
                    }
                    let nn = n.min(lim - pos);
                    for h in 0..g {
                        let row = j * g + h;
                        let qh = &q[j * gw + h * hd..j * gw + (h + 1) * hd];
                        let mut mx = m[row];
                        for r in 0..nn {
                            let v = dot8(qh, &kt[r * hd..(r + 1) * hd]);
                            sc[r] = v;
                            mx = mx.max(v);
                        }
                        let corr = (m[row] - mx).exp();
                        let a = &mut acc[row * hd..(row + 1) * hd];
                        if corr != 1.0 {
                            a.iter_mut().for_each(|x| *x *= corr);
                        }
                        let mut sum = l[row] * corr;
                        for r in 0..nn {
                            let w = (sc[r] - mx).exp();
                            sum += w;
                            for (x, v) in a.iter_mut().zip(&vt[r * hd..(r + 1) * hd]) {
                                *x += w * v;
                            }
                        }
                        m[row] = mx;
                        l[row] = sum;
                    }
                }
                pos = stop;
            }
        }
        for j in 0..tq {
            for h in 0..g {
                let row = j * g + h;
                let inv = if l[row] > 0.0 { 1.0 / l[row] } else { 0.0 };
                for (o, a) in out[j * out_stride + h * hd..j * out_stride + (h + 1) * hd].iter_mut().zip(&acc[row * hd..(row + 1) * hd]) {
                    *o = a * inv;
                }
            }
        }
    }
}

/// Dot product with 8 accumulators (vectorises).
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Decodes channels `ch0..ch0+out.len()` of an encoded row to f32.
#[inline]
fn row_decode(ty: KvType, row: &[u8], dim: usize, ch0: usize, out: &mut [f32]) {
    match ty {
        KvType::F16 => {
            let r: &[f16] = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const f16, dim) };
            let n = out.len();
            for (o, v) in out.iter_mut().zip(&r[ch0..ch0 + n]) {
                *o = v.to_f32();
            }
        }
        KvType::Q8 => {
            let (data, sc) = row.split_at(dim);
            for (gi, og) in out.chunks_exact_mut(GROUP).enumerate() {
                let g = ch0 / GROUP + gi;
                let s = f16s(sc, g);
                for (o, b) in og.iter_mut().zip(&data[g * GROUP..(g + 1) * GROUP]) {
                    *o = s * (*b as i8) as f32;
                }
            }
        }
        KvType::Q4 => {
            let (data, sc) = row.split_at(dim / 2);
            for (gi, og) in out.chunks_exact_mut(GROUP).enumerate() {
                let g = ch0 / GROUP + gi;
                let s = f16s(sc, g);
                let d = &data[g * GROUP / 2..(g + 1) * GROUP / 2];
                let (lo, hi) = og.split_at_mut(GROUP / 2);
                for i in 0..GROUP / 2 {
                    lo[i] = s * ((d[i] & 15) as i32 - 8) as f32;
                    hi[i] = s * ((d[i] >> 4) as i32 - 8) as f32;
                }
            }
        }
    }
}

/// Dot of `q` with channels `ch0..ch0+q.len()` of an encoded row.
#[inline]
fn row_dot(ty: KvType, row: &[u8], dim: usize, ch0: usize, q: &[f32]) -> f32 {
    match ty {
        KvType::F16 => {
            let r: &[f16] = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const f16, dim) };
            crate::tensor::ops::dot_f32_f16(q, &r[ch0..ch0 + q.len()])
        }
        KvType::Q8 => {
            let (data, sc) = row.split_at(dim);
            let mut s = 0f32;
            for (gi, qg) in q.chunks_exact(GROUP).enumerate() {
                let g = ch0 / GROUP + gi;
                let d = &data[g * GROUP..(g + 1) * GROUP];
                let mut acc = [0f32; 8];
                for i in 0..GROUP {
                    acc[i % 8] += qg[i] * (d[i] as i8) as f32;
                }
                s += acc.iter().sum::<f32>() * f16s(sc, g);
            }
            s
        }
        KvType::Q4 => {
            let (data, sc) = row.split_at(dim / 2);
            let mut s = 0f32;
            for (gi, qg) in q.chunks_exact(GROUP).enumerate() {
                let g = ch0 / GROUP + gi;
                let d = &data[g * GROUP / 2..(g + 1) * GROUP / 2];
                let mut acc = [0f32; 8];
                for i in 0..GROUP / 2 {
                    acc[i % 8] += qg[i] * ((d[i] & 15) as i32 - 8) as f32 + qg[i + GROUP / 2] * ((d[i] >> 4) as i32 - 8) as f32;
                }
                s += acc.iter().sum::<f32>() * f16s(sc, g);
            }
            s
        }
    }
}

/// `out += w · row[ch0..ch0+out.len()]`.
#[inline]
fn row_axpy(ty: KvType, row: &[u8], dim: usize, ch0: usize, w: f32, out: &mut [f32]) {
    match ty {
        KvType::F16 => {
            let r: &[f16] = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const f16, dim) };
            crate::tensor::ops::axpy_f16(out, w, &r[ch0..ch0 + out.len()]);
        }
        KvType::Q8 => {
            let (data, sc) = row.split_at(dim);
            for (gi, og) in out.chunks_exact_mut(GROUP).enumerate() {
                let g = ch0 / GROUP + gi;
                let ws = w * f16s(sc, g);
                for (o, b) in og.iter_mut().zip(&data[g * GROUP..(g + 1) * GROUP]) {
                    *o += ws * (*b as i8) as f32;
                }
            }
        }
        KvType::Q4 => {
            let (data, sc) = row.split_at(dim / 2);
            for (gi, og) in out.chunks_exact_mut(GROUP).enumerate() {
                let g = ch0 / GROUP + gi;
                let ws = w * f16s(sc, g);
                let d = &data[g * GROUP / 2..(g + 1) * GROUP / 2];
                let (lo, hi) = og.split_at_mut(GROUP / 2);
                for i in 0..GROUP / 2 {
                    lo[i] += ws * ((d[i] & 15) as i32 - 8) as f32;
                    hi[i] += ws * ((d[i] >> 4) as i32 - 8) as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    fn reference(keys: &[Vec<f32>], vals: &[Vec<f32>], q: &[f32], scale: f32) -> Vec<f32> {
        let s: Vec<f32> = keys.iter().map(|k| k.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let mut out = vec![0f32; q.len()];
        for (w, v) in e.iter().zip(vals) {
            for (o, x) in out.iter_mut().zip(v) {
                *o += w / z * x;
            }
        }
        out
    }

    /// Keys: small noise plus two "needle" keys (positions 300 and n/2)
    /// pointing in a distinct direction; the query looks for the needles.
    fn run(ty: KvType, budget: usize, n: usize) -> f32 {
        let hd = 64;
        let cfg = KvConfig { ty, budget, sink: PAGE, recent: 256 };
        let mut kv = KvCache::with_config(1, n, hd, cfg);
        kv.reserve(n);
        let u: Vec<f32> = (0..hd).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for p in 0..n {
            let mut k: Vec<f32> = rnd(hd, p as u64 * 2 + 1).iter().map(|x| x * 0.2).collect();
            if p == 300 || p == n / 2 {
                k = u.clone();
            }
            let v = rnd(hd, p as u64 * 2 + 2);
            unsafe { kv.write(0, p, &k, &v) };
            kv.summarize(0, p, &k);
            ks.push(k);
            vs.push(v);
        }
        let q: Vec<f32> = u.iter().map(|x| x * 3.0).collect();
        let r = reference(&ks, &vs, &q, 0.125);
        let mut out = vec![0f32; hd];
        kv.attend(0, 0, &q, hd, n, 0.125, &mut vec![0.0; n], &mut out);
        out.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum::<f32>() / r.iter().map(|x| x.abs()).sum::<f32>()
    }

    #[test]
    fn quantised_exact_attention_is_close() {
        assert!(run(KvType::F16, 0, 700) < 2e-3);
        assert!(run(KvType::Q8, 0, 700) < 2e-2);
        assert!(run(KvType::Q4, 0, 700) < 0.1);
    }

    #[test]
    fn sparse_attention_finds_needles() {
        // 4096 positions, only 1024 attended: the needles at 300 and 2048 must be selected.
        let e = run(KvType::Q8, 1024, 4096);
        assert!(e < 0.05, "relative error {e}");
    }

    #[test]
    fn split_k_matches_single_pass() {
        let (hd, n, g) = (64, 1500, 2);
        let mut kv = KvCache::with_config(1, n, hd, KvConfig { ty: KvType::Q8, budget: 0, ..Default::default() });
        kv.reserve(n);
        for p in 0..n {
            let (k, v) = (rnd(hd, p as u64 * 2 + 1), rnd(hd, p as u64 * 2 + 2));
            unsafe { kv.write(0, p, &k, &v) };
            kv.summarize(0, p, &k);
        }
        let q = rnd(g * hd, 99).iter().map(|x| x * 8.0).collect::<Vec<_>>();
        let mut one = vec![0f32; g * hd];
        kv.attend(0, 0, &q, hd, n, 0.125, &mut vec![0.0; g * n], &mut one);
        let parts: Vec<(Vec<f32>, Vec<f32>)> = (0..5)
            .map(|i| {
                let (mut o, mut st) = (vec![0f32; g * hd], vec![0f32; 2 * g]);
                kv.attend_part(0, 0, &q, hd, n, 0.125, i, 5, &mut vec![0.0; g * n], &mut o, &mut st);
                (o, st)
            })
            .collect();
        let refs: Vec<(&[f32], &[f32])> = parts.iter().map(|(a, b)| (&a[..], &b[..])).collect();
        let mut merged = vec![0f32; g * hd];
        merge_parts(&refs, hd, &mut merged);
        for (a, b) in one.iter().zip(&merged) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn block_attention_matches_per_query() {
        let (hd, g, n) = (64, 2, 300);
        let mut kv = KvCache::with_config(1, n, hd, KvConfig { ty: KvType::Q8, budget: 0, ..Default::default() });
        kv.reserve(n);
        for p in 0..n {
            let (k, v) = (rnd(hd, p as u64 * 2 + 1), rnd(hd, p as u64 * 2 + 2));
            unsafe { kv.write(0, p, &k, &v) };
        }
        let (pos0, tq) = (250, 50);
        let q: Vec<f32> = rnd(tq * g * hd, 5).iter().map(|x| x * 6.0).collect();
        let mut blk = vec![0f32; tq * g * hd];
        kv.attend_block(0, 0, &q, hd, pos0, tq, g * hd, g * hd, 0.125, &mut blk, g * hd);
        for j in 0..tq {
            let mut one = vec![0f32; g * hd];
            kv.attend(0, 0, &q[j * g * hd..(j + 1) * g * hd], hd, pos0 + j + 1, 0.125, &mut vec![0.0; g * n], &mut one);
            for (a, b) in one.iter().zip(&blk[j * g * hd..(j + 1) * g * hd]) {
                assert!((a - b).abs() < 1e-4, "query {j}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn paging_allocates_lazily() {
        let mut kv = KvCache::new(4, 262_144, 1024);
        assert_eq!(kv.nbytes(), 0);
        kv.reserve(1000);
        let used = kv.nbytes();
        // 1000 positions = 16 pages × 4 layers, not 262144 positions.
        assert!(used < 10_000_000, "{used}");
        kv.truncate(0);
        assert_eq!(kv.nbytes(), 0);
    }
}
