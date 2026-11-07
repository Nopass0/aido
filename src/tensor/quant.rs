//! Quantised weight formats (GGML-compatible) and SIMD dot-product kernels.
//!
//! ## Why integer kernels
//! Generating one token of an 8 B-parameter model touches every weight once.
//! In f32 that is 32 GB of traffic per token — hopeless. In 4.5-bit `Q4_K`
//! it is ~4.9 GB, so the *storage format* already gives a 7× speed-up on a
//! bandwidth-bound machine. To keep compute from becoming the new bottleneck
//! the activation vector is quantised **once per matmul** to signed 8-bit
//! blocks of 32 ([`Q8Act`]) and the inner loop runs on integer AVX2
//! instructions (`vpmaddubsw` = 32 multiply-adds per instruction, 4× the
//! density of f32 FMA). Block scales are applied once per 32 elements.
//!
//! The activation layout stores, per block of 32, the scale `d` **and the
//! pre-scaled block sum `s = d·Σq`**. Formats with an additive minimum
//! (`Q4_1`, `Q5_1`, `Q4_K`, `Q5_K`) then need only one scalar FMA per block
//! for their min term: `Σ (a·q_w + m)·(d·q_x) = a·d·Σq_w q_x + m·s`.
//!
//! Every kernel has a portable scalar reference implementation; AVX2/FMA/F16C
//! versions are selected at runtime and verified against it in the tests.

use half::{bf16, f16};

/// GGML tensor element types supported by the engine.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// IEEE half.
    F16,
    /// bfloat16.
    BF16,
    /// 4-bit, block 32, scale.
    Q4_0,
    /// 4-bit, block 32, scale + min.
    Q4_1,
    /// 5-bit, block 32, scale.
    Q5_0,
    /// 5-bit, block 32, scale + min.
    Q5_1,
    /// 8-bit, block 32, scale.
    Q8_0,
    /// 4-bit k-quant, super-block 256 with 6-bit sub-scales/mins.
    Q4_K,
    /// 5-bit k-quant.
    Q5_K,
    /// 6-bit k-quant, 16 sub-blocks with 8-bit scales.
    Q6_K,
}

impl GgmlType {
    /// Converts the numeric GGML type id found in GGUF files.
    ///
    /// * **Input** — `id`: GGML enum value. **Output** — `Some(type)` if supported.
    ///
    /// # Example
    /// ```
    /// use aido::tensor::quant::GgmlType;
    /// assert_eq!(GgmlType::from_id(12), Some(GgmlType::Q4_K));
    /// assert_eq!(GgmlType::from_id(999), None);
    /// ```
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            _ => return None,
        })
    }

    /// Human readable name for an arbitrary GGML id (for error messages).
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::tensor::quant::GgmlType::name_of(10), "Q2_K");
    /// ```
    pub fn name_of(id: u32) -> &'static str {
        match id {
            0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
            8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
            14 => "Q6_K", 15 => "Q8_K", 16..=29 => "IQ*", 30 => "BF16", _ => "unknown",
        }
    }

    /// Number of elements per block.
    ///
    /// # Example
    /// ```
    /// use aido::tensor::quant::GgmlType;
    /// assert_eq!(GgmlType::Q6_K.block_elems(), 256);
    /// assert_eq!(GgmlType::F16.block_elems(), 1);
    /// ```
    pub fn block_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
        }
    }

    /// Number of bytes per block.
    ///
    /// # Example
    /// ```
    /// use aido::tensor::quant::GgmlType;
    /// assert_eq!(GgmlType::Q4_K.block_bytes(), 144);
    /// ```
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    /// Bytes needed to store one row of `n` elements.
    ///
    /// * **Input** — `n` must be a multiple of [`GgmlType::block_elems`].
    ///
    /// # Example
    /// ```
    /// use aido::tensor::quant::GgmlType;
    /// assert_eq!(GgmlType::Q8_0.row_bytes(64), 68);
    /// ```
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_elems() * self.block_bytes()
    }

    /// Average storage bits per weight (e.g. 4.5 for `Q4_K`).
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::tensor::quant::GgmlType::Q4_K.bits_per_weight(), 4.5);
    /// ```
    pub fn bits_per_weight(self) -> f64 {
        self.block_bytes() as f64 * 8.0 / self.block_elems() as f64
    }
}

// ---------------------------------------------------------------------------
// Activation quantisation
// ---------------------------------------------------------------------------

/// An activation vector quantised to symmetric int8 blocks of 32.
///
/// `qs[i]` are the quantised values, `d[b]` the per-block scale and
/// `s[b] = d[b] * Σ qs[block b]` the scaled block sum used by min-formats.
#[derive(Debug, Clone, Default)]
pub struct Q8Act {
    /// Quantised values, `len = n`.
    pub qs: Vec<i8>,
    /// Block scales, `len = n/32`.
    pub d: Vec<f32>,
    /// Scaled block sums, `len = n/32`.
    pub s: Vec<f32>,
}

impl Q8Act {
    /// Allocates an activation buffer for vectors of `n` elements.
    ///
    /// # Example
    /// ```
    /// let a = aido::tensor::quant::Q8Act::with_len(64);
    /// assert_eq!(a.qs.len(), 64);
    /// assert_eq!(a.d.len(), 2);
    /// ```
    pub fn with_len(n: usize) -> Self {
        Q8Act { qs: vec![0; n], d: vec![0.0; n / 32], s: vec![0.0; n / 32] }
    }

    /// Quantises `x` into `self` (resizing if needed).
    ///
    /// * **Input** — `x`: f32 vector, `len % 32 == 0`.
    /// * **Behaviour** — `d = max|x| / 127`, `q = round(x / d)`; uses AVX2
    ///   when available.
    ///
    /// # Example
    /// ```
    /// use aido::tensor::quant::Q8Act;
    /// let x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    /// let mut a = Q8Act::default();
    /// a.quantize(&x);
    /// assert!((a.qs[0] as f32 * a.d[0] - -16.0).abs() < 0.1);
    /// ```
    pub fn quantize(&mut self, x: &[f32]) {
        let n = x.len();
        assert!(n % 32 == 0, "activation length must be a multiple of 32");
        if self.qs.len() != n {
            *self = Q8Act::with_len(n);
        }
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                unsafe { avx2::quantize_q8(x, &mut self.qs, &mut self.d, &mut self.s) };
                return;
            }
        }
        quantize_q8_scalar(x, &mut self.qs, &mut self.d, &mut self.s);
    }
}

/// Portable reference implementation of activation quantisation.
///
/// * **Input** — `x` (len n), output slices `qs` (n), `d` (n/32), `s` (n/32).
///
/// # Example
/// ```
/// let x = [1.0f32; 32];
/// let (mut q, mut d, mut s) = ([0i8; 32], [0f32; 1], [0f32; 1]);
/// aido::tensor::quant::quantize_q8_scalar(&x, &mut q, &mut d, &mut s);
/// assert_eq!(q[0], 127);
/// assert!((s[0] - 32.0).abs() < 1e-3);
/// ```
pub fn quantize_q8_scalar(x: &[f32], qs: &mut [i8], d: &mut [f32], s: &mut [f32]) {
    for b in 0..x.len() / 32 {
        let xb = &x[b * 32..b * 32 + 32];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
        let mut sum = 0i32;
        for i in 0..32 {
            let q = (xb[i] * id).round_ties_even() as i32;
            let q = q.clamp(-127, 127);
            qs[b * 32 + i] = q as i8;
            sum += q;
        }
        d[b] = dd;
        s[b] = dd * sum as f32;
    }
}

/// Returns `true` if AVX2 + FMA + F16C kernels can be used (cached).
///
/// # Example
/// ```
/// let _fast = aido::tensor::quant::has_avx2();
/// ```
#[inline]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(|| {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers / dequantisation
// ---------------------------------------------------------------------------

/// Reads a little-endian f16 at `b[0..2]` as f32.
#[inline]
fn rd_f16(b: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

/// Decodes the 6-bit (scale, min) pair `j` (0..8) of a `Q4_K`/`Q5_K` block.
///
/// * **Input** — `j`: sub-block index; `q`: 12 packed scale bytes.
/// * **Output** — `(scale, min)` in `0..64`.
///
/// # Example
/// ```
/// let q = [1u8, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0];
/// assert_eq!(aido::tensor::quant::scale_min_k4(0, &q), (1, 5));
/// ```
#[inline]
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantises one row of `n` elements of type `ty` into `out`.
///
/// * **Input** — `ty`: weight type; `src`: `ty.row_bytes(n)` bytes;
///   `out`: destination of length `n`.
/// * **Behaviour** — exact inverse of GGML's quantisation layout.
///
/// # Example
/// ```
/// use aido::tensor::quant::{dequantize_row, GgmlType};
/// // A Q8_0 block: scale 1.0 (f16 0x3C00) followed by 32 values.
/// let mut blk = vec![0x00u8, 0x3C];
/// blk.extend((0..32).map(|i| i as u8));
/// let mut out = vec![0f32; 32];
/// dequantize_row(GgmlType::Q8_0, &blk, &mut out);
/// assert_eq!(out[5], 5.0);
/// ```
pub fn dequantize_row(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    let n = out.len();
    match ty {
        GgmlType::F32 => {
            for i in 0..n {
                out[i] = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for i in 0..n {
                out[i] = rd_f16(&src[i * 2..]);
            }
        }
        GgmlType::BF16 => {
            for i in 0..n {
                out[i] = bf16::from_bits(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]])).to_f32();
            }
        }
        GgmlType::Q4_0 => {
            for (b, blk) in src.chunks_exact(18).take(n / 32).enumerate() {
                let d = rd_f16(blk);
                let o = &mut out[b * 32..b * 32 + 32];
                for j in 0..16 {
                    o[j] = ((blk[2 + j] & 0xF) as i32 - 8) as f32 * d;
                    o[j + 16] = ((blk[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_1 => {
            for (b, blk) in src.chunks_exact(20).take(n / 32).enumerate() {
                let d = rd_f16(blk);
                let m = rd_f16(&blk[2..]);
                let o = &mut out[b * 32..b * 32 + 32];
                for j in 0..16 {
                    o[j] = (blk[4 + j] & 0xF) as f32 * d + m;
                    o[j + 16] = (blk[4 + j] >> 4) as f32 * d + m;
                }
            }
        }
        GgmlType::Q5_0 => {
            for (b, blk) in src.chunks_exact(22).take(n / 32).enumerate() {
                let d = rd_f16(blk);
                let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                let o = &mut out[b * 32..b * 32 + 32];
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    o[j] = (((blk[6 + j] & 0xF) | h0) as i32 - 16) as f32 * d;
                    o[j + 16] = (((blk[6 + j] >> 4) | h1) as i32 - 16) as f32 * d;
                }
            }
        }
        GgmlType::Q5_1 => {
            for (b, blk) in src.chunks_exact(24).take(n / 32).enumerate() {
                let d = rd_f16(blk);
                let m = rd_f16(&blk[2..]);
                let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
                let o = &mut out[b * 32..b * 32 + 32];
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    o[j] = ((blk[8 + j] & 0xF) | h0) as f32 * d + m;
                    o[j + 16] = ((blk[8 + j] >> 4) | h1) as f32 * d + m;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (b, blk) in src.chunks_exact(34).take(n / 32).enumerate() {
                let d = rd_f16(blk);
                for j in 0..32 {
                    out[b * 32 + j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_K => {
            for (b, blk) in src.chunks_exact(144).take(n / 256).enumerate() {
                let d = rd_f16(blk);
                let dmin = rd_f16(&blk[2..]);
                let sc = &blk[4..16];
                let qs = &blk[16..144];
                let o = &mut out[b * 256..b * 256 + 256];
                for j in 0..4 {
                    let (s0, m0) = scale_min_k4(2 * j, sc);
                    let (s1, m1) = scale_min_k4(2 * j + 1, sc);
                    let (d1, m1f) = (d * s0 as f32, dmin * m0 as f32);
                    let (d2, m2f) = (d * s1 as f32, dmin * m1 as f32);
                    for l in 0..32 {
                        o[j * 64 + l] = d1 * (qs[j * 32 + l] & 0xF) as f32 - m1f;
                        o[j * 64 + 32 + l] = d2 * (qs[j * 32 + l] >> 4) as f32 - m2f;
                    }
                }
            }
        }
        GgmlType::Q5_K => {
            for (b, blk) in src.chunks_exact(176).take(n / 256).enumerate() {
                let d = rd_f16(blk);
                let dmin = rd_f16(&blk[2..]);
                let sc = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let o = &mut out[b * 256..b * 256 + 256];
                for j in 0..4 {
                    let (s0, m0) = scale_min_k4(2 * j, sc);
                    let (s1, m1) = scale_min_k4(2 * j + 1, sc);
                    let (u1, u2) = (1u8 << (2 * j), 1u8 << (2 * j + 1));
                    for l in 0..32 {
                        let lo = (qs[j * 32 + l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi = (qs[j * 32 + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                        o[j * 64 + l] = d * s0 as f32 * lo as f32 - dmin * m0 as f32;
                        o[j * 64 + 32 + l] = d * s1 as f32 * hi as f32 - dmin * m1 as f32;
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            for (b, blk) in src.chunks_exact(210).take(n / 256).enumerate() {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = rd_f16(&blk[208..]);
                let o = &mut out[b * 256..b * 256 + 256];
                for h in 0..2 {
                    let (ql, qh, sc, o) = (&ql[64 * h..], &qh[32 * h..], &sc[8 * h..], &mut o[128 * h..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dot products
// ---------------------------------------------------------------------------

/// Signature of a row·activation kernel: `(row_bytes, x_f32, x_q8) -> dot`.
pub type DotFn = fn(&[u8], &[f32], &Q8Act) -> f32;

/// Returns `true` if `ty` consumes the int8 activation (otherwise f32).
///
/// # Example
/// ```
/// use aido::tensor::quant::{uses_q8, GgmlType};
/// assert!(uses_q8(GgmlType::Q4_K));
/// assert!(!uses_q8(GgmlType::F16));
/// ```
pub fn uses_q8(ty: GgmlType) -> bool {
    !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// Selects the fastest available dot-product kernel for `ty`.
///
/// * **Input** — weight type.
/// * **Output** — a function computing `row · x` where the row has
///   `x.len()` elements. Integer formats read `xq`, float formats read `x`.
///
/// # Example
/// ```
/// use aido::tensor::quant::{dot_fn, GgmlType, Q8Act};
/// let row: Vec<u8> = (0..4).flat_map(|v| (v as f32).to_le_bytes()).collect();
/// let x = [1.0f32, 1.0, 1.0, 1.0];
/// let f = dot_fn(GgmlType::F32);
/// assert_eq!(f(&row, &x, &Q8Act::default()), 6.0);
/// ```
pub fn dot_fn(ty: GgmlType) -> DotFn {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return match ty {
                GgmlType::F32 => |r, x, _| unsafe { avx2::dot_f32(r, x) },
                GgmlType::F16 => |r, x, _| unsafe { avx2::dot_f16(r, x) },
                GgmlType::BF16 => dot_scalar_bf16,
                GgmlType::Q4_0 => |r, _, q| unsafe { avx2::dot_q4_0(r, q) },
                GgmlType::Q4_1 => |r, _, q| unsafe { avx2::dot_q4_1(r, q) },
                GgmlType::Q5_0 => |r, _, q| unsafe { avx2::dot_q5_0(r, q) },
                GgmlType::Q5_1 => |r, _, q| unsafe { avx2::dot_q5_1(r, q) },
                GgmlType::Q8_0 => |r, _, q| unsafe { avx2::dot_q8_0(r, q) },
                GgmlType::Q4_K => |r, _, q| unsafe { avx2::dot_q4_k(r, q) },
                GgmlType::Q5_K => |r, _, q| unsafe { avx2::dot_q5_k(r, q) },
                GgmlType::Q6_K => |r, _, q| unsafe { avx2::dot_q6_k(r, q) },
            };
        }
    }
    scalar_dot_fn(ty)
}

/// Kernel computing one weight row against **four** quantised activations
/// at once: the weight block is loaded and unpacked once and reused for all
/// four tokens (register blocking). Used for prompt processing and
/// speculative verification.
pub type Dot4Fn = fn(&[u8], &[Q8Act]) -> [f32; 4];

/// Returns a 4-token kernel for `ty` if one exists for this CPU.
///
/// * **Input** — weight type. **Output** — `Some(kernel)` for the hot
///   formats (`Q8_0`, `Q4_0`, `Q4_K`, `Q6_K`) on AVX2 machines, else `None`
///   (callers then fall back to [`dot_fn`] per token).
///
/// # Example
/// ```
/// use aido::tensor::quant::{dot4_fn, GgmlType};
/// let _maybe = dot4_fn(GgmlType::Q4_K);
/// assert!(dot4_fn(GgmlType::F32).is_none());
/// ```
pub fn dot4_fn(ty: GgmlType) -> Option<Dot4Fn> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return match ty {
                GgmlType::Q8_0 => Some(|r, q| unsafe { avx2::dot4_q8_0(r, q) }),
                GgmlType::Q4_0 => Some(|r, q| unsafe { avx2::dot4_q4_0(r, q) }),
                GgmlType::Q4_K => Some(|r, q| unsafe { avx2::dot4_q4_k(r, q) }),
                GgmlType::Q6_K => Some(|r, q| unsafe { avx2::dot4_q6_k(r, q) }),
                _ => None,
            };
        }
    }
    let _ = ty;
    None
}

/// Returns the portable scalar kernel for `ty` (reference implementation).
///
/// # Example
/// ```
/// use aido::tensor::quant::{scalar_dot_fn, GgmlType};
/// let _f = scalar_dot_fn(GgmlType::Q6_K);
/// ```
pub fn scalar_dot_fn(ty: GgmlType) -> DotFn {
    match ty {
        GgmlType::F32 | GgmlType::F16 => dot_scalar_float,
        GgmlType::BF16 => dot_scalar_bf16,
        _ => match ty {
            GgmlType::Q4_0 => |r, _, q| dot_scalar_generic(GgmlType::Q4_0, r, q),
            GgmlType::Q4_1 => |r, _, q| dot_scalar_generic(GgmlType::Q4_1, r, q),
            GgmlType::Q5_0 => |r, _, q| dot_scalar_generic(GgmlType::Q5_0, r, q),
            GgmlType::Q5_1 => |r, _, q| dot_scalar_generic(GgmlType::Q5_1, r, q),
            GgmlType::Q8_0 => |r, _, q| dot_scalar_generic(GgmlType::Q8_0, r, q),
            GgmlType::Q4_K => |r, _, q| dot_scalar_generic(GgmlType::Q4_K, r, q),
            GgmlType::Q5_K => |r, _, q| dot_scalar_generic(GgmlType::Q5_K, r, q),
            _ => |r, _, q| dot_scalar_generic(GgmlType::Q6_K, r, q),
        },
    }
}

/// Scalar f32/f16 row dot (type inferred from byte length).
fn dot_scalar_float(r: &[u8], x: &[f32], _q: &Q8Act) -> f32 {
    let n = x.len();
    if r.len() >= n * 4 {
        (0..n).map(|i| f32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap()) * x[i]).sum()
    } else {
        (0..n).map(|i| rd_f16(&r[i * 2..]) * x[i]).sum()
    }
}

/// Scalar bf16 row dot.
fn dot_scalar_bf16(r: &[u8], x: &[f32], _q: &Q8Act) -> f32 {
    (0..x.len()).map(|i| bf16::from_bits(u16::from_le_bytes([r[2 * i], r[2 * i + 1]])).to_f32() * x[i]).sum()
}

/// Scalar reference: dequantise blockwise and dot with the *dequantised*
/// int8 activation (bit-identical maths to the SIMD kernels up to float
/// summation order).
fn dot_scalar_generic(ty: GgmlType, r: &[u8], q: &Q8Act) -> f32 {
    let n = q.qs.len();
    let be = ty.block_elems();
    let bb = ty.block_bytes();
    let mut tmp = [0f32; 256];
    let mut acc = 0f32;
    for b in 0..n / be {
        dequantize_row(ty, &r[b * bb..(b + 1) * bb], &mut tmp[..be]);
        for i in 0..be {
            let k = b * be + i;
            acc += tmp[i] * q.qs[k] as f32 * q.d[k / 32];
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! AVX2 + FMA + F16C kernels. All functions are `unsafe` because they
    //! require the CPU features checked by [`super::has_avx2`].
    use super::Q8Act;
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 floats.
    #[inline(always)]
    pub unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    /// f16 -> f32 for a scalar.
    #[inline(always)]
    unsafe fn h2f(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((p as *const u16).read_unaligned() as i32)))
    }

    /// Signed int8 × signed int8 dot of 32 lanes → 8 × f32 partial sums.
    #[inline(always)]
    unsafe fn mul_i8(x: __m256i, y: __m256i) -> __m256 {
        let ax = _mm256_sign_epi8(x, x);
        let sy = _mm256_sign_epi8(y, x);
        let dot = _mm256_maddubs_epi16(ax, sy);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
    }

    /// Unsigned int8 × signed int8 dot → 8 × f32.
    #[inline(always)]
    unsafe fn mul_u8(x: __m256i, y: __m256i) -> __m256 {
        let dot = _mm256_maddubs_epi16(x, y);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
    }

    /// Expands 32 bits into 32 bytes of 0xFF / 0x00.
    #[inline(always)]
    unsafe fn bytes_from_bits(p: *const u8) -> __m256i {
        let x = (p as *const u32).read_unaligned();
        let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
        let b = _mm256_shuffle_epi8(_mm256_set1_epi32(x as i32), shuf);
        let b = _mm256_or_si256(b, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
        _mm256_cmpeq_epi8(b, _mm256_set1_epi64x(-1))
    }

    /// Unpacks 16 bytes of nibbles into 32 bytes (low nibbles first).
    #[inline(always)]
    unsafe fn nibbles(p: *const u8) -> __m256i {
        let t = _mm_loadu_si128(p as *const __m128i);
        let hi = _mm_srli_epi16(t, 4);
        let m = _mm256_set1_epi8(0xF);
        _mm256_and_si256(_mm256_set_m128i(hi, t), m)
    }

    /// Quantises activations (see [`super::Q8Act::quantize`]).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn quantize_q8(x: &[f32], qs: &mut [i8], d: &mut [f32], s: &mut [f32]) {
        let sign = _mm256_set1_ps(-0.0);
        for b in 0..x.len() / 32 {
            let p = x.as_ptr().add(b * 32);
            let v0 = _mm256_loadu_ps(p);
            let v1 = _mm256_loadu_ps(p.add(8));
            let v2 = _mm256_loadu_ps(p.add(16));
            let v3 = _mm256_loadu_ps(p.add(24));
            let m = _mm256_max_ps(
                _mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)),
            );
            let mut m4 = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            let amax = _mm_cvtss_f32(m4);
            let dd = amax / 127.0;
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            let mul = _mm256_set1_ps(id);
            let r = |v| _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            let (i0, i1, i2, i3) = (r(v0), r(v1), r(v2), r(v3));
            let sum = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
            let s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
            let s4 = _mm_hadd_epi32(s4, s4);
            let s4 = _mm_hadd_epi32(s4, s4);
            let isum = _mm_cvtsi128_si32(s4);
            let p01 = _mm256_packs_epi32(i0, i1);
            let p23 = _mm256_packs_epi32(i2, i3);
            let p = _mm256_packs_epi16(p01, p23);
            let p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(qs.as_mut_ptr().add(b * 32) as *mut __m256i, p);
            d[b] = dd;
            s[b] = dd * isum as f32;
        }
    }

    /// f32 row · f32 vector.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f32(r: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let w = r.as_ptr() as *const f32;
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(i)), _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(i + 8)), _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            s += w.add(i).read_unaligned() * x[i];
            i += 1;
        }
        s
    }

    /// f16 row · f32 vector.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(r: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let w = r.as_ptr() as *const u16;
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let h0 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(i) as *const __m128i));
            let h1 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(i + 8) as *const __m128i));
            a0 = _mm256_fmadd_ps(h0, _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(h1, _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            s += half::f16::from_bits(w.add(i).read_unaligned()).to_f32() * x[i];
            i += 1;
        }
        s
    }

    /// Q8_0 row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q8_0(r: &[u8], q: &Q8Act) -> f32 {
        let nb = q.d.len();
        let mut acc = _mm256_setzero_ps();
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        for b in 0..nb {
            let blk = rp.add(b * 34);
            let d = _mm256_set1_ps(h2f(blk) * *q.d.get_unchecked(b));
            let bx = _mm256_loadu_si256(blk.add(2) as *const __m256i);
            let by = _mm256_loadu_si256(qp.add(b * 32) as *const __m256i);
            acc = _mm256_fmadd_ps(d, mul_i8(bx, by), acc);
        }
        hsum(acc)
    }

    /// Q4_0 row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4_0(r: &[u8], q: &Q8Act) -> f32 {
        let nb = q.d.len();
        let mut acc = _mm256_setzero_ps();
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        let eight = _mm256_set1_epi8(8);
        for b in 0..nb {
            let blk = rp.add(b * 18);
            let d = _mm256_set1_ps(h2f(blk) * *q.d.get_unchecked(b));
            let bx = _mm256_sub_epi8(nibbles(blk.add(2)), eight);
            let by = _mm256_loadu_si256(qp.add(b * 32) as *const __m256i);
            acc = _mm256_fmadd_ps(d, mul_i8(bx, by), acc);
        }
        hsum(acc)
    }

    /// Q4_1 row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4_1(r: &[u8], q: &Q8Act) -> f32 {
        let nb = q.d.len();
        let mut acc = _mm256_setzero_ps();
        let mut summ = 0f32;
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        for b in 0..nb {
            let blk = rp.add(b * 20);
            let d = _mm256_set1_ps(h2f(blk) * *q.d.get_unchecked(b));
            summ += h2f(blk.add(2)) * *q.s.get_unchecked(b);
            let bx = nibbles(blk.add(4));
            let by = _mm256_loadu_si256(qp.add(b * 32) as *const __m256i);
            acc = _mm256_fmadd_ps(d, mul_u8(bx, by), acc);
        }
        hsum(acc) + summ
    }

    /// Q5_0 row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5_0(r: &[u8], q: &Q8Act) -> f32 {
        let nb = q.d.len();
        let mut acc = _mm256_setzero_ps();
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        for b in 0..nb {
            let blk = rp.add(b * 22);
            let d = _mm256_set1_ps(h2f(blk) * *q.d.get_unchecked(b));
            let mut bx = nibbles(blk.add(6));
            let bxhi = _mm256_andnot_si256(bytes_from_bits(blk.add(2)), _mm256_set1_epi8(0xF0u8 as i8));
            bx = _mm256_or_si256(bx, bxhi);
            let by = _mm256_loadu_si256(qp.add(b * 32) as *const __m256i);
            acc = _mm256_fmadd_ps(d, mul_i8(bx, by), acc);
        }
        hsum(acc)
    }

    /// Q5_1 row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5_1(r: &[u8], q: &Q8Act) -> f32 {
        let nb = q.d.len();
        let mut acc = _mm256_setzero_ps();
        let mut summ = 0f32;
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        for b in 0..nb {
            let blk = rp.add(b * 24);
            let d = _mm256_set1_ps(h2f(blk) * *q.d.get_unchecked(b));
            summ += h2f(blk.add(2)) * *q.s.get_unchecked(b);
            let mut bx = nibbles(blk.add(8));
            let bxhi = _mm256_and_si256(bytes_from_bits(blk.add(4)), _mm256_set1_epi8(0x10));
            bx = _mm256_or_si256(bx, bxhi);
            let by = _mm256_loadu_si256(qp.add(b * 32) as *const __m256i);
            acc = _mm256_fmadd_ps(d, mul_u8(bx, by), acc);
        }
        hsum(acc) + summ
    }

    /// Decodes the 8 (scale, min) pairs of a Q4_K/Q5_K block.
    #[inline(always)]
    unsafe fn kscales(sc: *const u8) -> ([u8; 8], [u8; 8]) {
        let q = std::slice::from_raw_parts(sc, 12);
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            let (a, b) = super::scale_min_k4(j, q);
            s[j] = a;
            m[j] = b;
        }
        (s, m)
    }

    /// Q4_K row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4_k(r: &[u8], q: &Q8Act) -> f32 {
        let nsb = q.qs.len() / 256;
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut summ = 0f32;
        for sb in 0..nsb {
            let blk = rp.add(sb * 144);
            let d = h2f(blk);
            let dmin = h2f(blk.add(2));
            let (sc, mn) = kscales(blk.add(4));
            let qs = blk.add(16);
            let qd = q.d.as_ptr().add(sb * 8);
            let qsum = q.s.as_ptr().add(sb * 8);
            for j in 0..4 {
                let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
                let lo = _mm256_and_si256(raw, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4);
                let y0 = _mm256_loadu_si256(qp.add(sb * 256 + j * 64) as *const __m256i);
                let y1 = _mm256_loadu_si256(qp.add(sb * 256 + j * 64 + 32) as *const __m256i);
                // Integer scale inside the madd: fewer float ops.
                let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), _mm256_set1_epi16(sc[2 * j] as i16));
                let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), _mm256_set1_epi16(sc[2 * j + 1] as i16));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * *qd.add(2 * j)), _mm256_cvtepi32_ps(p0), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * *qd.add(2 * j + 1)), _mm256_cvtepi32_ps(p1), acc);
                summ += dmin * (mn[2 * j] as f32 * *qsum.add(2 * j) + mn[2 * j + 1] as f32 * *qsum.add(2 * j + 1));
            }
        }
        hsum(acc) - summ
    }

    /// Q5_K row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5_k(r: &[u8], q: &Q8Act) -> f32 {
        let nsb = q.qs.len() / 256;
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        let m4 = _mm256_set1_epi8(0xF);
        let b16 = _mm256_set1_epi8(16);
        let mut acc = _mm256_setzero_ps();
        let mut summ = 0f32;
        for sb in 0..nsb {
            let blk = rp.add(sb * 176);
            let d = h2f(blk);
            let dmin = h2f(blk.add(2));
            let (sc, mn) = kscales(blk.add(4));
            let qh = _mm256_loadu_si256(blk.add(16) as *const __m256i);
            let qs = blk.add(48);
            let qd = q.d.as_ptr().add(sb * 8);
            let qsum = q.s.as_ptr().add(sb * 8);
            for j in 0..4 {
                let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
                let u1 = _mm256_set1_epi8((1u8 << (2 * j)) as i8);
                let u2 = _mm256_set1_epi8((1u8 << (2 * j + 1)) as i8);
                let h1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u1), u1), b16);
                let h2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u2), u2), b16);
                let lo = _mm256_or_si256(_mm256_and_si256(raw, m4), h1);
                let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), m4), h2);
                let y0 = _mm256_loadu_si256(qp.add(sb * 256 + j * 64) as *const __m256i);
                let y1 = _mm256_loadu_si256(qp.add(sb * 256 + j * 64 + 32) as *const __m256i);
                let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), _mm256_set1_epi16(sc[2 * j] as i16));
                let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), _mm256_set1_epi16(sc[2 * j + 1] as i16));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * *qd.add(2 * j)), _mm256_cvtepi32_ps(p0), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * *qd.add(2 * j + 1)), _mm256_cvtepi32_ps(p1), acc);
                summ += dmin * (mn[2 * j] as f32 * *qsum.add(2 * j) + mn[2 * j + 1] as f32 * *qsum.add(2 * j + 1));
            }
        }
        hsum(acc) - summ
    }

    /// Sums 4 accumulators horizontally.
    #[inline(always)]
    unsafe fn hsum4(a: &[__m256; 4]) -> [f32; 4] {
        [hsum(a[0]), hsum(a[1]), hsum(a[2]), hsum(a[3])]
    }

    /// Q8_0 row · 4 activations.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot4_q8_0(r: &[u8], q: &[Q8Act]) -> [f32; 4] {
        let nb = q[0].d.len();
        let mut acc = [_mm256_setzero_ps(); 4];
        let rp = r.as_ptr();
        for b in 0..nb {
            let blk = rp.add(b * 34);
            let dw = h2f(blk);
            let bx = _mm256_loadu_si256(blk.add(2) as *const __m256i);
            let ax = _mm256_sign_epi8(bx, bx);
            for t in 0..4 {
                let by = _mm256_loadu_si256(q[t].qs.as_ptr().add(b * 32) as *const __m256i);
                let dot = _mm256_maddubs_epi16(ax, _mm256_sign_epi8(by, bx));
                let f = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * *q[t].d.get_unchecked(b)), f, acc[t]);
            }
        }
        hsum4(&acc)
    }

    /// Q4_0 row · 4 activations.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot4_q4_0(r: &[u8], q: &[Q8Act]) -> [f32; 4] {
        let nb = q[0].d.len();
        let mut acc = [_mm256_setzero_ps(); 4];
        let rp = r.as_ptr();
        let eight = _mm256_set1_epi8(8);
        for b in 0..nb {
            let blk = rp.add(b * 18);
            let dw = h2f(blk);
            let bx = _mm256_sub_epi8(nibbles(blk.add(2)), eight);
            let ax = _mm256_sign_epi8(bx, bx);
            for t in 0..4 {
                let by = _mm256_loadu_si256(q[t].qs.as_ptr().add(b * 32) as *const __m256i);
                let dot = _mm256_maddubs_epi16(ax, _mm256_sign_epi8(by, bx));
                let f = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * *q[t].d.get_unchecked(b)), f, acc[t]);
            }
        }
        hsum4(&acc)
    }

    /// Q4_K row · 4 activations.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot4_q4_k(r: &[u8], q: &[Q8Act]) -> [f32; 4] {
        let nsb = q[0].qs.len() / 256;
        let rp = r.as_ptr();
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut summ = [0f32; 4];
        for sb in 0..nsb {
            let blk = rp.add(sb * 144);
            let d = h2f(blk);
            let dmin = h2f(blk.add(2));
            let (sc, mn) = kscales(blk.add(4));
            let qs = blk.add(16);
            for j in 0..4 {
                let raw = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
                let lo = _mm256_and_si256(raw, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4);
                let s0 = _mm256_set1_epi16(sc[2 * j] as i16);
                let s1 = _mm256_set1_epi16(sc[2 * j + 1] as i16);
                let (b0, b1) = (sb * 8 + 2 * j, sb * 8 + 2 * j + 1);
                for t in 0..4 {
                    let qp = q[t].qs.as_ptr();
                    let y0 = _mm256_loadu_si256(qp.add(b0 * 32) as *const __m256i);
                    let y1 = _mm256_loadu_si256(qp.add(b1 * 32) as *const __m256i);
                    let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), s0);
                    let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), s1);
                    let qd = q[t].d.as_ptr();
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * *qd.add(b0)), _mm256_cvtepi32_ps(p0), acc[t]);
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * *qd.add(b1)), _mm256_cvtepi32_ps(p1), acc[t]);
                    let qs_ = q[t].s.as_ptr();
                    summ[t] += dmin * (mn[2 * j] as f32 * *qs_.add(b0) + mn[2 * j + 1] as f32 * *qs_.add(b1));
                }
            }
        }
        let h = hsum4(&acc);
        [h[0] - summ[0], h[1] - summ[1], h[2] - summ[2], h[3] - summ[3]]
    }

    /// Q6_K row · 4 activations.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot4_q6_k(r: &[u8], q: &[Q8Act]) -> [f32; 4] {
        let nsb = q[0].qs.len() / 256;
        let rp = r.as_ptr();
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let c32 = _mm256_set1_epi8(32);
        let mut acc = [_mm256_setzero_ps(); 4];
        for sb in 0..nsb {
            let blk = rp.add(sb * 210);
            let d = h2f(blk.add(208));
            let scp = blk.add(192) as *const i8;
            for h in 0..2 {
                let ql = blk.add(64 * h);
                let qhv = _mm256_loadu_si256(blk.add(128 + 32 * h) as *const __m256i);
                let ql0 = _mm256_loadu_si256(ql as *const __m256i);
                let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                let qv = [
                    _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4)),
                    _mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4)),
                ];
                for k in 0..4 {
                    let blk_idx = sb * 8 + h * 4 + k;
                    let s0 = *scp.add(h * 8 + 2 * k) as i16;
                    let s1 = *scp.add(h * 8 + 2 * k + 1) as i16;
                    let scv = _mm256_setr_epi16(s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1, s1, s1, s1, s1);
                    for t in 0..4 {
                        let y = _mm256_loadu_si256(q[t].qs.as_ptr().add(blk_idx * 32) as *const __m256i);
                        let diff = _mm256_sub_epi16(_mm256_maddubs_epi16(qv[k], y), _mm256_maddubs_epi16(c32, y));
                        let p = _mm256_madd_epi16(diff, scv);
                        acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * *q[t].d.get_unchecked(blk_idx)), _mm256_cvtepi32_ps(p), acc[t]);
                    }
                }
            }
        }
        hsum4(&acc)
    }

    /// Q6_K row · Q8 activation.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q6_k(r: &[u8], q: &Q8Act) -> f32 {
        let nsb = q.qs.len() / 256;
        let rp = r.as_ptr();
        let qp = q.qs.as_ptr();
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let c32 = _mm256_set1_epi8(32);
        let mut acc = _mm256_setzero_ps();
        for sb in 0..nsb {
            let blk = rp.add(sb * 210);
            let d = h2f(blk.add(208));
            let scp = blk.add(192) as *const i8;
            for h in 0..2 {
                let ql = blk.add(64 * h);
                let qhv = _mm256_loadu_si256(blk.add(128 + 32 * h) as *const __m256i);
                let ql0 = _mm256_loadu_si256(ql as *const __m256i);
                let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                let qv = [
                    _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4)),
                    _mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4)),
                ];
                for k in 0..4 {
                    let blk_idx = sb * 8 + h * 4 + k;
                    let y = _mm256_loadu_si256(qp.add(blk_idx * 32) as *const __m256i);
                    let dot = _mm256_maddubs_epi16(qv[k], y);
                    let off = _mm256_maddubs_epi16(c32, y);
                    let diff = _mm256_sub_epi16(dot, off);
                    let s0 = *scp.add(h * 8 + 2 * k) as i16;
                    let s1 = *scp.add(h * 8 + 2 * k + 1) as i16;
                    let scv = _mm256_setr_epi16(s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1, s1, s1, s1, s1);
                    let p = _mm256_madd_epi16(diff, scv);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d * *q.d.get_unchecked(blk_idx)), _mm256_cvtepi32_ps(p), acc);
                }
            }
        }
        hsum(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes (xorshift).
    fn rnd(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    /// Builds a random but well-formed row (valid f16 scales).
    fn random_row(ty: GgmlType, n: usize, seed: u64) -> Vec<u8> {
        let mut r = rnd(ty.row_bytes(n), seed);
        let bb = ty.block_bytes();
        let small = |r: &mut [u8], at: usize, v: f32| {
            let b = f16::from_f32(v).to_bits().to_le_bytes();
            r[at] = b[0];
            r[at + 1] = b[1];
        };
        for (i, blk) in r.chunks_exact_mut(bb).enumerate() {
            let v = 0.01 + (i % 7) as f32 * 0.003;
            match ty {
                GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => small(blk, 0, v),
                GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => {
                    small(blk, 0, v);
                    small(blk, 2, v * 0.5)
                }
                GgmlType::Q6_K => small(blk, 208, v),
                GgmlType::F16 => small(blk, 0, (i % 13) as f32 * 0.1 - 0.6),
                _ => {}
            }
        }
        if ty == GgmlType::F32 {
            r = (0..n).flat_map(|i| ((i % 11) as f32 * 0.1 - 0.5).to_le_bytes()).collect();
        }
        r
    }

    #[test]
    fn dot4_matches_dot() {
        let n = 512;
        let acts: Vec<Q8Act> = (0..4)
            .map(|t| {
                let x: Vec<f32> = (0..n).map(|i| ((i * 13 + t * 7) % 29) as f32 / 7.0 - 2.0).collect();
                let mut a = Q8Act::default();
                a.quantize(&x);
                a
            })
            .collect();
        for ty in [GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_K, GgmlType::Q6_K] {
            if let Some(f4) = dot4_fn(ty) {
                let row = random_row(ty, n, 77);
                let r4 = f4(&row, &acts);
                for t in 0..4 {
                    let r1 = dot_fn(ty)(&row, &[], &acts[t]);
                    assert!((r4[t] - r1).abs() < 1e-3 * (1.0 + r1.abs()), "{ty:?} t{t}: {} vs {}", r4[t], r1);
                }
            }
        }
    }

    #[test]
    fn simd_matches_scalar_all_types() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let mut xq = Q8Act::default();
        xq.quantize(&x);
        // Quantiser itself vs scalar.
        let (mut q2, mut d2, mut s2) = (vec![0i8; n], vec![0f32; n / 32], vec![0f32; n / 32]);
        quantize_q8_scalar(&x, &mut q2, &mut d2, &mut s2);
        assert_eq!(xq.qs, q2);
        for ty in [
            GgmlType::F32, GgmlType::F16, GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1,
            GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K,
        ] {
            let row = random_row(ty, n, 1234 + ty as u64);
            let fast = dot_fn(ty)(&row, &x, &xq);
            let slow = scalar_dot_fn(ty)(&row, &x, &xq);
            // Reference against fully dequantised f32 maths as well.
            let mut deq = vec![0f32; n];
            dequantize_row(ty, &row, &mut deq);
            let exact: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            let tol = 1e-3 * (1.0 + slow.abs());
            assert!((fast - slow).abs() < tol, "{ty:?}: simd {fast} vs scalar {slow}");
            // int8 activation rounding error scales with Σ|w·x| (not |Σ w·x|).
            let l1: f32 = deq.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let tol2 = 0.01 * l1 + 0.05;
            assert!((slow - exact).abs() < tol2, "{ty:?}: q8 {slow} vs f32 {exact}");
        }
    }
}

// ---------------------------------------------------------------------------
// colibri int4 "gs64" expert rows
// ---------------------------------------------------------------------------

/// Reorders `x` so that every group of 64 becomes `[even elements | odd
/// elements]`. Quantising the reordered vector gives the activation layout
/// expected by [`dot_i4g64`]: the packed int4 weights store element `2k` in
/// the low and `2k+1` in the high nibble, so after this permutation both
/// nibble planes line up with contiguous 32-element activation blocks and
/// the integer `maddubs` kernel runs **directly on the original file bytes**
/// (no repacking, no extra memory).
///
/// * **Input** — `x` (length multiple of 64); `out` same length.
///
/// # Example
/// ```
/// let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
/// let mut o = vec![0f32; 64];
/// aido::tensor::quant::deinterleave64(&x, &mut o);
/// assert_eq!((o[0], o[1], o[32], o[33]), (0.0, 2.0, 1.0, 3.0));
/// ```
pub fn deinterleave64(x: &[f32], out: &mut [f32]) {
    for (g, (src, dst)) in x.chunks_exact(64).zip(out.chunks_exact_mut(64)).enumerate() {
        let _ = g;
        for k in 0..32 {
            dst[k] = src[2 * k];
            dst[32 + k] = src[2 * k + 1];
        }
    }
}

/// Dot product of one colibri int4-gs64 row with a de-interleaved Q8
/// activation (see [`deinterleave64`]).
///
/// * **Input** — `row`: `cols/2` packed bytes (signed nibbles, low = even
///   element); `scales`: `cols/64` little-endian f32 group scales (raw bytes,
///   any alignment); `xq`: quantised de-interleaved activation.
/// * **Output** — `Σ_g scale_g Σ_k w_k x_k`.
///
/// # Example
/// ```
/// use aido::tensor::quant::{dot_i4g64, deinterleave64, Q8Act};
/// // all weights = +1 (nibble 1), scale 0.5
/// let row = vec![0x11u8; 32];
/// let scales = 0.5f32.to_le_bytes();
/// let x = vec![1.0f32; 64];
/// let mut xd = vec![0f32; 64];
/// deinterleave64(&x, &mut xd);
/// let mut q = Q8Act::default();
/// q.quantize(&xd);
/// assert!((dot_i4g64(&row, &scales, &q) - 32.0).abs() < 0.5);
/// ```
pub fn dot_i4g64(row: &[u8], scales: &[u8], xq: &Q8Act) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx2_i4::dot(row, scales, xq) };
    }
    let mut acc = 0f32;
    for g in 0..row.len() / 32 {
        let sc = f32::from_le_bytes(scales[4 * g..4 * g + 4].try_into().unwrap());
        let (mut se, mut so) = (0i32, 0i32);
        for k in 0..32 {
            let b = row[g * 32 + k];
            let lo = ((b & 15) ^ 8) as i32 - 8;
            let hi = ((b >> 4) ^ 8) as i32 - 8;
            se += lo * xq.qs[g * 64 + k] as i32;
            so += hi * xq.qs[g * 64 + 32 + k] as i32;
        }
        acc += sc * (xq.d[2 * g] * se as f32 + xq.d[2 * g + 1] * so as f32);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod avx2_i4 {
    use super::Q8Act;
    use std::arch::x86_64::*;

    /// AVX2 kernel for [`super::dot_i4g64`].
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot(row: &[u8], scales: &[u8], xq: &Q8Act) -> f32 {
        let ng = row.len() / 32;
        let m4 = _mm256_set1_epi8(0xF);
        let x8 = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        let mut corr = 0f32;
        let rp = row.as_ptr();
        let sp = scales.as_ptr() as *const f32;
        let qp = xq.qs.as_ptr();
        for g in 0..ng {
            let sc = sp.add(g).read_unaligned();
            let raw = _mm256_loadu_si256(rp.add(g * 32) as *const __m256i);
            let lo = _mm256_xor_si256(_mm256_and_si256(raw, m4), x8);
            let hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), m4), x8);
            let ye = _mm256_loadu_si256(qp.add(g * 64) as *const __m256i);
            let yo = _mm256_loadu_si256(qp.add(g * 64 + 32) as *const __m256i);
            let pe = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, ye), ones));
            let po = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(hi, yo), ones));
            let de = *xq.d.get_unchecked(2 * g);
            let dd = *xq.d.get_unchecked(2 * g + 1);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc * de), pe, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc * dd), po, acc);
            corr += sc * (*xq.s.get_unchecked(2 * g) + *xq.s.get_unchecked(2 * g + 1));
        }
        super::avx2_hsum(acc) - 8.0 * corr
    }
}

/// Horizontal sum exported for sibling SIMD modules.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_hsum(v: std::arch::x86_64::__m256) -> f32 {
    crate::tensor::ops::simd_hsum(v)
}

#[cfg(test)]
mod i4_tests {
    use super::*;

    #[test]
    fn i4g64_simd_matches_reference() {
        let cols = 512;
        let row: Vec<u8> = (0..cols / 2).map(|i| (i * 37 % 256) as u8).collect();
        let scales: Vec<u8> = (0..cols / 64).flat_map(|g| (0.01f32 + g as f32 * 0.003).to_le_bytes()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i * 13 % 17) as f32 - 8.0) / 3.0).collect();
        // reference in f32 with natural order
        let mut want = 0f32;
        for i in 0..cols {
            let b = row[i / 2];
            let nib = if i % 2 == 0 { b & 15 } else { b >> 4 };
            let v = ((nib ^ 8) as i32 - 8) as f32;
            let sc = f32::from_le_bytes(scales[4 * (i / 64)..4 * (i / 64) + 4].try_into().unwrap());
            want += v * sc * x[i];
        }
        let mut xd = vec![0f32; cols];
        deinterleave64(&x, &mut xd);
        let mut q = Q8Act::default();
        q.quantize(&xd);
        let got = dot_i4g64(&row, &scales, &q);
        assert!((got - want).abs() < 0.02 * (1.0 + want.abs()), "{got} vs {want}");
    }
}
