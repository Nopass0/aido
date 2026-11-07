//! Cache-blocked single-precision GEMM (BLIS-style).
//!
//! Diffusion models are *compute*-bound (convolutions over 64×64 latents,
//! attention over 4096 tokens), the opposite of LLM decoding. For them the
//! engine uses a classic three-level blocking scheme:
//!
//! * the weight matrix `B` is **packed once at load time** into 16-column
//!   panels (`[panel][k][16]`) so the micro-kernel streams it linearly;
//! * a block of `A` (`MC × KC`, ≈ L2-sized) is packed per task into 6-row
//!   panels;
//! * a 6×16 AVX2/FMA micro-kernel keeps 12 accumulators in registers and
//!   performs 12 FMAs per 2 vector loads + 6 broadcasts, ~85 % of the core's
//!   theoretical FMA throughput.
//!
//! Work is split over `(M-block, N-block)` tasks on the [`ThreadPool`].

use crate::runtime::pool::ThreadPool;
use crate::tensor::SyncPtr;

const MR: usize = 6;
const NR: usize = 16;
const KC: usize = 256;
const MC: usize = 96;
const NC: usize = 256;

/// The right-hand matrix of a GEMM, pre-packed into 16-wide column panels.
#[derive(Clone)]
pub struct PackedB {
    /// Inner dimension.
    pub k: usize,
    /// Number of columns (output features).
    pub n: usize,
    data: Vec<f32>,
}

impl PackedB {
    /// Packs a weight given as `W[n][k]` (PyTorch `Linear`/`Conv` layout,
    /// i.e. `B = Wᵀ`).
    ///
    /// * **Input** — `w`: `n × k` row-major; `n`, `k`.
    /// * **Output** — packed matrix; columns are zero-padded to a multiple of 16.
    ///
    /// # Example
    /// ```
    /// let p = aido::tensor::gemm::PackedB::from_nk(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    /// assert_eq!((p.n, p.k), (2, 2));
    /// ```
    pub fn from_nk(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        let panels = n.div_ceil(NR);
        let mut data = vec![0f32; panels * k * NR];
        for p in 0..panels {
            for kk in 0..k {
                for j in 0..NR {
                    let col = p * NR + j;
                    if col < n {
                        data[(p * k + kk) * NR + j] = w[col * k + kk];
                    }
                }
            }
        }
        PackedB { k, n, data }
    }

    /// Packs a matrix given as `B[k][n]` (row-major, no transpose).
    ///
    /// # Example
    /// ```
    /// let p = aido::tensor::gemm::PackedB::from_kn(&[1.0, 2.0], 1, 2);
    /// assert_eq!(p.n, 2);
    /// ```
    pub fn from_kn(b: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(b.len(), n * k);
        let panels = n.div_ceil(NR);
        let mut data = vec![0f32; panels * k * NR];
        for kk in 0..k {
            for col in 0..n {
                let (p, j) = (col / NR, col % NR);
                data[(p * k + kk) * NR + j] = b[kk * n + col];
            }
        }
        PackedB { k, n, data }
    }

    /// Memory used by the packed data in bytes.
    ///
    /// # Example
    /// ```
    /// let p = aido::tensor::gemm::PackedB::from_kn(&[0.0; 16], 1, 16);
    /// assert_eq!(p.nbytes(), 64);
    /// ```
    pub fn nbytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// `C[m×n] = A[m×k] · B (+ bias) (+ C if accumulate)`.
///
/// * **Input** — `a`: row-major with leading dimension `lda`; `b`: packed;
///   `bias`: optional per-column bias (length `n`); `c`: output with
///   leading dimension `ldc`; `accumulate`: add into existing `C`.
/// * **Behaviour** — parallel over `(MC, NC)` blocks; each task packs its own
///   `A` slice (thread-local, no sharing → no false sharing).
///
/// # Example
/// ```
/// use aido::{runtime::pool::ThreadPool, tensor::gemm::{sgemm, PackedB}};
/// let pool = ThreadPool::new(2, None);
/// // A = [[1,2],[3,4]], B = I
/// let b = PackedB::from_kn(&[1.0, 0.0, 0.0, 1.0], 2, 2);
/// let mut c = [0f32; 4];
/// sgemm(&pool, 2, &[1.0, 2.0, 3.0, 4.0], 2, &b, None, &mut c, 2, false);
/// assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn sgemm(pool: &ThreadPool, m: usize, a: &[f32], lda: usize, b: &PackedB, bias: Option<&[f32]>, c: &mut [f32], ldc: usize, accumulate: bool) {
    let (n, k) = (b.n, b.k);
    if m == 0 || n == 0 {
        return;
    }
    let mb = m.div_ceil(MC);
    let nb = n.div_ceil(NC);
    let cp = SyncPtr::new(c);
    pool.parallel_for(mb * nb, |task, _| {
        let (bi, bj) = (task / nb, task % nb);
        let (m0, m1) = (bi * MC, ((bi + 1) * MC).min(m));
        let (n0, n1) = (bj * NC, ((bj + 1) * NC).min(n));
        let mut apack = vec![0f32; MC.div_ceil(MR) * MR * KC];
        let mut kc0 = 0;
        while kc0 < k {
            let kc = KC.min(k - kc0);
            // Pack A[m0..m1, kc0..kc0+kc] into MR-row panels: [panel][kk][MR].
            let mrows = m1 - m0;
            let mpanels = mrows.div_ceil(MR);
            for p in 0..mpanels {
                for r in 0..MR {
                    let row = m0 + p * MR + r;
                    let dst = &mut apack[p * kc * MR..];
                    if row < m1 {
                        let src = &a[row * lda + kc0..row * lda + kc0 + kc];
                        for kk in 0..kc {
                            dst[kk * MR + r] = src[kk];
                        }
                    } else {
                        for kk in 0..kc {
                            dst[kk * MR + r] = 0.0;
                        }
                    }
                }
            }
            let first = kc0 == 0;
            let mut np = n0 / NR;
            while np * NR < n1 {
                let bp = &b.data[(np * k + kc0) * NR..];
                let cols = (n1 - np * NR).min(NR);
                for p in 0..mpanels {
                    let rows = (mrows - p * MR).min(MR);
                    let row0 = m0 + p * MR;
                    let ap = &apack[p * kc * MR..];
                    let mut acc = [[0f32; NR]; MR];
                    kernel(kc, ap, bp, &mut acc);
                    for r in 0..rows {
                        let crow = unsafe { cp.slice((row0 + r) * ldc + np * NR, cols) };
                        for j in 0..cols {
                            let mut v = acc[r][j];
                            if first {
                                if let Some(bs) = bias {
                                    v += bs[np * NR + j];
                                }
                                if accumulate {
                                    v += crow[j];
                                }
                            } else {
                                v += crow[j];
                            }
                            crow[j] = v;
                        }
                    }
                }
                np += 1;
            }
            kc0 += kc;
        }
    });
}

/// 6×16 micro-kernel dispatch (AVX2 or scalar).
#[inline]
fn kernel(kc: usize, a: &[f32], b: &[f32], acc: &mut [[f32; NR]; MR]) {
    #[cfg(target_arch = "x86_64")]
    if crate::tensor::quant::has_avx2() {
        unsafe { kernel_avx2(kc, a.as_ptr(), b.as_ptr(), acc) };
        return;
    }
    for kk in 0..kc {
        for r in 0..MR {
            let av = a[kk * MR + r];
            for j in 0..NR {
                acc[r][j] += av * b[kk * NR + j];
            }
        }
    }
}

/// AVX2/FMA 6×16 micro-kernel: 12 ymm accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2(kc: usize, a: *const f32, b: *const f32, acc: &mut [[f32; NR]; MR]) {
    use std::arch::x86_64::*;
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();
    let mut c40 = _mm256_setzero_ps();
    let mut c41 = _mm256_setzero_ps();
    let mut c50 = _mm256_setzero_ps();
    let mut c51 = _mm256_setzero_ps();
    let mut ap = a;
    let mut bp = b;
    for _ in 0..kc {
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        let a0 = _mm256_broadcast_ss(&*ap);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        let a1 = _mm256_broadcast_ss(&*ap.add(1));
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        let a2 = _mm256_broadcast_ss(&*ap.add(2));
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        let a3 = _mm256_broadcast_ss(&*ap.add(3));
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        let a4 = _mm256_broadcast_ss(&*ap.add(4));
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        let a5 = _mm256_broadcast_ss(&*ap.add(5));
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
        ap = ap.add(MR);
        bp = bp.add(NR);
    }
    let st = |row: &mut [f32; NR], x: __m256, y: __m256| {
        _mm256_storeu_ps(row.as_mut_ptr(), x);
        _mm256_storeu_ps(row.as_mut_ptr().add(8), y);
    };
    st(&mut acc[0], c00, c01);
    st(&mut acc[1], c10, c11);
    st(&mut acc[2], c20, c21);
    st(&mut acc[3], c30, c31);
    st(&mut acc[4], c40, c41);
    st(&mut acc[5], c50, c51);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_naive() {
        let pool = ThreadPool::new(4, None);
        for &(m, n, k) in &[(1, 1, 1), (7, 17, 5), (100, 33, 300), (200, 300, 513)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
            let bias: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let pb = PackedB::from_nk(&w, n, k);
            let mut c = vec![0f32; m * n];
            sgemm(&pool, m, &a, k, &pb, Some(&bias), &mut c, n, false);
            for i in 0..m {
                for j in 0..n {
                    let r: f32 = (0..k).map(|kk| a[i * k + kk] * w[j * k + kk]).sum::<f32>() + bias[j];
                    assert!((r - c[i * n + j]).abs() < 1e-3 * (1.0 + r.abs()), "{m}x{n}x{k} at {i},{j}");
                }
            }
        }
    }
}
