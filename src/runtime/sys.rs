//! Hardware discovery: CPU features, physical/logical cores, cache sizes and
//! measured memory bandwidth.
//!
//! The engine uses these numbers to make *quantitative* scheduling decisions
//! (see [`crate::runtime::governor`]): LLM token generation is bounded by
//! `bytes_of_weights / memory_bandwidth`, so knowing the real bandwidth lets
//! the engine predict the best achievable speed (the *roofline*) and stop
//! adding cores once that limit is reached.

use std::time::Instant;

use super::pool::ThreadPool;

/// Static description of the host CPU.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Brand string, e.g. `"Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz"`.
    pub brand: String,
    /// Logical processors visible to the OS.
    pub logical: usize,
    /// Physical cores (logical / SMT width).
    pub physical: usize,
    /// Hardware threads per core.
    pub smt: usize,
    /// L1 data cache per core, bytes.
    pub l1d: usize,
    /// L2 cache per core, bytes.
    pub l2: usize,
    /// Last-level cache (shared), bytes.
    pub l3: usize,
    /// AVX2 available.
    pub avx2: bool,
    /// FMA3 available.
    pub fma: bool,
    /// F16C (half <-> float conversion) available.
    pub f16c: bool,
    /// AVX-512F available.
    pub avx512f: bool,
}

impl CpuInfo {
    /// Detects the CPU via `cpuid` (x86-64) with safe fallbacks elsewhere.
    ///
    /// * **Input** — none.
    /// * **Output** — a filled [`CpuInfo`]; unknown fields fall back to
    ///   conservative defaults (32 KiB L1, 256 KiB L2, 8 MiB L3).
    ///
    /// # Example
    /// ```
    /// let cpu = aido::runtime::sys::CpuInfo::detect();
    /// assert!(cpu.logical >= cpu.physical && cpu.physical >= 1);
    /// ```
    pub fn detect() -> Self {
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        #[allow(unused_mut)]
        let mut info = CpuInfo {
            brand: String::from("unknown"),
            logical,
            physical: logical,
            smt: 1,
            l1d: 32 << 10,
            l2: 256 << 10,
            l3: 8 << 20,
            avx2: false,
            fma: false,
            f16c: false,
            avx512f: false,
        };
        #[cfg(target_arch = "x86_64")]
        #[allow(unused_unsafe)]
        unsafe {
            use std::arch::x86_64::__cpuid_count;
            info.avx2 = std::is_x86_feature_detected!("avx2");
            info.fma = std::is_x86_feature_detected!("fma");
            info.f16c = std::is_x86_feature_detected!("f16c");
            info.avx512f = std::is_x86_feature_detected!("avx512f");
            // Brand string.
            let maxext = __cpuid_count(0x8000_0000, 0).eax;
            if maxext >= 0x8000_0004 {
                let mut b = Vec::with_capacity(48);
                for leaf in 0x8000_0002u32..=0x8000_0004 {
                    let r = __cpuid_count(leaf, 0);
                    for v in [r.eax, r.ebx, r.ecx, r.edx] {
                        b.extend_from_slice(&v.to_le_bytes());
                    }
                }
                info.brand = String::from_utf8_lossy(&b).trim_matches(char::from(0)).trim().to_string();
            }
            let max = __cpuid_count(0, 0).eax;
            // SMT width via leaf 0xB (x2APIC topology).
            if max >= 0xB {
                let r = __cpuid_count(0xB, 0);
                let smt = (r.ebx & 0xFFFF) as usize;
                if smt >= 1 && smt <= 8 {
                    info.smt = smt;
                }
            }
            // Deterministic cache parameters (Intel leaf 4, AMD 0x8000001D).
            let leaf = if max >= 4 && info.brand.contains("Intel") { Some(4u32) } else if maxext >= 0x8000_001D { Some(0x8000_001D) } else { None };
            if let Some(leaf) = leaf {
                for sub in 0..8 {
                    let r = __cpuid_count(leaf, sub);
                    let ty = r.eax & 0x1F;
                    if ty == 0 {
                        break;
                    }
                    let level = (r.eax >> 5) & 0x7;
                    let ways = ((r.ebx >> 22) & 0x3FF) + 1;
                    let parts = ((r.ebx >> 12) & 0x3FF) + 1;
                    let line = (r.ebx & 0xFFF) + 1;
                    let sets = r.ecx + 1;
                    let size = (ways * parts * line * sets) as usize;
                    match (level, ty) {
                        (1, 1) => info.l1d = size,
                        (2, _) => info.l2 = size,
                        (3, _) => info.l3 = size,
                        _ => {}
                    }
                }
            }
        }
        info.physical = (info.logical / info.smt.max(1)).max(1);
        info
    }

    /// Returns one logical CPU id per physical core (SMT siblings skipped).
    ///
    /// Windows and Linux enumerate SMT siblings adjacently (`0,1` = core 0),
    /// so the list is `0, smt, 2*smt, …`.
    ///
    /// * **Output** — vector of logical CPU ids of length `physical`.
    ///
    /// # Example
    /// ```
    /// let cpu = aido::runtime::sys::CpuInfo::detect();
    /// assert_eq!(cpu.physical_core_ids().len(), cpu.physical);
    /// ```
    pub fn physical_core_ids(&self) -> Vec<usize> {
        (0..self.physical).map(|c| c * self.smt).collect()
    }
}

/// Measures sustained read bandwidth of main memory in GB/s.
///
/// * **Input** — `pool`: threads to use; `mib`: buffer size in MiB (should be
///   ≫ L3, e.g. 512).
/// * **Output** — best of 3 passes, in GB/s (1e9 bytes/s).
/// * **Behaviour** — each worker sums 64-bit words of its own slice; the
///   result is kept alive so the compiler cannot elide the loads.
///
/// # Example
/// ```no_run
/// use aido::runtime::{pool::ThreadPool, sys::memory_bandwidth_gbs};
/// let pool = ThreadPool::new(4, None);
/// let bw = memory_bandwidth_gbs(&pool, 256);
/// println!("{bw:.1} GB/s");
/// ```
pub fn memory_bandwidth_gbs(pool: &ThreadPool, mib: usize) -> f64 {
    let n = mib * (1 << 20) / 8;
    let buf: Vec<u64> = (0..n as u64).collect();
    let sink = std::sync::atomic::AtomicU64::new(0);
    let mut best = 0.0f64;
    let chunk = 1 << 16;
    for _ in 0..3 {
        let t = Instant::now();
        pool.parallel_chunks(n, chunk, |s, e, _| {
            let mut acc = [0u64; 4];
            let sl = &buf[s..e];
            let mut i = 0;
            while i + 4 <= sl.len() {
                acc[0] = acc[0].wrapping_add(sl[i]);
                acc[1] = acc[1].wrapping_add(sl[i + 1]);
                acc[2] = acc[2].wrapping_add(sl[i + 2]);
                acc[3] = acc[3].wrapping_add(sl[i + 3]);
                i += 4;
            }
            sink.fetch_add(acc[0] ^ acc[1] ^ acc[2] ^ acc[3], std::sync::atomic::Ordering::Relaxed);
        });
        let dt = t.elapsed().as_secs_f64();
        best = best.max((n * 8) as f64 / dt / 1e9);
    }
    std::hint::black_box(sink.into_inner());
    best
}
