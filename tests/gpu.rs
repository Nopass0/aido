//! GPU backend correctness and speed (skipped when no GPU is present).

use aido::gpu::Gpu;
use aido::tensor::quant::GgmlType;
use aido::tensor::quantize::quantize_row;
use aido::tensor::{Bytes, QWeight};
use std::time::Instant;

fn matrix(ty: GgmlType, rows: usize, cols: usize) -> (QWeight, Vec<f32>) {
    let mut data = Vec::new();
    let mut f = Vec::new();
    for r in 0..rows {
        let row: Vec<f32> = (0..cols).map(|c| (((r * 31 + c * 17) % 97) as f32 - 48.0) / 40.0).collect();
        data.extend(quantize_row(ty, &row));
        f.extend(row);
    }
    (QWeight::new(ty, rows, cols, Bytes::from_vec(data)).unwrap(), f)
}

#[test]
fn gpu_matches_cpu() {
    let Ok(g) = Gpu::new() else {
        eprintln!("no GPU — skipped");
        return;
    };
    eprintln!("{}", g.describe());
    let (rows, cols, t) = (300, 512, 3);
    let x: Vec<f32> = (0..t * cols).map(|i| ((i * 7 % 23) as f32 - 11.0) / 9.0).collect();
    for ty in [GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::F16] {
        let (w, _) = matrix(ty, rows, cols);
        let m = g.upload(&w).unwrap();
        let mut out = vec![0f32; t * rows];
        g.matmul_many(&x, t, &[&m], &mut [&mut out[..]]).unwrap();
        let mut max_err = 0f32;
        let mut deq = vec![0f32; cols];
        for tt in 0..t {
            let xs = &x[tt * cols..(tt + 1) * cols];
            for r in 0..rows {
                w.dequant_row(r, &mut deq);
                let c: f32 = deq.iter().zip(xs).map(|(a, b)| a * b).sum();
                let e = (c - out[tt * rows + r]).abs() / (1.0 + c.abs());
                max_err = max_err.max(e);
            }
        }
        eprintln!("{ty:?}: max rel err {max_err:.2e}");
        assert!(max_err < if ty == GgmlType::Q6_K || ty == GgmlType::F16 { 0.1 } else { 1e-3 }, "{ty:?} error {max_err}");
    }
}

#[test]
fn gpu_bandwidth() {
    let Ok(g) = Gpu::new() else { return };
    let (rows, cols) = (16384, 4096);
    let (w, _) = matrix(GgmlType::Q4_K, rows, cols);
    let m = g.upload(&w).unwrap();
    let x = vec![0.5f32; cols];
    let mut out = vec![0f32; rows];
    for _ in 0..3 {
        g.matmul_many(&x, 1, &[&m], &mut [&mut out[..]]).unwrap();
    }
    // ramp clocks
    let w0 = Instant::now();
    while w0.elapsed().as_secs_f64() < 1.5 {
        g.matmul_many(&x, 1, &[&m], &mut [&mut out[..]]).unwrap();
    }
    let n = 50;
    let t = Instant::now();
    for _ in 0..n {
        g.matmul_many(&x, 1, &[&m], &mut [&mut out[..]]).unwrap();
    }
    let dt = t.elapsed().as_secs_f64() / n as f64;
    let m4: Vec<&aido::gpu::GpuMatrix> = vec![&m, &m, &m, &m];
    let mut o4 = vec![vec![0f32; rows]; 4];
    let t4 = Instant::now();
    for _ in 0..10 {
        let mut outs: Vec<&mut [f32]> = o4.iter_mut().map(|v| &mut v[..]).collect();
        g.matmul_many(&x, 1, &m4, &mut outs).unwrap();
    }
    let dt4 = t4.elapsed().as_secs_f64() / 10.0;
    eprintln!("GPU 4 matrices in one submit: {:.3} ms ({:.1} GB/s)", dt4 * 1e3, 4.0 * m.nbytes as f64 / dt4 / 1e9);
    eprintln!("GPU G4 matvec {}x{}: {:.3} ms/call, {:.1} GB/s effective", rows, cols, dt * 1e3, m.nbytes as f64 / dt / 1e9);
    let small = g.upload(&matrix(GgmlType::Q8_0, 64, 512).0).unwrap();
    let xs = vec![0.5f32; 512];
    let mut o = vec![0f32; 64];
    let t = Instant::now();
    for _ in 0..200 {
        g.matmul_many(&xs, 1, &[&small], &mut [&mut o[..]]).unwrap();
    }
    eprintln!("GPU round-trip latency (tiny matmul): {:.1} us", t.elapsed().as_secs_f64() / 200.0 * 1e6);
}

#[test]
fn gpu_expert_matches_cpu() {
    let Ok(g) = Gpu::new() else { return };
    use aido::tensor::quant::{deinterleave64, dot_i4g64, Q8Act};
    let (inter, hidden) = (128, 256);
    let n = 3 * inter * hidden;
    let w: Vec<u8> = (0..n / 2).map(|i| (i * 73 % 251) as u8).collect();
    let sc: Vec<u8> = (0..n / 64).flat_map(|i| (0.01 + (i % 7) as f32 * 0.002).to_le_bytes()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 11 % 19) as f32 - 9.0) / 7.0).collect();
    let e = g.upload_expert(&w, &sc, inter, hidden).unwrap();
    let e2 = g.upload_expert(&w, &sc, inter, hidden).unwrap();
    g.begin_experts(&x, &[&e, &e2]).unwrap();
    let mut out = vec![0f32; 2 * hidden];
    g.end_experts(&mut out).unwrap();
    // CPU reference (f32 maths on the same int4 weights).
    let deq = |r: usize, cols: usize, wo: usize, so: usize| -> Vec<f32> {
        (0..cols).map(|c| {
            let b = w[wo + (r * cols + c) / 2];
            let nib = if c % 2 == 0 { b & 15 } else { b >> 4 };
            let v = ((nib ^ 8) as i32 - 8) as f32;
            let si = so + (r * cols + c) / 64;
            v * f32::from_le_bytes(sc[si * 4..si * 4 + 4].try_into().unwrap())
        }).collect()
    };
    let mut gu = vec![0f32; 2 * inter];
    for r in 0..2 * inter {
        gu[r] = deq(r, hidden, 0, 0).iter().zip(&x).map(|(a, b)| a * b).sum();
    }
    let mid: Vec<f32> = (0..inter).map(|j| gu[j] / (1.0 + (-gu[j]).exp()) * gu[inter + j]).collect();
    let (wo, so) = (2 * inter * hidden / 2, 2 * inter * hidden / 64);
    let mut maxe = 0f32;
    for r in 0..hidden {
        let y: f32 = deq(r, inter, wo, so).iter().zip(&mid).map(|(a, b)| a * b).sum();
        maxe = maxe.max((y - out[r]).abs() / (1.0 + y.abs()));
        assert!((out[r] - out[hidden + r]).abs() < 1e-5);
    }
    let _ = (deinterleave64, dot_i4g64, Q8Act::default());
    eprintln!("expert max rel err {maxe:.2e}");
    assert!(maxe < 1e-3);
}

#[test]
fn gpu_latency_with_cpu_gaps() {
    let Ok(g) = Gpu::new() else { return };
    let (w1, _) = matrix(GgmlType::Q8_0, 12352, 2048);
    let m = g.upload(&w1).unwrap();
    let x = vec![0.5f32; 2048];
    let mut out = vec![0f32; 12352];
    for gap_us in [0u64, 200, 1000, 3000] {
        let mut tot = 0.0;
        let n = 60;
        for i in 0..n + 10 {
            let t = Instant::now();
            g.matmul_many(&x, 1, &[&m], &mut [&mut out[..]]).unwrap();
            if i >= 10 {
                tot += t.elapsed().as_secs_f64();
            }
            let s = Instant::now();
            while s.elapsed().as_micros() < gap_us as u128 {
                std::hint::spin_loop();
            }
        }
        eprintln!("DN in-proj (12352x2048 Q8) with {gap_us:>4} us CPU gaps: {:.3} ms/call", tot / n as f64 * 1e3);
    }
}

#[test]
fn gpu_gemm_matches_cpu() {
    let Ok(g) = Gpu::new() else { return };
    let (rows, cols, t) = (200, 256, 70);
    let x: Vec<f32> = (0..t * cols).map(|i| ((i * 7 % 23) as f32 - 11.0) / 9.0).collect();
    for ty in [GgmlType::Q8_0, GgmlType::Q4_K] {
        let (w, _) = matrix(ty, rows, cols);
        let m = g.upload(&w).unwrap();
        let mut out = vec![0f32; t * rows];
        g.gemm_many(&x, t, &[&m], &mut [&mut out[..]]).unwrap();
        let mut deq = vec![0f32; cols];
        let mut maxe = 0f32;
        for r in 0..rows {
            w.dequant_row(r, &mut deq);
            for tt in 0..t {
                let c: f32 = deq.iter().zip(&x[tt * cols..(tt + 1) * cols]).map(|(a, b)| a * b).sum();
                maxe = maxe.max((c - out[tt * rows + r]).abs() / (1.0 + c.abs()));
            }
        }
        eprintln!("gemm {ty:?} max rel err {maxe:.2e}");
        assert!(maxe < 1e-3);
    }
    // throughput
    let (w, _) = matrix(GgmlType::Q8_0, 4096, 2048);
    let m = g.upload(&w).unwrap();
    let x = vec![0.1f32; 512 * 2048];
    let mut out = vec![0f32; 512 * 4096];
    for _ in 0..3 {
        g.gemm_many(&x, 512, &[&m], &mut [&mut out[..]]).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..5 {
        g.gemm_many(&x, 512, &[&m], &mut [&mut out[..]]).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / 5.0;
    eprintln!("GPU GEMM 512x2048 · 2048x4096: {:.2} ms = {:.0} GFLOP/s (incl. transfers)", dt * 1e3, 2.0 * 512.0 * 2048.0 * 4096.0 / dt / 1e9);
}
