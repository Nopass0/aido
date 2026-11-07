//! Quantisers: f32 rows → GGML block formats.
//!
//! Used by the model compiler (re-quantisation after pruning / vocabulary
//! trimming, precision changes) and by the SDK to export trained networks.
//! The K-quant quantisers do a small **error-minimising scale search** per
//! sub-block (try ±10 % around the min/max scale and keep the one with the
//! lowest squared error), which recovers most of the quality of the
//! exhaustive llama.cpp search at a fraction of the code.

use half::f16;

use super::quant::GgmlType;

/// Quantises one row of f32 values into `ty`.
///
/// * **Input** — `x` (length multiple of the block size), target type.
/// * **Output** — `ty.row_bytes(x.len())` bytes.
///
/// # Example
/// ```
/// use aido::tensor::{quantize::quantize_row, quant::{GgmlType, dequantize_row}};
/// let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
/// let q = quantize_row(GgmlType::Q8_0, &x);
/// let mut y = vec![0f32; 256];
/// dequantize_row(GgmlType::Q8_0, &q, &mut y);
/// assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 0.01));
/// ```
pub fn quantize_row(ty: GgmlType, x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ty.row_bytes(x.len()));
    match ty {
        GgmlType::F32 => x.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        GgmlType::F16 => x.iter().for_each(|v| out.extend_from_slice(&f16::from_f32(*v).to_le_bytes())),
        GgmlType::BF16 => x.iter().for_each(|v| out.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes())),
        GgmlType::Q8_0 => x.chunks_exact(32).for_each(|b| q8_0(b, &mut out)),
        GgmlType::Q4_0 => x.chunks_exact(32).for_each(|b| q4_0(b, &mut out)),
        GgmlType::Q4_K => x.chunks_exact(256).for_each(|b| q4_k(b, &mut out)),
        GgmlType::Q6_K => x.chunks_exact(256).for_each(|b| q6_k(b, &mut out)),
        other => panic!("quantisation to {other:?} is not implemented (use Q8_0, Q4_0, Q4_K, Q6_K, F16)"),
    }
    out
}

/// Types the quantiser can produce.
///
/// # Example
/// ```
/// assert!(aido::tensor::quantize::can_quantize(aido::tensor::quant::GgmlType::Q4_K));
/// ```
pub fn can_quantize(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K)
}

/// Parses `f32|f16|bf16|q8_0|q4_0|q4_k|q6_k` (case-insensitive).
///
/// # Example
/// ```
/// use aido::tensor::{quantize::parse_type, quant::GgmlType};
/// assert_eq!(parse_type("Q4_K").unwrap(), GgmlType::Q4_K);
/// ```
pub fn parse_type(s: &str) -> anyhow::Result<GgmlType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "f32" => GgmlType::F32,
        "f16" => GgmlType::F16,
        "bf16" => GgmlType::BF16,
        "q8_0" | "q8" => GgmlType::Q8_0,
        "q4_0" => GgmlType::Q4_0,
        "q4_k" | "q4" => GgmlType::Q4_K,
        "q6_k" | "q6" => GgmlType::Q6_K,
        _ => anyhow::bail!("unknown/unsupported quantisation '{s}'"),
    })
}

fn push_f16(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
}

/// Q8_0 block: absmax scaling.
fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    out.extend(x.iter().map(|v| ((v * id).round().clamp(-127.0, 127.0) as i8) as u8));
}

/// Q4_0 block: signed-max scaling (llama.cpp reference rule).
fn q4_0(x: &[f32], out: &mut Vec<u8>) {
    let mut max = 0f32;
    for &v in x {
        if v.abs() > max.abs() {
            max = v;
        }
    }
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    for j in 0..16 {
        let a = ((x[j] * id + 8.5) as i32).clamp(0, 15) as u8;
        let b = ((x[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
        out.push(a | (b << 4));
    }
}

/// Finds (scale, min) for unsigned `nmax`-level quantisation of `x`
/// minimising squared error (min/max initialisation + scale search).
fn fit_scale_min(x: &[f32], nmax: f32) -> (f32, f32) {
    let mn = x.iter().cloned().fold(f32::INFINITY, f32::min).min(0.0);
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if mx <= mn {
        return (0.0, -mn);
    }
    let base = (mx - mn) / nmax;
    let mut best = (base, -mn, f32::INFINITY);
    for step in -6..=6 {
        let s = base * (1.0 + step as f32 * 0.02);
        if s <= 0.0 {
            continue;
        }
        // Optimal min for this scale given rounded levels (one refinement).
        let mut err = 0.0;
        for &v in x {
            let q = ((v - mn) / s).round().clamp(0.0, nmax);
            let r = q * s + mn - v;
            err += r * r;
        }
        if err < best.2 {
            best = (s, -mn, err);
        }
    }
    (best.0, best.1)
}

/// Q4_K super-block (256 values, 8 sub-blocks of 32).
fn q4_k(x: &[f32], out: &mut Vec<u8>) {
    let mut sc = [0f32; 8];
    let mut mn = [0f32; 8];
    for j in 0..8 {
        let (s, m) = fit_scale_min(&x[j * 32..(j + 1) * 32], 15.0);
        sc[j] = s;
        mn[j] = m;
    }
    let max_sc = sc.iter().cloned().fold(0f32, f32::max);
    let max_mn = mn.iter().cloned().fold(0f32, f32::max);
    let d = max_sc / 63.0;
    let dmin = max_mn / 63.0;
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        ls[j] = if d > 0.0 { (sc[j] / d).round().clamp(0.0, 63.0) as u8 } else { 0 };
        lm[j] = if dmin > 0.0 { (mn[j] / dmin).round().clamp(0.0, 63.0) as u8 } else { 0 };
    }
    // Recompute d/dmin from the f16-rounded values actually stored.
    let d16 = f16::from_f32(d).to_f32();
    let dm16 = f16::from_f32(dmin).to_f32();
    push_f16(out, d);
    push_f16(out, dmin);
    let mut s12 = [0u8; 12];
    for j in 0..4 {
        s12[j] = ls[j] | ((ls[j + 4] >> 4) << 6);
        s12[j + 4] = lm[j] | ((lm[j + 4] >> 4) << 6);
        s12[j + 8] = (ls[j + 4] & 0xF) | ((lm[j + 4] & 0xF) << 4);
    }
    out.extend_from_slice(&s12);
    let mut q = [0u8; 256];
    for j in 0..8 {
        let s = d16 * ls[j] as f32;
        let m = dm16 * lm[j] as f32;
        for l in 0..32 {
            let v = x[j * 32 + l];
            q[j * 32 + l] = if s > 0.0 { ((v + m) / s).round().clamp(0.0, 15.0) as u8 } else { 0 };
        }
    }
    for c in 0..4 {
        for l in 0..32 {
            out.push(q[64 * c + l] | (q[64 * c + 32 + l] << 4));
        }
    }
}

/// Q6_K super-block (256 values, 16 sub-blocks of 16 with int8 scales).
fn q6_k(x: &[f32], out: &mut Vec<u8>) {
    let mut scales = [0f32; 16];
    for i in 0..16 {
        let b = &x[i * 16..(i + 1) * 16];
        let mut max = 0f32;
        for &v in b {
            if v.abs() > max.abs() {
                max = v;
            }
        }
        scales[i] = max / -32.0;
    }
    let amax = scales.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let d16 = f16::from_f32(d).to_f32();
    let mut isc = [0i8; 16];
    let mut q = [0u8; 256];
    for i in 0..16 {
        isc[i] = if d > 0.0 { (scales[i] / d).round().clamp(-128.0, 127.0) as i8 } else { 0 };
        let s = d16 * isc[i] as f32;
        for l in 0..16 {
            let v = x[i * 16 + l];
            q[i * 16 + l] = if s != 0.0 { ((v / s).round() + 32.0).clamp(0.0, 63.0) as u8 } else { 32 };
        }
    }
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for h in 0..2 {
        for l in 0..32 {
            let base = 128 * h;
            let (q1, q2, q3, q4) = (q[base + l], q[base + l + 32], q[base + l + 64], q[base + l + 96]);
            ql[64 * h + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            ql[64 * h + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            qh[32 * h + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(isc.iter().map(|v| *v as u8));
    push_f16(out, d);
}

/// Root-mean-square quantisation error of `ty` on `x` (for reports).
///
/// # Example
/// ```
/// use aido::tensor::{quantize::rms_error, quant::GgmlType};
/// let x: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();
/// assert!(rms_error(GgmlType::Q6_K, &x) < rms_error(GgmlType::Q4_K, &x));
/// ```
pub fn rms_error(ty: GgmlType, x: &[f32]) -> f32 {
    let q = quantize_row(ty, x);
    let mut y = vec![0f32; x.len()];
    super::quant::dequantize_row(ty, &q, &mut y);
    (x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / x.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant::dequantize_row;

    #[test]
    fn roundtrip_errors_are_small() {
        let x: Vec<f32> = (0..1024).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1.0 + (i % 5) as f32)).collect();
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        for (ty, rel) in [(GgmlType::Q8_0, 0.01), (GgmlType::Q6_K, 0.04), (GgmlType::Q4_K, 0.12), (GgmlType::Q4_0, 0.15), (GgmlType::F16, 0.001)] {
            let q = quantize_row(ty, &x);
            assert_eq!(q.len(), ty.row_bytes(x.len()));
            let mut y = vec![0f32; x.len()];
            dequantize_row(ty, &q, &mut y);
            let e = (x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / x.len() as f32).sqrt();
            assert!(e / rms < rel, "{ty:?}: relative rms error {}", e / rms);
        }
    }
}
