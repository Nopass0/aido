//! GPU compute backend (Vulkan / DX12 / Metal through `wgpu`).
//!
//! No CUDA toolkit or vendor SDK is needed: any GPU with a modern driver
//! works. The backend accelerates the **dense, bandwidth-bound** part of a
//! model (projections, LM head) by keeping those matrices resident in VRAM,
//! while the CPU keeps the parts that are sparse or recurrent (routed MoE
//! experts, DeltaNet state updates). On the reference machine VRAM delivers
//! ~448 GB/s against ~13 GB/s for the single-channel DRAM, so every byte
//! moved from RAM to VRAM is read ~30× faster.
//!
//! ## GPU weight formats
//! Weights are repacked once at upload into two simple, coalescing-friendly
//! layouts (4-byte aligned, no bit-fiddling across word boundaries):
//!
//! * **G4** — 8 unsigned nibbles per `u32` in natural order, one `f32` scale
//!   and one `f32` min per 32 elements: `w = q·s − m`. Exact for `Q4_0`,
//!   `Q4_1`, `Q4_K` and colibri int4-gs64.
//! * **G8** — 4 signed bytes per `u32`, one `f32` scale per 32 elements.
//!   Exact for `Q8_0`; `Q5_K`/`Q6_K`/`F16` are re-quantised to 8 bit
//!   (error far below their own quantisation noise).
//!
//! ## Execution
//! One workgroup of 64 threads per output row; threads stride over the row
//! (perfectly coalesced 256-byte reads), keep up to 8 token accumulators in
//! registers and reduce through shared memory. Several matrices that read
//! the same input are encoded into **one command buffer with one readback**
//! ([`Gpu::matmul_many`]), so a whole projection group costs a single
//! CPU↔GPU synchronisation.

use std::sync::Mutex;

pub mod llama;

use anyhow::{bail, Context, Result};
use wgpu::util::DeviceExt;

use crate::tensor::quant::{dequantize_row, GgmlType};
use crate::tensor::QWeight;

/// Maximum tokens per GPU matmul call.
pub const MAX_T: usize = 8;
/// Capacity of the shared input buffer (floats).
const X_CAP: usize = MAX_T * 16384;
/// Capacity of the shared output buffer (floats).
const Y_CAP: usize = 4 << 20;

const SHADER: &str = r#"
struct Params { rows: u32, cols: u32, t: u32, out_stride: u32, out_offset: u32, kind: u32, x_offset: u32, flags: u32 };
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sc: array<f32>;
@group(0) @binding(2) var<storage, read> mn: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
var<workgroup> red: array<f32, 64>;

fn nib4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 15u), f32((w >> (sh + 4u)) & 15u), f32((w >> (sh + 8u)) & 15u), f32((w >> (sh + 12u)) & 15u));
}
fn byte4(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}

// Input vector element block `i` (vec4 index relative to the group start).
// flags & 1: the input is SwiGLU(gate, up) of an intermediate stored in the
// bound input buffer at x_offset (gate) and x_offset + cols (up).
fn xin(xb: u32, i: u32) -> vec4<f32> {
    if ((p.flags & 1u) != 0u) {
        let gv = x[xb + i];
        let uv = x[xb + i + p.cols / 4u];
        return gv / (vec4<f32>(1.0) + exp(-gv)) * uv;
    }
    return x[xb + i];
}

// Dot of one 32-element group of row `row` with token `t`.
fn group_dot(row: u32, gi: u32, t: u32) -> f32 {
    let groups = p.cols / 32u;
    let g = row * groups + gi;
    let xb = (p.x_offset + t * p.cols + gi * 32u) / 4u;
    var acc = 0.0;
    if (p.kind == 5u) {
        // symmetric int4, one scale per 64 elements: w = (q - 8) * s
        let q = qs[g];
        let s = sc[g / 2u];
        let e = vec4<f32>(8.0);
        acc += dot(nib4(q.x, 0u) - e, xin(xb, 0u)) + dot(nib4(q.x, 16u) - e, xin(xb, 1u));
        acc += dot(nib4(q.y, 0u) - e, xin(xb, 2u)) + dot(nib4(q.y, 16u) - e, xin(xb, 3u));
        acc += dot(nib4(q.z, 0u) - e, xin(xb, 4u)) + dot(nib4(q.z, 16u) - e, xin(xb, 5u));
        acc += dot(nib4(q.w, 0u) - e, xin(xb, 6u)) + dot(nib4(q.w, 16u) - e, xin(xb, 7u));
        acc *= s;
    } else if (p.kind == 4u) {
        let q = qs[g];
        let s = sc[g];
        let m = vec4<f32>(mn[g]);
        acc += dot(nib4(q.x, 0u) * s - m, xin(xb, 0u)) + dot(nib4(q.x, 16u) * s - m, xin(xb, 1u));
        acc += dot(nib4(q.y, 0u) * s - m, xin(xb, 2u)) + dot(nib4(q.y, 16u) * s - m, xin(xb, 3u));
        acc += dot(nib4(q.z, 0u) * s - m, xin(xb, 4u)) + dot(nib4(q.z, 16u) * s - m, xin(xb, 5u));
        acc += dot(nib4(q.w, 0u) * s - m, xin(xb, 6u)) + dot(nib4(q.w, 16u) * s - m, xin(xb, 7u));
    } else {
        let qa = qs[2u * g];
        let qb = qs[2u * g + 1u];
        var a2 = dot(byte4(qa.x), xin(xb, 0u)) + dot(byte4(qa.y), xin(xb, 1u)) + dot(byte4(qa.z), xin(xb, 2u)) + dot(byte4(qa.w), xin(xb, 3u));
        a2 += dot(byte4(qb.x), xin(xb, 4u)) + dot(byte4(qb.y), xin(xb, 5u)) + dot(byte4(qb.z), xin(xb, 6u)) + dot(byte4(qb.w), xin(xb, 7u));
        acc = a2 * sc[g];
    }
    return acc;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wg.x + wg.y * 65535u;
    let groups = p.cols / 32u;
    for (var t = 0u; t < p.t; t++) {
        var acc = 0.0;
        if (row < p.rows) {
            for (var gi = lid; gi < groups; gi += 64u) { acc += group_dot(row, gi, t); }
        }
        red[lid] = acc;
        workgroupBarrier();
        if (lid < 32u) { red[lid] += red[lid + 32u]; }
        workgroupBarrier();
        if (lid < 16u) { red[lid] += red[lid + 16u]; }
        workgroupBarrier();
        if (lid < 8u) { red[lid] += red[lid + 8u]; }
        workgroupBarrier();
        if (lid < 4u) { red[lid] += red[lid + 4u]; }
        workgroupBarrier();
        if (lid == 0u && row < p.rows) {
            y[p.out_offset + t * p.out_stride + row] = red[0] + red[1] + red[2] + red[3];
        }
        workgroupBarrier();
    }
}
"#;

/// A GPU device with compiled kernels and persistent I/O buffers.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    name: String,
    backend: String,
    max_binding: u64,
    xbuf: wgpu::Buffer,
    ybuf: wgpu::Buffer,
    stage: wgpu::Buffer,
    y2buf: wgpu::Buffer,
    stage2: wgpu::Buffer,
    lock: Mutex<()>,
    gemm: Mutex<Option<GemmState>>,
    pending: Mutex<Option<(usize, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>>,
    /// Bytes uploaded so far.
    pub vram_used: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gpu({})", self.describe())
    }
}

/// One row-range of an uploaded matrix with its cached bind group.
struct Part {
    rows: usize,
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
    _bufs: [wgpu::Buffer; 3],
}

/// A matrix resident in VRAM.
pub struct GpuMatrix {
    parts: Vec<Part>,
    /// Output dimension.
    pub rows: usize,
    /// Input dimension.
    pub cols: usize,
    kind: u32,
    /// VRAM bytes.
    pub nbytes: usize,
}

impl Gpu {
    /// Opens the highest-performance adapter and compiles the kernels.
    ///
    /// * **Output** — `Err` when no compatible GPU / driver is present.
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// println!("{}", g.describe());
    /// ```
    pub fn new() -> Result<Self> {
        let backends = match std::env::var("AIDO_GPU_BACKEND").as_deref() {
            Ok("dx12") => wgpu::Backends::DX12,
            Ok("vulkan") => wgpu::Backends::VULKAN,
            Ok("metal") => wgpu::Backends::METAL,
            _ => wgpu::Backends::PRIMARY,
        };
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("no GPU adapter found")?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("only a software adapter is available");
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor { label: Some("aido"), required_features: wgpu::Features::empty(), required_limits: limits.clone(), memory_hints: wgpu::MemoryHints::Performance },
            None,
        ))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("matvec"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipe.get_bind_group_layout(0);
        let max_binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size).min(1 << 30);
        let mk = |label, size: usize, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
        let xbuf = mk("x", X_CAP * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let ybuf = mk("y", Y_CAP * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let stage = mk("stage", Y_CAP * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let y2buf = mk("y2", 1 << 20, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let stage2 = mk("stage2", 1 << 20, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        Ok(Gpu {
            device,
            queue,
            pipe,
            layout,
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            max_binding,
            xbuf,
            ybuf,
            stage,
            y2buf,
            stage2,
            lock: Mutex::new(()),
            pending: Mutex::new(None),
            gemm: Mutex::new(None),
            vram_used: Default::default(),
        })
    }

    /// Adapter name and API, e.g. `NVIDIA GeForce RTX 3070 (Vulkan)`.
    ///
    /// # Example
    /// ```no_run
    /// println!("{}", aido::gpu::Gpu::new().unwrap().describe());
    /// ```
    pub fn describe(&self) -> String {
        format!("{} ({}) max binding {} MB", self.name, self.backend, self.max_binding >> 20)
    }

    /// Uploads a weight matrix, converting it to G4 or G8.
    ///
    /// * **Input** — CPU weight (any supported type, `cols % 32 == 0`).
    /// * **Output** — VRAM-resident matrix split into parts that respect the
    ///   adapter's binding-size limit.
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// let w = aido::tensor::QWeight::from_f32(2, 32, &[0.5; 64]);
    /// let m = g.upload(&w).unwrap();
    /// assert_eq!(m.rows, 2);
    /// ```
    pub fn upload(&self, w: &QWeight) -> Result<GpuMatrix> {
        anyhow::ensure!(w.cols % 32 == 0, "GPU matrices need cols % 32 == 0");
        let g4 = matches!(w.ty, GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K);
        let (qs, sc, mn) = if g4 { to_g4_rows(w) } else { to_g8_rows(w) };
        self.upload_packed(qs, sc, mn, w.rows, w.cols, if g4 { 4 } else { 8 })
    }

    /// Uploads a colibri int4-gs64 matrix (rows of `cols/2` bytes, f32 scale
    /// per 64) as G4 without any precision change.
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// let m = g.upload_i4g64(&vec![0x11u8; 32], &0.5f32.to_le_bytes(), 1, 64).unwrap();
    /// ```
    pub fn upload_i4g64(&self, w: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<GpuMatrix> {
        let groups = cols / 32;
        let mut qs = vec![0u32; rows * cols / 8];
        let mut sc = vec![0f32; rows * groups];
        let mut mn = vec![0f32; rows * groups];
        for r in 0..rows {
            let row = &w[r * cols / 2..(r + 1) * cols / 2];
            for e in 0..cols {
                let b = row[e / 2];
                let nib = if e % 2 == 0 { b & 15 } else { b >> 4 };
                let u = (nib ^ 8) as u32;
                qs[(r * cols + e) / 8] |= u << (4 * (e % 8));
            }
            for g in 0..groups {
                let at = (r * cols / 64 + g / 2) * 4;
                let s = f32::from_le_bytes(scales[at..at + 4].try_into().unwrap());
                sc[r * groups + g] = s;
                mn[r * groups + g] = 8.0 * s;
            }
        }
        self.upload_packed(qs, sc, mn, rows, cols, 4)
    }

    fn upload_packed(&self, qs: Vec<u32>, sc: Vec<f32>, mn: Vec<f32>, rows: usize, cols: usize, kind: u32) -> Result<GpuMatrix> {
        self.upload_packed_io(qs, sc, mn, rows, cols, kind, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_packed_io(&self, qs: Vec<u32>, sc: Vec<f32>, mn: Vec<f32>, rows: usize, cols: usize, kind: u32, chained: bool) -> Result<GpuMatrix> {
        let words_per_row = match kind { 8 => cols / 4, 0 => cols, _ => cols / 8 };
        let row_bytes = (words_per_row * 4) as u64;
        let max_rows = ((self.max_binding / row_bytes) as usize).clamp(1, rows.max(1));
        let groups = cols / 32;
        let mut parts = Vec::new();
        let mut nbytes = 0;
        let mut r0 = 0;
        while r0 < rows {
            let n = max_rows.min(rows - r0);
            let mk = |data: &[u8], label: &str| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
            let q = &qs[r0 * words_per_row..(r0 + n) * words_per_row];
            let sg = if kind == 5 { groups / 2 } else { groups };
            let s = &sc[r0 * sg..(r0 + n) * sg];
            let m = if kind == 4 { &mn[r0 * groups..(r0 + n) * groups] } else { &sc[0..1] };
            nbytes += q.len() * 4 + s.len() * 4 + m.len() * 4;
            let bufs = [mk(bytemuck::cast_slice(q), "qs"), mk(bytemuck::cast_slice(s), "sc"), mk(bytemuck::cast_slice(m), "mn")];
            let params = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("p"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bufs[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bufs[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bufs[2].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: if chained { self.ybuf.as_entire_binding() } else { self.xbuf.as_entire_binding() } },
                    wgpu::BindGroupEntry { binding: 4, resource: if chained { self.y2buf.as_entire_binding() } else { self.ybuf.as_entire_binding() } },
                    wgpu::BindGroupEntry { binding: 5, resource: params.as_entire_binding() },
                ],
            });
            parts.push(Part { rows: n, params, bind, _bufs: bufs });
            r0 += n;
        }
        self.vram_used.fetch_add(nbytes as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(GpuMatrix { parts, rows, cols, kind, nbytes })
    }

    /// Computes `out_i[t] = M_i · x[t]` for several matrices sharing the
    /// input, in one submission with one readback.
    ///
    /// * **Input** — `x`: `t × cols` floats (`t ≤ MAX_T`); `mats`; `outs`:
    ///   one slice of `t × rows_i` per matrix.
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// let m = g.upload(&aido::tensor::QWeight::from_f32(1, 32, &[1.0; 32])).unwrap();
    /// let mut out = [0f32; 1];
    /// g.matmul_many(&[1.0; 32], 1, &[&m], &mut [&mut out[..]]).unwrap();
    /// assert!((out[0] - 32.0).abs() < 0.5);
    /// ```
    pub fn matmul_many(&self, x: &[f32], t: usize, mats: &[&GpuMatrix], outs: &mut [&mut [f32]]) -> Result<()> {
        anyhow::ensure!(t >= 1 && t <= MAX_T, "GPU batch must be 1..={MAX_T}");
        let cols = mats[0].cols;
        anyhow::ensure!(mats.iter().all(|m| m.cols == cols), "matrices must share the input width");
        anyhow::ensure!(x.len() <= X_CAP, "GPU input too large");
        let total_rows: usize = mats.iter().map(|m| m.rows).sum();
        anyhow::ensure!(t * total_rows <= Y_CAP, "GPU output too large");
        let ybytes = (t * total_rows * 4) as u64;
        let _g = self.lock.lock().unwrap();
        self.queue.write_buffer(&self.xbuf, 0, bytemuck::cast_slice(x));
        let mut base = 0usize;
        for m in mats {
            let mut row0 = 0;
            for part in &m.parts {
                let params: [u32; 8] = [part.rows as u32, m.cols as u32, t as u32, m.rows as u32, (base + row0) as u32, m.kind, 0, 0];
                debug_assert!(m.parts.len() == 1 || row0 == 0 || true);
                self.queue.write_buffer(&part.params, 0, bytemuck::cast_slice(&params));
                row0 += part.rows;
            }
            base += t * m.rows;
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mm") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipe);
            for m in mats {
                for part in &m.parts {
                    pass.set_bind_group(0, &part.bind, &[]);
                    let gx = (part.rows as u32).min(65535);
                    let gy = (part.rows as u32).div_ceil(65535);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        }
        enc.copy_buffer_to_buffer(&self.ybuf, 0, &self.stage, 0, ybytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.stage.slice(0..ybytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        wait_spin(&self.device, &rx)?;
        {
            let data = slice.get_mapped_range();
            let all: &[f32] = bytemuck::cast_slice(&data);
            let mut base = 0;
            for (m, o) in mats.iter().zip(outs.iter_mut()) {
                o[..t * m.rows].copy_from_slice(&all[base..base + t * m.rows]);
                base += t * m.rows;
            }
        }
        self.stage.unmap();
        Ok(())
    }
}

/// A routed MoE expert resident in VRAM (SwiGLU FFN: `[gate|up]` then `down`).
pub struct GpuExpert {
    gu: GpuMatrix,
    down: GpuMatrix,
    /// VRAM bytes.
    pub nbytes: usize,
}

impl Gpu {
    /// Uploads one colibri int4-gs64 expert slab `[gate | up | down]` in the
    /// lossless symmetric-int4 GPU layout (one scale per 64, as in the file).
    ///
    /// * **Input** — packed weights, f32 scales (raw bytes), `inter`, `hidden`.
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// let (inter, hidden) = (64, 64);
    /// let w = vec![0u8; 3 * inter * hidden / 2];
    /// let s = vec![0u8; 3 * inter * hidden / 64 * 4];
    /// let e = g.upload_expert(&w, &s, inter, hidden).unwrap();
    /// ```
    pub fn upload_expert(&self, w: &[u8], scales: &[u8], inter: usize, hidden: usize) -> Result<GpuExpert> {
        let pack = |wb: &[u8], sb: &[u8], rows: usize, cols: usize| -> (Vec<u32>, Vec<f32>) {
            let mut qs = vec![0u32; rows * cols / 8];
            for r in 0..rows {
                for e in 0..cols {
                    let b = wb[(r * cols + e) / 2];
                    let nib = if e % 2 == 0 { b & 15 } else { b >> 4 };
                    qs[(r * cols + e) / 8] |= ((nib ^ 8) as u32) << (4 * (e % 8));
                }
            }
            let sc: Vec<f32> = sb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            (qs, sc)
        };
        let gu_rows = 2 * inter;
        let gu_bytes = gu_rows * hidden / 2;
        let gu_sc = gu_rows * hidden / 64 * 4;
        let (q1, s1) = pack(&w[..gu_bytes], &scales[..gu_sc], gu_rows, hidden);
        let (q2, s2) = pack(&w[gu_bytes..], &scales[gu_sc..], hidden, inter);
        let gu = self.upload_packed_io(q1, s1, vec![0.0], gu_rows, hidden, 5, false)?;
        let down = self.upload_packed_io(q2, s2, vec![0.0], hidden, inter, 5, true)?;
        let nbytes = gu.nbytes + down.nbytes;
        Ok(GpuExpert { gu, down, nbytes })
    }

    /// Starts computing `Σ`-ready outputs of `experts` for one token `x`
    /// asynchronously; collect them with [`Gpu::end_experts`]. The CPU is
    /// free to compute other experts meanwhile.
    ///
    /// # Example
    /// ```no_run
    /// # let g = aido::gpu::Gpu::new().unwrap();
    /// # let e: aido::gpu::GpuExpert = unimplemented!();
    /// g.begin_experts(&vec![0.0; 2048], &[&e]).unwrap();
    /// let mut out = vec![0f32; 2048];
    /// g.end_experts(&mut out).unwrap();
    /// ```
    pub fn begin_experts(&self, x: &[f32], experts: &[&GpuExpert]) -> Result<()> {
        if experts.is_empty() {
            return Ok(());
        }
        let hidden = experts[0].gu.cols;
        let inter2 = experts[0].gu.rows;
        let _g = self.lock.lock().unwrap();
        self.queue.write_buffer(&self.xbuf, 0, bytemuck::cast_slice(x));
        for (i, e) in experts.iter().enumerate() {
            let p1: [u32; 8] = [e.gu.parts[0].rows as u32, hidden as u32, 1, inter2 as u32, (i * inter2) as u32, 5, 0, 0];
            self.queue.write_buffer(&e.gu.parts[0].params, 0, bytemuck::cast_slice(&p1));
            let p2: [u32; 8] = [e.down.parts[0].rows as u32, e.down.cols as u32, 1, hidden as u32, (i * hidden) as u32, 5, (i * inter2) as u32, 1];
            self.queue.write_buffer(&e.down.parts[0].params, 0, bytemuck::cast_slice(&p2));
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("experts") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipe);
            for e in experts {
                pass.set_bind_group(0, &e.gu.parts[0].bind, &[]);
                pass.dispatch_workgroups(e.gu.rows as u32, 1, 1);
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipe);
            for e in experts {
                pass.set_bind_group(0, &e.down.parts[0].bind, &[]);
                pass.dispatch_workgroups(e.down.rows as u32, 1, 1);
            }
        }
        let bytes = (experts.len() * hidden * 4) as u64;
        enc.copy_buffer_to_buffer(&self.y2buf, 0, &self.stage2, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        self.stage2.slice(0..bytes).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        *self.pending.lock().unwrap() = Some((bytes as usize, rx));
        Ok(())
    }

    /// Waits for the experts started by [`Gpu::begin_experts`] and copies
    /// their outputs (`n_experts × hidden`) into `out`. No-op if none pending.
    ///
    /// # Example
    /// ```no_run
    /// # let g = aido::gpu::Gpu::new().unwrap();
    /// let mut out = vec![0f32; 0];
    /// g.end_experts(&mut out).unwrap();
    /// ```
    pub fn end_experts(&self, out: &mut [f32]) -> Result<()> {
        let Some((bytes, rx)) = self.pending.lock().unwrap().take() else { return Ok(()) };
        wait_spin(&self.device, &rx)?;
        {
            let slice = self.stage2.slice(0..bytes as u64);
            let data = slice.get_mapped_range();
            out[..bytes / 4].copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.stage2.unmap();
        Ok(())
    }
}


const GEMM_SHADER: &str = r#"
struct GP { t: u32, n: u32, k: u32, kind: u32, y_off: u32, y_stride: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> sc: array<f32>;
@group(0) @binding(2) var<storage, read> mn: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: GP;
var<workgroup> xs: array<f32, 2112>;
var<workgroup> ws: array<f32, 2112>;

fn weight(n: u32, kk: u32) -> f32 {
    let groups = p.k / 32u;
    if (p.kind == 8u) {
        let word = qs[(n * p.k + kk) / 4u];
        let b = i32(word << (24u - 8u * (kk % 4u))) >> 24u;
        return f32(b) * sc[n * groups + kk / 32u];
    } else if (p.kind == 4u) {
        let word = qs[(n * p.k + kk) / 8u];
        let q = f32((word >> (4u * (kk % 8u))) & 15u);
        let g = n * groups + kk / 32u;
        return q * sc[g] - mn[g];
    } else if (p.kind == 5u) {
        let word = qs[(n * p.k + kk) / 8u];
        let q = f32((word >> (4u * (kk % 8u))) & 15u) - 8.0;
        return q * sc[(n * groups + kk / 32u) / 2u];
    }
    return bitcast<f32>(qs[n * p.k + kk]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    for (var k0 = 0u; k0 < p.k; k0 += 32u) {
        for (var i = 0u; i < 8u; i++) {
            let idx = tid + i * 256u;
            let r = idx / 32u;
            let c = idx % 32u;
            let tr = row0 + r;
            var xv = 0.0;
            if (tr < p.t) { xv = x[tr * p.k + k0 + c]; }
            xs[r * 33u + c] = xv;
            let nc = col0 + r;
            var wv = 0.0;
            if (nc < p.n) { wv = weight(nc, k0 + c); }
            ws[r * 33u + c] = wv;
        }
        workgroupBarrier();
        for (var c = 0u; c < 32u; c++) {
            let a0 = xs[(lid.y * 4u + 0u) * 33u + c];
            let a1 = xs[(lid.y * 4u + 1u) * 33u + c];
            let a2 = xs[(lid.y * 4u + 2u) * 33u + c];
            let a3 = xs[(lid.y * 4u + 3u) * 33u + c];
            for (var j = 0u; j < 4u; j++) {
                let b = ws[(lid.x * 4u + j) * 33u + c];
                acc[j] += a0 * b;
                acc[4u + j] += a1 * b;
                acc[8u + j] += a2 * b;
                acc[12u + j] += a3 * b;
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i++) {
        let r = row0 + lid.y * 4u + i;
        if (r >= p.t) { continue; }
        for (var j = 0u; j < 4u; j++) {
            let c = col0 + lid.x * 4u + j;
            if (c < p.n) { y[p.y_off + r * p.y_stride + c] = acc[i * 4u + j]; }
        }
    }
}
"#;

/// Lazily created GEMM pipeline and growable I/O buffers.
struct GemmState {
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    x: Option<(wgpu::Buffer, u64)>,
    y: Option<(wgpu::Buffer, wgpu::Buffer, u64)>,
}

impl Gpu {
    fn gemm_state(&self) -> std::sync::MutexGuard<'_, Option<GemmState>> {
        let mut g = self.gemm.lock().unwrap();
        if g.is_none() {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gemm"), source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()) });
            let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gemm"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let layout = pipe.get_bind_group_layout(0);
            *g = Some(GemmState { pipe, layout, x: None, y: None });
        }
        g
    }

    /// Batched matrix product on the GPU: `out_i = X · M_iᵀ` for several
    /// matrices sharing the input `X` (`t × k`), in one submission. Tiled
    /// 64×64 output blocks with the weight tile dequantised into shared
    /// memory (G8 / G4 / symmetric G4 / f32 layouts).
    ///
    /// * **Input** — `x` (`t × k`), `t`, matrices with `cols == k`, outputs
    ///   (`t × rows_i` each).
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// let m = g.upload(&aido::tensor::QWeight::from_f32(2, 32, &[1.0; 64])).unwrap();
    /// let mut out = vec![0f32; 3 * 2];
    /// g.gemm_many(&vec![1.0; 3 * 32], 3, &[&m], &mut [&mut out[..]]).unwrap();
    /// assert!((out[5] - 32.0).abs() < 0.5);
    /// ```
    pub fn gemm_many(&self, x: &[f32], t: usize, mats: &[&GpuMatrix], outs: &mut [&mut [f32]]) -> Result<()> {
        let k = mats[0].cols;
        anyhow::ensure!(mats.iter().all(|m| m.cols == k) && k % 32 == 0, "gemm: shared k, multiple of 32");
        let total: usize = mats.iter().map(|m| m.rows).sum();
        let xbytes = (t * k * 4) as u64;
        let ybytes = (t * total * 4) as u64;
        let _l = self.lock.lock().unwrap();
        let mut st = self.gemm_state();
        let s = st.as_mut().unwrap();
        if s.x.as_ref().map_or(true, |b| b.1 < xbytes) {
            let sz = xbytes.next_power_of_two();
            s.x = Some((self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gx"), size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }), sz));
        }
        if s.y.as_ref().map_or(true, |b| b.2 < ybytes) {
            let sz = ybytes.next_power_of_two();
            let y = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gy"), size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let stg = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gs"), size: sz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            s.y = Some((y, stg, sz));
        }
        let (xb, (yb, sb, _)) = (&s.x.as_ref().unwrap().0, s.y.as_ref().unwrap());
        self.queue.write_buffer(xb, 0, bytemuck::cast_slice(&x[..t * k]));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gemm") });
        let mut binds = Vec::new();
        let mut base = 0usize;
        for m in mats {
            let mut row0 = 0;
            for part in &m.parts {
                let params: [u32; 8] = [t as u32, part.rows as u32, k as u32, m.kind, (base + row0) as u32, total as u32, 0, 0];
                let ub = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gp"), contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &s.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: part._bufs[0].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: part._bufs[1].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: part._bufs[2].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: xb.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: yb.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: ub.as_entire_binding() },
                    ],
                });
                binds.push((bg, ub, part.rows));
                row0 += part.rows;
            }
            base += m.rows;
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&s.pipe);
            for (bg, _, rows) in &binds {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups((*rows as u32).div_ceil(64), (t as u32).div_ceil(64), 1);
            }
        }
        enc.copy_buffer_to_buffer(yb, 0, sb, 0, ybytes);
        self.queue.submit(Some(enc.finish()));
        let slice = sb.slice(0..ybytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        {
            let data = slice.get_mapped_range();
            let all: &[f32] = bytemuck::cast_slice(&data);
            let mut col = 0;
            for (m, o) in mats.iter().zip(outs.iter_mut()) {
                for r in 0..t {
                    o[r * m.rows..(r + 1) * m.rows].copy_from_slice(&all[r * total + col..r * total + col + m.rows]);
                }
                col += m.rows;
            }
        }
        sb.unmap();
        Ok(())
    }

    /// Uploads an f32 matrix `[rows][cols]` exactly (no quantisation), for
    /// precision-sensitive weights (diffusion convolutions).
    ///
    /// # Example
    /// ```no_run
    /// let g = aido::gpu::Gpu::new().unwrap();
    /// let m = g.upload_f32(&[1.0; 64], 2, 32).unwrap();
    /// ```
    pub fn upload_f32(&self, w: &[f32], rows: usize, cols: usize) -> Result<GpuMatrix> {
        let qs: Vec<u32> = w.iter().map(|v| v.to_bits()).collect();
        self.upload_packed_io(qs, vec![0.0; rows * cols / 32], vec![0.0], rows, cols, 0, false)
    }
}

/// Waits for a mapping callback by busy-polling the device: on Windows a
/// blocking fence wait sleeps through the OS scheduler (≈0.3–0.5 ms wake-up),
/// while spinning returns as soon as the GPU is done (CPU cost is negligible
/// because the wait is short).
fn wait_spin(device: &wgpu::Device, rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>) -> Result<()> {
    let t = std::time::Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        match rx.try_recv() {
            Ok(r) => return Ok(r?),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if t.elapsed().as_millis() > 2 {
                    device.poll(wgpu::Maintain::Wait);
                    return Ok(rx.recv()??);
                }
                std::hint::spin_loop();
            }
            Err(e) => anyhow::bail!("GPU readback failed: {e}"),
        }
    }
}

/// Converts rows of a 4-bit weight to G4 (`w = q·s − m`, natural order),
/// decoding the GGML block structure exactly (no re-quantisation).
fn to_g4_rows(w: &QWeight) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let groups = w.cols / 32;
    let mut qs = vec![0u32; w.rows * w.cols / 8];
    let mut sc = vec![0f32; w.rows * groups];
    let mut mn = vec![0f32; w.rows * groups];
    let h = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
    let mut put = |e: usize, q: u8| qs[e / 8] |= (q as u32) << (4 * (e % 8));
    for r in 0..w.rows {
        let row = w.row(r);
        let base = r * w.cols;
        match w.ty {
            GgmlType::Q4_0 | GgmlType::Q4_1 => {
                let bb = if w.ty == GgmlType::Q4_0 { 18 } else { 20 };
                let off = bb - 16;
                for g in 0..groups {
                    let blk = &row[g * bb..(g + 1) * bb];
                    let d = h(blk);
                    sc[r * groups + g] = d;
                    mn[r * groups + g] = if w.ty == GgmlType::Q4_0 { 8.0 * d } else { -h(&blk[2..]) };
                    for j in 0..16 {
                        put(base + g * 32 + j, blk[off + j] & 15);
                        put(base + g * 32 + j + 16, blk[off + j] >> 4);
                    }
                }
            }
            GgmlType::Q4_K => {
                for sb in 0..w.cols / 256 {
                    let blk = &row[sb * 144..(sb + 1) * 144];
                    let (d, dmin) = (h(blk), h(&blk[2..]));
                    let scales = &blk[4..16];
                    let q = &blk[16..144];
                    for j in 0..4 {
                        for half_ in 0..2 {
                            let sub = 2 * j + half_;
                            let (s_, m_) = crate::tensor::quant::scale_min_k4(sub, scales);
                            let g = sb * 8 + sub;
                            sc[r * groups + g] = d * s_ as f32;
                            mn[r * groups + g] = dmin * m_ as f32;
                            for l in 0..32 {
                                let b = q[j * 32 + l];
                                put(base + g * 32 + l, if half_ == 0 { b & 15 } else { b >> 4 });
                            }
                        }
                    }
                }
            }
            _ => unreachable!("to_g4_rows called for {:?}", w.ty),
        }
    }
    (qs, sc, mn)
}

/// Converts rows of any weight to G8 (signed int8, scale per 32).
fn to_g8_rows(w: &QWeight) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let groups = w.cols / 32;
    let mut qs = vec![0u32; w.rows * w.cols / 4];
    let mut sc = vec![0f32; w.rows * groups];
    let mut deq = vec![0f32; w.cols];
    for r in 0..w.rows {
        if w.ty == GgmlType::Q8_0 {
            let row = w.row(r);
            for g in 0..groups {
                let blk = &row[g * 34..(g + 1) * 34];
                sc[r * groups + g] = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for k in 0..32 {
                    let e = r * w.cols + g * 32 + k;
                    qs[e / 4] |= (blk[2 + k] as u32) << (8 * (e % 4));
                }
            }
            continue;
        }
        dequantize_row(w.ty, w.row(r), &mut deq);
        for g in 0..groups {
            let v = &deq[g * 32..(g + 1) * 32];
            let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            sc[r * groups + g] = s;
            for (k, &x) in v.iter().enumerate() {
                let q = (x / s).round().clamp(-127.0, 127.0) as i32 as i8 as u8 as u32;
                let e = r * w.cols + g * 32 + k;
                qs[e / 4] |= q << (8 * (e % 4));
            }
        }
    }
    (qs, sc, vec![0.0])
}
