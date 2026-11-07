//! # Whole-model GPU decoding for Llama-family LLMs
//!
//! On a PC with slow (single-channel) RAM the CPU reads weights at
//! ~14 GB/s while the GPU reads VRAM at ~450 GB/s. Offloading single
//! matrix products does not help much: each CPU↔GPU round trip costs
//! ~0.15 ms and a token needs hundreds of them. This module therefore keeps
//! **everything** on the GPU — weights (G4/G8, see [`super`]), activations,
//! norm gains and the KV cache — and records the complete forward pass of a
//! token (embedding row in, logits out) as **one command buffer with one
//! readback**. A batch of prompt tokens is recorded into the same command
//! buffer, so prefill also costs a single synchronisation.
//!
//! Kernels (WGSL): quantised mat-vec with fused bias / residual-add /
//! SwiGLU-input, RMSNorm (full or per head), RoPE (NeoX or GPT-J), KV write
//! (packed f16 pairs), single-query attention (one workgroup per head,
//! softmax in shared memory).
//!
//! The KV cache lives in one buffer per layer (`[pos][K pairs | V pairs]`,
//! f16) and grows by doubling, so memory follows the actual context.
//! [`GpuLlama::try_new`] declines (and the CPU path is used) when weights do
//! not fit the VRAM budget or the model uses run-time layer skipping or
//! steering vectors.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use super::{to_g4_rows, to_g8_rows, wait_spin, Gpu};
use crate::llm::model::Model;
use crate::llm::{LanguageModel, ModelState};
use crate::runtime::pool::ThreadPool;
use crate::tensor::quant::GgmlType;
use crate::tensor::QWeight;

const MATVEC: &str = r#"
struct P { rows: u32, cols: u32, kind: u32, x_off: u32, y_off: u32, flags: u32, b_off: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sc: array<f32>;
@group(0) @binding(2) var<storage, read> mn: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> cst: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;

fn nib4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 15u), f32((w >> (sh + 4u)) & 15u), f32((w >> (sh + 8u)) & 15u), f32((w >> (sh + 12u)) & 15u));
}
fn byte4(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}
// 4 input elements starting at element e (offsets are multiples of 4: one vec4 load).
fn xin(e: u32) -> vec4<f32> {
    return act[(p.x_off + e) / 4u];
}
fn group_dot(row: u32, gi: u32) -> f32 {
    let g = row * (p.cols / 32u) + gi;
    let e = gi * 32u;
    if (p.kind == 4u) {
        let q = qs[g];
        let s = sc[g];
        let m = vec4<f32>(mn[g]);
        var a = dot(nib4(q.x, 0u) * s - m, xin(e)) + dot(nib4(q.x, 16u) * s - m, xin(e + 4u));
        a += dot(nib4(q.y, 0u) * s - m, xin(e + 8u)) + dot(nib4(q.y, 16u) * s - m, xin(e + 12u));
        a += dot(nib4(q.z, 0u) * s - m, xin(e + 16u)) + dot(nib4(q.z, 16u) * s - m, xin(e + 20u));
        a += dot(nib4(q.w, 0u) * s - m, xin(e + 24u)) + dot(nib4(q.w, 16u) * s - m, xin(e + 28u));
        return a;
    }
    let qa = qs[2u * g];
    let qb = qs[2u * g + 1u];
    var a2 = dot(byte4(qa.x), xin(e)) + dot(byte4(qa.y), xin(e + 4u)) + dot(byte4(qa.z), xin(e + 8u)) + dot(byte4(qa.w), xin(e + 12u));
    a2 += dot(byte4(qb.x), xin(e + 16u)) + dot(byte4(qb.y), xin(e + 20u)) + dot(byte4(qb.z), xin(e + 24u)) + dot(byte4(qb.w), xin(e + 28u));
    return a2 * sc[g];
}
// 8 rows per workgroup, 32 lanes per row striding over the 32-element groups.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let lane = lid % 32u;
    let r8 = lid / 32u;
    let row = (wg.x + wg.y * 65535u) * 8u + r8;
    var acc = 0.0;
    if (row < p.rows) {
        for (var gi = lane; gi < p.cols / 32u; gi += 32u) { acc += group_dot(row, gi); }
    }
    red[lid] = acc;
    workgroupBarrier();
    if (lane < 16u) { red[lid] += red[lid + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { red[lid] += red[lid + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { red[lid] += red[lid + 4u]; }
    workgroupBarrier();
    if (lane == 0u && row < p.rows) {
        var v = red[lid] + red[lid + 1u] + red[lid + 2u] + red[lid + 3u];
        if ((p.flags & 2u) != 0u) { v += cst[p.b_off + row]; }
        let yi = p.y_off + row;
        if ((p.flags & 4u) != 0u) { act[yi / 4u][yi % 4u] += v; } else { act[yi / 4u][yi % 4u] = v; }
    }
}
"#;

const NORM: &str = r#"
struct P { len: u32, x_off: u32, y_off: u32, w_off: u32, eps: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> act: array<f32>;
@group(0) @binding(1) var<storage, read> cst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
// One workgroup per vector of `len` (wg.x = head index for per-head norms).
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let xb = p.x_off + wg.x * p.len;
    let yb = p.y_off + wg.x * p.len;
    var s = 0.0;
    for (var i = lid; i < p.len; i += 256u) { let v = act[xb + i]; s += v * v; }
    red[lid] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    let r = inverseSqrt(red[0] / f32(p.len) + p.eps);
    workgroupBarrier();
    for (var i = lid; i < p.len; i += 256u) { act[yb + i] = act[xb + i] * r * cst[p.w_off + i]; }
}
"#;

const SILU: &str = r#"
struct P { gate: u32, up: u32, n: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32 };
@group(0) @binding(0) var<storage, read_write> act: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p: P;
// gate ← silu(gate) · up, 4 elements per thread.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n / 4u) { return; }
    let g = act[p.gate / 4u + i];
    act[p.gate / 4u + i] = g / (vec4<f32>(1.0) + exp(-g)) * act[p.up / 4u + i];
}
"#;

const ROPE: &str = r#"
struct P { q_off: u32, k_off: u32, n_q: u32, n_k: u32, hd: u32, half: u32, neox: u32, _a: u32 };
@group(0) @binding(0) var<storage, read_write> act: array<f32>;
@group(0) @binding(1) var<storage, read> step: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
// One thread per rotated pair of one head (query heads first, then key heads).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = (p.n_q + p.n_k) * p.half;
    if (i >= total) { return; }
    let head = i / p.half;
    let j = i % p.half;
    var base = p.q_off + head * p.hd;
    if (head >= p.n_q) { base = p.k_off + (head - p.n_q) * p.hd; }
    let c = step[4u + 2u * j];
    let s = step[5u + 2u * j];
    var a = base + 2u * j;
    var b = a + 1u;
    if (p.neox != 0u) { a = base + j; b = base + j + p.half; }
    let x = act[a];
    let y = act[b];
    act[a] = x * c - y * s;
    act[b] = x * s + y * c;
}
"#;

const KVSTORE: &str = r#"
struct P { k_off: u32, v_off: u32, kvd: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32 };
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> step: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let half = p.kvd / 2u;
    if (i >= half) { return; }
    let pos = u32(step[0]);
    let base = pos * p.kvd;
    kv[base + i] = pack2x16float(vec2<f32>(act[p.k_off + 2u * i], act[p.k_off + 2u * i + 1u]));
    kv[base + half + i] = pack2x16float(vec2<f32>(act[p.v_off + 2u * i], act[p.v_off + 2u * i + 1u]));
}
"#;

const ATTN: &str = r#"
struct P { q_off: u32, o_off: u32, s_off: u32, group: u32, hd: u32, kvd: u32, max_ctx: u32, scale: f32 };
@group(0) @binding(0) var<storage, read_write> act: array<f32>;
@group(0) @binding(1) var<storage, read> step: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
// One workgroup per query head; positions 0..=pos.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wg.x;
    let n = u32(step[0]) + 1u;
    let kvh = h / p.group;
    let hp = p.hd / 2u;
    let qb = p.q_off + h * p.hd;
    let sb = p.s_off + h * p.max_ctx;
    let kb = kvh * hp;
    var mx = -3.0e38;
    for (var j = lid; j < n; j += 256u) {
        let row = j * p.kvd + kb;
        var s = 0.0;
        for (var d = 0u; d < hp; d++) {
            let k = unpack2x16float(kv[row + d]);
            s += k.x * act[qb + 2u * d] + k.y * act[qb + 2u * d + 1u];
        }
        s *= p.scale;
        act[sb + j] = s;
        mx = max(mx, s);
    }
    red[lid] = mx;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] = max(red[lid], red[lid + k]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = lid; j < n; j += 256u) {
        let e = exp(act[sb + j] - m);
        act[sb + j] = e;
        sum += e;
    }
    red[lid] = sum;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    if (lid < hp) {
        var acc = vec2<f32>(0.0);
        let vb = p.kvd / 2u + kb + lid;
        for (var j = 0u; j < n; j++) {
            acc += act[sb + j] * unpack2x16float(kv[j * p.kvd + vb]);
        }
        act[p.o_off + h * p.hd + 2u * lid] = acc.x * inv;
        act[p.o_off + h * p.hd + 2u * lid + 1u] = acc.y * inv;
    }
}
"#;

/// A weight matrix in VRAM.
struct Mat {
    bufs: [wgpu::Buffer; 3],
    rows: usize,
    cols: usize,
    kind: u32,
}

/// One recorded dispatch.
enum Op {
    /// Fixed kernel and bindings.
    Fixed { pipe: usize, bind: wgpu::BindGroup, groups: (u32, u32) },
    /// KV write of `layer` (bindings depend on the state's KV buffer).
    KvStore(usize),
    /// Attention of `layer`.
    Attn(usize),
}

/// Offsets (in floats) of activations inside the shared buffer.
#[derive(Clone, Copy, Debug)]
struct Lay {
    x: usize,
    xb: usize,
    q: usize,
    k: usize,
    v: usize,
    att: usize,
    gate: usize,
    up: usize,
    logits: usize,
    scores: usize,
    total: usize,
}

struct Res {
    act: wgpu::Buffer,
    step: wgpu::Buffer,
    step_all: wgpu::Buffer,
    x_all: wgpu::Buffer,
    logits_all: wgpu::Buffer,
    stage: wgpu::Buffer,
}

/// Llama-family model running entirely on the GPU (see the module docs).
pub struct GpuLlama {
    cpu: Model,
    gpu: Arc<Gpu>,
    pipes: Vec<wgpu::ComputePipeline>,
    ops: Vec<Op>,
    kv_params: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    lay: Lay,
    max_ctx: usize,
    step_floats: usize,
    chunk: usize,
    res: Mutex<Res>,
    _mats: Vec<Mat>,
    _cst: wgpu::Buffer,
    /// VRAM bytes of weights.
    pub weight_bytes: usize,
}

/// Per-sequence GPU state: KV buffers (one per layer) and host logits.
pub struct GpuState {
    logits: Vec<f32>,
    n_ctx: usize,
    max_t: usize,
    cap: usize,
    kv: Vec<wgpu::Buffer>,
    binds: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    bytes: usize,
}

impl ModelState for GpuState {
    fn logits(&self) -> &[f32] {
        &self.logits
    }
    fn max_t(&self) -> usize {
        self.max_t
    }
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn nbytes(&self) -> usize {
        self.bytes
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn align4(x: usize) -> usize {
    x.div_ceil(4) * 4
}

impl GpuLlama {
    /// Bytes the weights would occupy in VRAM (G4 ≈ 5 bits, G8 ≈ 9 bits per weight).
    ///
    /// # Example
    /// ```no_run
    /// let m = aido::llm::model::Model::load("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf").unwrap();
    /// println!("{:.2} GB", aido::gpu::llama::GpuLlama::vram_needed(&m) as f64 / 1e9);
    /// ```
    pub fn vram_needed(m: &Model) -> usize {
        let one = |w: &QWeight| {
            let g4 = matches!(w.ty, GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K);
            w.rows * w.cols / if g4 { 2 } else { 1 } + w.rows * w.cols / 32 * if g4 { 8 } else { 4 }
        };
        m.layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].iter().map(|w| one(w)).sum::<usize>()).sum::<usize>() + one(&m.output)
    }

    /// Moves `m` to the GPU when it fits `budget` bytes of VRAM; otherwise
    /// returns the model back (`Err(m)`) so the caller keeps the CPU path.
    ///
    /// # Example
    /// ```no_run
    /// let gpu = std::sync::Arc::new(aido::gpu::Gpu::new().unwrap());
    /// let m = aido::llm::model::Model::load("models/Qwen3-0.6B-Q8_0.gguf").unwrap();
    /// match aido::gpu::llama::GpuLlama::try_new(m, gpu, 5 << 30, 8192) {
    ///     Ok(g) => println!("on GPU: {:.2} GB", g.weight_bytes as f64 / 1e9),
    ///     Err((m, why)) => println!("CPU: {why}"),
    /// }
    /// ```
    pub fn try_new(m: Model, gpu: Arc<Gpu>, budget: usize, max_ctx: usize) -> std::result::Result<Self, (Model, String)> {
        if m.skip.iter().any(|s| *s) || m.steer.iter().any(|s| s.is_some()) {
            return Err((m, "layer skipping / steering vectors run on the CPU path".into()));
        }
        let c = &m.cfg;
        if c.head_dim % 2 != 0 || c.head_dim > 512 || c.n_embd % 32 != 0 {
            return Err((m, "unsupported head/embedding size for the GPU path".into()));
        }
        let need = Self::vram_needed(&m);
        if need > budget {
            return Err((m, format!("weights need {:.2} GB of VRAM, budget {:.2} GB", need as f64 / 1e9, budget as f64 / 1e9)));
        }
        let max_ctx = max_ctx.min(c.ctx_train.max(2048)).max(256);
        match Self::build(&m, &gpu, max_ctx) {
            Ok((pipes, ops, kv_params, lay, step_floats, res, mats, cst, bytes)) => {
                let chunk = 256;
                gpu.vram_used.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(GpuLlama { cpu: m, gpu, pipes, ops, kv_params, lay, max_ctx, step_floats, chunk, res: Mutex::new(res), _mats: mats, _cst: cst, weight_bytes: bytes })
            }
            Err(e) => Err((m, format!("GPU setup failed: {e:#}"))),
        }
    }

    #[allow(clippy::type_complexity)]
    fn build(m: &Model, g: &Gpu, max_ctx: usize) -> Result<(Vec<wgpu::ComputePipeline>, Vec<Op>, Vec<(wgpu::Buffer, wgpu::Buffer)>, Lay, usize, Res, Vec<Mat>, wgpu::Buffer, usize)> {
        let c = &m.cfg;
        let dev = &g.device;
        let (d, qd, kvd, hd) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim);
        let ff = m.layers.iter().map(|l| l.w_gate.rows).max().unwrap_or(c.n_ff);
        // Activation layout.
        let mut o = 0;
        let mut take = |n: usize| {
            let a = o;
            o += align4(n);
            a
        };
        let lay = Lay { x: take(d), xb: take(d.max(qd)), q: take(qd), k: take(kvd), v: take(kvd), att: take(qd), gate: take(ff), up: take(ff), logits: take(c.vocab), scores: take(c.n_head * max_ctx), total: 0 };
        let lay = Lay { total: o, ..lay };
        anyhow::ensure!((lay.total * 4) as u64 <= g.max_binding, "activation buffer too large for this GPU");
        // Constants: norm gains and biases.
        let mut cst: Vec<f32> = Vec::new();
        let mut put = |v: &[f32]| {
            let a = cst.len();
            cst.extend_from_slice(v);
            while cst.len() % 4 != 0 {
                cst.push(0.0);
            }
            a
        };
        struct LOff {
            n1: usize,
            n2: usize,
            bq: Option<usize>,
            bk: Option<usize>,
            bv: Option<usize>,
            qn: Option<usize>,
            kn: Option<usize>,
        }
        let loffs: Vec<LOff> = m
            .layers
            .iter()
            .map(|l| LOff {
                n1: put(&l.attn_norm),
                n2: put(&l.ffn_norm),
                bq: l.bq.as_deref().map(&mut put),
                bk: l.bk.as_deref().map(&mut put),
                bv: l.bv.as_deref().map(&mut put),
                qn: l.q_norm.as_deref().map(&mut put),
                kn: l.k_norm.as_deref().map(&mut put),
            })
            .collect();
        let out_norm = put(&m.out_norm);
        let cst_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("cst"), contents: bytemuck::cast_slice(&cst), usage: wgpu::BufferUsages::STORAGE });
        let mk = |label: &str, size: usize, usage: wgpu::BufferUsages| dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let act = mk("act", lay.total * 4, st);
        let step_floats = align4(4 + c.rope_dim);
        let chunk = 256;
        let res = Res {
            act,
            step: mk("step", step_floats * 4, st),
            step_all: mk("step_all", step_floats * 4 * chunk, st),
            x_all: mk("x_all", d * 4 * chunk, st),
            logits_all: mk("logits_all", c.vocab * 4 * 16, st),
            stage: mk("stage", c.vocab * 4 * 16, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        };
        // Pipelines.
        let pipe = |src: &str, label: &str| {
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        };
        let pipes = vec![pipe(MATVEC, "mv"), pipe(NORM, "norm"), pipe(ROPE, "rope"), pipe(KVSTORE, "kvs"), pipe(ATTN, "attn"), pipe(SILU, "silu")];
        let uni = |words: &[u32]| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("p"), contents: bytemuck::cast_slice(words), usage: wgpu::BufferUsages::UNIFORM });
        // Weights.
        let mut mats: Vec<Mat> = Vec::new();
        let mut bytes = 0usize;
        // Uploads one matrix, or several stacked by rows (same input, same
        // GPU format) so that e.g. q|k|v run as a single dispatch.
        let mut upload_cat = |ws: &[&QWeight]| -> Result<usize> {
            let w = ws[0];
            anyhow::ensure!(w.cols % 32 == 0, "cols % 32");
            let g4 = matches!(w.ty, GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K);
            let (mut qs, mut sc, mut mn) = (Vec::new(), Vec::new(), Vec::new());
            let mut rows = 0;
            for x in ws {
                let (a, b, c2) = if g4 { to_g4_rows(x) } else { to_g8_rows(x) };
                qs.extend(a);
                sc.extend(b);
                mn.extend(c2);
                rows += x.rows;
            }
            let mnv = if g4 { mn } else { vec![0f32; 4] };
            let b = |data: &[u8], l: &str| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(l), contents: data, usage: wgpu::BufferUsages::STORAGE });
            bytes += qs.len() * 4 + sc.len() * 4 + mnv.len() * 4;
            anyhow::ensure!((qs.len() * 4) as u64 <= g.max_binding, "matrix larger than the GPU binding limit");
            mats.push(Mat { bufs: [b(bytemuck::cast_slice(&qs), "qs"), b(bytemuck::cast_slice(&sc), "sc"), b(bytemuck::cast_slice(&mnv), "mn")], rows, cols: w.cols, kind: if g4 { 4 } else { 8 } });
            Ok(mats.len() - 1)
        };
        let g4 = |w: &QWeight| matches!(w.ty, GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K);
        struct LM {
            /// `[q|k|v]` fused (None → separate q, k, v).
            qkv: Option<usize>,
            q: usize,
            k: usize,
            v: usize,
            o: usize,
            /// `[gate|up]` fused (None → separate).
            gu: Option<usize>,
            gate: usize,
            up: usize,
            down: usize,
        }
        let mut lms = Vec::new();
        let biased = m.layers.iter().any(|l| l.bq.is_some() || l.bk.is_some() || l.bv.is_some());
        for l in &m.layers {
            let same_qkv = !biased && g4(&l.wq) == g4(&l.wk) && g4(&l.wq) == g4(&l.wv);
            let same_gu = g4(&l.w_gate) == g4(&l.w_up) && l.w_gate.rows == l.w_up.rows;
            let (qkv, q, k, v) = if same_qkv { (Some(upload_cat(&[&l.wq, &l.wk, &l.wv])?), 0, 0, 0) } else { (None, upload_cat(&[&l.wq])?, upload_cat(&[&l.wk])?, upload_cat(&[&l.wv])?) };
            let o = upload_cat(&[&l.wo])?;
            let (gu, gate, up) = if same_gu { (Some(upload_cat(&[&l.w_gate, &l.w_up])?), 0, 0) } else { (None, upload_cat(&[&l.w_gate])?, upload_cat(&[&l.w_up])?) };
            let down = upload_cat(&[&l.w_down])?;
            lms.push(LM { qkv, q, k, v, o, gu, gate, up, down });
        }
        let out_m = upload_cat(&[&m.output])?;
        let mut ops = Vec::new();
        let mv = |ops: &mut Vec<Op>, mi: usize, x_off: usize, y_off: usize, flags: u32, b_off: usize| {
            let mt = &mats[mi];
            let p = uni(&[mt.rows as u32, mt.cols as u32, mt.kind, x_off as u32, y_off as u32, flags, b_off as u32, 0]);
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes[0].get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: mt.bufs[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: mt.bufs[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: mt.bufs[2].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: res.act.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: cst_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: p.as_entire_binding() },
                ],
            });
            let wgs = (mt.rows as u32).div_ceil(8);
            ops.push(Op::Fixed { pipe: 0, bind, groups: (wgs.min(65535), wgs.div_ceil(65535)) });
        };
        let norm = |ops: &mut Vec<Op>, len: usize, x_off: usize, y_off: usize, w_off: usize, heads: usize| {
            let p = uni(&[len as u32, x_off as u32, y_off as u32, w_off as u32, c.eps.to_bits(), 0, 0, 0]);
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes[1].get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: res.act.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cst_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: p.as_entire_binding() },
                ],
            });
            ops.push(Op::Fixed { pipe: 1, bind, groups: (heads as u32, 1) });
        };
        let (nq, nk) = (c.n_head, c.n_head_kv);
        let half = c.rope_dim / 2;
        let mut kv_params = Vec::new();
        for (li, (lm, lo)) in lms.iter().zip(&loffs).enumerate() {
            norm(&mut ops, d, lay.x, lay.xb, lo.n1, 1);
            match lm.qkv {
                // q, k, v are adjacent in the activation buffer (sizes multiple of 4).
                Some(i) => mv(&mut ops, i, lay.xb, lay.q, 0, 0),
                None => {
                    mv(&mut ops, lm.q, lay.xb, lay.q, if lo.bq.is_some() { 2 } else { 0 }, lo.bq.unwrap_or(0));
                    mv(&mut ops, lm.k, lay.xb, lay.k, if lo.bk.is_some() { 2 } else { 0 }, lo.bk.unwrap_or(0));
                    mv(&mut ops, lm.v, lay.xb, lay.v, if lo.bv.is_some() { 2 } else { 0 }, lo.bv.unwrap_or(0));
                }
            }
            if let Some(w) = lo.qn {
                norm(&mut ops, hd, lay.q, lay.q, w, nq);
            }
            if let Some(w) = lo.kn {
                norm(&mut ops, hd, lay.k, lay.k, w, nk);
            }
            let rp = uni(&[lay.q as u32, lay.k as u32, nq as u32, nk as u32, hd as u32, half as u32, m.rope.neox as u32, 0]);
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipes[2].get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: res.act.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: res.step.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rp.as_entire_binding() },
                ],
            });
            ops.push(Op::Fixed { pipe: 2, bind, groups: ((((nq + nk) * half) as u32).div_ceil(64).max(1), 1) });
            ops.push(Op::KvStore(li));
            ops.push(Op::Attn(li));
            kv_params.push((uni(&[lay.k as u32, lay.v as u32, kvd as u32, 0, 0, 0, 0, 0]), uni(&[lay.q as u32, lay.att as u32, lay.scores as u32, (nq / nk) as u32, hd as u32, kvd as u32, max_ctx as u32, (1.0 / (hd as f32).sqrt()).to_bits()])));
            mv(&mut ops, lm.o, lay.att, lay.x, 4, 0);
            norm(&mut ops, d, lay.x, lay.xb, lo.n2, 1);
            match lm.gu {
                Some(i) => mv(&mut ops, i, lay.xb, lay.gate, 0, 0),
                None => {
                    mv(&mut ops, lm.gate, lay.xb, lay.gate, 0, 0);
                    mv(&mut ops, lm.up, lay.xb, lay.up, 0, 0);
                }
            }
            {
                let sp = uni(&[lay.gate as u32, lay.up as u32, ff as u32, 0, 0, 0, 0, 0]);
                let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipes[5].get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: res.act.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: sp.as_entire_binding() }],
                });
                ops.push(Op::Fixed { pipe: 5, bind, groups: ((ff as u32 / 4).div_ceil(256), 1) });
            }
            mv(&mut ops, lm.down, lay.gate, lay.x, 4, 0);
        }
        norm(&mut ops, d, lay.x, lay.xb, out_norm, 1);
        mv(&mut ops, out_m, lay.xb, lay.logits, 0, 0);
        Ok((pipes, ops, kv_params, lay, step_floats, res, mats, cst_buf, bytes))
    }

    /// Makes sure `st` has KV capacity for `n` positions (doubling growth,
    /// copying the existing cache), rebuilding its bind groups.
    fn ensure_kv(&self, st: &mut GpuState, n: usize) -> Result<()> {
        if n <= st.cap {
            return Ok(());
        }
        anyhow::ensure!(n <= self.max_ctx, "GPU context limit {} reached", self.max_ctx);
        let c = &self.cpu.cfg;
        let kvd = c.kv_dim();
        let new_cap = (st.cap * 2).max(1024).max(n.next_power_of_two()).min(self.max_ctx);
        let row = kvd * 4; // K and V as packed f16 pairs: kvd u32 per position
        anyhow::ensure!((new_cap * row) as u64 <= self.gpu.max_binding, "KV layer buffer exceeds the GPU binding limit");
        let dev = &self.gpu.device;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kvgrow") });
        let mut kv = Vec::new();
        for li in 0..c.n_layer {
            let b = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("kv"), size: (new_cap * row) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            if st.cap > 0 {
                enc.copy_buffer_to_buffer(&st.kv[li], 0, &b, 0, (st.cap * row) as u64);
            }
            kv.push(b);
        }
        self.gpu.queue.submit(Some(enc.finish()));
        let res = self.res.lock().unwrap();
        st.binds = kv
            .iter()
            .enumerate()
            .map(|(li, b)| {
                let (pk, pa) = &self.kv_params[li];
                let bk = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipes[3].get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: res.act.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: res.step.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: pk.as_entire_binding() },
                    ],
                });
                let ba = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipes[4].get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: res.act.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: res.step.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: pa.as_entire_binding() },
                    ],
                });
                (bk, ba)
            })
            .collect();
        st.kv = kv;
        st.cap = new_cap;
        st.bytes = new_cap * row * c.n_layer + st.logits.capacity() * 4;
        Ok(())
    }

    /// Runs `tokens` (≤ chunk) and fills `st.logits` (all rows or the last).
    fn run_chunk(&self, tokens: &[u32], pos0: usize, st: &mut GpuState, all: bool) -> Result<usize> {
        let c = &self.cpu.cfg;
        let (d, t) = (c.n_embd, tokens.len());
        self.ensure_kv(st, pos0 + t)?;
        let res = self.res.lock().unwrap();
        let q = &self.gpu.queue;
        let sf = self.step_floats;
        let mut steps = vec![0f32; sf * t];
        let mut xs = vec![0f32; d * t];
        for (i, &tk) in tokens.iter().enumerate() {
            steps[i * sf] = (pos0 + i) as f32;
            self.cpu.rope.table(pos0 + i, &mut steps[i * sf + 4..i * sf + 4 + c.rope_dim]);
            self.cpu.tok_embd.dequant_row(tk as usize % c.vocab, &mut xs[i * d..(i + 1) * d]);
        }
        q.write_buffer(&res.step_all, 0, bytemuck::cast_slice(&steps));
        q.write_buffer(&res.x_all, 0, bytemuck::cast_slice(&xs));
        let dev = &self.gpu.device;
        let t_enc = std::time::Instant::now();
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd") });
        let vb = (c.vocab * 4) as u64;
        for i in 0..t {
            enc.copy_buffer_to_buffer(&res.step_all, (i * sf * 4) as u64, &res.step, 0, (sf * 4) as u64);
            enc.copy_buffer_to_buffer(&res.x_all, (i * d * 4) as u64, &res.act, (self.lay.x * 4) as u64, (d * 4) as u64);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                for op in &self.ops {
                    match op {
                        Op::Fixed { pipe, bind, groups } => {
                            pass.set_pipeline(&self.pipes[*pipe]);
                            pass.set_bind_group(0, bind, &[]);
                            pass.dispatch_workgroups(groups.0, groups.1, 1);
                        }
                        Op::KvStore(li) => {
                            pass.set_pipeline(&self.pipes[3]);
                            pass.set_bind_group(0, &st.binds[*li].0, &[]);
                            pass.dispatch_workgroups(((c.kv_dim() / 2) as u32).div_ceil(64), 1, 1);
                        }
                        Op::Attn(li) => {
                            pass.set_pipeline(&self.pipes[4]);
                            pass.set_bind_group(0, &st.binds[*li].1, &[]);
                            pass.dispatch_workgroups(c.n_head as u32, 1, 1);
                        }
                    }
                }
            }
            if all {
                enc.copy_buffer_to_buffer(&res.act, (self.lay.logits * 4) as u64, &res.logits_all, i as u64 * vb, vb);
            }
        }
        let rows = if all { t } else { 1 };
        if all {
            enc.copy_buffer_to_buffer(&res.logits_all, 0, &res.stage, 0, vb * t as u64);
        } else {
            enc.copy_buffer_to_buffer(&res.act, (self.lay.logits * 4) as u64, &res.stage, 0, vb);
        }
        let cb = enc.finish();
        let enc_s = t_enc.elapsed();
        let t_gpu = std::time::Instant::now();
        q.submit(Some(cb));
        let bytes = vb * rows as u64;
        let slice = res.stage.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        wait_spin(dev, &rx)?;
        if std::env::var("AIDO_GPU_PROF").is_ok() {
            eprintln!("[gpu-prof] t={t} encode {:.2} ms, submit→ready {:.2} ms", enc_s.as_secs_f64() * 1e3, t_gpu.elapsed().as_secs_f64() * 1e3);
        }
        {
            let data = slice.get_mapped_range();
            st.logits.clear();
            st.logits.extend_from_slice(bytemuck::cast_slice(&data));
        }
        res.stage.unmap();
        Ok(rows)
    }
}

impl Drop for GpuLlama {
    /// Returns the weights' VRAM to the shared budget.
    fn drop(&mut self) {
        self.gpu.vram_used.fetch_sub(self.weight_bytes as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

impl LanguageModel for GpuLlama {
    fn describe(&self) -> String {
        format!("{} [GPU: whole model in VRAM, {:.2} GB, ctx ≤ {}]", self.cpu.describe(), self.weight_bytes as f64 / 1e9, self.max_ctx)
    }
    fn vocab(&self) -> usize {
        self.cpu.cfg.vocab
    }
    fn store(&self) -> &dyn crate::format::TensorStore {
        self.cpu.store.as_ref()
    }
    fn tokenizer(&self) -> Result<crate::tokenizer::Tokenizer> {
        crate::tokenizer::Tokenizer::from_gguf(self.cpu.store.as_ref())
    }
    fn weight_bytes_per_token(&self) -> usize {
        self.cpu.weight_bytes_per_token()
    }
    fn new_state(&self, n_ctx: usize, max_t: usize, _n_threads: usize) -> Box<dyn ModelState> {
        Box::new(GpuState { logits: Vec::new(), n_ctx: n_ctx.min(self.max_ctx), max_t: max_t.max(1), cap: 0, kv: Vec::new(), binds: Vec::new(), bytes: 0 })
    }
    fn forward(&self, _pool: &ThreadPool, tokens: &[u32], pos0: usize, st: &mut dyn ModelState, all_logits: bool) -> Result<usize> {
        let s = st.as_any_mut().downcast_mut::<GpuState>().context("state type mismatch")?;
        anyhow::ensure!(pos0 + tokens.len() <= s.n_ctx, "context full ({} tokens)", s.n_ctx);
        let step = if all_logits { 16 } else { self.chunk };
        let mut all = Vec::new();
        let mut rows = 0;
        for (ci, ch) in tokens.chunks(step).enumerate() {
            rows = self.run_chunk(ch, pos0 + ci * step, s, all_logits)?;
            if all_logits {
                all.extend_from_slice(&s.logits);
            }
        }
        if all_logits {
            s.logits = all;
            return Ok(tokens.len());
        }
        Ok(rows)
    }
}
