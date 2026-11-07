//! Qwen3.5 / Qwen3.6 hybrid MoE (e.g. **Qwen3.6-35B-A3B**) running directly
//! from a colibri int4 container directory.
//!
//! Architecture (per block, 40 blocks, 3 of every 4 are DeltaNet):
//!
//! * **Gated DeltaNet** (linear attention, constant-size recurrent state):
//!   `qkv = W·x → causal depthwise conv1d(k=4) → SiLU`, per value head
//!   `S ← S·e^{g}; δ = (v − Sᵀk)·β; S ← S + k δᵀ; o = Sᵀq`, then a gated
//!   RMSNorm `norm(o)·w·SiLU(z)` and an output projection.
//! * **Gated full attention** (every 4th block): GQA with 16 query / 2 KV
//!   heads of 256, per-head zero-centred RMSNorm on q/k, partial RoPE on 64
//!   dims, output multiplied by `σ(gate)` (gate produced by `q_proj`).
//! * **MoE FFN**: router softmax over 256 experts, top-8 (weights not
//!   renormalised), SwiGLU experts of width 512 + a shared SwiGLU expert
//!   gated by `σ(w·x)`.
//!
//! ## Performance design
//! * Routed experts (~18 GB) are **never copied or repacked**: the int4
//!   bytes are used in place from the memory map; the activation is
//!   de-interleaved once so the integer kernel lines up with the nibble
//!   planes ([`crate::tensor::quant::dot_i4g64`]).
//! * Dense F16 matrices (3.8 GB/token if used as-is) are re-quantised at load
//!   to Q8_0 (default) or Q4_K, cutting per-token traffic by 2–3.5×.
//! * Tokens of a batch are **grouped by expert**, so during prompt processing
//!   each expert's weights are read once for all tokens routed to it.
//! * Expert usage is counted per layer, exposing the hot set for caching /
//!   pruning decisions.

use std::any::Any;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result};
use half::f16;

use super::{LanguageModel, ModelState};
use crate::format::safetensors::SafetensorsDir;
use crate::format::TensorStore;
use crate::runtime::pool::ThreadPool;
use crate::tensor::ops::{self, Rope};
use crate::tensor::quant::{deinterleave64, dot_i4g64, GgmlType, Q8Act};
use crate::tensor::quantize::quantize_row;
use crate::tensor::{Bytes, QWeight, SyncPtr};
use crate::gpu::{Gpu, GpuMatrix, MAX_T};
use std::sync::Arc;
use crate::tokenizer::Tokenizer;

/// Cumulative profiling counters in nanoseconds: deltanet, attention,
/// moe-dense (router+shared), moe-experts, lm-head.
pub static PROF: [std::sync::atomic::AtomicU64; 9] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn prof(i: usize, t: std::time::Instant) {
    PROF[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Returns and resets the profile as `(name, seconds)` pairs.
///
/// # Example
/// ```
/// let p = aido::llm::qwen35::take_profile();
/// assert_eq!(p.len(), 9);
/// ```
pub fn take_profile() -> Vec<(&'static str, f64)> {
    let names = ["deltanet", "attention", "moe-dense", "moe-experts", "lm-head", "dn.inproj", "dn.conv", "dn.recur", "dn.outproj"];
    names.iter().enumerate().map(|(i, n)| (*n, PROF[i].swap(0, Ordering::Relaxed) as f64 * 1e-9)).collect()
}

/// Hyper-parameters (from `qwen36_meta.json`).
#[derive(Debug, Clone)]
pub struct Q35Config {
    /// Hidden size.
    pub hidden: usize,
    /// Number of blocks.
    pub n_layers: usize,
    /// `true` = full attention block, `false` = DeltaNet.
    pub is_attn: Vec<bool>,
    /// Routed experts.
    pub n_experts: usize,
    /// Experts per token.
    pub topk: usize,
    /// Expert FFN width.
    pub moe_inter: usize,
    /// Shared expert width.
    pub shared_inter: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Renormalise top-k weights.
    pub norm_topk: bool,
    /// Attention query heads.
    pub q_heads: usize,
    /// Attention KV heads.
    pub kv_heads: usize,
    /// Attention head dim.
    pub head_dim: usize,
    /// Rotated dims per head.
    pub rotary: usize,
    /// RoPE θ.
    pub theta: f32,
    /// DeltaNet value heads.
    pub dn_vheads: usize,
    /// DeltaNet key heads.
    pub dn_kheads: usize,
    /// DeltaNet key dim.
    pub dn_kdim: usize,
    /// DeltaNet value dim.
    pub dn_vdim: usize,
    /// Conv kernel size.
    pub dn_convk: usize,
    /// Expert scale group size.
    pub gs: usize,
    /// Vocabulary.
    pub vocab: usize,
}

/// A dense projection: CPU weight plus an optional VRAM-resident copy.
pub struct Dense {
    /// CPU weight (always present: used for batches the GPU path does not take).
    pub w: QWeight,
    /// GPU copy, if it fitted into the VRAM budget.
    pub g: Option<GpuMatrix>,
}

impl std::ops::Deref for Dense {
    type Target = QWeight;
    fn deref(&self) -> &QWeight {
        &self.w
    }
}

/// DeltaNet block weights.
struct DnW {
    qkv: Dense,
    z: Dense,
    a: Dense,
    b: Dense,
    conv: Vec<f32>,
    a_neg: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    out: Dense,
}

/// Gated attention block weights.
struct AttnW {
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

enum Mixer {
    Dn(DnW),
    Attn(AttnW),
}

/// One routed expert: packed int4 slab `[gate | up | down]` and its scales,
/// living in one of three memory tiers:
/// VRAM (`g`), private RAM (owned `w`/`s` copies) or the file mapping.
struct Expert {
    w: Bytes,
    s: Bytes,
    g: Option<crate::gpu::GpuExpert>,
}

/// Where the experts ended up after tiering.
#[derive(Debug, Clone, Default)]
pub struct TierReport {
    /// Experts resident in VRAM.
    pub vram: usize,
    /// Experts copied into private RAM.
    pub ram: usize,
    /// Experts served from the memory-mapped files.
    pub mapped: usize,
    /// Share of historical activations served by VRAM experts.
    pub vram_hit: f64,
    /// Share served by VRAM + RAM experts.
    pub ram_hit: f64,
}

/// Reads a colibri `.coli_usage` profile: lines `layer expert count`.
fn read_usage(dir: &std::path::Path, layers: usize, experts: usize) -> Vec<Vec<u64>> {
    let mut u = vec![vec![0u64; experts]; layers];
    for name in [".aido_usage", ".coli_usage"] {
        if let Ok(t) = std::fs::read_to_string(dir.join(name)) {
            for line in t.lines() {
                let v: Vec<i64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if v.len() == 3 && v[0] >= 0 && (v[0] as usize) < layers && (v[1] as usize) < experts {
                    u[v[0] as usize][v[1] as usize] += v[2].max(0) as u64;
                }
            }
            break;
        }
    }
    u
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    router: QWeight,
    sh_gate: QWeight,
    sh_up: QWeight,
    sh_down: QWeight,
    sh_gate_w: Vec<f32>,
    experts: Vec<Expert>,
}

/// The model.
pub struct Qwen35 {
    /// Configuration.
    pub cfg: Q35Config,
    store: SafetensorsDir,
    embed: QWeight,
    head: Dense,
    gpu: Option<Arc<Gpu>>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    rope: Rope,
    skip: Vec<bool>,
    /// Expert hit counters `[layer][expert]`.
    pub usage: Vec<Vec<AtomicU32>>,
    dense_ty: GgmlType,
}

/// Zero-centred RMSNorm gain → plain gain (`1 + w`).
fn zc(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|w| 1.0 + w).collect()
}

/// Re-quantises a float matrix in parallel.
fn requant(pool: &ThreadPool, w: QWeight, ty: GgmlType) -> QWeight {
    if w.ty == ty || w.cols % ty.block_elems() != 0 {
        return w;
    }
    let rb = ty.row_bytes(w.cols);
    let mut out = vec![0u8; rb * w.rows];
    let op = SyncPtr::new(&mut out);
    let src = &w;
    pool.parallel_chunks(w.rows, 64, |r0, r1, _| {
        let mut tmp = vec![0f32; src.cols];
        for r in r0..r1 {
            src.dequant_row(r, &mut tmp);
            let q = quantize_row(ty, &tmp);
            unsafe { op.slice(r * rb, rb).copy_from_slice(&q) };
        }
    });
    QWeight::new(ty, w.rows, w.cols, Bytes::from_vec(out)).expect("size")
}

impl Qwen35 {
    /// Loads the container directory.
    ///
    /// * **Input** — directory with `qwen36_meta.json`, `config.json`,
    ///   `tokenizer.json` and `model-*.safetensors`; `dense`: type for the
    ///   dense matrices (`Q8_0` recommended, `Q4_K` fastest, `F16` exact);
    ///   pool used for the parallel conversion.
    /// * **Output** — model ready to run; routed experts stay memory-mapped.
    ///
    /// # Example
    /// ```no_run
    /// use aido::{llm::qwen35::Qwen35, tensor::quant::GgmlType, runtime::pool::ThreadPool};
    /// let m = Qwen35::load("D:/colibri-models/qwen36-35b", GgmlType::Q8_0, &ThreadPool::new(12, None), None, 0, 4 << 30).unwrap();
    /// ```
    pub fn load(dir: &str, dense: GgmlType, pool: &ThreadPool, gpu: Option<Arc<Gpu>>, vram_budget: usize, ram_cache: usize) -> Result<Self> {
        let st = SafetensorsDir::open(dir)?;
        let u = |k: &str| st.get_u64(&format!("qwen36_meta.{k}")).with_context(|| format!("qwen36_meta.{k}"));
        let hidden = u("hidden")? as usize;
        let n_layers = u("n_layers")? as usize;
        let types = st.meta().get("qwen36_meta.layer_types").and_then(|v| v.as_array().map(|a| a.to_vec())).context("layer_types")?;
        let is_attn: Vec<bool> = types.iter().map(|v| v.as_str() == Some("full_attention")).collect();
        let head_dim = u("head_dim")? as usize;
        let prf = st.get_f64("qwen36_meta.partial_rotary_factor").unwrap_or(0.25);
        let cfg = Q35Config {
            hidden,
            n_layers,
            is_attn,
            n_experts: u("num_experts")? as usize,
            topk: u("topk")? as usize,
            moe_inter: u("moe_inter")? as usize,
            shared_inter: u("shared_inter")? as usize,
            eps: st.get_f64("qwen36_meta.rms_eps").unwrap_or(1e-6) as f32,
            norm_topk: st.meta().get("qwen36_meta.norm_topk_prob").and_then(|v| v.as_u64()).unwrap_or(0) != 0,
            q_heads: u("q_heads")? as usize,
            kv_heads: u("kv_heads")? as usize,
            head_dim,
            rotary: (head_dim as f64 * prf + 0.5) as usize,
            theta: st.get_f64("qwen36_meta.rope_theta").unwrap_or(1e7) as f32,
            dn_vheads: u("dn_vheads")? as usize,
            dn_kheads: u("dn_kheads")? as usize,
            dn_kdim: u("dn_kdim")? as usize,
            dn_vdim: u("dn_vdim")? as usize,
            dn_convk: u("dn_convk")? as usize,
            gs: st.get_u64("qwen36_meta.expert_gs").unwrap_or(64) as usize,
            vocab: st.shape("lm_head.weight").context("lm_head")?[0],
        };
        anyhow::ensure!(cfg.gs == 64, "only expert_gs = 64 containers are supported (got {})", cfg.gs);
        // Dense projections: re-quantise for the CPU path, and upload to VRAM
        // while the budget lasts (layers in order → a contiguous GPU prefix
        // when the budget is smaller than the model: partial offload).
        let used = std::cell::Cell::new(0usize);
        let mk = |w: QWeight| -> Result<Dense> {
            let g = match &gpu {
                Some(gp) if used.get() + w.nbytes() * 9 / 8 <= vram_budget => {
                    let m = gp.upload(&w)?;
                    used.set(used.get() + m.nbytes);
                    Some(m)
                }
                _ => None,
            };
            // A VRAM-resident matrix keeps only a zero-byte CPU stub: RAM is
            // the scarcest tier on small machines (page cache for experts).
            let w = if g.is_some() { QWeight { ty: w.ty, rows: w.rows, cols: w.cols, data: Bytes::from_vec(Vec::new()) } } else { w };
            Ok(Dense { w, g })
        };
        let rq = |name: &str| -> Result<Dense> { mk(requant(pool, st.weight(name)?, dense)) };
        let raw_dense = |name: &str| -> Result<Dense> { mk(st.weight(name)?) };
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = |s: &str| format!("model.layers.{l}.{s}");
            let mixer = if cfg.is_attn[l] {
                Mixer::Attn(AttnW {
                    q: rq(&p("self_attn.q_proj.weight"))?,
                    k: rq(&p("self_attn.k_proj.weight"))?,
                    v: rq(&p("self_attn.v_proj.weight"))?,
                    o: rq(&p("self_attn.o_proj.weight"))?,
                    qn: zc(st.f32(&p("self_attn.q_norm.weight"))?),
                    kn: zc(st.f32(&p("self_attn.k_norm.weight"))?),
                })
            } else {
                Mixer::Dn(DnW {
                    qkv: rq(&p("linear_attn.in_proj_qkv.weight"))?,
                    z: rq(&p("linear_attn.in_proj_z.weight"))?,
                    a: raw_dense(&p("linear_attn.in_proj_a.weight"))?,
                    b: raw_dense(&p("linear_attn.in_proj_b.weight"))?,
                    conv: st.f32(&p("linear_attn.conv1d.weight"))?,
                    a_neg: st.f32(&p("linear_attn.A_log"))?.into_iter().map(|v| -v.exp()).collect(),
                    dt_bias: st.f32(&p("linear_attn.dt_bias"))?,
                    norm: st.f32(&p("linear_attn.norm.weight"))?,
                    out: rq(&p("linear_attn.out_proj.weight"))?,
                })
            };
            let experts = (0..cfg.n_experts)
                .map(|e| -> Result<Expert> {
                    Ok(Expert { w: st.raw(&p(&format!("mlp.experts.{e}.merged_weight")))?, s: st.raw(&p(&format!("mlp.experts.{e}.qs")))?, g: None })
                })
                .collect::<Result<Vec<_>>>()?;
            layers.push(Layer {
                in_norm: zc(st.f32(&p("input_layernorm.weight"))?),
                post_norm: zc(st.f32(&p("post_attention_layernorm.weight"))?),
                mixer,
                router: st.weight(&p("mlp.gate.weight"))?,
                sh_gate: requant(pool, st.weight(&p("mlp.shared_expert.gate_proj.weight"))?, dense),
                sh_up: requant(pool, st.weight(&p("mlp.shared_expert.up_proj.weight"))?, dense),
                sh_down: requant(pool, st.weight(&p("mlp.shared_expert.down_proj.weight"))?, dense),
                sh_gate_w: st.f32(&p("mlp.shared_expert_gate.weight"))?,
                experts,
            });
        }
        let mut head_pending = Some(());
        let _ = head_pending.take();
        let head = mk(requant(pool, st.weight("lm_head.weight")?, if dense == GgmlType::F16 { GgmlType::F16 } else { GgmlType::Q8_0 }))?;
        if gpu.is_some() {
            eprintln!("[gpu] {:.2} GB of dense weights resident in VRAM", used.get() as f64 / 1e9);
        }
        // ---- expert tiering: VRAM → private RAM → mapped file ----
        let usage_hist = read_usage(std::path::Path::new(dir), n_layers, cfg.n_experts);
        let mut rank: Vec<(u64, usize, usize)> = Vec::new();
        for (l, v) in usage_hist.iter().enumerate() {
            for (e, &c) in v.iter().enumerate() {
                rank.push((c, l, e));
            }
        }
        rank.sort_by(|a, b| b.0.cmp(&a.0));
        let total_hits: u64 = rank.iter().map(|r| r.0).sum::<u64>().max(1);
        let mut tiers = TierReport::default();
        let mut ram_left = ram_cache;
        let mut hits_v = 0u64;
        let mut hits_r = 0u64;
        for &(c, l, e) in &rank {
            let ex = &mut layers[l].experts[e];
            let sz = ex.w.len() + ex.s.len();
            let mut placed = false;
            if let Some(gp) = &gpu {
                if used.get() + sz + (64 << 20) <= vram_budget {
                    let ge = gp.upload_expert(ex.w.as_slice(), ex.s.as_slice(), cfg.moe_inter, cfg.hidden)?;
                    used.set(used.get() + ge.nbytes);
                    ex.g = Some(ge);
                    tiers.vram += 1;
                    hits_v += c;
                    hits_r += c;
                    placed = true;
                }
            }
            if !placed && ram_left >= sz {
                ex.w = Bytes::from_vec(ex.w.as_slice().to_vec());
                ex.s = Bytes::from_vec(ex.s.as_slice().to_vec());
                ram_left -= sz;
                tiers.ram += 1;
                hits_r += c;
                placed = true;
            }
            if !placed {
                tiers.mapped += 1;
            }
        }
        tiers.vram_hit = hits_v as f64 / total_hits as f64;
        tiers.ram_hit = hits_r as f64 / total_hits as f64;
        eprintln!(
            "[tiers] experts: VRAM {} | RAM {} | mapped {} — historical hit rate VRAM {:.1}%, VRAM+RAM {:.1}% (VRAM total {:.2} GB)",
            tiers.vram, tiers.ram, tiers.mapped, tiers.vram_hit * 100.0, tiers.ram_hit * 100.0, used.get() as f64 / 1e9
        );
        let embed = st.weight("model.embed_tokens.weight")?;
        let final_norm = zc(st.f32("model.norm.weight")?);
        let rope = Rope::new(cfg.rotary, cfg.theta, None, true);
        let usage = (0..n_layers).map(|_| (0..cfg.n_experts).map(|_| AtomicU32::new(0)).collect()).collect();
        Ok(Qwen35 { skip: vec![false; n_layers], cfg, store: st, embed, head, gpu, final_norm, layers, rope, usage, dense_ty: dense })
    }

    /// Fraction of expert activations served by the `n` most used experts of
    /// each layer (how skewed routing is — the basis for hot-expert caching).
    ///
    /// # Example
    /// ```no_run
    /// # let m: aido::llm::qwen35::Qwen35 = unimplemented!();
    /// println!("top-64 coverage {:.1}%", 100.0 * m.hot_coverage(64));
    /// ```
    pub fn hot_coverage(&self, n: usize) -> f64 {
        let (mut hit, mut tot) = (0u64, 0u64);
        for l in &self.usage {
            let mut v: Vec<u32> = l.iter().map(|a| a.load(Ordering::Relaxed)).collect();
            v.sort_unstable_by(|a, b| b.cmp(a));
            hit += v.iter().take(n).map(|&x| x as u64).sum::<u64>();
            tot += v.iter().map(|&x| x as u64).sum::<u64>();
        }
        hit as f64 / tot.max(1) as f64
    }
}

/// Per-sequence state.
pub struct Q35State {
    n_ctx: usize,
    max_t: usize,
    pos: usize,
    kv_k: Vec<Vec<f16>>,
    kv_v: Vec<Vec<f16>>,
    dn_s: Vec<Vec<f32>>,
    dn_conv: Vec<Vec<f32>>,
    logits: Vec<f32>,
    scores: Vec<Vec<f32>>,
}

impl ModelState for Q35State {
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
        self.kv_k.iter().chain(&self.kv_v).map(|v| v.len() * 2).sum::<usize>() + self.dn_s.iter().chain(&self.dn_conv).map(|v| v.len() * 4).sum::<usize>()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Fused parallel matmul of several weights sharing one input.
fn mm(pool: &ThreadPool, ws: &[(&QWeight, &mut [f32])], xs: &[f32], xq: &[Q8Act], t: usize) {
    let total: usize = ws.iter().map(|w| w.0.rows).sum();
    let threads = pool.width().max(1);
    let chunk = ((total / (threads * 6)).max(4) + 3) / 4 * 4;
    let mut tasks = Vec::new();
    for (i, (w, _)) in ws.iter().enumerate() {
        let mut r = 0;
        while r < w.rows {
            tasks.push((i, r, (r + chunk).min(w.rows)));
            r += chunk;
        }
    }
    let outs: Vec<(SyncPtr<f32>, usize)> = ws.iter().map(|(w, o)| (SyncPtr(o.as_ptr() as *mut f32), w.rows)).collect();
    let empty = Q8Act::default();
    pool.parallel_for(tasks.len(), |ti, _| {
        let (wi, r0, r1) = tasks[ti];
        let w = ws[wi].0;
        let f = crate::tensor::quant::dot_fn(w.ty);
        let need = crate::tensor::quant::uses_q8(w.ty);
        let (op, rows) = (&outs[wi].0, outs[wi].1);
        for tok in 0..t {
            let x = &xs[tok * w.cols..(tok + 1) * w.cols];
            let q = if need { &xq[tok] } else { &empty };
            for r in r0..r1 {
                unsafe { *op.ptr().add(tok * rows + r) = f(w.row(r), x, q) };
            }
        }
    });
}

impl Qwen35 {
    /// Dense matmul group: GPU when every matrix of the group is resident
    /// (one submission per ≤ 8 tokens), otherwise the CPU integer path.
    fn dmm(&self, pool: &ThreadPool, ops: &mut [(&Dense, &mut [f32])], x: &[f32], t: usize) {
        let cols = ops[0].0.cols;
        if let Some(gpu) = &self.gpu {
            if ops.iter().all(|o| o.0.g.is_some()) {
                let mats: Vec<&GpuMatrix> = ops.iter().map(|o| o.0.g.as_ref().unwrap()).collect();
                let mut ok = true;
                let mut c = 0;
                while c < t && ok {
                    let tc = (t - c).min(MAX_T);
                    let mut outs: Vec<&mut [f32]> = ops.iter_mut().map(|(d, o)| &mut o[c * d.rows..(c + tc) * d.rows]).collect();
                    ok = gpu.matmul_many(&x[c * cols..(c + tc) * cols], tc, &mats, &mut outs).is_ok();
                    c += tc;
                }
                if ok {
                    return;
                }
            }
        }
        assert!(ops.iter().all(|o| o.0.w.data.len() > 0), "GPU-resident matrix needs the GPU path (GPU call failed)");
        let mut xq = vec![Q8Act::default(); t];
        ops::quantize_rows(pool, x, cols, &mut xq);
        let v: Vec<(&QWeight, &mut [f32])> = ops.iter_mut().map(|(d, o)| (&d.w, &mut **o)).collect();
        mm(pool, &v, x, &xq, t);
    }

    /// Gated DeltaNet block for `t` tokens.
    fn deltanet(&self, pool: &ThreadPool, w: &DnW, l_dn: usize, h: &[f32], t: usize, st: &mut Q35State, out: &mut [f32]) {
        let c = &self.cfg;
        let d = c.hidden;
        let (vh, kh, kd, vd) = (c.dn_vheads, c.dn_kheads, c.dn_kdim, c.dn_vdim);
        let conv_dim = 2 * kh * kd + vh * vd;
        let vdim_tot = vh * vd;
        let mut qkv = vec![0f32; t * conv_dim];
        let mut z = vec![0f32; t * vdim_tot];
        let mut a = vec![0f32; t * vh];
        let mut b = vec![0f32; t * vh];
        let _ = d;
        let tm = std::time::Instant::now();
        self.dmm(pool, &mut [(&w.qkv, &mut qkv), (&w.z, &mut z), (&w.a, &mut a), (&w.b, &mut b)], h, t);
        prof(5, tm);
        let tm = std::time::Instant::now();
        // Causal depthwise conv + SiLU, parallel over channels (ring carried).
        let ck = c.dn_convk;
        let ring = SyncPtr::new(&mut st.dn_conv[l_dn]);
        let qp = SyncPtr::new(&mut qkv);
        pool.parallel_chunks(conv_dim, 256, |c0, c1, _| {
            for ch in c0..c1 {
                let wv = &w.conv[ch * ck..(ch + 1) * ck];
                let rg = unsafe { ring.slice(ch * (ck - 1), ck - 1) };
                for r in 0..t {
                    let xv = unsafe { *qp.ptr().add(r * conv_dim + ch) };
                    let mut acc = wv[ck - 1] * xv;
                    for k in 0..ck - 1 {
                        acc += wv[k] * rg[k];
                    }
                    for k in 0..ck - 2 {
                        rg[k] = rg[k + 1];
                    }
                    rg[ck - 2] = xv;
                    unsafe { *qp.ptr().add(r * conv_dim + ch) = acc / (1.0 + (-acc).exp()) };
                }
            }
        });
        prof(6, tm);
        let tm = std::time::Instant::now();
        // Recurrence, parallel over value heads; tokens sequential inside.
        let rep = vh / kh;
        let scale = 1.0 / (kd as f32).sqrt();
        let mut o = vec![0f32; t * vdim_tot];
        let op = SyncPtr::new(&mut o);
        let sp = SyncPtr::new(&mut st.dn_s[l_dn]);
        let eps = c.eps;
        let qkv = &qkv;
        pool.parallel_for(vh, |hh, _| {
            let s = unsafe { sp.slice(hh * kd * vd, kd * vd) };
            let ki = hh / rep;
            let mut q = vec![0f32; kd];
            let mut k = vec![0f32; kd];
            let mut kv = vec![0f32; vd];
            for r in 0..t {
                let row = &qkv[r * conv_dim..(r + 1) * conv_dim];
                q.copy_from_slice(&row[ki * kd..(ki + 1) * kd]);
                k.copy_from_slice(&row[kh * kd + ki * kd..kh * kd + (ki + 1) * kd]);
                let v = &row[2 * kh * kd + hh * vd..2 * kh * kd + (hh + 1) * vd];
                let nq = (q.iter().map(|x| (x * x) as f64).sum::<f64>() + 1e-6).sqrt() as f32;
                let nk = (k.iter().map(|x| (x * x) as f64).sum::<f64>() + 1e-6).sqrt() as f32;
                q.iter_mut().for_each(|x| *x = *x / nq * scale);
                k.iter_mut().for_each(|x| *x /= nk);
                let beta = 1.0 / (1.0 + (-b[r * vh + hh]).exp());
                let ar = a[r * vh + hh] + w.dt_bias[hh];
                let sp_ = if ar > 20.0 { ar } else { ar.exp().ln_1p() };
                let decay = (w.a_neg[hh] * sp_).exp();
                kv.fill(0.0);
                for kk in 0..kd {
                    let srow = &mut s[kk * vd..(kk + 1) * vd];
                    let kkv = k[kk];
                    for j in 0..vd {
                        srow[j] *= decay;
                        kv[j] += kkv * srow[j];
                    }
                }
                for j in 0..vd {
                    kv[j] = (v[j] - kv[j]) * beta; // delta
                }
                let ov = unsafe { op.slice(r * vdim_tot + hh * vd, vd) };
                ov.fill(0.0);
                for kk in 0..kd {
                    let srow = &mut s[kk * vd..(kk + 1) * vd];
                    let (kkv, qkv_) = (k[kk], q[kk]);
                    for j in 0..vd {
                        srow[j] += kkv * kv[j];
                        ov[j] += qkv_ * srow[j];
                    }
                }
                // Gated RMSNorm.
                let ms = ov.iter().map(|x| (x * x) as f64).sum::<f64>() / vd as f64;
                let rr = 1.0 / ((ms as f32) + eps).sqrt();
                let zr = &z[r * vdim_tot + hh * vd..r * vdim_tot + (hh + 1) * vd];
                for j in 0..vd {
                    let g = zr[j] / (1.0 + (-zr[j]).exp());
                    ov[j] = ov[j] * rr * w.norm[j] * g;
                }
            }
        });
        prof(7, tm);
        let tm = std::time::Instant::now();
        self.dmm(pool, &mut [(&w.out, out)], &o, t);
        prof(8, tm);
    }

    /// Gated attention block for `t` tokens at positions `pos0..`.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, pool: &ThreadPool, w: &AttnW, l_at: usize, h: &[f32], t: usize, pos0: usize, st: &mut Q35State, out: &mut [f32]) {
        let c = &self.cfg;
        let d = c.hidden;
        let (nh, nkv, hd) = (c.q_heads, c.kv_heads, c.head_dim);
        let qw = w.q.rows / nh; // query + gate per head
        let kvd = nkv * hd;
        let _ = d;
        let mut q = vec![0f32; t * w.q.rows];
        let mut k = vec![0f32; t * kvd];
        let mut v = vec![0f32; t * kvd];
        self.dmm(pool, &mut [(&w.q, &mut q), (&w.k, &mut k), (&w.v, &mut v)], h, t);
        let mut query = vec![0f32; t * nh * hd];
        let mut cs = vec![0f32; c.rotary];
        for r in 0..t {
            self.rope.table(pos0 + r, &mut cs);
            for hh in 0..nh {
                let qh = &mut query[(r * nh + hh) * hd..(r * nh + hh + 1) * hd];
                qh.copy_from_slice(&q[r * w.q.rows + hh * qw..r * w.q.rows + hh * qw + hd]);
                ops::rmsnorm_inplace(qh, &w.qn, c.eps);
                self.rope.apply(qh, hd, &cs);
            }
            for kh in 0..nkv {
                let kr = &mut k[r * kvd + kh * hd..r * kvd + (kh + 1) * hd];
                ops::rmsnorm_inplace(kr, &w.kn, c.eps);
                self.rope.apply(kr, hd, &cs);
            }
            let off = (pos0 + r) * kvd;
            ops::f32_to_f16(&k[r * kvd..(r + 1) * kvd], &mut st.kv_k[l_at][off..off + kvd]);
            ops::f32_to_f16(&v[r * kvd..(r + 1) * kvd], &mut st.kv_v[l_at][off..off + kvd]);
        }
        let g = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0f32; t * nh * hd];
        let cp = SyncPtr::new(&mut ctx);
        let sc = SyncPtr::new(&mut st.scores);
        let (kc, vc) = (&st.kv_k[l_at], &st.kv_v[l_at]);
        let query = &query;
        pool.parallel_for(t * nkv, |task, wk| {
            let (r, kh) = (task / nkv, task % nkv);
            let n_ctx = pos0 + r + 1;
            let scores = unsafe { &mut *sc.ptr().add(wk) };
            if scores.len() < g * n_ctx {
                scores.resize(g * n_ctx, 0.0);
            }
            let qs = &query[(r * nh + kh * g) * hd..(r * nh + (kh + 1) * g) * hd];
            let o = unsafe { cp.slice((r * nh + kh * g) * hd, g * hd) };
            ops::attend_group(qs, &kc[kh * hd..], &vc[kh * hd..], kvd, n_ctx, hd, scale, &mut scores[..g * n_ctx], o);
        });
        // Output gate.
        for r in 0..t {
            for hh in 0..nh {
                let gate = &q[r * w.q.rows + hh * qw + hd..r * w.q.rows + hh * qw + 2 * hd];
                let o = &mut ctx[(r * nh + hh) * hd..(r * nh + hh + 1) * hd];
                for j in 0..hd {
                    o[j] /= 1.0 + (-gate[j]).exp();
                }
            }
        }
        self.dmm(pool, &mut [(&w.o, out)], &ctx, t);
    }

    /// MoE feed-forward (routed + shared experts) for `t` tokens.
    fn moe(&self, pool: &ThreadPool, li: usize, l: &Layer, h: &[f32], t: usize, out: &mut [f32]) {
        let c = &self.cfg;
        let d = c.hidden;
        let e_n = c.n_experts;
        let inter = c.moe_inter;
        let tm = std::time::Instant::now();
        let mut xq = vec![Q8Act::default(); t];
        ops::quantize_rows(pool, h, d, &mut xq);
        // Router + shared expert projections in one region.
        let si = c.shared_inter;
        let mut logits = vec![0f32; t * e_n];
        let mut sg = vec![0f32; t * si];
        let mut su = vec![0f32; t * si];
        mm(pool, &[(&l.router, &mut logits), (&l.sh_gate, &mut sg), (&l.sh_up, &mut su)], h, &xq, t);
        // Routing.
        let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(t * c.topk); // (expert, token, weight)
        for r in 0..t {
            let lg = &mut logits[r * e_n..(r + 1) * e_n];
            ops::softmax_inplace(lg);
            let mut idx: Vec<usize> = (0..e_n).collect();
            idx.select_nth_unstable_by(c.topk - 1, |&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            let top = &idx[..c.topk];
            let s: f32 = if c.norm_topk { top.iter().map(|&e| lg[e]).sum() } else { 1.0 };
            for &e in top {
                pairs.push((e, r, lg[e] / s));
                self.usage[li][e].fetch_add(1, Ordering::Relaxed);
            }
        }
        pairs.sort_by_key(|p| p.0);
        prof(2, tm);
        let tm = std::time::Instant::now();
        // Experts resident in VRAM run asynchronously on the GPU while the
        // CPU handles the rest (decode path, one token).
        let mut gpu_pairs: Vec<(usize, usize, f32)> = Vec::new();
        if t == 1 {
            if let Some(gp) = &self.gpu {
                gpu_pairs = pairs.iter().filter(|p| l.experts[p.0].g.is_some()).cloned().collect();
                pairs.retain(|p| l.experts[p.0].g.is_none());
                let list: Vec<&crate::gpu::GpuExpert> = gpu_pairs.iter().map(|p| l.experts[p.0].g.as_ref().unwrap()).collect();
                if gp.begin_experts(h, &list).is_err() {
                    pairs.extend(gpu_pairs.drain(..));
                    pairs.sort_by_key(|p| p.0);
                }
            }
        }
        // De-interleaved activations for the int4 kernels.
        let mut hd_ = vec![0f32; t * d];
        for r in 0..t {
            deinterleave64(&h[r * d..(r + 1) * d], &mut hd_[r * d..(r + 1) * d]);
        }
        let mut hq = vec![Q8Act::default(); t];
        ops::quantize_rows(pool, &hd_, d, &mut hq);
        // Stage 1: gate/up rows for every (expert, token) pair, tasks = pair × row chunk.
        let np = pairs.len();
        let mut act = vec![0f32; np * 2 * inter];
        let ap = SyncPtr::new(&mut act);
        let rows_per = 64;
        let chunks = (2 * inter).div_ceil(rows_per);
        // group pairs by expert so one expert's rows are read once per chunk
        let mut groups: Vec<(usize, usize)> = Vec::new(); // (start, end) in pairs
        let mut s0 = 0;
        for i in 1..=np {
            if i == np || pairs[i].0 != pairs[s0].0 {
                groups.push((s0, i));
                s0 = i;
            }
        }
        let row_b = d / 2;
        let sc_b = d / 64 * 4;
        let pairs_r = &pairs;
        let hq_r = &hq;
        pool.parallel_for(groups.len() * chunks, |task, _| {
            let (gi, ch) = (task / chunks, task % chunks);
            let (g0, g1) = groups[gi];
            let ex = &l.experts[pairs_r[g0].0];
            let (wb, sb) = (ex.w.as_slice(), ex.s.as_slice());
            let r0 = ch * rows_per;
            let r1 = (r0 + rows_per).min(2 * inter);
            for r in r0..r1 {
                let row = &wb[r * row_b..(r + 1) * row_b];
                let scl = &sb[r * sc_b..(r + 1) * sc_b];
                for (pi, p) in pairs_r[g0..g1].iter().enumerate() {
                    let v = dot_i4g64(row, scl, &hq_r[p.1]);
                    unsafe { *ap.ptr().add((g0 + pi) * 2 * inter + r) = v };
                }
            }
        });
        // SwiGLU, de-interleave, quantise.
        let mut mid = vec![0f32; np * inter];
        let mut midq = vec![Q8Act::default(); np];
        {
            let mp = SyncPtr::new(&mut mid);
            let mq = SyncPtr::new(&mut midq);
            let act = &act;
            pool.parallel_for(np, |i, _| {
                let a = &act[i * 2 * inter..(i + 1) * 2 * inter];
                let mut tmp = vec![0f32; inter];
                for j in 0..inter {
                    let g = a[j];
                    tmp[j] = g / (1.0 + (-g).exp()) * a[inter + j];
                }
                let m = unsafe { mp.slice(i * inter, inter) };
                deinterleave64(&tmp, m);
                unsafe { (*mq.ptr().add(i)).quantize(m) };
            });
        }
        // Stage 2: down rows (d rows of `inter` columns) per pair.
        let mut y = vec![0f32; np * d];
        let yp = SyncPtr::new(&mut y);
        let down_off = 2 * inter * row_b;
        let down_soff = 2 * inter * sc_b;
        let drow_b = inter / 2;
        let dsc_b = inter / 64 * 4;
        let dchunks = d.div_ceil(rows_per * 2);
        let midq = &midq;
        pool.parallel_for(groups.len() * dchunks, |task, _| {
            let (gi, ch) = (task / dchunks, task % dchunks);
            let (g0, g1) = groups[gi];
            let ex = &l.experts[pairs_r[g0].0];
            let (wb, sb) = (ex.w.as_slice(), ex.s.as_slice());
            let r0 = ch * rows_per * 2;
            let r1 = (r0 + rows_per * 2).min(d);
            for r in r0..r1 {
                let row = &wb[down_off + r * drow_b..down_off + (r + 1) * drow_b];
                let scl = &sb[down_soff + r * dsc_b..down_soff + (r + 1) * dsc_b];
                for pi in g0..g1 {
                    let v = dot_i4g64(row, scl, &midq[pi]);
                    unsafe { *yp.ptr().add(pi * d + r) = v };
                }
            }
        });
        prof(3, tm);
        let tm = std::time::Instant::now();
        // Shared expert.
        for r in 0..t {
            let (g, u) = (&mut sg[r * si..(r + 1) * si], &su[r * si..(r + 1) * si]);
            ops::silu_mul(g, u);
        }
        let mut sq = vec![Q8Act::default(); t];
        ops::quantize_rows(pool, &sg, si, &mut sq);
        mm(pool, &[(&l.sh_down, out)], &sg, &sq, t);
        for r in 0..t {
            let hr = &h[r * d..(r + 1) * d];
            let gl: f32 = hr.iter().zip(&l.sh_gate_w).map(|(a, b)| a * b).sum();
            let gate = 1.0 / (1.0 + (-gl).exp());
            out[r * d..(r + 1) * d].iter_mut().for_each(|v| *v *= gate);
        }
        prof(2, tm);
        for (pi, p) in pairs.iter().enumerate() {
            let o = &mut out[p.1 * d..(p.1 + 1) * d];
            let yy = &y[pi * d..(pi + 1) * d];
            for j in 0..d {
                o[j] += p.2 * yy[j];
            }
        }
        if !gpu_pairs.is_empty() {
            let mut gy = vec![0f32; gpu_pairs.len() * d];
            if let Some(gp) = &self.gpu {
                gp.end_experts(&mut gy).expect("GPU experts");
            }
            for (pi, p) in gpu_pairs.iter().enumerate() {
                let o = &mut out[p.1 * d..(p.1 + 1) * d];
                for j in 0..d {
                    o[j] += p.2 * gy[pi * d + j];
                }
            }
        }
    }
}

impl LanguageModel for Qwen35 {
    fn describe(&self) -> String {
        let c = &self.cfg;
        format!(
            "qwen3.5-moe {}L d={} experts={}x{} top{} (+shared) attn {}x{} heads, deltanet {}v/{}k heads, dense {:?}, vocab {}",
            c.n_layers, c.hidden, c.n_experts, c.moe_inter, c.topk, c.q_heads, c.head_dim, c.dn_vheads, c.dn_kheads, self.dense_ty, c.vocab
        )
    }
    fn vocab(&self) -> usize {
        self.cfg.vocab
    }
    fn store(&self) -> &dyn TensorStore {
        &self.store
    }
    fn tokenizer(&self) -> Result<Tokenizer> {
        Tokenizer::from_hf_json(self.store.path.join("tokenizer.json"), None, Some("<|im_end|>"))
    }
    fn weight_bytes_per_token(&self) -> usize {
        let mut n = self.head.nbytes();
        for l in &self.layers {
            n += match &l.mixer {
                Mixer::Dn(w) => w.qkv.nbytes() + w.z.nbytes() + w.out.nbytes() + w.a.nbytes() + w.b.nbytes(),
                Mixer::Attn(w) => w.q.nbytes() + w.k.nbytes() + w.v.nbytes() + w.o.nbytes(),
            };
            n += l.router.nbytes() + l.sh_gate.nbytes() + l.sh_up.nbytes() + l.sh_down.nbytes();
            n += self.cfg.topk * (l.experts[0].w.len() + l.experts[0].s.len());
        }
        n
    }
    fn new_state(&self, n_ctx: usize, max_t: usize, n_threads: usize) -> Box<dyn ModelState> {
        let c = &self.cfg;
        let n_attn = c.is_attn.iter().filter(|a| **a).count();
        let n_dn = c.n_layers - n_attn;
        let conv_dim = 2 * c.dn_kheads * c.dn_kdim + c.dn_vheads * c.dn_vdim;
        let kvd = c.kv_heads * c.head_dim;
        Box::new(Q35State {
            n_ctx,
            max_t,
            pos: 0,
            kv_k: (0..n_attn).map(|_| vec![f16::ZERO; n_ctx * kvd]).collect(),
            kv_v: (0..n_attn).map(|_| vec![f16::ZERO; n_ctx * kvd]).collect(),
            dn_s: (0..n_dn).map(|_| vec![0.0; c.dn_vheads * c.dn_kdim * c.dn_vdim]).collect(),
            dn_conv: (0..n_dn).map(|_| vec![0.0; conv_dim * (c.dn_convk - 1)]).collect(),
            logits: Vec::new(),
            scores: (0..n_threads).map(|_| Vec::new()).collect(),
        })
    }
    fn forward(&self, pool: &ThreadPool, tokens: &[u32], pos0: usize, st: &mut dyn ModelState, all_logits: bool) -> Result<usize> {
        let st = st.as_any_mut().downcast_mut::<Q35State>().context("state type")?;
        let t = tokens.len();
        anyhow::ensure!(t >= 1 && t <= st.max_t, "batch size");
        anyhow::ensure!(pos0 + t <= st.n_ctx, "context full");
        if pos0 == 0 {
            st.dn_s.iter_mut().chain(st.dn_conv.iter_mut()).for_each(|v| v.fill(0.0));
            st.pos = 0;
        }
        anyhow::ensure!(pos0 == st.pos, "recurrent state is at {}, cannot jump to {}", st.pos, pos0);
        let d = self.cfg.hidden;
        let mut x = vec![0f32; t * d];
        for (i, &tk) in tokens.iter().enumerate() {
            self.embed.dequant_row(tk as usize % self.cfg.vocab, &mut x[i * d..(i + 1) * d]);
        }
        let mut hbuf = vec![0f32; t * d];
        let mut obuf = vec![0f32; t * d];
        let (mut i_dn, mut i_at) = (0, 0);
        for (li, l) in self.layers.iter().enumerate() {
            let is_attn = matches!(l.mixer, Mixer::Attn(_));
            if self.skip[li] {
                if is_attn { i_at += 1 } else { i_dn += 1 }
                continue;
            }
            for r in 0..t {
                ops::rmsnorm(&mut hbuf[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], &l.in_norm, self.cfg.eps);
            }
            let tm = std::time::Instant::now();
            match &l.mixer {
                Mixer::Dn(w) => {
                    self.deltanet(pool, w, i_dn, &hbuf, t, st, &mut obuf);
                    i_dn += 1;
                    prof(0, tm);
                }
                Mixer::Attn(w) => {
                    self.attention(pool, w, i_at, &hbuf, t, pos0, st, &mut obuf);
                    i_at += 1;
                    prof(1, tm);
                }
            }
            ops::add_inplace(&mut x, &obuf);
            for r in 0..t {
                ops::rmsnorm(&mut hbuf[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], &l.post_norm, self.cfg.eps);
            }
            self.moe(pool, li, l, &hbuf, t, &mut obuf);
            ops::add_inplace(&mut x, &obuf);
        }
        st.pos = pos0 + t;
        let n_out = if all_logits { t } else { 1 };
        let first = t - n_out;
        let mut xn = vec![0f32; n_out * d];
        for i in 0..n_out {
            ops::rmsnorm(&mut xn[i * d..(i + 1) * d], &x[(first + i) * d..(first + i + 1) * d], &self.final_norm, self.cfg.eps);
        }
        let tm = std::time::Instant::now();
        st.logits.resize(n_out * self.cfg.vocab, 0.0);
        self.dmm(pool, &mut [(&self.head, &mut st.logits)], &xn, n_out);
        prof(4, tm);
        Ok(n_out)
    }
    fn can_rewind(&self) -> bool {
        false
    }
    fn no_think_suffix(&self) -> &str {
        "<think>\n\n</think>\n\n"
    }
    fn set_skip(&mut self, layers: &[usize]) -> Result<()> {
        for &l in layers {
            anyhow::ensure!(l < self.cfg.n_layers, "layer {l} out of range");
            self.skip[l] = true;
        }
        Ok(())
    }
}
