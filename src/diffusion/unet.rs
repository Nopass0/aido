//! Stable Diffusion UNet (`UNet2DConditionModel`, SD 1.x / 2.x layout).

use anyhow::Result;

use super::nn::{attention, concat, upsample2x, Conv, GroupNorm, LayerNorm, Linear, T};
use crate::format::safetensors::SafetensorsDir;
use crate::runtime::pool::ThreadPool;

/// ResNet block with timestep conditioning.
struct Resnet {
    n1: GroupNorm,
    c1: Conv,
    temb: Linear,
    n2: GroupNorm,
    c2: Conv,
    short: Option<Conv>,
}

impl Resnet {
    fn load(st: &SafetensorsDir, p: &str, eps: f32) -> Result<Self> {
        Ok(Resnet {
            n1: GroupNorm::load(st, &format!("{p}.norm1"), 32, eps)?,
            c1: Conv::load(st, &format!("{p}.conv1"), 1)?,
            temb: Linear::load(st, &format!("{p}.time_emb_proj"))?,
            n2: GroupNorm::load(st, &format!("{p}.norm2"), 32, eps)?,
            c2: Conv::load(st, &format!("{p}.conv2"), 1)?,
            short: if st.shape(&format!("{p}.conv_shortcut.weight")).is_some() { Some(Conv::load(st, &format!("{p}.conv_shortcut"), 1)?) } else { None },
        })
    }

    fn forward(&self, pool: &ThreadPool, x: &T, temb_act: &[f32]) -> T {
        let mut h = x.clone();
        self.n1.apply(&mut h, true);
        let mut h = self.c1.forward(pool, &h);
        let tb = self.temb.forward(pool, temb_act, 1);
        for p in 0..h.pixels() {
            for c in 0..h.c {
                h.d[p * h.c + c] += tb[c];
            }
        }
        self.n2.apply(&mut h, true);
        let mut h = self.c2.forward(pool, &h);
        let skip = match &self.short {
            Some(s) => s.forward(pool, x),
            None => x.clone(),
        };
        crate::tensor::ops::add_inplace(&mut h.d, &skip.d);
        h
    }
}

/// Spatial transformer (self-attention, cross-attention, GEGLU FFN).
struct Xformer {
    norm: GroupNorm,
    proj_in: Linear,
    ln1: LayerNorm,
    q1: Linear,
    k1: Linear,
    v1: Linear,
    o1: Linear,
    ln2: LayerNorm,
    q2: Linear,
    k2: Linear,
    v2: Linear,
    o2: Linear,
    ln3: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    proj_out: Linear,
    heads: usize,
}

impl Xformer {
    fn load(st: &SafetensorsDir, p: &str, heads: usize) -> Result<Self> {
        let b = format!("{p}.transformer_blocks.0");
        let l = |s: &str| Linear::load(st, &format!("{b}.{s}"));
        Ok(Xformer {
            norm: GroupNorm::load(st, &format!("{p}.norm"), 32, 1e-6)?,
            proj_in: Linear::load(st, &format!("{p}.proj_in"))?,
            ln1: LayerNorm::load(st, &format!("{b}.norm1"), 1e-5)?,
            q1: l("attn1.to_q")?,
            k1: l("attn1.to_k")?,
            v1: l("attn1.to_v")?,
            o1: l("attn1.to_out.0")?,
            ln2: LayerNorm::load(st, &format!("{b}.norm2"), 1e-5)?,
            q2: l("attn2.to_q")?,
            k2: l("attn2.to_k")?,
            v2: l("attn2.to_v")?,
            o2: l("attn2.to_out.0")?,
            ln3: LayerNorm::load(st, &format!("{b}.norm3"), 1e-5)?,
            ff1: l("ff.net.0.proj")?,
            ff2: l("ff.net.2")?,
            proj_out: Linear::load(st, &format!("{p}.proj_out"))?,
            heads,
        })
    }

    fn forward(&self, pool: &ThreadPool, x: &T, ctx: &[f32], ctx_len: usize) -> T {
        let n = x.pixels();
        let mut h = x.clone();
        self.norm.apply(&mut h, false);
        let mut s = self.proj_in.forward(pool, &h.d, n);
        let c = self.proj_in.w.n;
        // self-attention
        let a = self.ln1.forward(&s);
        let (q, k, v) = (self.q1.forward(pool, &a, n), self.k1.forward(pool, &a, n), self.v1.forward(pool, &a, n));
        let o = attention(pool, &q, &k, &v, n, n, c, self.heads);
        crate::tensor::ops::add_inplace(&mut s, &self.o1.forward(pool, &o, n));
        // cross-attention
        let a = self.ln2.forward(&s);
        let q = self.q2.forward(pool, &a, n);
        let (k, v) = (self.k2.forward(pool, ctx, ctx_len), self.v2.forward(pool, ctx, ctx_len));
        let o = attention(pool, &q, &k, &v, n, ctx_len, c, self.heads);
        crate::tensor::ops::add_inplace(&mut s, &self.o2.forward(pool, &o, n));
        // GEGLU feed-forward
        let a = self.ln3.forward(&s);
        let g = self.ff1.forward(pool, &a, n);
        let inner = g.len() / n / 2;
        let mut f = vec![0f32; n * inner];
        for i in 0..n {
            let (val, gate) = (&g[i * 2 * inner..i * 2 * inner + inner], &g[i * 2 * inner + inner..(i + 1) * 2 * inner]);
            let mut gg = gate.to_vec();
            super::nn::gelu_erf(&mut gg);
            for j in 0..inner {
                f[i * inner + j] = val[j] * gg[j];
            }
        }
        crate::tensor::ops::add_inplace(&mut s, &self.ff2.forward(pool, &f, n));
        let mut out = T { h: x.h, w: x.w, c: x.c, d: self.proj_out.forward(pool, &s, n) };
        crate::tensor::ops::add_inplace(&mut out.d, &x.d);
        out
    }
}

struct Level {
    resnets: Vec<Resnet>,
    attns: Vec<Xformer>,
    sampler: Option<Conv>,
}

/// The UNet.
pub struct UNet {
    conv_in: Conv,
    t1: Linear,
    t2: Linear,
    down: Vec<Level>,
    mid_r1: Resnet,
    mid_a: Xformer,
    mid_r2: Resnet,
    up: Vec<Level>,
    norm_out: GroupNorm,
    conv_out: Conv,
    t_dim: usize,
}

impl UNet {
    /// Loads `unet/` (config.json + safetensors) converting weights to
    /// packed f32.
    ///
    /// # Example
    /// ```no_run
    /// let u = aido::diffusion::unet::UNet::load("models/sd-turbo/unet").unwrap();
    /// ```
    pub fn load(dir: &str) -> Result<Self> {
        let st = SafetensorsDir::open(dir)?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
        let ch: Vec<usize> = cfg["block_out_channels"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let head_cfg: Vec<usize> = match &cfg["attention_head_dim"] {
            serde_json::Value::Array(a) => a.iter().map(|v| v.as_u64().unwrap() as usize).collect(),
            v => vec![v.as_u64().unwrap_or(8) as usize; ch.len()],
        };
        let eps = cfg["norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let down_types: Vec<String> = cfg["down_block_types"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let up_types: Vec<String> = cfg["up_block_types"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        // SD1 configs give heads directly (8); SD2 gives head counts per level.
        let heads = |lvl: usize| if head_cfg.len() == ch.len() && head_cfg[lvl] < 64 { head_cfg[lvl] } else { ch[lvl] / 64 };
        let mut down = Vec::new();
        for (i, t) in down_types.iter().enumerate() {
            let p = format!("down_blocks.{i}");
            let cross = t.contains("CrossAttn");
            down.push(Level {
                resnets: (0..2).map(|j| Resnet::load(&st, &format!("{p}.resnets.{j}"), eps)).collect::<Result<_>>()?,
                attns: if cross { (0..2).map(|j| Xformer::load(&st, &format!("{p}.attentions.{j}"), heads(i))).collect::<Result<_>>()? } else { vec![] },
                sampler: if st.shape(&format!("{p}.downsamplers.0.conv.weight")).is_some() { Some(Conv::load(&st, &format!("{p}.downsamplers.0.conv"), 2)?) } else { None },
            });
        }
        let nl = ch.len();
        let mut up = Vec::new();
        for (i, t) in up_types.iter().enumerate() {
            let p = format!("up_blocks.{i}");
            let cross = t.contains("CrossAttn");
            up.push(Level {
                resnets: (0..3).map(|j| Resnet::load(&st, &format!("{p}.resnets.{j}"), eps)).collect::<Result<_>>()?,
                attns: if cross { (0..3).map(|j| Xformer::load(&st, &format!("{p}.attentions.{j}"), heads(nl - 1 - i))).collect::<Result<_>>()? } else { vec![] },
                sampler: if st.shape(&format!("{p}.upsamplers.0.conv.weight")).is_some() { Some(Conv::load(&st, &format!("{p}.upsamplers.0.conv"), 1)?) } else { None },
            });
        }
        Ok(UNet {
            conv_in: Conv::load(&st, "conv_in", 1)?,
            t1: Linear::load(&st, "time_embedding.linear_1")?,
            t2: Linear::load(&st, "time_embedding.linear_2")?,
            down,
            mid_r1: Resnet::load(&st, "mid_block.resnets.0", eps)?,
            mid_a: Xformer::load(&st, "mid_block.attentions.0", heads(nl - 1))?,
            mid_r2: Resnet::load(&st, "mid_block.resnets.1", eps)?,
            up,
            norm_out: GroupNorm::load(&st, "conv_norm_out", 32, eps)?,
            conv_out: Conv::load(&st, "conv_out", 1)?,
            t_dim: ch[0],
        })
    }

    /// Sinusoidal timestep embedding (`flip_sin_to_cos`, shift 0).
    fn timestep(&self, t: f32) -> Vec<f32> {
        let half = self.t_dim / 2;
        let mut e = vec![0f32; self.t_dim];
        for i in 0..half {
            let f = (-(10000f32).ln() * i as f32 / half as f32).exp();
            e[i] = (t * f).cos();
            e[half + i] = (t * f).sin();
        }
        e
    }

    /// Predicts noise for latent `x` at timestep `t` given text context.
    ///
    /// * **Input** — latent (HWC, 4 channels), timestep, context
    ///   (`ctx_len × 1024` for SD2).
    /// * **Output** — ε prediction with the shape of `x`.
    ///
    /// # Example
    /// ```no_run
    /// # let u = aido::diffusion::unet::UNet::load("models/sd-turbo/unet").unwrap();
    /// # let pool = aido::runtime::pool::ThreadPool::new(12, None);
    /// let eps = u.forward(&pool, &aido::diffusion::nn::T::zeros(64, 64, 4), 999.0, &vec![0.0; 77 * 1024], 77);
    /// ```
    pub fn forward(&self, pool: &ThreadPool, x: &T, t: f32, ctx: &[f32], ctx_len: usize) -> T {
        let mut temb = self.t1.forward(pool, &self.timestep(t), 1);
        crate::tensor::ops::silu_inplace(&mut temb);
        let mut temb = self.t2.forward(pool, &temb, 1);
        crate::tensor::ops::silu_inplace(&mut temb); // resnets apply SiLU before time_emb_proj
        let mut h = self.conv_in.forward(pool, x);
        let mut skips = vec![h.clone()];
        for lvl in &self.down {
            for (j, r) in lvl.resnets.iter().enumerate() {
                h = r.forward(pool, &h, &temb);
                if let Some(a) = lvl.attns.get(j) {
                    h = a.forward(pool, &h, ctx, ctx_len);
                }
                skips.push(h.clone());
            }
            if let Some(s) = &lvl.sampler {
                h = s.forward(pool, &h);
                skips.push(h.clone());
            }
        }
        h = self.mid_r1.forward(pool, &h, &temb);
        h = self.mid_a.forward(pool, &h, ctx, ctx_len);
        h = self.mid_r2.forward(pool, &h, &temb);
        for lvl in &self.up {
            for (j, r) in lvl.resnets.iter().enumerate() {
                let s = skips.pop().expect("skip");
                h = r.forward(pool, &concat(&h, &s), &temb);
                if let Some(a) = lvl.attns.get(j) {
                    h = a.forward(pool, &h, ctx, ctx_len);
                }
            }
            if let Some(s) = &lvl.sampler {
                h = s.forward(pool, &upsample2x(&h));
            }
        }
        self.norm_out.apply(&mut h, true);
        self.conv_out.forward(pool, &h)
    }
}
