//! TAESD — Tiny AutoEncoder for Stable Diffusion (decoder only).
//!
//! 1.2 M parameters instead of the 50 M-parameter KL VAE decoder: decoding a
//! 512×512 image costs ~40× fewer FLOPs, which on a CPU turns several seconds
//! into a fraction of a second, at a small loss of fine detail.

use anyhow::Result;

use super::nn::{upsample2x, Conv, T};
use crate::format::safetensors::SafetensorsDir;
use crate::runtime::pool::ThreadPool;

enum Layer {
    Conv(Conv),
    Relu,
    Block(Conv, Conv, Conv),
    Up,
}

/// TAESD decoder.
pub struct TaesdDecoder {
    layers: Vec<Layer>,
}

impl TaesdDecoder {
    /// Loads `diffusion_pytorch_model.safetensors` from a TAESD directory
    /// (`decoder.layers.N` layout).
    ///
    /// # Example
    /// ```no_run
    /// let d = aido::diffusion::taesd::TaesdDecoder::load("models/taesd").unwrap();
    /// ```
    pub fn load(dir: &str) -> Result<Self> {
        let st = SafetensorsDir::open(dir)?;
        let mut layers = Vec::new();
        for i in 0..64 {
            let p = format!("decoder.layers.{i}");
            if st.shape(&format!("{p}.weight")).is_some() {
                layers.push(Layer::Conv(Conv::load(&st, &p, 1)?));
            } else if st.shape(&format!("{p}.conv.0.weight")).is_some() {
                layers.push(Layer::Block(Conv::load(&st, &format!("{p}.conv.0"), 1)?, Conv::load(&st, &format!("{p}.conv.2"), 1)?, Conv::load(&st, &format!("{p}.conv.4"), 1)?));
            } else if i == 1 {
                layers.push(Layer::Relu);
            } else if layers.len() > 1 && i < 19 {
                layers.push(Layer::Up);
            }
        }
        Ok(TaesdDecoder { layers })
    }

    /// Decodes UNet latents (HWC, 4 ch) to RGB in `[0, 1]`.
    ///
    /// # Example
    /// ```no_run
    /// # let d = aido::diffusion::taesd::TaesdDecoder::load("models/taesd").unwrap();
    /// # let pool = aido::runtime::pool::ThreadPool::new(8, None);
    /// let rgb = d.decode(&pool, &aido::diffusion::nn::T::zeros(64, 64, 4));
    /// assert_eq!(rgb.c, 3);
    /// ```
    pub fn decode(&self, pool: &ThreadPool, z: &T) -> T {
        let relu = |t: &mut T| t.d.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut x = z.clone();
        x.d.iter_mut().for_each(|v| *v = (*v / 3.0).tanh() * 3.0);
        for l in &self.layers {
            x = match l {
                Layer::Conv(c) => c.forward(pool, &x),
                Layer::Relu => {
                    relu(&mut x);
                    x
                }
                Layer::Up => upsample2x(&x),
                Layer::Block(a, b, c) => {
                    let mut h = a.forward(pool, &x);
                    relu(&mut h);
                    let mut h = b.forward(pool, &h);
                    relu(&mut h);
                    let mut h = c.forward(pool, &h);
                    crate::tensor::ops::add_inplace(&mut h.d, &x.d);
                    relu(&mut h);
                    h
                }
            };
        }
        x.d.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        x
    }
}
