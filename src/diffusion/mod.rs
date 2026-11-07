//! Diffusion image generation: CLIP text encoder, UNet, TAESD decoder and
//! the Euler sampler, all running on the engine's own GEMM and thread pool.

pub mod clip;
pub mod nn;
pub mod pipeline;
pub mod taesd;
pub mod unet;
