//! # AIDO — Adaptive Inference Engine for local AI models
//!
//! A from-scratch Rust runtime for running local models on modest hardware:
//!
//! * **LLMs** from GGUF files (Llama 3.x, Mistral, Qwen 2.5/3 …, quantised
//!   `Q4_0…Q8_0`, `Q4_K/Q5_K/Q6_K`, `F16`, `BF16`) — [`llm`];
//! * **Diffusion image models** (Stable-Diffusion-family UNet + CLIP + VAE /
//!   TAESD from `safetensors`) — [`diffusion`];
//! * an **OpenAI-compatible HTTP API** (`/v1/chat/completions`,
//!   `/v1/completions`, `/v1/images/generations`, `/v1/models`) — [`server`].
//!
//! The design goal is *maximum tokens/images per second per watt on
//! ordinary CPUs and small GPUs*:
//!
//! | Technique | Module |
//! |-----------|--------|
//! | integer AVX2 kernels on quantised weights, activation quantised once | [`tensor::quant`] |
//! | spin-then-park pinned thread pool, fused matmul regions | [`runtime::pool`], [`llm::model`] |
//! | measured bandwidth roofline → minimal thread count, thermal guard | [`runtime::governor`] |
//! | adaptive speculative decoding (n-gram + draft model), exact sampling | [`llm::engine`] |
//! | prefix KV-cache reuse across requests, idle cache eviction | [`llm::engine`] |
//! | zero-copy mmap weights + parallel pre-faulting | [`gguf`] |
//! | BLIS-style packed f32 GEMM for convolutions | [`tensor::gemm`] |
//! | GPU (Vulkan/DX12/Metal via wgpu) compute for diffusion | [`gpu`] |

#![recursion_limit = "512"]

pub mod diffusion;
pub mod format;
pub mod gguf;
pub mod gpu;
pub mod hub;
pub mod llm;
pub mod memory;
pub mod models;
pub mod optimize;
pub mod runtime;
pub mod sdk;
#[cfg(feature = "server")]
pub mod server;
pub mod steer;
pub mod tensor;
pub mod tokenizer;
pub mod upscale;
pub mod vector;
pub mod weights;
pub mod evals;

pub use models::{Kind, Models, ModelsConfig};
