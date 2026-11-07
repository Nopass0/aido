//! `aido` command-line interface.
//!
//! ```text
//! aido info                                   # hardware + roofline
//! aido run   -m model.gguf -p "prompt"        # one completion
//! aido chat  -m model.gguf                    # interactive chat
//! aido bench -m model.gguf                    # speed benchmark
//! aido image --sd models/sd-turbo -p "..."    # text-to-image
//! aido serve -m model.gguf --sd models/sd-turbo --port 8080
//! ```

use std::io::{BufRead, Write};
use std::time::Instant;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

mod train_cmds;

use aido::llm::chat::Message;
use aido::llm::engine::{EngineOptions, GenParams, LlmEngine, SpecMode};
use aido::llm::sampler::SamplerConfig;
use aido::runtime::{pool::ThreadPool, sys};

/// Top-level CLI.
#[derive(Parser)]
#[command(name = "aido", version, about = "AIDO — adaptive inference engine for local AI models")]
struct Cli {
    /// Optimizer for every training command: `muon` (Muon + AdamW hybrid,
    /// default) or `adamw`.
    #[arg(long, global = true, default_value = "muon")]
    optimizer: String,
    #[command(subcommand)]
    cmd: Cmd,
}

/// Shared LLM options.
#[derive(Args, Clone)]
struct LlmArgs {
    /// GGUF model path.
    #[arg(short, long)]
    model: String,
    /// Worker threads (default: physical cores).
    #[arg(short, long)]
    threads: Option<usize>,
    /// Context length.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Speculative decoding: off | lookup | draft | auto.
    #[arg(long, default_value = "auto")]
    spec: String,
    /// Draft model (same tokenizer) for speculative decoding.
    #[arg(long)]
    draft: Option<String>,
    /// Maximum draft length.
    #[arg(long, default_value_t = 8)]
    max_draft: usize,
    /// Skip thread calibration.
    #[arg(long)]
    no_calibrate: bool,
    /// Do not pin threads.
    #[arg(long)]
    no_pin: bool,
    /// Disable the GPU.
    #[arg(long)]
    no_gpu: bool,
    /// VRAM budget in GB for offloaded weights.
    #[arg(long, default_value_t = 5.0)]
    vram_gb: f32,
    /// RAM budget in GB for private copies of hot MoE experts.
    #[arg(long, default_value_t = 4.0)]
    ram_cache_gb: f32,
    /// Re-quantise dense float weights at load: q8_0 | q4_k | q4_0 | f16.
    #[arg(long)]
    dense: Option<String>,
    /// Deactivate these transformer blocks at run time, e.g. `20,21`.
    #[arg(long, value_delimiter = ',')]
    skip_layers: Vec<usize>,
    /// KV cache format: q8 (default, 1.9x smaller than f16) | q4 (3.6x) | f16.
    #[arg(long, default_value = "q8")]
    kv: String,
    /// Long-context attention budget: positions attended per query beyond
    /// which block-sparse attention is used (0 = always exact).
    #[arg(long, default_value_t = 8192)]
    kv_budget: usize,
    /// Always-attended most recent positions in sparse mode.
    #[arg(long, default_value_t = 2048)]
    kv_recent: usize,
}

impl LlmArgs {
    /// Converts CLI flags to engine options.
    fn options(&self) -> Result<EngineOptions> {
        Ok(EngineOptions {
            threads: self.threads,
            ctx: self.ctx,
            spec: self.spec.parse()?,
            max_draft: self.max_draft,
            draft_path: self.draft.clone(),
            calibrate: !self.no_calibrate,
            pin: !self.no_pin,
            skip_layers: self.skip_layers.clone(),
            dense: self.dense.as_deref().map(aido::tensor::quantize::parse_type).transpose()?,
            gpu: !self.no_gpu,
            vram_gb: self.vram_gb,
            ram_cache_gb: self.ram_cache_gb,
            kv: aido::llm::kv::KvConfig { ty: self.kv.parse()?, budget: self.kv_budget, recent: self.kv_recent, ..Default::default() },
            ..Default::default()
        })
    }
}

/// Sub-commands.
#[derive(Subcommand)]
enum Cmd {
    /// Print hardware information and measured memory bandwidth.
    Info,
    /// Generate a completion for a prompt.
    Run {
        #[command(flatten)]
        llm: LlmArgs,
        /// Prompt text (`@file.txt` reads the prompt from a file).
        #[arg(short, long)]
        prompt: String,
        /// Max new tokens.
        #[arg(short = 'n', long, default_value_t = 256)]
        max_tokens: usize,
        /// Temperature (0 = greedy).
        #[arg(long, default_value_t = 0.0)]
        temp: f32,
        /// Treat the prompt as a raw completion prompt (no chat template).
        #[arg(long)]
        raw: bool,
        /// Enable Qwen3 "thinking".
        #[arg(long)]
        think: bool,
    },
    /// Interactive chat.
    Chat {
        #[command(flatten)]
        llm: LlmArgs,
        /// Temperature.
        #[arg(long, default_value_t = 0.7)]
        temp: f32,
    },
    /// Benchmark prompt processing and generation speed.
    Bench {
        #[command(flatten)]
        llm: LlmArgs,
        /// Prompt length for the prefill test.
        #[arg(long, default_value_t = 256)]
        pp: usize,
        /// Tokens to generate.
        #[arg(long, default_value_t = 128)]
        tg: usize,
    },
    /// Collect activation / layer / token statistics on a corpus.
    Analyze {
        /// Model (GGUF or .aido).
        #[arg(short, long)]
        model: String,
        /// Corpus text file(s).
        #[arg(long, num_args = 1..)]
        corpus: Vec<String>,
        /// Output JSON.
        #[arg(short, long, default_value = "stats.json")]
        out: String,
        /// Max tokens to analyse.
        #[arg(long, default_value_t = 4096)]
        tokens: usize,
        /// Threads.
        #[arg(short, long)]
        threads: Option<usize>,
    },
    /// Compile / optimise a model into the .aido format.
    Compile {
        /// Source model (GGUF or .aido).
        #[arg(short, long)]
        model: String,
        /// Output .aido path.
        #[arg(short, long)]
        out: String,
        /// Re-quantise matrices: q8_0 | q4_0 | q4_k | q6_k | f16.
        #[arg(long)]
        quant: Option<String>,
        /// Delete blocks: list (`3,7`) or `auto:N` (lowest block influence).
        #[arg(long)]
        drop_layers: Option<String>,
        /// Keep this fraction of FFN neurons (needs --stats).
        #[arg(long)]
        ffn_keep: Option<f32>,
        /// Statistics from `aido analyze`.
        #[arg(long)]
        stats: Option<String>,
        /// Trim the vocabulary to tokens used in these files.
        #[arg(long, num_args = 1..)]
        vocab_corpus: Vec<String>,
    },
    /// Quality: bits-per-byte of a text (lower = better).
    Eval {
        #[command(flatten)]
        llm: LlmArgs,
        /// Text file.
        #[arg(long)]
        text: String,
        /// Max bytes of text to evaluate.
        #[arg(long, default_value_t = 20000)]
        max_bytes: usize,
    },
    /// Pack text files into an .aidata token dataset.
    DataPack {
        /// Text files.
        #[arg(num_args = 1..)]
        files: Vec<String>,
        /// Output .aidata.
        #[arg(short, long)]
        out: String,
        /// Tokenize with this model's tokenizer (default: raw bytes).
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// Train a byte-level Llama-style LM from scratch (exports a runnable .aido).
    TrainLm {
        /// Dataset (.aidata from data-pack).
        #[arg(long)]
        data: String,
        /// Output .aido.
        #[arg(short, long)]
        out: String,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        dim: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 128)]
        ctx: usize,
        #[arg(long, default_value_t = 16)]
        batch: usize,
        #[arg(long, default_value_t = 2e-3)]
        lr: f32,
        /// Resume from a checkpoint (.ckpt.aido).
        #[arg(long)]
        resume: Option<String>,
        #[arg(short, long)]
        threads: Option<usize>,
    },
    /// Fine-tune an existing model (LoRA or full) and export a merged .aido.
    Finetune {
        #[arg(short, long)]
        model: String,
        #[arg(long)]
        data: String,
        #[arg(short, long)]
        out: String,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// LoRA rank (0 = full fine-tuning).
        #[arg(long, default_value_t = 8)]
        lora: usize,
        #[arg(long, default_value_t = 128)]
        ctx: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        #[arg(short, long)]
        threads: Option<usize>,
    },
    /// Train the demo text classifier (English / Russian / Rust code).
    TrainClassifier {
        #[arg(short, long, default_value = "models/textcls.aido")]
        out: String,
        #[arg(long, default_value_t = 600)]
        steps: usize,
    },
    /// Classify a text with a trained classifier.
    Classify {
        #[arg(short, long, default_value = "models/textcls.aido")]
        model: String,
        /// Text.
        text: String,
    },
    /// Train the tiny DDPM image generator.
    TrainImage {
        #[arg(short, long, default_value = "models/ddpm-shapes.aido")]
        out: String,
        #[arg(long, default_value_t = 3000)]
        steps: usize,
    },
    /// Sample images from a trained DDPM.
    SampleImage {
        #[arg(short, long, default_value = "models/ddpm-shapes.aido")]
        model: String,
        #[arg(short, long, default_value = "samples.png")]
        out: String,
        #[arg(short, long, default_value_t = 16)]
        n: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Inspect and edit weight slices: `list`, `stats`, `edit`.
    Weights {
        /// Action: list | stats | edit.
        action: String,
        #[arg(short, long)]
        model: String,
        /// Tensor name (stats/edit) or substring filter (list).
        #[arg(short, long)]
        tensor: Option<String>,
        /// Row range a:b (output neurons).
        #[arg(long)]
        rows: Option<String>,
        /// Column range a:b (input features).
        #[arg(long)]
        cols: Option<String>,
        /// Edit op: scale:k | set:v | add:v | noise:k | clamp:k | prune:k.
        #[arg(long)]
        op: Option<String>,
        /// Output .aido for edit.
        #[arg(short, long)]
        out: Option<String>,
        /// Outliers to show.
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// Benchmark a model on a task set: gsm8k | arith | seq.
    Evalset {
        #[command(flatten)]
        llm: LlmArgs,
        /// Task.
        #[arg(long, default_value = "arith")]
        task: String,
        /// Number of items.
        #[arg(short = 'n', long, default_value_t = 40)]
        n: usize,
        /// Max new tokens per item.
        #[arg(long, default_value_t = 24)]
        max_new: usize,
        /// Print every item.
        #[arg(long)]
        verbose: bool,
    },
    /// Find a behaviour direction from contrast prompts and edit it into the weights.
    Steer {
        #[arg(short, long)]
        model: String,
        /// Prompts showing the behaviour (one per line).
        #[arg(long)]
        pos: String,
        /// Contrast prompts.
        #[arg(long)]
        neg: String,
        /// 1 = remove the behaviour, negative = amplify it.
        #[arg(long, default_value_t = 1.0)]
        alpha: f32,
        /// Block whose residual defines the direction (default: best separation).
        #[arg(long)]
        layer: Option<usize>,
        /// Additive steering: add strength·(μ₊ − μ₋) after the block instead of
        /// projecting weights (use for *inducing* a behaviour).
        #[arg(long)]
        add: Option<f32>,
        /// Restrict the edit to blocks a:b (default: all blocks + embeddings).
        #[arg(long)]
        blocks: Option<String>,
        #[arg(short, long)]
        out: String,
    },
    /// Measure behaviour on prompts: refusal rate and rudeness.
    Behave {
        #[command(flatten)]
        llm: LlmArgs,
        /// Prompts file.
        #[arg(long)]
        prompts: String,
        #[arg(long, default_value_t = 60)]
        max_new: usize,
        /// Samples to print.
        #[arg(long, default_value_t = 3)]
        show: usize,
    },
    /// Train a Deep Q-Learning agent to play Snake.
    TrainSnake {
        #[arg(long, default_value_t = 800)]
        episodes: usize,
        #[arg(short, long, default_value = "models/snake-dqn.aido")]
        out: String,
    },
    /// Train the next-number (sequence pattern) model.
    TrainSeq {
        #[arg(long, default_value_t = 1500)]
        steps: usize,
        #[arg(short, long, default_value = "models/seq.aido")]
        out: String,
    },
    /// Download a model from the Hugging Face Hub into models/.
    Pull {
        /// Repository, e.g. Qwen/Qwen3-0.6B-GGUF.
        repo: String,
        /// File glob (default: best GGUF quant, or the whole repo).
        #[arg(short, long)]
        file: Option<String>,
        /// Target root directory.
        #[arg(long, default_value = "models")]
        dir: String,
    },
    /// Train the class-conditional pixel-art diffusion model (256x256 output).
    TrainPixelart {
        #[arg(long, default_value_t = 3000)]
        steps: usize,
        #[arg(long, default_value_t = 48)]
        channels: usize,
        #[arg(short, long, default_value = "models/pixelart.aido")]
        out: String,
    },
    /// Generate pixel art with a trained model.
    Pixelart {
        #[arg(short, long, default_value = "models/pixelart.aido")]
        model: String,
        /// Class: creature | potion | tree | house | all.
        #[arg(long, default_value = "all")]
        class: String,
        #[arg(short = 'n', long, default_value_t = 4)]
        n: usize,
        #[arg(long, default_value_t = 40)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(short, long, default_value = "pixelart.png")]
        out: String,
    },
    /// Generate the text-detection dataset and train the detector.
    TrainTextdet {
        #[arg(long, default_value_t = 1500)]
        steps: usize,
        #[arg(short, long, default_value = "models/textdet.aido")]
        out: String,
        /// Also export this many dataset samples to data/textdet/.
        #[arg(long, default_value_t = 300)]
        export: usize,
    },
    /// Train the own Russian+English tokenizer and LLM.
    TrainBilingual {
        #[arg(long, default_value_t = 6000)]
        vocab: usize,
        #[arg(long, default_value_t = 1500)]
        steps: usize,
        #[arg(long, default_value_t = 320)]
        dim: usize,
        #[arg(long, default_value_t = 6)]
        layers: usize,
        #[arg(short, long, default_value = "models/ruen-llm.aido")]
        out: String,
    },
    /// One-pass typed decisions with a d1 model (text / JSON state, pictures, audio).
    Decide {
        #[arg(short, long, default_value = "models/d1-omni")]
        model: String,
        /// State: text, or JSON (starting with { or [).
        #[arg(long)]
        state: Option<String>,
        /// Questions JSON (inline or @file.json).
        #[arg(short, long)]
        questions: String,
        /// Picture files.
        #[arg(long, num_args = 1..)]
        image: Vec<String>,
        /// 16 kHz mono WAV files.
        #[arg(long, num_args = 1..)]
        audio: Vec<String>,
    },
    /// Super-resolution: one image, a folder of frames, or a raw RGB24 stream.
    Upscale {
        /// Model (.safetensors), e.g. models/baikal/Baikal_LoopSR_x2_v2.safetensors.
        #[arg(short, long, default_value = "models/baikal/Baikal_LoopSR_x2_v2.safetensors")]
        model: String,
        /// Input image or folder of frames.
        #[arg(short, long)]
        input: Option<String>,
        /// Output image or folder.
        #[arg(short, long, default_value = "upscaled.png")]
        out: String,
        /// Body loops (1 = fastest, max = best).
        #[arg(long)]
        loops: Option<usize>,
        /// Raw RGB24 frames on stdin of this size (WxH), written x2 to stdout.
        #[arg(long)]
        stream: Option<String>,
        /// Disable the GPU.
        #[arg(long)]
        no_gpu: bool,
    },
    /// Make `aido` a global command (cmd, PowerShell, Windows Terminal, bash, zsh…).
    ///
    /// Windows: copies the binary to %LOCALAPPDATA%\\aido\\bin and adds that
    /// folder to the user PATH. Unix: links ~/.local/bin/aido and makes sure
    /// ~/.local/bin is on PATH in ~/.profile, ~/.bashrc and ~/.zshrc.
    Install {
        /// Remove the global command instead.
        #[arg(long)]
        uninstall: bool,
    },
    /// Show a model's I/O signature and metadata (SDK models: inputs/outputs/codec).
    ModelInfo {
        /// Model file.
        model: String,
        /// Attach the built-in signature for this task (e.g. `sequence-next`) and save in place.
        #[arg(long)]
        attach: Option<String>,
    },
    /// Run a custom SDK model on JSON input (validated against its signature).
    Predict {
        /// Model (.aido with an I/O signature).
        #[arg(short, long)]
        model: String,
        /// Input JSON object, or @file.json.
        #[arg(short, long)]
        input: String,
    },
    /// Verify checksums of an .aido file.
    Verify {
        /// File.
        file: String,
    },
    /// Generate an image with a diffusion model.
    Image {
        /// Diffusers-format model directory (e.g. models/sd-turbo).
        #[arg(long)]
        sd: String,
        /// Prompt.
        #[arg(short, long)]
        prompt: String,
        /// Output PNG.
        #[arg(short, long, default_value = "out.png")]
        out: String,
        /// Denoising steps.
        #[arg(long, default_value_t = 1)]
        steps: usize,
        /// Image size (multiple of 64).
        #[arg(long, default_value_t = 512)]
        size: usize,
        /// RNG seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Device: cpu | gpu | auto.
        #[arg(long, default_value = "auto")]
        device: String,
        /// Use the tiny TAESD decoder (directory with its safetensors).
        #[arg(long)]
        taesd: Option<String>,
        /// Classifier-free guidance scale (1 = off; SD-Turbo uses 1).
        #[arg(long, default_value_t = 1.0)]
        cfg: f32,
    },
    /// Start the OpenAI-compatible HTTP server.
    Serve {
        /// Model to serve as `name=path` or `path` (repeatable; the kind is detected:
        /// GGUF/.aido/MoE → llm, d1 dir → decision, LoopSR .safetensors → upscale,
        /// SD dir → diffusion). `hf:org/repo` paths are downloaded first.
        #[arg(short, long)]
        model: Vec<String>,
        /// Diffusion model directory (shortcut for `--model sd=DIR`).
        #[arg(long)]
        sd: Option<String>,
        /// TAESD directory for fast decoding.
        #[arg(long)]
        taesd: Option<String>,
        /// Bind address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Context length.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Threads.
        #[arg(short, long)]
        threads: Option<usize>,
        /// Diffusion device: cpu | gpu | auto.
        #[arg(long, default_value = "auto")]
        device: String,
        /// Unload idle models after this many seconds.
        #[arg(long, default_value_t = 600)]
        keep_alive: u64,
        /// RAM budget for simultaneously loaded models (GB); LRU eviction above it.
        #[arg(long, default_value_t = 12.0)]
        ram_gb: f32,
        /// VRAM budget shared by all models (GB); models beyond it run on the CPU.
        #[arg(long, default_value_t = 5.0)]
        vram_gb: f32,
        /// Load all models at start (otherwise on first request).
        #[arg(long)]
        preload: bool,
        /// Long-term memory directory (per-user stores; `none` disables).
        #[arg(long, default_value = "data/memory")]
        memory_dir: String,
    },
}

fn main() -> Result<()> {
    // Any argument of the form hf:org/repo[:pattern] is downloaded from the
    // Hugging Face Hub (once) and replaced by the local path.
    let raw: Vec<String> = std::env::args().collect();
    let is_pull = raw.get(1).map(|a| a == "pull").unwrap_or(false);
    let args: Vec<String> = raw.into_iter().map(|a| if a.starts_with("hf:") && !is_pull { aido::hub::resolve(&a).unwrap_or(a) } else { a }).collect();
    let cli = Cli::parse_from(args);
    std::env::set_var("AIDO_OPTIMIZER", &cli.optimizer);
    match cli.cmd {
        Cmd::Info => info(),
        Cmd::Run { llm, prompt, max_tokens, temp, raw, think } => {
            let prompt = match prompt.strip_prefix('@') {
                Some(f) => std::fs::read_to_string(f)?,
                None => prompt,
            };
            run(llm, prompt, max_tokens, temp, raw, think)
        }
        Cmd::Chat { llm, temp } => chat(llm, temp),
        Cmd::Bench { llm, pp, tg } => bench(llm, pp, tg),
        Cmd::Analyze { model, corpus, out, tokens, threads } => analyze_cmd(model, corpus, out, tokens, threads),
        Cmd::Compile { model, out, quant, drop_layers, ffn_keep, stats, vocab_corpus } => compile_cmd(model, out, quant, drop_layers, ffn_keep, stats, vocab_corpus),
        Cmd::Eval { llm, text, max_bytes } => eval_cmd(llm, text, max_bytes),
        Cmd::DataPack { files, out, tokenizer } => train_cmds::data_pack(&files, &out, tokenizer.as_deref()),
        Cmd::TrainLm { data, out, steps, dim, layers, heads, ctx, batch, lr, resume, threads } => train_cmds::train_lm(&data, &out, steps, dim, layers, heads, ctx, batch, lr, resume.as_deref(), threads),
        Cmd::Finetune { model, data, out, steps, lora, ctx, batch, lr, threads } => train_cmds::finetune(&model, &data, &out, steps, lora, ctx, batch, lr, threads),
        Cmd::TrainClassifier { out, steps } => train_cmds::train_classifier(&out, steps, None),
        Cmd::Classify { model, text } => train_cmds::classify(&model, &text),
        Cmd::TrainImage { out, steps } => train_cmds::train_image(&out, steps, None),
        Cmd::SampleImage { model, out, n, seed } => train_cmds::sample_image(&model, &out, n, seed),
        Cmd::Weights { action, model, tensor, rows, cols, op, out, top } => match action.as_str() {
            "list" => aido::weights::list(&model, tensor.as_deref()),
            "stats" => aido::weights::slice_stats(&model, &tensor.ok_or_else(|| anyhow::anyhow!("--tensor required"))?, rows.as_deref(), cols.as_deref(), top).map(|_| ()),
            "edit" => {
                let n = aido::weights::edit(&model, &out.ok_or_else(|| anyhow::anyhow!("--out required"))?, &tensor.ok_or_else(|| anyhow::anyhow!("--tensor required"))?, rows.as_deref(), cols.as_deref(), op.ok_or_else(|| anyhow::anyhow!("--op required"))?.parse()?, 1)?;
                println!("edited {n} values");
                Ok(())
            }
            a => anyhow::bail!("unknown action {a} (list|stats|edit)"),
        },
        Cmd::Evalset { llm, task, n, max_new, verbose } => {
            let mut o = llm.options()?;
            o.calibrate = false;
            let mut eng = LlmEngine::load(&llm.model, o)?;
            let items = match task.as_str() {
                "gsm8k" => aido::evals::load_gsm8k("data/gsm8k_test.jsonl", n)?,
                "seq" => aido::evals::sequence_items(n, 7),
                _ => aido::evals::arith_items(n, 7),
            };
            let s = aido::evals::run(&mut eng, &items, max_new, verbose)?;
            println!("{task}: {}/{} = {:.1}% in {:.1}s — {}", s.correct, s.total, s.percent(), s.secs, llm.model);
            Ok(())
        }
        Cmd::Steer { model, pos, neg, alpha, layer, add, blocks, out } => {
            let cpu = sys::CpuInfo::detect();
            let pool = ThreadPool::new(cpu.physical, Some(cpu.physical_core_ids()));
            let m = aido::llm::model::Model::load(&model)?;
            let tok = aido::tokenizer::Tokenizer::from_gguf(m.store.as_ref())?;
            let lines = |f: &str| -> Result<Vec<String>> { Ok(std::fs::read_to_string(f)?.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect()) };
            let (p, n) = (lines(&pos)?, lines(&neg)?);
            let (mp, mn) = (aido::steer::mean_resid(&m, &tok, &pool, &p)?, aido::steer::mean_resid(&m, &tok, &pool, &n)?);
            let sep = aido::steer::separation(&mp, &mn);
            let nl = sep.len();
            // Prefer middle layers: early ones mix token identity, the last ones the output distribution.
            let best = layer.unwrap_or_else(|| (nl / 4..nl * 3 / 4).max_by(|&a, &b| sep[a].partial_cmp(&sep[b]).unwrap()).unwrap());
            println!("separation per block: {}", sep.iter().enumerate().map(|(i, v)| format!("{i}:{v:.2}")).collect::<Vec<_>>().join(" "));
            println!("using block {best} (separation {:.3})", sep[best]);
            if let Some(k) = add {
                let v: Vec<f32> = mp[best].iter().zip(&mn[best]).map(|(a, b)| k * (a - b)).collect();
                drop(m);
                aido::steer::add_vector(&model, &out, best, &v, &format!("{pos} vs {neg}"))?;
                println!("added steering vector ×{k} after block {best} → {out}");
                return Ok(());
            }
            let r = aido::steer::direction(&mp[best], &mn[best]);
            drop(m);
            let br = blocks.as_deref().map(|b| aido::weights::parse_range(Some(b), nl)).transpose()?;
            let k = aido::steer::apply(&model, &out, &r, alpha, &format!("{pos} vs {neg} @ block {best}"), br)?;
            println!("edited {k} matrices with alpha {alpha} → {out}");
            Ok(())
        }
        Cmd::Behave { llm, prompts, max_new, show } => {
            let mut o = llm.options()?;
            o.calibrate = false;
            let mut eng = LlmEngine::load(&llm.model, o)?;
            let mut refusals = 0;
            let mut rude = 0;
            let ps: Vec<String> = std::fs::read_to_string(&prompts)?.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect();
            for (i, p) in ps.iter().enumerate() {
                let ids = eng.chat_prompt(&[Message::new("user", p)], false)?;
                let mut out = String::new();
                eng.generate(&ids, &GenParams { max_tokens: max_new, ..Default::default() }, &mut |s| {
                    out.push_str(s);
                    true
                })?;
                let rf = aido::steer::is_refusal(&out);
                refusals += rf as usize;
                rude += aido::steer::rudeness(&out);
                if i < show {
                    println!("> {p}\n  [{}] {}\n", if rf { "REFUSE" } else { "answer" }, out.replace('\n', " ").chars().take(160).collect::<String>());
                }
            }
            println!("{}: refusal rate {}/{} = {:.0}% | rude markers {} ({:.2} per answer)", llm.model, refusals, ps.len(), 100.0 * refusals as f64 / ps.len() as f64, rude, rude as f64 / ps.len() as f64);
            Ok(())
        }
        Cmd::TrainSnake { episodes, out } => train_cmds::train_snake(episodes, &out),
        Cmd::TrainSeq { steps, out } => train_cmds::train_seq(steps, &out),
        Cmd::Pull { repo, file, dir } => {
            // Accept `org/repo`, `hf:org/repo` and `hf:org/repo:pattern`.
            let r = repo.trim_start_matches("hf:");
            let (r, pat) = match r.split_once(':') {
                Some((a, b)) => (a, Some(b.to_string())),
                None => (r, file),
            };
            let p = aido::hub::pull(r, pat.as_deref(), &dir)?;
            println!("ready: {}", p.display());
            Ok(())
        }
        Cmd::TrainPixelart { steps, channels, out } => train_cmds::train_pixelart(steps, channels, &out),
        Cmd::Pixelart { model, class, n, steps, seed, out } => train_cmds::pixelart(&model, &class, n, steps, seed, &out),
        Cmd::TrainTextdet { steps, out, export } => train_cmds::train_textdet(steps, &out, export),
        Cmd::TrainBilingual { vocab, steps, dim, layers, out } => train_cmds::train_bilingual(vocab, steps, dim, layers, &out),
        Cmd::Decide { model, state, questions, image, audio } => {
            let cpu = sys::CpuInfo::detect();
            let pool = ThreadPool::new(cpu.physical, Some(cpu.physical_core_ids()));
            let t = Instant::now();
            let gpu = if std::env::var("AIDO_NO_GPU").is_ok() { None } else { aido::gpu::Gpu::new().ok().map(std::sync::Arc::new) };
            let vram: f64 = std::env::var("AIDO_VRAM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(4.5);
            let m = aido::llm::d1::D1::load_with(&model, &pool, &aido::llm::lfm3::Accel::new(gpu, (vram * 1e9) as usize))?;
            eprintln!("[d1] loaded in {:.1}s", t.elapsed().as_secs_f64());
            let qtext = match questions.strip_prefix('@') { Some(f) => std::fs::read_to_string(f)?, None => questions };
            let q: serde_json::Value = serde_json::from_str(&qtext)?;
            let st = match state {
                None => serde_json::Value::Null,
                Some(s) if s.trim_start().starts_with('{') || s.trim_start().starts_with('[') => serde_json::from_str(&s)?,
                Some(s) => serde_json::Value::String(s),
            };
            let mut media = Vec::new();
            for p in &image {
                media.push(aido::llm::d1_media::MediaInput::Image(image::open(p)?.to_rgb8()));
            }
            for p in &audio {
                media.push(aido::llm::d1_media::MediaInput::Audio(aido::llm::d1_media::read_wav(p)?));
            }
            let out = m.decide(&pool, &st, &q, &media)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
        Cmd::Upscale { model, input, out, loops, stream, no_gpu } => upscale_cmd(&model, input.as_deref(), &out, loops, stream.as_deref(), no_gpu),
        Cmd::Install { uninstall } => install_cmd(uninstall),
        Cmd::ModelInfo { model, attach } => {
            if let Some(task) = attach {
                anyhow::ensure!(task == "sequence-next", "built-in signatures: sequence-next");
                aido::sdk::io::attach(&model, &model, &aido::sdk::io::seq_signature())?;
            }
            use aido::format::TensorStore;
            let st = aido::format::open(&model)?;
            println!("file        : {model} ({:.2} MB)", std::fs::metadata(&model)?.len() as f64 / 1e6);
            println!("architecture: {}", st.get_str("general.architecture").unwrap_or("?"));
            println!("tensors     : {}", st.names().len());
            println!("kind        : {}", aido::models::detect_kind(&model).map(|k| k.name()).unwrap_or("unknown"));
            if model.ends_with(".aido") {
                if let Some(sig) = aido::sdk::io::Signature::read(&model)? {
                    println!("\n{}", sig.describe());
                    println!("\nJSON schema:\n{}", serde_json::to_string_pretty(&sig)?);
                }
            }
            Ok(())
        }
        Cmd::Predict { model, input } => {
            let txt = match input.strip_prefix('@') {
                Some(f) => std::fs::read_to_string(f)?,
                None => input,
            };
            let v: serde_json::Value = serde_json::from_str(&txt)?;
            let cpu = sys::CpuInfo::detect();
            let pool = ThreadPool::new(cpu.physical, Some(cpu.physical_core_ids()));
            let mut m = aido::sdk::io::SdkModel::load(&model)?;
            let t = Instant::now();
            let out = m.predict(&pool, &v)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            eprintln!("[predict] {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            Ok(())
        }
        Cmd::Verify { file } => {
            let f = aido::format::AidoFile::open(&file)?;
            let bad = f.verify();
            if bad.is_empty() {
                println!("OK — all tensors match their checksums; provenance:");
                for p in &f.provenance {
                    println!("  {p}");
                }
                Ok(())
            } else {
                anyhow::bail!("corrupted tensors: {bad:?}")
            }
        }
        Cmd::Image { sd, prompt, out, steps, size, seed, device, taesd, cfg } => {
            let mut p = aido::diffusion::pipeline::Pipeline::load(&sd, taesd.as_deref(), device.parse()?, None)?;
            let t = Instant::now();
            let img = p.generate(&aido::diffusion::pipeline::ImageParams { prompt, negative: String::new(), steps, width: size, height: size, seed, guidance: cfg })?;
            eprintln!("image generated in {:.2}s ({})", t.elapsed().as_secs_f64(), p.timings());
            img.save_png(&out)?;
            eprintln!("saved {out}");
            Ok(())
        }
        Cmd::Serve { model, sd, taesd, host, port, ctx, threads, device, keep_alive, ram_gb, vram_gb, preload, memory_dir } => {
            let mut models: Vec<(String, String)> = model.iter().map(|m| split_model_arg(m)).collect();
            if let Some(d) = sd {
                models.push(("sd".into(), d));
            }
            anyhow::ensure!(!models.is_empty(), "give at least one --model");
            aido::server::serve(aido::server::ServerConfig { models, taesd, host, port, ctx, threads, device: device.parse()?, keep_alive, ram_gb, vram_gb, preload, memory_dir: Some(memory_dir).filter(|d| d != "none") })
        }
    }
}

/// `aido info`.
fn info() -> Result<()> {
    let cpu = sys::CpuInfo::detect();
    println!("CPU      : {}", cpu.brand);
    println!("cores    : {} physical / {} logical (SMT {})", cpu.physical, cpu.logical, cpu.smt);
    println!("caches   : L1d {} KiB, L2 {} KiB, L3 {} MiB", cpu.l1d >> 10, cpu.l2 >> 10, cpu.l3 >> 20);
    println!("SIMD     : avx2={} fma={} f16c={} avx512f={}", cpu.avx2, cpu.fma, cpu.f16c, cpu.avx512f);
    for &n in &[1usize, 2, 4, cpu.physical / 2, cpu.physical, cpu.logical] {
        if n == 0 {
            continue;
        }
        let pool = ThreadPool::new(n, Some(cpu.physical_core_ids().into_iter().chain((0..cpu.logical).map(|x| x)).collect()));
        println!("DRAM bw  : {:>2} threads → {:.1} GB/s", n, sys::memory_bandwidth_gbs(&pool, 512));
    }
    match aido::gpu::Gpu::new() {
        Ok(g) => println!("GPU      : {}", g.describe()),
        Err(e) => println!("GPU      : unavailable ({e})"),
    }
    Ok(())
}

/// Builds the prompt token ids (chat template unless raw).
fn prompt_ids(eng: &LlmEngine, msgs: &[Message], raw: bool, think: bool) -> Result<Vec<u32>> {
    if raw {
        return Ok(eng.tok.encode(&msgs.last().unwrap().content, true));
    }
    eng.chat_prompt(msgs, think)
}

/// `aido run`.
fn run(a: LlmArgs, prompt: String, max_tokens: usize, temp: f32, raw: bool, think: bool) -> Result<()> {
    let t = Instant::now();
    let mut eng = LlmEngine::load(&a.model, a.options()?)?;
    eprintln!("loaded {} in {:.2}s; decode threads {} (calibration {:?})", eng.model.describe(), t.elapsed().as_secs_f64(), eng.decode_width, eng.calibration.iter().map(|(w, s)| format!("{w}:{:.1}ms", s * 1e3)).collect::<Vec<_>>());
    let ids = prompt_ids(&eng, &[Message::new("user", &prompt)], raw, think)?;
    let params = GenParams { max_tokens, sampler: SamplerConfig { temperature: temp, ..Default::default() }, ..Default::default() };
    let st = eng.generate(&ids, &params, &mut |s| {
        print!("{s}");
        std::io::stdout().flush().ok();
        true
    })?;
    println!();
    eprintln!("[kv] {:.1} MB in session caches", eng.state_bytes() as f64 / 1e6);
    eprintln!(
        "[prompt {} tok @ {:.1} tok/s | gen {} tok @ {:.2} tok/s | passes {} | drafts {}/{} accepted]",
        st.prompt_tokens, st.prefill_tps(), st.gen_tokens, st.tokens_per_s(), st.passes, st.accepted, st.drafted
    );
    let prof = aido::llm::qwen35::take_profile();
    if prof.iter().any(|p| p.1 > 0.0) {
        let tot: f64 = prof.iter().map(|p| p.1).sum();
        eprintln!("[profile] {}", prof.iter().map(|(n, s)| format!("{n} {:.2}s ({:.0}%)", s, 100.0 * s / tot)).collect::<Vec<_>>().join(" | "));
    }
    Ok(())
}

/// `aido chat`.
fn chat(a: LlmArgs, temp: f32) -> Result<()> {
    let mut eng = LlmEngine::load(&a.model, a.options()?)?;
    eprintln!("{} — type your message, empty line to quit", eng.model.describe());
    let mut msgs: Vec<Message> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        msgs.push(Message::new("user", line.trim()));
        let ids = prompt_ids(&eng, &msgs, false, false)?;
        let mut answer = String::new();
        let params = GenParams { max_tokens: 1024, sampler: SamplerConfig { temperature: temp, ..Default::default() }, ..Default::default() };
        let st = eng.generate(&ids, &params, &mut |s| {
            answer.push_str(s);
            print!("{s}");
            std::io::stdout().flush().ok();
            true
        })?;
        eprintln!("\n[{:.1} tok/s, {} cached prompt tokens]", st.tokens_per_s(), st.cached_tokens);
        msgs.push(Message::new("assistant", &answer));
    }
    Ok(())
}

/// `aido bench`.
fn bench(a: LlmArgs, pp: usize, tg: usize) -> Result<()> {
    let mut opts = a.options()?;
    let spec_requested = opts.spec;
    opts.spec = SpecMode::Off;
    let mut eng = LlmEngine::load(&a.model, opts)?;
    let wbytes = eng.model.weight_bytes_per_token() as f64;
    println!("model    : {}", eng.model.describe());
    println!("weights  : {:.2} GB streamed per token", wbytes / 1e9);
    println!("threads  : pool {} | decode width {} | batch width {}", eng.pool.n_threads(), eng.decode_width, eng.batch_width);
    println!("calib    : {}", eng.calibration.iter().map(|(w, s)| format!("{w}t={:.1}ms", s * 1e3)).collect::<Vec<_>>().join("  "));
    // Prefill.
    let prompt: Vec<u32> = (0..pp as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
    let params = GenParams { max_tokens: 1, ..Default::default() };
    let st = eng.generate(&prompt, &params, &mut |_| true)?;
    println!("prefill  : {pp} tokens in {:.3}s → {:.1} tok/s", st.prefill_s, st.prefill_tps());
    // Generation (plain).
    eng.reset();
    let p2 = eng.tok.encode("Once upon a time, in a small village,", true);
    let params = GenParams { max_tokens: tg, spec: Some(SpecMode::Off), ..Default::default() };
    let st = eng.generate(&p2, &params, &mut |_| true)?;
    let tps = st.tokens_per_s();
    println!("decode   : {} tokens → {:.2} tok/s  ({:.1} GB/s effective weight bandwidth)", st.gen_tokens, tps, tps * wbytes / 1e9);
    // Verification-cost curve T(t): the key quantity behind speculative decoding.
    {
        let m = eng.model.as_ref();
        let can_rewind = m.can_rewind();
        let mut st = m.new_state(512, 32, eng.pool.n_threads());
        let toks: Vec<u32> = (0..32u32).map(|i| 1000 + i).collect();
        m.forward(&eng.pool, &toks[..16], 0, st.as_mut(), false)?;
        let mut line = String::from("T(t)     :");
        let mut t1 = 0.0;
        for &t in &[1usize, 2, 4, 8, 16, 32] {
            eng.pool.set_width(if t == 1 { eng.decode_width } else { eng.batch_width });
            let mut v = Vec::new();
            for _ in 0..if can_rewind { 7 } else { 1 } {
                if !can_rewind {
                    m.forward(&eng.pool, &toks[..16], 0, st.as_mut(), false)?;
                }
                let t0 = Instant::now();
                m.forward(&eng.pool, &toks[..t], if can_rewind { 100 } else { 16 }, st.as_mut(), true)?;
                v.push(t0.elapsed().as_secs_f64());
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dt = v[v.len() / 2];
            if t == 1 {
                t1 = dt;
            }
            line += &format!("  t={t}:{:.1}ms({:.2}x)", dt * 1e3, dt / t1);
        }
        println!("{line}");
    }
    if spec_requested != SpecMode::Off {
        eng.reset();
        let params = GenParams { max_tokens: tg, spec: Some(spec_requested), ..Default::default() };
        let st2 = eng.generate(&p2, &params, &mut |_| true)?;
        println!(
            "spec     : {} tokens → {:.2} tok/s ({:.2}x), passes {}, drafts {}/{} accepted",
            st2.gen_tokens,
            st2.tokens_per_s(),
            st2.tokens_per_s() / tps,
            st2.passes,
            st2.accepted,
            st2.drafted
        );
    }
    Ok(())
}

/// Reads and concatenates text files.
fn read_texts(files: &[String]) -> Result<String> {
    let mut s = String::new();
    for f in files {
        s.push_str(&String::from_utf8_lossy(&std::fs::read(f)?));
        s.push('\n');
    }
    Ok(s)
}

/// `aido analyze`.
fn analyze_cmd(model: String, corpus: Vec<String>, out: String, tokens: usize, threads: Option<usize>) -> Result<()> {
    let cpu = sys::CpuInfo::detect();
    let pool = ThreadPool::new(threads.unwrap_or(cpu.physical), Some(cpu.physical_core_ids()));
    let m = aido::llm::model::Model::load(&model)?;
    let tok = aido::tokenizer::Tokenizer::from_gguf(m.store.as_ref())?;
    let text = read_texts(&corpus)?;
    let t = Instant::now();
    let st = aido::optimize::analyze(&m, &tok, &pool, &text, tokens, 512)?;
    println!("analysed {} tokens in {:.1}s — {}", st.tokens, t.elapsed().as_secs_f64(), m.describe());
    println!("block influence (1 - cos(x_in, x_out)), lower = more redundant:");
    let mut order: Vec<usize> = (0..st.block_influence.len()).collect();
    order.sort_by(|&a, &b| st.block_influence[a].partial_cmp(&st.block_influence[b]).unwrap());
    for (l, bi) in st.block_influence.iter().enumerate() {
        let bar = "#".repeat((bi * 200.0).min(60.0) as usize);
        println!("  L{l:>2} {bi:.4} {bar}");
    }
    println!("most redundant blocks: {:?}", &order[..order.len().min(6)]);
    for rel in [0.05f32, 0.1, 0.2] {
        println!("FFN neurons with mean |a| < {:.0}% of layer mean: {:.1}%", rel * 100.0, st.cold_fraction(rel) * 100.0);
    }
    let used = st.token_counts.iter().filter(|&&c| c > 0).count();
    println!("vocabulary: {used} of {} tokens used by the corpus ({:.1}%)", st.token_counts.len(), 100.0 * used as f64 / st.token_counts.len() as f64);
    let head = m.output.nbytes() as f64;
    println!("LM head share of per-token weight traffic: {:.1}%", 100.0 * head / m.weight_bytes_per_token() as f64);
    std::fs::write(&out, serde_json::to_vec(&st)?)?;
    println!("saved {out}");
    Ok(())
}

/// `aido compile`.
#[allow(clippy::too_many_arguments)]
fn compile_cmd(model: String, out: String, quant: Option<String>, drop_layers: Option<String>, ffn_keep: Option<f32>, stats: Option<String>, vocab_corpus: Vec<String>) -> Result<()> {
    // SDK models (own networks): inference compile keeping the I/O signature.
    if model.ends_with(".aido") {
        use aido::format::TensorStore;
        if aido::format::AidoFile::open(&model)?.get_str("general.architecture") == Some("aido-sdk") {
            let sig = aido::sdk::io::Signature::read(&model)?;
            let (a, b) = aido::sdk::io::compile(&model, &out, sig.as_ref(), quant.as_deref())?;
            println!("compiled SDK model {model} -> {out}: {:.2} MB -> {:.2} MB{}", a as f64 / 1e6, b as f64 / 1e6, if sig.is_some() { " (I/O signature kept)" } else { "" });
            return Ok(());
        }
    }
    let mut opt = aido::optimize::CompileOptions::default();
    if let Some(q) = quant {
        opt.quant = Some(aido::tensor::quantize::parse_type(&q)?);
    }
    if let Some(d) = drop_layers {
        if let Some(n) = d.strip_prefix("auto:") {
            opt.drop_auto = n.parse()?;
        } else {
            opt.drop_layers = d.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse()).collect::<Result<_, _>>()?;
        }
    }
    opt.ffn_keep = ffn_keep;
    if let Some(p) = stats {
        opt.stats = Some(serde_json::from_slice(&std::fs::read(p)?)?);
    }
    if !vocab_corpus.is_empty() {
        opt.vocab_corpus = Some(read_texts(&vocab_corpus)?);
    }
    let t = Instant::now();
    let r = aido::optimize::compile(&model, &out, &opt)?;
    println!(
        "compiled in {:.1}s: {:.1} MB → {:.1} MB ({:.1}%), layers {}, vocab {}, ffn per layer {:?}",
        t.elapsed().as_secs_f64(),
        r.bytes_in as f64 / 1e6,
        r.bytes_out as f64 / 1e6,
        100.0 * r.bytes_out as f64 / r.bytes_in as f64,
        r.layers,
        r.vocab,
        &r.ffn[..r.ffn.len().min(8)]
    );
    Ok(())
}

/// `aido eval`.
fn eval_cmd(a: LlmArgs, text: String, max_bytes: usize) -> Result<()> {
    let mut opts = a.options()?;
    opts.calibrate = false;
    let eng = LlmEngine::load(&a.model, opts)?;
    let mut t = String::from_utf8_lossy(&std::fs::read(&text)?).into_owned();
    if t.len() > max_bytes {
        let mut cut = max_bytes;
        while !t.is_char_boundary(cut) {
            cut -= 1;
        }
        t.truncate(cut);
    }
    let st = Instant::now();
    let (bpb, n) = aido::optimize::bits_per_byte(eng.model.as_ref(), &eng.tok, &eng.pool, &t, 512)?;
    println!("bits/byte {bpb:.4} over {n} tokens ({} bytes) in {:.1}s — {}", t.len(), st.elapsed().as_secs_f64(), eng.model.describe());
    Ok(())
}

/// `aido upscale`.
fn upscale_cmd(model: &str, input: Option<&str>, out: &str, loops: Option<usize>, stream: Option<&str>, no_gpu: bool) -> Result<()> {
    use std::io::{Read, Write};
    let cpu = sys::CpuInfo::detect();
    let pool = ThreadPool::new(cpu.physical, Some(cpu.physical_core_ids()));
    let gpu = if no_gpu { None } else { aido::gpu::Gpu::new().ok().map(std::sync::Arc::new) };
    let t = Instant::now();
    let mut m = aido::upscale::LoopSr::load(model, gpu)?;
    if let Some(l) = loops {
        m.set_loops(l);
    }
    eprintln!("[upscale] loaded in {:.2}s, x{} with {} loop(s)", t.elapsed().as_secs_f64(), m.scale, m.loops);
    let run = |img: &image::RgbImage| -> (image::RgbImage, f64) {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let x: Vec<f32> = img.as_raw().iter().map(|v| *v as f32 / 255.0).collect();
        let t = Instant::now();
        let (oh, ow, y) = m.upscale(&pool, &x, h, w);
        let dt = t.elapsed().as_secs_f64();
        let bytes: Vec<u8> = y.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        (image::RgbImage::from_raw(ow as u32, oh as u32, bytes).unwrap(), dt)
    };
    if let Some(size) = stream {
        let (w, h) = size.split_once('x').map(|(a, b)| (a.parse::<usize>().unwrap_or(0), b.parse::<usize>().unwrap_or(0))).unwrap_or((0, 0));
        anyhow::ensure!(w > 0 && h > 0, "--stream WxH");
        let mut buf = vec![0u8; w * h * 3];
        let (mut n, t0) = (0usize, Instant::now());
        let mut stdin = std::io::stdin().lock();
        let mut stdout = std::io::stdout().lock();
        while stdin.read_exact(&mut buf).is_ok() {
            let img = image::RgbImage::from_raw(w as u32, h as u32, buf.clone()).unwrap();
            let (o, _) = run(&img);
            stdout.write_all(o.as_raw())?;
            n += 1;
            eprint!("\r[upscale] frame {n} | {:.2} fps   ", n as f64 / t0.elapsed().as_secs_f64());
        }
        eprintln!();
        return Ok(());
    }
    let input = input.ok_or_else(|| anyhow::anyhow!("--input or --stream required"))?;
    if std::path::Path::new(input).is_dir() {
        std::fs::create_dir_all(out)?;
        let mut files: Vec<_> = std::fs::read_dir(input)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
        files.sort();
        let t0 = Instant::now();
        for (i, f) in files.iter().enumerate() {
            let (o, dt) = run(&image::open(f)?.to_rgb8());
            o.save(std::path::Path::new(out).join(f.file_name().unwrap()))?;
            eprintln!("[upscale] {}/{} {} in {:.0} ms", i + 1, files.len(), f.display(), dt * 1e3);
        }
        println!("{} frames in {:.2}s = {:.2} fps", files.len(), t0.elapsed().as_secs_f64(), files.len() as f64 / t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let img = image::open(input)?.to_rgb8();
    let (o, dt) = run(&img);
    o.save(out)?;
    println!("{}x{} -> {}x{} in {:.0} ms -> {out}", img.width(), img.height(), o.width(), o.height(), dt * 1e3);
    Ok(())
}

/// Splits a `--model` argument into `(name, path)`: `name=path`, or just a
/// path whose file stem becomes the name.
///
/// # Example
/// `split_model_arg("qwen=models/q.gguf")` → `("qwen", "models/q.gguf")`;
/// `split_model_arg("models/q.gguf")` → `("q", "models/q.gguf")`.
fn split_model_arg(m: &str) -> (String, String) {
    match m.split_once('=') {
        Some((n, p)) if !n.is_empty() && !n.contains(['/', '\\', ':', '.']) => (n.to_string(), p.to_string()),
        _ => {
            let p = std::path::Path::new(m.trim_end_matches(['/', '\\']));
            (p.file_stem().and_then(|s| s.to_str()).unwrap_or(m).to_string(), m.to_string())
        }
    }
}

/// `aido install`: puts the running binary on the user's PATH.
///
/// Idempotent: running it again only refreshes the copied binary.
fn install_cmd(uninstall: bool) -> Result<()> {
    let exe = std::env::current_exe()?;
    #[cfg(windows)]
    {
        let base = std::env::var("LOCALAPPDATA").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
        let dir = base.join("aido").join("bin");
        let dir_s = dir.to_string_lossy().to_string();
        // Read and write the *user* PATH through .NET so that the change is
        // broadcast (WM_SETTINGCHANGE) to Explorer and new terminals.
        let ps = |script: &str| -> Result<String> {
            let o = std::process::Command::new("powershell").args(["-NoProfile", "-NonInteractive", "-Command", script]).output()?;
            anyhow::ensure!(o.status.success(), "powershell: {}", String::from_utf8_lossy(&o.stderr));
            Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
        };
        let cur = ps("[Environment]::GetEnvironmentVariable('Path','User')")?;
        let parts: Vec<&str> = cur.split(';').filter(|p| !p.is_empty()).collect();
        let has = parts.iter().any(|p| p.trim_end_matches('\\').eq_ignore_ascii_case(&dir_s));
        if uninstall {
            let rest: Vec<&str> = parts.into_iter().filter(|p| !p.trim_end_matches('\\').eq_ignore_ascii_case(&dir_s)).collect();
            ps(&format!("[Environment]::SetEnvironmentVariable('Path','{}','User')", rest.join(";").replace('\'', "''")))?;
            let _ = std::fs::remove_dir_all(&dir);
            println!("removed {dir_s} from the user PATH");
            return Ok(());
        }
        std::fs::create_dir_all(&dir)?;
        let target = dir.join("aido.exe");
        if exe != target {
            // A running copy cannot be overwritten on Windows: move it aside first.
            if target.exists() {
                let old = dir.join("aido.old.exe");
                let _ = std::fs::remove_file(&old);
                let _ = std::fs::rename(&target, &old);
            }
            std::fs::copy(&exe, &target)?;
        }
        if !has {
            let new = if cur.is_empty() { dir_s.clone() } else { format!("{};{}", cur.trim_end_matches(';'), dir_s) };
            ps(&format!("[Environment]::SetEnvironmentVariable('Path','{}','User')", new.replace('\'', "''")))?;
        }
        println!("installed {} -> {}", exe.display(), target.display());
        println!("{}", if has { "PATH already contained it" } else { "added to the user PATH: open a new cmd / PowerShell window and run `aido info`" });
    }
    #[cfg(not(windows))]
    {
        let home = std::path::PathBuf::from(std::env::var("HOME")?);
        let bin = home.join(".local").join("bin");
        let link = bin.join("aido");
        if uninstall {
            let _ = std::fs::remove_file(&link);
            println!("removed {}", link.display());
            return Ok(());
        }
        std::fs::create_dir_all(&bin)?;
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&exe, &link)?;
        let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
        for rc in [".profile", ".bashrc", ".zshrc"] {
            let f = home.join(rc);
            if rc != ".profile" && !f.exists() {
                continue;
            }
            let txt = std::fs::read_to_string(&f).unwrap_or_default();
            if !txt.contains(".local/bin") {
                use std::io::Write as _;
                std::fs::OpenOptions::new().create(true).append(true).open(&f)?.write_all(format!("\n# added by aido install\n{line}\n").as_bytes())?;
            }
        }
        println!("linked {} -> {}; open a new shell and run `aido info`", link.display(), exe.display());
    }
    Ok(())
}
