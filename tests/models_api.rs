//! Integration tests of the embeddable [`aido::Models`] API on real models.
//!
//! Tests whose model files are missing are skipped (they print why), so the
//! suite also runs on machines without the downloaded models.

use aido::llm::chat::Message;
use aido::llm::engine::GenParams;
use aido::{Kind, Models, ModelsConfig};
use serde_json::json;

fn have(p: &str) -> bool {
    let ok = std::path::Path::new(p).exists();
    if !ok {
        eprintln!("skipped: {p} not found");
    }
    ok
}

fn models() -> Models {
    Models::new(ModelsConfig { ram_gb: 8.0, ..Default::default() }).unwrap()
}

#[test]
fn kind_detection() {
    for (p, k) in [("models/Qwen3-0.6B-Q8_0.gguf", Kind::Llm), ("models/d1-omni", Kind::Decision), ("models/baikal/Baikal_LoopSR_x2_v2.safetensors", Kind::Upscale), ("models/sd-turbo", Kind::Diffusion)] {
        if have(p) {
            assert_eq!(aido::models::detect_kind(p).unwrap(), k, "{p}");
        }
    }
    assert!(aido::models::detect_kind("does/not/exist.gguf").is_err());
}

#[test]
fn llm_chat_and_streaming() {
    let p = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    if !have(p) {
        return;
    }
    let m = models();
    m.add("q", p).unwrap();
    let mut pieces = 0;
    let out = m.chat("q", &[Message::new("user", "What is the capital of France? One word.")], &GenParams { max_tokens: 16, ..Default::default() }, &mut |_| {
        pieces += 1;
        true
    })
    .unwrap();
    assert!(out.text.to_lowercase().contains("paris"), "{}", out.text);
    assert!(pieces >= 1);
    // Early stop from the callback.
    let out = m.complete("q", "1, 2, 3,", &GenParams { max_tokens: 50, ..Default::default() }, &mut |_| false).unwrap();
    assert!(out.stats.gen_tokens <= 2, "{}", out.stats.gen_tokens);
    // Default model of a kind (empty name) and errors.
    assert!(m.complete("", "Hi", &GenParams { max_tokens: 1, ..Default::default() }, &mut |_| true).is_ok());
    assert!(m.complete("missing", "Hi", &Default::default(), &mut |_| true).is_err());
    assert!(m.unload("q"));
    assert!(!m.unload("q"));
}

#[test]
fn shared_instance_for_same_file() {
    let p = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    if !have(p) {
        return;
    }
    let m = models();
    m.add("a", p).unwrap();
    m.add("b", p).unwrap();
    m.preload("a").unwrap();
    let t = std::time::Instant::now();
    m.preload("b").unwrap(); // reuses a's instance: no second load
    assert!(t.elapsed().as_millis() < 200, "{:?}", t.elapsed());
}

#[test]
fn decision_text() {
    let p = "models/d1-omni";
    if !have(p) {
        return;
    }
    let m = models();
    m.add("d1", p).unwrap();
    let q = json!({"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}});
    let out = m.decide("d1", &json!("Please give me my money back, the item is broken."), &q, &[]).unwrap();
    assert!(out["answers"]["refund"]["noul"].as_f64().unwrap() > 0.5, "{out}");
}

#[test]
fn upscale_beats_bicubic() {
    let p = "models/baikal/Baikal_LoopSR_x2_v2.safetensors";
    if !have(p) || !have("data/test_media/a128_hr.png") {
        return;
    }
    let m = models();
    m.add("x2", p).unwrap();
    let hr = image::open("data/test_media/a128_hr.png").unwrap().to_rgb8();
    let lr = image::imageops::resize(&hr, 64, 64, image::imageops::FilterType::CatmullRom);
    let sr = m.upscale("x2", &lr, None).unwrap();
    assert_eq!((sr.width(), sr.height()), (128, 128));
    let bic = image::imageops::resize(&lr, 128, 128, image::imageops::FilterType::CatmullRom);
    let psnr = |a: &image::RgbImage| {
        let mse: f64 = a.as_raw().iter().zip(hr.as_raw()).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / hr.as_raw().len() as f64;
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    let (ps, pb) = (psnr(&sr), psnr(&bic));
    eprintln!("PSNR LoopSR {ps:.2} dB vs bicubic {pb:.2} dB");
    assert!(ps > pb + 3.0);
}

#[test]
fn bicubic_constant_is_identity() {
    let out = aido::upscale::bicubic(&vec![0.25; 5 * 7 * 3], 5, 7, 2);
    assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-5));
}
