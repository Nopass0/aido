//! Hugging Face Hub downloader: `aido pull <repo>` and `hf:` model paths.
//!
//! * Lists the repository files through the public API.
//! * For GGUF repos picks one file (preferring `Q4_K_M`, then `Q4_K_S`,
//!   `Q8_0`, any `.gguf`) unless a pattern is given; for other repos
//!   (diffusers / safetensors) downloads every file matching the pattern.
//! * Downloads with resume (HTTP range) into `models/<repo-name>/`.
//! * A model argument `hf:org/repo[:pattern]` anywhere in the CLI is
//!   resolved through [`resolve`], so `aido run -m hf:Qwen/Qwen3-0.6B-GGUF`
//!   just works.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Simple glob: `*` matches any run of characters; case-insensitive.
///
/// # Example
/// ```
/// assert!(aido::hub::glob("*q4_k_m*.gguf", "Qwen3-8B-Q4_K_M.gguf"));
/// assert!(!aido::hub::glob("*.json", "model.safetensors"));
/// ```
pub fn glob(pat: &str, s: &str) -> bool {
    let (p, s) = (pat.to_lowercase(), s.to_lowercase());
    let parts: Vec<&str> = p.split('*').collect();
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match s[pos..].find(part) {
            Some(f) if i > 0 || f == 0 => pos += f + part.len(),
            _ => return false,
        }
    }
    p.ends_with('*') || pos == s.len()
}

/// Adds `Authorization: Bearer $HF_TOKEN` when the variable is set
/// (private / gated repositories).
fn auth(r: ureq::Request) -> ureq::Request {
    match std::env::var("HF_TOKEN") {
        Ok(t) if !t.is_empty() => r.set("Authorization", &format!("Bearer {t}")),
        _ => r,
    }
}

/// Lists file paths of a model repository.
///
/// # Example
/// ```no_run
/// let files = aido::hub::list_files("Qwen/Qwen3-0.6B-GGUF").unwrap();
/// ```
pub fn list_files(repo: &str) -> Result<Vec<(String, u64)>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let body = auth(ureq::get(&url).set("User-Agent", "aido")).call().with_context(|| format!("repository {repo} not found or private"))?.into_string()?;
    let j: serde_json::Value = serde_json::from_str(&body)?;
    Ok(j.as_array()
        .context("unexpected API answer")?
        .iter()
        .filter(|f| f["type"] == "file")
        .map(|f| (f["path"].as_str().unwrap_or("").to_string(), f["size"].as_u64().unwrap_or(0)))
        .collect())
}

/// Chooses which files to download.
///
/// # Example
/// ```
/// let f = vec![("a-Q8_0.gguf".to_string(), 1), ("a-Q4_K_M.gguf".to_string(), 1)];
/// assert_eq!(aido::hub::select(&f, None), vec!["a-Q4_K_M.gguf".to_string()]);
/// ```
pub fn select(files: &[(String, u64)], pattern: Option<&str>) -> Vec<String> {
    if let Some(p) = pattern {
        return files.iter().filter(|f| glob(p, &f.0)).map(|f| f.0.clone()).collect();
    }
    let gguf: Vec<&String> = files.iter().map(|f| &f.0).filter(|f| f.to_lowercase().ends_with(".gguf")).collect();
    if !gguf.is_empty() {
        for pref in ["q4_k_m", "q4_k_s", "q8_0", ""] {
            if let Some(f) = gguf.iter().find(|f| f.to_lowercase().contains(pref)) {
                return vec![(*f).clone()];
            }
        }
    }
    files.iter().map(|f| f.0.clone()).filter(|f| !f.ends_with(".bin") && !f.ends_with(".pt") && !f.ends_with(".onnx") && !f.contains(".fp32.")).collect()
}

/// Downloads one file with resume and a progress line.
///
/// # Example
/// ```no_run
/// aido::hub::download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", std::path::Path::new("models/x.gguf")).unwrap();
/// ```
pub fn download(repo: &str, file: &str, dst: &Path) -> Result<()> {
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p)?;
    }
    let have = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let mut req = auth(ureq::get(&url).set("User-Agent", "aido"));
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(416, _)) => return Ok(()), // already complete
        Err(e) => bail!("download {file}: {e}"),
    };
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).map(|v| v + if resp.status() == 206 { have } else { 0 });
    let append = resp.status() == 206;
    let mut out = std::fs::OpenOptions::new().create(true).append(append).write(true).truncate(!append).open(dst)?;
    let mut rd = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = if append { have } else { 0 };
    let t = std::time::Instant::now();
    loop {
        let n = rd.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if let Some(tot) = total {
            eprint!("\r  {file}: {:.1}/{:.1} MB ({:.0} MB/s)   ", done as f64 / 1e6, tot as f64 / 1e6, done as f64 / 1e6 / t.elapsed().as_secs_f64().max(1e-3));
        }
    }
    eprintln!();
    Ok(())
}

/// Downloads a repository (or selected files) into `models/`. Returns the
/// path to use as a model argument (a single file or the directory).
///
/// # Example
/// ```no_run
/// let p = aido::hub::pull("Qwen/Qwen3-0.6B-GGUF", None, "models").unwrap();
/// ```
pub fn pull(repo: &str, pattern: Option<&str>, root: &str) -> Result<PathBuf> {
    let files = list_files(repo)?;
    let sel = select(&files, pattern);
    anyhow::ensure!(!sel.is_empty(), "no files match");
    let dir = Path::new(root).join(repo.replace('/', "--"));
    let total: u64 = files.iter().filter(|f| sel.contains(&f.0)).map(|f| f.1).sum();
    eprintln!("[pull] {repo}: {} file(s), {:.2} GB → {}", sel.len(), total as f64 / 1e9, dir.display());
    for f in &sel {
        download(repo, f, &dir.join(f))?;
    }
    Ok(if sel.len() == 1 { dir.join(&sel[0]) } else { dir })
}

/// Resolves `hf:org/repo[:pattern]` to a local path (downloading if
/// needed); other strings are returned unchanged.
///
/// # Example
/// ```
/// assert_eq!(aido::hub::resolve("models/x.gguf").unwrap(), "models/x.gguf");
/// ```
pub fn resolve(spec: &str) -> Result<String> {
    let Some(rest) = spec.strip_prefix("hf:") else { return Ok(spec.to_string()) };
    let (repo, pat) = match rest.split_once(':') {
        Some((r, p)) => (r, Some(p)),
        None => (rest, None),
    };
    Ok(pull(repo, pat, "models")?.to_string_lossy().into_owned())
}
