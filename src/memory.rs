//! Long-term memory: a persistent per-user store with **hybrid vector +
//! keyword retrieval**.
//!
//! Each user (the OpenAI `user` field, or any id you choose) has a JSON-lines
//! file `<dir>/<user>.jsonl` (the facts) and `<dir>/<user>.vec` (their
//! vectors, see [`crate::vector`]). Retrieval scores every memory by
//! `0.6 · cosine(embedding) + 0.4 · BM25 / max(BM25)`: the vector part finds
//! paraphrases and other word forms, the keyword part pins exact names and
//! numbers. Before a chat turn the best memories are given to the model as a
//! system note, and the model can also search and save memories itself
//! through the built-in tools [`MEMORY_TOOLS`] (`memory_search`,
//! `memory_save`), so it can look things up as many times as it needs while
//! reasoning instead of carrying everything in its context window.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::vector::{Embedder, HashEmbedder, VectorIndex};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// One stored memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryItem {
    /// Unique id (monotonic per store).
    pub id: u64,
    /// Unix seconds when stored.
    pub created: u64,
    /// The remembered text.
    pub text: String,
}

/// Persistent per-user memory store.
pub struct MemoryStore {
    dir: PathBuf,
    lock: Mutex<()>,
    embedder: Box<dyn Embedder>,
    cache: Mutex<HashMap<String, VectorIndex>>,
}

/// OpenAI-format definitions of the built-in memory tools the model may call
/// (executed by the engine, never returned to the client).
pub const MEMORY_TOOLS: &str = r#"[
 {"type":"function","function":{"name":"memory_search","description":"Search the long-term memory about this user (facts, preferences, earlier conversations). Use it whenever the answer may depend on something said before.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"what to look for"}},"required":["query"]}}},
 {"type":"function","function":{"name":"memory_save","description":"Save an important long-lived fact about the user or the task to long-term memory.","parameters":{"type":"object","properties":{"text":{"type":"string","description":"the fact, one sentence"}},"required":["text"]}}}
]"#;

/// Splits text into lower-cased word tokens (any script).
///
/// # Example
/// ```
/// assert_eq!(aido::memory::words("Привет, World 42!"), vec!["привет", "world", "42"]);
/// ```
pub fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl MemoryStore {
    /// Opens (creates) a store rooted at `dir`.
    ///
    /// # Example
    /// ```
    /// let m = aido::memory::MemoryStore::new(std::env::temp_dir().join("aido_mem_doc")).unwrap();
    /// ```
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(MemoryStore { dir, lock: Mutex::new(()), embedder: Box::new(HashEmbedder::default()), cache: Mutex::new(HashMap::new()) })
    }

    /// Replaces the embedder (e.g. a neural one); cached vectors are rebuilt.
    pub fn with_embedder(mut self, e: Box<dyn Embedder>) -> Self {
        self.embedder = e;
        self.cache.lock().unwrap().clear();
        self
    }

    /// The vector index of `user`, (re)built when stale.
    fn index(&self, user: &str, items: &[MemoryItem]) -> VectorIndex {
        let mut c = self.cache.lock().unwrap();
        let fresh = c.get(user).map(|i| i.rows.len() == items.len() && i.embedder == self.embedder.id() && i.rows.iter().zip(items).all(|(r, it)| r.id == it.id)).unwrap_or(false);
        if !fresh {
            let path = self.file(user).with_extension("vec");
            let mut idx = VectorIndex::load(&path).ok().filter(|i| i.embedder == self.embedder.id()).unwrap_or_else(|| VectorIndex::new(&self.embedder.id(), self.embedder.dim()));
            idx.rows.retain(|r| items.iter().any(|it| it.id == r.id));
            for it in items {
                if !idx.rows.iter().any(|r| r.id == it.id) {
                    idx.upsert(it.id, &it.text, self.embedder.embed(&it.text));
                }
            }
            idx.rows.sort_by_key(|r| items.iter().position(|it| it.id == r.id));
            let _ = idx.save(&path);
            c.insert(user.to_string(), idx);
        }
        c[user].clone()
    }

    fn file(&self, user: &str) -> PathBuf {
        let safe: String = user.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).take(64).collect();
        self.dir.join(format!("{}.jsonl", if safe.is_empty() { "default" } else { &safe }))
    }

    /// All memories of `user`, oldest first.
    ///
    /// # Example
    /// ```
    /// # let m = aido::memory::MemoryStore::new(std::env::temp_dir().join("aido_mem_doc2")).unwrap();
    /// let all = m.list("alice").unwrap();
    /// ```
    pub fn list(&self, user: &str) -> Result<Vec<MemoryItem>> {
        let txt = std::fs::read_to_string(self.file(user)).unwrap_or_default();
        Ok(txt.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
    }

    /// Stores `text` for `user`; returns the new item.
    ///
    /// # Example
    /// ```
    /// # let m = aido::memory::MemoryStore::new(std::env::temp_dir().join("aido_mem_doc3")).unwrap();
    /// let it = m.add("bob", "Bob's favourite colour is green").unwrap();
    /// assert!(m.list("bob").unwrap().iter().any(|x| x.id == it.id));
    /// ```
    pub fn add(&self, user: &str, text: &str) -> Result<MemoryItem> {
        use std::io::Write;
        let _g = self.lock.lock().unwrap();
        let id = self.list(user)?.iter().map(|x| x.id).max().map(|m| m + 1).unwrap_or(1);
        let it = MemoryItem { id, created: now(), text: text.trim().to_string() };
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.file(user))?;
        writeln!(f, "{}", serde_json::to_string(&it)?)?;
        Ok(it)
    }

    /// Deletes item `id` of `user` (`None` deletes everything). Returns how
    /// many items were removed.
    ///
    /// # Example
    /// ```
    /// # let m = aido::memory::MemoryStore::new(std::env::temp_dir().join("aido_mem_doc4")).unwrap();
    /// m.delete("carol", None).unwrap();
    /// ```
    pub fn delete(&self, user: &str, id: Option<u64>) -> Result<usize> {
        let _g = self.lock.lock().unwrap();
        let all = self.list(user)?;
        let keep: Vec<&MemoryItem> = all.iter().filter(|x| id.map(|i| x.id != i).unwrap_or(false)).collect();
        let removed = all.len() - keep.len();
        let body: String = keep.iter().map(|x| serde_json::to_string(x).unwrap() + "\n").collect();
        std::fs::write(self.file(user), body)?;
        Ok(removed)
    }

    /// The `k` memories of `user` most relevant to `query`, best first:
    /// hybrid of embedding cosine (0.6) and BM25 (0.4, k1 = 1.2, b = 0.75,
    /// normalised by the best BM25 score). Entries scoring below 0.12 are
    /// dropped.
    ///
    /// # Example
    /// ```
    /// let m = aido::memory::MemoryStore::new(std::env::temp_dir().join("aido_mem_doc5")).unwrap();
    /// m.delete("dave", None).unwrap();
    /// m.add("dave", "Dave lives in Kazan").unwrap();
    /// m.add("dave", "Dave has a cat named Murzik").unwrap();
    /// let hits = m.search("dave", "what is the name of my cat?", 1).unwrap();
    /// assert!(hits[0].text.contains("Murzik"));
    /// ```
    pub fn search(&self, user: &str, query: &str, k: usize) -> Result<Vec<MemoryItem>> {
        let all = self.list(user)?;
        if all.is_empty() {
            return Ok(vec![]);
        }
        let docs: Vec<Vec<String>> = all.iter().map(|x| words(&x.text)).collect();
        let avg = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / docs.len() as f64;
        let n = docs.len() as f64;
        let q = words(query);
        let bm: Vec<f64> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut s = 0.0;
                for w in &q {
                    let tf = d.iter().filter(|x| *x == w).count() as f64;
                    if tf == 0.0 {
                        continue;
                    }
                    let df = docs.iter().filter(|dd| dd.contains(w)).count() as f64;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * d.len() as f64 / avg.max(1.0)));
                }
                let _ = i;
                s
            })
            .collect();
        let bmax = bm.iter().cloned().fold(0.0, f64::max).max(1e-9);
        let idx = self.index(user, &all);
        let qv = self.embedder.embed(query);
        let mut scored: Vec<(f64, usize)> = idx.rows.iter().enumerate().map(|(i, r)| (0.6 * crate::vector::cosine(&qv, &r.vector) as f64 + 0.4 * bm[i] / bmax, i)).filter(|x| x.0 > 0.12).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(scored.into_iter().take(k).map(|(_, i)| all[i].clone()).collect())
    }
}
