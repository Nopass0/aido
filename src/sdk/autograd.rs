//! Reverse-mode automatic differentiation on 2-D tensors (define-by-run).
//!
//! A [`Graph`] records operations as they are executed; [`Graph::backward`]
//! walks the tape in reverse and accumulates gradients into the
//! [`Params`] store. Every tensor is a row-major matrix `rows × cols`
//! (a batch of sequences is flattened to `batch·seq` rows), which keeps all
//! heavy operations on the engine's packed GEMM.
//!
//! Supported ops: linear (`x·Wᵀ`, optional LoRA adapter), add, bias, mul,
//! scale, ReLU, SiLU, GELU, tanh, SwiGLU, RMSNorm, embedding, RoPE, fused
//! causal multi-head attention, softmax cross-entropy and MSE.

use crate::runtime::pool::ThreadPool;
use crate::tensor::gemm::{sgemm, PackedB};
use crate::tensor::ops::Rope;

/// Trainable (or frozen) parameters with gradients.
#[derive(Default, Clone)]
pub struct Params {
    /// Parameter names.
    pub names: Vec<String>,
    /// Shapes `(rows, cols)`.
    pub shapes: Vec<(usize, usize)>,
    /// Values.
    pub data: Vec<Vec<f32>>,
    /// Accumulated gradients.
    pub grads: Vec<Vec<f32>>,
    /// `false` for frozen parameters (no gradient kept).
    pub trainable: Vec<bool>,
}

impl Params {
    /// Adds a parameter and returns its index.
    ///
    /// # Example
    /// ```
    /// let mut p = aido::sdk::autograd::Params::default();
    /// let id = p.add("w", 2, 3, vec![0.0; 6], true);
    /// assert_eq!(p.find("w"), Some(id));
    /// ```
    pub fn add(&mut self, name: &str, rows: usize, cols: usize, data: Vec<f32>, trainable: bool) -> usize {
        assert_eq!(data.len(), rows * cols, "param {name} size");
        self.names.push(name.to_string());
        self.shapes.push((rows, cols));
        self.grads.push(if trainable { vec![0.0; data.len()] } else { Vec::new() });
        self.data.push(data);
        self.trainable.push(trainable);
        self.names.len() - 1
    }

    /// Index of a parameter by name.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::sdk::autograd::Params::default().find("x"), None);
    /// ```
    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Clears all gradients.
    ///
    /// # Example
    /// ```
    /// let mut p = aido::sdk::autograd::Params::default();
    /// p.add("w", 1, 1, vec![1.0], true);
    /// p.grads[0][0] = 5.0;
    /// p.zero_grad();
    /// assert_eq!(p.grads[0][0], 0.0);
    /// ```
    pub fn zero_grad(&mut self) {
        for g in &mut self.grads {
            g.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Number of trainable scalars.
    ///
    /// # Example
    /// ```
    /// let mut p = aido::sdk::autograd::Params::default();
    /// p.add("w", 2, 2, vec![0.0; 4], true);
    /// p.add("f", 2, 2, vec![0.0; 4], false);
    /// assert_eq!(p.n_trainable(), 4);
    /// ```
    pub fn n_trainable(&self) -> usize {
        self.data.iter().zip(&self.trainable).filter(|(_, t)| **t).map(|(d, _)| d.len()).sum()
    }
}

/// Node identifier inside a graph.
pub type Id = usize;

/// Recorded operation.
enum Op {
    Input,
    Param(usize),
    Linear { x: Id, w: usize, b: Option<usize>, lora: Option<(usize, usize, f32)> },
    Add(Id, Id),
    Mul(Id, Id),
    Scale(Id, f32),
    Relu(Id),
    Silu(Id),
    Tanh(Id),
    Gelu(Id),
    SwiGlu(Id, Id),
    RmsNorm { x: Id, g: usize, eps: f32, inv: Vec<f32> },
    Embed { table: usize, ids: Vec<u32> },
    Rope { x: Id, heads: usize, seq: usize, base: f32, neox: bool },
    Reshape(Id),
    Attn { q: Id, k: Id, v: Id, heads: usize, seq: usize, probs: Vec<f32> },
    SoftmaxCe { logits: Id, targets: Vec<u32>, probs: Vec<f32>, count: usize },
    Mse { a: Id, target: Vec<f32> },
    ConcatCols(Id, Id),
    Conv { x: Id, w: usize, b: Option<usize>, n: usize, h: usize, wd: usize, stride: usize, col: Vec<f32> },
    Up2 { x: Id, n: usize, h: usize, w: usize },
    AddImg { x: Id, v: Id, n: usize },
    Bce { a: Id, target: Vec<f32>, weight: Vec<f32> },
    Gather { x: Id, idx: Vec<usize> },
    Ctc { logits: Id, grad: Vec<f32> },
}

struct Node {
    rows: usize,
    cols: usize,
    val: Vec<f32>,
    grad: Vec<f32>,
    op: Op,
}

/// A computation tape.
pub struct Graph<'a> {
    nodes: Vec<Node>,
    /// Parameter store the graph reads from / writes gradients to.
    pub params: &'a mut Params,
    pool: &'a ThreadPool,
}

/// `y[m×n] = x[m×k] · W[n×k]ᵀ` on the packed GEMM.
fn mm_xwt(pool: &ThreadPool, x: &[f32], m: usize, k: usize, w: &[f32], n: usize, out: &mut [f32], acc: bool) {
    let pb = PackedB::from_nk(w, n, k);
    sgemm(pool, m, x, k, &pb, None, out, n, acc);
}

/// `y[m×n] = a[m×k] · b[k×n]`.
fn mm_ab(pool: &ThreadPool, a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32], acc: bool) {
    let pb = PackedB::from_kn(b, k, n);
    sgemm(pool, m, a, k, &pb, None, out, n, acc);
}

/// Transpose of a `r × c` matrix.
fn transpose(a: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut t = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

impl<'a> Graph<'a> {
    /// Starts a new tape.
    ///
    /// # Example
    /// ```
    /// use aido::sdk::autograd::{Graph, Params};
    /// let pool = aido::runtime::pool::ThreadPool::new(2, None);
    /// let mut p = Params::default();
    /// let g = Graph::new(&mut p, &pool);
    /// ```
    pub fn new(params: &'a mut Params, pool: &'a ThreadPool) -> Self {
        Graph { nodes: Vec::new(), params, pool }
    }

    fn push(&mut self, rows: usize, cols: usize, val: Vec<f32>, op: Op) -> Id {
        debug_assert_eq!(val.len(), rows * cols);
        self.nodes.push(Node { rows, cols, val, grad: Vec::new(), op });
        self.nodes.len() - 1
    }

    /// Value of a node.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 2, vec![1.0, 2.0]);
    /// assert_eq!(g.value(x), &[1.0, 2.0]);
    /// ```
    pub fn value(&self, id: Id) -> &[f32] {
        &self.nodes[id].val
    }

    /// `(rows, cols)` of a node.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(3, 2, vec![0.0; 6]);
    /// assert_eq!(g.shape(x), (3, 2));
    /// ```
    pub fn shape(&self, id: Id) -> (usize, usize) {
        (self.nodes[id].rows, self.nodes[id].cols)
    }

    /// Constant input (no gradient flows into it).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 1, vec![3.0]);
    /// ```
    pub fn input(&mut self, rows: usize, cols: usize, val: Vec<f32>) -> Id {
        self.push(rows, cols, val, Op::Input)
    }

    /// A parameter used directly as a tensor (e.g. a learned bias matrix).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut p = Params::default();
    /// let w = p.add("w", 1, 2, vec![1.0, 2.0], true);
    /// let mut g = Graph::new(&mut p, &pool);
    /// let n = g.param(w);
    /// ```
    pub fn param(&mut self, pid: usize) -> Id {
        let (r, c) = self.params.shapes[pid];
        let v = self.params.data[pid].clone();
        self.push(r, c, v, Op::Param(pid))
    }

    /// Linear layer `x·Wᵀ (+ b) (+ scale·(x·Aᵀ)·Bᵀ)` where `W` is `[out, in]`
    /// and the optional LoRA pair is `A: [r, in]`, `B: [out, r]`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut p = Params::default();
    /// let w = p.add("w", 1, 2, vec![1.0, 1.0], true);
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 2, vec![2.0, 3.0]);
    /// let y = g.linear(x, w, None, None);
    /// assert_eq!(g.value(y), &[5.0]);
    /// ```
    pub fn linear(&mut self, x: Id, w: usize, b: Option<usize>, lora: Option<(usize, usize, f32)>) -> Id {
        let (m, k) = self.shape(x);
        let (n, k2) = self.params.shapes[w];
        assert_eq!(k, k2, "linear: input width {k} vs weight {k2} ({})", self.params.names[w]);
        let mut y = vec![0f32; m * n];
        mm_xwt(self.pool, &self.nodes[x].val, m, k, &self.params.data[w], n, &mut y, false);
        if let Some(bi) = b {
            let bias = &self.params.data[bi];
            for row in y.chunks_exact_mut(n) {
                for (v, bb) in row.iter_mut().zip(bias) {
                    *v += bb;
                }
            }
        }
        if let Some((a, bb, s)) = lora {
            let r = self.params.shapes[a].0;
            let mut h = vec![0f32; m * r];
            mm_xwt(self.pool, &self.nodes[x].val, m, k, &self.params.data[a], r, &mut h, false);
            h.iter_mut().for_each(|v| *v *= s);
            mm_xwt(self.pool, &h, m, r, &self.params.data[bb], n, &mut y, true);
        }
        self.push(m, n, y, Op::Linear { x, w, b, lora })
    }

    fn unary(&mut self, a: Id, f: impl Fn(f32) -> f32, op: Op) -> Id {
        let (r, c) = self.shape(a);
        let v = self.nodes[a].val.iter().map(|x| f(*x)).collect();
        self.push(r, c, v, op)
    }

    /// Element-wise sum (shapes must match).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![1.0]);
    /// let s = g.add(a, a);
    /// assert_eq!(g.value(s), &[2.0]);
    /// ```
    pub fn add(&mut self, a: Id, b: Id) -> Id {
        let (r, c) = self.shape(a);
        assert_eq!((r, c), self.shape(b), "add: shape mismatch");
        let v = self.nodes[a].val.iter().zip(&self.nodes[b].val).map(|(x, y)| x + y).collect();
        self.push(r, c, v, Op::Add(a, b))
    }

    /// Element-wise product.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![3.0]);
    /// let s = g.mul(a, a);
    /// assert_eq!(g.value(s), &[9.0]);
    /// ```
    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        let (r, c) = self.shape(a);
        let v = self.nodes[a].val.iter().zip(&self.nodes[b].val).map(|(x, y)| x * y).collect();
        self.push(r, c, v, Op::Mul(a, b))
    }

    /// Multiplication by a constant.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![3.0]);
    /// let s = g.scale(a, 2.0);
    /// assert_eq!(g.value(s), &[6.0]);
    /// ```
    pub fn scale(&mut self, a: Id, s: f32) -> Id {
        self.unary(a, |x| x * s, Op::Scale(a, s))
    }

    /// ReLU.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 2, vec![-1.0, 2.0]);
    /// let r = g.relu(a);
    /// assert_eq!(g.value(r), &[0.0, 2.0]);
    /// ```
    pub fn relu(&mut self, a: Id) -> Id {
        self.unary(a, |x| x.max(0.0), Op::Relu(a))
    }

    /// SiLU `x·σ(x)`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![0.0]);
    /// let r = g.silu(a);
    /// assert_eq!(g.value(r), &[0.0]);
    /// ```
    pub fn silu(&mut self, a: Id) -> Id {
        self.unary(a, |x| x / (1.0 + (-x).exp()), Op::Silu(a))
    }

    /// tanh.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![0.0]);
    /// let r = g.tanh(a);
    /// assert_eq!(g.value(r), &[0.0]);
    /// ```
    pub fn tanh(&mut self, a: Id) -> Id {
        self.unary(a, |x| x.tanh(), Op::Tanh(a))
    }

    /// GELU (tanh approximation).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![0.0]);
    /// let r = g.gelu(a);
    /// assert_eq!(g.value(r), &[0.0]);
    /// ```
    pub fn gelu(&mut self, a: Id) -> Id {
        self.unary(a, |u| 0.5 * u * (1.0 + (0.797_884_6 * (u + 0.044715 * u * u * u)).tanh()), Op::Gelu(a))
    }

    /// SwiGLU `silu(g) · u`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![0.0]);
    /// let r = g.swiglu(a, a);
    /// assert_eq!(g.value(r), &[0.0]);
    /// ```
    pub fn swiglu(&mut self, gate: Id, up: Id) -> Id {
        let (r, c) = self.shape(gate);
        let v = self.nodes[gate].val.iter().zip(&self.nodes[up].val).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
        self.push(r, c, v, Op::SwiGlu(gate, up))
    }

    /// RMSNorm over rows with learned gain parameter `g` (`1 × cols`).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut p = Params::default();
    /// let gain = p.add("g", 1, 2, vec![1.0, 1.0], true);
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 2, vec![2.0, 2.0]);
    /// let y = g.rmsnorm(x, gain, 0.0);
    /// assert!((g.value(y)[0] - 1.0).abs() < 1e-6);
    /// ```
    pub fn rmsnorm(&mut self, x: Id, gain: usize, eps: f32) -> Id {
        let (r, c) = self.shape(x);
        let mut inv = vec![0f32; r];
        let mut y = vec![0f32; r * c];
        let g = &self.params.data[gain];
        for i in 0..r {
            let row = &self.nodes[x].val[i * c..(i + 1) * c];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
            inv[i] = 1.0 / (ms + eps).sqrt();
            for j in 0..c {
                y[i * c + j] = row[j] * inv[i] * g[j];
            }
        }
        self.push(r, c, y, Op::RmsNorm { x, g: gain, eps, inv })
    }

    /// Embedding lookup: rows of parameter `table` selected by `ids`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut p = Params::default();
    /// let t = p.add("emb", 2, 1, vec![5.0, 7.0], true);
    /// let mut g = Graph::new(&mut p, &pool);
    /// let e = g.embed(t, &[1, 0]);
    /// assert_eq!(g.value(e), &[7.0, 5.0]);
    /// ```
    pub fn embed(&mut self, table: usize, ids: &[u32]) -> Id {
        let (_, c) = self.params.shapes[table];
        let t = &self.params.data[table];
        let mut y = vec![0f32; ids.len() * c];
        for (i, &id) in ids.iter().enumerate() {
            y[i * c..(i + 1) * c].copy_from_slice(&t[id as usize * c..(id as usize + 1) * c]);
        }
        self.push(ids.len(), c, y, Op::Embed { table, ids: ids.to_vec() })
    }

    /// Rotary embedding (GPT-J / Llama pair layout) on `heads` heads; rows
    /// are sequences of length `seq` (position = row % seq).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 2, vec![1.0, 0.0]);
    /// let y = g.rope(x, 1, 1, 10000.0);
    /// assert_eq!(g.value(y), &[1.0, 0.0]); // position 0 = identity
    /// ```
    pub fn rope(&mut self, x: Id, heads: usize, seq: usize, base: f32) -> Id {
        self.rope_ex(x, heads, seq, base, false)
    }

    /// RoPE with explicit layout: `neox = true` rotates halves (Qwen,
    /// NeoX), `false` rotates adjacent pairs (Llama GGUF layout).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 2, vec![1.0, 0.0]);
    /// let y = g.rope_ex(x, 1, 1, 10000.0, true);
    /// assert_eq!(g.value(y), &[1.0, 0.0]);
    /// ```
    pub fn rope_ex(&mut self, x: Id, heads: usize, seq: usize, base: f32, neox: bool) -> Id {
        let (r, c) = self.shape(x);
        let hd = c / heads;
        let rope = Rope::new(hd, base, None, neox);
        let mut y = self.nodes[x].val.clone();
        let mut cs = vec![0f32; hd];
        for i in 0..r {
            rope.table(i % seq, &mut cs);
            rope.apply(&mut y[i * c..(i + 1) * c], hd, &cs);
        }
        self.push(r, c, y, Op::Rope { x, heads, seq, base, neox })
    }

    /// Reinterprets a node as `rows × cols` (same element count, no copy of
    /// semantics — e.g. `[t, heads·hd] → [t·heads, hd]` for per-head norms).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
    /// let y = g.reshape(x, 2, 2);
    /// assert_eq!(g.shape(y), (2, 2));
    /// ```
    pub fn reshape(&mut self, x: Id, rows: usize, cols: usize) -> Id {
        assert_eq!(rows * cols, self.nodes[x].val.len(), "reshape size");
        let v = self.nodes[x].val.clone();
        self.push(rows, cols, v, Op::Reshape(x))
    }

    /// Fused causal multi-head attention for sequences of length `seq`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let y = g.causal_attention(x, x, x, 1, 2);
    /// assert_eq!(&g.value(y)[..2], &[1.0, 0.0]); // first token sees only itself
    /// ```
    pub fn causal_attention(&mut self, q: Id, k: Id, v: Id, heads: usize, seq: usize) -> Id {
        let (r, c) = self.shape(q);
        let hd = c / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let nseq = r / seq;
        let mut probs = vec![0f32; nseq * heads * seq * seq];
        let mut out = vec![0f32; r * c];
        let (qv, kv, vv) = (&self.nodes[q].val, &self.nodes[k].val, &self.nodes[v].val);
        let pp = crate::tensor::SyncPtr::new(&mut probs);
        let op = crate::tensor::SyncPtr::new(&mut out);
        self.pool.parallel_for(nseq * heads, |task, _| {
            let (s, h) = (task / heads, task % heads);
            let pr = unsafe { pp.slice(task * seq * seq, seq * seq) };
            for i in 0..seq {
                let qi = &qv[(s * seq + i) * c + h * hd..(s * seq + i) * c + (h + 1) * hd];
                let row = &mut pr[i * seq..(i + 1) * seq];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    let kj = &kv[(s * seq + j) * c + h * hd..(s * seq + j) * c + (h + 1) * hd];
                    row[j] = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                    mx = mx.max(row[j]);
                }
                let mut sum = 0.0;
                for j in 0..=i {
                    row[j] = (row[j] - mx).exp();
                    sum += row[j];
                }
                for j in 0..=i {
                    row[j] /= sum;
                }
                let o = unsafe { op.slice((s * seq + i) * c + h * hd, hd) };
                for j in 0..=i {
                    let vj = &vv[(s * seq + j) * c + h * hd..(s * seq + j) * c + (h + 1) * hd];
                    for e in 0..hd {
                        o[e] += row[j] * vj[e];
                    }
                }
            }
        });
        self.push(r, c, out, Op::Attn { q, k, v, heads, seq, probs })
    }

    /// Concatenates columns `[a | b]` (same row count).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 1, vec![1.0]);
    /// let c = g.concat_cols(a, a);
    /// assert_eq!(g.value(c), &[1.0, 1.0]);
    /// ```
    pub fn concat_cols(&mut self, a: Id, b: Id) -> Id {
        let (r, ca) = self.shape(a);
        let (_, cb) = self.shape(b);
        let mut v = vec![0f32; r * (ca + cb)];
        for i in 0..r {
            v[i * (ca + cb)..i * (ca + cb) + ca].copy_from_slice(&self.nodes[a].val[i * ca..(i + 1) * ca]);
            v[i * (ca + cb) + ca..(i + 1) * (ca + cb)].copy_from_slice(&self.nodes[b].val[i * cb..(i + 1) * cb]);
        }
        self.push(r, ca + cb, v, Op::ConcatCols(a, b))
    }

    /// 3×3 convolution (padding 1, stride 1 or 2) over a batch of `n` NHWC
    /// images of `h × wd` pixels: rows = `n·h·wd` pixels, cols = channels.
    /// Weight `w` is `[cout, 9·cin]` with K ordered `(ky, kx, cin)`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut p = Params::default();
    /// let mut k = vec![0.0f32; 9];
    /// k[4] = 1.0; // centre tap = identity
    /// let w = p.add("w", 1, 9, k, true);
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    /// let y = g.conv3x3(x, w, None, 1, 2, 2, 1);
    /// assert_eq!(g.value(y), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn conv3x3(&mut self, x: Id, w: usize, b: Option<usize>, n: usize, h: usize, wd: usize, stride: usize) -> Id {
        let cin = self.nodes[x].cols;
        let (cout, k) = self.params.shapes[w];
        assert_eq!(k, 9 * cin, "conv3x3: weight K {k} != 9·{cin}");
        let (ho, wo) = (h.div_ceil(stride), wd.div_ceil(stride));
        let rows = n * ho * wo;
        let mut col = vec![0f32; rows * k];
        let xv = &self.nodes[x].val;
        let cp = crate::tensor::SyncPtr::new(&mut col);
        self.pool.parallel_chunks(rows, 64, |a, bnd, _| {
            for r in a..bnd {
                let (img, rem) = (r / (ho * wo), r % (ho * wo));
                let (oy, ox) = (rem / wo, rem % wo);
                let dst = unsafe { cp.slice(r * k, k) };
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = (oy * stride + ky) as isize - 1;
                        let ix = (ox * stride + kx) as isize - 1;
                        let d = &mut dst[(ky * 3 + kx) * cin..(ky * 3 + kx + 1) * cin];
                        if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                            d.fill(0.0);
                        } else {
                            let s = ((img * h + iy as usize) * wd + ix as usize) * cin;
                            d.copy_from_slice(&xv[s..s + cin]);
                        }
                    }
                }
            }
        });
        let mut y = vec![0f32; rows * cout];
        mm_xwt(self.pool, &col, rows, k, &self.params.data[w], cout, &mut y, false);
        if let Some(bi) = b {
            for row in y.chunks_exact_mut(cout) {
                for (v, bb) in row.iter_mut().zip(&self.params.data[bi]) {
                    *v += bb;
                }
            }
        }
        self.push(rows, cout, y, Op::Conv { x, w, b, n, h, wd, stride, col })
    }

    /// Nearest-neighbour 2× upsampling of `n` NHWC images of `h × w`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(1, 1, vec![5.0]);
    /// let y = g.upsample2x(x, 1, 1, 1);
    /// assert_eq!(g.value(y), &[5.0; 4]);
    /// ```
    pub fn upsample2x(&mut self, x: Id, n: usize, h: usize, w: usize) -> Id {
        let c = self.nodes[x].cols;
        let mut y = vec![0f32; n * 4 * h * w * c];
        let xv = &self.nodes[x].val;
        for img in 0..n {
            for oy in 0..2 * h {
                for ox in 0..2 * w {
                    let s = ((img * h + oy / 2) * w + ox / 2) * c;
                    let d = ((img * 2 * h + oy) * 2 * w + ox) * c;
                    y[d..d + c].copy_from_slice(&xv[s..s + c]);
                }
            }
        }
        self.push(n * 4 * h * w, c, y, Op::Up2 { x, n, h, w })
    }

    /// Adds a per-image vector `v` (`n × c`) to every pixel of image `i`
    /// in `x` (`n·hw × c`) — e.g. a timestep / class embedding.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(2, 1, vec![1.0, 2.0]);
    /// let v = g.input(1, 1, vec![10.0]);
    /// let y = g.add_per_image(x, v, 1);
    /// assert_eq!(g.value(y), &[11.0, 12.0]);
    /// ```
    pub fn add_per_image(&mut self, x: Id, v: Id, n: usize) -> Id {
        let (r, c) = self.shape(x);
        let per = r / n;
        let mut y = self.nodes[x].val.clone();
        for i in 0..r {
            let img = i / per;
            for j in 0..c {
                y[i * c + j] += self.nodes[v].val[img * c + j];
            }
        }
        self.push(r, c, y, Op::AddImg { x, v, n })
    }

    /// Row gather: output row `i` = input row `idx[i]` (permutations,
    /// e.g. turning `(img, y, x)` pixel order into `(img, x, y)`).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let x = g.input(2, 1, vec![1.0, 2.0]);
    /// let y = g.gather_rows(x, &[1, 0]);
    /// assert_eq!(g.value(y), &[2.0, 1.0]);
    /// ```
    pub fn gather_rows(&mut self, x: Id, idx: &[usize]) -> Id {
        let c = self.nodes[x].cols;
        let mut v = vec![0f32; idx.len() * c];
        for (i, &j) in idx.iter().enumerate() {
            v[i * c..(i + 1) * c].copy_from_slice(&self.nodes[x].val[j * c..(j + 1) * c]);
        }
        self.push(idx.len(), c, v, Op::Gather { x, idx: idx.to_vec() })
    }

    /// Connectionist Temporal Classification loss (blank = class 0) for `n`
    /// sequences of `t` frames: `logits` rows are `(sequence, frame)`.
    /// Returns the mean negative log-likelihood (`1×1`).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// // 3 frames, 3 classes, uniform logits, target "1"
    /// let z = g.input(3, 3, vec![0.0; 9]);
    /// let l = g.ctc(z, 1, 3, &[vec![1]]);
    /// assert!(g.value(l)[0] > 0.0);
    /// ```
    pub fn ctc(&mut self, logits: Id, n: usize, t: usize, targets: &[Vec<u32>]) -> Id {
        let c = self.nodes[logits].cols;
        let z = &self.nodes[logits].val;
        let mut grad = vec![0f32; z.len()];
        let mut total = 0f64;
        let lse = |a: f64, b: f64| if a == f64::NEG_INFINITY { b } else if b == f64::NEG_INFINITY { a } else { a.max(b) + (-(a - b).abs()).exp().ln_1p() };
        for b in 0..n {
            // log-softmax per frame
            let mut lp = vec![0f64; t * c];
            for f in 0..t {
                let row = &z[(b * t + f) * c..(b * t + f + 1) * c];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let s: f64 = row.iter().map(|v| (*v as f64 - m).exp()).sum();
                for k in 0..c {
                    lp[f * c + k] = row[k] as f64 - m - s.ln();
                }
            }
            let tg = &targets[b];
            let ext: Vec<usize> = std::iter::once(0).chain(tg.iter().flat_map(|&x| [x as usize, 0])).collect();
            let sl = ext.len();
            let ni = f64::NEG_INFINITY;
            let mut al = vec![ni; t * sl];
            let mut be = vec![ni; t * sl];
            al[0] = lp[ext[0]];
            if sl > 1 {
                al[1] = lp[ext[1]];
            }
            for f in 1..t {
                for s in 0..sl {
                    let mut a = al[(f - 1) * sl + s];
                    if s >= 1 {
                        a = lse(a, al[(f - 1) * sl + s - 1]);
                    }
                    if s >= 2 && ext[s] != 0 && ext[s] != ext[s - 2] {
                        a = lse(a, al[(f - 1) * sl + s - 2]);
                    }
                    al[f * sl + s] = a + lp[f * c + ext[s]];
                }
            }
            be[(t - 1) * sl + sl - 1] = lp[(t - 1) * c + ext[sl - 1]];
            if sl > 1 {
                be[(t - 1) * sl + sl - 2] = lp[(t - 1) * c + ext[sl - 2]];
            }
            for f in (0..t - 1).rev() {
                for s in 0..sl {
                    let mut a = be[(f + 1) * sl + s];
                    if s + 1 < sl {
                        a = lse(a, be[(f + 1) * sl + s + 1]);
                    }
                    if s + 2 < sl && ext[s] != 0 && ext[s] != ext[s + 2] {
                        a = lse(a, be[(f + 1) * sl + s + 2]);
                    }
                    be[f * sl + s] = a + lp[f * c + ext[s]];
                }
            }
            let logp = if sl > 1 { lse(al[(t - 1) * sl + sl - 1], al[(t - 1) * sl + sl - 2]) } else { al[(t - 1) * sl] };
            if !logp.is_finite() {
                continue; // target longer than the frames allow
            }
            total -= logp;
            for f in 0..t {
                let mut occ = vec![ni; c];
                for s in 0..sl {
                    // α·β double-counts the frame's emission: subtract it once.
                    let v = al[f * sl + s] + be[f * sl + s] - lp[f * c + ext[s]];
                    occ[ext[s]] = lse(occ[ext[s]], v);
                }
                for k in 0..c {
                    let post = (occ[k] - logp).exp();
                    grad[(b * t + f) * c + k] = ((lp[f * c + k].exp() - post) / n as f64) as f32;
                }
            }
        }
        let l = (total / n as f64) as f32;
        self.push(1, 1, vec![l], Op::Ctc { logits, grad })
    }

    /// Weighted binary cross-entropy with logits (mean over elements).
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let z = g.input(1, 1, vec![0.0]);
    /// let l = g.bce_logits(z, &[1.0], &[1.0]);
    /// assert!((g.value(l)[0] - 2f32.ln()).abs() < 1e-6);
    /// ```
    pub fn bce_logits(&mut self, a: Id, target: &[f32], weight: &[f32]) -> Id {
        let v = &self.nodes[a].val;
        let mut l = 0f64;
        for i in 0..v.len() {
            let z = v[i] as f64;
            let t = target[i] as f64;
            l += weight[i] as f64 * (z.max(0.0) - z * t + (1.0 + (-z.abs()).exp()).ln());
        }
        let l = (l / v.len() as f64) as f32;
        self.push(1, 1, vec![l], Op::Bce { a, target: target.to_vec(), weight: weight.to_vec() })
    }

    /// Mean softmax cross-entropy of `logits` rows against `targets`
    /// (`u32::MAX` = ignore). Returns a `1×1` loss node.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let l = g.input(1, 2, vec![0.0, 0.0]);
    /// let loss = g.cross_entropy(l, &[0]);
    /// assert!((g.value(loss)[0] - 2f32.ln()).abs() < 1e-6);
    /// ```
    pub fn cross_entropy(&mut self, logits: Id, targets: &[u32]) -> Id {
        let (r, c) = self.shape(logits);
        let mut probs = self.nodes[logits].val.clone();
        let mut loss = 0f64;
        let mut count = 0;
        for i in 0..r {
            let row = &mut probs[i * c..(i + 1) * c];
            crate::tensor::ops::softmax_inplace(row);
            if targets[i] != u32::MAX {
                loss -= (row[targets[i] as usize].max(1e-12) as f64).ln();
                count += 1;
            }
        }
        let l = (loss / count.max(1) as f64) as f32;
        self.push(1, 1, vec![l], Op::SoftmaxCe { logits, targets: targets.to_vec(), probs, count })
    }

    /// Mean squared error against a constant target. Returns `1×1`.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// # let mut p = Params::default();
    /// let mut g = Graph::new(&mut p, &pool);
    /// let a = g.input(1, 2, vec![1.0, 3.0]);
    /// let l = g.mse(a, &[1.0, 1.0]);
    /// assert_eq!(g.value(l)[0], 2.0);
    /// ```
    pub fn mse(&mut self, a: Id, target: &[f32]) -> Id {
        let v = &self.nodes[a].val;
        let l = v.iter().zip(target).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / v.len() as f32;
        self.push(1, 1, vec![l], Op::Mse { a, target: target.to_vec() })
    }

    /// Back-propagates from scalar node `loss`, accumulating parameter
    /// gradients into the store. Returns the loss value.
    ///
    /// # Example
    /// ```
    /// # use aido::sdk::autograd::{Graph, Params};
    /// # let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut p = Params::default();
    /// let w = p.add("w", 1, 1, vec![2.0], true);
    /// {
    ///     let mut g = Graph::new(&mut p, &pool);
    ///     let x = g.input(1, 1, vec![3.0]);
    ///     let y = g.linear(x, w, None, None);
    ///     let l = g.mse(y, &[0.0]);
    ///     g.backward(l);
    /// }
    /// assert_eq!(p.grads[w][0], 2.0 * 6.0 * 3.0); // d/dw (3w)^2 = 18w
    /// ```
    pub fn backward(&mut self, loss: Id) -> f32 {
        for n in &mut self.nodes {
            n.grad = vec![0.0; n.val.len()];
        }
        self.nodes[loss].grad[0] = 1.0;
        let pool = self.pool;
        for id in (0..self.nodes.len()).rev() {
            let g = std::mem::take(&mut self.nodes[id].grad);
            if g.iter().all(|v| *v == 0.0) {
                continue;
            }
            let (rows, cols) = (self.nodes[id].rows, self.nodes[id].cols);
            let op = std::mem::replace(&mut self.nodes[id].op, Op::Input);
            match &op {
                Op::Input => {}
                Op::Param(p) => {
                    if self.params.trainable[*p] {
                        for (a, b) in self.params.grads[*p].iter_mut().zip(&g) {
                            *a += b;
                        }
                    }
                }
                Op::Linear { x, w, b, lora } => {
                    let (m, k) = self.shape(*x);
                    let n = cols;
                    // dX = dY · W
                    let mut dx = vec![0f32; m * k];
                    mm_ab(pool, &g, m, n, &self.params.data[*w], k, &mut dx, false);
                    let gt = transpose(&g, m, n);
                    if self.params.trainable[*w] {
                        let mut dw = vec![0f32; n * k];
                        mm_ab(pool, &gt, n, m, &self.nodes[*x].val, k, &mut dw, false);
                        for (a, bb) in self.params.grads[*w].iter_mut().zip(&dw) {
                            *a += bb;
                        }
                    }
                    if let Some(bi) = b {
                        if self.params.trainable[*bi] {
                            for row in g.chunks_exact(n) {
                                for (a, bb) in self.params.grads[*bi].iter_mut().zip(row) {
                                    *a += bb;
                                }
                            }
                        }
                    }
                    if let Some((a, bb, s)) = lora {
                        let r = self.params.shapes[*a].0;
                        let xv = &self.nodes[*x].val;
                        let mut h = vec![0f32; m * r];
                        mm_xwt(pool, xv, m, k, &self.params.data[*a], r, &mut h, false);
                        h.iter_mut().for_each(|v| *v *= s);
                        // dB = dYᵀ·h ; dh = dY·B ; dA = s·dhᵀ·x ; dX += s·dh·A
                        let mut db = vec![0f32; n * r];
                        mm_ab(pool, &gt, n, m, &h, r, &mut db, false);
                        let mut dh = vec![0f32; m * r];
                        mm_ab(pool, &g, m, n, &self.params.data[*bb], r, &mut dh, false);
                        dh.iter_mut().for_each(|v| *v *= s);
                        let dht = transpose(&dh, m, r);
                        let mut da = vec![0f32; r * k];
                        mm_ab(pool, &dht, r, m, xv, k, &mut da, false);
                        mm_ab(pool, &dh, m, r, &self.params.data[*a], k, &mut dx, true);
                        for (x1, y1) in self.params.grads[*bb].iter_mut().zip(&db) {
                            *x1 += y1;
                        }
                        for (x1, y1) in self.params.grads[*a].iter_mut().zip(&da) {
                            *x1 += y1;
                        }
                    }
                    self.acc(*x, &dx);
                }
                Op::Add(a, b) => {
                    self.acc(*a, &g);
                    self.acc(*b, &g);
                }
                Op::Mul(a, b) => {
                    let da: Vec<f32> = g.iter().zip(&self.nodes[*b].val).map(|(x, y)| x * y).collect();
                    let db: Vec<f32> = g.iter().zip(&self.nodes[*a].val).map(|(x, y)| x * y).collect();
                    self.acc(*a, &da);
                    self.acc(*b, &db);
                }
                Op::Scale(a, s) => {
                    let d: Vec<f32> = g.iter().map(|x| x * s).collect();
                    self.acc(*a, &d);
                }
                Op::Relu(a) => {
                    let d: Vec<f32> = g.iter().zip(&self.nodes[*a].val).map(|(x, v)| if *v > 0.0 { *x } else { 0.0 }).collect();
                    self.acc(*a, &d);
                }
                Op::Silu(a) => {
                    let d: Vec<f32> = g
                        .iter()
                        .zip(&self.nodes[*a].val)
                        .map(|(x, v)| {
                            let s = 1.0 / (1.0 + (-v).exp());
                            x * s * (1.0 + v * (1.0 - s))
                        })
                        .collect();
                    self.acc(*a, &d);
                }
                Op::Tanh(a) => {
                    let d: Vec<f32> = g.iter().zip(&self.nodes[id].val).map(|(x, y)| x * (1.0 - y * y)).collect();
                    self.acc(*a, &d);
                }
                Op::Gelu(a) => {
                    let d: Vec<f32> = g
                        .iter()
                        .zip(&self.nodes[*a].val)
                        .map(|(x, u)| {
                            let c = 0.797_884_6f32;
                            let inner = c * (u + 0.044715 * u * u * u);
                            let t = inner.tanh();
                            let dinner = c * (1.0 + 3.0 * 0.044715 * u * u);
                            x * (0.5 * (1.0 + t) + 0.5 * u * (1.0 - t * t) * dinner)
                        })
                        .collect();
                    self.acc(*a, &d);
                }
                Op::SwiGlu(gt, up) => {
                    let (gv, uv) = (&self.nodes[*gt].val, &self.nodes[*up].val);
                    let mut dg = vec![0f32; g.len()];
                    let mut du = vec![0f32; g.len()];
                    for i in 0..g.len() {
                        let s = 1.0 / (1.0 + (-gv[i]).exp());
                        let silu = gv[i] * s;
                        du[i] = g[i] * silu;
                        dg[i] = g[i] * uv[i] * s * (1.0 + gv[i] * (1.0 - s));
                    }
                    self.acc(*gt, &dg);
                    self.acc(*up, &du);
                }
                Op::RmsNorm { x, g: gain, inv, .. } => {
                    let c = cols;
                    let xv = &self.nodes[*x].val;
                    let gw = &self.params.data[*gain];
                    let mut dx = vec![0f32; xv.len()];
                    let mut dgain = vec![0f32; c];
                    for i in 0..rows {
                        let xr = &xv[i * c..(i + 1) * c];
                        let gr = &g[i * c..(i + 1) * c];
                        let r = inv[i];
                        let dot: f32 = (0..c).map(|j| gr[j] * gw[j] * xr[j]).sum();
                        for j in 0..c {
                            dgain[j] += gr[j] * xr[j] * r;
                            dx[i * c + j] = r * gr[j] * gw[j] - r * r * r * xr[j] * dot / c as f32;
                        }
                    }
                    if self.params.trainable[*gain] {
                        for (a, b) in self.params.grads[*gain].iter_mut().zip(&dgain) {
                            *a += b;
                        }
                    }
                    self.acc(*x, &dx);
                }
                Op::Embed { table, ids } => {
                    if self.params.trainable[*table] {
                        let c = cols;
                        for (i, &t) in ids.iter().enumerate() {
                            for j in 0..c {
                                self.params.grads[*table][t as usize * c + j] += g[i * c + j];
                            }
                        }
                    }
                }
                Op::Reshape(x) => {
                    self.acc(*x, &g);
                }
                Op::Rope { x, heads, seq, base, neox } => {
                    let hd = cols / heads;
                    let rope = Rope::new(hd, *base, None, *neox);
                    let mut d = g.clone();
                    let mut cs = vec![0f32; hd];
                    for i in 0..rows {
                        rope.table(i % seq, &mut cs);
                        for k in (1..hd).step_by(2) {
                            cs[k] = -cs[k]; // inverse rotation
                        }
                        rope.apply(&mut d[i * cols..(i + 1) * cols], hd, &cs);
                    }
                    self.acc(*x, &d);
                }
                Op::Attn { q, k, v, heads, seq, probs } => {
                    let c = cols;
                    let hd = c / heads;
                    let scale = 1.0 / (hd as f32).sqrt();
                    let (qv, kv, vv) = (&self.nodes[*q].val, &self.nodes[*k].val, &self.nodes[*v].val);
                    let mut dq = vec![0f32; qv.len()];
                    let mut dk = vec![0f32; kv.len()];
                    let mut dv = vec![0f32; vv.len()];
                    let nseq = rows / seq;
                    let (dqp, dkp, dvp) = (crate::tensor::SyncPtr::new(&mut dq), crate::tensor::SyncPtr::new(&mut dk), crate::tensor::SyncPtr::new(&mut dv));
                    let (heads, seq) = (*heads, *seq);
                    pool.parallel_for(nseq * heads, |task, _| {
                        let (s, h) = (task / heads, task % heads);
                        let pr = &probs[task * seq * seq..(task + 1) * seq * seq];
                        let at = |i: usize| (s * seq + i) * c + h * hd;
                        let mut dp = vec![0f32; seq];
                        for i in 0..seq {
                            let go = &g[at(i)..at(i) + hd];
                            for j in 0..=i {
                                let vj = &vv[at(j)..at(j) + hd];
                                dp[j] = go.iter().zip(vj).map(|(a, b)| a * b).sum();
                                let dvj = unsafe { dvp.slice(at(j), hd) };
                                for e in 0..hd {
                                    dvj[e] += pr[i * seq + j] * go[e];
                                }
                            }
                            let dot: f32 = (0..=i).map(|j| dp[j] * pr[i * seq + j]).sum();
                            let dqi = unsafe { dqp.slice(at(i), hd) };
                            for j in 0..=i {
                                let ds = pr[i * seq + j] * (dp[j] - dot) * scale;
                                let kj = &kv[at(j)..at(j) + hd];
                                let qi = &qv[at(i)..at(i) + hd];
                                let dkj = unsafe { dkp.slice(at(j), hd) };
                                for e in 0..hd {
                                    dqi[e] += ds * kj[e];
                                    dkj[e] += ds * qi[e];
                                }
                            }
                        }
                    });
                    self.acc(*q, &dq);
                    self.acc(*k, &dk);
                    self.acc(*v, &dv);
                }
                Op::SoftmaxCe { logits, targets, probs, count } => {
                    let c = self.nodes[*logits].cols;
                    let mut d = probs.clone();
                    let scale = g[0] / (*count).max(1) as f32;
                    for (i, &t) in targets.iter().enumerate() {
                        let row = &mut d[i * c..(i + 1) * c];
                        if t == u32::MAX {
                            row.iter_mut().for_each(|v| *v = 0.0);
                            continue;
                        }
                        row[t as usize] -= 1.0;
                        row.iter_mut().for_each(|v| *v *= scale);
                    }
                    self.acc(*logits, &d);
                }
                Op::Mse { a, target } => {
                    let n = target.len() as f32;
                    let d: Vec<f32> = self.nodes[*a].val.iter().zip(target).map(|(x, y)| g[0] * 2.0 * (x - y) / n).collect();
                    self.acc(*a, &d);
                }
                Op::Conv { x, w, b, n, h, wd, stride, col } => {
                    let cin = self.nodes[*x].cols;
                    let (cout, k) = self.params.shapes[*w];
                    let rows = rows;
                    let _ = cols;
                    if self.params.trainable[*w] {
                        let gt = transpose(&g, rows, cout);
                        let mut dw = vec![0f32; cout * k];
                        mm_ab(pool, &gt, cout, rows, col, k, &mut dw, false);
                        for (a, bb) in self.params.grads[*w].iter_mut().zip(&dw) {
                            *a += bb;
                        }
                    }
                    if let Some(bi) = b {
                        if self.params.trainable[*bi] {
                            for row in g.chunks_exact(cout) {
                                for (a, bb) in self.params.grads[*bi].iter_mut().zip(row) {
                                    *a += bb;
                                }
                            }
                        }
                    }
                    // dcol = dY · W  → col2im
                    let mut dcol = vec![0f32; rows * k];
                    mm_ab(pool, &g, rows, cout, &self.params.data[*w], k, &mut dcol, false);
                    let (h, wd, stride, n) = (*h, *wd, *stride, *n);
                    let (ho, wo) = (h.div_ceil(stride), wd.div_ceil(stride));
                    let mut dx = vec![0f32; n * h * wd * cin];
                    for r in 0..rows {
                        let (img, rem) = (r / (ho * wo), r % (ho * wo));
                        let (oy, ox) = (rem / wo, rem % wo);
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = (oy * stride + ky) as isize - 1;
                                let ix = (ox * stride + kx) as isize - 1;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                    continue;
                                }
                                let s = ((img * h + iy as usize) * wd + ix as usize) * cin;
                                let src = &dcol[r * k + (ky * 3 + kx) * cin..r * k + (ky * 3 + kx + 1) * cin];
                                for (a, bb) in dx[s..s + cin].iter_mut().zip(src) {
                                    *a += bb;
                                }
                            }
                        }
                    }
                    self.acc(*x, &dx);
                }
                Op::Up2 { x, n, h, w } => {
                    let c = cols;
                    let mut dx = vec![0f32; n * h * w * c];
                    for img in 0..*n {
                        for oy in 0..2 * h {
                            for ox in 0..2 * w {
                                let s = ((img * h + oy / 2) * w + ox / 2) * c;
                                let d = ((img * 2 * h + oy) * 2 * w + ox) * c;
                                for j in 0..c {
                                    dx[s + j] += g[d + j];
                                }
                            }
                        }
                    }
                    self.acc(*x, &dx);
                }
                Op::AddImg { x, v, n } => {
                    let c = cols;
                    let per = rows / n;
                    let mut dv = vec![0f32; n * c];
                    for i in 0..rows {
                        for j in 0..c {
                            dv[(i / per) * c + j] += g[i * c + j];
                        }
                    }
                    self.acc(*x, &g);
                    self.acc(*v, &dv);
                }
                Op::Bce { a, target, weight } => {
                    let v = &self.nodes[*a].val;
                    let nn = v.len() as f32;
                    let d: Vec<f32> = (0..v.len()).map(|i| {
                        let s = 1.0 / (1.0 + (-v[i]).exp());
                        g[0] * weight[i] * (s - target[i]) / nn
                    }).collect();
                    self.acc(*a, &d);
                }
                Op::Gather { x, idx } => {
                    let (xr, c) = (self.nodes[*x].rows, cols);
                    let mut dx = vec![0f32; xr * c];
                    for (i, &j) in idx.iter().enumerate() {
                        for k in 0..c {
                            dx[j * c + k] += g[i * c + k];
                        }
                    }
                    self.acc(*x, &dx);
                }
                Op::Ctc { logits, grad } => {
                    let d: Vec<f32> = grad.iter().map(|v| v * g[0]).collect();
                    self.acc(*logits, &d);
                }
                Op::ConcatCols(a, b) => {
                    let (ca, cb) = (self.nodes[*a].cols, self.nodes[*b].cols);
                    let mut da = vec![0f32; rows * ca];
                    let mut db = vec![0f32; rows * cb];
                    for i in 0..rows {
                        da[i * ca..(i + 1) * ca].copy_from_slice(&g[i * cols..i * cols + ca]);
                        db[i * cb..(i + 1) * cb].copy_from_slice(&g[i * cols + ca..(i + 1) * cols]);
                    }
                    self.acc(*a, &da);
                    self.acc(*b, &db);
                }
            }
            self.nodes[id].op = op;
        }
        self.nodes[loss].val[0]
    }

    /// Adds `d` to the gradient of node `id`.
    fn acc(&mut self, id: Id, d: &[f32]) {
        let n = &mut self.nodes[id];
        if n.grad.len() != d.len() {
            n.grad = vec![0.0; d.len()];
        }
        for (a, b) in n.grad.iter_mut().zip(d) {
            *a += b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_gradient() {
        let pool = ThreadPool::new(1, None);
        let mut p = Params::default();
        let w = p.add("w", 4, 3, (0..12).map(|i| ((i * 7 % 5) as f32 - 2.0) / 3.0).collect(), true);
        let x: Vec<f32> = (0..15).map(|i| ((i * 3 % 7) as f32 - 3.0) / 4.0).collect();
        let tg = vec![vec![1u32, 2], vec![3]];
        let f = |p: &mut Params, back: bool| -> f32 {
            let mut g = Graph::new(p, &pool);
            let xi = g.input(5, 3, x.clone());
            let a = g.linear(xi, w, None, None);
            let a2 = g.gather_rows(a, &[0, 1, 2, 3, 4, 4, 3, 2, 1, 0]);
            let l = g.ctc(a2, 2, 5, &tg);
            if back { g.backward(l) } else { g.value(l)[0] }
        };
        p.zero_grad();
        f(&mut p, true);
        for idx in [0usize, 5, 11] {
            let o = p.data[w][idx];
            p.data[w][idx] = o + 1e-3;
            let lp = f(&mut p, false);
            p.data[w][idx] = o - 1e-3;
            let lm = f(&mut p, false);
            p.data[w][idx] = o;
            let num = (lp - lm) / 2e-3;
            assert!((num - p.grads[w][idx]).abs() < 1e-3 + 0.03 * num.abs(), "ctc grad {num} vs {}", p.grads[w][idx]);
        }
    }

    #[test]
    fn conv_gradients() {
        let pool = ThreadPool::new(2, None);
        let mut p = Params::default();
        let w1 = p.add("w1", 3, 18, (0..54).map(|i| ((i * 7 % 13) as f32 - 6.0) / 15.0).collect(), true);
        let b1 = p.add("b1", 1, 3, vec![0.1, -0.1, 0.05], true);
        let w2 = p.add("w2", 2, 27, (0..54).map(|i| ((i * 5 % 11) as f32 - 5.0) / 15.0).collect(), true);
        let tv = p.add("tv", 2, 3, vec![0.2, -0.3, 0.1, 0.0, 0.4, -0.2], true);
        let x: Vec<f32> = (0..2 * 16 * 2).map(|i| ((i * 3 % 7) as f32 - 3.0) / 3.0).collect();
        let tgt: Vec<f32> = (0..2 * 16 * 2).map(|i| (i % 2) as f32).collect();
        let f = |p: &mut Params, back: bool| -> f32 {
            let mut g = Graph::new(p, &pool);
            let xi = g.input(32, 2, x.clone());
            let h = g.conv3x3(xi, w1, Some(b1), 2, 4, 4, 2);
            let t = g.param(tv);
            let h = g.add_per_image(h, t, 2);
            let h = g.silu(h);
            let u = g.upsample2x(h, 2, 2, 2);
            let o = g.conv3x3(u, w2, None, 2, 4, 4, 1);
            let l = g.bce_logits(o, &tgt, &vec![1.5; 64]);
            if back { g.backward(l) } else { g.value(l)[0] }
        };
        p.zero_grad();
        f(&mut p, true);
        for pid in 0..p.names.len() {
            for idx in [0usize, 2] {
                let o = p.data[pid][idx];
                p.data[pid][idx] = o + 1e-3;
                let lp = f(&mut p, false);
                p.data[pid][idx] = o - 1e-3;
                let lm = f(&mut p, false);
                p.data[pid][idx] = o;
                let num = (lp - lm) / 2e-3;
                let ana = p.grads[pid][idx];
                assert!((num - ana).abs() < 1e-3 + 0.05 * num.abs(), "{}[{idx}] {num} vs {ana}", p.names[pid]);
            }
        }
    }

    /// Finite-difference check of every differentiable op in one graph.
    #[test]
    fn gradients_match_finite_differences() {
        let pool = ThreadPool::new(2, None);
        let mut p = Params::default();
        let emb = p.add("emb", 5, 8, (0..40).map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0).collect(), true);
        let w = p.add("w", 8, 8, (0..64).map(|i| ((i * 5 % 13) as f32 - 6.0) / 20.0).collect(), true);
        let wb = p.add("wb", 1, 8, vec![0.05; 8], true);
        let gn = p.add("gn", 1, 8, vec![1.1; 8], true);
        let la = p.add("la", 2, 8, (0..16).map(|i| (i as f32 - 8.0) / 30.0).collect(), true);
        let lb = p.add("lb", 8, 2, (0..16).map(|i| (i as f32 - 7.0) / 25.0).collect(), true);
        let head = p.add("head", 5, 8, (0..40).map(|i| ((i * 3 % 7) as f32 - 3.0) / 10.0).collect(), true);
        let ids = [1u32, 3, 2, 4];
        let targets = [3u32, 2, 4, 0];
        let f = |p: &mut Params, back: bool| -> f32 {
            let mut g = Graph::new(p, &pool);
            let x = g.embed(emb, &ids);
            let n = g.rmsnorm(x, gn, 1e-5);
            let q = g.linear(n, w, Some(wb), Some((la, lb, 0.5)));
            let qr = g.rope(q, 2, 2, 10000.0);
            let a = g.causal_attention(qr, qr, n, 2, 2);
            let s = g.swiglu(a, n);
            let gl = g.gelu(s);
            let t = g.tanh(gl);
            let r = g.add(t, x);
            let logits = g.linear(r, head, None, None);
            let l = g.cross_entropy(logits, &targets);
            if back { g.backward(l) } else { g.value(l)[0] }
        };
        p.zero_grad();
        f(&mut p, true);
        let eps = 1e-3;
        for pid in 0..p.names.len() {
            for idx in [0usize, 3, 7] {
                if idx >= p.data[pid].len() {
                    continue;
                }
                let orig = p.data[pid][idx];
                p.data[pid][idx] = orig + eps;
                let lp = f(&mut p, false);
                p.data[pid][idx] = orig - eps;
                let lm = f(&mut p, false);
                p.data[pid][idx] = orig;
                let num = (lp - lm) / (2.0 * eps);
                let ana = p.grads[pid][idx];
                assert!((num - ana).abs() < 2e-3 + 0.05 * num.abs(), "{}[{idx}]: numeric {num} vs analytic {ana}", p.names[pid]);
            }
        }
    }
}
