//! Tensors, weight storage and the numeric kernels shared by all models.
//!
//! * [`quant`] — GGML block formats and integer SIMD dot products.
//! * [`ops`] — element-wise ops, normalisation, RoPE, attention, matmul
//!   dispatch on the thread pool.
//! * [`gemm`] — cache-blocked f32 GEMM (BLIS-style 6×16 micro-kernel) used
//!   by convolution / diffusion models.

pub mod gemm;
pub mod ops;
pub mod quant;
pub mod quantize;

use std::sync::Arc;

use quant::GgmlType;

/// Raw, immutable bytes of a weight tensor, either memory-mapped from the
/// model file (zero-copy, paged in on demand by the OS) or owned.
#[derive(Clone)]
pub struct Bytes {
    ptr: *const u8,
    len: usize,
    _keep: Arc<dyn std::any::Any + Send + Sync>,
}

unsafe impl Send for Bytes {}
unsafe impl Sync for Bytes {}

impl Bytes {
    /// Wraps a sub-range of a shared owner (e.g. an `Arc<Mmap>`).
    ///
    /// * **Input** — `owner`: keeps the memory alive; `data`: slice borrowed
    ///   from that owner.
    /// * **Safety contract** — `data` must point into memory owned by
    ///   `owner` (checked only by convention, the function is safe because the
    ///   `Arc` keeps the allocation alive for the lifetime of `Bytes`).
    ///
    /// # Example
    /// ```
    /// use std::sync::Arc;
    /// let owner: Arc<Vec<u8>> = Arc::new(vec![1, 2, 3, 4]);
    /// let b = aido::tensor::Bytes::from_owner(owner.clone(), &owner[1..3]);
    /// assert_eq!(b.as_slice(), &[2, 3]);
    /// ```
    pub fn from_owner<T: std::any::Any + Send + Sync>(owner: Arc<T>, data: &[u8]) -> Self {
        Bytes { ptr: data.as_ptr(), len: data.len(), _keep: owner }
    }

    /// Takes ownership of a vector.
    ///
    /// # Example
    /// ```
    /// let b = aido::tensor::Bytes::from_vec(vec![7u8; 3]);
    /// assert_eq!(b.len(), 3);
    /// ```
    pub fn from_vec(v: Vec<u8>) -> Self {
        let a = Arc::new(v);
        let (ptr, len) = (a.as_ptr(), a.len());
        Bytes { ptr, len, _keep: a }
    }

    /// Returns the bytes as a slice.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::tensor::Bytes::from_vec(vec![1]).as_slice(), &[1]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length in bytes.
    ///
    /// # Example
    /// ```
    /// assert!(aido::tensor::Bytes::from_vec(vec![]).is_empty());
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A 2-D weight matrix `[rows, cols]` stored row-major in a GGML type.
///
/// Each row is `ty.row_bytes(cols)` bytes; `y = W·x` produces `rows` outputs.
#[derive(Clone)]
pub struct QWeight {
    /// Element type.
    pub ty: GgmlType,
    /// Output dimension.
    pub rows: usize,
    /// Input dimension.
    pub cols: usize,
    /// Backing bytes.
    pub data: Bytes,
}

impl QWeight {
    /// Creates a weight view, validating the byte length.
    ///
    /// * **Input** — type, `rows`, `cols`, and data of exactly
    ///   `rows * ty.row_bytes(cols)` bytes.
    /// * **Output** — `Err` if the length does not match.
    ///
    /// # Example
    /// ```
    /// use aido::tensor::{QWeight, Bytes, quant::GgmlType};
    /// let w = QWeight::new(GgmlType::F32, 2, 3, Bytes::from_vec(vec![0; 24])).unwrap();
    /// assert_eq!(w.row(1).len(), 12);
    /// ```
    pub fn new(ty: GgmlType, rows: usize, cols: usize, data: Bytes) -> anyhow::Result<Self> {
        let need = rows * ty.row_bytes(cols);
        anyhow::ensure!(data.len() >= need, "weight needs {need} bytes, got {}", data.len());
        Ok(QWeight { ty, rows, cols, data })
    }

    /// Builds an F32 weight from a float slice (mainly for tests).
    ///
    /// # Example
    /// ```
    /// let w = aido::tensor::QWeight::from_f32(1, 2, &[1.0, 2.0]);
    /// assert_eq!(w.rows, 1);
    /// ```
    pub fn from_f32(rows: usize, cols: usize, v: &[f32]) -> Self {
        assert_eq!(v.len(), rows * cols);
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        QWeight { ty: GgmlType::F32, rows, cols, data: Bytes::from_vec(b) }
    }

    /// Returns the raw bytes of row `r`.
    ///
    /// # Example
    /// ```
    /// let w = aido::tensor::QWeight::from_f32(2, 1, &[1.0, 2.0]);
    /// assert_eq!(w.row(1), &2.0f32.to_le_bytes());
    /// ```
    #[inline]
    pub fn row(&self, r: usize) -> &[u8] {
        let rb = self.ty.row_bytes(self.cols);
        &self.data.as_slice()[r * rb..(r + 1) * rb]
    }

    /// Dequantises row `r` into `out` (length `cols`).
    ///
    /// # Example
    /// ```
    /// let w = aido::tensor::QWeight::from_f32(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// let mut o = [0f32; 2];
    /// w.dequant_row(1, &mut o);
    /// assert_eq!(o, [3.0, 4.0]);
    /// ```
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        quant::dequantize_row(self.ty, self.row(r), out);
    }

    /// Total size in bytes.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::tensor::QWeight::from_f32(2, 2, &[0.0; 4]).nbytes(), 16);
    /// ```
    pub fn nbytes(&self) -> usize {
        self.rows * self.ty.row_bytes(self.cols)
    }
}

/// A raw pointer that may be shared across pool workers. Each worker must
/// write a *disjoint* region — the type exists only to move that promise
/// past the borrow checker inside parallel loops.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// Wraps a mutable slice.
    ///
    /// # Example
    /// ```
    /// let mut v = vec![0u8; 4];
    /// let p = aido::tensor::SyncPtr::new(&mut v);
    /// unsafe { *p.0.add(2) = 9 };
    /// assert_eq!(v[2], 9);
    /// ```
    pub fn new(s: &mut [T]) -> Self {
        SyncPtr(s.as_mut_ptr())
    }

    /// Returns the raw pointer (method call so closures capture the whole
    /// `Send + Sync` wrapper rather than the bare pointer field).
    ///
    /// # Example
    /// ```
    /// let mut v = vec![1u8];
    /// assert_eq!(unsafe { *aido::tensor::SyncPtr::new(&mut v).ptr() }, 1);
    /// ```
    #[inline]
    pub fn ptr(&self) -> *mut T {
        self.0
    }

    /// Returns a mutable sub-slice `[start, start+len)`.
    ///
    /// # Safety
    /// The caller guarantees the range is in bounds and not aliased by any
    /// other live reference.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}
