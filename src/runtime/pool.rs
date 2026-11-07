//! Low-latency persistent thread pool ("spin-then-park").
//!
//! Token-by-token LLM decoding issues **hundreds of tiny parallel regions per
//! token** (one per matrix-vector product). Generic pools (rayon, OpenMP with
//! passive wait) pay 5–50 µs per region to wake sleeping threads on Windows,
//! which at ~250 regions/token is several milliseconds — a 20–40 % slowdown
//! for a small model. This pool keeps workers *hot* for a short, configurable
//! spin window (default 300 µs, a few times longer than a typical gap between
//! regions inside one token), and only then parks them on the OS. Between
//! requests the CPU therefore truly idles (no heat, no wasted power), while
//! inside a token wake-up costs ~100–300 ns.
//!
//! Work distribution uses a single shared atomic counter (dynamic
//! self-scheduling): every worker grabs the next chunk index with
//! `fetch_add`. A core that is slowed down (thermal throttling, an SMT
//! sibling busy with the OS, an E-core) simply grabs fewer chunks, so the
//! region finishes when the *total* work is done instead of waiting for the
//! slowest statically assigned slice.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

/// Type-erased reference to the job closure of the current parallel region.
///
/// The closure lives on the caller's stack; the pool guarantees that every
/// worker has finished with it before [`ThreadPool::parallel_for`] returns,
/// which is what makes the `'static` transmute in `parallel_for` sound.
#[derive(Clone, Copy)]
struct JobRef {
    /// Fat pointer to `dyn Fn(task_index, worker_index) + Sync`.
    f: *const (dyn Fn(usize, usize) + Sync),
}

unsafe impl Send for JobRef {}
unsafe impl Sync for JobRef {}

/// Per-worker sleep flag used for the Dekker-style park/unpark handshake.
struct WorkerSlot {
    /// `true` while the worker is (about to be) parked in the OS.
    sleeping: AtomicBool,
    /// Handle used to unpark the worker thread.
    thread: UnsafeCell<Option<Thread>>,
}

unsafe impl Sync for WorkerSlot {}

/// State shared between the owning thread and all workers.
struct Shared {
    /// `(region_counter << 16) | active_workers`, bumped once per parallel
    /// region; workers wait for it to change and read their participation
    /// from the *same* value, so a slow worker can never mix two regions.
    epoch: AtomicU64,
    /// Job of the current region (written before `epoch` is bumped).
    job: UnsafeCell<Option<JobRef>>,
    /// Next task index to hand out.
    next: AtomicUsize,
    /// Number of tasks in the current region.
    n_tasks: AtomicUsize,
    /// Number of workers that finished the current region.
    done: AtomicUsize,
    /// Number of background workers participating in the current region.
    active: AtomicUsize,
    /// Pool shutdown flag.
    shutdown: AtomicBool,
    /// Spin budget in nanoseconds before a worker parks.
    spin_ns: AtomicU64,
    /// Per background worker sleep slots.
    slots: Vec<WorkerSlot>,
}

unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// A persistent pool of pinned worker threads with spin-then-park waiting.
///
/// The calling thread always participates as worker `0`, so a pool created
/// with `n_threads = 8` spawns 7 background threads.
pub struct ThreadPool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    n_threads: usize,
    /// Number of threads used by the next regions (`1..=n_threads`).
    width: AtomicUsize,
}

impl ThreadPool {
    /// Creates a pool with `n_threads` total workers (including the caller).
    ///
    /// * **Input** — `n_threads`: total number of threads (clamped to ≥ 1);
    ///   `pin`: optional list of logical CPU ids; worker `i` (background
    ///   workers are `1..n`) is pinned to `pin[i % pin.len()]`. Pinning keeps
    ///   each worker's slice of the weights hot in *its own* L1/L2 cache
    ///   across consecutive tokens and avoids SMT siblings fighting for the
    ///   same core.
    /// * **Output** — a ready pool; threads start parked-spinning.
    /// * **Behaviour** — background threads are spawned immediately.
    ///
    /// # Example
    /// ```
    /// use aido::runtime::pool::ThreadPool;
    /// let pool = ThreadPool::new(4, None);
    /// assert_eq!(pool.n_threads(), 4);
    /// ```
    pub fn new(n_threads: usize, pin: Option<Vec<usize>>) -> Self {
        let n_threads = n_threads.max(1);
        let slots = (1..n_threads)
            .map(|_| WorkerSlot { sleeping: AtomicBool::new(false), thread: UnsafeCell::new(None) })
            .collect();
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            next: AtomicUsize::new(0),
            n_tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            active: AtomicUsize::new(n_threads - 1),
            shutdown: AtomicBool::new(false),
            spin_ns: AtomicU64::new(300_000),
            slots,
        });
        if let Some(p) = &pin {
            if !p.is_empty() {
                pin_current(p[0]);
            }
        }
        let mut handles = Vec::with_capacity(n_threads - 1);
        for wid in 1..n_threads {
            let sh = shared.clone();
            let cpu = pin.as_ref().filter(|p| !p.is_empty()).map(|p| p[wid % p.len()]);
            let h = thread::Builder::new()
                .name(format!("aido-worker-{wid}"))
                .spawn(move || worker_main(sh, wid, cpu))
                .expect("spawn worker");
            unsafe { *shared.slots[wid - 1].thread.get() = Some(h.thread().clone()) };
            handles.push(h);
        }
        ThreadPool { shared, handles, n_threads, width: AtomicUsize::new(n_threads) }
    }

    /// Returns the total number of threads owned by the pool (incl. caller).
    ///
    /// * **Input** — none. **Output** — thread count `>= 1`.
    ///
    /// # Example
    /// ```
    /// let p = aido::runtime::pool::ThreadPool::new(3, None);
    /// assert_eq!(p.n_threads(), 3);
    /// ```
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Returns how many threads the next parallel regions will use.
    ///
    /// * **Output** — current width set by [`ThreadPool::set_width`].
    ///
    /// # Example
    /// ```
    /// let p = aido::runtime::pool::ThreadPool::new(4, None);
    /// p.set_width(2);
    /// assert_eq!(p.width(), 2);
    /// ```
    pub fn width(&self) -> usize {
        self.width.load(Ordering::Relaxed)
    }

    /// Limits how many threads take part in subsequent regions.
    ///
    /// Used by the [`crate::runtime::governor::Governor`]: memory-bound
    /// decoding saturates DRAM bandwidth with fewer cores than exist, so the
    /// extra cores are kept parked (cool, idle) at no loss of speed.
    ///
    /// * **Input** — `w`: desired width, clamped to `1..=n_threads`.
    ///
    /// # Example
    /// ```
    /// let p = aido::runtime::pool::ThreadPool::new(4, None);
    /// p.set_width(100);
    /// assert_eq!(p.width(), 4);
    /// ```
    pub fn set_width(&self, w: usize) {
        self.width.store(w.clamp(1, self.n_threads), Ordering::Relaxed);
    }

    /// Sets the spin window (how long idle workers busy-wait before parking).
    ///
    /// * **Input** — `d`: spin duration. Longer → lower latency between
    ///   regions, more idle power. `Duration::ZERO` parks immediately (eco).
    ///
    /// # Example
    /// ```
    /// use std::time::Duration;
    /// let p = aido::runtime::pool::ThreadPool::new(2, None);
    /// p.set_spin(Duration::from_micros(50));
    /// ```
    pub fn set_spin(&self, d: Duration) {
        self.shared.spin_ns.store(d.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Runs `f(task, worker)` for every `task` in `0..n_tasks` in parallel and
    /// blocks until all tasks finished.
    ///
    /// * **Input** — `n_tasks`: number of independent tasks; `f`: closure
    ///   receiving the task index and the worker index (`0..width`), which can
    ///   be used to index per-thread scratch buffers.
    /// * **Output** — none; all side effects of `f` are visible on return.
    /// * **Behaviour** — tasks are handed out dynamically; with `n_tasks <= 1`
    ///   or width 1 the closure runs inline without touching other threads.
    ///
    /// # Example
    /// ```
    /// use aido::runtime::pool::ThreadPool;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// let pool = ThreadPool::new(4, None);
    /// let sum = AtomicUsize::new(0);
    /// pool.parallel_for(100, |i, _w| { sum.fetch_add(i, Ordering::Relaxed); });
    /// assert_eq!(sum.load(Ordering::Relaxed), 4950);
    /// ```
    pub fn parallel_for<F>(&self, n_tasks: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let width = self.width().min(n_tasks.max(1));
        if n_tasks == 0 {
            return;
        }
        if width <= 1 || self.n_threads == 1 {
            for i in 0..n_tasks {
                f(i, 0);
            }
            return;
        }
        let sh = &*self.shared;
        let bg = width - 1;
        let fref: &(dyn Fn(usize, usize) + Sync) = &f;
        // SAFETY: we wait for all participants before returning, so the
        // closure outlives every use.
        let fptr: *const (dyn Fn(usize, usize) + Sync) = unsafe { std::mem::transmute(fref) };
        unsafe { *sh.job.get() = Some(JobRef { f: fptr }) };
        sh.n_tasks.store(n_tasks, Ordering::Relaxed);
        sh.next.store(0, Ordering::Relaxed);
        sh.done.store(0, Ordering::Relaxed);
        sh.active.store(bg, Ordering::Relaxed);
        let old = sh.epoch.load(Ordering::Relaxed);
        sh.epoch.store((((old >> 16) + 1) << 16) | bg as u64, Ordering::SeqCst);
        // Wake parked participants (only the ones we need).
        for slot in &sh.slots[..bg] {
            if slot.sleeping.load(Ordering::SeqCst) {
                if let Some(t) = unsafe { &*slot.thread.get() } {
                    t.unpark();
                }
            }
        }
        // The caller works too.
        loop {
            let i = sh.next.fetch_add(1, Ordering::Relaxed);
            if i >= n_tasks {
                break;
            }
            f(i, 0);
        }
        // Wait for background participants.
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < bg {
            spins += 1;
            if spins < 1 << 20 {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
        unsafe { *sh.job.get() = None };
    }

    /// Splits `0..n` into contiguous chunks of `chunk` elements and runs
    /// `f(start, end, worker)` for each chunk in parallel.
    ///
    /// * **Input** — `n`: total length; `chunk`: elements per task (≥ 1);
    ///   `f`: closure over a half-open range.
    /// * **Behaviour** — convenience wrapper over [`ThreadPool::parallel_for`].
    ///
    /// # Example
    /// ```
    /// use aido::runtime::pool::ThreadPool;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// let pool = ThreadPool::new(3, None);
    /// let c = AtomicUsize::new(0);
    /// pool.parallel_chunks(10, 4, |s, e, _| { c.fetch_add(e - s, Ordering::Relaxed); });
    /// assert_eq!(c.load(Ordering::Relaxed), 10);
    /// ```
    pub fn parallel_chunks<F>(&self, n: usize, chunk: usize, f: F)
    where
        F: Fn(usize, usize, usize) + Sync,
    {
        let chunk = chunk.max(1);
        let tasks = n.div_ceil(chunk);
        self.parallel_for(tasks, |t, w| {
            let s = t * chunk;
            let e = (s + chunk).min(n);
            f(s, e, w)
        });
    }
}

impl Drop for ThreadPool {
    /// Signals shutdown, wakes every worker and joins the threads.
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.epoch.fetch_add(1 << 16, Ordering::SeqCst);
        for slot in &self.shared.slots {
            if let Some(t) = unsafe { &*slot.thread.get() } {
                t.unpark();
            }
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Pins the current thread to logical CPU `cpu` (best effort).
///
/// * **Input** — logical processor index. **Output** — none; failures are
///   ignored (pinning is an optimisation, never a requirement).
fn pin_current(cpu: usize) {
    if let Some(ids) = core_affinity::get_core_ids() {
        if let Some(id) = ids.into_iter().find(|c| c.id == cpu) {
            core_affinity::set_for_current(id);
        }
    }
}

/// Main loop of a background worker.
///
/// * **Input** — shared state, worker id (`1..n`), optional CPU to pin to.
/// * **Behaviour** — waits for a new epoch (spin, then park), runs tasks of
///   the region if it is among the active participants, reports completion.
fn worker_main(sh: Arc<Shared>, wid: usize, cpu: Option<usize>) {
    if let Some(c) = cpu {
        pin_current(c);
    }
    let slot = &sh.slots[wid - 1];
    // Start from the initial value (0), NOT a fresh load: a region may already
    // have been published before this thread got scheduled.
    let mut seen = 0u64;
    loop {
        // ---- wait for the next region ----
        let start = Instant::now();
        let spin_ns = sh.spin_ns.load(Ordering::Relaxed);
        let mut k = 0u32;
        loop {
            let e = sh.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            k = k.wrapping_add(1);
            if k & 63 == 0 && start.elapsed().as_nanos() as u64 >= spin_ns {
                slot.sleeping.store(true, Ordering::SeqCst);
                if sh.epoch.load(Ordering::SeqCst) == seen && !sh.shutdown.load(Ordering::SeqCst) {
                    thread::park_timeout(Duration::from_millis(50));
                }
                slot.sleeping.store(false, Ordering::SeqCst);
            } else {
                std::hint::spin_loop();
            }
        }
        if sh.shutdown.load(Ordering::SeqCst) {
            return;
        }
        // ---- participate if active ----
        if wid as u64 > (seen & 0xFFFF) {
            continue;
        }
        let job = unsafe { (*sh.job.get()).expect("job set before epoch") };
        let n = sh.n_tasks.load(Ordering::Relaxed);
        let f = unsafe { &*job.f };
        loop {
            let i = sh.next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            f(i, wid);
        }
        sh.done.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn many_regions_are_consistent() {
        let pool = ThreadPool::new(6, None);
        for r in 0..2000 {
            let v: Vec<AtomicUsize> = (0..37).map(|_| AtomicUsize::new(0)).collect();
            pool.parallel_for(37, |i, _| {
                v[i].fetch_add(i + r, Ordering::Relaxed);
            });
            for (i, x) in v.iter().enumerate() {
                assert_eq!(x.load(Ordering::Relaxed), i + r);
            }
        }
    }

    #[test]
    fn width_and_park() {
        let pool = ThreadPool::new(4, None);
        pool.set_spin(Duration::ZERO);
        pool.set_width(2);
        let c = AtomicUsize::new(0);
        for _ in 0..100 {
            pool.parallel_for(10, |_, w| {
                assert!(w < 2);
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(c.load(Ordering::Relaxed), 1000);
    }
}
