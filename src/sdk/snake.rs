//! Reinforcement learning demo: the Snake game solved with Deep Q-Learning.
//!
//! * Environment: `size × size` grid, the snake dies on walls / itself,
//!   +10 for food, −10 for death, a step limit prevents endless loops.
//! * State (11 features): danger straight / right / left, current direction
//!   (one-hot), food left / right / up / down.
//! * Actions: go straight, turn right, turn left.
//! * Agent: MLP Q-network `11 → 128 → 128 → 3`, experience replay, target
//!   network, ε-greedy exploration with linear decay, AdamW.

use std::collections::VecDeque;

use super::autograd::{Graph, Params};
use super::Rng;
use crate::runtime::pool::ThreadPool;

/// The game.
#[derive(Clone)]
pub struct Snake {
    /// Grid size.
    pub size: i32,
    /// Body, head first.
    pub body: VecDeque<(i32, i32)>,
    /// Direction index 0 up, 1 right, 2 down, 3 left.
    pub dir: usize,
    /// Food position.
    pub food: (i32, i32),
    /// Food eaten.
    pub score: usize,
    steps_since_food: usize,
    rng: Rng,
}

const DIRS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

impl Snake {
    /// New game.
    ///
    /// # Example
    /// ```
    /// let s = aido::sdk::snake::Snake::new(10, 1);
    /// assert_eq!(s.body.len(), 3);
    /// ```
    pub fn new(size: i32, seed: u64) -> Self {
        let c = size / 2;
        let mut s = Snake { size, body: VecDeque::from(vec![(c, c), (c - 1, c), (c - 2, c)]), dir: 1, food: (0, 0), score: 0, steps_since_food: 0, rng: Rng(seed | 1) };
        s.place_food();
        s
    }

    fn place_food(&mut self) {
        loop {
            let p = ((self.rng.uniform() * self.size as f32) as i32, (self.rng.uniform() * self.size as f32) as i32);
            if !self.body.contains(&p) {
                self.food = p;
                return;
            }
        }
    }

    fn collides(&self, p: (i32, i32)) -> bool {
        p.0 < 0 || p.1 < 0 || p.0 >= self.size || p.1 >= self.size || self.body.iter().take(self.body.len() - 1).any(|&b| b == p)
    }

    /// The 11-feature observation.
    ///
    /// # Example
    /// ```
    /// assert_eq!(aido::sdk::snake::Snake::new(10, 1).state().len(), 11);
    /// ```
    pub fn state(&self) -> Vec<f32> {
        let h = self.body[0];
        let ahead = |d: usize| (h.0 + DIRS[d].0, h.1 + DIRS[d].1);
        let (s, r, l) = (self.dir, (self.dir + 1) % 4, (self.dir + 3) % 4);
        let mut v = vec![self.collides(ahead(s)) as u8 as f32, self.collides(ahead(r)) as u8 as f32, self.collides(ahead(l)) as u8 as f32];
        for d in 0..4 {
            v.push((self.dir == d) as u8 as f32);
        }
        v.push((self.food.0 < h.0) as u8 as f32);
        v.push((self.food.0 > h.0) as u8 as f32);
        v.push((self.food.1 < h.1) as u8 as f32);
        v.push((self.food.1 > h.1) as u8 as f32);
        v
    }

    /// Applies action (0 straight, 1 right, 2 left). Returns `(reward, done)`.
    ///
    /// # Example
    /// ```
    /// let mut s = aido::sdk::snake::Snake::new(10, 1);
    /// let (_r, done) = s.step(0);
    /// assert!(!done);
    /// ```
    pub fn step(&mut self, action: usize) -> (f32, bool) {
        self.dir = match action {
            1 => (self.dir + 1) % 4,
            2 => (self.dir + 3) % 4,
            _ => self.dir,
        };
        let h = self.body[0];
        let nh = (h.0 + DIRS[self.dir].0, h.1 + DIRS[self.dir].1);
        self.steps_since_food += 1;
        if self.collides(nh) || self.steps_since_food > 100 * self.body.len() {
            return (-10.0, true);
        }
        self.body.push_front(nh);
        if nh == self.food {
            self.score += 1;
            self.steps_since_food = 0;
            self.place_food();
            return (10.0, false);
        }
        self.body.pop_back();
        (-0.01, false)
    }

    /// ASCII rendering of the board.
    ///
    /// # Example
    /// ```
    /// assert!(aido::sdk::snake::Snake::new(5, 1).render().contains('@'));
    /// ```
    pub fn render(&self) -> String {
        let mut s = String::new();
        for y in 0..self.size {
            for x in 0..self.size {
                s.push(if self.body[0] == (x, y) { '@' } else if self.body.contains(&(x, y)) { 'o' } else if self.food == (x, y) { '*' } else { '.' });
            }
            s.push('\n');
        }
        s
    }
}

/// Q-network.
pub struct QNet {
    /// Parameters.
    pub params: Params,
}

impl QNet {
    /// Random network.
    ///
    /// # Example
    /// ```
    /// let q = aido::sdk::snake::QNet::new(1);
    /// assert_eq!(q.params.names.len(), 6);
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed | 1);
        let mut p = Params::default();
        for (i, (a, b)) in [(11, 128), (128, 128), (128, 3)].iter().enumerate() {
            p.add(&format!("l{i}.weight"), *b, *a, r.normals(a * b, (2.0 / *a as f32).sqrt()), true);
            p.add(&format!("l{i}.bias"), 1, *b, vec![0.0; *b], true);
        }
        QNet { params: p }
    }

    fn forward(g: &mut Graph, x: Vec<f32>, n: usize) -> usize {
        let mut h = g.input(n, 11, x);
        for i in 0..3 {
            let w = g.params.find(&format!("l{i}.weight")).unwrap();
            let b = g.params.find(&format!("l{i}.bias")).unwrap();
            h = g.linear(h, w, Some(b), None);
            if i < 2 {
                h = g.relu(h);
            }
        }
        h
    }

    /// Q-values for `n` states.
    ///
    /// # Example
    /// ```
    /// let pool = aido::runtime::pool::ThreadPool::new(1, None);
    /// let mut q = aido::sdk::snake::QNet::new(1);
    /// assert_eq!(q.q(&pool, &vec![0.0; 11], 1).len(), 3);
    /// ```
    pub fn q(&mut self, pool: &ThreadPool, x: &[f32], n: usize) -> Vec<f32> {
        let mut p = std::mem::take(&mut self.params);
        let v = {
            let mut g = Graph::new(&mut p, pool);
            let o = Self::forward(&mut g, x.to_vec(), n);
            g.value(o).to_vec()
        };
        self.params = p;
        v
    }
}

/// Training statistics.
#[derive(Debug, Clone)]
pub struct RlReport {
    /// Episodes played.
    pub episodes: usize,
    /// Mean score of the last 100 training episodes.
    pub train_mean: f64,
    /// Mean score of greedy evaluation games.
    pub eval_mean: f64,
    /// Best evaluation score.
    pub eval_best: usize,
}

/// Trains a DQN agent and evaluates it greedily.
///
/// * **Input** — episodes, grid size, pool, progress callback
///   `(episode, mean_score_last_50, epsilon)`.
/// * **Output** — trained network and report.
///
/// # Example
/// ```no_run
/// let pool = aido::runtime::pool::ThreadPool::new(4, None);
/// let (q, rep) = aido::sdk::snake::train(300, 10, &pool, &mut |_, _, _| {});
/// println!("{rep:?}");
/// ```
pub fn train(episodes: usize, size: i32, pool: &ThreadPool, progress: &mut dyn FnMut(usize, f64, f32)) -> (QNet, RlReport) {
    let mut net = QNet::new(7);
    let mut target = QNet { params: net.params.clone() };
    let mut opt = crate::sdk::optimizer(&net.params, 1e-3, episodes * 200);
    opt.wd = 0.0;
    opt.warmup = 1;
    let mut replay: VecDeque<(Vec<f32>, usize, f32, Vec<f32>, bool)> = VecDeque::new();
    let mut rng = Rng(99);
    let mut scores = Vec::new();
    let mut steps = 0usize;
    let gamma = 0.9;
    for ep in 0..episodes {
        let eps = (1.0 - ep as f32 / (episodes as f32 * 0.6)).max(0.01);
        let mut game = Snake::new(size, ep as u64 * 31 + 1);
        loop {
            let s = game.state();
            let a = if rng.uniform() < eps {
                (rng.uniform() * 3.0) as usize % 3
            } else {
                let q = net.q(pool, &s, 1);
                (0..3).max_by(|&i, &j| q[i].partial_cmp(&q[j]).unwrap()).unwrap()
            };
            let (r, done) = game.step(a);
            replay.push_back((s, a, r, game.state(), done));
            if replay.len() > 50_000 {
                replay.pop_front();
            }
            steps += 1;
            if replay.len() >= 256 && steps % 2 == 0 {
                let b = 64;
                let batch: Vec<usize> = (0..b).map(|_| (rng.uniform() * replay.len() as f32) as usize % replay.len()).collect();
                let s1: Vec<f32> = batch.iter().flat_map(|&i| replay[i].0.clone()).collect();
                let s2: Vec<f32> = batch.iter().flat_map(|&i| replay[i].3.clone()).collect();
                let qn = target.q(pool, &s2, b);
                let mut p = std::mem::take(&mut net.params);
                {
                    let mut g = Graph::new(&mut p, pool);
                    let out = QNet::forward(&mut g, s1, b);
                    let mut tgt = g.value(out).to_vec();
                    for (k, &i) in batch.iter().enumerate() {
                        let (_, a, r, _, d) = &replay[i];
                        let best = qn[k * 3..k * 3 + 3].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        tgt[k * 3 + a] = r + if *d { 0.0 } else { gamma * best };
                    }
                    let l = g.mse(out, &tgt);
                    g.backward(l);
                }
                net.params = p;
                opt.update(&mut net.params);
                if steps % 500 == 0 {
                    target.params = net.params.clone();
                }
            }
            if done {
                break;
            }
        }
        scores.push(game.score);
        if (ep + 1) % 50 == 0 {
            let m = scores[scores.len() - 50..].iter().sum::<usize>() as f64 / 50.0;
            progress(ep + 1, m, eps);
        }
    }
    let train_mean = scores[scores.len().saturating_sub(100)..].iter().sum::<usize>() as f64 / scores.len().min(100) as f64;
    let mut evals = Vec::new();
    for g in 0..50 {
        let mut game = Snake::new(size, 10_000 + g);
        loop {
            let q = net.q(pool, &game.state(), 1);
            let a = (0..3).max_by(|&i, &j| q[i].partial_cmp(&q[j]).unwrap()).unwrap();
            if game.step(a).1 {
                break;
            }
        }
        evals.push(game.score);
    }
    let rep = RlReport { episodes, train_mean, eval_mean: evals.iter().sum::<usize>() as f64 / evals.len() as f64, eval_best: *evals.iter().max().unwrap() };
    (net, rep)
}
