//! Hebbian plasticity engine — pure math, no UI dependencies.
//!
//! Faithfully replicates the Python HebbianPlasticity from training.py:
//!   novelty = sync - baseline
//!   gate = |novelty| > percentile(|novelty|, pct)
//!   gated = novelty * gate
//!   action = W @ gated         (matvec [n_output, n_synch] × [n_synch])
//!   delta = momentum * delta + (1-momentum) * lr * outer(gated, action)
//!   correction = sync @ delta   (matvec [n_synch, n_output])

/// Hebbian adaptation engine operating on cached sync signals.
pub struct HebbianEngine {
    pub n_synch: usize,
    pub n_output: usize,

    // User-controllable parameters
    pub lr: f32,
    pub momentum: f32,
    pub gate_percentile: f32, // 0-100, default 50 (median)

    // Accumulated correction matrix [n_synch × n_output], row-major
    pub delta: Vec<f32>,

    // Pre-loaded from model (immutable after init)
    pub baseline_sync: Vec<f32>,    // [n_synch]
    pub output_weights: Vec<f32>,   // [n_output × n_synch], row-major
}

impl HebbianEngine {
    pub fn new(
        n_synch: usize,
        n_output: usize,
        baseline_sync: Vec<f32>,
        output_weights: Vec<f32>,
    ) -> Self {
        assert_eq!(baseline_sync.len(), n_synch);
        assert_eq!(output_weights.len(), n_output * n_synch);
        Self {
            n_synch,
            n_output,
            lr: 0.3,
            momentum: 0.95,
            gate_percentile: 50.0,
            delta: vec![0.0; n_synch * n_output],
            baseline_sync,
            output_weights,
        }
    }

    pub fn reset(&mut self) {
        self.delta.fill(0.0);
    }

    /// Reward-modulated Hebbian update from one sync observation.
    /// Only updates when `reward` is true (positive reinforcement).
    pub fn update(&mut self, sync: &[f32], reward: bool) {
        if !reward {
            return;
        }

        // novelty = sync - baseline
        let mut novelty = vec![0.0f32; self.n_synch];
        for i in 0..self.n_synch {
            novelty[i] = sync[i] - self.baseline_sync[i];
        }

        // Gate: keep dimensions where |novelty| > percentile threshold
        let threshold = percentile_abs(&novelty, self.gate_percentile);
        let mut gated = vec![0.0f32; self.n_synch];
        for i in 0..self.n_synch {
            if novelty[i].abs() > threshold {
                gated[i] = novelty[i];
            }
        }

        // action_signal = output_weights @ gated  → [n_output]
        // W is [n_output, n_synch] row-major
        let mut action = vec![0.0f32; self.n_output];
        for o in 0..self.n_output {
            let row_start = o * self.n_synch;
            let mut sum = 0.0f32;
            for s in 0..self.n_synch {
                sum += self.output_weights[row_start + s] * gated[s];
            }
            action[o] = sum;
        }

        // delta = momentum * delta + (1-momentum) * lr * outer(gated, action)
        // delta is [n_synch, n_output] row-major
        let scale = (1.0 - self.momentum) * self.lr;
        let mom = self.momentum;
        for s in 0..self.n_synch {
            let row_start = s * self.n_output;
            let g = gated[s];
            if g == 0.0 {
                // Only apply momentum decay for this row
                for o in 0..self.n_output {
                    self.delta[row_start + o] *= mom;
                }
            } else {
                for o in 0..self.n_output {
                    self.delta[row_start + o] =
                        mom * self.delta[row_start + o] + scale * g * action[o];
                }
            }
        }
    }

    /// Apply correction: returns corrected logits = base_logits + sync @ delta
    pub fn apply(&self, sync: &[f32], base_logits: &[f32]) -> Vec<f32> {
        assert_eq!(sync.len(), self.n_synch);
        assert_eq!(base_logits.len(), self.n_output);

        // correction = sync @ delta  where delta is [n_synch, n_output]
        let mut result = vec![0.0f32; self.n_output];
        for o in 0..self.n_output {
            result[o] = base_logits[o];
        }
        for s in 0..self.n_synch {
            let sv = sync[s];
            if sv == 0.0 {
                continue;
            }
            let row_start = s * self.n_output;
            for o in 0..self.n_output {
                result[o] += sv * self.delta[row_start + o];
            }
        }
        result
    }

    /// Apply and return argmax prediction.
    pub fn predict(&self, sync: &[f32], base_logits: &[f32]) -> usize {
        let logits = self.apply(sync, base_logits);
        argmax(&logits)
    }

    /// Current delta matrix L2 norm.
    pub fn delta_norm(&self) -> f32 {
        self.delta.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Evaluation state: runs Hebbian engine over cached dataset.
pub struct EvalState {
    // Data [all flat, row-major]
    pub sync_signals: Vec<f32>,   // [N × n_synch]
    pub logits_final: Vec<f32>,   // [N × n_output]
    pub logits_t10: Vec<f32>,     // [N × n_output]
    pub labels: Vec<u16>,         // [N]
    pub sync_pca: Vec<f32>,       // [N × 2]
    pub n_images: usize,
    pub n_synch: usize,
    pub n_output: usize,
    pub class_names: Vec<String>,

    // Results (recomputed when params change)
    pub base_correct: Vec<bool>,
    pub hebbian_correct: Vec<bool>,
    pub hebbian_preds: Vec<u16>,
    pub cumulative_acc: Vec<f32>,   // running accuracy at each step
    pub delta_norms: Vec<f32>,

    // Animation state
    pub current_step: usize,
    pub dirty: bool,
}

impl EvalState {
    pub fn new(
        sync_signals: Vec<f32>,
        logits_final: Vec<f32>,
        logits_t10: Vec<f32>,
        labels: Vec<u16>,
        sync_pca: Vec<f32>,
        n_images: usize,
        n_synch: usize,
        n_output: usize,
        class_names: Vec<String>,
    ) -> Self {
        // Compute base accuracy (no adaptation)
        let mut base_correct = vec![false; n_images];
        for i in 0..n_images {
            let offset = i * n_output;
            let logits = &logits_final[offset..offset + n_output];
            let pred = argmax(logits);
            base_correct[i] = pred == labels[i] as usize;
        }

        Self {
            sync_signals, logits_final, logits_t10, labels, sync_pca,
            n_images, n_synch, n_output, class_names,
            base_correct,
            hebbian_correct: vec![false; n_images],
            hebbian_preds: vec![0; n_images],
            cumulative_acc: vec![0.0; n_images],
            delta_norms: vec![0.0; n_images],
            current_step: 0,
            dirty: true,
        }
    }

    /// Run full evaluation with the given engine. ~2s for 200 images in WASM.
    pub fn run_full(&mut self, engine: &mut HebbianEngine, use_t10: bool) {
        engine.reset();
        let mut correct_so_far = 0u32;

        for i in 0..self.n_images {
            let sync = &self.sync_signals[i * self.n_synch..(i + 1) * self.n_synch];
            let logits = if use_t10 {
                &self.logits_t10[i * self.n_output..(i + 1) * self.n_output]
            } else {
                &self.logits_final[i * self.n_output..(i + 1) * self.n_output]
            };

            let pred = engine.predict(sync, logits);
            let label = self.labels[i] as usize;
            let correct = pred == label;

            self.hebbian_correct[i] = correct;
            self.hebbian_preds[i] = pred as u16;

            // Reward-modulated update (positive reinforcement only)
            engine.update(sync, correct);

            if correct {
                correct_so_far += 1;
            }
            self.cumulative_acc[i] = correct_so_far as f32 / (i + 1) as f32;
            self.delta_norms[i] = engine.delta_norm();
        }

        self.current_step = self.n_images;
        self.dirty = false;
    }

    /// Run one step (for animation). Returns true when done.
    pub fn run_step(&mut self, engine: &mut HebbianEngine, use_t10: bool) -> bool {
        if self.current_step >= self.n_images {
            return true;
        }

        let i = self.current_step;
        let sync = &self.sync_signals[i * self.n_synch..(i + 1) * self.n_synch];
        let logits = if use_t10 {
            &self.logits_t10[i * self.n_output..(i + 1) * self.n_output]
        } else {
            &self.logits_final[i * self.n_output..(i + 1) * self.n_output]
        };

        let pred = engine.predict(sync, logits);
        let label = self.labels[i] as usize;
        let correct = pred == label;

        self.hebbian_correct[i] = correct;
        self.hebbian_preds[i] = pred as u16;
        engine.update(sync, correct);

        let prev_correct: u32 = if i > 0 {
            (self.cumulative_acc[i - 1] * i as f32).round() as u32
        } else {
            0
        };
        let total_correct = prev_correct + if correct { 1 } else { 0 };
        self.cumulative_acc[i] = total_correct as f32 / (i + 1) as f32;
        self.delta_norms[i] = engine.delta_norm();

        self.current_step += 1;
        self.current_step >= self.n_images
    }

    pub fn base_accuracy(&self) -> f32 {
        let c = self.base_correct.iter().filter(|&&x| x).count();
        c as f32 / self.n_images as f32
    }

    pub fn hebbian_accuracy(&self) -> f32 {
        if self.current_step == 0 {
            return 0.0;
        }
        let n = self.current_step.min(self.n_images);
        let c = self.hebbian_correct[..n].iter().filter(|&&x| x).count();
        c as f32 / n as f32
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn percentile_abs(v: &[f32], pct: f32) -> f32 {
    let mut abs_vals: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((pct / 100.0) * (abs_vals.len() - 1) as f32) as usize;
    abs_vals[idx.min(abs_vals.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -3.0, -2.0]), 0);
    }

    #[test]
    fn test_percentile() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_abs(&v, 50.0) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_hebbian_update() {
        let n_s = 4;
        let n_o = 2;
        let baseline = vec![0.0; n_s];
        // Asymmetric weights so action signal doesn't cancel out
        let weights = vec![1.0, 0.5, 0.3, 0.1,
                           0.2, 0.8, 0.4, 0.6];

        let mut engine = HebbianEngine::new(n_s, n_o, baseline, weights);
        engine.lr = 1.0;
        engine.momentum = 0.0;

        // Sync with clear asymmetry — gated values won't sum to zero
        let sync = vec![2.0, 0.1, 0.8, 1.5];
        engine.update(&sync, true);

        assert!(engine.delta_norm() > 0.0, "delta should be non-zero after positive reward");

        // No update on negative reward
        let norm_before = engine.delta_norm();
        engine.update(&sync, false);
        // With momentum=0, delta should stay the same (no decay applied when reward=false)
        // Actually our code returns early before touching delta, so it stays identical
        assert_eq!(engine.delta_norm(), norm_before);
    }

    #[test]
    fn test_apply_no_delta() {
        let engine = HebbianEngine::new(
            3, 2, vec![0.0; 3], vec![0.0; 6],
        );
        let result = engine.apply(&[1.0, 2.0, 3.0], &[10.0, 20.0]);
        assert_eq!(result, vec![10.0, 20.0]); // no correction when delta is zero
    }
}
