//! Per-tick forward correction via least-squares optimal NLM weights.
//!
//! At each thinking step, compute what optimal weights would produce
//! and nudge the activations toward that. This is the core innovation:
//! SDP bound analysis → inline correction → better thinking.

/// Forward corrector: computes optimal activation per tick via least-squares.
pub struct ForwardCorrector {
    pub alpha: f32,  // correction strength [0, 1]
    pub n_dims: usize,
    pub memory_length: usize,
}

impl ForwardCorrector {
    pub fn new(n_dims: usize, memory_length: usize, alpha: f32) -> Self {
        Self { alpha, n_dims, memory_length }
    }

    /// Compute optimal activation from trace history via least-squares.
    ///
    /// For neuron d: h_d = w_d^T @ trace_d
    /// Optimal w_d minimizes ||w_d^T @ trace_d - target_d||²
    /// Solution: w* = (trace^T trace)^{-1} trace^T target
    ///
    /// Here we use the trace mean as homeostatic target.
    ///
    /// trace: [n_dims, memory_length] (column-major per neuron)
    /// activated: [n_dims] current activations
    ///
    /// Returns: corrected [n_dims] activations
    pub fn correct(&self, activated: &[f32], trace: &[f32]) -> Vec<f32> {
        if self.alpha == 0.0 {
            return activated.to_vec();
        }

        let d = self.n_dims;
        let m = self.memory_length;
        assert_eq!(activated.len(), d);
        assert_eq!(trace.len(), d * m);

        // Homeostatic target: mean over memory for each neuron
        let mut target = vec![0.0f32; d];
        for dim in 0..d {
            let base = dim * m;
            let mut sum = 0.0f32;
            for j in 0..m {
                sum += trace[base + j];
            }
            target[dim] = sum / m as f32;
        }

        // Gram matrix: M×M (shared across neurons for efficiency)
        // gram[i][j] = Σ_d trace[d,i] * trace[d,j]
        let mut gram = vec![0.0f32; m * m];
        let mut cross = vec![0.0f32; m]; // Σ_d trace[d,:] * target[d]

        for dim in 0..d {
            let base = dim * m;
            for i in 0..m {
                let ti = trace[base + i];
                cross[i] += ti * target[dim];
                for j in i..m {
                    let tj = trace[base + j];
                    gram[i * m + j] += ti * tj;
                    if i != j {
                        gram[j * m + i] += ti * tj; // symmetric
                    }
                }
            }
        }

        // Regularize
        for i in 0..m {
            gram[i * m + i] += 1e-6;
        }

        // Solve gram @ w = cross via Cholesky (M is small, 6-16)
        let w_opt = solve_symmetric(m, &gram, &cross);

        // Compute optimal activations: for each neuron, h* = w_opt^T @ trace_d
        let mut optimal = vec![0.0f32; d];
        for dim in 0..d {
            let base = dim * m;
            let mut sum = 0.0f32;
            for j in 0..m {
                sum += w_opt[j] * trace[base + j];
            }
            optimal[dim] = sum;
        }

        // Nudge: corrected = activated + alpha * (optimal - activated)
        let mut corrected = vec![0.0f32; d];
        for dim in 0..d {
            corrected[dim] = activated[dim] + self.alpha * (optimal[dim] - activated[dim]);
        }

        corrected
    }

    /// Compute the gap at this tick: ||activated - optimal||² / ||activated||²
    pub fn compute_gap(&self, activated: &[f32], trace: &[f32]) -> f32 {
        let optimal = self.correct_full(activated, trace);
        let mut gap_sq = 0.0f32;
        let mut act_sq = 0.0f32;
        for i in 0..self.n_dims {
            let diff = activated[i] - optimal[i];
            gap_sq += diff * diff;
            act_sq += activated[i] * activated[i];
        }
        if act_sq > 1e-10 { gap_sq / act_sq } else { 0.0 }
    }

    /// Full optimal (alpha=1.0) for gap computation
    fn correct_full(&self, activated: &[f32], trace: &[f32]) -> Vec<f32> {
        let temp = Self { alpha: 1.0, ..*self };
        temp.correct(activated, trace)
    }
}

/// Solve Ax = b for symmetric positive definite A via Cholesky.
/// Falls back to diagonal solve if Cholesky fails.
fn solve_symmetric(n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // Try Cholesky: A = L L^T
    let mut l = vec![0.0f32; n * n];
    let mut ok = true;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = a[i * n + i] - sum;
                if val <= 0.0 {
                    ok = false;
                    break;
                }
                l[i * n + j] = val.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
        if !ok { break; }
    }

    if !ok {
        // Fallback: diagonal solve
        let mut x = vec![0.0f32; n];
        for i in 0..n {
            let diag = a[i * n + i];
            x[i] = if diag.abs() > 1e-10 { b[i] / diag } else { 0.0 };
        }
        return x;
    }

    // Forward solve: L y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    // Backward solve: L^T x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f32;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k]; // L^T[i,k] = L[k,i]
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_symmetric() {
        // 2x2: [[4, 2], [2, 3]] x = [8, 7] → x = [1, 1] (check: 4+2=6≠8... let me fix)
        // Actually: [[2, 1], [1, 2]] x = [3, 3] → x = [1, 1]
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let b = vec![3.0, 3.0];
        let x = solve_symmetric(2, &a, &b);
        assert!((x[0] - 1.0).abs() < 0.01);
        assert!((x[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_zero_alpha_no_change() {
        let corr = ForwardCorrector::new(4, 2, 0.0);
        let activated = vec![1.0, 2.0, 3.0, 4.0];
        let trace = vec![0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0];
        let result = corr.correct(&activated, &trace);
        assert_eq!(result, activated);
    }

    #[test]
    fn test_correction_moves_toward_optimal() {
        let corr = ForwardCorrector::new(2, 2, 0.5);
        let activated = vec![10.0, 10.0]; // far from trace mean
        let trace = vec![1.0, 1.0, 1.0, 1.0]; // trace mean = 1.0
        let result = corr.correct(&activated, &trace);
        // Should move toward ~1.0 from 10.0
        assert!(result[0] < 10.0);
        assert!(result[1] < 10.0);
    }
}
