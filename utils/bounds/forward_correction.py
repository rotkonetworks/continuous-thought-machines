"""Per-tick forward correction guided by optimality bounds.

At each thinking step, compute the optimal NLM weights via
least-squares on the observed trace, then nudge the activations
toward what's optimal. Feed the corrected state to the next tick.

This is Hebbian learning done right:
- The SDP bound gives the target (what optimal weights produce)
- The correction is the difference (achieved - optimal)
- Applied additively at each tick (no backprop)
- The next tick gets better information

Usage:
    corrector = ForwardCorrector(model)
    preds = corrector.forward_with_correction(x)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ForwardCorrector:
    """Inline per-tick correction during CTM forward pass.

    At each tick t:
    1. Model computes pre-activation (synapse output)
    2. NLM maps trace → post-activation
    3. We compute: what WOULD optimal NLM weights produce?
    4. Nudge post-activation toward optimal by α * (optimal - achieved)
    5. Feed corrected activation to next tick

    The "optimal" is computed per-neuron via least-squares on the
    trace history — the same thing the SDP bounds optimize, but
    solved in closed form.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: correction strength [0, 1]. 0 = no correction, 1 = full replacement.
                   Start small — even α=0.05 can help without destabilizing.
        """
        self.alpha = alpha

    @torch.no_grad()
    def compute_optimal_activation(self, trace: torch.Tensor,
                                     target: Optional[torch.Tensor] = None):
        """Compute what optimal NLM weights would produce.

        For a linear NLM: h = W @ trace
        Optimal W minimizes ||W @ trace - target||²
        Solution: W* = target @ trace^T @ (trace @ trace^T)^{-1}

        If no target is given, use the mean activation as target
        (homeostatic: drive toward the average behavior).

        Args:
            trace: [B, D, M] trace history at current tick
            target: [B, D] target activations (optional)

        Returns:
            [B, D] optimal activations
        """
        B, D, M = trace.shape

        if target is None:
            # Homeostatic target: mean over batch
            # This encourages diverse activations (anti-collapse)
            target = trace.mean(dim=-1)  # [B, D] — mean over memory

        # Per-neuron least squares: for each neuron d,
        # find w_d that minimizes ||w_d^T @ trace_d - target_d||²
        # across the batch.
        #
        # trace_d: [B, M], target_d: [B, 1]
        # w_d* = (Σ trace_d @ trace_d^T)^{-1} @ (Σ trace_d * target_d)
        #
        # But this is expensive for D=4096 neurons.
        # Simpler: use the current trace to estimate what better weights
        # would produce by projecting toward the trace mean.

        # Fast approximation: optimal activation ≈ trace @ (trace^T trace)^{-1} trace^T @ target
        # For M=8, (M×M) inverse is trivial
        # trace: [B, D, M]

        # Batch-level statistics per neuron
        # trace_flat: [B*D, M]
        trace_flat = trace.reshape(B * D, M)
        target_flat = target.reshape(B * D)

        # Gram matrix: [M, M] (averaged over batch*neurons)
        gram = trace_flat.T @ trace_flat  # [M, M]
        gram += 1e-6 * torch.eye(M, device=trace.device)  # regularize

        # Cross-correlation: [M]
        cross = trace_flat.T @ target_flat  # [M]

        # Optimal weights: [M]
        w_opt = torch.linalg.solve(gram, cross)  # [M]

        # Optimal activations: [B*D]
        optimal = trace_flat @ w_opt  # [B*D]

        return optimal.reshape(B, D)

    @torch.no_grad()
    def correct(self, activated_state: torch.Tensor,
                trace: torch.Tensor,
                target: Optional[torch.Tensor] = None):
        """Apply per-tick correction to activated state.

        Args:
            activated_state: [B, D] current post-activation from NLM
            trace: [B, D, M] current trace history
            target: [B, D] optional target (default: homeostatic)

        Returns:
            [B, D] corrected activation
        """
        optimal = self.compute_optimal_activation(trace, target)

        # Nudge toward optimal
        correction = optimal - activated_state
        corrected = activated_state + self.alpha * correction

        return corrected


def test_forward_correction():
    """Test on QEC decoder."""
    from tasks.qec.surface_code import SurfaceCode
    from tasks.qec.ctm_qec import ContinuousThoughtMachineQEC

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    code = SurfaceCode(5)
    syndrome_dim = 5 * code.n_stab

    ckpt = torch.load('checkpoints/qec/best.pt', map_location='cpu', weights_only=False)
    model = ContinuousThoughtMachineQEC(
        syndrome_dim=syndrome_dim, d_model=256, iterations=16,
        n_synch_out=64, synapse_depth=1, memory_length=8,
    ).to(device)
    with torch.no_grad():
        model(torch.randn(2, syndrome_dim).to(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Test: run with and without correction
    N = 2000
    corrector = ForwardCorrector(alpha=0.1)

    for alpha in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
        corrector.alpha = alpha
        correct = 0

        with torch.no_grad():
            for _ in range(N // 64):
                syndromes, labels = code.generate_syndromes(64, 0.05, 5, device=device)
                x = syndromes.flatten(1)

                # Manual forward pass with per-tick correction
                B = x.size(0)
                state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1).clone()
                activated_state = model.start_activated_state.unsqueeze(0).expand(B, -1).clone()

                r_out = torch.exp(-torch.clamp(model.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
                _, decay_alpha_out, decay_beta_out = model.compute_synchronisation(
                    activated_state, None, None, r_out, synch_type='out')

                for t in range(model.iterations):
                    pre_synapse = torch.cat((x, activated_state), dim=-1)
                    state = model.synapses(pre_synapse)
                    state_trace = torch.cat(
                        (state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

                    activated_state = model.trace_processor(state_trace)

                    # === CORRECTION POINT ===
                    if alpha > 0:
                        activated_state = corrector.correct(
                            activated_state, state_trace)

                    sync_out, decay_alpha_out, decay_beta_out = \
                        model.compute_synchronisation(
                            activated_state, decay_alpha_out, decay_beta_out,
                            r_out, synch_type='out')

                # Final prediction from last sync
                pred = model.output_projector(sync_out)
                correct += (pred.argmax(1) == labels).sum().item()

        acc = correct / N
        print(f'  alpha={alpha:.2f}: {acc:.1%}')


if __name__ == '__main__':
    print('Forward correction test on QEC decoder:')
    test_forward_correction()
