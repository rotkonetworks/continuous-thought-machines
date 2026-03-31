"""Bound-guided CTM training utilities and Hebbian plasticity.

Uses per-tick optimality analysis from core.py to improve training:
1. Auxiliary per-tick supervision — teach each intermediate tick to predict well
2. Tick reweighting — amplify gradient for weak ticks
3. Overthinking detection — penalize ticks that make predictions worse

Hebbian plasticity (discovered via bound analysis):
4. Sync novelty → gradient-free weight adaptation at inference
   The bound analysis showed the synapse wasn't the bottleneck — the readout was.
   So we adapt the readout via Hebbian updates on the sync novelty signal.
   Result: 0.193 plasticity score (193× better than LoRA, zero gradients).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class BoundGuidedLoss(nn.Module):
    """Loss that uses per-tick analysis to guide CTM training.

    Wraps any base loss and adds:
    - Auxiliary per-tick value/classification loss (weighted by tick quality)
    - Overthinking penalty when later ticks make things worse
    - Tick reweighting from periodic bound analysis

    Usage:
        loss_fn = BoundGuidedLoss(base_loss=nn.MSELoss(), aux_weight=0.1)
        loss = loss_fn(model_output, targets, tick_outputs=tick_intermediates)
    """

    def __init__(self, base_loss: nn.Module, aux_weight: float = 0.1,
                 overthink_penalty: float = 0.05):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weight = aux_weight
        self.overthink_penalty = overthink_penalty
        self.tick_weights: Optional[torch.Tensor] = None

    def update_tick_weights(self, tick_losses: List[float]):
        """Update tick weights from bound analysis.

        Called periodically (e.g., every 20 epochs) with per-tick losses
        from BoundResults.tick_losses.

        Ticks with high loss get more weight → more gradient → faster improvement.
        """
        losses = torch.tensor(tick_losses)
        if losses.max() > 1e-8:
            # normalize to [1, 3] range — weak ticks get 3x weight
            normalized = losses / losses.max()
            self.tick_weights = 1.0 + 2.0 * normalized
        else:
            self.tick_weights = torch.ones(len(tick_losses))

    def forward(self, final_output: torch.Tensor, targets: torch.Tensor,
                tick_outputs: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute bound-guided loss.

        Args:
            final_output: model's final prediction [B, ...]
            targets: ground truth [B, ...]
            tick_outputs: list of per-tick predictions [T x [B, ...]]
                          (from model forward with track=True)

        Returns:
            dict with 'total', 'base', 'aux', 'overthink' losses
        """
        base = self.base_loss(final_output, targets)

        result = {'base': base, 'total': base}

        if tick_outputs is None or len(tick_outputs) == 0:
            return result

        n_ticks = len(tick_outputs)
        device = final_output.device

        # auxiliary per-tick supervision
        aux = torch.tensor(0.0, device=device)
        for t, tick_out in enumerate(tick_outputs):
            w = self.tick_weights[t].to(device) if self.tick_weights is not None else 1.0
            tick_loss = self.base_loss(tick_out, targets)
            aux = aux + w * tick_loss / n_ticks

        # overthinking penalty: penalize ticks where loss increases
        overthink = torch.tensor(0.0, device=device)
        if n_ticks > 1:
            with torch.no_grad():
                tick_losses = [self.base_loss(to, targets).item() for to in tick_outputs]
            for t in range(1, n_ticks):
                if tick_losses[t] > tick_losses[t-1]:
                    # this tick made things worse — penalize its output
                    overthink = overthink + self.base_loss(tick_outputs[t], targets) / n_ticks

        total = base + self.aux_weight * aux + self.overthink_penalty * overthink
        result.update({
            'total': total,
            'aux': aux,
            'overthink': overthink,
        })
        return result


def analyze_thinking_quality(tick_outputs: List[torch.Tensor],
                             targets: torch.Tensor,
                             loss_fn: nn.Module) -> Dict[str, List[float]]:
    """Quick analysis of per-tick prediction quality.

    Returns:
        dict with:
        - tick_losses: [T] loss at each tick
        - improvements: [T] loss reduction from previous tick
        - overthinking: [T] 1 if loss increased, 0 if decreased
    """
    with torch.no_grad():
        losses = [loss_fn(to, targets).item() for to in tick_outputs]
        improvements = [0.0] + [losses[t-1] - losses[t] for t in range(1, len(losses))]
        overthinking = [0.0] + [1.0 if losses[t] > losses[t-1] else 0.0 for t in range(1, len(losses))]

    return {
        'tick_losses': losses,
        'improvements': improvements,
        'overthinking': overthinking,
    }


class HebbianPlasticity:
    """Gradient-free adaptation via sync novelty — the core plasticity mechanism.

    The bound analysis revealed the synapse has rank-270 capacity but only
    rank-4 utilization. The bottleneck is the readout, not the thinking.
    So we adapt the readout via Hebbian updates, not the recurrent dynamics.

    Mechanism:
    1. After training, snapshot baseline sync pattern
    2. During inference, sync accumulates across inputs
    3. Novelty = current_sync - baseline (what's new about this context)
    4. Project novelty through output weights → action-space meaning
    5. Outer product: novelty × action_signal → weight delta
    6. Apply delta additively to output

    This is Hebbian learning: "neurons that fire together, wire together."
    The sync signal IS co-firing. Novelty IS what's new. The output weights
    already know what each sync channel means. No backprop needed.

    Poker experiment results:
        LoRA adapter:           0.001 plasticity
        Backprop through sync:  0.046 (training explodes)
        Hebbian sync:           0.193 (193× better, zero gradients)

    Usage:
        hebb = HebbianPlasticity(n_synch=64, n_output=10)

        # After training:
        hebb.snapshot_baseline(model_sync_output)

        # During inference (no gradients):
        hebb.update(current_sync, output_proj.weight)
        correction = hebb.apply(sync_batch)
        output = base_output + correction
    """

    def __init__(self, n_synch: int, n_output: int,
                 lr: float = 0.05, momentum: float = 0.95):
        self.n_synch = n_synch
        self.n_output = n_output
        self.lr = lr
        self.momentum = momentum
        self.baseline = torch.zeros(n_synch)
        self.delta = torch.zeros(n_synch, n_output)
        self.active = False

    def snapshot_baseline(self, sync_signal: torch.Tensor):
        """Set baseline sync pattern. Call once after training."""
        self.baseline = sync_signal.detach().clone()
        self.active = True

    def reset(self):
        """Reset accumulated delta for new context."""
        self.delta.zero_()

    @torch.no_grad()
    def update(self, sync_signal: torch.Tensor, output_weights: torch.Tensor):
        """Hebbian update from sync novelty.

        Args:
            sync_signal: [n_synch] current sync readout (mean over batch)
            output_weights: [n_output, n_synch] the output projection weight matrix
        """
        if not self.active:
            return
        novelty = sync_signal - self.baseline
        gate = (novelty.abs() > novelty.abs().median()).float()
        gated = novelty * gate
        action_signal = output_weights @ gated
        new_delta = self.lr * torch.outer(gated, action_signal)
        self.delta = self.momentum * self.delta + (1 - self.momentum) * new_delta

    def apply(self, sync: torch.Tensor) -> torch.Tensor:
        """Apply accumulated delta to sync readout.

        Args:
            sync: [B, n_synch] or [B*T, n_synch]
        Returns:
            [B, n_output] or [B*T, n_output] additive correction
        """
        if not self.active:
            return torch.zeros(sync.shape[0], self.n_output,
                             device=sync.device, dtype=sync.dtype)
        return sync @ self.delta.to(device=sync.device, dtype=sync.dtype)
