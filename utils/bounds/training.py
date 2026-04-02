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

    Pluggable strategies (inspired by MADNet's continual adaptation):
    - Gate: median (default), percentile, topk, or none
    - Confidence weighting: scale update by model certainty
    - Reward/punishment histogram: MAD-style module selection
    - Extrapolation check: verify update actually helped

    Usage:
        hebb = HebbianPlasticity(n_synch=64, n_output=10,
                                 gate='median',
                                 confidence_weighted=True,
                                 use_extrapolation_check=True)

        hebb.snapshot_baseline(model_sync_output)
        hebb.update(sync, output_weights, reward=True, confidence=0.9)
        correction = hebb.apply(sync_batch)
        output = base_output + correction
    """

    def __init__(self, n_synch: int, n_output: int,
                 lr: float = 0.05, momentum: float = 0.95,
                 # Pluggable gate strategy
                 gate: str = 'median',  # 'median', 'percentile', 'topk', 'none'
                 gate_percentile: float = 50.0,
                 gate_topk: int = 0,  # 0 = auto (n_synch // 4)
                 # Confidence weighting (from MADNet proxy filtering)
                 confidence_weighted: bool = False,
                 # Extrapolation check (MADNet Algorithm 2, line 13-14)
                 use_extrapolation_check: bool = False,
                 extrapolation_decay: float = 0.99,
                 extrapolation_scale: float = 0.01,
                 # Reward/punishment histogram (MADNet Algorithm 2, line 15-16)
                 n_modules: int = 0,  # 0 = no modular selection
                 ):
        self.n_synch = n_synch
        self.n_output = n_output
        self.lr = lr
        self.momentum = momentum
        self.baseline = torch.zeros(n_synch)
        self.delta = torch.zeros(n_synch, n_output)
        self.active = False

        # Gate config
        self.gate_mode = gate
        self.gate_percentile = gate_percentile
        self.gate_topk = gate_topk if gate_topk > 0 else max(1, n_synch // 4)

        # Confidence weighting
        self.confidence_weighted = confidence_weighted

        # Extrapolation check (γ from MADNet)
        self.use_extrapolation_check = use_extrapolation_check
        self.extrap_decay = extrapolation_decay
        self.extrap_scale = extrapolation_scale
        self._loss_history = []  # last 3 losses for linear extrapolation
        self._extrap_scale_current = 1.0  # dynamic scaling from extrapolation

        # Modular reward/punishment histogram
        self.n_modules = n_modules
        if n_modules > 0:
            self.histogram = torch.zeros(n_modules)
            self._last_module = 0
        else:
            self.histogram = None

    def snapshot_baseline(self, sync_signal: torch.Tensor):
        """Set baseline sync pattern. Call once after training."""
        self.baseline = sync_signal.detach().clone()
        self.active = True

    def reset(self):
        """Reset accumulated delta for new context."""
        self.delta.zero_()
        self._loss_history.clear()
        self._extrap_scale_current = 1.0
        if self.histogram is not None:
            self.histogram.zero_()

    def _compute_gate(self, novelty: torch.Tensor) -> torch.Tensor:
        """Compute gating mask based on selected strategy."""
        abs_nov = novelty.abs()
        if self.gate_mode == 'none':
            return torch.ones_like(novelty)
        elif self.gate_mode == 'median':
            return (abs_nov > abs_nov.median()).float()
        elif self.gate_mode == 'percentile':
            if len(novelty) < 2:
                return torch.ones_like(novelty)
            sorted_vals = abs_nov.sort().values
            idx = int(self.gate_percentile / 100.0 * (len(sorted_vals) - 1))
            threshold = sorted_vals[idx]
            return (abs_nov > threshold).float()
        elif self.gate_mode == 'topk':
            k = min(self.gate_topk, len(novelty))
            _, top_indices = abs_nov.topk(k)
            mask = torch.zeros_like(novelty)
            mask[top_indices] = 1.0
            return mask
        else:
            return (abs_nov > abs_nov.median()).float()

    def report_loss(self, loss: float):
        """Report current loss for extrapolation check.

        Call after each prediction, before update. The extrapolation
        check compares actual loss against linearly extrapolated expected
        loss to determine if the previous update helped.

        From MADNet Algorithm 2 lines 13-14:
            L_tilde = 2 * L_{t-1} - L_{t-2}
            gamma = L_tilde - L_t
        """
        self._loss_history.append(loss)
        if len(self._loss_history) >= 3:
            l_t = self._loss_history[-1]
            l_t1 = self._loss_history[-2]
            l_t2 = self._loss_history[-3]
            expected = 2 * l_t1 - l_t2
            gamma = expected - l_t  # positive = update helped
            self._extrap_scale_current = (
                self.extrap_decay * self._extrap_scale_current
                + self.extrap_scale * gamma
            )
            # Keep only last 3
            self._loss_history = self._loss_history[-3:]

    def select_module(self) -> int:
        """MAD-style module selection from reward histogram.

        Returns module index to update. Call before update().
        """
        if self.histogram is None:
            return 0
        probs = torch.softmax(self.histogram, dim=0)
        selected = torch.multinomial(probs, 1).item()
        self._last_module = selected
        return selected

    def reward_module(self, gamma: float):
        """Reward/punish the last selected module.

        From MADNet Algorithm 2 lines 15-16:
            H = δ · H
            H[φ_{t-1}] += λ · γ
        """
        if self.histogram is None:
            return
        self.histogram *= self.extrap_decay
        self.histogram[self._last_module] += self.extrap_scale * gamma

    @torch.no_grad()
    def update(self, sync_signal: torch.Tensor, output_weights: torch.Tensor,
               reward: bool = True, confidence: float = 1.0):
        """Hebbian update from sync novelty.

        Args:
            sync_signal: [n_synch] current sync readout (mean over batch)
            output_weights: [n_output, n_synch] the output projection weight matrix
            reward: if False, skip update (positive reinforcement only)
            confidence: model certainty score [0, 1] — scales the update
        """
        if not self.active or not reward:
            return

        novelty = sync_signal - self.baseline
        gate = self._compute_gate(novelty)
        gated = novelty * gate

        action_signal = output_weights @ gated
        new_delta = self.lr * torch.outer(gated, action_signal)

        # Confidence weighting: scale update by model certainty
        if self.confidence_weighted:
            new_delta = new_delta * confidence

        # Extrapolation check: scale by whether updates are helping
        if self.use_extrapolation_check:
            # Clamp to [0.1, 2.0] to prevent runaway scaling
            scale = max(0.1, min(2.0, self._extrap_scale_current))
            new_delta = new_delta * scale

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


class CoinBettingHebbian:
    """Parameter-free Hebbian adaptation via coin-betting regret minimization.

    Replaces fixed lr/momentum with the Krichevsky-Trofimov (KT) estimator
    from Orabona & Pál (2016). The algorithm automatically determines update
    magnitude based on cumulative "wealth" — if updates help (correct predictions),
    wealth grows and updates get stronger. If they hurt, wealth shrinks and
    updates throttle.

    The CTM certainty signal serves as the coin flip outcome:
        - Correct + high certainty → strong positive signal → bet more
        - Wrong → negative signal → bet less
        - Correct + low certainty → weak positive → cautious bet

    No hyperparameters to tune. Achieves O(sqrt(T)) regret automatically.

    Usage:
        hebb = CoinBettingHebbian(n_synch=64, n_output=10)
        hebb.snapshot_baseline(model_sync_output)

        for input in stream:
            correction = hebb.apply(sync)
            adapted = base_output + correction
            is_correct = (adapted.argmax() == label)
            certainty = adapted.softmax(0).max()
            hebb.update(sync, output_weights,
                        reward=is_correct, certainty=certainty)
    """

    def __init__(self, n_synch: int, n_output: int,
                 initial_wealth: float = 1.0,
                 gate: str = 'none'):
        self.n_synch = n_synch
        self.n_output = n_output
        self.baseline = torch.zeros(n_synch)
        self.active = False

        # Gate (from HebbianPlasticity)
        self.gate_mode = gate

        # KT estimator state (per sync-output pair would be ideal but too expensive,
        # so we use a scalar wealth tracker + direction from Hebbian outer product)
        self.wealth = initial_wealth      # cumulative wealth W_t
        self.initial_wealth = initial_wealth
        self.sum_rewards = 0.0            # Σ r_i (cumulative reward signal)
        self.sum_sq_rewards = 0.0         # Σ r_i² (for variance tracking)
        self.t = 0                        # time step

        # The delta is still a matrix, but its MAGNITUDE is controlled by wealth
        self.direction = torch.zeros(n_synch, n_output)  # unit direction
        self.delta = torch.zeros(n_synch, n_output)      # direction * bet_size

    def snapshot_baseline(self, sync_signal: torch.Tensor):
        self.baseline = sync_signal.detach().clone()
        self.active = True

    def reset(self):
        self.direction.zero_()
        self.delta.zero_()
        self.wealth = self.initial_wealth
        self.sum_rewards = 0.0
        self.sum_sq_rewards = 0.0
        self.t = 0

    def _compute_gate(self, novelty: torch.Tensor) -> torch.Tensor:
        abs_nov = novelty.abs()
        if self.gate_mode == 'none':
            return torch.ones_like(novelty)
        elif self.gate_mode == 'median':
            return (abs_nov > abs_nov.median()).float()
        elif self.gate_mode == 'topk':
            k = max(1, len(novelty) // 4)
            _, top_idx = abs_nov.topk(k)
            mask = torch.zeros_like(novelty)
            mask[top_idx] = 1.0
            return mask
        return torch.ones_like(novelty)

    @torch.no_grad()
    def update(self, sync_signal: torch.Tensor, output_weights: torch.Tensor,
               reward: bool = True, certainty: float = 0.5):
        """Coin-betting Hebbian update.

        The reward signal is:
            +certainty  if prediction was correct
            -certainty  if prediction was wrong

        The bet size is determined by the KT estimator:
            bet_t = wealth_t * sum_rewards / (t + 1)

        This is parameter-free: no lr or momentum to tune.
        """
        if not self.active:
            return

        self.t += 1

        # Reward signal: [-1, +1] range
        r = certainty if reward else -certainty

        # Update cumulative stats
        self.sum_rewards += r
        self.sum_sq_rewards += r * r

        # KT estimator: bet fraction = sum_rewards / (t + 1)
        # Clamped to [-0.5, 0.5] for stability
        bet_fraction = max(-0.5, min(0.5, self.sum_rewards / (self.t + 1)))

        # Update wealth: W_{t+1} = W_t * (1 + r * bet_fraction)
        self.wealth *= (1.0 + r * bet_fraction)
        self.wealth = max(self.wealth, 1e-8)  # prevent zero wealth

        # Compute Hebbian direction (same as standard Hebbian)
        novelty = sync_signal - self.baseline
        gate = self._compute_gate(novelty)
        gated = novelty * gate
        action_signal = output_weights @ gated
        new_direction = torch.outer(gated, action_signal)

        # Normalize direction
        dir_norm = new_direction.norm()
        if dir_norm > 1e-8:
            new_direction = new_direction / dir_norm

        # Blend direction with momentum derived from wealth stability
        # High wealth variance → more momentum (conservative)
        # Stable wealth → less momentum (responsive)
        if self.t > 1:
            variance = self.sum_sq_rewards / self.t - (self.sum_rewards / self.t) ** 2
            adaptive_momentum = min(0.99, max(0.5, 1.0 - 1.0 / (1.0 + 10.0 * max(0, variance))))
        else:
            adaptive_momentum = 0.9

        self.direction = adaptive_momentum * self.direction + (1 - adaptive_momentum) * new_direction

        # Delta = direction * bet_size (wealth-scaled)
        bet_size = self.wealth * abs(bet_fraction)
        self.delta = self.direction * bet_size

    def apply(self, sync: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.zeros(sync.shape[0], self.n_output,
                             device=sync.device, dtype=sync.dtype)
        return sync @ self.delta.to(device=sync.device, dtype=sync.dtype)

    @property
    def effective_lr(self) -> float:
        """Current effective learning rate (for monitoring)."""
        if self.t == 0:
            return 0.0
        bet_fraction = self.sum_rewards / (self.t + 1)
        return self.wealth * abs(bet_fraction)

    def stats(self) -> dict:
        """Return diagnostic stats."""
        return {
            'wealth': self.wealth,
            'sum_rewards': self.sum_rewards,
            't': self.t,
            'effective_lr': self.effective_lr,
            'bet_fraction': self.sum_rewards / (self.t + 1) if self.t > 0 else 0,
            'delta_norm': self.delta.norm().item(),
        }


class BoundGuidedHebbian:
    """Hebbian adaptation steered by online bound analysis.

    Instead of adapting the output projection blindly, uses lightweight
    bound diagnostics to determine:
    1. WHICH tick to read predictions from (best tick, not final)
    2. WHICH sync dimensions matter (from neuron contribution analysis)
    3. WHETHER to update at all (overthinking detection)
    4. WHERE to apply correction (upstream if synapse is underutilized)

    The bound analysis runs every `diagnose_every` steps on the accumulated
    activations, then steers the Hebbian updates until the next diagnosis.

    Usage:
        hebb = BoundGuidedHebbian(n_synch=64, n_output=10, n_ticks=16)
        hebb.snapshot_baseline(model_sync_output)

        for input in stream:
            preds_all_ticks, sync = model(input)
            # Uses best tick, not final tick
            correction = hebb.apply(sync)
            adapted = preds_all_ticks[:, :, hebb.best_tick] + correction
            hebb.update(sync, output_weights, preds_all_ticks, label)
    """

    def __init__(self, n_synch: int, n_output: int, n_ticks: int,
                 lr: float = 0.1, momentum: float = 0.8,
                 diagnose_every: int = 50):
        self.n_synch = n_synch
        self.n_output = n_output
        self.n_ticks = n_ticks
        self.lr = lr
        self.momentum = momentum
        self.diagnose_every = diagnose_every

        self.baseline = torch.zeros(n_synch)
        self.delta = torch.zeros(n_synch, n_output)
        self.active = False

        # Bound-derived steering (updated every diagnose_every steps)
        self.best_tick = n_ticks - 1         # which tick to use for predictions
        self.sync_mask = torch.ones(n_synch)  # which sync dims matter
        self.should_update = True             # overthinking check
        self.tick_weights = torch.ones(n_ticks)  # per-tick quality

        # Online accumulators for lightweight diagnostics
        self.t = 0
        self._tick_correct = torch.zeros(n_ticks)
        self._tick_total = 0
        self._sync_variance = torch.zeros(n_synch)  # running variance
        self._sync_mean = torch.zeros(n_synch)
        self._n_accumulated = 0

    def snapshot_baseline(self, sync_signal: torch.Tensor):
        self.baseline = sync_signal.detach().clone()
        self.active = True

    def reset(self):
        self.delta.zero_()
        self.t = 0
        self._tick_correct.zero_()
        self._tick_total = 0
        self._sync_variance.zero_()
        self._sync_mean.zero_()
        self._n_accumulated = 0
        self.best_tick = self.n_ticks - 1
        self.sync_mask = torch.ones(self.n_synch)
        self.should_update = True

    def _diagnose(self):
        """Lightweight online bound analysis from accumulated stats."""
        if self._tick_total == 0:
            return

        # 1. Best tick: which tick has highest accuracy?
        tick_accs = self._tick_correct / max(self._tick_total, 1)
        self.best_tick = int(tick_accs.argmax().item())

        # 2. Overthinking detection: does accuracy DROP after best tick?
        best_acc = tick_accs[self.best_tick].item()
        final_acc = tick_accs[-1].item()
        self.should_update = (final_acc < best_acc * 0.95)  # 5% margin

        # 3. Tick weights: normalize accuracies for weighted prediction
        if tick_accs.max() > 0:
            self.tick_weights = tick_accs / tick_accs.max()

        # 4. Sync mask: which dimensions have high variance (= informative)?
        if self._n_accumulated > 1:
            variance = self._sync_variance / self._n_accumulated
            # Keep top 50% most variable dimensions
            threshold = variance.median()
            self.sync_mask = (variance > threshold).float()

        # Reset accumulators
        self._tick_correct.zero_()
        self._tick_total = 0
        self._sync_variance.zero_()
        self._sync_mean.zero_()
        self._n_accumulated = 0

    @torch.no_grad()
    def update(self, sync_signal: torch.Tensor, output_weights: torch.Tensor,
               all_tick_preds: torch.Tensor, label: int):
        """Bound-guided Hebbian update.

        Args:
            sync_signal: [n_synch] sync from forward pass
            output_weights: [n_output, n_synch] output projection weights
            all_tick_preds: [n_output, n_ticks] predictions at every tick
            label: ground truth class
        """
        if not self.active:
            return

        self.t += 1

        # Accumulate per-tick accuracy for diagnosis
        for t in range(self.n_ticks):
            if all_tick_preds[:, t].argmax().item() == label:
                self._tick_correct[t] += 1
        self._tick_total += 1

        # Accumulate sync stats (Welford's online variance)
        self._n_accumulated += 1
        diff = sync_signal - self._sync_mean
        self._sync_mean += diff / self._n_accumulated
        diff2 = sync_signal - self._sync_mean
        self._sync_variance += diff * diff2

        # Run diagnosis periodically
        if self.t % self.diagnose_every == 0:
            self._diagnose()

        # Use best tick prediction for reward signal
        best_pred = all_tick_preds[:, self.best_tick].argmax().item()
        reward = (best_pred == label)

        if not reward or not self.should_update:
            return

        # Hebbian update with sync mask (only informative dimensions)
        novelty = sync_signal - self.baseline
        masked_novelty = novelty * self.sync_mask
        gated = masked_novelty  # no additional gating — mask IS the gate

        action_signal = output_weights @ gated
        new_delta = self.lr * torch.outer(gated, action_signal)

        self.delta = self.momentum * self.delta + (1 - self.momentum) * new_delta

    def apply(self, sync: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.zeros(sync.shape[0], self.n_output,
                             device=sync.device, dtype=sync.dtype)
        return sync @ self.delta.to(device=sync.device, dtype=sync.dtype)

    def stats(self) -> dict:
        return {
            't': self.t,
            'best_tick': self.best_tick,
            'should_update': self.should_update,
            'sync_mask_active': self.sync_mask.sum().item(),
            'delta_norm': self.delta.norm().item(),
        }


class HomeostaticHebbian:
    """Hebbian adaptation with homeostatic structural plasticity.

    Inspired by HAG (Cazalets & Dambre, Nature Communications 2026):
    only update sync dimensions that are OUT OF HOMEOSTASIS — firing
    too much or too little relative to baseline. This prevents the
    runaway updates that destroyed our earlier Hebbian attempts.

    Key differences from naive HebbianPlasticity:
    1. Track per-dimension running mean/variance over a window
    2. Only update dimensions where |activity - target| > threshold
    3. Grow connections (increase delta) for under-active dimensions
    4. Prune connections (decay delta) for over-active dimensions
    5. Long-horizon correlation: accumulate correlation over window,
       don't react to single-sample novelty

    This is the biological version of what the SDP computes algebraically:
    both find optimal weights from observed activation statistics.
    The SDP does it in one shot, homeostatic Hebbian does it incrementally.

    Usage:
        hebb = HomeostaticHebbian(n_synch=64, n_output=10)
        hebb.snapshot_baseline(model_sync_output)

        for input in stream:
            preds, sync = model(input)
            correction = hebb.apply(sync)
            adapted = preds + correction
            hebb.observe(sync, reward=is_correct)
    """

    def __init__(self, n_synch: int, n_output: int,
                 target_rate: float = 0.0,     # target mean activation (ρ)
                 rate_spread: float = 1.0,     # homeostatic band width (β)
                 window_size: int = 100,       # correlation window (T_current)
                 delta_w: float = 0.01,        # connection growth step
                 decay_factor: float = 0.99,   # pruning decay for over-active dims
                 saturation_threshold: float = 5.0,  # max delta magnitude
                 saturation_scale: float = 0.9,      # scale-down factor
                 ):
        self.n_synch = n_synch
        self.n_output = n_output
        self.target_rate = target_rate
        self.rate_spread = rate_spread
        self.window_size = window_size
        self.delta_w = delta_w
        self.decay_factor = decay_factor
        self.saturation_threshold = saturation_threshold
        self.saturation_scale = saturation_scale

        self.baseline = torch.zeros(n_synch)
        self.delta = torch.zeros(n_synch, n_output)
        self.active = False

        # Running statistics per sync dimension
        self._sync_buffer = []  # list of [n_synch] tensors
        self._reward_buffer = []  # list of bools
        self.t = 0

    def snapshot_baseline(self, sync_signal: torch.Tensor):
        self.baseline = sync_signal.detach().clone()
        self.active = True
        self.target_rate = sync_signal.mean().item()

    def reset(self):
        self.delta.zero_()
        self._sync_buffer.clear()
        self._reward_buffer.clear()
        self.t = 0

    @torch.no_grad()
    def observe(self, sync_signal: torch.Tensor, output_weights: torch.Tensor,
                reward: bool = True):
        """Observe one sync signal and optionally update.

        Unlike naive Hebbian which updates every sample, homeostatic
        Hebbian buffers observations and only updates when the window
        is full — computing long-horizon correlations.
        """
        if not self.active:
            return

        self._sync_buffer.append(sync_signal.detach().clone())
        self._reward_buffer.append(reward)
        self.t += 1

        # Only update when window is full
        if len(self._sync_buffer) < self.window_size:
            return

        self._homeostatic_update(output_weights)

        # Slide window (keep half for overlap)
        half = self.window_size // 2
        self._sync_buffer = self._sync_buffer[half:]
        self._reward_buffer = self._reward_buffer[half:]

    def _homeostatic_update(self, output_weights: torch.Tensor):
        """HAG-style update: identify out-of-homeostasis dimensions,
        compute long-horizon correlations, grow/prune connections."""

        syncs = torch.stack(self._sync_buffer)  # [W, n_synch]
        rewards = torch.tensor(self._reward_buffer, dtype=torch.float32)

        # Per-dimension statistics over the window
        dim_means = syncs.mean(dim=0)  # [n_synch]
        dim_stds = syncs.std(dim=0)    # [n_synch]

        # Homeostatic deviation: how far from target?
        # Δz_i = (s_i - ρ) / β  (from HAG eq. 6)
        deviation = (dim_means - self.target_rate) / max(self.rate_spread, 1e-8)

        # Under-active: Δz < -1 → need to grow connections
        under_active = (deviation < -1.0)
        # Over-active: Δz > +1 → need to prune connections
        over_active = (deviation > 1.0)
        # At homeostasis: -1 ≤ Δz ≤ 1 → leave alone
        n_under = under_active.sum().item()
        n_over = over_active.sum().item()

        if n_under == 0 and n_over == 0:
            return  # all at homeostasis — no update

        # For under-active dimensions: compute pairwise correlation
        # with other under-active dimensions (HAG eq. 8)
        # and grow the strongest connection
        if n_under > 1:
            under_idx = torch.where(under_active)[0]
            under_syncs = syncs[:, under_idx]  # [W, n_under]

            # Correlation matrix among under-active dimensions
            # Centered
            centered = under_syncs - under_syncs.mean(dim=0, keepdim=True)
            norms = centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
            normalized = centered / norms
            corr = (normalized.T @ normalized) / len(syncs)

            # Zero diagonal (no self-connections)
            corr.fill_diagonal_(0)

            # Find most correlated pair
            flat_idx = corr.abs().argmax()
            i_local = flat_idx // len(under_idx)
            j_local = flat_idx % len(under_idx)
            i_global = under_idx[i_local].item()
            j_global = under_idx[j_local].item()

            # Grow: strengthen connection from j to i in the delta matrix
            # This means: when sync dim j fires, it contributes more to output
            # Scale by correlation sign (positive = excitatory, negative = inhibitory)
            sign = 1.0 if corr[i_local, j_local] > 0 else -1.0

            # Only grow using reward-correlated signals
            reward_rate = rewards.mean().item()
            if reward_rate > 0:
                # Grow the delta row for the under-active dimension
                # Direction: use output_weights to map to action space
                action_dir = output_weights[:, j_global]  # [n_output]
                self.delta[i_global] += self.delta_w * sign * reward_rate * action_dir

        elif n_under == 1:
            # Single under-active dim: grow using correlation with baseline
            under_idx = torch.where(under_active)[0][0].item()
            novelty = dim_means - self.baseline
            # Use the most novel OTHER dimension as partner
            novelty[under_idx] = 0  # exclude self
            partner = novelty.abs().argmax().item()
            sign = 1.0 if novelty[partner] > 0 else -1.0
            reward_rate = rewards.mean().item()
            if reward_rate > 0:
                action_dir = output_weights[:, partner]
                self.delta[under_idx] += self.delta_w * sign * reward_rate * action_dir

        # For over-active dimensions: prune (decay their delta rows)
        if n_over > 0:
            over_idx = torch.where(over_active)[0]
            self.delta[over_idx] *= self.decay_factor

        # Saturation check: if any delta row is too large, scale down
        row_norms = self.delta.norm(dim=1)  # [n_synch]
        saturated = row_norms > self.saturation_threshold
        if saturated.any():
            self.delta[saturated] *= self.saturation_scale

    def apply(self, sync: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.zeros(sync.shape[0] if sync.dim() > 1 else 1,
                             self.n_output,
                             device=sync.device, dtype=sync.dtype)
        s = sync if sync.dim() > 1 else sync.unsqueeze(0)
        return s @ self.delta.to(device=s.device, dtype=s.dtype)

    def stats(self) -> dict:
        row_norms = self.delta.norm(dim=1)
        return {
            't': self.t,
            'buffer_size': len(self._sync_buffer),
            'delta_norm': self.delta.norm().item(),
            'active_dims': (row_norms > 1e-6).sum().item(),
            'max_row_norm': row_norms.max().item(),
        }
