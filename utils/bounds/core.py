"""Core bound analysis for ContinuousThoughtMachine.

Works with any CTM instance — ImageNet, maze, QA, or custom tasks.
Only requires track=True forward pass.

Key distinction this analysis makes:
  - "Dead neuron" = low WEIGHT norm (truly unused capacity, architecture waste)
  - "Inactive neuron" = low activation on THIS input (sparse activation, expected)
  - "Synapse capacity" = weight matrix rank (what it CAN express)
  - "Synapse utilization" = activation rank (what it DOES express on this input)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class BoundResults:
    """Complete bound analysis results for a CTM on one input."""
    # Per-neuron analysis (weights — architecture capacity)
    neuron_weight_norms: np.ndarray   # [D] NLM weight L2 norm per neuron
    neuron_diversity: float           # mean pairwise cosine sim (0=diverse, 1=collapsed)
    dead_neurons: List[int]           # truly dead: low weight norm (not just inactive)
    n_dead: int                       # count of dead neurons

    # Per-neuron analysis (activations — this-input behavior)
    neuron_act_energy: np.ndarray     # [D] activation energy on this input
    inactive_neurons: List[int]       # inactive on THIS input (sparse activation)
    n_inactive: int                   # count of inactive neurons
    neuron_contributions: np.ndarray  # [D] fraction of total Jacobian energy
    neuron_effective_ranks: np.ndarray  # [D] effective rank of NLM Jacobian

    # Per-tick analysis
    tick_losses: np.ndarray           # [T] loss at each tick
    tick_improvements: np.ndarray     # [T] loss improvement from previous tick
    tick_jac_energy: np.ndarray       # [T] total Jacobian energy
    tick_act_energy: np.ndarray       # [T] total activation energy
    best_tick: int                    # tick with lowest loss
    overthinking_ticks: List[int]     # ticks where loss increased

    # Synapse analysis — separate capacity (weights) from utilization (activations)
    synapse_weight_rank_90: int       # effective rank of W at 90% energy
    synapse_weight_rank_99: int       # effective rank of W at 99% energy
    synapse_weight_dims: tuple        # (rows, cols) of synapse weight matrix
    synapse_activation_rank: int      # effective rank of activations flowing through
    synapse_utilization_pct: float    # activation_rank / weight_rank * 100
    synapse_top_svs: list             # top 5 singular values of weight matrix
    synapse_condition: float          # condition number of weight matrix

    # Global
    synapse_gap: float                # achieved - optimal residual
    synapse_gap_pct: float            # gap as percentage
    bottleneck: str                   # what limits performance

    # Summary
    model_dim: int
    n_ticks: int
    n_synapse_params: int


def _get_nlm_weight_norms(model):
    """Extract per-neuron NLM weight norms from model parameters.

    Handles both SuperLinear (w1 shaped [in, hidden, N]) and standard Linear.
    Returns array of shape [D] with L2 norm per neuron, or None if not found.
    """
    norms = None
    for name, param in model.named_parameters():
        if 'trace_processor' not in name or 'weight' not in name.split('.')[-1]:
            # Also check for SuperLinear's 'w1' parameter
            if 'trace_processor' not in name or 'w1' not in name.split('.')[-1]:
                continue

        w = param.detach().cpu()
        if w.dim() == 3:
            # SuperLinear: could be [in, out, N] or [N, in, out]
            # Figure out which dim is neurons (D = d_model)
            D = model.d_model
            if w.shape[0] == D:
                per_neuron = w.reshape(D, -1).norm(dim=1)
            elif w.shape[2] == D:
                per_neuron = w.reshape(-1, D).norm(dim=0)
            elif w.shape[1] == D:
                per_neuron = w.permute(1, 0, 2).reshape(D, -1).norm(dim=1)
            else:
                continue

            if norms is None:
                norms = per_neuron.numpy()
            else:
                norms = norms + per_neuron.numpy()

    return norms


def _get_nlm_diversity(model):
    """Compute pairwise cosine similarity between neuron NLM weights.

    Returns mean cosine similarity: 0 = fully diverse, 1 = all identical.
    """
    # Find the largest NLM weight tensor
    best_w = None
    for name, param in model.named_parameters():
        if 'trace_processor' not in name:
            continue
        if 'w1' in name.split('.')[-1] or 'weight' in name.split('.')[-1]:
            w = param.detach().cpu()
            if w.dim() == 3 and (best_w is None or w.numel() > best_w.numel()):
                best_w = w

    if best_w is None:
        return 0.0

    D = model.d_model
    # Reshape so each neuron is a row
    if best_w.shape[2] == D:
        vectors = best_w.reshape(-1, D).T  # [D, features]
    elif best_w.shape[0] == D:
        vectors = best_w.reshape(D, -1)    # [D, features]
    else:
        return 0.0

    # Normalize
    vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)

    # Pairwise cosine (upper triangle)
    cos = vectors @ vectors.T
    mask = torch.triu(torch.ones(D, D, dtype=torch.bool), diagonal=1)
    return float(cos[mask].mean().abs())


def _get_synapse_weight_info(model):
    """Extract synapse weight matrix and compute rank/SVD info."""
    for name, param in model.synapses.named_parameters():
        if 'weight' in name and param.dim() == 2:
            w = param.detach().cpu().float()
            svs = torch.linalg.svdvals(w)
            total = svs.sum()
            cumsum = svs.cumsum(0) / total
            rank_90 = int((cumsum < 0.9).sum().item()) + 1
            rank_99 = int((cumsum < 0.99).sum().item()) + 1
            condition = float(svs[0] / (svs[-1] + 1e-10))
            return {
                'dims': (w.shape[0], w.shape[1]),
                'rank_90': rank_90,
                'rank_99': rank_99,
                'condition': condition,
                'top_svs': svs[:5].tolist(),
                'n_params': w.numel(),
            }
    return None


def analyze_ctm(model, x, device=None):
    """Run complete bound analysis on a CTM for a single input.

    Args:
        model: ContinuousThoughtMachine instance (or subclass)
        x: input tensor [1, ...] (single sample, batched)
        device: torch device (inferred from x if None)

    Returns:
        BoundResults with per-neuron, per-tick, and synapse diagnostics
    """
    if device is None:
        if isinstance(x, dict):
            device = next(iter(x.values())).device
        else:
            device = x.device if hasattr(x, 'device') else 'cpu'

    model.eval()
    model = model.to(device)
    if isinstance(x, dict):
        x = {k: v.to(device) for k, v in x.items()}
    else:
        x = x.to(device)

    D = model.d_model
    T = model.iterations
    eps = 1e-8

    # ─── Forward pass with tracking ──────────────────────────────────
    with torch.no_grad():
        out = model(x, track=True)

    predictions = out[0]
    pre_act = out[3]
    post_act = out[4]

    if isinstance(pre_act, torch.Tensor):
        pre_act = pre_act.cpu().numpy()
    if isinstance(post_act, torch.Tensor):
        post_act = post_act.cpu().numpy()

    pre_act = pre_act[:, 0, :]   # [T, D]
    post_act = post_act[:, 0, :]  # [T, D]

    # ─── Per-neuron: WEIGHT analysis (architecture capacity) ─────────
    weight_norms = _get_nlm_weight_norms(model)
    if weight_norms is None:
        weight_norms = np.ones(D)

    neuron_diversity = _get_nlm_diversity(model)

    # Truly dead = weight norm below 10% of mean
    mean_wn = weight_norms.mean()
    dead_neurons = list(np.where(weight_norms < mean_wn * 0.1)[0])

    # ─── Per-neuron: ACTIVATION analysis (this-input behavior) ───────
    neuron_act_energy = np.sum(post_act ** 2, axis=0)  # [D]
    act_threshold = np.percentile(neuron_act_energy, 25)
    inactive_neurons = list(np.where(neuron_act_energy < act_threshold * 0.01)[0])

    # Jacobian-based contribution
    nlm_jac = np.where(
        np.abs(pre_act) > eps,
        post_act / (pre_act + np.sign(pre_act) * eps),
        1.0
    )
    neuron_jac_energy = np.sum(nlm_jac ** 2, axis=0)
    total_jac_energy = neuron_jac_energy.sum()
    neuron_contributions = neuron_jac_energy / (total_jac_energy + eps)

    # Effective rank per neuron
    neuron_eff_ranks = np.zeros(D)
    for d in range(D):
        jac_abs = np.abs(nlm_jac[:, d])
        s = jac_abs.sum()
        if s > eps:
            p = jac_abs / s
            neuron_eff_ranks[d] = np.exp(-np.sum(p * np.log(p + 1e-12)))

    # ─── Per-tick analysis ───────────────────────────────────────────
    pred_np = predictions[0, 0, :].detach().cpu().numpy()
    tick_losses = pred_np ** 2
    tick_improvements = np.zeros(T)
    tick_improvements[1:] = tick_losses[:-1] - tick_losses[1:]

    tick_jac_energy = np.sum(nlm_jac ** 2, axis=1)
    tick_act_energy = np.sum(post_act ** 2, axis=1)

    best_tick = int(np.argmin(tick_losses))
    overthinking_ticks = list(np.where(tick_improvements < -0.01 * tick_losses.max())[0])

    # ─── Synapse: WEIGHT analysis (capacity) ─────────────────────────
    syn_info = _get_synapse_weight_info(model)
    if syn_info is None:
        syn_info = {'dims': (0, 0), 'rank_90': 0, 'rank_99': 0,
                    'condition': 0, 'top_svs': [], 'n_params': 0}

    # ─── Synapse: ACTIVATION analysis (utilization) ──────────────────
    captured = {'in': [], 'out': []}
    def hook(module, inp, out):
        captured['in'].append(inp[0].detach().cpu())
        captured['out'].append(out.detach().cpu())

    handle = model.synapses.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    if captured['in']:
        syn_in = np.array([t[0].numpy() for t in captured['in']])
        syn_out = np.array([t[0].numpy() for t in captured['out']])

        # Activation effective rank (what the synapse actually uses)
        _, S_in, _ = np.linalg.svd(syn_in, full_matrices=False)
        act_eff_rank = int(np.sum(S_in > S_in[0] * 0.01))

        # Optimal shared W residual
        A_cross = sum(np.outer(syn_out[t], syn_in[t]) for t in range(len(syn_in)))
        B_inp = sum(np.outer(syn_in[t], syn_in[t]) for t in range(len(syn_in)))
        try:
            W_opt = A_cross @ np.linalg.pinv(B_inp)
            optimal_resid = sum(
                np.sum((syn_out[t] - W_opt @ syn_in[t]) ** 2) for t in range(len(syn_in)))
            achieved_resid = sum(np.sum(syn_out[t] ** 2) for t in range(len(syn_in)))
            synapse_gap = achieved_resid - optimal_resid
            synapse_gap_pct = synapse_gap / (achieved_resid + eps) * 100
        except np.linalg.LinAlgError:
            act_eff_rank = 0
            synapse_gap = 0
            synapse_gap_pct = 0
    else:
        act_eff_rank = 0
        synapse_gap = 0
        synapse_gap_pct = 0

    # Utilization: how much of the synapse capacity is actually used
    utilization = act_eff_rank / (syn_info['rank_90'] + eps) * 100

    # Bottleneck determination
    if len(dead_neurons) > D * 0.3:
        bottleneck = 'dead_neurons'
    elif utilization < 10:
        bottleneck = 'upstream (low activation rank through synapse)'
    elif syn_info['condition'] > 500:
        bottleneck = 'synapse conditioning (ill-conditioned weight matrix)'
    elif len(overthinking_ticks) > T * 0.3:
        bottleneck = 'overthinking (too many ticks)'
    else:
        bottleneck = 'none detected'

    return BoundResults(
        neuron_weight_norms=weight_norms,
        neuron_diversity=neuron_diversity,
        dead_neurons=dead_neurons,
        n_dead=len(dead_neurons),
        neuron_act_energy=neuron_act_energy,
        inactive_neurons=inactive_neurons,
        n_inactive=len(inactive_neurons),
        neuron_contributions=neuron_contributions,
        neuron_effective_ranks=neuron_eff_ranks,
        tick_losses=tick_losses,
        tick_improvements=tick_improvements,
        tick_jac_energy=tick_jac_energy,
        tick_act_energy=tick_act_energy,
        best_tick=best_tick,
        overthinking_ticks=overthinking_ticks,
        synapse_weight_rank_90=syn_info['rank_90'],
        synapse_weight_rank_99=syn_info['rank_99'],
        synapse_weight_dims=syn_info['dims'],
        synapse_activation_rank=act_eff_rank,
        synapse_utilization_pct=utilization,
        synapse_top_svs=syn_info['top_svs'],
        synapse_condition=syn_info['condition'],
        synapse_gap=synapse_gap,
        synapse_gap_pct=synapse_gap_pct,
        bottleneck=bottleneck,
        model_dim=D,
        n_ticks=T,
        n_synapse_params=syn_info['n_params'],
    )


def print_report(results: BoundResults):
    """Print human-readable bound analysis report."""
    D = results.model_dim
    T = results.n_ticks

    print("=" * 60)
    print("CTM OPTIMALITY BOUND ANALYSIS")
    print("Angeris (2022) + §3.1 improved characterization")
    print("=" * 60)
    print(f"Architecture: {D} neurons, {T} ticks, {results.n_synapse_params:,} synapse params")

    # Neurons — weights (capacity)
    print(f"\n--- Neuron capacity (NLM weights) ---")
    wn = results.neuron_weight_norms
    print(f"  Weight norms: min={wn.min():.3f} mean={wn.mean():.3f} max={wn.max():.3f}")
    print(f"  Dead neurons (low weight):  {results.n_dead}/{D}")
    print(f"  Diversity (cosine sim):     {results.neuron_diversity:.4f} "
          f"({'diverse' if results.neuron_diversity < 0.1 else 'some collapse' if results.neuron_diversity < 0.5 else 'collapsed'})")

    # Neurons — activations (this input)
    print(f"\n--- Neuron activation (this input) ---")
    print(f"  Inactive on this input: {results.n_inactive}/{D} (sparse activation)")
    top5 = np.argsort(results.neuron_contributions)[-5:][::-1]
    print(f"  Top contributors: {[(int(i), f'{results.neuron_contributions[i]:.3f}') for i in top5]}")
    print(f"  Mean eff rank: {results.neuron_effective_ranks.mean():.1f}/{T}")

    # Ticks
    print(f"\n--- Per-tick thinking trajectory ---")
    print(f"  Best tick:    {results.best_tick} (loss={results.tick_losses[results.best_tick]:.4f})")
    print(f"  Final tick:   loss={results.tick_losses[-1]:.4f}")
    if results.overthinking_ticks:
        print(f"  Overthinking: ticks {results.overthinking_ticks}")
    else:
        print(f"  Overthinking: none")
    imp = results.tick_improvements
    print(f"  Improvement:  first3=[{', '.join(f'{x:.4f}' for x in imp[:3])}] "
          f"last3=[{', '.join(f'{x:.4f}' for x in imp[-3:])}]")

    # Synapse — separate capacity from utilization
    print(f"\n--- Synapse analysis (§3.1) ---")
    print(f"  Weight matrix:     {results.synapse_weight_dims}")
    print(f"  Weight rank (90%): {results.synapse_weight_rank_90} (capacity)")
    print(f"  Weight rank (99%): {results.synapse_weight_rank_99}")
    print(f"  Activation rank:   {results.synapse_activation_rank} (utilization)")
    print(f"  Utilization:       {results.synapse_utilization_pct:.1f}% "
          f"(activation rank / weight rank)")
    print(f"  Condition number:  {results.synapse_condition:.0f}")
    if results.synapse_top_svs:
        print(f"  Top singular vals: {[f'{s:.2f}' for s in results.synapse_top_svs]}")

    # Diagnosis
    print(f"\n--- Diagnosis ---")
    print(f"  Bottleneck: {results.bottleneck}")

    if results.n_dead > 0:
        print(f"  ! {results.n_dead} truly dead neurons — wasted parameters. "
              f"Consider smaller d_model or reinitializing.")

    if results.n_dead == 0 and results.n_inactive > D * 0.5:
        print(f"  * {results.n_inactive} neurons inactive on this input — "
              f"this is sparse activation (normal/healthy). "
              f"Different inputs activate different subsets.")

    if results.synapse_utilization_pct < 10:
        print(f"  ! Synapse utilization {results.synapse_utilization_pct:.0f}% — "
              f"the synapse has rank-{results.synapse_weight_rank_90} capacity but "
              f"only rank-{results.synapse_activation_rank} activations flow through it. "
              f"Bottleneck is UPSTREAM (attention/input projection).")

    if results.synapse_condition > 500:
        print(f"  ! Condition number {results.synapse_condition:.0f} — "
              f"ill-conditioned. Consider spectral normalization.")

    if results.overthinking_ticks and len(results.overthinking_ticks) > T * 0.2:
        print(f"  * Model overthinks on {len(results.overthinking_ticks)}/{T} ticks. "
              f"Consider early-exit or reducing T.")

    if results.neuron_diversity > 0.5:
        print(f"  ! Neuron collapse: diversity={results.neuron_diversity:.2f}. "
              f"Many neurons learned similar functions. Consider dropout or repulsion loss.")

    if (results.n_dead == 0 and results.synapse_utilization_pct > 30
            and not results.overthinking_ticks and results.neuron_diversity < 0.1):
        print(f"  OK: Model appears healthy — diverse neurons, good synapse utilization, "
              f"no overthinking.")
