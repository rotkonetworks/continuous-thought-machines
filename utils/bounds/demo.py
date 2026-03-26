#!/usr/bin/env python3
"""
Demo: Optimality bound analysis for Continuous Thought Machines.

Builds a standalone CTM-like recurrent model, trains it briefly on a toy
task, then runs the full bound analysis showing per-neuron, per-tick,
and synapse diagnostics.

No GPU or external data required — runs in ~10 seconds on CPU.

Usage:
    python utils/bounds/demo.py

For use with the full ContinuousThoughtMachine class:
    from utils.bounds import analyze_ctm, print_report
    model = ContinuousThoughtMachine.from_pretrained(...)
    results = analyze_ctm(model, input_tensor)
    print_report(results)

References:
    Angeris, G. (2022). "A Note on Generalizing Power Bounds for Physical
    Design." arXiv:2208.04411v2.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


# ─── Minimal CTM for demo (same structure, no heavy attention) ───────

class SuperLinear(nn.Module):
    """Per-neuron linear: applies N independent linear transforms."""
    def __init__(self, in_dims, out_dims, N):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, in_dims, out_dims) * 0.02)
        self.bias = nn.Parameter(torch.zeros(N, out_dims))

    def forward(self, x):
        # x: [B, N, in_dims]
        return torch.einsum('bni,nio->bno', x, self.weight) + self.bias


class DemoCTM(nn.Module):
    """Minimal CTM with synapse + NLMs + synchronization.

    Same architecture as ContinuousThoughtMachine but without the
    attention mechanism, for fast CPU demo.
    """
    def __init__(self, input_dim=8, d_model=32, iterations=12,
                 memory_length=6, n_synch_out=16, out_dims=10):
        super().__init__()
        self.d_model = d_model
        self.iterations = iterations
        self.memory_length = memory_length
        self.out_dims = out_dims

        # Input projection (replaces backbone + attention)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Synapse: shared communication between neurons
        self.synapses = nn.Sequential(
            nn.Linear(d_model + d_model, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model),
        )

        # NLMs: simplified — linear map from trace to activation per neuron
        # (In the full CTM this is SuperLinear with per-neuron weights)
        self.trace_processor = nn.Linear(memory_length, 1)

        # Start states
        self.start_state = nn.Parameter(torch.randn(d_model) * 0.01)
        self.start_trace = nn.Parameter(torch.randn(d_model, memory_length) * 0.01)

        # Synchronization indices (random pairing)
        indices = torch.randperm(d_model)
        self.register_buffer('sync_left', indices[:n_synch_out])
        self.register_buffer('sync_right', indices[n_synch_out:2*n_synch_out]
                             if 2*n_synch_out <= d_model
                             else torch.randperm(d_model)[:n_synch_out])

        # Output projection from synchronization
        self.output_proj = nn.Linear(n_synch_out, out_dims)

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # Project input
        input_features = self.input_proj(x)  # [B, d_model]

        # Initialize
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_state.unsqueeze(0).expand(B, -1) + input_features

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device)
        certainties = torch.empty(B, 2, self.iterations, device=device)

        pre_act_track = []
        post_act_track = []

        for t in range(self.iterations):
            # Synapse: mix all neurons
            syn_input = torch.cat([input_features, activated_state], dim=-1)
            state = self.synapses(syn_input)

            # Update trace (shift + append)
            state_trace = torch.cat([state_trace[:, :, 1:], state.unsqueeze(-1)], dim=-1)

            # NLM: process trace history → activation
            activated_state = self.trace_processor(state_trace).squeeze(-1)
            activated_state = torch.tanh(activated_state)

            # Synchronization: pairwise products
            left = activated_state[:, self.sync_left]
            right = activated_state[:, self.sync_right]
            sync = left * right

            # Output
            pred = self.output_proj(sync)
            predictions[:, :, t] = pred
            certainties[:, 0, t] = pred.abs().mean(dim=-1)
            certainties[:, 1, t] = 1 - pred.abs().mean(dim=-1)

            if track:
                pre_act_track.append(state.detach().cpu().numpy())
                post_act_track.append(activated_state.detach().cpu().numpy())

        if track:
            return (predictions, certainties, None,
                    np.array(pre_act_track),
                    np.array(post_act_track),
                    None)
        return predictions, certainties, sync


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("CTM Optimality Bound Analysis — Demo")
    print("=" * 60)

    # ─── 1. Toy task: input features → class label ──────────────────
    n_samples = 300
    input_dim = 8
    n_classes = 10

    X = torch.randn(n_samples, input_dim)
    # Target: nonlinear function of inputs (needs "thinking" to solve)
    hidden = torch.tanh(X @ torch.randn(input_dim, 16)) @ torch.randn(16, 1)
    Y = torch.bucketize(hidden.squeeze(), torch.linspace(-2, 2, n_classes - 1))
    print(f"\nTask: {input_dim}D input → {n_classes} classes ({n_samples} samples)")

    # ─── 2. Build and train ──────────────────────────────────────────
    model = DemoCTM(input_dim=input_dim, d_model=32, iterations=12,
                    memory_length=6, n_synch_out=16, out_dims=n_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (32 neurons, 12 ticks, 6 memory)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining 30 epochs...")
    for epoch in range(30):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0
        correct = 0
        for i in range(0, n_samples, 64):
            idx = perm[i:i+64]
            pred, _, _ = model(X[idx])
            logits = pred[:, :, -1]
            loss = criterion(logits, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == Y[idx]).sum().item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}: loss={total_loss/(n_samples//64+1):.3f} "
                  f"acc={correct/n_samples:.1%}")

    # ─── 3. Bound analysis ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running bound analysis...")
    print("=" * 60)

    from utils.bounds.core import analyze_ctm, print_report

    # Analyze on 3 samples
    all_results = []
    for i in range(3):
        results = analyze_ctm(model, X[i:i+1])
        all_results.append(results)
        if i == 0:
            print_report(results)

    # ─── 4. Aggregate ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS 3 SAMPLES")
    print("=" * 60)
    print(f"  Dead neurons:       {np.mean([len(r.dead_neurons) for r in all_results]):.1f}/32")
    print(f"  Best tick:          {np.mean([r.best_tick for r in all_results]):.1f}/12")
    print(f"  Overthinking ticks: {np.mean([len(r.overthinking_ticks) for r in all_results]):.1f}")
    print(f"  Synapse gap:        {np.mean([r.synapse_gap_pct for r in all_results]):.1f}%")
    print(f"  Input eff rank:     {np.mean([r.input_effective_rank for r in all_results]):.1f}")
    print(f"  Bottleneck:         {all_results[0].bottleneck}")


if __name__ == '__main__':
    main()
