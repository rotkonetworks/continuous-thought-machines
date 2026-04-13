#!/usr/bin/env python3
"""Train CTM QEC decoder on Bayes-optimal soft posteriors.

Instead of hard labels (argmax class), train on the full posterior
P(error_class | syndrome_bucket). This teaches the CTM:
- WHEN to be uncertain (ambiguous syndromes)
- HOW to resolve ambiguity (more thinking ticks)
- WHAT to attend to (soft target guides sync attention)

Pipeline (mirrors poker CFR → CTM):
1. Generate syndromes and compute exact posterior per coarsened bucket
2. Train CTM with KL divergence loss against soft posteriors
3. Evaluate: does soft training close the gap to Bayes optimal?

Usage:
    python -m tasks.qec.bayes_train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import defaultdict

from tasks.qec.surface_code import SurfaceCode
from tasks.qec.ctm_qec import ContinuousThoughtMachineQEC


def compute_syndrome_buckets(code, n_samples=200000, p=0.05, num_rounds=5,
                              coarsen='weight'):
    """Compute Bayes-optimal posterior per coarsened syndrome bucket.

    Like CFR in poker: compute the optimal strategy for each
    information set (syndrome bucket).

    Args:
        coarsen: 'weight' (by syndrome Hamming weight) or
                 'spatial' (by which half of lattice fired) or
                 'raw' (no coarsening — exact syndrome)
    """
    print(f"Computing Bayes posteriors ({n_samples} samples, coarsen={coarsen})...")

    bucket_counts = defaultdict(lambda: np.zeros(4))
    total = 0

    batch = 1000
    for _ in range(n_samples // batch):
        syndromes, labels = code.generate_syndromes(batch, p, num_rounds)
        final_syn = syndromes[:, -1, :]  # [B, n_stab]

        for i in range(batch):
            syn = final_syn[i]
            label = labels[i].item()

            if coarsen == 'weight':
                # Bucket by syndrome weight (how many stabilizers fired)
                # Split into X-weight and Z-weight for more resolution
                n_x = code.n_x_stab
                x_weight = int(syn[:n_x].sum().item())
                z_weight = int(syn[n_x:].sum().item())
                key = (x_weight, z_weight)
            elif coarsen == 'spatial':
                # Bucket by quadrant of lattice
                n_x = code.n_x_stab
                half = code.n_stab // 2
                q1 = int(syn[:half//2].sum().item() > 0)
                q2 = int(syn[half//2:half].sum().item() > 0)
                q3 = int(syn[half:half+half//2].sum().item() > 0)
                q4 = int(syn[half+half//2:].sum().item() > 0)
                key = (q1, q2, q3, q4)
            else:  # raw
                key = tuple(syn.numpy().astype(int))

            bucket_counts[key][label] += 1
            total += 1

    # Convert to posteriors
    bucket_posteriors = {}
    for key, counts in bucket_counts.items():
        total_count = counts.sum()
        if total_count > 0:
            bucket_posteriors[key] = counts / total_count

    print(f"  {len(bucket_posteriors)} buckets from {total} samples")
    print(f"  Mean samples/bucket: {total / max(len(bucket_posteriors), 1):.0f}")

    # Show example posteriors
    sorted_buckets = sorted(bucket_posteriors.items(),
                           key=lambda x: -sum(bucket_counts[x[0]]))
    print(f"  Top buckets:")
    for key, post in sorted_buckets[:5]:
        n = int(bucket_counts[key].sum())
        print(f"    {key}: [{post[0]:.2f}, {post[1]:.2f}, {post[2]:.2f}, {post[3]:.2f}] (n={n})")

    return bucket_posteriors, bucket_counts


def get_soft_target(syndrome, bucket_posteriors, coarsen='weight', n_x_stab=12):
    """Look up the Bayes posterior for a syndrome."""
    if coarsen == 'weight':
        x_weight = int(syndrome[:n_x_stab].sum().item())
        z_weight = int(syndrome[n_x_stab:].sum().item())
        key = (x_weight, z_weight)
    elif coarsen == 'spatial':
        n_stab = len(syndrome)
        half = n_stab // 2
        q1 = int(syndrome[:half//2].sum().item() > 0)
        q2 = int(syndrome[half//2:half].sum().item() > 0)
        q3 = int(syndrome[half:half+half//2].sum().item() > 0)
        q4 = int(syndrome[half+half//2:].sum().item() > 0)
        key = (q1, q2, q3, q4)
    else:
        key = tuple(syndrome.numpy().astype(int))

    if key in bucket_posteriors:
        return torch.tensor(bucket_posteriors[key], dtype=torch.float32)
    else:
        return torch.ones(4) / 4  # uniform if unseen


def train_bayes(args=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/qec'
    os.makedirs(save_dir, exist_ok=True)

    distance = 5
    p = 0.05
    num_rounds = 5
    iterations = 16
    d_model = 256
    n_epochs = 20
    coarsen = 'weight'

    code = SurfaceCode(distance)
    syndrome_dim = num_rounds * code.n_stab

    # Step 1: Compute Bayes posteriors (the "CFR solve")
    bucket_posteriors, bucket_counts = compute_syndrome_buckets(
        code, n_samples=200000, p=p, num_rounds=num_rounds, coarsen=coarsen)

    # Step 2: Build CTM decoder
    model = ContinuousThoughtMachineQEC(
        syndrome_dim=syndrome_dim, d_model=d_model, iterations=iterations,
        n_synch_out=64, synapse_depth=1, memory_length=8,
    ).to(device)
    with torch.no_grad():
        model(torch.randn(2, syndrome_dim).to(device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params, T={iterations}, d_model={d_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Step 3: Train with KL divergence against soft posteriors
    best_acc = 0
    batch_size = 128
    n_batches = 50000 // batch_size

    for epoch in range(n_epochs):
        model.train()
        total_kl = 0
        correct = 0
        total = 0
        t0 = time.time()

        for _ in range(n_batches):
            syndromes, labels = code.generate_syndromes(
                batch_size, p, num_rounds, device=device)
            flat = syndromes.flatten(1)

            # Get soft targets for this batch
            final_syn = syndromes[:, -1, :]  # [B, n_stab]
            soft_targets = torch.stack([
                get_soft_target(final_syn[i].cpu(), bucket_posteriors,
                               coarsen=coarsen, n_x_stab=code.n_x_stab)
                for i in range(batch_size)
            ]).to(device)

            preds, _, _ = model(flat)

            # KL divergence loss at each tick, weighted toward later ticks
            kl_loss = torch.tensor(0.0, device=device)
            T = preds.shape[2]
            for t in range(T):
                log_probs = F.log_softmax(preds[:, :, t], dim=1)
                tick_kl = F.kl_div(log_probs, soft_targets, reduction='batchmean')
                # Weight: later ticks should match posterior better
                tick_weight = (t + 1) / T
                kl_loss = kl_loss + tick_weight * tick_kl

            kl_loss = kl_loss / T

            # Also add hard-label CE on final tick for grounding
            ce_loss = F.cross_entropy(preds[:, :, -1], labels)
            loss = 0.7 * kl_loss + 0.3 * ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_kl += kl_loss.item()
            correct += (preds[:, :, -1].argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Evaluate
        model.eval()
        tc = 0
        tt = 5000
        with torch.no_grad():
            for _ in range(tt // 64):
                syn, lab = code.generate_syndromes(64, p, num_rounds, device=device)
                preds, _, _ = model(syn.flatten(1))
                tc += (preds[:, :, -1].argmax(1) == lab).sum().item()

        test_acc = tc / tt
        train_acc = correct / total
        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
                'coarsen': coarsen,
                'n_buckets': len(bucket_posteriors),
            }, os.path.join(save_dir, 'best_bayes.pt'))

        print(f"epoch {epoch+1:2d}: train={train_acc:.1%} test={test_acc:.1%} "
              f"best={best_acc:.1%} kl={total_kl/n_batches:.4f} ({elapsed:.0f}s)",
              flush=True)

    print(f"\nBest test accuracy: {best_acc:.1%}")
    print(f"Hard-label baseline was: 47.8%")
    print(f"Bayes optimal: 65.6%")
    print(f"Gap closed: {(best_acc - 0.478) / (0.656 - 0.478) * 100:.0f}%")


if __name__ == '__main__':
    train_bayes()
