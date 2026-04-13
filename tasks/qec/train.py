#!/usr/bin/env python3
"""Train CTM decoder for quantum error correction.

Trains on d=5 rotated surface code with depolarizing noise.
~15-30 min on consumer GPU.

Usage:
    python -m tasks.qec.train
    python -m tasks.qec.train --distance 7 --noise_rate 0.08
"""

import torch
import torch.nn as nn
import time
import os
import argparse

from tasks.qec.surface_code import SurfaceCode, SyndromeDataset, DriftingSyndromeStream
from tasks.qec.ctm_qec import ContinuousThoughtMachineQEC
from utils.bounds.training import HebbianPlasticity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--distance', type=int, default=5)
    p.add_argument('--noise_rate', type=float, default=0.05)
    p.add_argument('--num_rounds', type=int, default=5)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--iterations', type=int, default=16)
    p.add_argument('--n_synch_out', type=int, default=64)
    p.add_argument('--synapse_depth', type=int, default=1)
    p.add_argument('--memory_length', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--train_size', type=int, default=50000)
    p.add_argument('--test_size', type=int, default=5000)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_dir', type=str, default='checkpoints/qec')
    return p.parse_args()


def evaluate(model, code, args):
    """Evaluate on fresh test syndromes."""
    model.eval()
    correct = 0
    total = 0
    per_tick_correct = torch.zeros(model.iterations)

    with torch.no_grad():
        for _ in range(args.test_size // args.batch_size):
            syndromes, labels = code.generate_syndromes(
                args.batch_size, args.noise_rate, args.num_rounds,
                device=args.device)
            flat = syndromes.flatten(1)
            preds, _, _ = model(flat)

            # Final tick accuracy
            final_pred = preds[:, :, -1].argmax(1)
            correct += (final_pred == labels).sum().item()
            total += labels.size(0)

            # Per-tick accuracy
            for t in range(model.iterations):
                tick_pred = preds[:, :, t].argmax(1)
                per_tick_correct[t] += (tick_pred == labels).sum().item()

    acc = correct / total
    per_tick_acc = per_tick_correct / total
    return acc, per_tick_acc


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    code = SurfaceCode(args.distance)
    syndrome_dim = args.num_rounds * code.n_stab
    print(f"Surface code d={args.distance}: {code.n_data} qubits, "
          f"{code.n_stab} stabilizers, syndrome_dim={syndrome_dim}")

    model = ContinuousThoughtMachineQEC(
        syndrome_dim=syndrome_dim,
        d_model=args.d_model,
        iterations=args.iterations,
        n_synch_out=args.n_synch_out,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
    ).to(args.device)

    # Init lazy
    dummy = torch.randn(2, syndrome_dim).to(args.device)
    with torch.no_grad():
        model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, {args.iterations} ticks")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        n_batches = args.train_size // args.batch_size
        for _ in range(n_batches):
            syndromes, labels = code.generate_syndromes(
                args.batch_size, args.noise_rate, args.num_rounds,
                device=args.device)
            flat = syndromes.flatten(1)

            preds, certs, _ = model(flat)
            final_logits = preds[:, :, -1]

            # Dual loss: final tick + best tick
            loss_final = criterion(final_logits, labels)

            # Best tick loss (tick with lowest CE)
            tick_losses = torch.stack([
                criterion(preds[:, :, t], labels) for t in range(model.iterations)
            ])
            loss_best = tick_losses.min()

            loss = 0.5 * loss_final + 0.5 * loss_best

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (final_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Evaluate
        test_acc, per_tick_acc = evaluate(model, code, args)
        train_acc = correct / total
        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'test_acc': test_acc,
                'per_tick_acc': per_tick_acc.tolist(),
            }, os.path.join(args.save_dir, 'best.pt'))

        # Per-tick summary
        tick_str = ' '.join(f'{per_tick_acc[t]:.0%}' for t in
                            [0, args.iterations//4, args.iterations//2,
                             3*args.iterations//4, args.iterations-1])

        print(f'epoch {epoch+1:2d}: train={train_acc:.1%} test={test_acc:.1%} '
              f'best={best_acc:.1%} loss={total_loss/n_batches:.3f} '
              f'ticks=[{tick_str}] ({elapsed:.0f}s)', flush=True)

    # ─── Hebbian adaptation test under noise drift ───────────────────
    print(f'\n{"─"*60}')
    print(f'Hebbian adaptation under noise drift (p: 0.03 → 0.10)')
    print(f'{"─"*60}')

    model.eval()
    W = model.output_projector[0].weight.detach()
    n_synch = W.shape[1]

    # Baseline sync from training distribution
    with torch.no_grad():
        syn_cal, _ = code.generate_syndromes(256, args.noise_rate, args.num_rounds,
                                              device=args.device)
        _, _, sync_cal = model(syn_cal.flatten(1))
    baseline_sync = sync_cal.mean(0)

    # Evaluate: frozen model vs Hebbian-adapted
    drift_stream = DriftingSyndromeStream(
        code, args.num_rounds, p_start=0.03, p_end=0.10,
        n_samples=2000, batch_size=32, device=args.device)

    frozen_correct = 0
    hebb_correct = 0
    total_drift = 0

    hebb = HebbianPlasticity(n_synch, 4, lr=0.3, momentum=0.95)
    hebb.snapshot_baseline(baseline_sync.cpu())

    for flat, labels, p in drift_stream:
        with torch.no_grad():
            preds, _, sync = model(flat)
            base_logits = preds[:, :, -1]

            # Frozen
            frozen_pred = base_logits.argmax(1)
            frozen_correct += (frozen_pred == labels).sum().item()

            # Hebbian
            correction = hebb.apply(sync.cpu())
            adapted = base_logits.cpu() + correction
            hebb_pred = adapted.argmax(1)
            hebb_correct += (hebb_pred == labels.cpu()).sum().item()

            # Reward: positive reinforcement from correct predictions
            for i in range(flat.size(0)):
                if hebb_pred[i] == labels[i]:
                    hebb.update(sync[i].cpu(), W.cpu())

            total_drift += labels.size(0)

    print(f'  Frozen model:  {frozen_correct/total_drift:.1%}')
    print(f'  + Hebbian:     {hebb_correct/total_drift:.1%}')
    print(f'  Improvement:   {(hebb_correct-frozen_correct)/total_drift:+.1%}')
    print(f'  Zero backward passes.')

    print(f'\nCheckpoint: {args.save_dir}/best.pt')


if __name__ == '__main__':
    main()
