#!/usr/bin/env python3
"""Benchmark all Hebbian strategy combinations.

Tests gate modes, confidence weighting, and extrapolation check
on cached data. Run after demo.py has generated results.

Usage:
    python utils/bounds/test_strategies.py
"""

import torch
import itertools
import time
from utils.bounds.training import HebbianPlasticity


def run_strategy(sync_signals, logits, labels, baseline_sync, output_weights,
                 **kwargs):
    """Run one Hebbian strategy, return accuracy."""
    N = len(labels)
    n_synch = sync_signals.shape[1]
    n_output = logits.shape[1]

    hebb = HebbianPlasticity(n_synch, n_output, **kwargs)
    hebb.snapshot_baseline(baseline_sync)

    correct = 0
    for i in range(N):
        sync_i = sync_signals[i]
        base_logits = logits[i]

        correction = hebb.apply(sync_i.unsqueeze(0))[0]
        adapted = base_logits + correction
        pred = adapted.argmax().item()
        label = labels[i].item()

        is_correct = (pred == label)
        if is_correct:
            correct += 1

        # Confidence from softmax
        conf = adapted.softmax(0).max().item()

        # Report loss for extrapolation check
        loss = torch.nn.functional.cross_entropy(
            adapted.unsqueeze(0), labels[i:i+1]).item()
        hebb.report_loss(loss)

        hebb.update(sync_i, output_weights,
                    reward=is_correct, confidence=conf)

    return correct / N


def main():
    # Try to load cached data from the demo
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    # Generate synthetic test data if real data not available
    print("Generating test data...")
    N = 200
    n_synch = 64
    n_output = 10
    torch.manual_seed(42)

    # Simulate: baseline sync, shifted sync signals, random logits with some signal
    baseline_sync = torch.randn(n_synch) * 0.1
    sync_signals = torch.randn(N, n_synch) * 0.1 + 0.05  # slight shift
    true_labels = torch.randint(0, n_output, (N,))

    # Logits with some correctness built in (~40% base accuracy)
    logits = torch.randn(N, n_output) * 0.5
    for i in range(N):
        if torch.rand(1).item() < 0.4:
            logits[i, true_labels[i]] += 2.0  # make it correct sometimes

    output_weights = torch.randn(n_output, n_synch) * 0.1

    base_acc = (logits.argmax(1) == true_labels).float().mean().item()
    print(f"Base accuracy: {base_acc:.1%}")

    # Strategy grid
    gates = ['median', 'percentile', 'topk', 'none']
    confidence_opts = [False, True]
    extrapolation_opts = [False, True]
    lrs = [0.1, 0.3, 1.0]

    results = []

    print(f"\n{'Gate':<12s} {'Conf':>5s} {'Extrap':>7s} {'LR':>5s} {'Acc':>7s} {'Δ':>7s}")
    print("-" * 50)

    for gate, conf, extrap, lr in itertools.product(
        gates, confidence_opts, extrapolation_opts, lrs
    ):
        acc = run_strategy(
            sync_signals, logits, true_labels, baseline_sync, output_weights,
            lr=lr, momentum=0.95,
            gate=gate,
            confidence_weighted=conf,
            use_extrapolation_check=extrap,
        )
        delta = acc - base_acc
        results.append({
            'gate': gate, 'confidence': conf, 'extrapolation': extrap,
            'lr': lr, 'accuracy': acc, 'delta': delta,
        })

        if delta > 0.01:  # only print improvements
            print(f"{gate:<12s} {str(conf):>5s} {str(extrap):>7s} {lr:>5.1f} "
                  f"{acc:>6.1%} {delta:>+6.1%}")

    # Top 5
    results.sort(key=lambda r: r['accuracy'], reverse=True)
    print(f"\nTop 5 strategies:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. gate={r['gate']:<12s} conf={r['confidence']!s:<6s} "
              f"extrap={r['extrapolation']!s:<6s} lr={r['lr']:.1f} "
              f"→ {r['accuracy']:.1%} ({r['delta']:+.1%})")

    # Worst 3
    print(f"\nWorst 3:")
    for r in results[-3:]:
        print(f"     gate={r['gate']:<12s} conf={r['confidence']!s:<6s} "
              f"extrap={r['extrapolation']!s:<6s} lr={r['lr']:.1f} "
              f"→ {r['accuracy']:.1%} ({r['delta']:+.1%})")


if __name__ == '__main__':
    main()
