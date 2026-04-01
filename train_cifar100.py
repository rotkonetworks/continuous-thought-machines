#!/usr/bin/env python3
"""Train small CTM on CIFAR-100 with BoundGuidedLoss.

~50 min on RX 7600M XT. Saves checkpoint to checkpoints/cifar100/best.pt

Usage:
    python train_cifar100.py
"""
import torch
import torch.nn as nn
import time
import os

from models.ctm import ContinuousThoughtMachine
from torchvision import transforms, datasets
from utils.bounds.training import BoundGuidedLoss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/cifar100'
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_ds = datasets.CIFAR100('data/', train=True, transform=transforms.Compose([
        transforms.Resize(64), transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ]), download=True)
    test_ds = datasets.CIFAR100('data/', train=False, transform=transforms.Compose([
        transforms.Resize(64), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ]), download=True)

    # Model: small CTM with ResNet18 backbone
    model = ContinuousThoughtMachine(
        iterations=10, d_model=128, d_input=64, heads=2,
        n_synch_out=64, n_synch_action=32, synapse_depth=1, memory_length=6,
        deep_nlms=True, memory_hidden_dims=16, do_layernorm_nlm=False,
        backbone_type='resnet18-2', positional_embedding_type='none',
        out_dims=100, prediction_reshaper=[-1],
        neuron_select_type='random-pairing',
    ).to(device)

    # Init lazy layers
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64).to(device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params:,} params on {device}')

    # BoundGuidedLoss: per-tick auxiliary supervision + overthinking penalty
    base_loss = nn.CrossEntropyLoss()
    bound_loss = BoundGuidedLoss(base_loss, aux_weight=0.1, overthink_penalty=0.05)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_test = 0
    for epoch in range(30):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        t0 = time.time()

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _, _ = model(xb)  # pred: [B, 100, T=10]

            # Final tick for base prediction
            final_logits = pred[:, :, -1]

            # Per-tick outputs for BoundGuidedLoss
            tick_outputs = [pred[:, :, t] for t in range(pred.shape[2])]

            losses = bound_loss(final_logits, yb, tick_outputs=tick_outputs)
            loss = losses['total']

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            correct += (final_logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
            epoch_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        tc = 0
        tt = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _, _ = model(xb)
                tc += (pred[:, :, -1].argmax(1) == yb).sum().item()
                tt += yb.size(0)

        test_acc = tc / tt
        if test_acc > best_test:
            best_test = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
                'args': {
                    'iterations': 10, 'd_model': 128, 'd_input': 64,
                    'heads': 2, 'n_synch_out': 64, 'n_synch_action': 32,
                    'synapse_depth': 1, 'memory_length': 6,
                    'backbone_type': 'resnet18-2', 'out_dims': 100,
                },
            }, os.path.join(save_dir, 'best.pt'))

        elapsed = time.time() - t0
        print(f'epoch {epoch+1:2d}: train={correct/total:.1%} test={test_acc:.1%} '
              f'best={best_test:.1%} loss={epoch_loss/len(loader):.3f} ({elapsed:.0f}s)',
              flush=True)

    print(f'\nDone. Best test accuracy: {best_test:.1%}')
    print(f'Checkpoint: {save_dir}/best.pt')


if __name__ == '__main__':
    main()
