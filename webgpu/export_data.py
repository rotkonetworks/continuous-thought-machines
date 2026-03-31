#!/usr/bin/env python3
"""Export binary assets for the WebGPU Hebbian demo.

Runs the pretrained ImageNet CTM, caches sync signals + logits,
writes flat binary files for loading in the browser.

Output files (~40MB total):
    assets/hebbian/sync_signals.bin   [N, n_synch] float32
    assets/hebbian/logits_final.bin   [N, 1000] float32
    assets/hebbian/logits_t10.bin     [N, 1000] float32
    assets/hebbian/output_weights.bin [1000, n_synch] float32
    assets/hebbian/baseline_sync.bin  [n_synch] float32
    assets/hebbian/labels.bin         [N] uint16
    assets/hebbian/sync_pca.bin       [N, 2] float32
    assets/hebbian/metadata.json

Usage:
    cd continuous-thought-machines
    python webgpu/export_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import json

from torchvision import transforms
from datasets import load_dataset

from models.ctm import ContinuousThoughtMachine
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES

CKPT = 'checkpoints/imagenet/ctm_imagenet_D=4096_T=50_M=25.pt'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'hebbian')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EXPORT = 200
N_BASELINE = 20

IMAGENET_SYNSETS = list(IMAGENET2012_CLASSES.keys())
IMAGENET_CLASSES = list(IMAGENET2012_CLASSES.values())

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model from {CKPT}...")
    checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
    ma = checkpoint['args']
    if not hasattr(ma, 'backbone_type'):
        ma.backbone_type = (f'{ma.resnet_type}-'
                            f'{getattr(ma, "resnet_feature_scales", [4])[-1]}')
    if not hasattr(ma, 'neuron_select_type'):
        ma.neuron_select_type = 'first-last'

    model = ContinuousThoughtMachine(
        iterations=50, d_model=ma.d_model, d_input=ma.d_input, heads=ma.heads,
        n_synch_out=ma.n_synch_out, n_synch_action=ma.n_synch_action,
        synapse_depth=ma.synapse_depth, memory_length=ma.memory_length,
        deep_nlms=ma.deep_memory, memory_hidden_dims=ma.memory_hidden_dims,
        do_layernorm_nlm=ma.do_normalisation,
        backbone_type=ma.backbone_type,
        positional_embedding_type=ma.positional_embedding_type,
        out_dims=ma.out_dims, prediction_reshaper=[-1], dropout=0,
        neuron_select_type=ma.neuron_select_type,
        n_random_pairing_self=ma.n_random_pairing_self,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    del checkpoint
    model = model.to(DEVICE).eval()

    W = model.output_projector[0].weight.detach().cpu()
    n_out, n_synch = W.shape
    print(f"  Output projector: [{n_out}, {n_synch}]")

    # Label mapping
    ds_info = load_dataset('zh-plus/tiny-imagenet', split='valid')
    label_names = ds_info.features['label'].names
    tiny_to_imagenet = {}
    for ti, wnid in enumerate(label_names):
        if wnid in IMAGENET_SYNSETS:
            tiny_to_imagenet[ti] = IMAGENET_SYNSETS.index(wnid)

    # Forward passes
    print(f"Running forward passes on {N_EXPORT + N_BASELINE} images...")
    ds = load_dataset('zh-plus/tiny-imagenet', split='valid', streaming=True)

    baseline_syncs = []
    syncs = []
    logits_final = []
    logits_t10 = []
    labels = []

    for i, sample in enumerate(ds):
        if len(syncs) >= N_EXPORT and len(baseline_syncs) >= N_BASELINE:
            break

        il = tiny_to_imagenet.get(sample['label'])
        if il is None:
            continue

        x = TRANSFORM(sample['image'].convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds, _, sync = model(x)

        if len(baseline_syncs) < N_BASELINE:
            baseline_syncs.append(sync[0].cpu())
        else:
            syncs.append(sync[0].cpu().numpy())
            logits_final.append(preds[0, :, -1].cpu().numpy())
            logits_t10.append(preds[0, :, 9].cpu().numpy())
            labels.append(il)

        if (len(syncs) + len(baseline_syncs)) % 50 == 0:
            print(f"  {len(baseline_syncs)} baseline + {len(syncs)} eval...")

    baseline_sync = torch.stack(baseline_syncs).mean(0).numpy()
    N = len(syncs)
    print(f"  Done: {N} images")

    # PCA of sync signals (avoid sklearn — use SVD)
    sync_arr = np.array(syncs, dtype=np.float32)
    centered = sync_arr - sync_arr.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_2d = (U[:, :2] * S[:2]).astype(np.float32)

    # Write binary files
    print(f"Writing to {OUT_DIR}/...")
    sync_arr.tofile(os.path.join(OUT_DIR, 'sync_signals.bin'))
    np.array(logits_final, dtype=np.float32).tofile(os.path.join(OUT_DIR, 'logits_final.bin'))
    np.array(logits_t10, dtype=np.float32).tofile(os.path.join(OUT_DIR, 'logits_t10.bin'))
    W.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, 'output_weights.bin'))
    baseline_sync.astype(np.float32).tofile(os.path.join(OUT_DIR, 'baseline_sync.bin'))
    np.array(labels, dtype=np.uint16).tofile(os.path.join(OUT_DIR, 'labels.bin'))
    pca_2d.tofile(os.path.join(OUT_DIR, 'sync_pca.bin'))

    metadata = {
        'n_images': N,
        'n_synch': n_synch,
        'n_output': n_out,
        'class_names': [c.split(',')[0] for c in IMAGENET_CLASSES],
    }
    with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    # Report sizes
    total = 0
    for fname in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, fname)
        sz = os.path.getsize(fpath)
        total += sz
        print(f"  {fname:<25s} {sz/1e6:>6.1f} MB")
    print(f"  {'TOTAL':<25s} {total/1e6:>6.1f} MB")


if __name__ == '__main__':
    main()
