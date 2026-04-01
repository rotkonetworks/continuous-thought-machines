"""MWPM baseline decoder using PyMatching.

Minimum Weight Perfect Matching is the standard QEC decoder.
Near-optimal for code-capacity noise with known noise rate.
Cannot adapt online to noise drift without recalibration.

Usage:
    python -m tasks.qec.mwpm_baseline
"""

import torch
import numpy as np
from scipy.sparse import csc_matrix
import pymatching

from tasks.qec.surface_code import SurfaceCode


def build_matching(code: SurfaceCode, noise_rate: float):
    """Build PyMatching decoder from surface code parity-check matrix.

    We decode X and Z errors separately:
    - X errors detected by Z-stabilizers (Hz)
    - Z errors detected by X-stabilizers (Hx)
    """
    # PyMatching needs scipy sparse matrices
    hx_sparse = csc_matrix(code.hx.numpy())
    hz_sparse = csc_matrix(code.hz.numpy())

    # Build matchers with noise weights
    # Weight = log((1-p)/p) for depolarizing noise at rate p per qubit
    p_eff = noise_rate * 2 / 3  # prob of X or Z error (Y counts as both)
    weights = np.full(code.n_data, np.log((1 - p_eff) / max(p_eff, 1e-10)))

    matcher_x = pymatching.Matching(hz_sparse, weights=weights)  # decodes X errors
    matcher_z = pymatching.Matching(hx_sparse, weights=weights)  # decodes Z errors

    return matcher_x, matcher_z


def mwpm_decode(code, syndromes, labels, noise_rate):
    """Decode a batch of syndromes using MWPM.

    Args:
        code: SurfaceCode instance
        syndromes: [B, R, n_stab] tensor
        labels: [B] true logical error classes
        noise_rate: physical error rate (for weight calculation)

    Returns:
        accuracy: fraction correct
    """
    matcher_x, matcher_z = build_matching(code, noise_rate)

    B = syndromes.shape[0]
    n_x = code.n_x_stab
    n_z = code.n_z_stab

    correct = 0
    for i in range(B):
        # Use final-round syndrome (collapse R rounds)
        syn = syndromes[i, -1].numpy().astype(np.uint8)
        syn_x = syn[:n_x]  # X-stabilizer syndrome (detects Z errors)
        syn_z = syn[n_x:]  # Z-stabilizer syndrome (detects X errors)

        # Decode
        x_correction = matcher_x.decode(syn_z)  # X errors from Z-syndrome
        z_correction = matcher_z.decode(syn_x)  # Z errors from X-syndrome

        # Logical error: check if correction differs from true error by a logical op
        # In our simplified setup: check if the predicted class matches
        x_logical = int(np.dot(x_correction, code.logical_z.numpy()) % 2)
        z_logical = int(np.dot(z_correction, code.logical_x.numpy()) % 2)
        predicted_class = x_logical + 2 * z_logical

        if predicted_class == labels[i].item():
            correct += 1

    return correct / B


def evaluate_mwpm(distance=5, noise_rates=None, num_rounds=5, n_test=2000):
    """Evaluate MWPM across noise rates."""
    if noise_rates is None:
        noise_rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

    code = SurfaceCode(distance)
    print(f"MWPM baseline: d={distance}, {code.n_stab} stabilizers, R={num_rounds}")
    print(f"{'p':>6s}  {'accuracy':>8s}  {'log_err_rate':>12s}")
    print("-" * 30)

    results = {}
    for p in noise_rates:
        syndromes, labels = code.generate_syndromes(n_test, p, num_rounds)
        acc = mwpm_decode(code, syndromes, labels, p)
        ler = 1 - acc
        results[p] = {'accuracy': acc, 'logical_error_rate': ler}
        print(f"{p:>6.3f}  {acc:>8.1%}  {ler:>12.4f}")

    return results


if __name__ == '__main__':
    for d in [3, 5]:
        print(f"\n{'='*40}")
        evaluate_mwpm(distance=d)
