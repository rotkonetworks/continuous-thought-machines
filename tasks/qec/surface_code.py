"""Rotated surface code simulator for QEC decoding experiments.

Pure PyTorch — no quantum hardware needed. Generates syndrome histories
from depolarizing noise for training neural decoders.

A distance-d rotated surface code has:
    d² data qubits, (d²-1) stabilizers, 1 logical qubit

Each QEC round: measure all stabilizers → binary syndrome vector.
The decoder's job: given R rounds of syndromes, identify the logical
error class (I, X, Z, or Y).

Reference: Fowler et al., "Surface codes: Towards practical large-scale
quantum computation", PRA 86, 032324 (2012).
"""

import torch
import numpy as np
from typing import Tuple, Optional


class SurfaceCode:
    """Rotated surface code of distance d.

    Constructs the parity-check matrices for X and Z stabilizers
    and provides batched syndrome generation under depolarizing noise.
    """

    def __init__(self, distance: int = 5):
        assert distance % 2 == 1 and distance >= 3, "distance must be odd >= 3"
        self.d = distance
        self.n_data = distance * distance
        self.n_x_stab = (distance * distance - 1) // 2
        self.n_z_stab = (distance * distance - 1) // 2
        self.n_stab = self.n_x_stab + self.n_z_stab  # = d²-1

        # Build parity-check matrices
        self.hx, self.hz = self._build_parity_checks()

        # Logical operators (for determining logical error class)
        self.logical_x, self.logical_z = self._build_logicals()

    def _build_parity_checks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build X and Z stabilizer parity-check matrices.

        For the rotated surface code, we place qubits on a d×d grid.
        X-stabilizers are on white faces, Z-stabilizers on grey faces
        of the checkerboard pattern.

        Returns:
            hx: [n_x_stab, n_data] binary matrix (X stabilizers)
            hz: [n_z_stab, n_data] binary matrix (Z stabilizers)
        """
        d = self.d
        hx_rows = []
        hz_rows = []

        for r in range(d - 1):
            for c in range(d - 1):
                # Each face involves 4 qubits: (r,c), (r,c+1), (r+1,c), (r+1,c+1)
                row = torch.zeros(self.n_data, dtype=torch.float32)
                row[r * d + c] = 1
                row[r * d + c + 1] = 1
                row[(r + 1) * d + c] = 1
                row[(r + 1) * d + c + 1] = 1

                # Checkerboard: even faces = X-stab, odd = Z-stab
                if (r + c) % 2 == 0:
                    hx_rows.append(row)
                else:
                    hz_rows.append(row)

        # Boundary stabilizers (weight-2)
        # Top and bottom edges
        for c in range(0, d - 1, 2):
            row = torch.zeros(self.n_data, dtype=torch.float32)
            row[c] = 1
            row[c + 1] = 1
            hz_rows.append(row)

        for c in range(1, d - 1, 2):
            row = torch.zeros(self.n_data, dtype=torch.float32)
            row[(d - 1) * d + c] = 1
            row[(d - 1) * d + c + 1] = 1
            hz_rows.append(row)

        # Left and right edges
        for r in range(0, d - 1, 2):
            row = torch.zeros(self.n_data, dtype=torch.float32)
            row[r * d] = 1
            row[(r + 1) * d] = 1
            hx_rows.append(row)

        for r in range(1, d - 1, 2):
            row = torch.zeros(self.n_data, dtype=torch.float32)
            row[r * d + d - 1] = 1
            row[(r + 1) * d + d - 1] = 1
            hx_rows.append(row)

        hx = torch.stack(hx_rows) if hx_rows else torch.zeros(0, self.n_data)
        hz = torch.stack(hz_rows) if hz_rows else torch.zeros(0, self.n_data)

        # Pad/trim to exact sizes
        # The rotated surface code should have exactly (d²-1)/2 of each type
        # Adjust if boundary stabilizer construction over/under-counted
        target_x = self.n_x_stab
        target_z = self.n_z_stab
        if hx.shape[0] > target_x:
            hx = hx[:target_x]
        if hz.shape[0] > target_z:
            hz = hz[:target_z]
        if hx.shape[0] < target_x:
            pad = torch.zeros(target_x - hx.shape[0], self.n_data)
            hx = torch.cat([hx, pad])
        if hz.shape[0] < target_z:
            pad = torch.zeros(target_z - hz.shape[0], self.n_data)
            hz = torch.cat([hz, pad])

        return hx, hz

    def _build_logicals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build logical X and Z operators.

        Logical X: horizontal chain across the code.
        Logical Z: vertical chain across the code.
        """
        d = self.d
        # Logical X: first row of data qubits
        logical_x = torch.zeros(self.n_data, dtype=torch.float32)
        for c in range(d):
            logical_x[c] = 1

        # Logical Z: first column of data qubits
        logical_z = torch.zeros(self.n_data, dtype=torch.float32)
        for r in range(d):
            logical_z[r * d] = 1

        return logical_x, logical_z

    @torch.no_grad()
    def generate_syndromes(
        self,
        batch_size: int,
        noise_rate: float,
        num_rounds: int = 1,
        measurement_noise: Optional[float] = None,
        device: str = 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batched syndrome histories with depolarizing noise.

        Args:
            batch_size: number of syndrome instances
            noise_rate: physical error rate p (depolarizing)
            num_rounds: R, number of QEC measurement rounds
            measurement_noise: probability of syndrome bit flip (default: noise_rate)
            device: torch device

        Returns:
            syndromes: [batch_size, num_rounds, n_stab] float32 (0 or 1)
            labels: [batch_size] int64, logical error class:
                0 = I (no logical error)
                1 = X logical error
                2 = Z logical error
                3 = Y logical error (both X and Z)
        """
        if measurement_noise is None:
            measurement_noise = noise_rate

        hx = self.hx.to(device)
        hz = self.hz.to(device)
        lx = self.logical_x.to(device)
        lz = self.logical_z.to(device)

        B = batch_size
        n = self.n_data

        all_syndromes = []

        # Accumulate errors across rounds
        total_x_errors = torch.zeros(B, n, device=device)
        total_z_errors = torch.zeros(B, n, device=device)

        for r in range(num_rounds):
            # Depolarizing noise: each qubit gets X, Y, or Z error with prob p/3
            error_mask = (torch.rand(B, n, device=device) < noise_rate).float()
            error_type = torch.randint(0, 3, (B, n), device=device)

            x_errors = error_mask * (error_type != 2).float()  # X or Y
            z_errors = error_mask * (error_type != 0).float()  # Z or Y

            total_x_errors = (total_x_errors + x_errors) % 2
            total_z_errors = (total_z_errors + z_errors) % 2

            # Syndrome: X-stabilizers detect Z-errors, Z-stabilizers detect X-errors
            syn_x = (total_z_errors @ hx.T) % 2  # [B, n_x_stab]
            syn_z = (total_x_errors @ hz.T) % 2  # [B, n_z_stab]
            syndrome = torch.cat([syn_x, syn_z], dim=1)  # [B, n_stab]

            # Measurement noise
            if measurement_noise > 0:
                meas_flip = (torch.rand_like(syndrome) < measurement_noise).float()
                syndrome = (syndrome + meas_flip) % 2

            all_syndromes.append(syndrome)

        syndromes = torch.stack(all_syndromes, dim=1)  # [B, R, n_stab]

        # Logical error class from accumulated errors
        x_logical = ((total_x_errors @ lz) % 2).long()  # X errors on Z-logical
        z_logical = ((total_z_errors @ lx) % 2).long()  # Z errors on X-logical
        labels = x_logical + 2 * z_logical  # 0=I, 1=X, 2=Z, 3=Y

        return syndromes, labels

    def __repr__(self):
        return (f"SurfaceCode(d={self.d}, data_qubits={self.n_data}, "
                f"stabilizers={self.n_stab})")


class SyndromeDataset(torch.utils.data.Dataset):
    """On-the-fly syndrome generation for CTM training.

    Generates fresh random syndromes each time — no overfitting to
    specific error instances.
    """

    def __init__(self, distance: int = 5, noise_rate: float = 0.05,
                 num_rounds: int = 5, length: int = 100_000):
        self.code = SurfaceCode(distance)
        self.noise_rate = noise_rate
        self.num_rounds = num_rounds
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        syndromes, labels = self.code.generate_syndromes(
            batch_size=1,
            noise_rate=self.noise_rate,
            num_rounds=self.num_rounds,
        )
        # Flatten syndrome history: [R, n_stab] -> [R * n_stab]
        flat = syndromes[0].flatten().float()
        label = labels[0].long()
        return flat, label


class DriftingSyndromeStream:
    """Generates syndromes with smoothly drifting noise rate.

    For evaluating Hebbian online adaptation.
    """

    def __init__(self, code: SurfaceCode, num_rounds: int = 5,
                 p_start: float = 0.03, p_end: float = 0.10,
                 n_samples: int = 5000, batch_size: int = 32,
                 device: str = 'cpu'):
        self.code = code
        self.num_rounds = num_rounds
        self.p_start = p_start
        self.p_end = p_end
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        n_batches = self.n_samples // self.batch_size
        for i in range(n_batches):
            t = i / max(n_batches - 1, 1)
            p = self.p_start + t * (self.p_end - self.p_start)

            syndromes, labels = self.code.generate_syndromes(
                batch_size=self.batch_size,
                noise_rate=p,
                num_rounds=self.num_rounds,
                device=self.device,
            )
            # Flatten: [B, R, n_stab] -> [B, R*n_stab]
            flat = syndromes.flatten(1).float()
            yield flat, labels, p

    def __len__(self):
        return self.n_samples // self.batch_size


if __name__ == '__main__':
    # Quick test
    code = SurfaceCode(5)
    print(code)
    print(f"Hx: {code.hx.shape}, Hz: {code.hz.shape}")
    print(f"Logical X: {code.logical_x.nonzero().flatten().tolist()}")
    print(f"Logical Z: {code.logical_z.nonzero().flatten().tolist()}")

    syndromes, labels = code.generate_syndromes(1000, noise_rate=0.05, num_rounds=5)
    print(f"\nSyndromes: {syndromes.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Class distribution: {torch.bincount(labels, minlength=4).tolist()}")
    print(f"  I={labels.eq(0).sum()}, X={labels.eq(1).sum()}, "
          f"Z={labels.eq(2).sum()}, Y={labels.eq(3).sum()}")
