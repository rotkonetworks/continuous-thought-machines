//! Surface code simulator + QEC visualization for the browser demo.
//!
//! Generates syndromes from depolarizing noise, visualizes the lattice,
//! and feeds syndromes to the Hebbian engine for online adaptation.

/// Rotated surface code of distance d.
pub struct SurfaceCode {
    pub d: usize,
    pub n_data: usize,        // d²
    pub n_stab: usize,        // d² - 1
    pub n_x_stab: usize,      // (d²-1)/2
    pub n_z_stab: usize,      // (d²-1)/2
    pub hx: Vec<Vec<usize>>,  // X-stabilizer → data qubit indices
    pub hz: Vec<Vec<usize>>,  // Z-stabilizer → data qubit indices
    pub logical_x: Vec<usize>, // qubits in logical X operator
    pub logical_z: Vec<usize>, // qubits in logical Z operator
    rng_state: u64,
}

impl SurfaceCode {
    pub fn new(d: usize) -> Self {
        assert!(d >= 3 && d % 2 == 1, "distance must be odd >= 3");
        let n_data = d * d;
        let n_x_stab = (d * d - 1) / 2;
        let n_z_stab = (d * d - 1) / 2;

        let (hx, hz) = Self::build_parity_checks(d);
        let logical_x = (0..d).collect(); // first row
        let logical_z = (0..d).map(|r| r * d).collect(); // first column

        Self {
            d, n_data,
            n_stab: n_x_stab + n_z_stab,
            n_x_stab, n_z_stab,
            hx, hz, logical_x, logical_z,
            rng_state: 0x12345678_9abcdef0,
        }
    }

    fn build_parity_checks(d: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut hx = Vec::new();
        let mut hz = Vec::new();

        // Bulk stabilizers (weight-4 faces)
        for r in 0..d - 1 {
            for c in 0..d - 1 {
                let face = vec![
                    r * d + c,
                    r * d + c + 1,
                    (r + 1) * d + c,
                    (r + 1) * d + c + 1,
                ];
                if (r + c) % 2 == 0 {
                    hx.push(face);
                } else {
                    hz.push(face);
                }
            }
        }

        // Boundary stabilizers (weight-2)
        // Top edge
        let mut c = 0;
        while c < d - 1 {
            hz.push(vec![c, c + 1]);
            c += 2;
        }
        // Bottom edge
        c = 1;
        while c < d - 1 {
            hz.push(vec![(d - 1) * d + c, (d - 1) * d + c + 1]);
            c += 2;
        }
        // Left edge
        let mut r = 0;
        while r < d - 1 {
            hx.push(vec![r * d, (r + 1) * d]);
            r += 2;
        }
        // Right edge
        r = 1;
        while r < d - 1 {
            hx.push(vec![r * d + d - 1, (r + 1) * d + d - 1]);
            r += 2;
        }

        // Trim to exact counts
        let target = (d * d - 1) / 2;
        hx.truncate(target);
        hz.truncate(target);

        (hx, hz)
    }

    /// Simple xoshiro128+ PRNG (deterministic, no std dependency)
    fn rand_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }

    pub fn seed(&mut self, s: u64) {
        self.rng_state = s | 1; // ensure non-zero
    }

    /// Generate one syndrome measurement with depolarizing noise.
    ///
    /// Returns (syndrome, x_errors, z_errors) where syndrome is
    /// [n_stab] booleans and errors are [n_data] booleans.
    pub fn generate_syndrome(&mut self, noise_rate: f32) -> SyndromeResult {
        let n = self.n_data;
        let mut x_errors = vec![false; n];
        let mut z_errors = vec![false; n];

        // Depolarizing noise: each qubit gets X, Y, or Z with prob p/3
        for q in 0..n {
            if self.rand_f32() < noise_rate {
                let pauli = (self.rand_f32() * 3.0) as usize; // 0=X, 1=Y, 2=Z
                match pauli {
                    0 => x_errors[q] = true,
                    1 => { x_errors[q] = true; z_errors[q] = true; }
                    _ => z_errors[q] = true,
                }
            }
        }

        // Syndrome: X-stabs detect Z-errors, Z-stabs detect X-errors
        let mut syndrome = vec![false; self.n_stab];

        for (si, stab) in self.hx.iter().enumerate() {
            let mut parity = false;
            for &q in stab {
                if z_errors[q] { parity = !parity; }
            }
            syndrome[si] = parity;
        }

        for (si, stab) in self.hz.iter().enumerate() {
            let mut parity = false;
            for &q in stab {
                if x_errors[q] { parity = !parity; }
            }
            syndrome[self.n_x_stab + si] = parity;
        }

        // Logical error class
        let mut x_logical = false;
        for &q in &self.logical_z {
            if x_errors[q] { x_logical = !x_logical; }
        }
        let mut z_logical = false;
        for &q in &self.logical_x {
            if z_errors[q] { z_logical = !z_logical; }
        }
        let label = (x_logical as u8) + 2 * (z_logical as u8);

        SyndromeResult {
            syndrome,
            x_errors,
            z_errors,
            label,
        }
    }

    /// Generate R rounds of syndromes (with accumulating errors).
    pub fn generate_rounds(&mut self, noise_rate: f32, num_rounds: usize)
        -> (Vec<Vec<bool>>, u8)
    {
        let mut all_syndromes = Vec::with_capacity(num_rounds);
        let mut total_x = vec![false; self.n_data];
        let mut total_z = vec![false; self.n_data];

        for _ in 0..num_rounds {
            let result = self.generate_syndrome(noise_rate);

            // Accumulate errors (XOR)
            for q in 0..self.n_data {
                total_x[q] ^= result.x_errors[q];
                total_z[q] ^= result.z_errors[q];
            }

            all_syndromes.push(result.syndrome);
        }

        // Final logical class from accumulated errors
        let mut x_logical = false;
        for &q in &self.logical_z {
            if total_x[q] { x_logical = !x_logical; }
        }
        let mut z_logical = false;
        for &q in &self.logical_x {
            if total_z[q] { z_logical = !z_logical; }
        }
        let label = (x_logical as u8) + 2 * (z_logical as u8);

        (all_syndromes, label)
    }

    /// Flatten R rounds of syndromes into a single f32 vector
    /// (for feeding to the Hebbian engine).
    pub fn syndromes_to_vec(syndromes: &[Vec<bool>]) -> Vec<f32> {
        syndromes.iter()
            .flat_map(|s| s.iter().map(|&b| if b { 1.0 } else { 0.0 }))
            .collect()
    }

    /// Qubit position on the lattice (for visualization).
    pub fn qubit_pos(&self, q: usize) -> (f32, f32) {
        let r = q / self.d;
        let c = q % self.d;
        (c as f32, r as f32)
    }

    /// Stabilizer center position (for visualization).
    pub fn stab_pos(&self, stab_idx: usize) -> (f32, f32) {
        let stab = if stab_idx < self.n_x_stab {
            &self.hx[stab_idx]
        } else {
            &self.hz[stab_idx - self.n_x_stab]
        };

        let (mut cx, mut cy) = (0.0f32, 0.0f32);
        for &q in stab {
            let (x, y) = self.qubit_pos(q);
            cx += x;
            cy += y;
        }
        cx /= stab.len() as f32;
        cy /= stab.len() as f32;
        (cx, cy)
    }
}

pub struct SyndromeResult {
    pub syndrome: Vec<bool>,
    pub x_errors: Vec<bool>,
    pub z_errors: Vec<bool>,
    pub label: u8, // 0=I, 1=X, 2=Z, 3=Y
}

/// Stream of syndromes with drifting noise rate.
pub struct DriftStream {
    pub code: SurfaceCode,
    pub num_rounds: usize,
    pub p_start: f32,
    pub p_end: f32,
    pub n_total: usize,
    pub current: usize,
}

impl DriftStream {
    pub fn new(d: usize, num_rounds: usize, p_start: f32, p_end: f32, n_total: usize) -> Self {
        Self {
            code: SurfaceCode::new(d),
            num_rounds,
            p_start, p_end, n_total,
            current: 0,
        }
    }

    /// Get next syndrome with current noise rate.
    /// Returns None when stream is exhausted.
    pub fn next(&mut self) -> Option<(Vec<f32>, u8, f32)> {
        if self.current >= self.n_total {
            return None;
        }
        let t = self.current as f32 / (self.n_total - 1).max(1) as f32;
        let p = self.p_start + t * (self.p_end - self.p_start);

        let (syndromes, label) = self.code.generate_rounds(p, self.num_rounds);
        let flat = SurfaceCode::syndromes_to_vec(&syndromes);

        self.current += 1;
        Some((flat, label, p))
    }

    pub fn reset(&mut self) {
        self.current = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_code_d3() {
        let code = SurfaceCode::new(3);
        assert_eq!(code.n_data, 9);
        assert_eq!(code.n_stab, 8);
        assert_eq!(code.n_x_stab, 4);
        assert_eq!(code.n_z_stab, 4);
    }

    #[test]
    fn test_surface_code_d5() {
        let code = SurfaceCode::new(5);
        assert_eq!(code.n_data, 25);
        assert_eq!(code.n_stab, 24);
        assert_eq!(code.n_x_stab, 12);
        assert_eq!(code.n_z_stab, 12);
    }

    #[test]
    fn test_syndrome_generation() {
        let mut code = SurfaceCode::new(5);
        code.seed(42);
        let result = code.generate_syndrome(0.05);
        assert_eq!(result.syndrome.len(), 24);
        assert!(result.label < 4);
    }

    #[test]
    fn test_no_noise_no_errors() {
        let mut code = SurfaceCode::new(5);
        let result = code.generate_syndrome(0.0);
        // Zero noise → no errors → trivial syndrome → class I
        assert!(result.syndrome.iter().all(|&s| !s));
        assert_eq!(result.label, 0);
    }

    #[test]
    fn test_drift_stream() {
        let mut stream = DriftStream::new(3, 3, 0.01, 0.10, 10);
        let mut count = 0;
        while let Some((flat, label, p)) = stream.next() {
            assert_eq!(flat.len(), 3 * 8); // 3 rounds × 8 stabilizers
            assert!(label < 4);
            assert!(p >= 0.01 && p <= 0.10);
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_class_distribution() {
        let mut code = SurfaceCode::new(5);
        code.seed(123);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let result = code.generate_syndrome(0.05);
            counts[result.label as usize] += 1;
        }
        // With p=0.05, most should be class 0 (no error)
        // but all classes should appear
        assert!(counts[0] > 0);
        assert!(counts[1] > 0);
        assert!(counts[2] > 0);
        // Y errors are rarer
    }
}
