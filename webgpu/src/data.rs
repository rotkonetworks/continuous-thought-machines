//! Data structures and loading for CTM tick snapshots.

use serde::Deserialize;
use std::collections::VecDeque;

#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufRead, BufReader};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Single tick's metrics within a training step.
#[derive(Debug, Clone, Deserialize)]
pub struct TickData {
    pub k: u32,
    pub loss: f64,
    pub selected_pct: f64,
}

/// One training step's complete tick snapshot.
#[derive(Debug, Clone, Deserialize)]
pub struct Snapshot {
    pub step: u64,
    pub loss: f64,
    pub ticks: Vec<TickData>,
    #[serde(default)]
    pub certainty_mean: f64,
    #[serde(default)]
    pub grad_tick_frac: f64,
    // Optional GPU metrics
    #[serde(default)]
    pub gpu_util: Option<f64>,
    #[serde(default)]
    pub gpu_temp: Option<f64>,
    #[serde(default)]
    pub gpu_power: Option<f64>,
    #[serde(default)]
    pub vram_used: Option<f64>,
    #[serde(default)]
    pub tok_per_sec: Option<f64>,
    #[serde(default)]
    pub lr: Option<f64>,
    // Spectral metrics for c_proj plasticity tracking
    #[serde(default)]
    pub c_proj_rank90: Option<u32>,
    #[serde(default)]
    pub c_proj_rank99: Option<u32>,
    #[serde(default)]
    pub c_proj_concentration: Option<f64>,
    #[serde(default)]
    pub c_proj_sigma_max: Option<f64>,
    #[serde(default)]
    pub c_proj_condition: Option<f64>,
}

/// Ring buffer of snapshots with fast append and range access.
pub struct SnapshotBuffer {
    pub snapshots: VecDeque<Snapshot>,
    pub capacity: usize,
}

impl SnapshotBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, snap: Snapshot) {
        if self.snapshots.len() >= self.capacity {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snap);
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn last(&self) -> Option<&Snapshot> {
        self.snapshots.back()
    }

    /// Get the last N snapshots as a slice-like iterator.
    pub fn tail(&self, n: usize) -> impl Iterator<Item = &Snapshot> {
        let skip = self.snapshots.len().saturating_sub(n);
        self.snapshots.iter().skip(skip)
    }

    /// Get a window of N snapshots ending at a given index.
    /// If end_idx is None or >= len, returns the last N.
    pub fn window(&self, n: usize, end_idx: Option<usize>) -> impl Iterator<Item = &Snapshot> {
        let total = self.snapshots.len();
        let end = end_idx.map(|e| e.min(total)).unwrap_or(total);
        let start = end.saturating_sub(n);
        self.snapshots.iter().skip(start).take(end - start)
    }

    /// Find the index of the snapshot closest to a given step.
    pub fn index_of_step(&self, step: u64) -> Option<usize> {
        self.snapshots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| (s.step as i64 - step as i64).unsigned_abs())
            .map(|(i, _)| i)
    }

    /// Load from a JSON array string (for WASM).
    pub fn from_json_array(json: &str) -> Result<Self, serde_json::Error> {
        let snapshots: Vec<Snapshot> = serde_json::from_str(json)?;
        let mut buf = Self::new(10_000);
        // Deduplicate by step: keep last entry per step with >1 tick
        let mut seen = std::collections::HashMap::new();
        for snap in snapshots {
            if snap.ticks.len() > 1 {
                seen.insert(snap.step, snap);
            }
        }
        let mut ordered: Vec<_> = seen.into_iter().collect();
        ordered.sort_by_key(|(step, _)| *step);
        for (_, snap) in ordered {
            buf.push(snap);
        }
        Ok(buf)
    }

    /// Load from JSONL file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_jsonl(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut buf = Self::new(10_000);

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(snap) = serde_json::from_str::<Snapshot>(line) {
                buf.push(snap);
            }
        }

        Ok(buf)
    }

    /// Append new lines from JSONL file starting at byte offset.
    /// Returns new offset.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_new_lines(&mut self, path: &Path, offset: u64) -> std::io::Result<u64> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(path)?;
        let end = file.metadata()?.len();
        if end <= offset {
            return Ok(offset);
        }

        file.seek(SeekFrom::Start(offset))?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;

        for line in buf.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(snap) = serde_json::from_str::<Snapshot>(line) {
                self.push(snap);
            }
        }

        Ok(end)
    }

    /// Get a sparse sample of all data, returning exactly `n` evenly-spaced snapshots.
    /// This lets users see beginning to end at once.
    pub fn sparse(&self, n: usize) -> Vec<&Snapshot> {
        let total = self.snapshots.len();
        if total <= n {
            return self.snapshots.iter().collect();
        }
        let step = total as f64 / n as f64;
        (0..n)
            .map(|i| {
                let idx = (i as f64 * step) as usize;
                &self.snapshots[idx.min(total - 1)]
            })
            .collect()
    }

    /// Get K (number of ticks) from first snapshot.
    pub fn k(&self) -> usize {
        self.snapshots
            .front()
            .map(|s| s.ticks.len())
            .unwrap_or(32)
    }

    /// Get step range.
    pub fn step_range(&self) -> (u64, u64) {
        let first = self.snapshots.front().map(|s| s.step).unwrap_or(0);
        let last = self.snapshots.back().map(|s| s.step).unwrap_or(0);
        (first, last)
    }

    /// Find snapshot closest to a given step.
    pub fn at_step(&self, step: u64) -> Option<&Snapshot> {
        self.snapshots
            .iter()
            .min_by_key(|s| (s.step as i64 - step as i64).unsigned_abs())
    }
}

// ─── Binary data parsing for Hebbian demo ───────────────────────────

/// Parse a flat f32 buffer from little-endian bytes.
pub fn parse_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Parse a flat u16 buffer from little-endian bytes.
pub fn parse_u16_buffer(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Metadata for the Hebbian demo dataset.
#[derive(Debug, Clone, Deserialize)]
pub struct HebbianMetadata {
    pub n_images: usize,
    pub n_synch: usize,
    pub n_output: usize,
    pub class_names: Vec<String>,
    #[serde(default)]
    pub bounds: Option<BoundsInfo>,
}

/// Bound analysis results from the pretrained model.
#[derive(Debug, Clone, Deserialize)]
pub struct BoundsInfo {
    pub synapse_rank_90: usize,
    pub synapse_activation_rank: usize,
    pub synapse_utilization_pct: f64,
    pub synapse_condition: f64,
    #[serde(default)]
    pub synapse_top_svs: Vec<f64>,
    pub n_dead: usize,
    pub n_inactive: usize,
    pub neuron_diversity: f64,
    pub best_tick: usize,
    pub n_overthinking: usize,
    pub n_ticks: usize,
    pub model_dim: usize,
    pub bottleneck: String,
    // Per-tick data (if available)
    #[serde(default)]
    pub tick_losses: Vec<f64>,         // [n_ticks] loss at each tick
    #[serde(default)]
    pub tick_improvements: Vec<f64>,   // [n_ticks] improvement from prev tick
    #[serde(default)]
    pub overthinking_ticks: Vec<usize>, // which ticks are overthinking
    #[serde(default)]
    pub neuron_contributions: Vec<f64>, // [top_k] sorted neuron contributions
    // SDP per-tick bounds (achieved vs optimal)
    #[serde(default)]
    pub tick_achieved: Vec<f64>,   // [n_ticks] NLM achieved loss per tick
    #[serde(default)]
    pub tick_bounds: Vec<f64>,     // [n_ticks] SDP optimal bound per tick
    #[serde(default)]
    pub tick_gaps: Vec<f64>,       // [n_ticks] achieved - bound = wasted
}
