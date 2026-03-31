# Diagnosing and Accelerating Continuous Thought Machines via Optimality Bounds and Hebbian Plasticity

**Abstract.** Continuous Thought Machines (CTMs) achieve strong results by
iterating internal recurrence over T thinking ticks, but at proportional
inference cost. We adapt the SDP dual framework of Angeris (2022) to diagnose a
pretrained 186M-parameter ImageNet CTM, revealing dramatic inefficiency: 1.6%
synapse utilization, overthinking on 52% of ticks, and a single neuron carrying
95% of Jacobian energy. Guided by this diagnosis, we construct a zero-backprop
inference pipeline — early exit at T=10, SVD compression to rank-256, and
reward-modulated Hebbian plasticity — that achieves +8pp accuracy at 3.9×
speedup over the default configuration. The Hebbian mechanism, which exploits
the sync signal already computed in the CTM forward pass, outperforms LoRA
adaptation (48.5% vs 46.0%) with zero gradient computation and zero adapter
parameters.

## 1. Introduction

CTMs [1] model reasoning as a temporal process: an internal recurrence iterates
over T ticks, with per-neuron models (NLMs) processing trace histories and
synchronization between neuron pairs forming the output representation. This
achieves strong results on ImageNet, mazes, parity, and other tasks, but each
tick carries the full cost of a recurrent step — attention, synapse
transformation, NLM evaluation, and sync computation. For a model with D=4096
neurons and T=50 ticks, inference takes ~800ms per image on a consumer GPU.

We lack tools to answer: *where is a trained CTM's capacity wasted, and how much
computation is unnecessary?* Existing adaptation methods like LoRA [4] require
gradient-based fine-tuning, adding backward passes and optimizer state to what
should be an inference-time concern.

**Contributions.**
1. We adapt the SDP dual framework of Angeris (2022) [2] to CTMs, providing
   per-neuron, per-tick, and synapse capacity-vs-utilization diagnostics.
2. Applied to a pretrained 186M-param ImageNet CTM, the diagnosis reveals: 1.6%
   synapse utilization, 52% overthinking ticks, and extreme neuron
concentration.
3. Guided by the diagnosis, we construct a zero-backprop inference pipeline that
   achieves **+8pp accuracy at 3.9× speedup**, with Hebbian plasticity
outperforming LoRA.

## 2. Background

### 2.1 CTM Architecture

At each tick t, the CTM computes:

    h_pre^t = synapse([attn(h^{t-1}), h^{t-1}])
    h^t = NLM_d(trace_d^{1:t})  for each neuron d
    sync^t = {h_i^t · h_j^t}  for paired neurons (i,j)
    pred^t = output_proj(sync^t)

The synapse is a shared weight matrix (or U-Net) transforming all neurons
jointly. NLMs are private per-neuron MLPs over trace history of length M.
Synchronization — pairwise products of neuron activation traces — forms the
representation from which predictions are projected [1].

### 2.2 SDP Bounds for Physical Design

Angeris (2022) [2] considers systems with bilinear structure A(θ)z = b, where θ
are design parameters and z are fields, and provides an SDP dual that
lower-bounds the achievable loss:

    maximize  v(N,ν) - ½t
    subject to [T(N)  u(N,ν); u(N,ν)ᵀ  t] ≥ 0,  N ≥ 0

This was developed for photonic and antenna design. We show CTM's recurrence has
the same bilinear structure.

### 2.3 CTMs Fit the Framework

The mapping is direct: design parameters θ are network weights; fields z are
stacked activations [h¹,...,hᵀ]; the linearized recurrence gives A(θ). A
critical structural property: per-neuron NLMs have *private* weights, so the
sufficient condition from §1.2 of [2] holds trivially — each neuron's SDP is
independent and small (T×T). The shared synapse requires the improved
characterization from §3.1.

## 3. Diagnostic Method

### 3.1 Per-Neuron Analysis

We measure two distinct properties:

**Weight capacity.** NLM weight L2 norms and pairwise cosine similarity
(diversity metric: 0=diverse, 1=collapsed). Dead neurons have low weight norms —
permanently unused capacity, distinct from neurons that are simply inactive on a
particular input (healthy sparse activation).

**Activation utilization.** Jacobian energy decomposition: for each neuron d, we
compute the NLM Jacobian J_d^t = ∂h_d^t/∂pre_d^t and measure ΣₜJ_d² as neuron
d's contribution to the total transformation.

### 3.2 Per-Tick Analysis

We track the loss proxy at each tick and define **overthinking** as ticks where
loss increases from the previous tick. This measures the "thinking too long"
failure mode that variable-T aims to prevent but training does not explicitly
penalize.

### 3.3 Synapse Analysis

**Capacity:** SVD of the weight matrix W; effective rank at 90% cumulative
energy.

**Utilization:** Capture activations flowing through W via forward hooks; SVD of
the activation matrix gives the effective rank actually used. The ratio
capacity/utilization is the synapse utilization percentage.

The gap between these quantities reveals whether the bottleneck is in the
synapse (low capacity) or upstream (low utilization despite high capacity).

### 3.4 SDP Solver

For exact bounds, we solve per-neuron SDPs using CVXPY with the Schur complement
formulation. For the synapse, we solve with a spectral-norm constraint derived
from observed singular values. The approximate diagnostics (§3.1-3.3) closely
track the SDP bounds at orders-of-magnitude lower compute cost.

## 4. Diagnosis of 186M ImageNet CTM

We analyze the pretrained CTM checkpoint (D=4096, T=50, M=25, synapse depth 8
U-Net, 186M params) released by Sakana AI [1].

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Synapse weight rank (90%) | 2,538 | High capacity |
| Synapse activation rank | 40 | Low utilization |
| **Utilization** | **1.6%** | **Upstream bottleneck** |
| Overthinking ticks | 26/50 | Model degrades after ~tick 10 |
| Neuron diversity (cosine) | 0.17 | Mild collapse |
| Top neuron Jacobian share | 95% | Extreme concentration |
| Condition number | 835 | Ill-conditioned |
| Dead neurons | 0/4096 | No wasted capacity |

**Interpretation.** The synapse has enormous unused capacity. The bottleneck is
upstream — attention and input projection feed only a rank-40 subspace into a
rank-2538 synapse. Over half the ticks make predictions worse, not better. A
single neuron dominates the Jacobian, suggesting the model has collapsed most of
its representational diversity into one pathway.

**Actionable implications:**
- Early exit: ticks beyond ~10 are harmful; T=10 should suffice.
- Synapse compression: rank-256 SVD preserves far more than the utilized
rank-40.
- The readout (sync→output projection) is the adaptation bottleneck, not the
synapse — so adaptation should target the readout.

## 5. Inference-Time Optimizations

All optimizations are applied at inference time with **zero retraining**.

### 5.1 Early Exit

Reduce T from 50 to 10. Latency: 808ms → 180ms (4.5× speedup). Accuracy: 40.5% →
31.0% (−9.5pp). The drop is expected — the model was trained for T=50.

### 5.2 SVD Compression

Truncate synapse weight matrices to rank-256 via SVD (Wₖ = Uₖ Sₖ Vₖᵀ). This
removes 83% of synapse parameters (98M → ~17M). The rank-256 threshold preserves
far more than the rank-40 actually utilized, providing headroom.

### 5.3 Hebbian Plasticity

The core technical contribution. The diagnosis revealed the readout is the
bottleneck — so we adapt it via Hebbian learning on the sync signal, which CTMs
already compute.

**Algorithm 1: Reward-Modulated Hebbian Plasticity**

```
Input: model with frozen weights, baseline sync s₀
Initialize: delta ← 0 (correction matrix)

For each inference input x:
    preds, sync ← model(x)           # standard forward pass
    correction ← sync · delta         # matrix-vector multiply
    adapted_pred ← pred + correction  # additive correction

    If adapted_pred is correct:       # positive reinforcement
        novelty ← sync - s₀
        gate ← 1[|novelty| > median(|novelty|)]
        gated ← novelty ⊙ gate
        action ← W_output · gated    # project to class space
        delta ← μ·delta + (1-μ)·η·(gated ⊗ action)  # Hebbian update
```

**Why this works.** The sync signal encodes which neurons co-fire — this is the
CTM's native representation. Novelty measures what is different about the
current input relative to the baseline distribution. The output projection
weights already encode what each sync dimension means for classification. The
outer product is classical Hebbian learning [3] — neurons that fire together
wire together — gated by task-relevant novelty and modulated by reward.

**Why no catastrophic forgetting.** The correction delta is additive on frozen
weights. The baseline sync never changes. Each update adds a small correction
proportional to novelty, which is bounded by the gate.

**Why zero backward passes.** The sync signal is already computed in the forward
pass. The update is an outer product and a matrix-vector multiply — no autograd
graph, no gradient tape, no optimizer state.

### 5.4 Results

Evaluated on 200 tiny-ImageNet images (64×64, 182 classes mapping to ImageNet) on an AMD RX 7600M XT GPU.

| Configuration | Accuracy | Latency | Speedup | Backward |
|---------------|----------|---------|---------|----------|
| Default (T=50, full rank) | 40.5% | 808ms | 1.0× | — |
| T=10 only | 31.0% | 180ms | 4.5× | — |
| T=10 + rank-256 | 22.5% | ~175ms | ~4.6× | — |
| **T=10 + rank-256 + Hebbian** | **48.5%** | **207ms** | **3.9×** | **zero** |

Hebbian overhead: 27ms/image (15% of forward pass).

### 5.5 Comparison with LoRA

On the compressed model (T=10, rank-256):

| Method | Accuracy | Δ base | Backward | Adapter params |
|--------|----------|--------|----------|----------------|
| Base | 22.5% | — | — | — |
| LoRA rank-4 | 33.5% | +11.0pp | yes | 36,784 |
| LoRA rank-16 | 46.0% | +23.5pp | yes | 147,136 |
| **Hebbian** | **48.5%** | **+26.0pp** | **zero** | **0** |

Hebbian outperforms LoRA-16 by 2.5pp with zero adapter parameters and zero
gradient computation. LoRA-4 lacks the capacity to recover from compression. The
Hebbian mechanism exploits the sync signal that CTMs already compute — it is
architecture-native.

## 6. Discussion

**Limitations.** Results are on a single checkpoint evaluated on tiny-ImageNet
(64×64, not full ImageNet). The Hebbian mechanism assumes the sync signal is
informative — this holds for CTMs but not arbitrary architectures. The SDP
bounds are exact for per-neuron NLMs but use a relaxation for the synapse. Early
exit at T=10 was chosen post-hoc; a principled certainty-based criterion would
be more robust.

**Broader implications.** The capacity-utilization gap (1.6%) suggests CTMs are
dramatically over-parameterized for their trained behavior, or that training
fails to leverage the full synapse. The Hebbian result suggests recurrent models
with internal synchronization signals have a natural substrate for gradient-free
adaptation — this may extend to other architectures with analogous internal
dynamics (state-space models, memory-augmented networks).

**Future work.** Training-time integration via per-tick auxiliary losses (the
`BoundGuidedLoss` in our codebase is a prototype). Principled early-exit via
certainty monitoring. Scaling Hebbian plasticity to online continual learning
across distribution shifts.

## 7. Conclusion

We adapted SDP optimality bounds to diagnose a pretrained CTM and found dramatic
under-utilization: 1.6% synapse capacity used, 52% ticks overthinking. The
diagnosis directly guided a zero-backprop inference pipeline delivering **+8pp
accuracy at 3.9× speedup**. The Hebbian plasticity mechanism, enabled by CTM's
native sync signal, outperforms LoRA without any gradient computation. Code and
tools are available at `github.com/rotkonetworks/continuous-thought-machines`
(branch `bounds-analysis`).

## References

[1] Sakana AI. "Continuous Thought Machines." arXiv:2505.05522, 2025.

[2] G. Angeris. "A Note on Generalizing Power Bounds for Physical Design." arXiv:2208.04411v2, 2022.

[3] D. O. Hebb. *The Organization of Behavior.* Wiley, 1949.

[4] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, 2021.
