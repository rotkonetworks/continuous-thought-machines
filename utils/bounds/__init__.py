"""Optimality bound analysis for Continuous Thought Machines.

Adapts the SDP dual framework from Angeris (2022) "Generalizing Power Bounds
for Physical Design" to diagnose CTM training and architecture.

The CTM's recurrent dynamics are a bilinear system:
    h_{t+1} = NLM(synapse([attn(h_t), h_t]))

This maps directly to the physical design problem A(θ)z = b where:
    θ = network weights (design parameters)
    z = stacked activations across all T thinking ticks (fields)
    A(θ) = linearized recurrence

The SDP dual gives LOWER BOUNDS on the best achievable loss, enabling:
    - Per-neuron gap analysis (which NLMs are well-optimized?)
    - Per-tick gap trajectory (where in the thinking process is optimization weakest?)
    - Synapse capacity analysis (is the communication backbone a bottleneck?)
    - Dead neuron detection (which neurons contribute nothing?)

Key theoretical result: CTM's per-neuron NLMs have PRIVATE weights, so the
sufficient condition from §1.2 of the paper holds trivially. The improved
characterization from §3.1 (March 2026 addendum) extends this to the shared
synapse weights without any conditions.

References:
    Angeris, G. (2022). "A Note on Generalizing Power Bounds for Physical
    Design." arXiv:2208.04411v2.

Usage:
    from utils.bounds import analyze_ctm, print_report

    model = ContinuousThoughtMachine.from_pretrained(...)
    results = analyze_ctm(model, input_tensor)
    print_report(results)
"""

from utils.bounds.core import analyze_ctm, print_report
