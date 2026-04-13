"""SDP dual bounds for CTM using Angeris (2022) formulation.

Uses the actual solver from the paper: given fields z, design
parameters θ with box constraints, and affine constraint A(θ)z = b,
compute the lower bound on the achievable objective.

CTM mapping:
    z = stacked activations [h_d^1, ..., h_d^T] per neuron d
    θ = NLM weights w_d (per-neuron, private)
    A(θ)z = b  ↔  h_d^t - w_d^T @ trace_d^t = synapse_output_d^t

    Expanding: z_d - Trace_d @ θ_d = b_d
    Where Trace_d = [trace_d^1; ...; trace_d^T] is the T×M trace matrix.

    This is affine in θ: A(θ) = I - Σ_d θ_{d,m} * E_{d,m}
    where E_{d,m} selects the m-th trace column for neuron d.

The SDP dual gives the MINIMUM achievable loss for ANY choice of NLM
weights θ_d, subject to box constraints 0 ≤ θ ≤ t_max.

Requires: cvxpy (pip install cvxpy)

Reference:
    Angeris, G. (2022). "A Note on Generalizing Power Bounds for Physical
    Design." arXiv:2208.04411v2.
"""

import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def generate_lower_bounds(z_hat, weights, A, b, t_max, idx_constrained=None,
                          suggested_design_params=True, eps=1e-8,
                          verbose=False, solver="SCS"):
    """Compute SDP lower bound — direct from Angeris (2022).

    Solves the dual of:
        minimize  ½ ||W(z - z_hat)||²
        subject to A(θ)z = b,  0 ≤ θ ≤ t_max

    Args:
        z_hat: [n] operating point (current field values)
        weights: [n] per-element weights W in the objective
        A: [n, m] constraint matrix (fixed part)
        b: [n] right-hand side
        t_max: [n] upper bounds on design parameters
        idx_constrained: list of lists — groups of row indices sharing
                         a design parameter. Default: each row independent.
        suggested_design_params: if True, return suggested θ and z
        eps: numerical tolerance
        verbose: solver verbosity
        solver: CVXPY solver name

    Returns:
        obj_value: lower bound on achievable objective
        nu_value: dual variable values
        init_design: suggested design parameters (if requested)
        init_field: suggested field values (if requested)
    """
    if not HAS_CVXPY:
        raise ImportError("cvxpy required: pip install cvxpy")

    assert A.shape[0] == b.shape[0] == t_max.shape[0]
    assert np.all(t_max >= 0)

    n = A.shape[0]

    if idx_constrained is None:
        idx_constrained = [[i] for i in range(n)]

    nu = cp.Variable(n)
    constraints = []
    dual_obj = -nu @ b + 0.5 * np.linalg.norm(weights * z_hat) ** 2

    ATnu = A.T @ nu

    for S_k in idx_constrained:
        max_left = 0
        max_right = 0
        for j in S_k:
            if abs(weights[j]) <= eps:
                constraints.append(ATnu[j] == 0)
                constraints.append(ATnu[j] + nu[j] * t_max[j] == 0)
            else:
                w2 = weights[j] ** 2
                max_left += cp.square(ATnu[j] - w2 * z_hat[j]) / w2
                if t_max[j] > eps:
                    max_right += cp.square(
                        ATnu[j] + t_max[j] * nu[j] - w2 * z_hat[j]) / w2

        dual_obj += -0.5 * cp.maximum(max_left, max_right)

    prob = cp.Problem(cp.Maximize(dual_obj), constraints)
    obj_value = prob.solve(solver=solver, verbose=verbose, max_iters=10000)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'bound': None, 'status': prob.status}

    result = {'bound': float(obj_value), 'status': prob.status,
              'nu': nu.value}

    if suggested_design_params and nu.value is not None:
        nu_val = nu.value
        Anu = A.T @ nu_val
        max_left = (Anu - (weights ** 2) * z_hat) ** 2
        max_right = (Anu + nu_val * t_max - (weights ** 2) * z_hat) ** 2

        init_design = np.zeros(n)
        for S_k in idx_constrained:
            use_right = np.sum(max_left[S_k]) <= np.sum(max_right[S_k])
            init_design[S_k] = float(use_right) * t_max[S_k]

        pinv_w2 = np.copy(weights)
        small = weights < eps
        pinv_w2[~small] = 1.0 / (weights[~small] ** 2)
        pinv_w2[small] = 0.0

        init_field = z_hat - pinv_w2 * (Anu + init_design * nu_val)

        result['design'] = init_design
        result['field'] = init_field

    return result


def ctm_per_neuron_bound(pre_act, post_act, trace_history,
                         synapse_output, loss_weight=1.0,
                         nlm_weight_max=5.0, solver='SCS', verbose=False):
    """Compute the SDP bound for a single CTM neuron.

    Maps the CTM per-neuron NLM to Angeris' framework:
        z = [h_d^1, ..., h_d^T]  (post-activations)
        θ = NLM weight vector w_d ∈ R^M  (memory_length params)
        Constraint: h_d^t = w_d^T @ trace_d^t  (linearized NLM)
        Objective: minimize ||h_d - h_d*||²  (match operating point)

    The constraint h_d^t - w_d^T @ trace_d^t = 0 has the form:
        For each tick t, memory slot m:
            row (t): coeff of z_t = 1, affected by θ_m with magnitude trace_d^t[m]

    Args:
        pre_act: [T] pre-activations (synapse output for this neuron)
        post_act: [T] post-activations (NLM output for this neuron)
        trace_history: [T, M] trace values at each tick
        synapse_output: [T] synapse contribution (right-hand side)
        loss_weight: scalar weight for this neuron's contribution to loss
        nlm_weight_max: box constraint on NLM weights
        solver: CVXPY solver
        verbose: print solver output

    Returns:
        dict with 'bound', 'achieved', 'gap', 'status', 'optimal_weights'
    """
    if not HAS_CVXPY:
        return {'bound': None, 'status': 'cvxpy not installed'}

    T = len(post_act)
    M = trace_history.shape[1] if trace_history.ndim > 1 else 1

    # z_hat = current post-activations (operating point)
    z_hat = post_act.copy()

    # weights = sqrt(loss_weight) for uniform weighting across ticks
    weights = np.full(T, np.sqrt(loss_weight))

    # A = identity (each tick's activation is a separate field variable)
    A = np.eye(T)

    # b = synapse output (what the NLM input is before NLM processing)
    # Actually b = 0 because constraint is z - Trace @ θ = 0
    # But Angeris' form is A(θ)z = b, and θ modifies A.
    # We need: z_t = Σ_m θ_m * trace_t_m
    # Rewrite: z_t - Σ_m θ_m * trace_t_m = 0
    # In Angeris: (I)z = 0 with design params θ_m adding trace_t_m to row t
    b = np.zeros(T)

    # t_max: for each row t, the design parameter θ affects it through trace.
    # We need to expand: T rows, each affected by M design parameters.
    # But Angeris' code groups rows by design parameter (idx_constrained).
    #
    # Reformulation: expand to T*M rows, where row (t,m) represents
    # the contribution of weight m at tick t.
    # Actually, simpler: for a scalar NLM (M=1), each tick IS one constraint.

    if M == 1:
        # Scalar NLM: θ is a single weight, trace is [T]
        trace_flat = trace_history.flatten()
        t_max_vec = np.full(T, nlm_weight_max * np.abs(trace_flat).max())
        idx_constrained = [list(range(T))]  # all ticks share one design param

        result = generate_lower_bounds(
            z_hat, weights, A, b, t_max_vec,
            idx_constrained=idx_constrained,
            solver=solver, verbose=verbose)
    else:
        # Vector NLM: θ ∈ R^M, each tick affected by all M components.
        # Expand to T rows, M design parameter groups.
        # For Angeris: idx_constrained[m] = all T rows (each weight affects all ticks)
        # t_max[t] = max over m of (nlm_weight_max * |trace_t_m|)

        # The design parameter effect on row t:
        # θ_m adds trace_t_m * θ_m to the diagonal at row t
        # This means t_max[t] = Σ_m nlm_weight_max * |trace_t_m|
        t_max_vec = np.sum(np.abs(trace_history) * nlm_weight_max, axis=1)

        # All rows share the same M design parameters
        idx_constrained = [list(range(T))]

        result = generate_lower_bounds(
            z_hat, weights, A, b, t_max_vec,
            idx_constrained=idx_constrained,
            solver=solver, verbose=verbose)

    if isinstance(result, dict):
        # Compute achieved loss at operating point
        achieved = 0.5 * np.sum((weights * z_hat) ** 2)
        result['achieved'] = float(achieved)
        if result.get('bound') is not None:
            result['gap'] = float(achieved - result['bound'])
        return result

    # Old-style tuple return from generate_lower_bounds
    bound = result[0] if isinstance(result, tuple) else result
    return {'bound': float(bound), 'achieved': 0.5 * np.sum((weights * z_hat) ** 2),
            'status': 'ok'}


def ctm_full_bound(model, x, device='cpu', n_neurons=10, solver='SCS',
                    verbose=False):
    """Run full SDP bound analysis on a CTM.

    For each sampled neuron, computes the Angeris SDP dual to get
    the minimum achievable loss. Compares against the approximate
    diagnostics from core.py.

    Args:
        model: CTM model instance
        x: single input [1, ...]
        device: torch device
        n_neurons: number of neurons to analyze
        solver: CVXPY solver
        verbose: print details

    Returns:
        dict with per-neuron SDP results and comparison to approximate
    """
    import torch
    from utils.bounds.core import analyze_ctm

    if not HAS_CVXPY:
        return {'error': 'cvxpy not installed'}

    # Get approximate diagnostics
    approx = analyze_ctm(model, x, device=device)
    D = approx.model_dim
    T = approx.n_ticks

    # Get activations with tracking
    model.eval()
    model = model.to(device)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.to(device)

    with torch.no_grad():
        out = model(x, track=True)

    pre_act = out[3]  # [T, B, D]
    post_act = out[4]  # [T, B, D]

    if isinstance(pre_act, torch.Tensor):
        pre_act = pre_act.cpu().numpy()
    if isinstance(post_act, torch.Tensor):
        post_act = post_act.cpu().numpy()

    pre_act = pre_act[:, 0, :]   # [T, D]
    post_act = post_act[:, 0, :]  # [T, D]

    # Select top-contributing neurons
    top_neurons = np.argsort(approx.neuron_contributions)[-n_neurons:]

    results = []
    for d in top_neurons:
        pre_d = pre_act[:, d]    # [T]
        post_d = post_act[:, d]  # [T]

        # Trace: the memory history. For a linear NLM, the trace
        # at tick t is the last M pre-activations.
        # We approximate trace as just [pre_d] (M=1 case)
        trace_d = pre_d.reshape(T, 1)

        # Synapse output for this neuron = pre-activation
        syn_d = pre_d

        r = ctm_per_neuron_bound(
            pre_d, post_d, trace_d, syn_d,
            loss_weight=approx.neuron_contributions[d],
            solver=solver, verbose=verbose)
        r['neuron'] = int(d)
        r['approx_contribution'] = float(approx.neuron_contributions[d])
        results.append(r)

        status = r.get('status', '?')
        bound = r.get('bound', None)
        achieved = r.get('achieved', None)
        gap = r.get('gap', None)

        if verbose or True:
            b_str = f'{bound:.6f}' if bound is not None else 'N/A'
            a_str = f'{achieved:.6f}' if achieved is not None else 'N/A'
            g_str = f'{gap:.6f}' if gap is not None else 'N/A'
            print(f'  neuron {d:>4d}: bound={b_str} achieved={a_str} '
                  f'gap={g_str} [{status}]')

    return {
        'neurons': results,
        'approximate': {
            'synapse_utilization_pct': approx.synapse_utilization_pct,
            'n_dead': approx.n_dead,
            'n_overthinking': len(approx.overthinking_ticks),
            'bottleneck': approx.bottleneck,
        },
    }
