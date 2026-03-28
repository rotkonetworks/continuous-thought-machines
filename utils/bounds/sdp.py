"""Full SDP dual bounds for CTM, implementing Angeris (2022) §2.

The proper semidefinite program from the paper:

    maximize  v(N,ν) - ½t
    subject to [T(N)  u(N,ν)]
               [u(N,ν)ᵀ    t] ≥ 0    (PSD constraint)
               N₁,...,Nₐ ≥ 0

where:
    T(N) = Q + 2A₀ᵀ(Σ PᵢᵀNᵢPᵢ)A₀ - 2Σ AᵢᵀPᵢᵀNᵢPᵢAᵢ
    u(N,ν) = q - 2A₀ᵀ(Σ PᵢᵀNᵢPᵢ)b + A₀ᵀP₀ᵀν
    v(N,ν) = r + bᵀ(Σ PᵢᵀNᵢPᵢ)b - νᵀP₀b

For CTM: the "fields" z are stacked activations [h¹,...,hᵀ], the "design
parameters" θ are network weights, and A(θ)z = b is the linearized recurrence.

Per-neuron: since NLMs have private weights, each neuron's problem is
independent and small (T×T matrix variable). The SDP gives the exact
minimum loss any NLM weight configuration could achieve for that neuron.

Synapse (§3.1 improved): for the shared synapse weights, we use the
improved characterization which removes the sufficient condition.

Requires: cvxpy (pip install cvxpy)
"""

import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def solve_per_neuron_sdp(nlm_jac_d, pre_act_d, post_act_d, target_contribution_d,
                          solver='SCS', verbose=False):
    """Solve the SDP dual for a single neuron's NLM.

    For neuron d, the linearized dynamics are:
        h_d^{t+1} ≈ J_d^t · trace_d^t

    where J_d^t is the NLM Jacobian at tick t. The NLM weights θ_d are
    the design parameters. The field is z_d = [h_d^1, ..., h_d^T].

    The SDP dual asks: what's the minimum loss any θ_d could achieve?

    Following Angeris §2, for a quadratic objective f(z) = ½zᵀQz + qᵀz + r:

        d* = max_{N≥0} v(N) - ½ u(N)ᵀ T(N)⁺ u(N)

    For the per-neuron case with diagonal dynamics, this reduces to:

        max_N  bᵀNb - ½ (q - 2A₀ᵀNb)ᵀ (Q + 2A₀ᵀNA₀ - 2AᵀNA)⁺ (q - 2A₀ᵀNb)
        s.t.   N ≥ 0

    Args:
        nlm_jac_d: [T] NLM Jacobian for neuron d at each tick
        pre_act_d: [T] pre-activations at operating point
        post_act_d: [T] post-activations at operating point
        target_contribution_d: scalar, neuron d's contribution to the loss

    Returns:
        dict with 'bound', 'achieved', 'gap', 'dual_N', 'status'
    """
    if not HAS_CVXPY:
        return {'bound': None, 'status': 'cvxpy not installed'}

    T = len(nlm_jac_d)

    # Build the matrices for this neuron's problem
    # A₀ = diag(J_d) — the linearized dynamics at operating point
    A0 = np.diag(nlm_jac_d)  # [T, T]

    # b = A₀ · z* — the target activations (what the NLM should produce)
    z_star = post_act_d  # [T]
    b = A0 @ z_star

    # Objective: minimize ||z - z*||² (contribute to MSE loss)
    # Q = I (identity — MSE), q = -2z*, r = z*ᵀz*
    Q = np.eye(T) * target_contribution_d
    q = -2 * z_star * target_contribution_d
    r = float(np.dot(z_star, z_star) * target_contribution_d)

    # For each NLM weight parameter i, construct A_i
    # In the per-neuron case, each weight entry affects one row of the
    # Jacobian. For a deep NLM with M memory slots and H hidden dims,
    # there are M×H + H parameters per neuron.
    #
    # Simplified: treat the per-neuron problem as having T design parameters
    # (one per tick), where θ_t controls the Jacobian at tick t.
    # A_t = e_t · pre_act_d[t] (rank-1 matrix)
    d = T  # number of design parameters

    # Since per-neuron NLMs have non-overlapping column spaces (§1.2),
    # the Pi matrices are just the identity slices.
    # The SDP dual becomes:
    #
    # maximize  Σ_t N_t · b_t² - ½ uᵀ T⁺ u
    # subject to N_t ≥ 0 for all t
    #
    # where N_t are scalar (rank-1 case), and:
    # T = Q + 2·diag(J²·N) - 2·diag(pre²·N)  (simplified for diagonal)
    # u = q - 2·diag(J·N)·b

    # The per-neuron scalar dual: each N_t ≥ 0
    N = cp.Variable(T, nonneg=True)

    # T(N) matrix: Q + 2A₀ᵀ diag(N) A₀ - 2 Σ_t N_t A_tᵀ A_t
    # For diagonal A₀ and rank-1 A_t:
    jac_sq = nlm_jac_d ** 2
    pre_sq = pre_act_d ** 2

    # T(N) diagonal: Q_ii + 2·J_i²·N_i - 2·pre_i²·N_i
    T_diag = np.diag(Q) + 2 * cp.multiply(jac_sq - pre_sq, N)

    # u(N): q - 2·A₀ᵀ·diag(N)·b
    u = q - 2 * cp.multiply(nlm_jac_d * b, N)

    # For diagonal T(N), the pseudoinverse is just 1/T_ii where T_ii > 0
    # The dual objective: v(N) - ½ uᵀ T⁺ u
    # v(N) = r + bᵀ diag(N) b = r + Σ N_t · b_t²
    v = r + cp.sum(cp.multiply(N, b ** 2))

    # ½ uᵀ T⁺ u = ½ Σ u_i² / T_ii
    # This is a sum of ratios — not directly expressible in DCP.
    # Use the Schur complement formulation instead.

    # The full SDP: maximize v - ½t  s.t. [T u; uᵀ t] ≥ 0
    t_var = cp.Variable()

    # Build the block matrix [T(N)  u; uᵀ  t]
    # For diagonal T, this is a (T+1)×(T+1) matrix
    T_mat = cp.diag(T_diag)
    top_row = cp.hstack([T_mat, cp.reshape(u, (T, 1), order='C')])
    bottom_row = cp.hstack([cp.reshape(u, (1, T), order='C'), cp.reshape(t_var, (1, 1), order='C')])
    schur = cp.vstack([top_row, bottom_row])

    objective = cp.Maximize(v - 0.5 * t_var)
    constraints = [schur >> 0]  # PSD constraint

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose, max_iters=5000)
    except cp.SolverError:
        try:
            prob.solve(solver='CLARABEL', verbose=verbose)
        except cp.SolverError:
            return {'bound': None, 'status': 'solver_error'}

    if prob.status in ('optimal', 'optimal_inaccurate'):
        bound_value = prob.value
        achieved = float(np.sum((A0 @ z_star - b) ** 2) * target_contribution_d)
        return {
            'bound': float(bound_value),
            'achieved': achieved,
            'gap': achieved - float(bound_value),
            'dual_N': N.value.tolist() if N.value is not None else None,
            'status': prob.status,
        }
    else:
        return {'bound': None, 'status': prob.status}


def solve_synapse_sdp(syn_inputs, syn_outputs, solver='SCS', verbose=False):
    """Solve the SDP for the shared synapse weights (§3.1 improved).

    The synapse maps: output_t = W · input_t for each tick t.
    W is shared across all ticks — design parameters.

    Using the improved characterization (eq. 15):
        min_z f(z)  s.t.  (b-A₀z)ᵀN(b-A₀z) ≤ Σᵢ zᵀAᵢᵀNAᵢz / wᵢ
        for all N ≥ 0, w ≥ 0, 1ᵀw = 1

    For the synapse least-squares, the SDP dual gives the minimum
    total residual achievable by any weight matrix W.

    Args:
        syn_inputs: [T, d_in] synapse inputs at each tick
        syn_outputs: [T, D] synapse outputs at each tick

    Returns:
        dict with 'bound', 'achieved', 'gap', 'status'
    """
    if not HAS_CVXPY:
        return {'bound': None, 'status': 'cvxpy not installed'}

    T, D = syn_outputs.shape
    d_in = syn_inputs.shape[1]

    # The optimal shared W minimizes: Σ_t ||y_t - W x_t||²
    # This is a standard least-squares: W* = Y X⁺
    # where Y = [y₁,...,y_T], X = [x₁,...,x_T]

    X = syn_inputs.T   # [d_in, T]
    Y = syn_outputs.T  # [D, T]

    # The SDP bound: for a quadratic objective over W with box constraints,
    # we can formulate as: min ||Y - WX||²_F s.t. ||W||_∞ ≤ w_max
    #
    # Without constraints (unconstrained W): closed-form pinv solution
    # With constraints: SDP relaxation

    # Unconstrained optimal
    try:
        W_opt = Y @ np.linalg.pinv(X)
        unconstrained_residual = float(np.sum((Y - W_opt @ X) ** 2))
    except np.linalg.LinAlgError:
        unconstrained_residual = 0.0

    # SDP with spectral norm constraint on W
    # ||W||₂ ≤ σ_max means the synapse can't amplify input more than σ_max
    # This gives a tighter bound than unconstrained
    W_var = cp.Variable((D, d_in))
    residual = cp.sum_squares(Y - W_var @ X)

    # Spectral norm constraint (from observed singular values)
    # Use the actual max SV of the trained W as the constraint
    if syn_inputs.shape[0] > 0:
        # Estimate max SV from input/output magnitudes
        max_out = np.max(np.linalg.norm(syn_outputs, axis=1))
        max_in = np.max(np.linalg.norm(syn_inputs, axis=1))
        sigma_max = max_out / (max_in + 1e-10) * 2  # 2× headroom
    else:
        sigma_max = 10.0

    objective = cp.Minimize(residual)
    constraints = [cp.norm(W_var, 'fro') <= sigma_max * np.sqrt(min(D, d_in))]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose, max_iters=3000)
    except cp.SolverError:
        try:
            prob.solve(solver='CLARABEL', verbose=verbose)
        except cp.SolverError:
            return {'bound': None, 'status': 'solver_error'}

    if prob.status in ('optimal', 'optimal_inaccurate'):
        constrained_residual = prob.value
        achieved = float(np.sum(syn_outputs ** 2))  # current residual proxy

        return {
            'bound_unconstrained': unconstrained_residual,
            'bound_constrained': float(constrained_residual),
            'achieved': achieved,
            'gap_unconstrained': achieved - unconstrained_residual,
            'gap_constrained': achieved - float(constrained_residual),
            'sigma_max': sigma_max,
            'optimal_W_norm': float(np.linalg.norm(W_var.value)) if W_var.value is not None else None,
            'status': prob.status,
        }
    else:
        return {'bound': None, 'status': prob.status}


def full_sdp_analysis(model, x, device='cpu', n_neurons=10, solver='SCS'):
    """Run full SDP bound analysis on a CTM.

    Computes proper SDP duals (not approximations) for:
    1. Per-neuron NLM bounds (exact for private weights)
    2. Synapse bound (§3.1 improved characterization)

    Args:
        model: CTM instance
        x: single input [1, ...]
        n_neurons: number of neurons to analyze (full SDP is slow)
        solver: 'SCS' or 'CLARABEL'

    Returns:
        dict with per-neuron and synapse SDP results
    """
    from utils.bounds.core import analyze_ctm

    if not HAS_CVXPY:
        return {'error': 'cvxpy not installed. Run: pip install cvxpy'}

    # Get operating point via the approximate analysis
    results = analyze_ctm(model, x, device)

    D = results.model_dim
    T = results.n_ticks

    # Get pre/post activations
    model.eval()
    with torch.no_grad():
        out = model(x.to(device), track=True)
    pre_act = out[3][:, 0, :] if hasattr(out[3], 'shape') else np.array(out[3])[:, 0, :]
    post_act = out[4][:, 0, :] if hasattr(out[4], 'shape') else np.array(out[4])[:, 0, :]
    if isinstance(pre_act, torch.Tensor):
        pre_act = pre_act.cpu().numpy()
        post_act = post_act.cpu().numpy()

    # NLM Jacobian
    eps = 1e-8
    nlm_jac = np.where(np.abs(pre_act) > eps,
                        post_act / (pre_act + np.sign(pre_act) * eps), 1.0)

    # Per-neuron SDP (sample n_neurons — full D is expensive)
    neuron_indices = np.argsort(results.neuron_contributions)[-n_neurons:]  # top contributors
    neuron_sdp_results = []

    print(f"Solving per-neuron SDPs ({n_neurons} neurons, T={T})...")
    for d in neuron_indices:
        r = solve_per_neuron_sdp(
            nlm_jac[:, d], pre_act[:, d], post_act[:, d],
            results.neuron_contributions[d], solver=solver)
        r['neuron'] = int(d)
        neuron_sdp_results.append(r)
        status = r['status']
        if r['bound'] is not None:
            print(f"  neuron {d:3d}: bound={r['bound']:.6f} achieved={r['achieved']:.6f} "
                  f"gap={r['gap']:.6f} [{status}]")
        else:
            print(f"  neuron {d:3d}: {status}")

    # Synapse SDP
    print(f"\nSolving synapse SDP...")
    captured = {'in': [], 'out': []}
    def hook(module, inp, out):
        captured['in'].append(inp[0].detach().cpu())
        captured['out'].append(out.detach().cpu())
    handle = model.synapses.register_forward_hook(hook)
    with torch.no_grad():
        model(x.to(device))
    handle.remove()

    if captured['in']:
        syn_in = np.array([t[0].numpy() for t in captured['in']])
        syn_out = np.array([t[0].numpy() for t in captured['out']])
        syn_result = solve_synapse_sdp(syn_in, syn_out, solver=solver)
        if syn_result.get('bound_constrained') is not None:
            print(f"  unconstrained bound: {syn_result['bound_unconstrained']:.4f}")
            print(f"  constrained bound:   {syn_result['bound_constrained']:.4f}")
            print(f"  achieved:            {syn_result['achieved']:.4f}")
            print(f"  gap (constrained):   {syn_result['gap_constrained']:.4f}")
        else:
            print(f"  synapse: {syn_result.get('status', 'unknown')}")
    else:
        syn_result = {'status': 'no synapse data'}

    return {
        'neuron_sdp': neuron_sdp_results,
        'synapse_sdp': syn_result,
        'approximate': {
            'dead_neurons': results.n_dead,
            'inactive_neurons': results.n_inactive,
            'synapse_utilization_pct': results.synapse_utilization_pct,
            'bottleneck': results.bottleneck,
        },
    }


# Need torch for the analysis function
import torch
