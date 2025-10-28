"""
GENERIC Decomposition for Three Binary Variables
=================================================

Function library for computational demonstration of GENERIC structure emergence
in constrained information dynamics for exponential family distributions.

This code accompanies the paper "The Inaccessible Game" and demonstrates
how constraint geometry creates antisymmetric (conservative) flow alongside
symmetric (dissipative) dynamics.

Main Functions:
    Core Analysis:
    analyze_generic_structure() - Compute M = S + A decomposition at a point
    analyze_correlation_structure() - Analyze frustration patterns
        compute_joint_entropy_trajectory() - H(X₁,...,Xₙ) along trajectory
        compute_marginal_entropy_trajectory() - Σᵢ H(Xᵢ) along trajectory
        compute_regime_along_trajectory() - ||A||/||S|| along trajectory
    
    Solvers:
        solve_constrained_maxent() - Gradient descent with constraint
        solve_unconstrained_maxent() - Gradient descent without constraint
    
    Single-Panel Publication Figures (Section 4):
        save_constraint_maintenance() - |Σh_i(t) - C| vs time
        save_convergence() - ||F(θ)|| vs time
        save_trajectory_comparison() - Parameter space trajectories
        save_joint_entropy_evolution() - H(t) for both dynamics
        save_marginal_entropy_evolution() - Σh_i(t) for both
        save_flow_comparison() - ||F|| comparison
        save_distance_evolution() - ||θ|| comparison
        save_marginal_parameters() - θ₁,θ₂,θ₃ evolution
        save_interaction_parameters() - θ₁₂,θ₁₃,θ₂₃ evolution
        save_regime_variation() - ||A||/||S|| vs time
        save_component_norms() - ||S|| and ||A|| separately
        save_correlation_structure() - Correlation matrix
        save_decomposition_table() - LaTeX table
    
    Multi-Panel Exploration Figures:
        plot_phase_space_decomposition() - Visualize S, A, M dynamics
        plot_correlation_analysis() - Full correlation analysis
        plot_constrained_trajectory() - Trajectory visualization
        plot_constrained_vs_unconstrained() - Full comparison (9 panels)
        temperature_scaling_experiment() - Physics validation

Usage:
    >>> import generic_decomposition_n3 as gd
    
    # Run constrained dynamics
    >>> sol_c = gd.solve_constrained_maxent(theta_init, N=3, verbose=True)
    >>> sol_u = gd.solve_unconstrained_maxent(theta_init, N=3)
    
    # Generate all Section 4 figures
    >>> gd.save_constraint_maintenance(sol_c)
    >>> gd.save_trajectory_comparison(sol_c, sol_u)
    >>> gd.save_regime_variation(sol_c, N=3)
    
    # Analyze at specific point
    >>> result = gd.analyze_generic_structure(theta, N=3)
    >>> gd.save_decomposition_table(result)

Dependencies:
    numpy, scipy, matplotlib

Author: Neil D. Lawrence
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
from itertools import product

# Publication-quality figure settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})


# ============================================================================
# Exponential Family Functions
# ============================================================================

def generate_states(N):
    """Generate all 2^N binary states."""
    return np.array(list(product([0, 1], repeat=N)))


def compute_features_pairwise(states):
    """
    Compute sufficient statistics for pairwise exponential family.
    
    For N binary variables: [x_1, x_2, ..., x_N, x_1*x_2, x_1*x_3, ...]
    """
    N = states.shape[1]
    features = [states]  # Linear (marginal) features
    
    # Pairwise interaction features
    for i in range(N):
        for j in range(i+1, N):
            features.append((states[:, i] * states[:, j]).reshape(-1, 1))
    
    return np.hstack(features)


def compute_marginals(theta, N):
    """Compute marginal probabilities p(x_i) for each variable."""
    states = generate_states(N)
    features = compute_features_pairwise(states)
    
    # Joint distribution: p(x) ∝ exp(θ^T φ(x))
    logits = features @ theta
    log_Z = np.logaddexp.reduce(logits)
    probs = np.exp(logits - log_Z)
    
    # Marginalize
    marginals = []
    for i in range(N):
        p_i = np.array([probs[states[:, i] == 0].sum(), 
                       probs[states[:, i] == 1].sum()])
        marginals.append(p_i)
    
    return marginals, probs


def compute_fisher(theta, N):
    """Compute Fisher information matrix G(θ) = Cov[φ(X)]."""
    states = generate_states(N)
    features = compute_features_pairwise(states)
    
    logits = features @ theta
    log_Z = np.logaddexp.reduce(logits)
    probs = np.exp(logits - log_Z)
    
    # Fisher = E[φφ^T] - E[φ]E[φ]^T
    mean_phi = probs @ features
    cov_matrix = sum(p * np.outer(phi, phi) for p, phi in zip(probs, features))
    
    return cov_matrix - np.outer(mean_phi, mean_phi)


def marginal_entropy(p):
    """Shannon entropy H(X) = -∑ p(x) log p(x)."""
    p_clean = p[p > 1e-10]
    return -np.sum(p_clean * np.log(p_clean))


def compute_constraint_gradient(theta, N, eps=1e-5):
    """
    Numerical gradient of total marginal entropy: ∇(∑_i H(X_i)).
    
    This is the constraint gradient a(θ) = ∇_θ C(θ).
    """
    d = N + N*(N-1)//2  # Parameter dimension
    marginals, _ = compute_marginals(theta, N)
    h_base = sum(marginal_entropy(m) for m in marginals)
    
    grad = np.zeros(d)
    for i in range(d):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        marginals_plus, _ = compute_marginals(theta_plus, N)
        h_plus = sum(marginal_entropy(m) for m in marginals_plus)
        grad[i] = (h_plus - h_base) / eps
    
    return grad


# ============================================================================
# GENERIC Decomposition Analysis
# ============================================================================

def analyze_generic_structure(theta, N, eps_diff=1e-5):
    """
    Perform GENERIC decomposition of constrained information dynamics.
    
    Dynamics: dθ/dt = -G(θ)θ - ν(θ)a(θ)
    where ν enforces tangency to constraint manifold.
    
    Returns decomposition M = S + A where:
    - S = (M + M^T)/2 is symmetric (dissipative)
    - A = (M - M^T)/2 is antisymmetric (conservative)
    """
    d = N + N*(N-1)//2
    
    # System properties at θ
    G = compute_fisher(theta, N)
    a = compute_constraint_gradient(theta, N)
    
    # Lagrange multiplier from tangency condition
    F_unc = -G @ theta  # Unconstrained flow
    # Tangency: a^T F = 0 => ν = (a^T F_unc)/(a^T a)
    nu = np.dot(F_unc, a) / np.dot(a, a)
    F = F_unc - nu * a  # Constrained flow
    
    # Linearization: M = ∂F/∂θ (numerical Jacobian)
    M = np.zeros((d, d))
    for i in range(d):
        theta_plus = theta.copy()
        theta_plus[i] += eps_diff
        
        G_plus = compute_fisher(theta_plus, N)
        a_plus = compute_constraint_gradient(theta_plus, N)
        F_unc_plus = -G_plus @ theta_plus
        nu_plus = np.dot(F_unc_plus, a_plus) / np.dot(a_plus, a_plus)
        F_plus = F_unc_plus - nu_plus * a_plus
        
        M[:, i] = (F_plus - F) / eps_diff
    
    # GENERIC decomposition
    S = (M + M.T) / 2  # Symmetric part
    A = (M - M.T) / 2  # Antisymmetric part
    
    # Norms and eigenvalues
    norm_S = np.linalg.norm(S, 'fro')
    norm_A = np.linalg.norm(A, 'fro')
    ratio = norm_A / norm_S if norm_S > 1e-10 else 0.0
    
    eigs_S = np.linalg.eigvals(S)
    eigs_A = np.linalg.eigvals(A)
    
    return {
        'theta': theta,
        'N': N,
        'd': d,
        'F': F,
        'M': M,
        'S': S,
        'A': A,
        'norm_S': norm_S,
        'norm_A': norm_A,
        'ratio': ratio,
        'eigs_S': eigs_S,
        'eigs_A': eigs_A,
        'nu': nu,
        'G': G,
        'a': a,
    }


def solve_constrained_maxent(theta_init, N, n_steps=2000, dt=0.01, 
                              convergence_tol=1e-6, verbose=False):
    """
    Solve constrained max ent dynamics via gradient descent.
    
    Dynamics: dθ/dt = F(θ) = -G(θ)θ - ν(θ)a(θ)
    
    This performs gradient ascent on joint entropy H(X₁,...,Xₙ) subject to
    the constraint Σᵢ H(Xᵢ) = C (enforced via Lagrange multiplier ν).
    
    Parameters
    ----------
    theta_init : array
        Initial parameter values
    N : int
        Number of binary variables
    n_steps : int
        Maximum number of gradient steps
    dt : float
        Step size for gradient descent
    convergence_tol : float
        Stop when ||F|| < convergence_tol
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict with keys:
        trajectory : array of shape (n_actual_steps, d)
            Parameter trajectory θ(t)
        flow_norms : array
            ||F(θ)|| at each step
        constraint_values : array
            Σᵢ H(Xᵢ) at each step (should be approximately constant)
        converged : bool
            Whether convergence criterion was met
    """
    d = N + N*(N-1)//2
    trajectory = [theta_init.copy()]
    flow_norms = []
    constraint_values = []
    theta = theta_init.copy()
    
    # Initial constraint value
    marginals, _ = compute_marginals(theta, N)
    C_init = sum(marginal_entropy(m) for m in marginals)
    
    for step in range(n_steps):
        # Compute constrained flow at current point
        G = compute_fisher(theta, N)
        a = compute_constraint_gradient(theta, N)
        F_unc = -G @ theta  # Unconstrained: maximize entropy
        
        # Lagrange multiplier to enforce constraint tangency
        # Tangency: a^T F = 0 => a^T(F_unc - ν*a) = 0 => ν = (a^T F_unc)/(a^T a)
        nu = np.dot(F_unc, a) / np.dot(a, a)
        F = F_unc - nu * a  # Constrained flow
        
        # Gradient descent step
        theta = theta + dt * F
        
        # Track metrics
        flow_norm = np.linalg.norm(F)
        flow_norms.append(flow_norm)
        trajectory.append(theta.copy())
        
        # Check constraint preservation
        marginals, _ = compute_marginals(theta, N)
        C_current = sum(marginal_entropy(m) for m in marginals)
        constraint_values.append(C_current)
        
        # Verbose output
        if verbose and step % 100 == 0:
            print(f"Step {step:4d}: ||F|| = {flow_norm:.6f}, "
                  f"ΔC = {abs(C_current - C_init):.8f}")
        
        # Check convergence
        if flow_norm < convergence_tol:
            if verbose:
                print(f"\nConverged at step {step}")
            return {
                'trajectory': np.array(trajectory),
                'flow_norms': np.array(flow_norms),
                'constraint_values': np.array(constraint_values),
                'C_init': C_init,
                'converged': True,
                'n_steps': step + 1
            }
    
    if verbose:
        print(f"\nReached maximum steps ({n_steps})")
    
    return {
        'trajectory': np.array(trajectory),
        'flow_norms': np.array(flow_norms),
        'constraint_values': np.array(constraint_values),
        'C_init': C_init,
        'converged': False,
        'n_steps': n_steps
    }


# ============================================================================
# Correlation Structure Analysis
# ============================================================================

def analyze_correlation_structure(theta, N):
    """
    Analyze the correlation structure that creates strong antisymmetric flow.
    
    Returns:
    - Correlation matrix
    - Marginal probabilities
    - Interaction strengths
    """
    marginals, probs = compute_marginals(theta, N)
    states = generate_states(N)
    
    # Compute correlation matrix
    means = np.array([m[1] for m in marginals])  # P(X_i = 1)
    
    # Cov(X_i, X_j) for binary variables
    corr_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                corr_matrix[i, j] = means[i] * (1 - means[i])  # Variance
            else:
                # E[X_i X_j] - E[X_i]E[X_j]
                joint_prob = probs[np.logical_and(states[:, i] == 1, 
                                                  states[:, j] == 1)].sum()
                corr_matrix[i, j] = joint_prob - means[i] * means[j]
    
    # Normalize to correlation coefficients
    std_devs = np.sqrt(np.diag(corr_matrix))
    corr_coef = corr_matrix / np.outer(std_devs, std_devs)
    
    return {
        'correlation_matrix': corr_coef,
        'marginal_probs': means,
        'covariance': corr_matrix,
        'theta_marginal': theta[:N],
        'theta_pairwise': theta[N:],
    }


# ============================================================================
# Analysis Helper Functions (for trajectory-based analysis)
# ============================================================================

def compute_joint_entropy_trajectory(trajectory, N, sample_every=10):
    """
    Compute joint entropy H(X₁,...,Xₙ) along trajectory.
    
    Parameters
    ----------
    trajectory : array, shape (n_steps, d)
        Parameter trajectory θ(t)
    N : int
        Number of binary variables
    sample_every : int
        Sample every nth point for efficiency
        
    Returns
    -------
    sample_indices : array
        Step numbers where entropy was computed
    joint_entropy : array
        H(X₁,...,Xₙ) at each sampled point
    """
    states = generate_states(N)
    features = compute_features_pairwise(states)
    
    sample_indices = np.arange(0, len(trajectory), sample_every)
    joint_entropy = []
    
    for idx in sample_indices:
        theta = trajectory[idx]
        logits = features @ theta
        log_Z = np.logaddexp.reduce(logits)
        probs = np.exp(logits - log_Z)
        
        # H = -Σ p(x) log p(x)
        p_clean = probs[probs > 1e-10]
        H = -np.sum(p_clean * np.log(p_clean))
        joint_entropy.append(H)
    
    return sample_indices, np.array(joint_entropy)


def compute_marginal_entropy_trajectory(trajectory, N, sample_every=10):
    """
    Compute marginal entropy sum Σᵢ H(Xᵢ) along trajectory.
    
    Parameters
    ----------
    trajectory : array, shape (n_steps, d)
        Parameter trajectory θ(t)
    N : int
        Number of binary variables
    sample_every : int
        Sample every nth point for efficiency
        
    Returns
    -------
    sample_indices : array
        Step numbers where entropy was computed
    marginal_entropy_sum : array
        Σᵢ H(Xᵢ) at each sampled point
    """
    sample_indices = np.arange(0, len(trajectory), sample_every)
    marginal_entropy_sum = []
    
    for idx in sample_indices:
        theta = trajectory[idx]
        marginals, _ = compute_marginals(theta, N)
        sum_h = sum(marginal_entropy(m) for m in marginals)
        marginal_entropy_sum.append(sum_h)
    
    return sample_indices, np.array(marginal_entropy_sum)


def compute_regime_along_trajectory(trajectory, N, sample_every=50):
    """
    Compute ||A||/||S|| ratio at sampled points along trajectory.
    
    This shows how the balance between conservative and dissipative dynamics
    varies as the system evolves on the constraint manifold.
    
    Parameters
    ----------
    trajectory : array, shape (n_steps, d)
        Parameter trajectory θ(t)
    N : int
        Number of binary variables
    sample_every : int
        Sample every nth point (GENERIC analysis is expensive)
        
    Returns
    -------
    dict with keys:
        sample_indices : array
            Step numbers where decomposition was computed
        ratios : array
            ||A||/||S|| at each point
        norms_S : array
            ||S|| at each point
        norms_A : array
            ||A|| at each point
    """
    sample_indices = np.arange(0, len(trajectory), sample_every)
    ratios = []
    norms_S = []
    norms_A = []
    
    for idx in sample_indices:
        theta = trajectory[idx]
        try:
            result = analyze_generic_structure(theta, N)
            ratios.append(result['ratio'])
            norms_S.append(result['norm_S'])
            norms_A.append(result['norm_A'])
        except:
            # If analysis fails, append NaN
            ratios.append(np.nan)
            norms_S.append(np.nan)
            norms_A.append(np.nan)
    
    return {
        'sample_indices': sample_indices,
        'ratios': np.array(ratios),
        'norms_S': np.array(norms_S),
        'norms_A': np.array(norms_A)
    }


# ============================================================================
# Single-Panel Publication Figures (Section 4 of paper)
# ============================================================================

def save_constraint_maintenance(solution, filename='fig_constraint_maintenance.pdf'):
    """
    Plot |Σh_i(t) - C| vs time (log scale).
    
    Shows that marginal entropy constraint is maintained during evolution.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    steps = np.arange(len(solution['constraint_values']))
    deviation = np.abs(solution['constraint_values'] - solution['C_init'])
    
    ax.semilogy(steps, deviation, 'b-', linewidth=1.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$|\sum_i h_i(t) - C|$')
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(1e-8, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_convergence(solution, filename='fig_convergence.pdf'):
    """
    Plot ||F(θ)|| vs time (log scale).
    
    Shows convergence to stationary point where constrained flow vanishes.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    steps = np.arange(len(solution['flow_norms']))
    ax.semilogy(steps, solution['flow_norms'], 'b-', linewidth=1.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$\|F(\theta)\|$')
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(1e-6, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_trajectory_comparison(sol_constrained, sol_unconstrained, 
                                filename='fig_trajectory_comparison.pdf'):
    """
    Plot θ₁ vs θ₂ projection with both trajectories.
    
    Shows how constraint shapes the flow in parameter space.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    traj_c = sol_constrained['trajectory']
    traj_u = sol_unconstrained['trajectory']
    
    # Plot trajectories
    ax.plot(traj_u[:, 0], traj_u[:, 1], 'r-', alpha=0.6, linewidth=2, 
            label='Unconstrained')
    ax.plot(traj_c[:, 0], traj_c[:, 1], 'b-', alpha=0.7, linewidth=2,
            label='Constrained')
    
    # Mark initial and final points
    ax.plot(traj_c[0, 0], traj_c[0, 1], 'go', markersize=8, 
            label='Initial', zorder=5)
    ax.plot(traj_u[-1, 0], traj_u[-1, 1], 'r*', markersize=12, zorder=5)
    ax.plot(traj_c[-1, 0], traj_c[-1, 1], 'bs', markersize=8, zorder=5)
    
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_joint_entropy_evolution(sol_constrained, sol_unconstrained, N,
                                  filename='fig_joint_entropy_evolution.pdf'):
    """
    Plot H(X₁,X₂,X₃) vs time for both dynamics.
    
    Shows that both increase joint entropy (second law), but at different rates.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Compute joint entropy for both trajectories
    steps_u, H_u = compute_joint_entropy_trajectory(sol_unconstrained['trajectory'], N, sample_every=10)
    steps_c, H_c = compute_joint_entropy_trajectory(sol_constrained['trajectory'], N, sample_every=10)
    
    ax.plot(steps_u, H_u, 'r-', linewidth=2, label='Unconstrained')
    ax.plot(steps_c, H_c, 'b-', linewidth=2, label='Constrained')
    ax.axhline(N*np.log(2), color='k', linestyle='--', linewidth=1, 
               alpha=0.5, label='Maximum')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$H(X_1,X_2,X_3)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_marginal_entropy_evolution(sol_constrained, sol_unconstrained, N,
                                     filename='fig_marginal_entropy_evolution.pdf'):
    """
    Plot Σh_i vs time for both dynamics.
    
    Shows constrained maintains constant sum, unconstrained does not.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Compute marginal entropy sum for both
    steps_u, sum_h_u = compute_marginal_entropy_trajectory(sol_unconstrained['trajectory'], N, sample_every=10)
    steps_c, sum_h_c = compute_marginal_entropy_trajectory(sol_constrained['trajectory'], N, sample_every=10)
    
    ax.plot(steps_u, sum_h_u, 'r-', linewidth=2, label='Unconstrained')
    ax.plot(steps_c, sum_h_c, 'b-', linewidth=2, label='Constrained')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$\sum_i h_i$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_flow_comparison(sol_constrained, sol_unconstrained,
                         filename='fig_flow_comparison.pdf'):
    """
    Plot ||F(θ)|| vs time for both dynamics.
    
    Shows different convergence rates due to constraint.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    steps_u = np.arange(len(sol_unconstrained['flow_norms']))
    steps_c = np.arange(len(sol_constrained['flow_norms']))
    
    ax.semilogy(steps_u, sol_unconstrained['flow_norms'], 'r-', 
                linewidth=2, label='Unconstrained')
    ax.semilogy(steps_c, sol_constrained['flow_norms'], 'b-', 
                linewidth=2, label='Constrained')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$\|F(\theta)\|$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_distance_evolution(sol_constrained, sol_unconstrained,
                            filename='fig_distance_evolution.pdf'):
    """
    Plot ||θ|| vs time for both dynamics.
    
    Shows convergence to origin (unconstrained) vs convergence on manifold (constrained).
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    traj_u = sol_unconstrained['trajectory']
    traj_c = sol_constrained['trajectory']
    
    dist_u = np.linalg.norm(traj_u, axis=1)
    dist_c = np.linalg.norm(traj_c, axis=1)
    
    steps_u = np.arange(len(dist_u))
    steps_c = np.arange(len(dist_c))
    
    ax.semilogy(steps_u, dist_u, 'r-', linewidth=2, label='Unconstrained')
    ax.semilogy(steps_c, dist_c, 'b-', linewidth=2, label='Constrained')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$\|\theta\|$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_marginal_parameters(sol_constrained, sol_unconstrained,
                             filename='fig_marginal_parameters.pdf'):
    """
    Plot θ₁, θ₂, θ₃ vs time for both dynamics.
    
    Shows how marginal bias parameters evolve differently under constraint.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    traj_u = sol_unconstrained['trajectory']
    traj_c = sol_constrained['trajectory']
    
    steps_u = np.arange(len(traj_u))
    steps_c = np.arange(len(traj_c))
    
    labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
    colors = ['C0', 'C1', 'C2']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(steps_u, traj_u[:, i], '--', alpha=0.5, linewidth=1.5, 
                color=color, label=f'{label} (unc.)')
        ax.plot(steps_c, traj_c[:, i], '-', linewidth=2, 
                color=color, label=f'{label} (con.)')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Parameter value')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_interaction_parameters(sol_constrained, sol_unconstrained,
                                filename='fig_interaction_parameters.pdf'):
    """
    Plot θ₁₂, θ₁₃, θ₂₃ vs time for both dynamics.
    
    Shows how pairwise interactions evolve differently under constraint.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    traj_u = sol_unconstrained['trajectory']
    traj_c = sol_constrained['trajectory']
    
    steps_u = np.arange(len(traj_u))
    steps_c = np.arange(len(traj_c))
    
    labels = [r'$\theta_{12}$', r'$\theta_{13}$', r'$\theta_{23}$']
    colors = ['C0', 'C1', 'C2']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        param_idx = 3 + i  # Interaction parameters start at index 3
        ax.plot(steps_u, traj_u[:, param_idx], '--', alpha=0.5, linewidth=1.5,
                color=color, label=f'{label} (unc.)')
        ax.plot(steps_c, traj_c[:, param_idx], '-', linewidth=2,
                color=color, label=f'{label} (con.)')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Parameter value')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_regime_variation(solution, N, sample_every=50,
                          filename='fig_regime_variation.pdf'):
    """
    Plot ||A||/||S|| along trajectory.
    
    Shows transition between thermodynamic and mechanical regimes.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Compute regime variation
    regime_data = compute_regime_along_trajectory(solution['trajectory'], N, sample_every)
    
    ax.plot(regime_data['sample_indices'], regime_data['ratios'], 
            'g-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$\|A\|/\|S\|$')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_component_norms(solution, N, sample_every=50,
                         filename='fig_component_norms.pdf'):
    """
    Plot ||S|| and ||A|| separately vs time.
    
    Shows absolute magnitudes, not just ratio.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Compute regime variation
    regime_data = compute_regime_along_trajectory(solution['trajectory'], N, sample_every)
    
    ax.plot(regime_data['sample_indices'], regime_data['norms_S'], 
            'r-', linewidth=2, label=r'$\|S\|$ (dissipative)')
    ax.plot(regime_data['sample_indices'], regime_data['norms_A'], 
            'b-', linewidth=2, label=r'$\|A\|$ (conservative)')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Frobenius norm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_correlation_structure(theta, N, 
                               filename='fig_correlation_structure.pdf'):
    """
    Plot correlation matrix and frustration pattern.
    
    Single-panel version showing just the correlation heatmap with values.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Compute correlation structure
    corr_info = analyze_correlation_structure(theta, N)
    corr_matrix = corr_info['correlation_matrix']
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'])
    ax.set_yticklabels([r'$X_1$', r'$X_2$', r'$X_3$'])
    
    # Add correlation values
    for i in range(N):
        for j in range(N):
            if i != j:
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


def save_decomposition_table(result, filename='table_decomposition_table.txt'):
    """
    Generate LaTeX table code for decomposition results.
    
    Creates a simple table showing ||S||, ||A||, and ratio.
    """
    table_code = r"""\begin{center}
\begin{tabular}{lcc}
\toprule
Component & Frobenius Norm & Eigenvalue Structure \\
\midrule
Symmetric ($S$) & $\|S\| = %.3f$ & Real, negative (dissipation) \\
Antisymmetric ($A$) & $\|A\| = %.3f$ & Pure imaginary (oscillation) \\
\textbf{Ratio} & $\|A\|/\|S\| = %.2f$ & \textbf{%s} \\
\bottomrule
\end{tabular}
\end{center}
""" % (result['norm_S'], result['norm_A'], result['ratio'],
       'Conservation-dominated' if result['ratio'] > 1 else 'Dissipation-dominated')
    
    with open(filename, 'w') as f:
        f.write(table_code)
    
    print(f"✓ Saved: {filename}")


# ============================================================================
# Multi-Panel Visualization Functions (Original - for exploration)
# ============================================================================

def plot_phase_space_decomposition(result, filename='generic_n3_phase_space.pdf'):
    """Create phase space visualization showing S, A, and M dynamics."""
    
    M, S, A = result['M'], result['S'], result['A']
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    t_span = np.linspace(0, 15, 200)
    
    # Initial conditions (perturbations in first 2 dims)
    q0_list = [
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.07, 0.07, 0.0, 0.0, 0.0, 0.0])
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    dynamics = [
        (r"Pure $S$ (dissipation)", S),
        (r"Pure $A$ (rotation)", A),
        (r"$M = S + A$ (combined)", M)
    ]
    
    for ax, (title, matrix) in zip(axes, dynamics):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'$q_1$')
        ax.set_ylabel(r'$q_2$')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        
        for q0, color in zip(q0_list, colors):
            trajectory = np.array([expm(matrix * t) @ q0 for t in t_span])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, alpha=0.7, linewidth=1.5)
            ax.plot(q0[0], q0[1], 'o', color=color, markersize=5)
            
            # Direction arrow
            if len(trajectory) > 20:
                mid = len(trajectory) // 3
                dx = trajectory[mid+3, 0] - trajectory[mid, 0]
                dy = trajectory[mid+3, 1] - trajectory[mid, 1]
                if np.sqrt(dx**2 + dy**2) > 1e-4:
                    ax.arrow(trajectory[mid, 0], trajectory[mid, 1], 
                           dx*0.7, dy*0.7,
                           head_width=0.008, head_length=0.008, 
                           fc=color, ec=color, alpha=0.7, linewidth=1)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")


def plot_correlation_analysis(theta, N, corr_info, 
                              filename='generic_n3_correlation_structure.pdf'):
    """
    Visualize why this parameter pattern creates strong antisymmetric flow.
    
    Shows:
    1. Parameter values (marginals and pairwise interactions)
    2. Correlation matrix
    3. Marginal distributions
    """
    
    fig = plt.figure(figsize=(10, 3.5))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    # 1. Parameter values
    ax1 = fig.add_subplot(gs[0, 0])
    param_labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', 
                   r'$\theta_{12}$', r'$\theta_{13}$', r'$\theta_{23}$']
    colors = ['blue']*3 + ['red']*3
    bars = ax1.bar(range(6), theta, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(6))
    ax1.set_xticklabels(param_labels)
    ax1.set_ylabel('Parameter value')
    ax1.set_title('Parameter Pattern')
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight the frustrated structure
    ax1.text(0.5, -0.5, 'Marginals', ha='center', fontsize=8, color='blue')
    ax1.text(4, -0.5, 'Pairwise\nInteractions', ha='center', fontsize=8, color='red')
    
    # 2. Correlation matrix (heatmap)
    ax2 = fig.add_subplot(gs[0, 1])
    corr_matrix = corr_info['correlation_matrix']
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'])
    ax2.set_yticklabels([r'$X_1$', r'$X_2$', r'$X_3$'])
    ax2.set_title('Correlation Matrix')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            if i != j:
                text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Marginal probabilities
    ax3 = fig.add_subplot(gs[0, 2])
    marginal_probs = corr_info['marginal_probs']
    x_pos = np.array([0, 1, 2])
    width = 0.35
    
    ax3.bar(x_pos - width/2, 1 - marginal_probs, width, 
           label=r'$P(X_i=0)$', color='lightblue', edgecolor='black')
    ax3.bar(x_pos + width/2, marginal_probs, width, 
           label=r'$P(X_i=1)$', color='darkblue', edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'])
    ax3.set_ylabel('Probability')
    ax3.set_title('Marginal Distributions')
    ax3.legend(fontsize=8)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Geometric frustration explanation (text panel spanning bottom)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Analysis text
    text = (
        r"$\mathbf{Why\ This\ Pattern\ Creates\ Strong\ Antisymmetric\ Flow:}$" + "\n\n"
        
        r"$\bullet$ $\theta_{12} \approx -\theta_{13}$: Variable $X_1$ has strong but $\mathit{opposite}$ " +
        "pairwise interactions with $X_2$ and $X_3$\n\n"
        
        r"$\bullet$ $\mathbf{Geometric\ frustration}$: The constraint surface " +
        r"$\sum_i H(X_i) = C$ curves sharply because $X_1$ " +
        "cannot simultaneously\n" +
        "      satisfy both interaction preferences — it creates a \"twisted\" geometry in parameter space\n\n"
        
        r"$\bullet$ $\mathbf{Antisymmetric\ emergence}$: Moving along the constraint manifold induces " +
        "geometric phases (Berry-like effects)\n" +
        "      from parallel transport on this curved surface, manifesting as the antisymmetric component $A$\n\n"
        
        r"$\bullet$ $\mathbf{Result}$: " +
        rf"$\|A\|/\|S\| = {corr_matrix[0,1]:.2f}$ means conservative (rotational) dynamics dominate " +
        "dissipative (relaxation) dynamics\n" +
        "      at this point — the system exhibits more Hamiltonian-like than thermodynamic behavior"
    )
    
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, 
            fontsize=9, verticalalignment='top', family='serif',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")


def plot_constrained_trajectory(solution, theta_final, 
                                filename='generic_n3_constrained_trajectory.pdf'):
    """
    Visualize the constrained max ent dynamics trajectory.
    
    Shows:
    1. Parameter trajectory in 2D projection
    2. Flow norm convergence
    3. Constraint preservation
    """
    
    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.35)
    
    trajectory = solution['trajectory']
    flow_norms = solution['flow_norms']
    constraint_values = solution['constraint_values']
    C_init = solution['C_init']
    n_steps = solution['n_steps']
    
    # 1. Parameter trajectory (project to first 2 dimensions)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=1.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, 
            label='Initial', zorder=5)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=12, 
            label='Final', zorder=5)
    
    # Add arrows showing direction
    n_arrows = 5
    arrow_indices = np.linspace(10, len(trajectory)-10, n_arrows, dtype=int)
    for idx in arrow_indices:
        dx = trajectory[idx+5, 0] - trajectory[idx, 0]
        dy = trajectory[idx+5, 1] - trajectory[idx, 1]
        ax1.arrow(trajectory[idx, 0], trajectory[idx, 1], dx*0.5, dy*0.5,
                 head_width=0.02, head_length=0.02, fc='blue', ec='blue', 
                 alpha=0.4, linewidth=0.8)
    
    ax1.set_xlabel(r'$\theta_1$ (marginal bias $X_1$)')
    ax1.set_ylabel(r'$\theta_2$ (marginal bias $X_2$)')
    ax1.set_title('Trajectory on Constraint Manifold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Flow norm convergence
    ax2 = fig.add_subplot(gs[0, 1])
    steps = np.arange(len(flow_norms))
    ax2.semilogy(steps, flow_norms, 'b-', linewidth=1.5)
    ax2.set_xlabel('Gradient step')
    ax2.set_ylabel(r'$\|F(\theta)\|$ (flow magnitude)')
    ax2.set_title('Convergence to Stationary Point')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axhline(1e-6, color='r', linestyle='--', linewidth=1, 
               alpha=0.5, label='Convergence threshold')
    ax2.legend(fontsize=8)
    
    # 3. Constraint preservation
    ax3 = fig.add_subplot(gs[0, 2])
    steps = np.arange(len(constraint_values))
    constraint_deviation = np.abs(np.array(constraint_values) - C_init)
    ax3.semilogy(steps, constraint_deviation, 'g-', linewidth=1.5)
    ax3.set_xlabel('Gradient step')
    ax3.set_ylabel(r'$|\sum_i H(X_i) - C|$ (constraint violation)')
    ax3.set_title('Constraint Preservation')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Add text annotation
    final_violation = constraint_deviation[-1]
    ax3.text(0.5, 0.95, f'Final violation: {final_violation:.2e}', 
            transform=ax3.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")


def solve_unconstrained_maxent(theta_init, N, n_steps=2000, dt=0.01,
                                convergence_tol=1e-6, verbose=False):
    """
    Solve pure (unconstrained) max ent dynamics via gradient ascent.
    
    Dynamics: dθ/dt = -G(θ)θ
    
    This performs gradient ascent on joint entropy H(X₁,...,Xₙ) WITHOUT
    any constraint. The system converges to θ=0 (uniform distribution).
    
    Parameters
    ----------
    theta_init : array
        Initial parameter values
    N : int
        Number of binary variables
    n_steps : int
        Maximum number of gradient steps
    dt : float
        Step size for gradient descent
    convergence_tol : float
        Stop when ||F|| < convergence_tol
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict with keys:
        trajectory : array of shape (n_actual_steps, d)
            Parameter trajectory θ(t)
        flow_norms : array
            ||F(θ)|| at each step
        converged : bool
            Whether convergence criterion was met
    """
    d = N + N*(N-1)//2
    trajectory = [theta_init.copy()]
    flow_norms = []
    theta = theta_init.copy()
    
    for step in range(n_steps):
        # Unconstrained flow: maximize joint entropy
        G = compute_fisher(theta, N)
        F = -G @ theta  # Pure gradient ascent
        
        # Gradient descent step
        theta = theta + dt * F
        
        # Track metrics
        flow_norm = np.linalg.norm(F)
        flow_norms.append(flow_norm)
        trajectory.append(theta.copy())
        
        # Verbose output
        if verbose and step % 100 == 0:
            print(f"Step {step:4d}: ||F|| = {flow_norm:.6f}")
        
        # Check convergence
        if flow_norm < convergence_tol:
            if verbose:
                print(f"\nConverged at step {step}")
            return {
                'trajectory': np.array(trajectory),
                'flow_norms': np.array(flow_norms),
                'converged': True,
                'n_steps': step + 1
            }
    
    if verbose:
        print(f"\nReached maximum steps ({n_steps})")
    
    return {
        'trajectory': np.array(trajectory),
        'flow_norms': np.array(flow_norms),
        'converged': False,
        'n_steps': n_steps
    }


def plot_constrained_vs_unconstrained(solution_constrained, solution_unconstrained,
                                      filename='generic_n3_constrained_vs_unconstrained.pdf'):
    """
    Compare constrained vs unconstrained max ent dynamics.
    
    Shows how the marginal entropy constraint affects the convergence path.
    """
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    traj_c = solution_constrained['trajectory']
    traj_u = solution_unconstrained['trajectory']
    flow_c = solution_constrained['flow_norms']
    flow_u = solution_unconstrained['flow_norms']
    
    # Diagnostic output
    print("\n=== Diagnostic Info ===")
    print(f"Constrained trajectory shape: {traj_c.shape}")
    print(f"Unconstrained trajectory shape: {traj_u.shape}")
    print(f"\nInitial θ (constrained):   {traj_c[0]}")
    print(f"Initial θ (unconstrained): {traj_u[0]}")
    print(f"\nFinal θ (constrained):   {traj_c[-1]}")
    print(f"Final θ (unconstrained): {traj_u[-1]}")
    print(f"\nDistance between final points: {np.linalg.norm(traj_c[-1] - traj_u[-1]):.4f}")
    
    # Compute marginal and joint entropy at key points
    def compute_joint_entropy(theta, N):
        """Compute joint entropy H(X₁,X₂,X₃)"""
        marginals, probs = compute_marginals(theta, N)
        # Joint entropy: -Σ p(x) log p(x)
        p_clean = probs[probs > 1e-10]
        return -np.sum(p_clean * np.log(p_clean))
    
    marginals_c_init, _ = compute_marginals(traj_c[0], 3)
    marginals_c_final, _ = compute_marginals(traj_c[-1], 3)
    marginals_u_init, _ = compute_marginals(traj_u[0], 3)
    marginals_u_final, _ = compute_marginals(traj_u[-1], 3)
    
    sum_H_c_init = sum(marginal_entropy(m) for m in marginals_c_init)
    sum_H_c_final = sum(marginal_entropy(m) for m in marginals_c_final)
    sum_H_u_init = sum(marginal_entropy(m) for m in marginals_u_init)
    sum_H_u_final = sum(marginal_entropy(m) for m in marginals_u_final)
    
    H_joint_c_init = compute_joint_entropy(traj_c[0], 3)
    H_joint_c_final = compute_joint_entropy(traj_c[-1], 3)
    H_joint_u_init = compute_joint_entropy(traj_u[0], 3)
    H_joint_u_final = compute_joint_entropy(traj_u[-1], 3)
    
    print(f"\nMarginal Entropy Σᵢ H(Xᵢ):")
    print(f"  Constrained:   {sum_H_c_init:.6f} → {sum_H_c_final:.6f} (Δ = {abs(sum_H_c_final - sum_H_c_init):.6f})")
    print(f"  Unconstrained: {sum_H_u_init:.6f} → {sum_H_u_final:.6f} (Δ = {abs(sum_H_u_final - sum_H_u_init):.6f})")
    
    print(f"\nJoint Entropy H(X₁,X₂,X₃):")
    print(f"  Constrained:   {H_joint_c_init:.6f} → {H_joint_c_final:.6f} (Δ = {H_joint_c_final - H_joint_c_init:+.6f})")
    print(f"  Unconstrained: {H_joint_u_init:.6f} → {H_joint_u_final:.6f} (Δ = {H_joint_u_final - H_joint_u_init:+.6f})")
    print(f"  Max possible: {3*np.log(2):.6f} (uniform distribution)")
    
    # Check if constrained is really staying constant
    if 'constraint_values' in solution_constrained:
        C_values = solution_constrained['constraint_values']
        C_std = np.std(C_values)
        print(f"\nConstrained Σᵢ H(Xᵢ) std dev over trajectory: {C_std:.8f}")
        print(f"  Min: {np.min(C_values):.6f}, Max: {np.max(C_values):.6f}, Range: {np.max(C_values) - np.min(C_values):.6f}")
    
    print("=" * 70 + "\n")
    
    # 1. Trajectory comparison (θ₁ vs θ₂ projection)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj_u[:, 0], traj_u[:, 1], 'r-', alpha=0.6, linewidth=2, 
            label='Unconstrained')
    ax1.plot(traj_c[:, 0], traj_c[:, 1], 'b-', alpha=0.6, linewidth=2,
            label='Constrained')
    ax1.plot(traj_c[0, 0], traj_c[0, 1], 'go', markersize=10, 
            label='Initial', zorder=5)
    ax1.plot(traj_u[-1, 0], traj_u[-1, 1], 'r*', markersize=14,
            label='Unconstrained end', zorder=5)
    ax1.plot(traj_c[-1, 0], traj_c[-1, 1], 'bs', markersize=10,
            label='Constrained end', zorder=5)
    
    ax1.set_xlabel(r'$\theta_1$')
    ax1.set_ylabel(r'$\theta_2$')
    ax1.set_title('Trajectories in Parameter Space')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # 2. Joint entropy evolution (check monotonicity)
    ax2 = fig.add_subplot(gs[0, 1])
    # Compute joint entropy for both trajectories
    joint_entropy_u = []
    joint_entropy_c = []
    
    # Sample every 10th point for efficiency
    sample_indices = np.arange(0, max(len(traj_u), len(traj_c)), 10)
    for idx in sample_indices:
        if idx < len(traj_u):
            H_u = compute_joint_entropy(traj_u[idx], 3)
            joint_entropy_u.append(H_u)
        if idx < len(traj_c):
            H_c = compute_joint_entropy(traj_c[idx], 3)
            joint_entropy_c.append(H_c)
    
    sample_u = sample_indices[:len(joint_entropy_u)]
    sample_c = sample_indices[:len(joint_entropy_c)]
    ax2.plot(sample_u, joint_entropy_u, 'r-', linewidth=2, label='Unconstrained')
    ax2.plot(sample_c, joint_entropy_c, 'b-', linewidth=2, label='Constrained')
    ax2.axhline(3*np.log(2), color='k', linestyle='--', linewidth=1, 
               alpha=0.5, label='Max (uniform)')
    ax2.set_xlabel('Gradient step')
    ax2.set_ylabel(r'$H(X_1,X_2,X_3)$ (joint entropy)')
    ax2.set_title('Joint Entropy Evolution (should increase)')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distance from origin (convergence to uniform)
    ax3 = fig.add_subplot(gs[0, 2])
    dist_u = np.linalg.norm(traj_u, axis=1)
    dist_c = np.linalg.norm(traj_c, axis=1)
    steps_u = np.arange(len(dist_u))
    steps_c = np.arange(len(dist_c))
    
    ax3.semilogy(steps_u, dist_u, 'r-', linewidth=2, label='Unconstrained')
    ax3.semilogy(steps_c, dist_c, 'b-', linewidth=2, label='Constrained')
    ax3.set_xlabel('Gradient step')
    ax3.set_ylabel(r'$\|\theta\|$ (distance from $\theta=0$)')
    ax3.set_title('Convergence to Equilibrium')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Flow norm comparison
    ax4 = fig.add_subplot(gs[1, 0])
    steps_u = np.arange(len(flow_u))
    steps_c = np.arange(len(flow_c))
    ax4.semilogy(steps_u, flow_u, 'r-', linewidth=1.5, label='Unconstrained')
    ax4.semilogy(steps_c, flow_c, 'b-', linewidth=1.5, label='Constrained')
    ax4.set_xlabel('Gradient step')
    ax4.set_ylabel(r'$\|F(\theta)\|$ (flow magnitude)')
    ax4.set_title('Flow Norm Convergence')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, which='both')
    
    # 5. Marginal entropy over time
    ax5 = fig.add_subplot(gs[1, 1])
    # Compute marginal entropy sum for both trajectories
    marginal_entropy_u = []
    marginal_entropy_c = []
    
    # Sample every 10th point for efficiency
    sample_indices = np.arange(0, max(len(traj_u), len(traj_c)), 10)
    for idx in sample_indices:
        if idx < len(traj_u):
            marginals_u, _ = compute_marginals(traj_u[idx], 3)
            marginal_entropy_u.append(sum(marginal_entropy(m) for m in marginals_u))
        if idx < len(traj_c):
            marginals_c, _ = compute_marginals(traj_c[idx], 3)
            marginal_entropy_c.append(sum(marginal_entropy(m) for m in marginals_c))
    
    sample_u = sample_indices[:len(marginal_entropy_u)]
    sample_c = sample_indices[:len(marginal_entropy_c)]
    ax5.plot(sample_u, marginal_entropy_u, 'r-', linewidth=2, label='Unconstrained')
    ax5.plot(sample_c, marginal_entropy_c, 'b-', linewidth=2, label='Constrained')
    ax5.set_xlabel('Gradient step')
    ax5.set_ylabel(r'$\sum_i H(X_i)$ (marginal entropy sum)')
    ax5.set_title('Marginal Entropy Evolution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Text summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = (
        r"$\mathbf{Key\ Observations:}$" + "\n\n"
        
        r"$\bullet$ $\mathbf{Unconstrained}$ (red): Pure max ent $\rightarrow$ uniform" + "\n"
        f"   • Converges to θ=0 (all parameters → 0)\n"
        f"   • Final ||θ|| = {dist_u[-1]:.2e}\n"
        f"   • Marginal entropy → {marginal_entropy_u[-1]:.3f} (max)\n\n"
        
        r"$\bullet$ $\mathbf{Constrained}$ (blue): Max ent on manifold" + "\n"
        f"   • Stays on Σᵢ H(Xᵢ) = C surface\n"
        f"   • Final ||θ|| = {dist_c[-1]:.3f}\n"
        f"   • Marginal entropy = {marginal_entropy_c[-1]:.3f} (fixed)\n\n"
        
        r"$\bullet$ $\mathbf{Constraint\ effect}$:" + "\n"
        "   • Unconstrained: Straight path to origin\n"
        "   • Constrained: Curved path along manifold\n"
        "   • Different equilibria!"
    )
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='serif',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 7. Marginal parameters (θ₁, θ₂, θ₃) over time
    ax7 = fig.add_subplot(gs[2, 0])
    steps_c = np.arange(len(traj_c))
    steps_u = np.arange(len(traj_u))
    
    for i, label in enumerate([r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']):
        ax7.plot(steps_u, traj_u[:, i], '--', alpha=0.5, linewidth=1.5, 
                color=f'C{i}', label=f'{label} (unconstrained)')
        ax7.plot(steps_c, traj_c[:, i], '-', linewidth=2, 
                color=f'C{i}', label=f'{label} (constrained)')
    
    ax7.set_xlabel('Gradient step')
    ax7.set_ylabel('Parameter value')
    ax7.set_title('Marginal Bias Parameters')
    ax7.legend(fontsize=7, ncol=2, loc='best')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    # 8. Pairwise parameters (θ₁₂, θ₁₃, θ₂₃) over time
    ax8 = fig.add_subplot(gs[2, 1])
    
    for i, label in enumerate([r'$\theta_{12}$', r'$\theta_{13}$', r'$\theta_{23}$']):
        param_idx = 3 + i
        ax8.plot(steps_u, traj_u[:, param_idx], '--', alpha=0.5, linewidth=1.5,
                color=f'C{i}', label=f'{label} (unconstrained)')
        ax8.plot(steps_c, traj_c[:, param_idx], '-', linewidth=2,
                color=f'C{i}', label=f'{label} (constrained)')
    
    ax8.set_xlabel('Gradient step')
    ax8.set_ylabel('Parameter value')
    ax8.set_title('Pairwise Interaction Parameters')
    ax8.legend(fontsize=7, ncol=2, loc='best')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    # 9. Parameter trajectory in 3D (θ₁ vs θ₂ vs θ₃)
    ax9 = fig.add_subplot(gs[2, 2], projection='3d')
    
    # Downsample for clarity
    skip = 20
    ax9.plot(traj_u[::skip, 0], traj_u[::skip, 1], traj_u[::skip, 2], 
            'r-', alpha=0.5, linewidth=1.5, label='Unconstrained')
    ax9.plot(traj_c[::skip, 0], traj_c[::skip, 1], traj_c[::skip, 2],
            'b-', alpha=0.7, linewidth=2, label='Constrained')
    ax9.scatter(traj_c[0, 0], traj_c[0, 1], traj_c[0, 2], 
               c='g', s=50, marker='o', label='Initial')
    ax9.scatter(traj_u[-1, 0], traj_u[-1, 1], traj_u[-1, 2],
               c='r', s=100, marker='*', label='Unc. end')
    ax9.scatter(traj_c[-1, 0], traj_c[-1, 1], traj_c[-1, 2],
               c='b', s=80, marker='s', label='Con. end')
    
    ax9.set_xlabel(r'$\theta_1$', fontsize=8)
    ax9.set_ylabel(r'$\theta_2$', fontsize=8)
    ax9.set_zlabel(r'$\theta_3$', fontsize=8)
    ax9.set_title('3D Trajectory (Marginal Params)', fontsize=9)
    ax9.legend(fontsize=6, loc='upper left')
    ax9.grid(True, alpha=0.3)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")


def temperature_scaling_experiment(theta_base, N, beta_values=None,
                                   filename='generic_n3_temperature_scaling.pdf'):
    """
    Explore temperature scaling of GENERIC structure.
    
    Physics motivation: In Ising/Boltzmann models, parameters scale with inverse 
    temperature β. At high T (β→0), thermal fluctuations dominate (dissipative).
    At low T (β→∞), system freezes (also dissipative). At intermediate T, 
    frustrated interactions create strong conservative dynamics.
    
    This experiment validates the physics prediction that ||A||/||S|| peaks at 
    intermediate temperature.
    
    Parameters:
    -----------
    theta_base : ndarray, shape (6,)
        Base parameter vector (champion parameters at β=1)
    N : int
        Number of binary variables (should be 3)
    beta_values : ndarray, optional
        Array of inverse temperatures to scan. Default: np.linspace(0.1, 3.0, 30)
    filename : str
        Output filename for figure
        
    Returns:
    --------
    dict with keys:
        'beta_values' : array of β values
        'ratio_values' : array of ||A||/||S|| ratios
        'norm_S_values' : array of ||S|| norms
        'norm_A_values' : array of ||A|| norms
        'peak_beta' : β value at peak ratio
        'peak_ratio' : maximum ||A||/||S|| ratio
    """
    if beta_values is None:
        beta_values = np.linspace(0.1, 3.0, 30)
    
    ratio_values = []
    norm_S_values = []
    norm_A_values = []
    
    print("Running temperature scaling experiment...")
    print(f"  Scanning β ∈ [{beta_values[0]:.2f}, {beta_values[-1]:.2f}] with {len(beta_values)} points")
    
    for beta in beta_values:
        theta_scaled = beta * theta_base
        try:
            result = analyze_generic_structure(theta_scaled, N)
            ratio = result['ratio']
            ratio_values.append(ratio)
            norm_S_values.append(result['norm_S'])
            norm_A_values.append(result['norm_A'])
        except:
            # If analysis fails (e.g., numerical issues), append NaN
            ratio_values.append(np.nan)
            norm_S_values.append(np.nan)
            norm_A_values.append(np.nan)
    
    ratio_values = np.array(ratio_values)
    norm_S_values = np.array(norm_S_values)
    norm_A_values = np.array(norm_A_values)
    
    # Find peak
    valid_mask = ~np.isnan(ratio_values)
    if np.any(valid_mask):
        peak_idx = np.nanargmax(ratio_values)
        peak_beta = beta_values[peak_idx]
        peak_ratio = ratio_values[peak_idx]
    else:
        peak_beta = np.nan
        peak_ratio = np.nan
    
    print(f"  Peak ratio: ||A||/||S|| = {peak_ratio:.3f} at β = {peak_beta:.3f}")
    print(f"  Base parameters (β=1.0): ||A||/||S|| = {ratio_values[np.argmin(np.abs(beta_values - 1.0))]:.3f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: ||A||/||S|| ratio vs temperature
    ax1 = axes[0]
    ax1.plot(beta_values, ratio_values, 'b-', linewidth=2, label=r'$\|A\|/\|S\|$')
    ax1.axvline(peak_beta, color='r', linestyle='--', linewidth=1, alpha=0.7, 
                label=f'Peak: β={peak_beta:.2f}')
    ax1.axvline(1.0, color='g', linestyle='--', linewidth=1, alpha=0.7, 
                label='Base: β=1.0')
    ax1.set_xlabel(r'Inverse Temperature $\beta$', fontsize=11)
    ax1.set_ylabel(r'Ratio $\|A\|/\|S\|$', fontsize=11)
    ax1.set_title('Conservative vs Dissipative Balance', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Add regime labels
    ax1.text(0.2, ax1.get_ylim()[1]*0.9, 'High T\n(Thermal)', 
             ha='center', va='top', fontsize=9, alpha=0.7)
    ax1.text(2.7, ax1.get_ylim()[1]*0.9, 'Low T\n(Frozen)', 
             ha='center', va='top', fontsize=9, alpha=0.7)
    ax1.text(peak_beta, ax1.get_ylim()[1]*0.6, 'Frustrated\nRegime', 
             ha='center', va='top', fontsize=9, color='r', weight='bold')
    
    # Plot 2: Individual norms
    ax2 = axes[1]
    ax2.plot(beta_values, norm_S_values, 'r-', linewidth=2, label=r'$\|S\|$ (Dissipative)')
    ax2.plot(beta_values, norm_A_values, 'b-', linewidth=2, label=r'$\|A\|$ (Conservative)')
    ax2.axvline(peak_beta, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel(r'Inverse Temperature $\beta$', fontsize=11)
    ax2.set_ylabel('Frobenius Norm', fontsize=11)
    ax2.set_title('Component Magnitudes', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {filename}")
    
    return {
        'beta_values': beta_values,
        'ratio_values': ratio_values,
        'norm_S_values': norm_S_values,
        'norm_A_values': norm_A_values,
        'peak_beta': peak_beta,
        'peak_ratio': peak_ratio
    }


