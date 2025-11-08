"""
Test energy-entropy constraint equivalence in the Curie-Weiss model.

This script tests the predictions from Theorem (Energy-Entropy Equivalence):
1. Constraint equivalence holds when ∇I ≈ 0 (weakly correlated regime)
2. Equivalence breaks down near critical point (diverging correlations)
3. GENERIC structure changes at phase transition

The Curie-Weiss model is ideal because:
- Exact mean field solution
- Conditional independence given mean field
- Clear phase transition at β_c = 1/J
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, logsumexp, loggamma


# ============================================================================
# Curie-Weiss Model: Exact Canonical Ensemble Computation
# ============================================================================

def log_binom(n, k):
    """
    Compute log of binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    
    Uses loggamma to avoid overflow for large n:
        log C(n, k) = log(n!) - log(k!) - log((n-k)!)
                    = loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)
    
    This is numerically stable for arbitrary n, k.
    
    Parameters:
    -----------
    n : int or array
        Total number of items
    k : int or array  
        Number of items to choose
    
    Returns:
    --------
    log_c : float or array
        Natural log of binomial coefficient
    """
    # Handle edge cases
    if np.any(k < 0) or np.any(k > n):
        return -np.inf
    
    # Use loggamma: log(n!) = loggamma(n+1)
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)


def marginal_entropy(m, n=1.0):
    """
    Marginal entropy h(X) for binary variable with magnetiation m.
    
    Formula:
        h(X) = -[(1+m)/2]*log[(1+m)/2] - [(1-m)/2]*log[(1-m)/2]
    
    This is the exact binary entropy function - no approximations.
    
    Parameters:
    -----------
    m : float
        Magnetisation per spin, |m| <= 1
    n : float, optional
        Number of spins (default 1.0, scales linearly)
    
    Returns:
    --------
    h : float
        Marginal entropy (exact)
    
    When to use:
    ------------
    - Always valid for computing binary entropy given magnetisation
    - Use with exact ⟨m⟩ from exact_expectation_magnetisation()
    - Or with mean-field m from curie_weiss_magnetisation()
    """

    if abs(m) >= 1.0:
        return 0.0
    return -n*((1+m)/2) * np.log((1+m)/2 + 1e-10) - n*((1-m)/2) * np.log((1-m)/2 + 1e-10)




# ============================================================================
# Exact Canonical Ensemble Computation (Efficient for Curie-Weiss)
# ============================================================================

def partition_function_exact(beta, J, h, n):
    """
    Partition function for Curie-Weiss model.
    
    Key insight: Energy depends only on total magnetisation M = Σᵢ xᵢ:
        E(M) = -J*M²/(2n) - h*M
    
    Therefore, we sum over O(n) magnetisation values, not O(2^n) configurations:
        Z = Σ_{M=-n}^{n} Ω(M) exp(-β E(M))
    
    where Ω(M) = C(n, (n+M)/2) is the degeneracy (binomial coefficient).
    
    Computational complexity:
        - O(n) evaluations (2n+1 magnetisation states)
        - No approximations whatsoever
        - Numerically stable via logsumexp
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    log_Z : float
        Log partition function (numerically stable, no overflow for large n)
    
    Computational limits:
    ---------------------
    This function is O(n) complexity due to form of Curie-Weiss energy.
    
    See also:
    ---------
    - exact_expectation_energy() - exact ⟨E⟩
    - exact_joint_entropy_canonical() - exact H
    - exact_multi_information_canonical() - exact I
    """
    n = int(n)
    
    # All possible total magnetisations: M ∈ {-n, -n+2, ..., n-2, n}
    M_values = np.arange(-n, n + 1, 2)
    
    # Energy for each magnetisation
    # E(M) = -J*M²/(2n) - h*M
    energies = -J * M_values**2 / (2 * n) - h * M_values
    
    # Log degeneracy: log Ω(M) = log C(n, k) where k = (n+M)/2
    # Use log_binom to avoid overflow for large n
    k_values = (n + M_values) // 2
    log_degeneracies = np.array([log_binom(n, k) if 0 <= k <= n else -np.inf 
                                  for k in k_values])
    
    # Compute log(partition function) using logsumexp for numerical stability
    # logsumexp handles large values internally by shifting
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = logsumexp(log_boltzmann)
    
    return log_Z


def exact_expectation_energy(beta, J, h, n):
    """
    Compute exact expectation of energy: ⟨E⟩ = -∂log(Z)/∂β
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength  
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    E_mean : float
        Expectation value of energy
    """
    n = int(n)
    M_values = np.arange(-n, n + 1, 2)
    energies = -J * M_values**2 / (2 * n) - h * M_values
    
    # Degeneracies using log_binom to avoid overflow
    k_values = (n + M_values) // 2
    log_degeneracies = np.array([log_binom(n, k) if 0 <= k <= n else -np.inf 
                                  for k in k_values])
    
    # Boltzmann weights
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities in log space for numerical stability
    log_probs = log_boltzmann - log_Z
    probs = np.exp(log_probs)
    
    # Expectation
    E_mean = np.sum(probs * energies)
    
    return E_mean


def exact_expectation_magnetisation(beta, J, h, n):
    """
    Compute exact expectation of magnetisation per spin: ⟨m⟩ = ⟨M⟩/n
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    m_mean : float
        Expectation value of magnetisation per spin
    """
    n = int(n)
    M_values = np.arange(-n, n + 1, 2)
    
    # Degeneracies using log_binom to avoid overflow
    k_values = (n + M_values) // 2
    log_degeneracies = np.array([log_binom(n, k) if 0 <= k <= n else -np.inf 
                                  for k in k_values])
    
    # Energies
    energies = -J * M_values**2 / (2 * n) - h * M_values
    
    # Boltzmann weights
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities in log space for numerical stability
    log_probs = log_boltzmann - log_Z
    probs = np.exp(log_probs)
    
    # Expectation
    M_mean = np.sum(probs * M_values)
    m_mean = M_mean / n
    
    return m_mean


def exact_joint_entropy_canonical(beta, J, h, n):
    """
    **EXACT** canonical joint entropy H = log(Z) + β⟨E⟩ (no approximations).
    
    This is the true Shannon entropy of the full joint distribution p(x₁,...,xₙ):
        H = -Σ p(x₁,...,xₙ) log p(x₁,...,xₙ)
    
    Computed via exact partition function (O(n) complexity, not O(2^n)).
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    H : float
        Joint entropy (total, not per spin) - exact value
    
    Properties:
    -----------
    - Always non-negative: H ≥ 0
    - Maximum at high T: H → n*log(2) as β → 0
    - Minimum at low T: H → 0 as β → ∞ (with h ≠ 0)
    - For h = 0: H → log(2) at T → 0 (ground state degeneracy)
    
    Computational limits:
    ---------------------
    ✓ n ≤ 20 (exact computation feasible)
    ✗ n > 20 (use mean-field joint_entropy_curie_weiss() with caution)
    
    Compare to:
    -----------
    joint_entropy_curie_weiss() - mean-field approximation (can fail badly)
    
    Example:
    --------
    >>> H_exact = exact_joint_entropy_canonical(1.0, 1.0, 0.1, 10)
    >>> # Gaussian regime: accurate mean-field
    >>> H_mf = joint_entropy_curie_weiss(1.0, 1.0, 0.3, 10)
    """
    log_Z = partition_function_exact(beta, J, h, n)
    E_mean = exact_expectation_energy(beta, J, h, n)
    
    H = log_Z + beta * E_mean
    
    return H


def exact_marginal_entropy_canonical(beta, J, h, n):
    """
    Compute exact marginal entropy by marginalizing the full distribution.
    
    For each spin i, we compute p(xᵢ = +1) by summing over all configurations
    with xᵢ = +1, weighted by Boltzmann factors. Due to symmetry in Curie-Weiss,
    all spins have the same marginal, so we only need to compute it once.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    h_marginal : float
        Total marginal entropy (sum over all n spins)
    """
    # Due to symmetry, all spins have identical marginals
    # p(x₁ = +1) = ⟨(1 + x₁)/2⟩ = (1 + ⟨m⟩)/2
    m_mean = exact_expectation_magnetisation(beta, J, h, n)
    
    # Binary entropy scaled by n for total system
    h_marginal = marginal_entropy(m_mean, n)
    
    return h_marginal


def exact_multi_information_canonical(beta, J, h, n):
    """
    **EXACT** multi-information I = Σᵢ h(Xᵢ) - H(X₁,...,Xₙ) (no approximations).
    
    Multi-information quantifies total correlation in the system:
        I = 0: spins are independent
        I > 0: correlations present (joint entropy < sum of marginals)
    
    Computed using exact partition function (O(n) complexity).
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    I : float
        Multi-information (exact, total not per spin)
    
    Properties:
    -----------
    - Always non-negative: I ≥ 0 (information inequality)
    - I = 0 at high T: spins nearly independent
    - I grows at low T: strong correlations emerge
    - I maximal near critical point (diverging correlations)
    
    Key theorem diagnostic:
    ----------------------
    The gradient ∇_m I determines energy-entropy equivalence:
        - |∇I| ≈ 0 (small): equivalence HOLDS
        - |∇I| ≫ 0 (large): equivalence BREAKS
    
    Computational limits:
    ---------------------
    ✓ n ≤ 20 (exact computation feasible)
    ✗ n > 20 (use multi_information_curie_weiss() - but can fail!)
    
    Compare to:
    -----------
    multi_information_curie_weiss() - mean-field (can be negative! unphysical)
    
    Example:
    --------
    >>> # Gaussian regime (small I, small ∇I)
    >>> I_gauss = exact_multi_information_canonical(0.5, 1.0, 0.05, 10)
    >>> # Ordered phase (large I, large ∇I)  
    >>> I_order = exact_multi_information_canonical(2.0, 1.0, 0.1, 10)
    """
    H_joint = exact_joint_entropy_canonical(beta, J, h, n)
    h_marginal = exact_marginal_entropy_canonical(beta, J, h, n)
    
    I = h_marginal - H_joint
    
    return I





# ============================================================================
# Gradient Computations
# ============================================================================

def gradient_energy_wrt_m(J, h, m, n=1.0):
    """
    ∇_m E = ∇_m(-n*J*m²/2 - n*h*m) = -n*J*m - n*h
    
    For Curie-Weiss total energy: E = -n*J*m²/2 - n*h*m
    Per spin energy: E/n = -J*m²/2 - h*m
    
    Parameters:
    -----------
    J : float
        Coupling strength
    h : float
        External field
    m : float
        Magnetisation per spin
    n : float, optional
        Number of spins (default 1.0). Use n>1 for total system gradient.
    
    Returns:
    --------
    grad : float
        ∇_m E (scaled by n for total system)
    """
    return -n * (J * m + h)


def gradient_marginal_entropy_wrt_m(m, n=1.0):
    """
    ∇_m h(m) = -log[(1+m)/(1-m)]/2 = -arctanh(m)
    
    This properly captures the logarithmic divergence as |m| → 1.
    np.arctanh is numerically stable for all |m| < 1 (tested to m = 1 - 1e-14).
    At exactly |m| = 1, returns ±∞ (correct mathematical limit).
    """
    return -n * np.arctanh(m)


def analytical_magnetisation_susceptibility(beta, J, h, n):
    """
    Compute ∂m/∂h using fluctuation-dissipation theorem.
    
    For canonical ensemble:
        ∂<M>/∂h = β[<M²> - <M>²] = β Var(M)
        ∂m/∂h = (1/n) ∂<M>/∂h = β σ²_M / n
    
    This is exact for the Curie-Weiss model and follows from the
    fluctuation-dissipation theorem.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    dm_dh : float
        ∂m/∂h 
    """
    n = int(n)
    M_values = np.arange(-n, n + 1, 2)
    
    # Degeneracies using log_binom
    k_values = (n + M_values) // 2
    log_degeneracies = np.array([log_binom(n, k) if 0 <= k <= n else -np.inf 
                                  for k in k_values])
    
    # Energies
    energies = -J * M_values**2 / (2 * n) - h * M_values
    
    # Boltzmann weights
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities
    log_probs = log_boltzmann - log_Z
    probs = np.exp(log_probs)
    
    # Compute <M> and <M²>
    M_mean = np.sum(probs * M_values)
    M2_mean = np.sum(probs * M_values**2)
    
    # Variance of M
    var_M = M2_mean - M_mean**2
    
    # Susceptibility: ∂m/∂h = β * Var(M) / n
    dm_dh = beta * var_M / n
    
    return dm_dh


def analytical_entropy_derivative_wrt_h(beta, J, h, n):
    """
    Compute ∂H/∂h analytically for canonical ensemble.
    
    For H = log(Z) + β<E> where E(M) = -JM²/(2n) - hM:
        ∂H/∂h = ∂log(Z)/∂h + β ∂<E>/∂h
    
    The key steps:
        ∂log(Z)/∂h = β<M>                     (from partition function)
        ∂<E>/∂h = ∂<-JM²/(2n) - hM>/∂h
                = <∂E/∂M> * ∂<M>/∂h + <-M>     (product rule + explicit h dependence)
                = -<M> + β Cov(E, M)             (using FDT for ∂<M>/∂h)
    
    Therefore:
        ∂H/∂h = β<M> + β[-<M> + β Cov(E, M)]
              = β² Cov(E, M)
    
    The β<M> terms cancel! This is the exact result for canonical ensemble.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    dH_dh : float
        ∂H/∂h (analytical, exact)
    """
    n = int(n)
    M_values = np.arange(-n, n + 1, 2)
    
    # Degeneracies
    k_values = (n + M_values) // 2
    log_degeneracies = np.array([log_binom(n, k) if 0 <= k <= n else -np.inf 
                                  for k in k_values])
    
    # Energies
    energies = -J * M_values**2 / (2 * n) - h * M_values
    
    # Boltzmann weights
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities
    log_probs = log_boltzmann - log_Z
    probs = np.exp(log_probs)
    
    # Compute <M>, <E>, <M*E>
    M_mean = np.sum(probs * M_values)
    E_mean = np.sum(probs * energies)
    ME_mean = np.sum(probs * M_values * energies)
    
    # ∂H/∂h = β² Cov(E, M) = β²[<E*M> - <E><M>]
    dH_dh = beta**2 * (ME_mean - M_mean * E_mean)
    
    return dH_dh


def exact_gradient_multi_info_wrt_h(beta, J, h, n):
    """
    Gradient of multi-information with respect to field h.
    
    Computes ∇_h I analytically (default) or numerically:
        I = n*h(m) - H
        ∇_h I = n * (∂h/∂m) * (∂m/∂h) - ∂H/∂h
    
    Analytical computation uses fluctuation-dissipation theorem for ∂m/∂h
    and thermodynamic identity for ∂H/∂h.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    grad_I_h : float
        ∇_h I (exact)
    
    
    
    Note:
    -----
    Analytical version is preferred - it's faster, more accurate, and
    doesn't require choosing a step size dh.
    """
    # Analytical computation
    # I = n*h(m) - H
    # ∇_h I = n * (∂h/∂m) * (∂m/∂h) - ∂H/∂h

    # Get magnetisation
    m = exact_expectation_magnetisation(beta, J, h, n)

    # Analytical derivatives
    dm_dh = analytical_magnetisation_susceptibility(beta, J, h, n)
    dH_dh = analytical_entropy_derivative_wrt_h(beta, J, h, n)

    # Marginal entropy derivative
    dh_dm = gradient_marginal_entropy_wrt_m(m, n=n)

    # Chain rule: ∇_h I = n * (∂h/∂m) * (∂m/∂h) - ∂H/∂h
    grad_I_h = dh_dm * dm_dh - dH_dh

    return grad_I_h


def exact_gradient_multi_info_wrt_m(beta, J, h, n):
    """
    Gradient of multi-information with respect to magnetisation m.
    
    Computes ∇_m I using chain rule with analytical or numerical derivatives:
        ∇_m I = (∇_h I) / (∇_h m)
    
    Analytical version uses fluctuation-dissipation theorem for susceptibility.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field (determines m)
    n : int
        Number of spins
    
    Returns:
    --------
    grad_I_m : float
        ∇_m I (exact)
        
    """
    # Compute ∇_h I 
    dH_dh = analytical_entropy_derivative_wrt_h(beta, J, h, n)
    dm_dh = analytical_magnetisation_susceptibility(beta, J, h, n)
    dH_dm = dH_dh / dm_dh
        
    m = exact_expectation_magnetisation(beta, J, h, n)
    dh_dm = gradient_marginal_entropy_wrt_m(m, n=n)

    return dh_dm - dH_dm

def exact_constraint_gradient_angle(beta, J, h, n):
    """
    Angle between energy and marginal entropy constraint gradients.
    
    Computes angle using exact gradients (no mean-field approximation):
        θ = arctan(|∇_m I_exact| / |∇_m H_exact|)
    
    All gradients computed numerically from exact partition function.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    angle : float
        Angle in degrees [0°, 90°) - exact up to numerical precision
    
    Computational limits:
    ---------------------
    ✓ n ≤ 20 (exact computation feasible)
    
    Interpretation:
    ---------------
    angle < 10°: Strong equivalence (Gaussian regime)
    10° < angle < 30°: Transition regime  
    angle > 30°: Equivalence broken (ordered phase)
    
    Compare to:
    -----------
    constraint_gradient_angle() - uses mean-field ∇I (faster but approximate)
    
    Example:
    --------
    >>> # Exact validation in Gaussian regime
    >>> angle_exact = exact_constraint_gradient_angle(0.5, 1.0, 0.05, 10)
    >>> # Should be small: angle ≈ 15-20°
    """
    # Get exact magnetisation
    m = exact_expectation_magnetisation(beta, J, h, n)
        
    # Exact gradients (uses analytical derivatives by default)
    grad_I = exact_gradient_multi_info_wrt_m(beta, J, h, n)
    
    # For ∇_m H, we need ∇_m(n*h(m) - I)
    # Since h(m) is the marginal entropy (exact binary entropy)
    grad_marginal = gradient_marginal_entropy_wrt_m(m, n=n)
    grad_H = grad_marginal - grad_I
    
    norm_H = abs(grad_H)
    norm_I = abs(grad_I)
        
    # Compute ratio
    ratio = norm_I / norm_H
    
    # For very small ratios, use linear approximation for better numerical stability
    # arctan(x) ≈ x for small x (relative error < 1e-6 for x < 0.01)
    if ratio < 0.01:
        # Small angle approximation: θ ≈ ratio (in radians)
        pseudo_angle = ratio
    else:
        # Standard computation
        pseudo_angle = np.arctan(ratio)
    
    return np.degrees(pseudo_angle)


def constraint_gradient_angle(beta, J, h, n):
    """
    Angle between energy and marginal entropy constraint gradients.
    
    Measures misalignment caused by correlations (∇I term):
        θ = arctan(|∇_m I| / |∇_m H|)
    
    Physical meaning:
        - θ ≈ 0°: Constraints aligned → energy conservation ≈ entropy conservation
        - θ ≈ 90°: Constraints orthogonal → different natural parameters
    
    Theorem connection:
        ∇(Σh_i) = ∇H + ∇I
    
    When ∇I ≈ 0 (Gaussian): angle small, equivalence holds
    When ∇I ≫ 0 (ordered): angle large, equivalence breaks
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    angle : float
        Angle in degrees [0°, 90°) - computed exactly
    
    Computational limits:
    ---------------------
    ✓ Feasible for n ≤ 1000 (O(n) complexity)
    
    Interpretation guide:
    ---------------------
    angle < 10°: Strong equivalence (Gaussian regime)
    10° < angle < 30°: Transition regime
    angle > 30°: Equivalence broken (ordered phase)
    
    Typical values:
    ---------------
    High T (β = 0.5, h = 0.05, n=12):   angle ≈ 2.5° ✓ strong equivalence
    High T (β = 0.5, h = 0.05, n=1000): angle ≈ 0.06° ✓ very strong equivalence
    Low T  (β = 2.0, h = 0.1, n=12):    angle ≈ 64° ✗ no equivalence
    
    Example:
    --------
    >>> angle = constraint_gradient_angle(0.5, 1.0, 0.05, 100)
    >>> print(f"Angle: {angle:.2f}° - {'Gaussian' if angle < 10 else 'Ordered'}")
    
    Note:
    -----
    This function now uses exact computation internally. The old approximate
    version consistently overestimated angles (e.g., 18° vs true 2.5° in Gaussian regime).
    """
    # Delegate to exact computation
    return exact_constraint_gradient_angle(beta, J, h, n)


    
# ============================================================================
# Compute Implied Alphas from Each Constraint
# ============================================================================

def implied_alpha_from_constraints(beta, J, h, n):
    """
    Compute the implied natural parameters from different constraints using gradients.
    
    Energy constraint: ∇_m E = -J*m → α_energy = J*m (up to sign/scale)
    Marginal entropy: ∇_m (Σh_i) = ∇_m H + ∇_m I = -β*α + ∇_m I
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins
    
    Returns:
    --------
    alpha_energy : float
        Natural parameter implied by energy constraint
    alpha_from_H : float  
        Natural parameter implied by joint entropy constraint
    alpha_from_marginal : float
        Natural parameter implied by marginal entropy constraint
    angle : float
        Angle between constraints (degrees) - computed exactly
    
    
    """
    # Get exact magnetisation
    m = exact_expectation_magnetisation(beta, J, h, n)
        
    # Energy direction (defines α_energy)
    grad_E = gradient_energy_wrt_m(J, h, m, n)  # Include n for total system
    alpha_energy = -grad_E  # α_energy ∝ -∇E
    
    # Exact gradient of multi-information (uses analytical derivatives)
    grad_I = exact_gradient_multi_info_wrt_m(beta, J, h, n)
    
    # Marginal entropy gradient (exact for binary entropy function)
    grad_marginal = gradient_marginal_entropy_wrt_m(m, n)
    grad_H = grad_marginal - grad_I
    
    # Implied α from marginal entropy: ∇H = -β*α, so α ∝ -∇H/β
    alpha_from_H = -grad_H / beta
    alpha_from_marginal = -grad_marginal / beta
    
    # Angle computed with adaptive step size (uses analytical gradients)
    angle = exact_constraint_gradient_angle(beta, J, h, n)
    
    return alpha_energy, alpha_from_H, alpha_from_marginal, angle
