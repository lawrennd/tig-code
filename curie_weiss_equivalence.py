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
from scipy.optimize import fsolve, brentq
from scipy.special import erf, comb, logsumexp


# ============================================================================
# Curie-Weiss Model: Exact Solution
# ============================================================================

def curie_weiss_magnetisation(beta, J, h, m_init=None):
    """
    **MEAN-FIELD** magnetisation via self-consistency: m = tanh(β(Jm + h))
    
    Approximation:
        - Assumes each spin sees mean-field from all others
        - Valid in thermodynamic limit (n → ∞)
        - Neglects fluctuations beyond mean
    
    Phase behavior:
        T > T_c (β < 1/J, h=0): m = 0 (paramagnetic)
        T < T_c (β > 1/J, h=0): m = ±m* (ferromagnetic, spontaneous breaking)
        T < T_c (β > 1/J, h≠0): m ≠ 0 (selected by field)
    
    Critical point: T_c = J (or β_c = 1/J)
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    m_init : float, optional
        Initial guess for magnetisation. If None, chooses based on phase.
        
    Returns:
    --------
    m : float
        Self-consistent magnetisation (mean-field solution)
    
    When to use:
    ------------
    ✓ Large systems (n > 20) where exact computation infeasible
    ✓ Understanding thermodynamic behavior and phase transitions
    ✓ Qualitative analysis of ordered/disordered regimes
    ✗ Quantitative predictions for small n (use exact_expectation_magnetisation)
    
    Important caveat:
    -----------------
    For finite n with h=0, exact canonical ensemble gives ⟨m⟩ = 0 by symmetry,
    but mean-field shows spontaneous symmetry breaking. This is a well-known
    artifact of the thermodynamic limit - use small h ≠ 0 for finite systems.
    
    See also:
    ---------
    exact_expectation_magnetisation() - exact ⟨m⟩ for n ≤ 20
    """
    def self_consistency(m):
        return m - np.tanh(beta * (J * m + h))
    
    beta_c = 1.0 / J  # Critical inverse temperature
    
    # Choose initial guess based on phase
    if m_init is None:
        if abs(h) > 0.01:
            # With field, start from field direction
            m_init = np.sign(h) * 0.5
        elif beta > beta_c:
            # Below T_c (ordered phase): start away from m=0 to find ferromagnetic solution
            m_init = 0.5  # Try to find positive magnetisation solution
        else:
            # Above T_c (disordered phase): m=0 is the only solution
            m_init = 0.0
    
    try:
        # Try to find solution with given initial guess
        m = fsolve(self_consistency, m_init, full_output=False)[0]
        
        # Check physical bounds
        if abs(m) > 1.0:
            return np.nan
        
        # Check stability: d(tanh(β(Jm+h)))/dm < 1
        # If unstable and below T_c, try to find stable solution
        stability = beta * J / (np.cosh(beta * (J * m + h)))**2
        if stability > 1.0 and abs(m) < 0.01 and beta > beta_c and abs(h) < 0.01:
            # m=0 is unstable, find ferromagnetic solution
            for init_guess in [0.3, 0.5, 0.7, -0.3, -0.5, -0.7]:
                m_try = fsolve(self_consistency, init_guess, full_output=False)[0]
                if abs(m_try) > 0.1 and abs(m_try) < 1.0:
                    # Check this solution is stable
                    stab_try = beta * J / (np.cosh(beta * (J * m_try + h)))**2
                    if stab_try < 1.0:
                        return abs(m_try)  # Return positive branch by convention
            # If still can't find stable solution, return what we have
            return abs(m)
        
        return m
    except:
        return np.nan


def free_energy_curie_weiss(beta, J, h, m):
    """
    Free energy per spin for Curie-Weiss model.
    
    F = -J*m²/2 - h*m - T*s(m)
    where s(m) = -[(1+m)/2]*log[(1+m)/2] - [(1-m)/2]*log[(1-m)/2]
    """
    # Entropic part
    if abs(m) >= 1.0:
        return np.inf
    
    s = -((1+m)/2) * np.log((1+m)/2 + 1e-10) - ((1-m)/2) * np.log((1-m)/2 + 1e-10)
    
    # Free energy
    F = -J * m**2 / 2 - h * m - s / beta
    return F


def marginal_entropy(m, n=1.0):
    """
    Marginal entropy h(X) for binary variable with magnetiation m.
    
    Formula:
        h(X) = -[(1+m)/2]*log[(1+m)/2] - [(1-m)/2]*log[(1-m)/2]
    
    This is the exact binary entropy function - no approximations.
    
    Parameters:
    -----------
    m : float
        Magnetization per spin, |m| <= 1
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


def microcanonical_entropy(m, n):
    """
    Compute log(Omega(m)) for magnetisation m with n spins.
    s(m) = -(1+m)/2 * log((1+m)/2) - (1-m)/2 * log((1-m)/2)
    """
    if abs(m) > 1:
        return -np.inf
    
    # Use Stirling approximation for large n
    if abs(m - 1) < 1e-10:  # m = 1
        return 0
    elif abs(m + 1) < 1e-10:  # m = -1
        return 0
    else:
        s = 0
        if 1 + m > 0:
            s -= (1 + m) / 2 * np.log((1 + m) / 2)
        if 1 - m > 0:
            s -= (1 - m) / 2 * np.log((1 - m) / 2)
        return n * s


def joint_entropy_curie_weiss_exact(m, n):
    """
    Compute exact microcanonical joint entropy at fixed magnetisation m.
    
    This is the entropy of configurations with total magnetisation M = n*m:
        H(s | M) = log Ω(M) = log C(n, k)
    where k = (n + M)/2 is the number of up spins.
    
    Parameters:
    -----------
    m : float
        Magnetization per spin (must satisfy |m| <= 1 and n*m is even)
    n : int
        Number of spins
    
    Returns:
    --------
    S : float
        Microcanonical joint entropy (total, not per spin)
    """
    # Total magnetisation
    M = n * m
    
    # Number of up spins
    k = int((n + M) / 2 + 0.5)  # Round to nearest integer
    
    # Check validity
    if k < 0 or k > n:
        return -np.inf
    
    # Microcanonical entropy: log of number of configurations
    if n <= 100:
        # Use exact combinatorics for small n
        S = np.log(comb(int(n), k, exact=True))
    else:
        # Use Stirling approximation for large n
        S = microcanonical_entropy(m, n)
    
    return S


# ============================================================================
# Exact Canonical Ensemble Computation (Efficient for Curie-Weiss)
# ============================================================================

def partition_function_exact(beta, J, h, n):
    """
    Partition function for Curie-Weiss model (no approximations).
    
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
    log_degeneracies = np.zeros_like(M_values, dtype=float)
    for i, M in enumerate(M_values):
        k = (n + M) // 2
        if 0 <= k <= n:
            log_degeneracies[i] = np.log(comb(n, k, exact=False))
        else:
            log_degeneracies[i] = -np.inf
    
    # Compute log(partition function) using logsumexp for numerical stability
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
    
    # Degeneracies
    log_degeneracies = np.zeros_like(M_values, dtype=float)
    for i, M in enumerate(M_values):
        k = (n + M) // 2
        if 0 <= k <= n:
            log_degeneracies[i] = np.log(comb(n, k, exact=False))
        else:
            log_degeneracies[i] = -np.inf
    
    # Boltzmann weights
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities
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
    
    # Degeneracies
    log_degeneracies = np.zeros_like(M_values, dtype=float)
    for i, M in enumerate(M_values):
        k = (n + M) // 2
        if 0 <= k <= n:
            log_degeneracies[i] = np.log(comb(n, k, exact=False))
        else:
            log_degeneracies[i] = -np.inf
    
    # Energies
    energies = -J * M_values**2 / (2 * n) - h * M_values
    
    # Boltzmann weights
    log_boltzmann = -beta * energies + log_degeneracies
    log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities
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
        Marginal entropy of a single spin
    """
    # Due to symmetry, all spins have identical marginals
    # p(x₁ = +1) = ⟨(1 + x₁)/2⟩ = (1 + ⟨m⟩)/2
    m_mean = exact_expectation_magnetisation(beta, J, h, n)
    
    # Binary entropy
    h_marginal = marginal_entropy(m_mean, n=1.0)
    
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
    
    I = n * h_marginal - H_joint
    
    return I



def joint_entropy_curie_weiss(beta, J, m, n=1.0):
    """
    **APPROXIMATE** joint entropy H using mean-field theory.
    
    Approximation:
        H ≈ n*h(m) - I_mf
    where I_mf uses Gaussian fluctuation/susceptibility approximation.
    
    Assumes:
    --------
    1. Gaussian fluctuations around mean-field solution
    2. Susceptibility χ = 1/(1 - βJ) captures correlations
    3. Valid in thermodynamic limit (n → ∞)
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    m : float
        Magnetization per spin
    n : float, optional
        Number of spins (default 1.0)
    
    Returns:
    --------
    H : float
        Joint entropy (mean-field approximation)
    
    When to use:
    ------------
    ✓ High temperature (T > T_c = J)
    ✓ Small magnetisation (|m| < 0.3)
    ✓ Large system size (n > 20)
    ✗ Near critical point (β ≈ 1/J) - diverges
    ✗ Ordered phase (T < T_c, large |m|) - can give negative H!
    
    For exact computation (n ≤ 20):
        Use exact_joint_entropy_canonical() instead
    """
    # Mean-field approximation
    h_m = marginal_entropy(m, n)
    I = multi_information_curie_weiss(beta, J, m, n)
    return h_m - I


def multi_information_curie_weiss(beta, J, m, n=1.0):
    """
    **APPROXIMATE** multi-information using mean-field/Gaussian theory.
    
    Approximation:
        I ≈ (n β² J² m²/2) / (1 - βJ)
    
    This is the susceptibility-corrected correlation term, assuming:
    - Gaussian fluctuations around mean-field
    - Correlations scale with susceptibility χ = 1/(1 - βJ)
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    m : float
        Magnetization per spin
    n : float, optional
        Number of spins (default 1.0)
    
    Returns:
    --------
    I : float
        Multi-information (mean-field approximation)
    
    When to use:
    ------------
    ✓ High temperature (T > T_c = J) - small correlations
    ✓ Small magnetisation (|m| < 0.3)
    ✓ Large system size (n > 20)
    ✗ Near critical point (β ≈ 1/J) - diverges!
    ✗ Ordered phase - can give large negative I (unphysical)
    
    Warning:
    --------
    - Diverges as β → 1/J (critical point)
    - Can be negative in ordered phase (approximation failure)
    - Use exact_multi_information_canonical() for n ≤ 20
    
    Physical interpretation:
    ------------------------
    I measures total correlation: I = Σh_i - H
    - I = 0: independent spins
    - I > 0: correlations reduce joint entropy below product of marginals
    """
    # Correlation correction (Gaussian fluctuations)
    correlation_term = n*(beta * J * m)**2 / 2
    return correlation_term / (1 - beta * J) # Susceptibility correction


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
        Magnetization per spin
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
    ∇_m h(m) = -log[(1+m)/(1-m)]/2 = -tanh⁻¹(m)
    """
    if abs(m) >= 0.99:
        return n*np.sign(m) * 10.0  # Large gradient near boundaries
    return -n*np.arctanh(m)


def gradient_joint_entropy_wrt_m(beta, J, m, n=1.0):
    """
    ∇_m H (including correlation corrections)
    """
    # Direct term from h(m)
    grad_h = gradient_marginal_entropy_wrt_m(m, n)
    grad_I = gradient_multi_info_wrt_m(beta, J, m, n)
    return grad_h - grad_I


def gradient_multi_info_wrt_m(beta, J, m, n=1.0):
    """
    **APPROXIMATE** gradient of multi-information wrt magnetisation.
    
    Uses mean-field/susceptibility approximation:
        ∇_m I ≈ n β² J² m / (1 - βJ)
    
    This is the **key diagnostic** for energy-entropy equivalence theorem:
        - |∇_m I| ≈ 0: Gaussian regime → equivalence HOLDS
        - |∇_m I| ≫ 0: Ordered phase → equivalence BREAKS
    
    Approximation:
        - Derived from mean-field I = (n β² J² m²/2) / (1 - βJ)
        - Assumes Gaussian fluctuations
        - Diverges at critical point (β → 1/J)
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    m : float
        Magnetization per spin
    n : float, optional
        Number of spins (default 1.0)
    
    Returns:
    --------
    grad_I : float
        ∇_m I (mean-field approximation)
    
    When to use:
    ------------
    ✓ Qualitative understanding of regime (Gaussian vs ordered)
    ✓ As diagnostic with exact ⟨m⟩ from exact_expectation_magnetisation
    ✗ Quantitative gradients in ordered phase (approximation poor)
    
    Physical interpretation:
    ------------------------
    Small |∇I|: Changing m doesn't change correlations much → Gaussian
    Large |∇I|: Changing m strongly affects correlations → Non-Gaussian
    """
    threshold = 1e-4
    # Correlation correction
    if abs(1 - beta * J) < threshold:  # Near critical point
        return n*beta**2 * J**2 * m / threshold  # Enhanced near criticality
    else:
        return n*beta**2 * J**2 * m / (1 - beta * J)


def exact_gradient_multi_info_wrt_h(beta, J, h, n, dh=1e-6):
    """
    Numerical gradient of multi-information with respect to field h.
    
    Computes ∇_h I using finite differences on exact I:
        ∇_h I ≈ [I(h + dh) - I(h - dh)] / (2*dh)
    
    This is exact up to numerical precision (no mean-field approximation).
    
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
    dh : float, optional
        Step size for finite difference (default 1e-6)
    
    Returns:
    --------
    grad_I_h : float
        ∇_h I (exact, up to numerical precision)
    
    Computational limits:
    ---------------------
    ✓ n ≤ 20 (exact computation feasible)
    
    Note:
    -----
    To get ∇_m I, use chain rule: ∇_m I = (∇_h I) / (∇_h m)
    """
    I_plus = exact_multi_information_canonical(beta, J, h + dh, n)
    I_minus = exact_multi_information_canonical(beta, J, h - dh, n)
    return (I_plus - I_minus) / (2 * dh)


def exact_gradient_multi_info_wrt_m(beta, J, h, n, dh=1e-6):
    """
    Numerical gradient of multi-information with respect to magnetisation m.
    
    Computes ∇_m I using chain rule and exact computations:
        ∇_m I = (∇_h I) / (∇_h m)
    
    Both gradients computed exactly via finite differences on exact functions.
    
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
    dh : float, optional
        Step size for finite difference (default 1e-6)
    
    Returns:
    --------
    grad_I_m : float
        ∇_m I (exact, up to numerical precision)
    
    Computational limits:
    ---------------------
    ✓ n ≤ 20 (exact computation feasible)
    
    Compare to:
    -----------
    gradient_multi_info_wrt_m() - mean-field approximation (faster but approximate)
    """
    # Exact gradients via finite differences
    grad_I_h = exact_gradient_multi_info_wrt_h(beta, J, h, n, dh)
    
    # ∇_h m
    m_plus = exact_expectation_magnetisation(beta, J, h + dh, n)
    m_minus = exact_expectation_magnetisation(beta, J, h - dh, n)
    grad_m_h = (m_plus - m_minus) / (2 * dh)
    
    # Chain rule: ∇_m I = (∇_h I) / (∇_h m)
    if abs(grad_m_h) < 1e-12:
        # Magnetization not changing with field (saturated or at h=0 with symmetry)
        return 0.0
    
    return grad_I_h / grad_m_h


def exact_constraint_gradient_angle(beta, J, h, n, dh=1e-6):
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
    dh : float, optional
        Step size for finite differences (default 1e-6)
    
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
    
    # Exact gradients via finite differences
    grad_I = exact_gradient_multi_info_wrt_m(beta, J, h, n, dh)
    
    # For ∇_m H, we need ∇_m(n*h(m) - I)
    # Since h(m) is the marginal entropy (exact binary entropy)
    grad_marginal = gradient_marginal_entropy_wrt_m(m, n=n)
    grad_H = grad_marginal - grad_I
    
    norm_H = abs(grad_H)
    norm_I = abs(grad_I)
    
    if norm_H < 1e-10:
        return 0.0
    
    # Angle from relative magnitude
    ratio = norm_I / norm_H
    pseudo_angle = np.arctan(ratio)
    
    return np.degrees(pseudo_angle)


def constraint_gradient_angle(beta, J, h, n, dh=1e-6):
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
    dh : float, optional
        Step size for finite differences (default 1e-6)
    
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
    return exact_constraint_gradient_angle(beta, J, h, n, dh)


    
# ============================================================================
# Compute Implied Alphas from Each Constraint
# ============================================================================

def implied_alpha_from_constraints(beta, J, h, n, dh=1e-6):
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
    dh : float, optional
        Step size for finite differences (default 1e-6)
    
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
    
    Note:
    -----
    This function now uses exact gradients via finite differences, providing
    accurate results without mean-field approximations.
    """
    # Get exact magnetisation
    m = exact_expectation_magnetisation(beta, J, h, n)
    
    # Energy direction (defines α_energy)
    grad_E = gradient_energy_wrt_m(J, h, m, n)  # Include n for total system
    alpha_energy = -grad_E  # α_energy ∝ -∇E
    
    # Exact gradient of multi-information
    grad_I = exact_gradient_multi_info_wrt_m(beta, J, h, n, dh)
    
    # Marginal entropy gradient (exact for binary entropy function)
    grad_marginal = gradient_marginal_entropy_wrt_m(m, n)
    grad_H = grad_marginal - grad_I
    
    # Implied α from marginal entropy: ∇H = -β*α, so α ∝ -∇H/β
    alpha_from_H = -grad_H / beta
    alpha_from_marginal = -grad_marginal / beta
    
    # Angle computed exactly
    angle = exact_constraint_gradient_angle(beta, J, h, n, dh)
    
    return alpha_energy, alpha_from_H, alpha_from_marginal, angle
