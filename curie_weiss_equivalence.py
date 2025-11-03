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
    Solve self-consistency equation: m = tanh(β(Jm + h))
    
    For h=0 and β > β_c = 1/J, there are three solutions:
    - m = 0 (unstable)
    - m = ±m* (stable, ferromagnetic)
    
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
        Self-consistent magnetisation (stable solution)
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
    
    h(X) = -[(1+m)/2]*log[(1+m)/2] - [(1-m)/2]*log[(1-m)/2]
    """

    if abs(m) >= 1.0:
        return 0.0
    return -n*((1+m)/2) * np.log((1+m)/2 + 1e-10) - n*((1-m)/2) * np.log((1-m)/2 + 1e-10)


def microcanonical_entropy(m, n):
    """
    Compute log(Omega(m)) for magnetization m with n spins.
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
    Compute exact microcanonical joint entropy at fixed magnetization m.
    
    This is the entropy of configurations with total magnetization M = n*m:
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
    # Total magnetization
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
    Compute exact partition function for Curie-Weiss model.
    
    Exploits the fact that energy only depends on total magnetization M:
        E(M) = -J*M²/(2n) - h*M
    
    So we sum over magnetizations rather than all 2^n configurations:
        Z = Σ_M Ω(M) exp(-β E(M))
    
    where Ω(M) = C(n, (n+M)/2) is the degeneracy.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    J : float
        Coupling strength
    h : float
        External field
    n : int
        Number of spins (should be reasonably small, e.g., n <= 20)
    
    Returns:
    --------
    Z : float
        Partition function
    log_Z : float
        Log partition function (more numerically stable)
    """
    n = int(n)
    
    # All possible total magnetizations: M ∈ {-n, -n+2, ..., n-2, n}
    M_values = np.arange(-n, n + 1, 2)
    
    # Energy for each magnetization
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
    
    return np.exp(log_Z), log_Z


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
    _, log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities
    log_probs = log_boltzmann - log_Z
    probs = np.exp(log_probs)
    
    # Expectation
    E_mean = np.sum(probs * energies)
    
    return E_mean


def exact_expectation_magnetization(beta, J, h, n):
    """
    Compute exact expectation of magnetization per spin: ⟨m⟩ = ⟨M⟩/n
    
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
        Expectation value of magnetization per spin
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
    _, log_Z = partition_function_exact(beta, J, h, n)
    
    # Probabilities
    log_probs = log_boltzmann - log_Z
    probs = np.exp(log_probs)
    
    # Expectation
    M_mean = np.sum(probs * M_values)
    m_mean = M_mean / n
    
    return m_mean


def exact_joint_entropy_canonical(beta, J, h, n):
    """
    Compute exact canonical joint entropy: H = log(Z) + β⟨E⟩
    
    This is the Shannon entropy of the full joint distribution p(x₁,...,xₙ).
    
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
        Joint entropy (total, not per spin)
    """
    _, log_Z = partition_function_exact(beta, J, h, n)
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
    m_mean = exact_expectation_magnetization(beta, J, h, n)
    
    # Binary entropy
    h_marginal = marginal_entropy(m_mean, n=1.0)
    
    return h_marginal


def exact_multi_information_canonical(beta, J, h, n):
    """
    Compute exact multi-information: I = Σᵢ h(Xᵢ) - H(X₁,...,Xₙ)
    
    This measures the total correlation in the system.
    
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
        Multi-information (total correlation)
    """
    H_joint = exact_joint_entropy_canonical(beta, J, h, n)
    h_marginal = exact_marginal_entropy_canonical(beta, J, h, n)
    
    I = n * h_marginal - H_joint
    
    return I



def joint_entropy_curie_weiss(beta, J, m, n=1.0):
    """
    Joint entropy H for N-spin Curie-Weiss model at fixed magnetization m.
    
    Uses mean-field approximation: H ≈ h(m) - β²J²m²/2 (Gaussian fluctuations)
    
    This approximation captures the temperature-dependent correlations that
    are relevant for testing the energy-entropy equivalence theorem.
    """
    # Mean-field approximation
    h_m = marginal_entropy(m, n)
    I = multi_information_curie_weiss(beta, J, m, n)
    return h_m - I


def multi_information_curie_weiss(beta, J, m, n=1.0):
    """
    Multi-information I = Σh_i(m) - H
    
    For Curie-Weiss mean-field: I ≈ β²J²nm²/(1-βJ) (Gaussian fluctuations)
    
    This captures the temperature-dependent correlations that grow as T → T_c.
    """
    # Correlation correction (Gaussian fluctuations)
    correlation_term = n*(beta * J * m)**2 / 2
    return correlation_term / (1 - beta * J) # Susceptibility correction


# ============================================================================
# Gradient Computations
# ============================================================================

def gradient_energy_wrt_m(J, m):
    """
    ∇_m ⟨E⟩ = ∇_m(-Jm²/2) = -Jm
    """
    return -J * m


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
    ∇_m I = ∇_m h(m) - ∇_m H
    
    This should be ≈ 0 in the Gaussian regime (equivalence holds)
    """
    threshold = 1e-4
    # Correlation correction
    if abs(1 - beta * J) < threshold:  # Near critical point
        return n*beta**2 * J**2 * m / threshold  # Enhanced near criticality
    else:
        return n*beta**2 * J**2 * m / (1 - beta * J)


def constraint_gradient_angle(beta, J, m, n=1.0):
    """
    Compute relative misalignment between constraint gradients.
    
    The theorem states: ∇(Σh_i) = ∇H + ∇I
    
    When ∇I ≈ 0, we have ∇(Σh_i) ≈ ∇H ∝ ∇E (equivalence holds)
    
    We measure the "angle" as the relative contribution of ∇I:
        θ = arctan(|∇I| / |∇H|)
    
    This is dimensionless and captures when correlations break equivalence.
    """
    grad_H = gradient_joint_entropy_wrt_m(beta, J, m, n)
    grad_I = gradient_multi_info_wrt_m(beta, J, m, n)
    
    norm_H = abs(grad_H)
    norm_I = abs(grad_I)
    
    if norm_H < 1e-10:
        return 0.0
    
    # Return relative magnitude as a "pseudo-angle" in degrees
    # When grad_I << grad_H: angle ≈ 0 (equivalence holds)
    # When grad_I ~ grad_H: angle ≈ 45 (equivalence breaking)
    # When grad_I >> grad_H: angle → 90 (equivalence fails)
    ratio = norm_I / norm_H
    pseudo_angle = np.arctan(ratio)  # Maps [0,∞) → [0, 90°)
    
    return np.degrees(pseudo_angle)


    
# ============================================================================
# Compute Implied Alphas from Each Constraint
# ============================================================================

def implied_alpha_from_constraints(beta, J, m, n=1.0):
    """
    Compute the implied natural parameters from different constraints.
    
    Energy constraint: ∇_m E = -J*m → α_energy = J*m (up to sign/scale)
    Marginal entropy: ∇_m (Σh_i) = ∇_m H + ∇_m I = -β*α + ∇_m I
    
    Returns:
    --------
    alpha_energy : float
        Natural parameter implied by energy constraint
    alpha_entropy : float  
        Natural parameter implied by marginal entropy constraint
    angle : float
        Angle between them (degrees)
    """
    # Energy direction (defines α_energy)
    grad_E = gradient_energy_wrt_m(J, m)
    alpha_energy = -grad_E  # α_energy ∝ -∇E
    
    # Marginal entropy direction
    grad_I = gradient_multi_info_wrt_m(beta, J, m, n)
    grad_marginal = gradient_marginal_entropy_wrt_m(m, n)
    grad_H = grad_marginal - grad_I
    
    # Implied α from marginal entropy: ∇H = -β*α, so α ∝ -∇H/β
    alpha_from_H = -grad_H / beta
    alpha_from_marginal = -grad_marginal / beta
    
    # Use the proper angle metric: arctan(|∇I| / |∇H|)
    # This measures how much multi-information deflects α from energy direction
    norm_H = abs(grad_H)
    norm_I = abs(grad_I)
    
    if norm_H < 1e-10:
        angle = 0.0
    else:
        # Return relative magnitude as a "pseudo-angle" in degrees
        # When grad_I << grad_H: angle ≈ 0 (equivalence holds)
        # When grad_I ~ grad_H: angle ≈ 45 (equivalence breaking)
        # When grad_I >> grad_H: angle → 90 (equivalence fails)
        ratio = norm_I / norm_H
        pseudo_angle = np.arctan(ratio)  # Maps [0,∞) → [0, 90°)
        
        angle =  np.degrees(pseudo_angle)
    
    return alpha_energy, alpha_from_H, alpha_from_marginal, angle
