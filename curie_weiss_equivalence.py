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
from scipy.special import erf


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



def joint_entropy_curie_weiss(beta, J, m, n=1.0):
    """
    Joint entropy H for N-spin Curie-Weiss model (per spin).
    
    In mean field: H ≈ h(m) - β²J²m²/2 (fluctuations)
    """
    h_m = marginal_entropy(m, n)
    I = multi_information_curie_weiss(beta, J, m, n)
    return h_m - I


def multi_information_curie_weiss(beta, J, m, n=1.0):
    """
    Multi-information I = N*h(m) - H
    
    For Curie-Weiss: I ≈ β²J²nm² (correlations scale with system size)
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
