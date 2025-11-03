"""
Tests for generic_decomposition_n3.py

Validates core computational functions for GENERIC structure emergence
in exponential family distributions with marginal entropy constraints.

Test Categories:
    - Exponential family functions (states, features, marginals, Fisher)
    - Entropy calculations (marginal entropy, joint entropy)
    - Constraint gradient (analytical vs numerical)
    - GENERIC decomposition (symmetry, antisymmetry, reconstruction)
    - Constraint preservation (projection, dynamics)
    - Physics consistency (non-negative quantities, bounds)

Author: Neil D. Lawrence
Date: November 2025
"""

import numpy as np
import pytest
import generic_decomposition_n3 as gd


# ============================================================================
# Test Exponential Family Functions
# ============================================================================

class TestExponentialFamily:
    """Test basic exponential family computations."""
    
    def test_generate_states_dimension(self):
        """States should have 2^N configurations."""
        N = 3
        states = gd.generate_states(N)
        assert states.shape == (2**N, N)
        assert np.all((states == 0) | (states == 1))
    
    def test_generate_states_uniqueness(self):
        """All states should be unique."""
        N = 3
        states = gd.generate_states(N)
        # Convert rows to tuples for set comparison
        state_tuples = [tuple(row) for row in states]
        assert len(state_tuples) == len(set(state_tuples))
    
    def test_compute_features_dimension(self):
        """Features should have dimension N + N(N-1)/2."""
        N = 3
        states = gd.generate_states(N)
        features = gd.compute_features_pairwise(states)
        
        expected_dim = N + N*(N-1)//2  # 3 marginals + 3 pairwise = 6
        assert features.shape == (2**N, expected_dim)
    
    def test_compute_marginals_normalization(self):
        """Marginal probabilities should sum to 1."""
        N = 3
        theta = np.array([0.1, -0.2, 0.3, -0.5, 0.4, -0.1])
        
        marginals, joint_probs = gd.compute_marginals(theta, N)
        
        # Joint probabilities should sum to 1
        assert np.abs(joint_probs.sum() - 1.0) < 1e-10
        
        # Each marginal should sum to 1
        for i, p_i in enumerate(marginals):
            assert len(p_i) == 2  # Binary variable
            assert np.abs(p_i.sum() - 1.0) < 1e-10
    
    def test_compute_marginals_non_negative(self):
        """All probabilities should be non-negative."""
        N = 3
        theta = np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.3])
        
        marginals, joint_probs = gd.compute_marginals(theta, N)
        
        assert np.all(joint_probs >= 0)
        for p_i in marginals:
            assert np.all(p_i >= 0)
    
    def test_compute_fisher_symmetric(self):
        """Fisher information matrix should be symmetric."""
        N = 3
        theta = np.array([0.2, -0.1, 0.3, -0.2, 0.4, -0.3])
        
        G = gd.compute_fisher(theta, N)
        
        # Check symmetry
        assert np.allclose(G, G.T, atol=1e-10)
    
    def test_compute_fisher_positive_definite(self):
        """Fisher information should be positive semi-definite."""
        N = 3
        theta = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.1])
        
        G = gd.compute_fisher(theta, N)
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors


# ============================================================================
# Test Entropy Calculations
# ============================================================================

class TestEntropyCalculations:
    """Test entropy-related functions."""
    
    def test_marginal_entropy_bounds(self):
        """Marginal entropy should be between 0 and log(2) for binary variables."""
        # Test extreme cases
        p_uniform = np.array([0.5, 0.5])
        p_deterministic = np.array([1.0, 0.0])
        p_near_uniform = np.array([0.49, 0.51])
        
        h_uniform = gd.marginal_entropy(p_uniform)
        h_deterministic = gd.marginal_entropy(p_deterministic)
        h_near_uniform = gd.marginal_entropy(p_near_uniform)
        
        # Uniform has maximum entropy
        assert np.abs(h_uniform - np.log(2)) < 1e-10
        
        # Deterministic has zero entropy
        assert h_deterministic < 1e-10
        
        # All entropies in valid range
        assert 0 <= h_uniform <= np.log(2) + 1e-10
        assert 0 <= h_near_uniform <= np.log(2) + 1e-10
    
    def test_marginal_entropy_concavity(self):
        """Entropy should be concave in probability."""
        # For binary variable, h(p) = -p*log(p) - (1-p)*log(1-p)
        # Should be concave
        p_values = np.linspace(0.1, 0.9, 9)
        entropies = [gd.marginal_entropy(np.array([p, 1-p])) for p in p_values]
        
        # Check that entropy increases then decreases (concave)
        # Maximum should be at p=0.5
        max_idx = np.argmax(entropies)
        assert 3 <= max_idx <= 5  # Middle indices
    
    def test_joint_entropy_non_negative(self):
        """Joint entropy should be non-negative."""
        N = 3
        theta = np.array([0.3, -0.2, 0.1, -0.4, 0.3, -0.2])
        
        marginals, joint_probs = gd.compute_marginals(theta, N)
        
        # Compute joint entropy: -Σ p(x) log p(x)
        p_clean = joint_probs[joint_probs > 1e-10]
        H_joint = -np.sum(p_clean * np.log(p_clean))
        
        assert H_joint >= 0
        assert H_joint <= N * np.log(2) + 1e-10  # Bounded by N*log(2)
    
    def test_multi_information_non_negative(self):
        """Multi-information I = Σh_i - H should be non-negative."""
        N = 3
        theta = np.array([0.2, -0.3, 0.1, 0.5, -0.4, 0.2])
        
        marginals, joint_probs = gd.compute_marginals(theta, N)
        
        # Marginal entropy sum
        sum_h = sum(gd.marginal_entropy(m) for m in marginals)
        
        # Joint entropy
        p_clean = joint_probs[joint_probs > 1e-10]
        H_joint = -np.sum(p_clean * np.log(p_clean))
        
        # Multi-information
        I = sum_h - H_joint
        
        assert I >= -1e-10  # Allow small numerical errors


# ============================================================================
# Test Constraint Gradient
# ============================================================================

class TestConstraintGradient:
    """Test constraint gradient computations."""
    
    def test_analytical_vs_numerical_gradient(self):
        """Analytical and numerical constraint gradients should match."""
        N = 3
        theta = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
        
        grad_analytical = gd.compute_constraint_gradient(theta, N)
        grad_numerical = gd.compute_constraint_gradient_numerical(theta, N, eps=1e-5)
        
        # Should match to reasonable precision
        assert np.allclose(grad_analytical, grad_numerical, rtol=1e-3, atol=1e-5)
    
    def test_gradient_dimension(self):
        """Gradient should have dimension d = N + N(N-1)/2."""
        N = 3
        theta = np.array([0.2, 0.1, -0.1, 0.3, -0.2, 0.1])
        
        grad = gd.compute_constraint_gradient(theta, N)
        
        expected_dim = N + N*(N-1)//2
        assert grad.shape == (expected_dim,)
    
    def test_gradient_finite(self):
        """Gradient should be finite (no NaNs or Infs)."""
        N = 3
        theta = np.array([0.5, -0.3, 0.2, 0.4, -0.1, 0.3])
        
        grad = gd.compute_constraint_gradient(theta, N)
        
        assert np.all(np.isfinite(grad))


# ============================================================================
# Test GENERIC Decomposition
# ============================================================================

class TestGENERICDecomposition:
    """Test GENERIC structure analysis."""
    
    def test_symmetric_antisymmetric_decomposition(self):
        """M should equal S + A."""
        N = 3
        theta = np.array([0.1, 0.2, -0.1, -0.3, 0.4, -0.2])
        
        result = gd.analyse_generic_structure(theta, N)
        
        M = result['M']
        S = result['S']
        A = result['A']
        
        # Check decomposition
        assert np.allclose(M, S + A, atol=1e-10)
    
    def test_symmetric_part_symmetric(self):
        """S should be symmetric."""
        N = 3
        theta = np.array([0.2, -0.1, 0.3, -0.2, 0.1, 0.3])
        
        result = gd.analyse_generic_structure(theta, N)
        S = result['S']
        
        assert np.allclose(S, S.T, atol=1e-10)
    
    def test_antisymmetric_part_antisymmetric(self):
        """A should be antisymmetric."""
        N = 3
        theta = np.array([0.3, 0.1, -0.2, 0.2, -0.3, 0.1])
        
        result = gd.analyse_generic_structure(theta, N)
        A = result['A']
        
        assert np.allclose(A, -A.T, atol=1e-10)
    
    def test_antisymmetric_zero_diagonal(self):
        """Antisymmetric matrix should have zero diagonal."""
        N = 3
        theta = np.array([0.1, -0.2, 0.2, 0.3, -0.1, 0.2])
        
        result = gd.analyse_generic_structure(theta, N)
        A = result['A']
        
        assert np.allclose(np.diag(A), 0, atol=1e-10)
    
    def test_ratio_non_negative(self):
        """||A||/||S|| ratio should be non-negative."""
        N = 3
        theta = np.array([0.2, 0.1, -0.1, -0.4, 0.3, -0.2])
        
        result = gd.analyse_generic_structure(theta, N)
        
        assert result['ratio'] >= 0
        assert result['norm_S'] >= 0
        assert result['norm_A'] >= 0


# ============================================================================
# Test Constraint Preservation
# ============================================================================

class TestConstraintPreservation:
    """Test constraint preservation in dynamics."""
    
    def test_project_onto_constraint_accuracy(self):
        """Projection should satisfy constraint to high precision."""
        N = 3
        theta = np.array([0.3, -0.1, 0.2, -0.2, 0.3, -0.1])
        
        # Compute initial constraint value
        marginals, _ = gd.compute_marginals(theta, N)
        C_target = sum(gd.marginal_entropy(m) for m in marginals)
        
        # Perturb theta
        theta_perturbed = theta + np.array([0.1, -0.05, 0.08, -0.03, 0.06, -0.04])
        
        # Project back
        theta_projected = gd.project_onto_constraint(theta_perturbed, N, C_target)
        
        assert theta_projected is not None
        
        # Check constraint is satisfied
        marginals_proj, _ = gd.compute_marginals(theta_projected, N)
        C_projected = sum(gd.marginal_entropy(m) for m in marginals_proj)
        
        assert np.abs(C_projected - C_target) < 1e-10
    
    def test_solve_constrained_preserves_constraint(self):
        """Constrained dynamics should preserve marginal entropy sum."""
        N = 3
        theta_init = np.array([0.2, -0.1, 0.15, -0.3, 0.25, -0.2])
        
        # Run constrained dynamics with projection
        solution = gd.solve_constrained_maxent(
            theta_init, N, 
            n_steps=100, 
            dt=0.01, 
            project=True,
            verbose=False
        )
        
        C_values = solution['constraint_values']
        C_init = solution['C_init']
        
        # Constraint should be preserved throughout
        max_deviation = np.max(np.abs(C_values - C_init))
        assert max_deviation < 1e-10
    
    def test_solve_constrained_convergence(self):
        """Constrained dynamics should reduce flow norm."""
        N = 3
        theta_init = np.array([0.3, -0.2, 0.1, -0.2, 0.3, -0.1])
        
        solution = gd.solve_constrained_maxent(
            theta_init, N,
            n_steps=1000,
            dt=0.01,
            convergence_tol=1e-5,
            verbose=False
        )
        
        # Flow norm should decrease or stay stable
        initial_norm = solution['flow_norms'][0]
        final_norm = solution['flow_norms'][-1]
        assert final_norm <= initial_norm  # Non-increasing (stable dynamics)
    
    def test_solve_constrained_flow_tangent(self):
        """Constrained flow should be tangent to constraint (a·F ≈ 0)."""
        N = 3
        theta_init = np.array([0.1, 0.2, -0.1, -0.3, 0.2, -0.1])
        
        solution = gd.solve_constrained_maxent(
            theta_init, N,
            n_steps=50,
            dt=0.01,
            verbose=False
        )
        
        # Check tangency at a point along trajectory
        theta_mid = solution['trajectory'][len(solution['trajectory'])//2]
        
        # Compute flow and constraint gradient
        G = gd.compute_fisher(theta_mid, N)
        a = gd.compute_constraint_gradient(theta_mid, N)
        F_unc = -G @ theta_mid
        nu = np.dot(F_unc, a) / np.dot(a, a)
        F = F_unc - nu * a
        
        # Flow should be tangent: a·F ≈ 0
        tangency = np.abs(np.dot(a, F))
        assert tangency < 1e-8


# ============================================================================
# Test Physics Consistency
# ============================================================================

class TestPhysicsConsistency:
    """Test physical consistency of results."""
    
    def test_uniform_distribution_maximum_entropy(self):
        """Uniform distribution (θ=0) should have maximum entropy."""
        N = 3
        theta_zero = np.zeros(6)
        
        marginals, joint_probs = gd.compute_marginals(theta_zero, N)
        
        # Joint entropy should be N*log(2)
        p_clean = joint_probs[joint_probs > 1e-10]
        H_joint = -np.sum(p_clean * np.log(p_clean))
        
        assert np.abs(H_joint - N*np.log(2)) < 1e-10
        
        # All marginals should be uniform
        for m in marginals:
            assert np.allclose(m, [0.5, 0.5], atol=1e-10)
    
    def test_zero_interactions_independence(self):
        """Zero pairwise parameters should give independent variables."""
        N = 3
        # Only marginal parameters, no interactions
        theta = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])
        
        marginals, joint_probs = gd.compute_marginals(theta, N)
        
        # Multi-information should be near zero
        sum_h = sum(gd.marginal_entropy(m) for m in marginals)
        p_clean = joint_probs[joint_probs > 1e-10]
        H_joint = -np.sum(p_clean * np.log(p_clean))
        I = sum_h - H_joint
        
        assert I < 1e-6  # Should be essentially zero
    
    def test_strong_interactions_create_correlations(self):
        """Strong pairwise interactions should create correlations."""
        N = 3
        # Strong pairwise interactions
        theta = np.array([0.0, 0.0, 0.0, 2.0, -2.0, 2.0])
        
        marginals, joint_probs = gd.compute_marginals(theta, N)
        
        # Multi-information should be positive
        sum_h = sum(gd.marginal_entropy(m) for m in marginals)
        p_clean = joint_probs[joint_probs > 1e-10]
        H_joint = -np.sum(p_clean * np.log(p_clean))
        I = sum_h - H_joint
        
        assert I > 0.1  # Significant correlations


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_large_parameters_stability(self):
        """Should handle large parameter values without overflow."""
        N = 3
        theta_large = np.array([5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
        
        # Should not raise exceptions
        marginals, joint_probs = gd.compute_marginals(theta_large, N)
        G = gd.compute_fisher(theta_large, N)
        grad = gd.compute_constraint_gradient(theta_large, N)
        
        # Results should be finite
        assert np.all(np.isfinite(joint_probs))
        assert np.all(np.isfinite(G))
        assert np.all(np.isfinite(grad))
    
    def test_small_parameters_stability(self):
        """Should handle small parameter values accurately."""
        N = 3
        theta_small = np.array([0.001, -0.001, 0.001, -0.001, 0.001, -0.001])
        
        # Should not raise exceptions
        marginals, joint_probs = gd.compute_marginals(theta_small, N)
        
        # Should be close to uniform
        for m in marginals:
            assert np.allclose(m, [0.5, 0.5], atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

