"""
Test suite for harmonic oscillator energy-entropy equivalence validation.

Tests all analytical gradients against numerical differentiation.
"""

import numpy as np
import pytest
import harmonic_oscillator as ho


# Helper functions for testing
def numerical_constraint_gradient(theta, eps=1e-6):
    """
    Compute constraint gradient a(θ) = ∇_θ (h(X) + h(P)) numerically.
    
    Uses finite differences for validation/testing purposes.
    """
    _, _, C_base = ho.marginal_entropies(theta)
    if C_base is None:
        return None
    
    grad = np.zeros(3)
    for i in range(3):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        _, _, C_plus = ho.marginal_entropies(theta_plus)
        if C_plus is None:
            return None
        grad[i] = (C_plus - C_base) / eps
    
    return grad


class TestPrecisionToCovariance:
    """Tests for precision to covariance conversion."""
    
    def test_exact_2x2_inverse(self):
        """Test exact 2x2 inverse formula."""
        theta = np.array([2.0, 3.0, 0.5])
        theta_xx, theta_pp, theta_xp = theta
        
        Sigma = ho.precision_to_covariance(theta)
        
        # Reconstruct K from Sigma
        K = np.linalg.inv(Sigma)
        
        # Check K matches original
        assert np.abs(K[0, 0] - theta_xx) < 1e-12
        assert np.abs(K[1, 1] - theta_pp) < 1e-12
        assert np.abs(K[0, 1] - theta_xp) < 1e-12
        assert np.abs(K[1, 0] - theta_xp) < 1e-12
    
    def test_positive_definite_check(self):
        """Should return None for non-positive definite."""
        # det(K) = 1*1 - 2^2 = -3 < 0
        theta_bad = np.array([1.0, 1.0, 2.0])
        assert ho.precision_to_covariance(theta_bad) is None
    
    def test_determinant_formula(self):
        """Verify det(Σ) = 1/det(K)."""
        theta = np.array([5.0, 2.0, -0.3])
        theta_xx, theta_pp, theta_xp = theta
        
        Sigma = ho.precision_to_covariance(theta)
        det_K = theta_xx * theta_pp - theta_xp**2
        det_Sigma = np.linalg.det(Sigma)
        
        assert np.abs(det_Sigma * det_K - 1.0) < 1e-12


class TestLogPartitionFunction:
    """Tests for log partition function."""
    
    def test_direct_computation(self):
        """Test ψ(θ) = -(1/2) log det(K)."""
        theta = np.array([2.0, 3.0, 0.5])
        
        # Direct computation
        psi = ho.log_partition_function(theta)
        
        # Via Sigma
        Sigma = ho.precision_to_covariance(theta)
        psi_via_Sigma = 0.5 * np.log(np.linalg.det(Sigma))
        
        assert np.abs(psi - psi_via_Sigma) < 1e-12
    
    def test_consistency_with_covariance(self):
        """ψ(θ) = (1/2) log det(Σ) = -(1/2) log det(K)."""
        theta = np.array([1.0, 1.0, 0.2])
        theta_xx, theta_pp, theta_xp = theta
        
        psi = ho.log_partition_function(theta)
        det_K = theta_xx * theta_pp - theta_xp**2
        
        expected = -0.5 * np.log(det_K)
        assert np.abs(psi - expected) < 1e-12


class TestMarginalEntropies:
    """Tests for marginal entropy computation."""
    
    def test_gaussian_entropy_formula(self):
        """h(X) = (1/2)log(2πe σ_x²)."""
        theta = np.array([2.0, 3.0, 0.5])
        
        Sigma = ho.precision_to_covariance(theta)
        h_X, h_P, h_total = ho.marginal_entropies(theta)
        
        # Manual computation
        h_X_expected = 0.5 * np.log(2 * np.pi * np.e * Sigma[0, 0])
        h_P_expected = 0.5 * np.log(2 * np.pi * np.e * Sigma[1, 1])
        
        assert np.abs(h_X - h_X_expected) < 1e-12
        assert np.abs(h_P - h_P_expected) < 1e-12
        assert np.abs(h_total - (h_X + h_P)) < 1e-12
    
    def test_positive_entropies(self):
        """Marginal entropies should be positive."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            h_X, h_P, h_total = ho.marginal_entropies(theta)
            assert h_X > 0
            assert h_P > 0
            assert h_total > 0


class TestConstraintGradient:
    """Tests for constraint gradient ∇_θ (h(X) + h(P))."""
    
    def test_analytical_vs_numerical(self):
        """Analytical gradient should match numerical."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            grad_analytical = ho.constraint_gradient(theta)
            grad_numerical = numerical_constraint_gradient(theta)
            
            diff = np.linalg.norm(grad_analytical - grad_numerical)
            rel_error = diff / np.linalg.norm(grad_numerical)
            
            assert rel_error < 1e-4, f"Relative error {rel_error} too large for theta={theta}"
    
    def test_consistency_check(self):
        """Verify gradient is computing marginal sum, not joint."""
        theta = np.array([2.0, 3.0, 0.5])
        
        # Gradient of marginal sum
        grad_marginals = ho.constraint_gradient(theta)
        
        # Gradient of joint entropy (different!)
        grad_joint = ho.joint_entropy_gradient(theta)
        
        # These should NOT be equal (differ by ∇I)
        diff = np.linalg.norm(grad_marginals - grad_joint)
        assert diff > 0.01, "Marginal and joint gradients should differ significantly"


class TestJointEntropyGradient:
    """Tests for joint entropy gradient ∇_θ H(X,P)."""
    
    def test_analytical_formula(self):
        """∇_θ H = -(1/2)[Σ_11, Σ_22, 2Σ_12]."""
        theta = np.array([2.0, 3.0, 0.5])
        
        Sigma = ho.precision_to_covariance(theta)
        grad = ho.joint_entropy_gradient(theta)
        
        expected = -0.5 * np.array([Sigma[0, 0], Sigma[1, 1], 2 * Sigma[0, 1]])
        
        diff = np.linalg.norm(grad - expected)
        assert diff < 1e-12
    
    def test_numerical_verification(self):
        """Joint entropy gradient vs numerical."""
        theta = np.array([2.0, 3.0, 0.5])
        
        def joint_entropy(theta):
            """H(X,P) = log(2πe) + (1/2)log(det(Σ))."""
            psi = ho.log_partition_function(theta)
            if psi is None:
                return None
            return np.log(2 * np.pi * np.e) + psi
        
        # Numerical gradient
        eps = 1e-6
        grad_num = np.zeros(3)
        H_base = joint_entropy(theta)
        
        for i in range(3):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            H_plus = joint_entropy(theta_plus)
            grad_num[i] = (H_plus - H_base) / eps
        
        # Analytical gradient
        grad_analytical = ho.joint_entropy_gradient(theta)
        
        diff = np.linalg.norm(grad_analytical - grad_num)
        rel_error = diff / np.linalg.norm(grad_num)
        
        assert rel_error < 1e-4


class TestFisherInformation:
    """Tests for Fisher information matrix G(θ) = ∇²ψ(θ)."""
    
    def test_analytical_vs_numerical(self):
        """Analytical Fisher info should match numerical Hessian."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            G_analytical = ho.fisher_information(theta)
            G_numerical = self._numerical_hessian(theta)
            
            diff = np.linalg.norm(G_analytical - G_numerical, 'fro')
            rel_error = diff / np.linalg.norm(G_numerical, 'fro')
            
            assert rel_error < 0.001, f"Relative error {rel_error} too large for theta={theta}"
    
    def _numerical_hessian(self, theta, eps=1e-6):
        """Compute numerical Hessian of log partition function."""
        d = 3
        H = np.zeros((d, d))
        
        for i in range(d):
            for j in range(i, d):
                theta_pp = theta.copy()
                theta_pm = theta.copy()
                theta_mp = theta.copy()
                theta_mm = theta.copy()
                
                theta_pp[i] += eps; theta_pp[j] += eps
                theta_pm[i] += eps; theta_pm[j] -= eps
                theta_mp[i] -= eps; theta_mp[j] += eps
                theta_mm[i] -= eps; theta_mm[j] -= eps
                
                f_pp = ho.log_partition_function(theta_pp)
                f_pm = ho.log_partition_function(theta_pm)
                f_mp = ho.log_partition_function(theta_mp)
                f_mm = ho.log_partition_function(theta_mm)
                
                H[i,j] = H[j,i] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
        
        return H
    
    def test_positive_definite(self):
        """Fisher information should be positive definite."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            G = ho.fisher_information(theta)
            eigenvalues = np.linalg.eigvals(G)
            assert np.all(eigenvalues > 0), f"Fisher info not positive definite for theta={theta}"
    
    def test_symmetry(self):
        """Fisher information should be symmetric."""
        theta = np.array([2.0, 3.0, 0.5])
        G = ho.fisher_information(theta)
        
        diff = np.linalg.norm(G - G.T, 'fro')
        assert diff < 1e-12


class TestThirdDerivatives:
    """Tests for third derivatives T_ijk = ∂³ψ/∂θ_i∂θ_j∂θ_k."""
    
    def test_third_derivatives_via_fisher_differentiation(self):
        """
        Third derivatives should match numerical differentiation of Fisher info.
        
        This is much more stable than third-order finite differences because:
        T[i,j,k] = ∂³ψ/∂θ_i∂θ_j∂θ_k = ∂G_jk/∂θ_i
        
        So we only need first-order finite differences of G.
        """
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        eps = 1e-7
        
        for theta in test_thetas:
            T_analytical = ho.third_derivatives_log_partition(theta)
            T_numerical = np.zeros((3, 3, 3))
            
            G_base = ho.fisher_information(theta)
            
            for i in range(3):
                theta_plus = theta.copy()
                theta_plus[i] += eps
                
                G_plus = ho.fisher_information(theta_plus)
                
                # T[i,:,:] = ∂G/∂θ_i
                T_numerical[i, :, :] = (G_plus - G_base) / eps
            
            diff = np.linalg.norm(T_analytical - T_numerical)
            rel_error = diff / np.linalg.norm(T_numerical)
            
            assert rel_error < 1e-4, \
                f"Third derivatives relative error {rel_error} too large for theta={theta}"
    
    def test_symmetry(self):
        """Third derivatives tensor should be fully symmetric."""
        theta = np.array([2.0, 3.0, 0.5])
        T = ho.third_derivatives_log_partition(theta)
        
        # Check all permutations are equal
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # All 6 permutations should be equal
                    permutations = [
                        T[i, j, k],
                        T[i, k, j],
                        T[j, i, k],
                        T[j, k, i],
                        T[k, i, j],
                        T[k, j, i],
                    ]
                    
                    for perm in permutations:
                        assert np.abs(perm - T[i, j, k]) < 1e-12, \
                            f"Symmetry violated at ({i},{j},{k})"
    
    def test_sample_values(self):
        """Test specific known values at a test point."""
        theta = np.array([2.0, 3.0, 0.5])
        T = ho.third_derivatives_log_partition(theta)
        
        # These values verified with SymPy
        assert np.abs(T[0, 0, 0] - (-0.142024)) < 1e-5  # T_111
        assert np.abs(T[1, 1, 1] - (-0.042081)) < 1e-5  # T_222
        assert np.abs(T[2, 2, 2] - 0.095997) < 1e-5     # T_333
        assert np.abs(T[0, 1, 2] - 0.016438) < 1e-5     # T_123


class TestConstrainedFlow:
    """Tests for constrained flow F(θ) = -G(θ)θ + ν(θ)a(θ)."""
    
    def test_flow_tangent_to_constraint(self):
        """F(θ) should be tangent to constraint manifold: a·F = 0."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            F = ho.compute_constrained_flow(theta)
            a = ho.constraint_gradient(theta)
            
            dot_product = np.dot(F, a)
            assert np.abs(dot_product) < 1e-10, f"Flow not tangent: a·F = {dot_product}"
    
    def test_projection_formula_equivalence(self):
        """F(θ) should equal -Π_∥ G θ where Π_∥ = I - (a⊗a)/(a·a)."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            # Current implementation
            F_current = ho.compute_constrained_flow(theta)
            
            # Explicit projection formula
            G = ho.fisher_information(theta)
            a = ho.constraint_gradient(theta)
            F_unc = -G @ theta
            
            # Π_∥ = I - (a ⊗ a) / (a·a)
            d = len(theta)
            I = np.eye(d)
            a_norm_sq = np.dot(a, a)
            Pi_parallel = I - np.outer(a, a) / a_norm_sq
            
            F_projection = Pi_parallel @ F_unc
            
            # Verify equivalence
            diff = np.linalg.norm(F_current - F_projection)
            assert diff < 1e-12, f"Projection formula differs: {diff}"
    
    def test_constraint_preservation_integration(self):
        """Taking a step along F should preserve constraint (to first order)."""
        theta = np.array([2.0, 3.0, 0.5])
        _, _, C_init = ho.marginal_entropies(theta)
        
        # Take small step
        F = ho.compute_constrained_flow(theta)
        dtau = 0.001
        theta_new = theta + dtau * F
        
        _, _, C_new = ho.marginal_entropies(theta_new)
        
        # Constraint should be preserved to O(dtau²)
        constraint_drift = np.abs(C_new - C_init)
        assert constraint_drift < 1e-6, f"Constraint drift {constraint_drift} too large"


class TestJacobian:
    """Tests for Jacobian M = ∂F/∂θ."""
    
    def test_numerical_jacobian(self):
        """Numerical Jacobian should be computable."""
        theta = np.array([2.0, 3.0, 0.5])
        
        M = ho.compute_jacobian_numerical(theta)
        
        assert M is not None
        assert M.shape == (3, 3)
        assert np.all(np.isfinite(M))
    
    def test_antisymmetric_direct_vs_jacobian(self):
        """Direct analytical A should match A from Jacobian decomposition."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
            np.array([2.0, 2.0, 0.1]),  # Symmetric case
        ]
        
        for theta in test_thetas:
            # From Jacobian decomposition
            M, S, A_from_jacobian = ho.compute_jacobian_analytical(theta)
            
            # Direct analytical computation
            A_direct = ho.antisymmetric_part_analytical(theta)
            
            # Should match to machine precision
            diff = np.linalg.norm(A_direct - A_from_jacobian, 'fro')
            assert diff < 1e-10, \
                f"Direct A differs from Jacobian A by {diff} for theta={theta}"
    
    def test_antisymmetric_properties(self):
        """Direct antisymmetric part should have correct properties."""
        theta = np.array([2.0, 3.0, 0.5])
        
        A = ho.antisymmetric_part_analytical(theta)
        
        # Should be anti-symmetric
        assert np.linalg.norm(A + A.T, 'fro') < 1e-12
        
        # Diagonal should be zero
        assert np.all(np.abs(np.diag(A)) < 1e-12)
        
        # trace should be zero
        assert np.abs(np.trace(A)) < 1e-12
    
    def test_antisymmetric_vanishes_symmetric(self):
        """A[0,1] should vanish when θ_pp = θ_xx."""
        # Symmetric precisions
        theta = np.array([2.5, 2.5, 0.3])
        
        A = ho.antisymmetric_part_analytical(theta)
        
        # A[0,1] should be very small (vanishes in factored form)
        assert np.abs(A[0, 1]) < 1e-10, \
            f"A[0,1] = {A[0,1]} should vanish for symmetric precisions"


class TestDynamicsSimulation:
    """Tests for dynamics simulation."""
    
    def test_constraint_preservation_over_time(self):
        """Constraint should be preserved during simulation."""
        theta_init = np.array([2.0, 3.0, 0.5])
        _, _, C_init = ho.marginal_entropies(theta_init)
        
        tau_span = 1.0
        dtau = 0.01
        taus, trajectory, constraint_vals = ho.simulate_dynamics(theta_init, tau_span, dtau)
        
        # Check constraint is preserved
        max_drift = np.max(np.abs(constraint_vals - C_init))
        
        # Should be preserved to O(dtau²) at each step
        # Over n steps, accumulates to O(n*dtau²) = O(tau_span*dtau)
        expected_drift = tau_span * dtau
        
        assert max_drift < 0.1, f"Constraint drift {max_drift} too large"
    
    def test_trajectory_valid(self):
        """All points in trajectory should have valid theta."""
        theta_init = np.array([2.0, 3.0, 0.5])
        
        tau_span = 1.0
        dtau = 0.01
        taus, trajectory, constraint_vals = ho.simulate_dynamics(theta_init, tau_span, dtau)
        
        # All trajectory points should have positive definite K
        for theta in trajectory:
            theta_xx, theta_pp, theta_xp = theta
            det_K = theta_xx * theta_pp - theta_xp**2
            assert det_K > 0, f"Invalid theta in trajectory: det(K) = {det_K}"


class TestMultiInformation:
    """Tests for multi-information I = h(X) + h(P) - H(X,P)."""
    
    def test_positive_multi_information(self):
        """Multi-information should be non-negative."""
        test_thetas = [
            np.array([2.0, 3.0, 0.5]),
            np.array([1.0, 1.0, 0.2]),
            np.array([5.0, 2.0, -0.3]),
        ]
        
        for theta in test_thetas:
            # Marginal sum
            _, _, h_sum = ho.marginal_entropies(theta)
            
            # Joint entropy
            psi = ho.log_partition_function(theta)
            H_joint = np.log(2 * np.pi * np.e) + psi
            
            # Multi-information
            I = h_sum - H_joint
            
            assert I >= -1e-10, f"Multi-information {I} should be non-negative"
    
    def test_gradient_difference(self):
        """∇(h_sum) = ∇H + ∇I."""
        theta = np.array([2.0, 3.0, 0.5])
        
        grad_marginals = ho.constraint_gradient(theta)
        grad_joint = ho.joint_entropy_gradient(theta)
        grad_I = grad_marginals - grad_joint
        
        # Verify gradient of I numerically
        def multi_info(theta):
            _, _, h_sum = ho.marginal_entropies(theta)
            psi = ho.log_partition_function(theta)
            H_joint = np.log(2 * np.pi * np.e) + psi
            return h_sum - H_joint
        
        eps = 1e-6
        grad_I_num = np.zeros(3)
        I_base = multi_info(theta)
        
        for i in range(3):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            I_plus = multi_info(theta_plus)
            grad_I_num[i] = (I_plus - I_base) / eps
        
        diff = np.linalg.norm(grad_I - grad_I_num)
        rel_error = diff / np.linalg.norm(grad_I_num)
        
        assert rel_error < 1e-3, f"Gradient difference formula failed: error {rel_error}"


class TestProjection:
    """Test constraint projection method."""
    
    def test_projection_accuracy(self):
        """Projection should restore constraint to target value."""
        # Start with a valid state
        theta = np.array([2.0, 3.0, 0.5])
        C_target = ho.marginal_entropies(theta)[2]
        
        # Perturb slightly off constraint
        theta_perturbed = theta + np.array([0.1, 0.0, 0.0])
        C_perturbed = ho.marginal_entropies(theta_perturbed)[2]
        
        # Project back
        theta_proj = ho.project_onto_constraint(theta_perturbed, C_target)
        assert theta_proj is not None
        
        C_proj = ho.marginal_entropies(theta_proj)[2]
        
        # Should be very close to target
        assert abs(C_proj - C_target) < 1e-12
    
    def test_projection_preserves_positive_definite(self):
        """Projection should maintain positive-definiteness."""
        theta = np.array([2.0, 3.0, 0.5])
        C_target = ho.marginal_entropies(theta)[2]
        
        # Perturb
        theta_perturbed = theta + np.array([0.05, -0.03, 0.02])
        
        # Project
        theta_proj = ho.project_onto_constraint(theta_perturbed, C_target)
        assert theta_proj is not None
        
        # Check positive definiteness
        Sigma = ho.precision_to_covariance(theta_proj)
        assert Sigma is not None, "Projected state should be valid"
    
    def test_simulate_with_projection(self):
        """Simulation with projection should preserve constraint to machine precision."""
        theta_init = np.array([5.0, 0.2, 0.5])
        C_init = ho.marginal_entropies(theta_init)[2]
        
        # Run simulation with projection
        taus, trajectory, constraint_vals = ho.simulate_dynamics(
            theta_init, tau_span=2.0, dtau=0.001, project=True
        )
        
        # Check constraint preservation
        deviations = np.abs(constraint_vals - C_init)
        max_deviation = deviations.max()
        
        assert max_deviation < 1e-10, \
            f"With projection, constraint should be preserved to ~1e-12, got {max_deviation}"
    
    def test_simulate_without_projection_drifts(self):
        """Simulation without projection should show constraint drift."""
        theta_init = np.array([5.0, 0.2, 0.5])
        C_init = ho.marginal_entropies(theta_init)[2]
        
        # Run simulation without projection
        taus, trajectory, constraint_vals = ho.simulate_dynamics(
            theta_init, tau_span=5.0, dtau=0.001, project=False
        )
        
        # Check constraint drift
        deviations = np.abs(constraint_vals - C_init)
        final_deviation = deviations[-1]
        
        # Should drift noticeably
        assert final_deviation > 1e-6, \
            f"Without projection, constraint should drift, got {final_deviation}"
    
    def test_projection_convergence(self):
        """Projection should converge quickly (2-3 iterations)."""
        theta = np.array([2.0, 3.0, 0.5])
        C_target = ho.marginal_entropies(theta)[2]
        
        # Large perturbation
        theta_perturbed = theta + np.array([0.5, -0.3, 0.1])
        
        # Project with iteration tracking
        theta_proj = ho.project_onto_constraint(
            theta_perturbed, C_target, max_iter=5, tol=1e-12
        )
        
        assert theta_proj is not None, "Projection should converge"
        
        # Verify convergence
        C_proj = ho.marginal_entropies(theta_proj)[2]
        assert abs(C_proj - C_target) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

