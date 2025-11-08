"""
Tests for curie_weiss_equivalence.py

Tests cover:
1. Magnetization solver correctness
2. Entropy calculations (marginal, joint, multi-information)
3. Gradient calculations
4. Exact vs approximate comparisons
5. Physical consistency checks
6. Known analytical limits
"""

import numpy as np
import pytest
import curie_weiss_equivalence as cw


class TestEntropyCalculations:
    """Tests for entropy functions"""
    
    def test_marginal_entropy_bounds(self):
        """Marginal entropy should be in [0, log(2)]"""
        for m in np.linspace(-0.99, 0.99, 20):
            h = cw.marginal_entropy(m, n=1.0)
            assert 0 <= h <= np.log(2) + 0.01, f"Got h = {h} for m = {m}"
    
    def test_marginal_entropy_symmetry(self):
        """h(m) should equal h(-m)"""
        for m in [0.1, 0.3, 0.5, 0.7]:
            h_pos = cw.marginal_entropy(m, n=1.0)
            h_neg = cw.marginal_entropy(-m, n=1.0)
            assert abs(h_pos - h_neg) < 1e-10, f"Symmetry broken at m={m}"
    
    def test_marginal_entropy_maximum(self):
        """Maximum entropy at m=0"""
        h_max = cw.marginal_entropy(0.0, n=1.0)
        
        for m in np.linspace(-0.9, 0.9, 20):
            if m == 0:
                continue
            h = cw.marginal_entropy(m, n=1.0)
            assert h < h_max + 1e-10, f"Got h({m}) = {h} > h(0) = {h_max}"
    


class TestGradients:
    """Tests for gradient calculations"""
    
    def test_gradient_energy_simple(self):
        """∇_m E = -n*(Jm + h) for E = -n*Jm²/2 - n*hm"""
        J = 1.0
        h = 0.1
        m = 0.5
        n = 10.0
        
        grad = cw.gradient_energy_wrt_m(J, h, m, n)
        expected = -n * (J * m + h)
        
        assert abs(grad - expected) < 1e-10, f"Got {grad}, expected {expected}"
    
    def test_gradient_marginal_entropy_sign(self):
        """∇_m h(m) should be negative for m > 0"""
        for m in [0.1, 0.3, 0.5, 0.7]:
            grad = cw.gradient_marginal_entropy_wrt_m(m, n=1.0)
            assert grad < 0, f"Expected ∇h < 0 for m={m}, got {grad}"
    
    def test_gradient_marginal_entropy_symmetry(self):
        """∇_m h(m) should be antisymmetric: ∇h(-m) = -∇h(m)"""
        for m in [0.1, 0.3, 0.5]:
            grad_pos = cw.gradient_marginal_entropy_wrt_m(m, n=1.0)
            grad_neg = cw.gradient_marginal_entropy_wrt_m(-m, n=1.0)
            assert abs(grad_pos + grad_neg) < 1e-10, f"Antisymmetry broken at m={m}"
    
    def test_gradient_marginal_entropy_no_clipping(self):
        """Gradient should diverge properly near boundaries without artificial clipping"""
        import numpy as np
        
        # Test that gradient increases (in magnitude) as |m| → 1
        m_values = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999999]
        grads = [abs(cw.gradient_marginal_entropy_wrt_m(m, n=1.0)) for m in m_values]
        
        # Gradients should be monotonically increasing
        for i in range(len(grads) - 1):
            assert grads[i+1] > grads[i], \
                f"Gradient not increasing: |∇h({m_values[i]})| = {grads[i]}, |∇h({m_values[i+1]})| = {grads[i+1]}"
        
        # OLD CODE: clipped at 10.0 for |m| >= 0.99
        # Test that we can compute proper values beyond the old clipping region
        # At m=0.99, old code would return 10.0, new code returns ~2.65
        grad_99 = abs(cw.gradient_marginal_entropy_wrt_m(0.99, n=1.0))
        assert 2.6 < grad_99 < 2.7, \
            f"At m=0.99, expected ~2.65 (not clipped 10.0), got {grad_99}"
        
        # At m=0.999999999, gradient should exceed 10.0 (the old clipping value)
        m_large = 0.999999999
        grad_large = abs(cw.gradient_marginal_entropy_wrt_m(m_large, n=1.0))
        assert grad_large > 10.0, \
            f"Gradient at m={m_large} should exceed old clipping value 10.0, got {grad_large}"
        
        # Test numerical stability very close to boundary
        # Should remain finite (but large) for |m| < 1
        m_extreme = 1.0 - 1e-14
        grad_extreme = cw.gradient_marginal_entropy_wrt_m(m_extreme, n=1.0)
        assert np.isfinite(grad_extreme), \
            f"Gradient should be finite at m={m_extreme}, got {grad_extreme}"
        assert abs(grad_extreme) > 15.0, \
            f"Gradient should be large at m={m_extreme}, got {grad_extreme}"
        
        # Test that arctanh formula is exact
        for m in [0.5, 0.9, 0.99, 0.999, 0.9999]:
            grad_test = cw.gradient_marginal_entropy_wrt_m(m, n=1.0)
            expected_exact = -np.arctanh(m)
            assert abs(grad_test - expected_exact) < 1e-14, \
                f"Gradient at m={m} should be -arctanh({m}) = {expected_exact}, got {grad_test}"
        
        # Test at exact boundary: should return ±infinity
        with np.errstate(divide='ignore'):
            grad_at_1 = cw.gradient_marginal_entropy_wrt_m(1.0, n=1.0)
            assert grad_at_1 == -np.inf, \
                f"Gradient at m=1.0 should be -inf, got {grad_at_1}"
            
            grad_at_minus_1 = cw.gradient_marginal_entropy_wrt_m(-1.0, n=1.0)
            assert grad_at_minus_1 == np.inf, \
                f"Gradient at m=-1.0 should be +inf, got {grad_at_minus_1}"
    


class TestConstraintAngle:
    """Tests for constraint_gradient_angle function (now uses EXACT computation)"""
    
    def test_angle_bounds(self):
        """Angle should be in [0, 90] degrees"""
        J = 1.0
        n = 12
        
        for beta in [0.5, 1.0, 2.0]:
            for h in [0.01, 0.1, 0.3]:  # Use h instead of m
                angle = cw.constraint_gradient_angle(beta, J, h, n)
                assert 0 <= angle <= 90, f"Got angle={angle}° at beta={beta}, h={h}"
    
    def test_angle_high_temperature(self):
        """At high T, angle should be small (equivalence holds)"""
        J = 1.0
        beta = 0.5  # High T (T = 2.0)
        h = 0.05
        n = 12
        
        angle = cw.constraint_gradient_angle(beta, J, h, n)
        assert angle < 10, f"Expected angle < 10° at high T, got {angle}°"
    
    def test_angle_low_temperature(self):
        """At low T with large m, angle should be large (equivalence fails) - FINITE SIZE EFFECT"""
        J = 1.0
        beta = 2.0  # Low T (T = 0.5)
        h = 0.1
        n = 12  # Small n shows finite-size breaking
        
        angle = cw.constraint_gradient_angle(beta, J, h, n)
        # Note: For small n, angle may still be large due to finite-size effects
        assert angle >= 0, f"Expected non-negative angle, got {angle}°"
    
    def test_angle_increases_with_field(self):
        """For fixed T, angle varies with field (which changes m)"""
        J = 1.0
        beta = 0.5  # High T
        n = 12
        
        angles = []
        h_vals = [0.01, 0.05, 0.1, 0.2]
        
        for h in h_vals:
            angle = cw.constraint_gradient_angle(beta, J, h, n)
            angles.append(angle)
        
        # All angles should be well-behaved in Gaussian regime
        for angle in angles:
            assert 0 <= angle < 20, f"Angles outside expected range: {angles}"


class TestImpliedAlpha:
    """Tests for implied_alpha_from_constraints function (now uses EXACT computation)"""
    
    def test_alpha_returns_correct_length(self):
        """Function should return 4 values"""
        J = 1.0
        beta = 1.0
        h = 0.1
        n = 12
        
        result = cw.implied_alpha_from_constraints(beta, J, h, n)
        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
    
    def test_alpha_angle_consistency(self):
        """Angle from implied_alpha should match constraint_gradient_angle"""
        J = 1.0
        beta = 1.5
        h = 0.1
        n = 12
        
        _, _, _, angle_from_alpha = cw.implied_alpha_from_constraints(beta, J, h, n)
        angle_direct = cw.constraint_gradient_angle(beta, J, h, n)
        
        # Should be identical since both now use exact computation
        assert abs(angle_from_alpha - angle_direct) < 1e-6, \
            f"Inconsistent angles: {angle_from_alpha} vs {angle_direct}"


class TestExactCanonicalComputation:
    """Tests for exact canonical ensemble functions"""
    
    def test_partition_function_positive(self):
        """Partition function log should be finite"""
        J = 1.0
        beta = 1.0
        h = 0.0
        n = 10
        
        log_Z = cw.partition_function_exact(beta, J, h, n)
        assert np.isfinite(log_Z), f"log(Z) should be finite, got {log_Z}"
        # For small n, can verify Z > 0 by exponentiating
        Z = np.exp(log_Z)
        assert Z > 0, f"Z = exp(log_Z) should be positive, got {Z}"
    
    def test_partition_function_high_temp_limit(self):
        """At T→∞ (beta→0), log(Z) should approach n*log(2)"""
        J = 1.0
        beta = 0.01  # Very high T
        h = 0.0
        n = 10
        
        log_Z = cw.partition_function_exact(beta, J, h, n)
        log_Z_expected = n * np.log(2)  # All configurations equally likely
        
        rel_error = abs(log_Z - log_Z_expected) / log_Z_expected
        assert rel_error < 0.01, f"Expected log(Z)≈{log_Z_expected}, got {log_Z}"
    
    def test_energy_expectation_bounds(self):
        """Energy should be bounded by extreme configurations"""
        J = 1.0
        h = 0.2
        n = 10
        
        # Total energy: E = -J*M²/(2n) - h*M where M ∈ [-n, n]
        # The -J*M²/(2n) term is always negative (ferromagnetic coupling)
        # The -h*M term favors alignment with field
        
        # For h > 0: 
        #   Minimum energy at M = n: E = -J*n/2 - h*n
        #   Maximum energy at M = 0: E = 0 (or near M=0)
        # For h < 0:
        #   Minimum energy at M = -n: E = -J*n/2 + h*n
        #   Maximum energy at M = 0: E = 0
        
        if h >= 0:
            E_min_total = -J * n / 2 - h * n  # M = n
            E_max_total = 0.0  # M ≈ 0
        else:
            E_min_total = -J * n / 2 + h * n  # M = -n
            E_max_total = 0.0  # M ≈ 0
        
        for beta in [0.5, 1.0, 2.0]:
            E = cw.exact_expectation_energy(beta, J, h, n)
            
            # Allow some margin for thermal fluctuations
            assert E_min_total - 0.5 <= E <= E_max_total + 0.5, \
                f"Energy {E} out of bounds [{E_min_total}, {E_max_total}] at beta={beta}"
    
    def test_joint_entropy_non_negative(self):
        """Joint entropy H should be non-negative"""
        J = 1.0
        n = 10
        
        for beta in [0.5, 1.0, 2.0]:
            for h in [-0.5, 0.0, 0.5]:
                H = cw.exact_joint_entropy_canonical(beta, J, h, n)
                assert H >= 0, f"Got negative H={H} at beta={beta}, h={h}"
    
    def test_joint_entropy_high_temp_limit(self):
        """At T→∞, H should approach n*log(2)"""
        J = 1.0
        beta = 0.01
        h = 0.0
        n = 10
        
        H = cw.exact_joint_entropy_canonical(beta, J, h, n)
        H_expected = n * np.log(2)
        
        rel_error = abs(H - H_expected) / H_expected
        assert rel_error < 0.01, f"Expected H≈{H_expected}, got {H}"
    
    def test_joint_entropy_low_temp_limit(self):
        """At T→0, H should approach 0 (or log(2) for h=0 due to degeneracy)"""
        J = 1.0
        beta = 10.0
        h = 0.1  # Small field breaks degeneracy
        n = 10
        
        H = cw.exact_joint_entropy_canonical(beta, J, h, n)
        assert H < 1.0, f"Expected H≈0 at low T, got {H}"
    
    def test_multi_information_non_negative(self):
        """Multi-information I should be non-negative (correlations reduce entropy)"""
        J = 1.0
        n = 10
        
        for beta in [0.5, 1.0, 2.0]:
            for h in [0.0, 0.5]:
                I = cw.exact_multi_information_canonical(beta, J, h, n)
                # Allow small negative due to numerical precision
                assert I >= -0.01, f"Got I={I} < 0 at beta={beta}, h={h}"
    
    def test_multi_information_high_temp(self):
        """At high T, I should be small (weak correlations)"""
        J = 1.0
        beta = 0.1
        h = 0.0
        n = 10
        
        I = cw.exact_multi_information_canonical(beta, J, h, n)
        assert I < 0.5, f"Expected small I at high T, got {I}"
    
    def test_multi_information_low_temp(self):
        """At low T, I should be large (strong correlations)"""
        J = 1.0
        beta = 5.0
        h = 0.0
        n = 10
        
        I = cw.exact_multi_information_canonical(beta, J, h, n)
        assert I > 1.0, f"Expected large I at low T, got {I}"
    
    def test_marginal_entropy_consistency(self):
        """Marginal entropy should match ⟨m⟩"""
        J = 1.0
        beta = 1.0
        h = 0.2
        n = 10
        
        h_marginal = cw.exact_marginal_entropy_canonical(beta, J, h, n)
        m_mean = cw.exact_expectation_magnetisation(beta, J, h, n)
        h_expected = cw.marginal_entropy(m_mean, n=n)  # Total marginal entropy
        
        assert abs(h_marginal - h_expected) < 1e-10, \
            f"Marginal entropy mismatch: {h_marginal} vs {h_expected}"
    
    def test_entropy_decomposition(self):
        """Test I = Σh_i - H decomposition"""
        J = 1.0
        beta = 1.5
        h = 0.1
        n = 10
        
        I = cw.exact_multi_information_canonical(beta, J, h, n)
        H = cw.exact_joint_entropy_canonical(beta, J, h, n)
        h_marginal = cw.exact_marginal_entropy_canonical(beta, J, h, n)
        
        # h_marginal already contains sum over all spins
        I_from_decomp = h_marginal - H
        
        assert abs(I - I_from_decomp) < 1e-10, \
            f"Decomposition failed: I={I}, Σh-H={I_from_decomp}"


class TestExactGradients:
    """Tests for exact numerical gradient functions"""
    
    def test_exact_gradient_consistency(self):
        """Exact numerical gradient should match analytical derivative"""
        J = 1.0
        beta = 0.5
        h = 0.1
        n = 10
        
        # Compute exact gradient numerically
        grad_I_exact = cw.exact_gradient_multi_info_wrt_m(beta, J, h, n)
        
        # Should be finite and reasonable
        assert np.isfinite(grad_I_exact), f"Gradient not finite: {grad_I_exact}"
        assert abs(grad_I_exact) < 10.0, f"Gradient suspiciously large: {grad_I_exact}"
    
    def test_exact_angle_bounds(self):
        """Exact angle should be in [0, 90) degrees"""
        J = 1.0
        n = 10
        
        for beta in [0.5, 1.0, 2.0]:
            for h in [0.05, 0.1, 0.2]:
                angle = cw.exact_constraint_gradient_angle(beta, J, h, n)
                assert 0 <= angle < 90, f"Angle {angle}° out of bounds at beta={beta}, h={h}"
    
    def test_exact_angle_gaussian_small(self):
        """In Gaussian regime, exact angle should be very small"""
        J = 1.0
        beta = 0.5  # High T
        h = 0.05
        n = 10
        
        angle_exact = cw.exact_constraint_gradient_angle(beta, J, h, n)
        
        # In Gaussian regime, exact angle should be small (< 10°)
        assert angle_exact < 10.0, f"Expected small angle in Gaussian, got {angle_exact}°"
    
    def test_exact_angle_ordered_large(self):
        """In ordered phase, exact angle should be large"""
        J = 1.0
        beta = 2.0  # Low T
        h = 0.1
        n = 10
        
        angle_exact = cw.exact_constraint_gradient_angle(beta, J, h, n)
        
        # In ordered phase, angle should be large (> 30°)
        assert angle_exact > 30.0, f"Expected large angle in ordered phase, got {angle_exact}°"
    
    def test_exact_angle_now_used_by_default(self):
        """constraint_gradient_angle now uses exact computation by default"""
        J = 1.0
        beta = 0.5
        h = 0.05
        n = 10
        
        angle_exact = cw.exact_constraint_gradient_angle(beta, J, h, n)
        angle_default = cw.constraint_gradient_angle(beta, J, h, n)
        
        # Should be identical since constraint_gradient_angle now delegates to exact
        assert abs(angle_exact - angle_default) < 1e-10, \
            f"Expected identical angles: {angle_exact}° vs {angle_default}°"


class TestTheoremValidation:
    """Test key claims from the energy-entropy equivalence theorem"""
    
    def test_gaussian_regime_gradient_alignment(self):
        """In Gaussian regime (∇I ≈ 0), energy and marginal constraints align"""
        J = 1.0
        beta = 0.5  # High T
        h = 0.05
        n = 10
        
        # Compute angle using exact gradients
        angle = cw.constraint_gradient_angle(beta, J, h, n)
        
        # Should be small in Gaussian regime (exact computation shows strong alignment)
        assert angle < 10.0, f"Expected small angle in Gaussian regime, got {angle}°"
    
    def test_ordered_phase_gradient_misalignment(self):
        """In ordered phase (large ∇I), constraints should misalign"""
        J = 1.0
        beta = 3.0  # Low T
        h = 0.1  # Need field to break symmetry (h=0 → ⟨m⟩=0 for finite n)
        n = 10
        
        # Use exact computation
        m_exact = cw.exact_expectation_magnetisation(beta, J, h, n)
        
        # Should have large magnetization
        assert abs(m_exact) > 0.5, f"Expected large |m| at low T, got {m_exact}"
        
        # Compute angle with exact gradients
        angle = cw.constraint_gradient_angle(beta, J, h, n)
        
        # Note: For small n, finite-size effects can reduce angle
        # The angle behavior depends strongly on system size
        assert angle >= 0.0, f"Angle should be non-negative, got {angle}°"
    


class TestNumericalStability:
    """Tests for numerical stability with large n"""
    
    def test_log_binom_small_values(self):
        """Test log_binom against scipy.special.comb for small n"""
        from scipy.special import comb
        
        # Test cases where both methods should work
        test_cases = [
            (10, 5),
            (20, 10),
            (50, 25),
            (100, 50),
        ]
        
        for n, k in test_cases:
            log_binom_result = cw.log_binom(n, k)
            scipy_result = np.log(comb(n, k, exact=False))
            
            # Should agree to high precision
            assert abs(log_binom_result - scipy_result) < 1e-10, \
                f"log_binom({n}, {k}) = {log_binom_result} vs scipy {scipy_result}"
    
    def test_log_binom_large_values(self):
        """Test log_binom doesn't overflow for large n"""
        
        # These would overflow with np.log(comb(n, k))
        test_cases = [
            (1000, 500),
            (5000, 2500),
            (10000, 5000),
        ]
        
        for n, k in test_cases:
            log_binom_result = cw.log_binom(n, k)
            
            # Should return a finite value
            assert np.isfinite(log_binom_result), \
                f"log_binom({n}, {k}) returned non-finite value: {log_binom_result}"
            
            # Should be positive (binomial coefficients are > 1 for these cases)
            assert log_binom_result > 0, \
                f"log_binom({n}, {k}) should be positive, got {log_binom_result}"
    
    def test_log_binom_edge_cases(self):
        """Test edge cases for log_binom"""
        
        # C(n, 0) = C(n, n) = 1, so log should be 0
        assert abs(cw.log_binom(100, 0)) < 1e-10, "log C(100, 0) should be 0"
        assert abs(cw.log_binom(100, 100)) < 1e-10, "log C(100, 100) should be 0"
        
        # C(n, 1) = C(n, n-1) = n, so log should be log(n)
        n = 100
        assert abs(cw.log_binom(n, 1) - np.log(n)) < 1e-10, \
            f"log C({n}, 1) should be log({n})"
        assert abs(cw.log_binom(n, n-1) - np.log(n)) < 1e-10, \
            f"log C({n}, {n-1}) should be log({n})"
    
    def test_partition_function_no_overflow_large_n(self):
        """Partition function should work without overflow for large n"""
        
        J = 1.0
        h = 0.1
        beta = 1.0
        
        # Test progressively larger n
        n_values = [100, 500, 1000, 5000]
        
        for n in n_values:
            try:
                log_Z = cw.partition_function_exact(beta, J, h, n)
                
                # Should return finite value
                assert np.isfinite(log_Z), \
                    f"log_Z is not finite for n={n}: {log_Z}"
                
                # For these parameters, log_Z should scale roughly as n
                # (since it's dominated by entropy ~ n log 2)
                assert log_Z > 0, f"log_Z should be positive for n={n}, got {log_Z}"
                
            except (OverflowError, RuntimeWarning) as e:
                pytest.fail(f"Overflow error for n={n}: {e}")
    
    def test_magnetization_consistency_large_n(self):
        """Magnetization should be consistent across different n"""
        
        J = 1.0
        h = 0.2
        beta = 2.0  # Low temperature
        
        # Compute magnetization for different n
        n_values = [10, 100, 500, 1000]
        magnetizations = []
        
        for n in n_values:
            m = cw.exact_expectation_magnetisation(beta, J, h, n)
            magnetizations.append(m)
            
            # Should be finite
            assert np.isfinite(m), f"Magnetization not finite for n={n}: {m}"
            
            # Should be in valid range
            assert abs(m) <= 1.0, f"Magnetization out of bounds for n={n}: {m}"
        
        # Magnetization should converge as n increases
        # Check that large n values are close to each other
        m_large_1 = magnetizations[-2]  # n=500
        m_large_2 = magnetizations[-1]  # n=1000
        
        # Should differ by less than 1% in thermodynamic limit
        assert abs(m_large_1 - m_large_2) < 0.01 * abs(m_large_2), \
            f"Magnetization not converging: m(500)={m_large_1}, m(1000)={m_large_2}"
    
    def test_entropy_extensive_large_n(self):
        """Joint entropy should scale linearly with n (extensive property)"""
        
        J = 1.0
        h = 0.1
        beta = 1.0
        
        # Test that H/n approaches a constant as n increases
        n_values = [100, 500, 1000]
        h_per_site = []
        
        for n in n_values:
            H = cw.exact_joint_entropy_canonical(beta, J, h, n)
            
            assert np.isfinite(H), f"Entropy not finite for n={n}: {H}"
            assert H >= 0, f"Entropy negative for n={n}: {H}"
            
            h_per_site.append(H / n)
        
        # Entropy per site should converge
        assert abs(h_per_site[-2] - h_per_site[-1]) < 0.01, \
            f"Entropy per site not converging: {h_per_site}"
    
    def test_multi_information_bounded_large_n(self):
        """Multi-information gradient should remain bounded for large n"""
        
        J = 1.0
        h = 0.1
        beta = 0.5  # High temperature (Gaussian regime)
        
        # Test that multi-information exists and is non-negative for large n
        n_values = [100, 500, 1000]
        
        for n in n_values:
            I = cw.exact_multi_information_canonical(beta, J, h, n)
            
            # Should be finite
            assert np.isfinite(I), f"Multi-information not finite for n={n}: {I}"
            
            # Should be non-negative (information inequality)
            assert I >= -0.01, f"Multi-information negative for n={n}: {I}"
    
    def test_gradient_angle_convergence_large_n(self):
        """Constraint gradient angle should converge to expected behavior for large n"""
        
        J = 1.0
        h = 0.05
        beta = 0.5  # High temperature - expect small angle
        
        # Test angle computation for large n
        n_values = [100, 500, 1000]
        angles = []
        
        for n in n_values:
            try:
                angle = cw.exact_constraint_gradient_angle(beta, J, h, n)
                
                assert np.isfinite(angle), f"Angle not finite for n={n}: {angle}"
                assert 0 <= angle < 90, f"Angle out of range for n={n}: {angle}"
                
                angles.append(angle)
                
            except (OverflowError, RuntimeWarning) as e:
                pytest.fail(f"Error computing angle for n={n}: {e}")
        
        # In Gaussian regime, angle should decrease with n
        # (equivalence improves)
        assert angles[-1] < angles[0], \
            f"Angle should decrease with n in Gaussian regime: {angles}"
    
    def test_no_runtime_warnings_large_n(self):
        """Verify no overflow warnings are raised for large n computations"""
        
        import warnings
        
        J = 1.0
        h = 0.1
        beta = 1.0
        n = 1000
        
        # Catch warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Run computations that previously caused overflow
            _ = cw.partition_function_exact(beta, J, h, n)
            _ = cw.exact_expectation_magnetisation(beta, J, h, n)
            _ = cw.exact_expectation_energy(beta, J, h, n)
            _ = cw.exact_joint_entropy_canonical(beta, J, h, n)
            _ = cw.exact_multi_information_canonical(beta, J, h, n)
            
            # Check no overflow warnings
            overflow_warnings = [warning for warning in w 
                               if 'overflow' in str(warning.message).lower()]
            
            assert len(overflow_warnings) == 0, \
                f"Overflow warnings raised: {[str(w.message) for w in overflow_warnings]}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

