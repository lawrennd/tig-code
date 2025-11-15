# The Inaccessible Game (TIG)

[![Tests](https://github.com/lawrennd/tig-code/workflows/Tests/badge.svg)](https://github.com/lawrennd/tig-code/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Python library for **information-theoretic dynamics** and **GENERIC structure emergence**, accompanying the paper:

> **The Inaccessible Game**  
> Neil D. Lawrence (2025)  
> [arXiv:2511.06795](https://arxiv.org/abs/2511.06795)

## Overview

The Inaccessible Game introduces a dynamical system derived from four information-theoretic axioms. The framework demonstrates how:

- **Marginal entropy conservation** ($\sum_i h_i = C$) acts as an isolated-system constraint
- **Maximum entropy production** subject to this constraint generates dynamics
- **GENERIC-like decomposition** emerges automatically (symmetric dissipative + antisymmetric conservative components)
- **Energy-entropy equivalence** holds in the thermodynamic limit under specific conditions

This library provides computational tools to explore these phenomena in:
- **Three-variable binary systems** (Ising models / Boltzmann machines)
- **Curie-Weiss model** (mean-field ferromagnetism with phase transitions)
- **Harmonic oscillators** (mass-spring systems)

## Installation

### From source

```bash
git clone https://github.com/lawrennd/tig-code.git
cd tig-code
pip install -e .
```

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.3 (for examples)

## Quick Start

### 1. GENERIC Decomposition for N=3 Binary Variables

```python
import numpy as np
from tig import generic_decomposition_n3 as gd

# Initialize frustrated system (Section 4.3 of paper)
theta = np.array([0, 0, 0, 1, -1, 1])  # [θ₁, θ₂, θ₃, θ₁₂, θ₁₃, θ₂₃]
theta = theta / np.linalg.norm(theta)

# Analyze GENERIC structure at this point
result = gd.analyse_generic_structure(theta, N=3)
print(f"Symmetric (dissipative):     ||S|| = {result['norm_S']:.3f}")
print(f"Antisymmetric (conservative): ||A|| = {result['norm_A']:.3f}")
print(f"Ratio:                        ||A||/||S|| = {result['ratio']:.3f}")

# Run constrained dynamics
sol = gd.solve_constrained_maxent(theta, N=3, n_steps=10000, dt=0.01)
print(f"Converged: {sol['converged']}")
print(f"Final constraint violation: {abs(sol['constraint_values'][-1] - sol['constraint_values'][0]):.2e}")
```

### 2. Curie-Weiss Model: Energy-Entropy Equivalence

```python
from tig import curie_weiss_equivalence as cw

# Parameters
J = 1.0       # Coupling strength
h = 0.01      # External field
beta_c = 1/J  # Critical inverse temperature
n = 1000      # System size

# Compute magnetization across phase transition
beta_range = np.linspace(0.5 * beta_c, 2.0 * beta_c, 100)
magnetizations = [cw.exact_expectation_magnetisation(beta, J, h, n) 
                  for beta in beta_range]

# Test equivalence condition (Section 5.2)
grad_I = cw.exact_gradient_multi_info_wrt_m(beta_c, J, h, n)
print(f"Multi-information gradient at βc: {grad_I:.3f}")
print(f"Equivalence {'holds' if abs(grad_I) < 5 else 'breaks down'}")
```

### 3. Harmonic Oscillator

```python
from tig import harmonic_oscillator as ho

# Natural parameters: [θ_xx, θ_pp, θ_xp]
theta = np.array([1.0, 1.0, 0.0])  # Isotropic case

# Compute antisymmetric operator
A = ho.compute_antisymmetric_operator(theta)
print(f"Antisymmetric operator shape: {A.shape}")
print(f"Antisymmetry check: ||A + A^T|| = {np.linalg.norm(A + A.T):.2e}")

# Verify Jacobi identity
jacobi_max = ho.verify_jacobi_identity(theta)
print(f"Jacobi identity violation: {jacobi_max:.2e}")
```

## Examples

The `examples/` directory contains Jupyter notebooks reproducing all paper figures and demonstrating key concepts:

### Quick Start: All Paper Figures

**`generate_all_paper_figures.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/tig-code/blob/main/examples/generate_all_paper_figures.ipynb)

Single notebook generating all figures from the paper:
- **N=3 Binary System** (Section 4.3): GENERIC component norms vs temperature, parameter trajectories, constrained vs unconstrained dynamics
- **Curie-Weiss Model** (Section 5.2): Magnetization phase transition, multi-information gradient scaling with system size

### Detailed Explorations

**`simulation_experiments_inaccessible_game.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/tig-code/blob/main/examples/simulation_experiments_inaccessible_game.ipynb)

Computational validation of GENERIC structure for three binary variables (Ising model):
- Temperature dependence of ||A||/||S|| ratio (frustrated systems)
- Constraint maintenance during evolution (Σᵢhᵢ = C preserved)
- Trajectory comparison: constrained vs unconstrained maximum entropy
- Marginal and joint entropy evolution
- **Jacobi identity verification**: Numerical tests across symmetric, asymmetric, and frustrated parameter regimes

**`curie_weiss_experiments_inaccessible_game.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/tig-code/blob/main/examples/curie_weiss_experiments_inaccessible_game.ipynb)

Analytical verification using mean-field ferromagnetism:
- Phase transition at critical temperature βc = 1/J
- Magnetization scaling with system size across transition
- Multi-information gradient remains O(1) as n→∞ (intensive behavior)
- Energy-entropy equivalence theorem validation
- Constraint gradient angles and alignment

**`mass-spring-simulation-example.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lawrennd/tig-code/blob/main/examples/mass-spring-simulation-example.ipynb)

Harmonic oscillator GENERIC dynamics (Section 6.2):
- Thermalisation from non-equilibrium initial states
- Constraint preservation: h(X) + h(P) = constant
- Distribution evolution in natural parameter space (θ_xx, θ_pp, θ_xp)
- Particle trajectories under time-varying distributions
- Equipartition approach and variance trade-offs
- Ensemble measurements of energy redistribution

### Running Locally

```bash
# Install with examples dependencies
pip install -e .

# Launch notebook
jupyter notebook examples/generate_all_paper_figures.ipynb
```

All notebooks are **Colab-ready**: they automatically install the `tig` package from GitHub when run in Google Colab (no local installation needed).

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_generic_decomposition_n3.py -v
pytest tests/test_curie_weiss_equivalence.py -v
pytest tests/test_harmonic_oscillator.py -v
```

Run notebook smoke tests (fast, ~2 minutes):

```bash
pytest tests/test_notebooks.py -v
```

Run full notebook tests (slow, ~30 minutes):

```bash
pytest tests/test_notebooks.py -v -m slow
```

Tests cover:
- Exponential family computations
- Entropy calculations (marginal, joint, multi-information)
- GENERIC decomposition properties (symmetry, antisymmetry, degeneracy conditions)
- Constraint preservation during dynamics
- Physical consistency (bounds, conservation laws)
- **Notebook smoke tests**: Verify imports and setup cells execute correctly

## API Documentation

### `tig.generic_decomposition_n3`

**Core Functions:**
- `analyse_generic_structure(theta, N)` - Compute M = S + A decomposition
- `solve_constrained_maxent(theta, N, n_steps, dt)` - Integrate constrained dynamics
- `solve_unconstrained_maxent(theta, N, n_steps, dt)` - Pure maximum entropy flow
- `verify_jacobi_identity(theta, N)` - Check Poisson bracket structure

**Key Outputs:**
- `norm_S`: Frobenius norm of symmetric part (dissipation strength)
- `norm_A`: Frobenius norm of antisymmetric part (conservative strength)
- `ratio`: ||A||/||S|| (regime indicator: thermodynamic vs mechanical)

### `tig.curie_weiss_equivalence`

**Exact Canonical Ensemble:**
- `exact_expectation_magnetisation(beta, J, h, n)` - Magnetization ⟨m⟩
- `exact_joint_entropy_canonical(beta, J, h, n)` - Joint entropy H
- `exact_marginal_entropy_canonical(beta, J, h, n)` - Sum of marginals Σᵢhᵢ
- `exact_multi_information_canonical(beta, J, h, n)` - Multi-information I

**Gradients:**
- `exact_gradient_multi_info_wrt_m(beta, J, h, n)` - ∇ₘI (equivalence diagnostic)
- `implied_alpha_from_constraints(beta, J, h, n)` - Natural parameter alignment

### `tig.harmonic_oscillator`

**Operators:**
- `compute_fisher_information(theta)` - Fisher information G(θ)
- `compute_antisymmetric_operator(theta)` - Antisymmetric part A
- `compute_symmetric_operator(theta)` - Symmetric part S

**Verification:**
- `verify_jacobi_identity(theta, eps)` - Test Poisson bracket condition
- `verify_degeneracy_conditions(theta)` - Check S∇E = 0, A∇H = 0

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lawrence2025inaccessible,
  title={The Inaccessible Game},
  author={Lawrence, Neil D.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Paper Sections → Code Mapping

| Paper Section | Module | Key Functions |
|---------------|--------|---------------|
| 4.3 Simulation Study | `generic_decomposition_n3` | `analyse_generic_structure`, `solve_constrained_maxent` |
| 5.2 Thermodynamic Limit | `curie_weiss_equivalence` | `exact_gradient_multi_info_wrt_m`, `exact_expectation_magnetisation` |
| 6.2 Mass Spring System | `harmonic_oscillator` | `compute_antisymmetric_operator`, `verify_jacobi_identity` |

## Key Results

The code validates:

1. **GENERIC emergence** (Figure 1): ||A||/||S|| varies from 0.16 (frustrated system) to >0.4 (near-critical), demonstrating regime diversity from thermodynamic (dissipative-dominated) to mechanical (conservative-dominated).

2. **Energy-entropy equivalence** (Figure 3-4): The multi-information gradient ∇ₘI remains O(1) as n→∞ in the Curie-Weiss model, confirming intensive behavior that enables the equivalence theorem.

3. **Antisymmetric complexity** (Section 6.2): For the harmonic oscillator, the antisymmetric operator A involves degree-7 polynomials, yet emerges automatically from constraint geometry—illustrating why GENERIC is typically hard to construct by hand.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Neil D. Lawrence**  
University of Cambridge  
📧 lawrennd@gmail.com  
🌐 [inverseprobability.com](https://inverseprobability.com)

## Acknowledgments

This work builds on foundational contributions from:
- Baez & Fritz (2014) - Information loss axioms
- Jaynes (1957, 1980) - Maximum entropy principles  
- Grmela & Öttinger (1997) - GENERIC framework
- Curie (1895) & Weiss (1907) - Ferromagnetism theory

See paper for complete references.


