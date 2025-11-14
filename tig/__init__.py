"""
The Inaccessible Game (TIG) - Core Library
===========================================

A Python library for information-theoretic dynamics and GENERIC structure emergence.

This package accompanies the paper "The Inaccessible Game" by Neil D. Lawrence
and provides computational tools for:

- GENERIC decomposition (symmetric/antisymmetric flow analysis)
- Curie-Weiss model energy-entropy equivalence validation
- Harmonic oscillator constrained dynamics
- Information geometry and Fisher information calculations

Modules:
--------
generic_decomposition_n3
    GENERIC decomposition for three binary variables with pairwise interactions
    
curie_weiss_equivalence
    Exact canonical ensemble computations for the Curie-Weiss model
    
harmonic_oscillator
    Harmonic oscillator dynamics under marginal entropy constraints

Quick Start:
------------
>>> import tig
>>> from tig import generic_decomposition_n3 as gd
>>> 
>>> # Analyze GENERIC structure for frustrated system
>>> theta = np.array([0, 0, 0, 1, -1, 1])
>>> theta = theta / np.linalg.norm(theta)
>>> result = gd.analyse_generic_structure(theta, N=3)
>>> print(f"||S|| = {result['norm_S']:.3f}, ||A|| = {result['norm_A']:.3f}")

>>> # Run constrained dynamics
>>> sol = gd.solve_constrained_maxent(theta, N=3, n_steps=1000, dt=0.01)

>>> # Curie-Weiss model
>>> from tig import curie_weiss_equivalence as cw
>>> m = cw.exact_expectation_magnetisation(beta=1.0, J=1.0, h=0.01, n=100)

References:
-----------
Lawrence, N. D. (2025). The Inaccessible Game. arXiv:XXXX.XXXXX

License:
--------
MIT License - see LICENSE file for details

Author:
-------
Neil D. Lawrence <lawrennd@gmail.com>
"""

__version__ = "0.1.0"
__author__ = "Neil D. Lawrence"
__email__ = "lawrennd@gmail.com"

# Import main modules for convenience
from . import generic_decomposition_n3
from . import curie_weiss_equivalence
from . import harmonic_oscillator

__all__ = [
    "generic_decomposition_n3",
    "curie_weiss_equivalence",
    "harmonic_oscillator",
]

