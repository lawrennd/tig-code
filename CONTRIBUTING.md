# Contributing to The Inaccessible Game (TIG)

Thank you for your interest in contributing to TIG! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/tig-code.git
   cd tig-code
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install in development mode**:
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/new-analysis` for new features
- `bugfix/issue-123` for bug fixes
- `docs/improve-readme` for documentation
- `test/add-coverage` for test improvements

### 2. Make Your Changes

- Write clean, readable code following PEP 8 style guidelines
- Add docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Keep functions focused and modular

### 3. Add Tests

All new functionality must include tests:

```python
# tests/test_your_module.py
import pytest
from tig import your_module

class TestYourFeature:
    def test_basic_functionality(self):
        result = your_module.your_function(input_data)
        assert result == expected_output
    
    def test_edge_cases(self):
        with pytest.raises(ValueError):
            your_module.your_function(invalid_input)
```

Run tests locally:
```bash
pytest tests/ -v
```

Check coverage:
```bash
pytest tests/ --cov=tig --cov-report=term
```

### 4. Working with Notebooks

If you're contributing example notebooks or modifying existing ones:

**First-time setup**: Install and configure `nbstripout` to automatically clean notebook outputs before committing:

```bash
poetry install --with dev  # or: pip install nbstripout
poetry run nbstripout --install  # or: nbstripout --install
```

This creates a git filter that automatically strips outputs, execution counts, and metadata from notebooks when you commit, preventing:
- Large binary data (images, plots) from bloating the repository
- Merge conflicts from execution counts and timestamps
- Accidentally committing sensitive data in outputs

**Manual stripping**: To manually strip outputs from notebooks:

```bash
poetry run nbstripout examples/*.ipynb
# or for specific files:
poetry run nbstripout examples/my_notebook.ipynb
```

**Testing notebooks**: Before submitting, ensure your notebook runs cleanly:

```bash
# Quick smoke test (first N cells only)
pytest tests/test_notebooks.py -v -k "your_notebook"

# Full execution test (slow)
pytest tests/test_notebooks.py -v -m slow -k "your_notebook"
```

**Best practices**:
- Clear all outputs before committing (nbstripout does this automatically)
- Test that notebooks run from a fresh kernel
- Include clear markdown explanations
- Keep computational cells efficient
- Add the notebook to `test_notebooks.py` with an appropriate cell limit

### 5. Update Documentation

- Update README.md if adding new features
- Add docstrings following NumPy style:
  ```python
  def your_function(param1, param2):
      """
      Brief description.
      
      Detailed explanation of what the function does.
      
      Parameters
      ----------
      param1 : type
          Description of param1
      param2 : type
          Description of param2
      
      Returns
      -------
      return_type
          Description of return value
      
      Examples
      --------
      >>> result = your_function(1, 2)
      >>> print(result)
      3
      """
  ```

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description

Detailed explanation of:
- What changed
- Why it changed
- Any breaking changes or migration notes"
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what the PR does
- Reference to any related issues
- Screenshots/plots if adding visualizations

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Maximum line length: 100 characters (flexible for readability)
- Use meaningful variable names
- Prefer explicit over implicit

Example:
```python
# Good
def compute_fisher_information(theta: np.ndarray) -> np.ndarray:
    """Compute Fisher information matrix for given parameters."""
    # Implementation
    
# Avoid
def calc_g(th):  # Unclear abbreviations
    # Implementation
```

### Docstring Style

Use NumPy-style docstrings consistently:

```python
"""
Brief one-line description.

Extended description providing more details about the function,
its purpose, and usage context.

Parameters
----------
theta : np.ndarray, shape (n,)
    Natural parameters of exponential family
N : int
    Number of variables

Returns
-------
result : dict
    Dictionary containing:
    - 'matrix': np.ndarray - The computed matrix
    - 'norm': float - Frobenius norm

Raises
------
ValueError
    If theta has invalid shape

Examples
--------
>>> theta = np.array([1.0, 2.0, 3.0])
>>> result = compute_something(theta, N=3)
>>> print(result['norm'])
2.449

See Also
--------
related_function : Related functionality

References
----------
.. [1] Lawrence, N.D. (2025). The Inaccessible Game.
"""
```

## Testing Guidelines

### Test Structure

Organize tests by module:
```
tests/
├── test_generic_decomposition_n3.py
├── test_curie_weiss_equivalence.py
├── test_harmonic_oscillator.py
└── test_integration.py  # For end-to-end tests
```

### Test Categories

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test module interactions
3. **Property tests**: Test mathematical properties (symmetry, bounds, etc.)
4. **Regression tests**: Ensure bugs stay fixed

### Test Naming

Use descriptive test names:
```python
def test_fisher_matrix_is_symmetric():
    """Fisher information matrix should always be symmetric."""
    
def test_entropy_is_non_negative():
    """Entropy values should never be negative."""
```

## Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass locally (`pytest tests/ -v`)
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] Code is well-commented for complex logic

## Types of Contributions

### Bug Reports

When reporting bugs, include:
- TIG version (`python -c "import tig; print(tig.__version__)"`)
- Python version (`python --version`)
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback if applicable

### Feature Requests

For new features, describe:
- The problem it solves
- Proposed API/interface
- Example usage
- Why it fits the project scope

### Code Contributions

Areas where contributions are especially welcome:
- Additional test coverage
- Performance optimizations
- Documentation improvements
- Example notebooks
- New analysis tools building on core functionality

## Code Review Process

1. Maintainer will review your PR within a few days
2. Address any requested changes
3. Once approved, maintainer will merge
4. Your contribution will be acknowledged in release notes

## Questions?

- Open an issue for general questions
- Email lawrennd@gmail.com for direct inquiries
- Reference the paper for theoretical background

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to The Inaccessible Game! 🎲


