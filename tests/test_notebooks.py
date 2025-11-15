"""
Test that example notebooks execute without errors.

These are smoke tests - they verify notebooks can start executing
(imports, setup) without running expensive computations.
"""
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


@pytest.mark.parametrize("notebook_path,max_cells", [
    ("examples/generate_all_paper_figures.ipynb", 5),
    ("examples/simulation_experiments_inaccessible_game.ipynb", 8),
    ("examples/curie_weiss_experiments_inaccessible_game.ipynb", 8),
    ("examples/mass-spring-simulation-example.ipynb", 6),
])
def test_notebook_smoke(notebook_path, max_cells):
    """
    Smoke test: Execute first N cells to verify imports and basic setup.
    
    This catches 90% of issues (import errors, syntax errors, missing deps)
    without running expensive computations.
    """
    # Read notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Only execute first N cells (imports, setup, configuration)
    original_cells = nb.cells.copy()
    nb.cells = nb.cells[:max_cells]
    
    # Configure executor with short timeout
    ep = ExecutePreprocessor(
        timeout=120,  # 2 minutes should be plenty for setup cells
        kernel_name='python3',
        allow_errors=False
    )
    
    # Execute notebook
    try:
        ep.preprocess(nb, {'metadata': {'path': 'examples/'}})
        print(f"✓ {notebook_path} smoke test passed ({max_cells} cells)")
    except CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_path} failed at cell {e.cell_index}: {str(e)}")
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed: {str(e)}")
    finally:
        # Restore original cells
        nb.cells = original_cells


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", [
    "examples/generate_all_paper_figures.ipynb",
])
def test_full_notebook_execution(notebook_path):
    """
    Test full notebook execution (slower, marked with @pytest.mark.slow).
    
    Run with: pytest tests/test_notebooks.py -v -m slow
    
    This runs without QUICK_TEST mode for complete validation.
    """
    # Temporarily disable quick test mode
    os.environ.pop('QUICK_TEST', None)
    
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Longer timeout for full execution
    ep = ExecutePreprocessor(
        timeout=1800,  # 30 minutes
        kernel_name='python3',
        allow_errors=False
    )
    
    try:
        ep.preprocess(nb, {'metadata': {'path': 'examples/'}})
        print(f"✓ {notebook_path} executed successfully (full run)")
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed: {str(e)}")
    finally:
        # Restore quick test mode
        os.environ['QUICK_TEST'] = 'true'

