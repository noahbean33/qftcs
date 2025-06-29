import pandas as pd
from pandas.testing import assert_frame_equal
import os
import pytest
import numpy as np
import sympy
import matplotlib

# Use a non-interactive backend for tests to prevent GUI windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.utils import export_to_csv, load_from_csv, plot_metric_component

def test_csv_export_import(tmp_path):
    """
    Tests that data can be exported to and imported from a CSV file.
    """
    # Create sample data
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df_original = pd.DataFrame(data)
    
    # Define file path in the temporary directory
    file_path = tmp_path / "test_data.csv"
    
    # Export data
    export_to_csv(df_original, str(file_path))
    
    # Check if file was created
    assert os.path.exists(file_path)
    
    # Load data back
    df_loaded = load_from_csv(str(file_path))
    
    # Check that the loaded data is correct
    assert_frame_equal(df_original, df_loaded)

def test_load_nonexistent_csv(capsys):
    """
    Tests that loading a non-existent CSV file returns None and prints an error.
    """
    df = load_from_csv("non_existent_file.csv")
    assert df is None
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.out

# This warning seems to be triggered by numba via clifford.
@pytest.mark.filterwarnings("ignore:The 'nopython' keyword argument is not supported")
def test_plot_metric_component_smoke_test(monkeypatch):
    """
    Smoke test for plot_metric_component to ensure it runs without errors.
    This test mocks plt.show() to prevent it from blocking execution.
    """
    # Mock plt.show() to do nothing and prevent windows from popping up
    monkeypatch.setattr(plt, "show", lambda: None)
    
    # Setup a spacetime
    schwarzschild = PredefinedSpacetime('Schwarzschild')
    M = sympy.Symbol('M')
    
    # Define ranges for plotting
    t, r, theta, phi = schwarzschild.coords
    ranges = {
        t: 0,
        r: (2.1, 10.0),  # Avoid singularity at r=2M, assuming M=1
        theta: np.pi / 2,
        phi: (0, 2 * np.pi)
    }
    
    # The plotting function requires a fully numerical expression.
    # We substitute the symbolic mass M with a numerical value in the metric.
    metric_with_value = schwarzschild.metric.subs({M: 1.0})
    schwarzschild.metric = metric_with_value

    # Call the function and expect no exceptions
    try:
        plot_metric_component(schwarzschild, component=(0, 0), ranges=ranges)
    except Exception as e:
        pytest.fail(f"plot_metric_component raised an exception: {e}")

def test_plot_metric_component_error_handling(capsys):
    """
    Tests the error handling of the plot_metric_component function.
    """
    schwarzschild = PredefinedSpacetime('Schwarzschild')
    
    # Test with no ranges
    plot_metric_component(schwarzschild)
    captured = capsys.readouterr()
    assert "Error: 'ranges' dictionary must be provided" in captured.out
    
    # Test with incorrect number of plot variables (1 instead of 2)
    t, r, theta, phi = schwarzschild.coords
    ranges_wrong = {
        t: 0,
        r: (2.1, 10.0),
        theta: np.pi / 2,
        phi: 0
    }
    plot_metric_component(schwarzschild, ranges=ranges_wrong)
    captured = capsys.readouterr()
    assert "Error: Exactly two coordinates must be given a range to plot." in captured.out

    # Test with missing coordinate in ranges
    ranges_missing = {
        t: 0,
        r: (2.1, 10.0),
        theta: (0, np.pi)
    }
    plot_metric_component(schwarzschild, ranges=ranges_missing)
    captured = capsys.readouterr()
    assert "Error: 'ranges' keys must exactly match spacetime coordinates" in captured.out
