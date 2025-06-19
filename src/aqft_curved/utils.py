"""
This module contains utility functions for visualization, data I/O,
and other helper tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy

def plot_metric_component(spacetime, component=(0, 0), ranges=None, n_points=50):
    """
    Visualizes a component of the metric tensor over a 2D slice of spacetime.

    Parameters:
        spacetime (Spacetime): The spacetime object with a defined metric.
        component (tuple): The (row, col) index of the metric component to plot.
        ranges (dict): A dictionary defining the plot slice. Keys should be the
                       coordinate symbols from spacetime.coords. Exactly two coordinates
                       should have a (min, max) tuple as their value, defining the plot axes.
                       The other coordinates must have a fixed numerical value.
                       Example: {{'t': 0, 'r': (2*M, 10*M), 'theta': np.pi/2, 'phi': (0, 2*np.pi)}}
        n_points (int): The number of points for the grid in each dimension.
    """
    if spacetime.metric is None:
        print("Spacetime metric is not set. Cannot plot.")
        return

    if ranges is None:
        print("Error: 'ranges' dictionary must be provided to define the plot slice.")
        return

    # Validate ranges and identify plot vs fixed coordinates
    plot_vars = []
    plot_ranges = []
    fixed_vars = {}
    
    all_coord_symbols = spacetime.coords
    
    # Check that all coordinates are present in ranges
    if set(all_coord_symbols) != set(ranges.keys()):
        print(f"Error: 'ranges' keys must exactly match spacetime coordinates: {{[c.name for c in all_coord_symbols]}}")
        return

    for var, value in ranges.items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            plot_vars.append(var)
            plot_ranges.append(value)
        else:
            fixed_vars[var] = value
            
    if len(plot_vars) != 2:
        print(f"Error: Exactly two coordinates must be given a range to plot. Found {{len(plot_vars)}}.")
        return

    # Get the symbolic metric component and substitute fixed values
    metric_comp_expr = spacetime.metric[component].subs(fixed_vars)

    # Lambdify the expression
    try:
        f = sympy.lambdify(plot_vars, metric_comp_expr, 'numpy')
    except Exception as e:
        print(f"Error during lambdify: {{e}}")
        print("This may happen if the expression contains symbolic parameters (e.g., 'M' for mass)")
        print("that were not substituted with numerical values in the spacetime object.")
        return

    # Create grid
    x_vals = np.linspace(plot_ranges[0][0], plot_ranges[0][1], n_points)
    y_vals = np.linspace(plot_ranges[1][0], plot_ranges[1][1], n_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluate metric component on the grid
    Z = f(X, Y)

    # Plotting
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    ax.set_title(f'Metric Component $g_{{{component[0]},{component[1]}}}$')
    ax.set_xlabel(str(plot_vars[0]))
    ax.set_ylabel(str(plot_vars[1]))
    fig.colorbar(c, ax=ax, label=f'Value of $g_{{{component[0]},{component[1]}}}$')
    plt.show()

def export_to_csv(data, filename):
    """
    Exports a dictionary or pandas DataFrame to a CSV file.

    Parameters:
        data (dict or pd.DataFrame): The data to export.
        filename (str): The path to the output CSV file.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Successfully exported data to {{filename}}")
    except Exception as e:
        print(f"Error exporting data to CSV: {{e}}")

def load_from_csv(filename):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
        filename (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The loaded data, or None if an error occurs.
    """
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded data from {{filename}}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {{filename}}")
        return None
    except Exception as e:
        print(f"Error loading data from CSV: {{e}}")
        return None
