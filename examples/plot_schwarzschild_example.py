import numpy as np
from aqft.spacetime import PredefinedSpacetime
from aqft.utils import plot_metric_component

def run_plot_schwarzschild_example():
    """
    Demonstrates the metric plotting functionality for a non-trivial spacetime.
    """
    # 1. Define Schwarzschild spacetime with a specific mass M
    # The symbolic 'M' must be replaced with a numerical value for plotting.
    M = 1.0
    schwarzschild = PredefinedSpacetime('Schwarzschild', M=M)
    print(f"Initialized Schwarzschild spacetime with M = {M}")

    # 2. Define the slice to plot. We'll visualize the g_tt component
    # on the equatorial plane (theta = pi/2) as a function of r and phi.
    t, r, theta, phi = schwarzschild.coords
    
    plot_ranges = {
        t: 0,                      # Fixed time
        r: (2 * M + 0.1, 10 * M), # Radial coordinate from just outside the horizon
        theta: (0, np.pi),         # Plotting vs. theta
        phi: 0                     # Fixed phi angle
    }

    # 3. Plot the g_rr component (1,1)
    print("\nPlotting the g_rr metric component...")
    plot_metric_component(
        schwarzschild, 
        component=(1, 1), 
        ranges=plot_ranges, 
        n_points=100
    )

    # 4. Plot the g_tt component (0,0)
    print("\nPlotting the g_tt metric component...")
    plot_metric_component(
        schwarzschild, 
        component=(0, 0), 
        ranges=plot_ranges, 
        n_points=100
    )

if __name__ == "__main__":
    run_plot_schwarzschild_example()
