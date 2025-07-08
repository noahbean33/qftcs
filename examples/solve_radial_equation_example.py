import numpy as np
import matplotlib.pyplot as plt
from aqft.field import ScalarField
from aqft.spacetime import PredefinedSpacetime

def solve_and_plot_radial_equation():
    """
    This example demonstrates how to use the `solve_radial_equation` method
    of the ScalarField class to find and plot a numerical solution for the
    radial part of the Klein-Gordon equation in Schwarzschild spacetime.
    """
    print("--- Numerical Radial Equation Solver Example ---")

    # 1. Define Schwarzschild spacetime with M=1
    M = 1
    schwarzschild_spacetime = PredefinedSpacetime(name='Schwarzschild', M=M)
    print(f"Initialized Schwarzschild spacetime with M = {M}")

    # 2. Define a massless scalar field
    scalar_field = ScalarField(spacetime=schwarzschild_spacetime, name='phi', mass=0.0)
    print("Defined a massless scalar field 'phi'.")

    # 3. Set parameters for the radial equation
    omega = 0.5  # Frequency of the mode
    l = 1        # Angular momentum quantum number
    r_start = 2.1 * M  # Start integration just outside the event horizon
    r_end = 20 * M     # End integration at a larger radius
    num_points = 2000

    # Set initial conditions for R(r) and R'(r) at r_start
    # These are arbitrary for demonstration; physical solutions require specific boundary conditions.
    initial_conditions = [1.0, 0.0]  # R(r_start) = 1, R'(r_start) = 0

    print(f"Solving radial equation for omega={omega}, l={l} from r={r_start} to r={r_end}")

    # 4. Solve the radial equation
    try:
        r_vals, R_vals = scalar_field.solve_radial_equation(
            omega=omega,
            l=l,
            r_start=r_start,
            r_end=r_end,
            initial_conditions=initial_conditions,
            num_points=num_points
        )

        # 5. Plot the solution
        plt.figure(figsize=(10, 6))
        plt.plot(r_vals, R_vals, label=fr'$\omega={omega}, l={l}$')
        plt.title('Numerical Solution of the Radial Wave Equation $R(r)$')
        plt.xlabel('Radial Coordinate $r/M$')
        plt.ylabel('Radial Function $R(r)$')
        plt.grid(True)
        plt.legend()
        plt.axvline(x=2*M, color='r', linestyle='--', label='Event Horizon (2M)')
        plt.legend()
        print("Plot generated. Please close the plot window to exit.")
        plt.show()

    except Exception as e:
        print(f"An error occurred during calculation or plotting: {e}")

    print("--- Example Complete ---")

if __name__ == "__main__":
    solve_and_plot_radial_equation()
