import sympy
import numpy as np
import matplotlib.pyplot as plt

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField
from aqft_curved.utils import plot_metric_component

def run_schwarzschild_scalar_field_tutorial():
    """
    Runs the Schwarzschild scalar field tutorial.
    """
    print("--- Schwarzschild Scalar Field Tutorial ---")

    # Define Schwarzschild mass M (can be symbolic or numerical)
    M_symbol = sympy.symbols('M')
    # For actual calculations like plotting or numerical solutions, M needs a value.
    # We'll use M=1 for this tutorial when plotting.
    M_value = 1 

    print(f"\nInitializing Schwarzschild spacetime with mass M = {M_value}...")

    # 1. Initialize Schwarzschild Spacetime
    # -------------------------------------
    # The PredefinedSpacetime class handles symbolic M internally and substitutes it if a numerical value is given.
    # For symbolic calculations (like the EOM), M can remain symbolic.
    # For plotting, we need to ensure M is substituted.
    schwarzschild_spacetime_symbolic_M = PredefinedSpacetime('Schwarzschild', M=M_symbol)
    schwarzschild_spacetime_numeric_M = PredefinedSpacetime('Schwarzschild', M=M_value)

    print("Schwarzschild Metric Tensor (symbolic M):")
    # Using str() to avoid potential Unicode issues on some consoles
    print(str(schwarzschild_spacetime_symbolic_M.metric.subs(M_symbol, M_value))) # Display with M=1 for compactness

    # 2. Define a Massless Scalar Field
    # ---------------------------------
    print("\nDefining a massless scalar field on Schwarzschild spacetime...")
    # We use the spacetime with symbolic M for deriving the EOM, as it's more general.
    # The scalar field 'phi' will depend on (t, r, theta, phi)
    # The name 'phi' is assigned by default within the ScalarField constructor.
    scalar_field = ScalarField(spacetime=schwarzschild_spacetime_symbolic_M, mass=0)
    print(f"Scalar field symbol: {scalar_field.name}")

    # 3. Compute Klein-Gordon Equation
    # --------------------------------
    print("\nComputing the Klein-Gordon equation (Box phi = 0) for the massless scalar field...")
    # This will be symbolic in M
    kg_equation = scalar_field.equation_of_motion()
    print("Klein-Gordon Equation (Box phi) (symbolic M):")
    print(str(kg_equation))
    print("Substituting M=1 for a more specific form:")
    print(str(kg_equation.subs(M_symbol, M_value)))
    print("(Should be equal to 0 for a solution)")

    # 4. Plot a Metric Component (Optional)
    # -------------------------------------
    print("\nPlotting the g_tt metric component...")
    # We will plot the g_tt component (index (0,0))
    # as a function of 'r' and 'theta', keeping 't' and 'phi' fixed.
    # We use the spacetime instance where M has been substituted with a numerical value.
    
    # Coordinates are (t, r, theta, phi)
    # Let's plot g_tt(r, theta) at t=0, phi=0
    # Ranges for r: (2M+epsilon to 10M), theta: (0 to pi)
    # Ensure r_min > 2M to avoid singularity
    r_min = 2 * M_value + 0.1 
    r_max = 10 * M_value

    ranges_dict = {
        schwarzschild_spacetime_numeric_M.coords[0]: 0,              # t = 0 (fixed)
        schwarzschild_spacetime_numeric_M.coords[1]: (r_min, r_max), # r: range
        schwarzschild_spacetime_numeric_M.coords[2]: (0, np.pi),     # theta: range
        schwarzschild_spacetime_numeric_M.coords[3]: 0               # phi = 0 (fixed)
    }

    plot_metric_component(
        spacetime=schwarzschild_spacetime_numeric_M,
        component=(0, 0),  # g_tt
        ranges=ranges_dict,
        n_points=50
    )
    print(f"Plot for g_tt generated with M={M_value}. Please close the plot window to continue.")

    print("\n--- Schwarzschild Scalar Field Tutorial Complete ---")

if __name__ == "__main__":
    run_schwarzschild_scalar_field_tutorial()
