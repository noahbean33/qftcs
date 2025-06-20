"""
Tutorial: Scalar Field on FLRW Spacetime

This script demonstrates the definition of a massless scalar field
on a Friedmann-Lemaître-Robertson-Walker (FLRW) spacetime and computes
its Klein-Gordon equation. It also visualizes a component of the FLRW metric.
"""

import sympy
from sympy.printing import pretty_print

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField
from aqft_curved.utils import plot_metric_component

# Enable Unicode printing for SymPy in Windows console if needed
from sympy.interactive import init_printing
init_printing(use_unicode=True)

def run_flrw_scalar_field_tutorial():
    """Runs the FLRW scalar field tutorial."""

    print("--- FLRW Scalar Field Tutorial ---")

    # 1. Define FLRW Spacetime
    # -------------------------
    # Define symbolic time coordinate and scale factor a(t)
    t_sym = sympy.symbols('t')
    # For this example, let's use a simple scale factor a(t) = t (matter-dominated universe, simplified)
    # For a radiation-dominated universe, one might use a(t) = sqrt(t), etc.
    # For de Sitter, a(t) = exp(H*t)
    scale_factor_a = t_sym

    # Initialize FLRW spacetime with k=0 (flat spatial curvature) and our defined a(t)
    # The PredefinedSpacetime class expects 'a' as a sympy.Function or expression involving 't'.
    print(f"\nInitializing FLRW spacetime with scale factor a(t) = {scale_factor_a} and k=0...")
    flrw_spacetime = PredefinedSpacetime('FLRW', a=scale_factor_a, k=0)

    print("FLRW Metric Tensor:")
    print(str(flrw_spacetime.metric))

    # 2. Define a Massless Scalar Field
    # ---------------------------------
    print("\nDefining a massless scalar field on FLRW spacetime...")
    # The field symbol will be phi(t, x, y, z)
    scalar_field = ScalarField(flrw_spacetime, mass=0)

    print(f"Scalar field symbol: {scalar_field.name}")

    # 3. Compute Klein-Gordon Equation
    # --------------------------------
    print("\nComputing the Klein-Gordon equation (Box phi = 0) for the massless scalar field...")
    kg_equation = scalar_field.equation_of_motion()
    print("Klein-Gordon Equation (Box phi):")
    print(str(kg_equation))
    print("(Should be equal to 0 for a solution)")

    # 4. Plot a Metric Component (Optional)
    # -------------------------------------
    # Let's plot the g_xx component. For FLRW with k=0, g_xx = a(t)^2 = t^2.
    # We need to choose a slice. Let's plot g_xx(t, x) for fixed y=0, z=0.
    # The component g_xx only depends on t.
    print("\nPlotting the g_xx metric component...")
    # Define the coordinate ranges for the plot
    # x_coord is 'x', y_coord is 't' for the plot axes
    # The actual component g_xx = metric[1,1] in (t,x,y,z) order
    # The plot_metric_component function expects a 'ranges' dictionary where:
    # - keys are the coordinate symbols
    # - values for plot axes are (min, max) tuples
    # - values for fixed coordinates are numbers
    ranges_dict = {
        flrw_spacetime.coords[0]: (1, 5),  # t: range (1, 5) - y-axis of plot
        flrw_spacetime.coords[1]: (0, 5),  # x: range (0, 5) - x-axis of plot
        flrw_spacetime.coords[2]: 0,       # y: fixed at 0
        flrw_spacetime.coords[3]: 0        # z: fixed at 0
    }

    plot_metric_component(
        spacetime=flrw_spacetime,
        component=(1, 1),  # g_xx
        ranges=ranges_dict,
        # title=f"FLRW Metric Component g_xx(x, t) for a(t)={scale_factor_a}, k=0", # Title is now auto-generated
        n_points=50
    )
    print("Plot for g_xx generated. Please close the plot window to continue.")

    print("\n--- Tutorial Complete ---")

if __name__ == "__main__":
    run_flrw_scalar_field_tutorial()
