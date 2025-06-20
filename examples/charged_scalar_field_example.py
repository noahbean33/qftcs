import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ElectromagneticField, ChargedScalarField

def run_charged_scalar_field_example():
    """
    Demonstrates the equation of motion for a charged scalar field coupled to an electromagnetic field.
    """
    print("--- Charged Scalar Field Example ---")

    # 1. Initialize Spacetime and Fields
    # ----------------------------------
    print("\nInitializing Minkowski spacetime, an EM field, and a charged scalar field...")
    minkowski = PredefinedSpacetime('Minkowski')
    em_field = ElectromagneticField(minkowski)
    
    # Define parameters for the scalar field
    m = sympy.Symbol('m', positive=True)  # Mass
    q = sympy.Symbol('q', real=True)      # Charge

    charged_field = ChargedScalarField(spacetime=minkowski, mass=m, charge=q, em_field=em_field)

    print(f"Spacetime Coordinates: {minkowski.coords}")
    print(f"Scalar Field Symbol: {charged_field.name}")
    print(f"EM Field 4-Potential: {em_field.potential.T}") # Transpose for display

    # 2. Compute the Equation of Motion
    # ---------------------------------
    print("\nComputing the equation of motion (D_mu D^mu - m^2) * phi = 0...")
    # In Minkowski space, xi*R term is zero.
    eom = charged_field.equation_of_motion()

    print("\nEquation of Motion for the Charged Scalar Field (0 =):")
    # The expression is large, so we display it.
    print(eom.lhs)

    print("\n--- Charged Scalar Field Example Complete ---")

if __name__ == "__main__":
    run_charged_scalar_field_example()
