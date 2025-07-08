import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft.spacetime import PredefinedSpacetime
from aqft.field import ScalarField

def run_stress_energy_tensor_example():
    """
    Demonstrates the calculation of the stress-energy tensor for a scalar field
    in Minkowski spacetime.
    """
    print("--- Stress-Energy Tensor Example (Scalar Field in Minkowski) ---")

    # 1. Initialize Minkowski Spacetime
    # ---------------------------------
    print("\nInitializing 4D Minkowski spacetime...")
    minkowski_spacetime = PredefinedSpacetime('Minkowski')
    print(f"Spacetime Coordinates: {minkowski_spacetime.coords}")
    print("Minkowski Metric Tensor:")
    print(str(minkowski_spacetime.metric))

    # 2. Define a Scalar Field
    # ------------------------
    # Define symbolic mass 'm' and coupling constant 'xi'
    m_sym = sympy.Symbol('m')
    xi_sym = sympy.Symbol('xi')

    print(f"\nDefining a scalar field with mass m={m_sym} and coupling xi={xi_sym}...")
    scalar_field = ScalarField(spacetime=minkowski_spacetime, name='phi', mass=m_sym, coupling_xi=xi_sym)
    print(f"Scalar field symbol: {scalar_field.name}")

    # 3. Compute Stress-Energy Tensor
    # -------------------------------
    print("\nComputing the stress-energy tensor T_munu...")
    T_munu = scalar_field.stress_energy_tensor()
    
    print("Stress-Energy Tensor T_munu:")
    # Using str() to avoid potential Unicode issues on some consoles for the matrix
    print(str(T_munu))

    # For Minkowski spacetime, R_munu = 0 and R = 0.
    # The formula simplifies to:
    # T_μν = (∂_μ φ)(∂_ν φ) - η_μν [1/2 ((∂_α φ)(∂^α φ) + m² φ²)]
    #          + ξ [- (∂_μ ∂_ν φ²) + η_μν (□ φ²)]
    # Note: ∂_μ ∂_ν φ² is not the same as ∇_μ ∇_ν φ² in general, but for Minkowski, Christoffels are zero.
    # The _second_covariant_derivative_scalar helper correctly handles this.

    print("\n--- Stress-Energy Tensor Example Complete ---")

if __name__ == "__main__":
    run_stress_energy_tensor_example()
