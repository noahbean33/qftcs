import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField

def run_scalar_field_example():
    """
    Demonstrates the functionality of the ScalarField class.
    """
    # 1. Initialize a predefined spacetime (Minkowski)
    print("Initializing Minkowski spacetime...")
    minkowski = PredefinedSpacetime('Minkowski')
    print("Spacetime initialized.")

    # 2. Define a scalar field on this spacetime
    # We'll use a massless (m=0) and minimally coupled (xi=0) field.
    print("\nDefining a massless, minimally coupled scalar field...")
    field = ScalarField(minkowski, mass=0, coupling_xi=0)
    print(f"Field defined: {field.name}")

    # 3. Compute the equation of motion
    print("\nCalculating the Klein-Gordon equation...")
    eom = field.equation_of_motion()

    # 4. Print the result
    print("\nEquation of motion for the scalar field in Minkowski space:")
    sympy.pprint(eom, use_unicode=False)

    # Expected: Box(phi) = 0, which is -d^2/dt^2(phi) + nabla^2(phi) = 0

if __name__ == "__main__":
    run_scalar_field_example()
