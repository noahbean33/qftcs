import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField
from aqft_curved.state import VacuumState

def run_state_example():
    """
    Demonstrates the functionality of the State and VacuumState classes.
    """
    # 1. Set up spacetime and field
    minkowski = PredefinedSpacetime('Minkowski')
    field = ScalarField(minkowski)
    print(f"Created scalar field {field.name} on Minkowski spacetime.")

    # 2. Initialize a vacuum state for the field
    vacuum = VacuumState(field, state_type='Hadamard')

    # 3. Compute the expectation value of a single field operator
    print("\nComputing expectation value of the field...")
    exp_value = vacuum.expectation_value(field)
    print(f"<| {field.name} |> = {exp_value}")
    assert exp_value == 0
    print("Assertion passed: One-point function is zero as expected.")

    # 4. Compute the symbolic two-point function
    print("\nComputing the symbolic two-point function...")
    coords = minkowski.coords
    x1 = sympy.symbols('t1, x1, y1, z1')
    x2 = sympy.symbols('t2, x2, y2, z2')
    two_point = vacuum.two_point_function(x1, x2)
    print(f"G(x1, x2) = {two_point}")

if __name__ == "__main__":
    run_state_example()
