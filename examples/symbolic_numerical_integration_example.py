"""
This example demonstrates the integration between the symbolic and numerical frameworks.
"""
import numpy as np
import sympy

from aqft_curved.spacetime import Spacetime
from aqft_curved.field import FieldOperator, ScalarField
from aqft_curved.algebra import AlgebraicProduct
from aqft_curved.state import VacuumState, State
from aqft_curved.quantum_state import basis

def main():
    print("--- Symbolic-Numerical Integration Example ---")

    # --- 1. Setup the Spacetime and Field ---
    # A simple 1D spacetime is sufficient as the structure is what matters
    spacetime = Spacetime(dimension=1, coordinates=('t',))
    phi = ScalarField(spacetime, 'phi')
    hilbert_dim = 4
    print(f"Hilbert space dimension: {hilbert_dim}")

    # --- 2. Create a State with Numerical Backend ---
    # Create a symbolic VacuumState, but also give it a numerical dimension
    vacuum = VacuumState(field=phi, hilbert_dim=hilbert_dim)
    print(f"Symbolic state has numerical backend: {vacuum.has_numerical}")

    # --- 3. Define Symbolic Operators ---
    # Define symbolic creation and annihilation operators
    a_dag = FieldOperator("a_dag", spacetime, is_creation=True)
    a = FieldOperator("a", spacetime, is_creation=False)
    print(f"\nSymbolic operators defined: {a_dag}, {a}")

    # --- 4. Define a Product of Operators (Number Operator) ---
    # The number operator N = a_dag * a
    number_op_symbolic = AlgebraicProduct([a_dag, a])
    print(f"Symbolic number operator N = {number_op_symbolic}")

    # --- 5. Calculate Expectation Value ---
    # This will use the numerical backend automatically
    exp_N_vacuum = vacuum.expectation_value(number_op_symbolic)
    print(f"Expectation value <0|N|0>: {np.real(exp_N_vacuum):.1f}")

    # --- 6. Test on an Excited State ---
    # Create a numerical one-particle state |1>
    one_particle_state_numerical = basis(hilbert_dim, 1)

    # Wrap it in a symbolic State object
    one_particle_state = State(
        field=phi,
        numerical_state=one_particle_state_numerical,
        hilbert_dim=hilbert_dim
    )
    print(f"\nCreated symbolic wrapper for numerical state |1>")

    exp_N_one_particle = one_particle_state.expectation_value(number_op_symbolic)
    print(f"Expectation value <1|N|1>: {np.real(exp_N_one_particle):.1f}")

if __name__ == "__main__":
    main()
