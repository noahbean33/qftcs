"""
Numerical Quantum State Tutorial

This tutorial demonstrates how to work with numerical quantum states in the aqft_py library.
It covers:
1. Creating a numerical vacuum state.
2. Defining annihilation and creation operators.
3. Constructing a one-particle state.
4. Calculating expectation values of the number operator.
"""

import numpy as np
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import FieldOperator, ScalarField
from aqft_curved.state import VacuumState
from aqft_curved.algebra import AlgebraicProduct

def main():
    # Define a Hilbert space dimension
    # A larger dimension provides a better approximation of the infinite-dimensional Fock space.
    hilbert_dim = 5

    # 1. Setup the spacetime and a scalar field
    # Although the operators here are abstract, they are formally associated with a field.
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')

    # 2. Create a numerical vacuum state |0>
    vacuum = VacuumState(field=field, hilbert_dim=hilbert_dim)
    print(f"Created a numerical vacuum state in a Hilbert space of dimension {hilbert_dim}.")
    print("Vacuum state |0>:")
    print(vacuum.numerical_state)

    # 3. Define annihilation (a) and creation (a_dag) operators
    a = FieldOperator(name='a', spacetime=st, is_creation=False)
    a_dag = FieldOperator(name='a_dag', spacetime=st, is_creation=True)

    # Get their numerical representations
    a_num = a.to_numerical(hilbert_dim)
    a_dag_num = a_dag.to_numerical(hilbert_dim)

    print("\nAnnihilation operator a:")
    print(a_num)
    print("\nCreation operator a_dag:")
    print(a_dag_num)

    # 4. Construct the number operator N = a_dag * a
    number_operator = AlgebraicProduct([a_dag, a])

    # 5. Calculate the expectation value of N in the vacuum state
    # We expect <0|N|0> = 0
    exp_N_vacuum = vacuum.expectation_value(number_operator)
    print(f"\nExpectation value of Number Operator in vacuum <0|N|0> = {exp_N_vacuum.real:.4f}")

    # 6. Create a one-particle state |1> = a_dag |0>
    # We use the @ operator for matrix multiplication (operator application)
    one_particle_state_numerical = a_dag_num @ vacuum.numerical_state
    one_particle_state_numerical.normalize() # Normalize the state
    
    # Create a new State object for the one-particle state
    from aqft_curved.state import State
    one_particle_state = State(field=field, numerical_state=one_particle_state_numerical, hilbert_dim=hilbert_dim)

    print("\nOne-particle state |1> = a_dag|0>:")
    print(one_particle_state.numerical_state)

    # 7. Calculate the expectation value of N in the one-particle state
    # We expect <1|N|1> = 1
    exp_N_one_particle = one_particle_state.expectation_value(number_operator)
    print(f"\nExpectation value of Number Operator in one-particle state <1|N|1> = {exp_N_one_particle.real:.4f}")

if __name__ == "__main__":
    main()
