import sympy
from sympy import Eq, Symbol
from sympy.physics.quantum import Commutator

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import FieldOperator
from aqft_curved.algebra import OperatorAlgebra

def run_algebra_example():
    """
    Demonstrates the functionality of the OperatorAlgebra class.
    """
    # 1. Define a dummy spacetime (not strictly needed for this abstract example)
    minkowski = PredefinedSpacetime('Minkowski')

    # 2. Create symbolic operators
    phi = FieldOperator(Symbol('phi', commutative=False), minkowski)
    pi = FieldOperator(Symbol('pi', commutative=False), minkowski)
    print(f"Created operators: {phi} and {pi}")

    # 3. Define the canonical commutation relation (CCR)
    # [phi, pi] = i
    ccr = Eq(Commutator(phi.name, pi.name), sympy.I)
    print("\nDefining relation:")
    sympy.pprint(ccr, use_unicode=False)

    # 4. Create an algebra with these operators and relations
    algebra = OperatorAlgebra(generators=[phi, pi], relations=[ccr])
    print("\nOperatorAlgebra created.")

    # 5. Compute commutators using the algebra
    print("\nComputing commutators...")
    comm_phi_pi = algebra.commutator(phi, pi)
    comm_pi_phi = algebra.commutator(pi, phi)
    comm_phi_phi = algebra.commutator(phi, phi)

    # 6. Print the results
    print(f"[phi, pi] = {comm_phi_pi}")
    print(f"[pi, phi] = {comm_pi_phi}")
    print(f"[phi, phi] = {comm_phi_phi}")

    # Verify results
    assert comm_phi_pi == sympy.I
    assert comm_pi_phi == -sympy.I
    assert comm_phi_phi == 0
    print("\nAll assertions passed.")

if __name__ == "__main__":
    run_algebra_example()
