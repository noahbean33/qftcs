import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField
from aqft_curved.quantization import CanonicalQuantization

def canonical_quantization_example():
    """
    An example to demonstrate canonical quantization by computing
    the conjugate momentum for a scalar field.
    """
    print("--- Canonical Quantization Example ---")

    # Initialize a 4D Minkowski spacetime
    minkowski = PredefinedSpacetime("Minkowski")
    print("\nInitializing Minkowski spacetime and a scalar field...")
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # Initialize the scalar field
    scalar_field = ScalarField(minkowski, mass=sympy.Symbol('m'))
    print(f"Scalar Field Symbol: {scalar_field.name}")

    # Perform canonical quantization
    quantization = CanonicalQuantization(scalar_field)

    # Compute the conjugate momentum
    print("\nComputing the conjugate momentum pi...")
    pi = quantization.conjugate_momentum()

    print("\nConjugate Momentum (pi):")
    sympy.pprint(pi, use_unicode=False)

    # Compute the commutation relations
    print("\nComputing the equal-time commutation relations...")
    comm_relations = quantization.commutation_relations()

    print("\nCommutation Relations:")
    print("1. [phi(t, x), pi(t, y)] =")
    sympy.pprint(comm_relations[0], use_unicode=False)
    print("\n2. [phi(t, x), phi(t, y)] =")
    sympy.pprint(comm_relations[1], use_unicode=False)
    print("\n3. [pi(t, x), pi(t, y)] =")
    sympy.pprint(comm_relations[2], use_unicode=False)

    # Display the mode expansion
    print("\nComputing the mode expansion for the scalar field...")
    mode_exp = quantization.mode_expansion()
    print("Mode Expansion (phi(x) = ...):")
    sympy.pprint(mode_exp, use_unicode=False)

    # Display the creation and annihilation operator commutation relations
    print("\nComputing the commutation relations for creation and annihilation operators...")
    ca_comm_relations = quantization.creation_annihilation_commutation()
    print("\nCreation/Annihilation Commutation Relations:")
    print("1. [a(k), a_dag(p)] =")
    sympy.pprint(ca_comm_relations[0], use_unicode=False)
    print("\n2. [a(k), a(p)] =")
    sympy.pprint(ca_comm_relations[1], use_unicode=False)
    print("\n3. [a_dag(k), a_dag(p)] =")
    sympy.pprint(ca_comm_relations[2], use_unicode=False)

    # Display the mode expansion for the conjugate momentum
    print("\nComputing the mode expansion for the conjugate momentum...")
    pi_mode_exp = quantization.pi_mode_expansion()
    print("Pi Mode Expansion (pi(x) = ...):")
    sympy.pprint(pi_mode_exp, use_unicode=False)

    # Display the Hamiltonian
    print("\nComputing the Hamiltonian...")
    hamiltonian = quantization.hamiltonian()
    print("Hamiltonian (H = ...):")
    sympy.pprint(hamiltonian, use_unicode=False)

    print("\n--- Canonical Quantization Example Complete ---")

if __name__ == "__main__":
    canonical_quantization_example()
