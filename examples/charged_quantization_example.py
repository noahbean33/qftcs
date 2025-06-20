import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ChargedScalarField
from aqft_curved.quantization import CanonicalQuantization

def charged_quantization_example():
    """
    An example demonstrating the canonical quantization of a charged scalar field.
    """
    print("--- Charged Canonical Quantization Example ---")

    # 1. Initialize Minkowski spacetime and a charged scalar field
    print("\nInitializing Minkowski spacetime and a charged scalar field...")
    minkowski = PredefinedSpacetime("Minkowski")
    t, x, y, z = minkowski.coords
    phi = ChargedScalarField(minkowski, 'psi', 'm', 'q')

    print(f"Spacetime Coordinates: {minkowski.coords}")
    print(f"Charged Scalar Field Symbol: {phi.name}")

    # 2. Perform canonical quantization
    quantization = CanonicalQuantization(phi)

    # 3. Compute the conjugate momenta
    print("\nComputing the conjugate momenta (pi, pi_dag)...")
    pi, pi_dag = quantization.conjugate_momentum()
    print("Conjugate Momentum (pi):")
    sympy.pprint(pi, use_unicode=False)
    print("\nConjugate Momentum (pi_dag):")
    sympy.pprint(pi_dag, use_unicode=False)

    # 4. Compute the equal-time commutation relations
    print("\nComputing the equal-time commutation relations...")
    comm_relations = quantization.commutation_relations()

    print("\nCommutation Relations:")
    print("1. [psi(t, x), pi(t, y)] =")
    sympy.pprint(comm_relations[0], use_unicode=False)
    print("\n2. [psi_dag(t, x), pi_dag(t, y)] =")
    sympy.pprint(comm_relations[1], use_unicode=False)
    print("\n3. [psi(t, x), psi(t, y)] =")
    sympy.pprint(comm_relations[2], use_unicode=False)
    print("\n4. [pi(t, x), pi(t, y)] =")
    sympy.pprint(comm_relations[3], use_unicode=False)
    print("\n5. [psi(t, x), psi_dag(t, y)] =")
    sympy.pprint(comm_relations[4], use_unicode=False)
    print("\n6. [pi(t, x), pi_dag(t, y)] =")
    sympy.pprint(comm_relations[5], use_unicode=False)

    print("\n--- Charged Canonical Quantization Example Complete ---")

if __name__ == "__main__":
    charged_quantization_example()
