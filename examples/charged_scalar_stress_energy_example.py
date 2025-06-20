import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ChargedScalarField, ElectromagneticField

def charged_scalar_stress_energy_example():
    """
    An example to demonstrate the computation of the stress-energy tensor
    for a charged scalar field coupled to an electromagnetic field.
    """
    print("--- Charged Scalar Field Stress-Energy Tensor Example ---")

    # Initialize a 4D Minkowski spacetime
    minkowski = PredefinedSpacetime("Minkowski")
    print("\nInitializing Minkowski spacetime, an EM field, and a charged scalar field...")
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # Initialize the fields
    em_field = ElectromagneticField(minkowski)
    charged_field = ChargedScalarField(minkowski, mass=sympy.Symbol('m'), charge=sympy.Symbol('q'), em_field=em_field)
    print(f"Charged Scalar Field Symbol: {charged_field.name}")
    print(f"EM Field 4-Potential: {em_field.name.T}")

    # Compute the stress-energy tensor
    print("\nComputing the stress-energy tensor T_munu for the charged scalar field...")
    T_munu = charged_field.stress_energy_tensor()

    print("\nStress-Energy Tensor for the Charged Scalar Field (T_munu):")
    sympy.pprint(T_munu, use_unicode=False)

    print("\n--- Charged Scalar Field Stress-Energy Tensor Example Complete ---")

if __name__ == "__main__":
    charged_scalar_stress_energy_example()
