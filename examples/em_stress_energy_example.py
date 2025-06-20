import sympy
from aqft_curved.spacetime import Spacetime, PredefinedSpacetime
from aqft_curved.field import ElectromagneticField

def em_stress_energy_example():
    """
    An example to demonstrate the computation of the stress-energy tensor
    for the electromagnetic field in Minkowski spacetime.
    """
    print("--- Electromagnetic Field Stress-Energy Tensor Example ---")

    # Initialize a 4D Minkowski spacetime
    minkowski = PredefinedSpacetime("Minkowski")
    print("\nInitializing Minkowski spacetime and an EM field...")
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # Initialize the EM field. The 4-potential is created symbolically by default.
    em_field = ElectromagneticField(minkowski)
    print(f"EM Field 4-Potential: {em_field.name.T}")

    # Compute the stress-energy tensor
    print("\nComputing the stress-energy tensor T_munu...")
    T_munu = em_field.stress_energy_tensor()

    print("\nStress-Energy Tensor for the Electromagnetic Field (T_munu):")
    sympy.pprint(T_munu, use_unicode=False)

    print("\n--- Electromagnetic Field Stress-Energy Tensor Example Complete ---")

if __name__ == "__main__":
    em_stress_energy_example()
