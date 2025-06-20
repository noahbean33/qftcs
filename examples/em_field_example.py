import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ElectromagneticField

def run_em_field_example():
    """
    Demonstrates the basic properties of the ElectromagneticField class.
    """
    print("--- Electromagnetic Field Example ---")

    # 1. Initialize a Spacetime
    # --------------------------
    print("\nInitializing Minkowski spacetime...")
    minkowski = PredefinedSpacetime('Minkowski')
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # 2. Initialize the Electromagnetic Field
    # ---------------------------------------
    print("\nInitializing the electromagnetic field...")
    em_field = ElectromagneticField(minkowski)
    print("Symbolic 4-potential A_mu:")
    print(em_field.potential)

    # 3. Compute the Field Strength Tensor
    # ------------------------------------
    print("\nComputing the field strength tensor F_munu = d_mu(A_nu) - d_nu(A_mu)...")
    F_munu = em_field.field_strength_tensor()
    print("Field Strength Tensor F_munu:")
    print(F_munu)

    # The components of F_munu correspond to electric (E) and magnetic (B) fields.
    # For example, in Cartesian coordinates (t, x, y, z):
    # E_x = F_10, E_y = F_20, E_z = F_30
    # B_x = F_32, B_y = F_13, B_z = F_21
    t, x, y, z = minkowski.coords
    A_t, A_x, A_y, A_z = em_field.potential
    print("\nExample components:")
    print(f"E_x = F_10 = {F_munu[1, 0]}")
    print(f"B_x = F_32 = {F_munu[3, 2]}")

    # Verify anti-symmetry: F_munu = -F_numu
    is_antisymmetric = sympy.simplify(F_munu + F_munu.T) == sympy.zeros(4)
    print(f"\nIs the tensor anti-symmetric (F_munu = -F_numu)? {is_antisymmetric}")

    # 4. Compute the Equation of Motion
    # ---------------------------------
    print("\nComputing the equation of motion (nabla_mu F^munu = 0)...")
    eom = em_field.equation_of_motion()
    print("Maxwell's Equations in vector form (0 =):")
    print(eom.lhs)  # eom is Eq(lhs, 0), so we print the left-hand side

    # 5. Compute the Stress-Energy Tensor
    # -----------------------------------
    print("\nComputing the stress-energy tensor T_munu...")
    T_munu = em_field.stress_energy_tensor()
    print("Stress-Energy Tensor T_munu (showing T_00 component):")
    # The full tensor is very large, so we'll just display one component.
    # T_00 represents the energy density.
    print(f"T_00 = {T_munu[0, 0]}")

    print("\n--- Electromagnetic Field Example Complete ---")

if __name__ == "__main__":
    run_em_field_example()
