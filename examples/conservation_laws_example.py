import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_conservation_laws_example():
    """
    Demonstrates the conservation of the Einstein tensor in Schwarzschild spacetime.
    This serves as a validation of the `einstein_tensor` and `covariant_divergence` methods.
    """
    print("--- Conservation Laws Example (Einstein Tensor in Schwarzschild) ---")

    # 1. Initialize Schwarzschild Spacetime
    # -------------------------------------
    print("\nInitializing Schwarzschild spacetime...")
    # Use a symbolic mass 'M' for generality
    M = sympy.Symbol('M')
    schwarzschild_spacetime = PredefinedSpacetime('Schwarzschild', M=M)
    print(f"Spacetime Coordinates: {schwarzschild_spacetime.coords}")

    # 2. Compute the Einstein Tensor (Covariant)
    # ------------------------------------------
    print("\nComputing the covariant Einstein tensor G_munu...")
    G_munu = schwarzschild_spacetime.einstein_tensor()
    # For a vacuum solution like Schwarzschild, the Einstein tensor should be zero.
    # Our symbolic computation should confirm this.
    print("Einstein Tensor G_munu:")
    print(str(G_munu))

    # 3. Compute the Contravariant Einstein Tensor
    # --------------------------------------------
    print("\nComputing the contravariant Einstein tensor G^munu...")
    g_inv = schwarzschild_spacetime.inverse_metric
    G_con_immutable = g_inv * G_munu * g_inv

    # Create a mutable matrix to store the simplified result
    dim = schwarzschild_spacetime.dimension
    G_con = sympy.MutableDenseMatrix.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            G_con[i, j] = sympy.simplify(G_con_immutable[i, j])
    print("Contravariant Einstein Tensor G^munu:")
    print(str(G_con))

    # 4. Compute the Covariant Divergence of the Einstein Tensor
    # --------------------------------------------------------
    print("\nComputing the covariant divergence of the Einstein tensor (nabla_mu G^munu)...")
    div_G = schwarzschild_spacetime.covariant_divergence(G_con)
    print("Covariant Divergence of G^munu:")
    print(str(div_G))

    # Verification
    is_conserved = all(c == 0 for c in div_G)
    print(f"\nIs the Einstein tensor conserved (nabla_mu G^munu == 0)? {is_conserved}")
    if not is_conserved:
        print("Warning: The Einstein tensor does not appear to be conserved. Check the implementation.")

    print("\n--- Conservation Laws Example Complete ---")

from aqft_curved.field import ScalarField

def run_stress_energy_conservation_example():
    """
    Demonstrates the on-shell conservation of the stress-energy tensor for a scalar field.
    """
    print("\n--- Conservation Laws Example (Stress-Energy Tensor) ---")

    # 1. Initialize Minkowski Spacetime
    # ---------------------------------
    print("\nInitializing 4D Minkowski spacetime...")
    minkowski_spacetime = PredefinedSpacetime('Minkowski')

    # 2. Define a Scalar Field with mass and coupling
    # -----------------------------------------------
    m = sympy.Symbol('m')
    xi = sympy.Symbol('xi')
    scalar_field = ScalarField(spacetime=minkowski_spacetime, mass=m, coupling_xi=xi)
    print(f"Defined scalar field with mass={m} and coupling={xi}")

    # 3. Compute the Covariant Divergence of the Stress-Energy Tensor
    # ---------------------------------------------------------------
    print("\nComputing the contravariant stress-energy tensor T^munu...")
    T_con = scalar_field.stress_energy_tensor_contravariant()

    print("\nComputing the covariant divergence (nabla_mu T^munu)...")
    div_T = minkowski_spacetime.covariant_divergence(T_con)
    print("Covariant Divergence of T^munu:")
    print(str(div_T))

    # 4. Verify On-Shell Conservation
    # -------------------------------
    # The divergence should be zero when the equation of motion is satisfied.
    # Let's get the equation of motion.
    eom = scalar_field.equation_of_motion().lhs
    print(f"\nEquation of Motion (should be zero): {eom}")

    # The divergence of T^munu is proportional to the EOM.
    # For a scalar field, nabla_mu T^munu = - (nabla^nu phi) * (EOM)
    # Let's check if each component of the divergence is zero when EOM is zero.
    # We can do this by substituting eom=0 into the divergence expression.
    # This is tricky with symbolic derivatives, but we can see the proportionality.
    
    # Let's test the first component of the divergence vector.
    # It should contain the EOM as a factor.
    div_T_simplified = sympy.simplify(div_T)
    print("\nSimplified Divergence of T^munu:")
    print(str(div_T_simplified))
    
    # A full symbolic check can be complex. For Minkowski space, R=0, so EOM is (Box + m^2)phi = 0.
    # The divergence should simplify to a form that is clearly proportional to the EOM.
    # For example, the first component of the divergence vector should be (d^0 phi) * EOM.
    dphi_con = minkowski_spacetime.inverse_metric * sympy.Matrix([sympy.diff(scalar_field.name, c) for c in minkowski_spacetime.coords])
    expected_div = dphi_con * eom

    # We will check if the difference is zero
    diff = sympy.simplify(div_T_simplified - expected_div)
    is_conserved_on_shell = all(c == 0 for c in diff)

    print(f"\nIs the stress-energy tensor conserved on-shell? {is_conserved_on_shell}")
    if not is_conserved_on_shell:
        print("Warning: The stress-energy tensor does not appear to be conserved on-shell.")

    print("\n--- Stress-Energy Conservation Example Complete ---")


if __name__ == "__main__":
    run_conservation_laws_example()
    run_stress_energy_conservation_example()
