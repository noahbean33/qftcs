import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_kerr_example():
    """
    Demonstrates the properties of Kerr spacetime.
    """
    print("--- Kerr Spacetime Example ---")

    # 1. Initialize Kerr Spacetime
    # ----------------------------
    print("\nInitializing Kerr spacetime...")
    # Use symbolic mass 'M' and spin parameter 'a'
    M = sympy.Symbol('M', positive=True)
    a = sympy.Symbol('a', real=True)
    kerr_spacetime = PredefinedSpacetime('kerr', M=M, a=a)
    print(f"Spacetime Coordinates: {kerr_spacetime.coords}")
    print("Metric g_munu (showing a few components due to complexity):")
    print(f"g_tt = {kerr_spacetime.metric[0, 0]}")
    print(f"g_rr = {kerr_spacetime.metric[1, 1]}")
    print(f"g_tphi = {kerr_spacetime.metric[0, 3]}")


    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For Kerr spacetime, the Ricci scalar should be zero.
    # This is a very intensive calculation and may take a long time.
    print("\nComputing the Ricci scalar R...")
    print("(This is a very intensive calculation and may take several minutes)")
    ricci_scalar = kerr_spacetime.ricci_scalar()
    print(f"Ricci Scalar (before simplification): {ricci_scalar}")

    # Verification
    print("\nSimplifying the Ricci scalar...")
    simplified_ricci = sympy.simplify(ricci_scalar)
    is_correct = (simplified_ricci == 0)
    print(f"\nSimplified Ricci Scalar: {simplified_ricci}")
    print(f"Is the Ricci scalar correct (R = 0)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar is non-zero, which is unexpected for Kerr spacetime.")

    print("\n--- Kerr Example Complete ---")

if __name__ == "__main__":
    run_kerr_example()
