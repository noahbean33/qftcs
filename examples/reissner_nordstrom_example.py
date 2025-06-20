import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_reissner_nordstrom_example():
    """
    Demonstrates the properties of Reissner-Nordström spacetime.
    """
    print("--- Reissner-Nordström Spacetime Example ---")

    # 1. Initialize Reissner-Nordström Spacetime
    # -------------------------------------------
    print("\nInitializing Reissner-Nordström spacetime...")
    # Use symbolic mass 'M' and charge 'Q'
    M = sympy.Symbol('M', positive=True)
    Q = sympy.Symbol('Q', positive=True)
    rn_spacetime = PredefinedSpacetime('reissner-nordstrom', M=M, Q=Q)
    print(f"Spacetime Coordinates: {rn_spacetime.coords}")
    print("Metric g_munu:")
    print(str(rn_spacetime.metric))

    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For Reissner-Nordström spacetime, the Ricci scalar should be zero.
    print("\nComputing the Ricci scalar R...")
    ricci_scalar = rn_spacetime.ricci_scalar()
    print(f"Ricci Scalar: {ricci_scalar}")

    # Verification
    # The calculation can be complex, so we simplify the result.
    simplified_ricci = sympy.simplify(ricci_scalar)
    is_correct = (simplified_ricci == 0)
    print(f"\nSimplified Ricci Scalar: {simplified_ricci}")
    print(f"Is the Ricci scalar correct (R = 0)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar is non-zero, which is unexpected.")

    print("\n--- Reissner-Nordström Example Complete ---")

if __name__ == "__main__":
    run_reissner_nordstrom_example()
