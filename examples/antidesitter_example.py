import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_antidesitter_example():
    """
    Demonstrates the properties of Anti-de Sitter (AdS) spacetime.
    """
    print("--- Anti-de Sitter (AdS) Spacetime Example ---")

    # 1. Initialize Anti-de Sitter Spacetime
    # -----------------------------------------
    print("\nInitializing Anti-de Sitter spacetime...")
    # Use a symbolic radius 'alpha'
    alpha = sympy.Symbol('alpha', positive=True)
    ads_spacetime = PredefinedSpacetime('anti-desitter', alpha=alpha)
    print(f"Spacetime Coordinates: {ads_spacetime.coords}")
    print("Metric g_munu:")
    print(str(ads_spacetime.metric))

    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For AdS spacetime, the Ricci scalar should be a negative constant:
    # R = -12 / alpha^2
    print("\nComputing the Ricci scalar R...")
    ricci_scalar = ads_spacetime.ricci_scalar()
    print(f"Ricci Scalar: {ricci_scalar}")

    # Verification
    expected_ricci_scalar = -12 / alpha**2
    is_correct = sympy.simplify(ricci_scalar - expected_ricci_scalar) == 0
    print(f"\nIs the Ricci scalar correct (R = -12/alpha^2)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar does not match the expected value.")

    print("\n--- Anti-de Sitter Example Complete ---")

if __name__ == "__main__":
    run_antidesitter_example()
