import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_desitter_example():
    """
    Demonstrates the properties of de Sitter spacetime.
    """
    print("--- de Sitter Spacetime Example ---")

    # 1. Initialize de Sitter Spacetime
    # -----------------------------------
    print("\nInitializing de Sitter spacetime...")
    # Use a symbolic radius 'alpha'
    alpha = sympy.Symbol('alpha', positive=True)
    desitter_spacetime = PredefinedSpacetime('deSitter', alpha=alpha)
    print(f"Spacetime Coordinates: {desitter_spacetime.coords}")
    print("Metric g_munu:")
    print(str(desitter_spacetime.metric))

    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For de Sitter spacetime, the Ricci scalar should be a positive constant:
    # R = 12 / alpha^2
    print("\nComputing the Ricci scalar R...")
    ricci_scalar = desitter_spacetime.ricci_scalar()
    print(f"Ricci Scalar: {ricci_scalar}")

    # Verification
    expected_ricci_scalar = 12 / alpha**2
    is_correct = sympy.simplify(ricci_scalar - expected_ricci_scalar) == 0
    print(f"\nIs the Ricci scalar correct (R = 12/alpha^2)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar does not match the expected value.")

    print("\n--- de Sitter Example Complete ---")

if __name__ == "__main__":
    run_desitter_example()
