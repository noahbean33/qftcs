"""
Example: Minkowski Spacetime

This script demonstrates how to use the `spacetime` module to define
a predefined spacetime (Minkowski space) and compute a key geometric
quantity, the Ricci scalar.
"""

import sympy
from aqft.spacetime import PredefinedSpacetime

def run_minkowski_example():
    """Initializes Minkowski spacetime and computes its Ricci scalar."""
    print("Initializing Minkowski spacetime...")
    minkowski = PredefinedSpacetime('Minkowski')

    print("\nMetric Tensor:")
    sympy.pprint(minkowski.metric)

    print("\nCalculating Ricci Scalar...")
    ricci_scalar = minkowski.ricci_scalar()

    print("\nRicci Scalar:")
    sympy.pprint(ricci_scalar)

    # Verify that the Ricci scalar is zero
    if ricci_scalar == 0:
        print("\nAs expected, the Ricci scalar for Minkowski space is 0.")
    else:
        print("\nUnexpected result: The Ricci scalar is non-zero.")

if __name__ == "__main__":
    run_minkowski_example()
