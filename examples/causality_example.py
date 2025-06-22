"""
Example demonstrating the CausalRegion class and causality checks.

This script defines several hyperrectangular regions in Minkowski spacetime and uses
the `is_causally_separated` method to check if they are spacelike separated.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sympy import Symbol
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.algebra import CausalRegion
from aqft_curved.category import AQFTFunctor, LocObject

def run_causality_example():
    """
    Runs the demonstration for causal region separation.
    """
    print("--- Einstein Causality Example ---")

    # 1. Initialize Minkowski spacetime
    # ---------------------------------
    minkowski = PredefinedSpacetime('Minkowski')
    t, x, y, z = minkowski.coords
    print(f"\nInitialized {minkowski.name} spacetime with coordinates {minkowski.coords}")

    # 2. Define two causally separated regions
    # ----------------------------------------
    # Region 1: Centered at x = -5
    # Region 2: Centered at x = +5
    # They are simultaneous (t in [0,1]) and far enough apart to be spacelike.
    print("\nDefining two causally separated regions...")
    region1 = CausalRegion(minkowski, {t: (0, 1), x: (-6, -4)})
    region2 = CausalRegion(minkowski, {t: (0, 1), x: (4, 6)})
    print(f"   Region 1: {region1}")
    print(f"   Region 2: {region2}")

    # 3. Define a functor for a free scalar field
    # -------------------------------------------
    scalar_functor = AQFTFunctor(mass=Symbol('m'))

    # 4. Generate the local algebras for these regions
    # ----------------------------------------------
    # In this simplified model, the functor maps the whole spacetime to an algebra.
    # A more advanced model would map each region to a subalgebra.
    # For now, we use the same functor to generate algebras we can test.
    loc_object = LocObject(minkowski)
    alg_obj1 = scalar_functor.map_object(loc_object)
    alg_obj2 = scalar_functor.map_object(loc_object)

    # 5. Verify the causality axiom
    # -----------------------------
    scalar_functor.verify_causality(region1, alg_obj1, region2, alg_obj2)

    # 6. Define two regions that are NOT causally separated
    # -----------------------------------------------------
    print("\nDefining two regions that are NOT causally separated...")
    region3 = CausalRegion(minkowski, {t: (0, 0.5), x: (0, 0.5)})
    region4 = CausalRegion(minkowski, {t: (2, 2.5), x: (1, 1.5)})
    print(f"   Region 3: {region3}")
    print(f"   Region 4: {region4}")

    # Verify that the causality axiom does not apply
    alg_obj3 = scalar_functor.map_object(loc_object)
    alg_obj4 = scalar_functor.map_object(loc_object)
    scalar_functor.verify_causality(region3, alg_obj3, region4, alg_obj4)

    print("\n--- Causality Example Complete ---")

if __name__ == "__main__":
    run_causality_example()
