"""
Example demonstrating the basic structure of the new categorical framework.

This script imports and instantiates the core classes from the `aqft_curved.category`
module to verify that the package structure is correct and the components are accessible.
"""

# Since we are running this from the project root, we need to adjust the Python path
# to ensure the `aqft_curved` package can be found.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.category import LocObject, AlgObject, AQFTFunctor
from aqft_curved.algebra import OperatorAlgebra, FieldOperator
from sympy import Symbol, Eq
from sympy.physics.quantum import Commutator

def run_category_framework_example():
    """
    Runs the demonstration for the categorical framework.
    """
    print("--- AQFT Categorical Framework Example ---")

    # 1. Create a concrete Spacetime object
    # -------------------------------------
    print("\n1. Initializing Minkowski spacetime...")
    minkowski_spacetime = PredefinedSpacetime('Minkowski')
    print(f"   Spacetime Coordinates: {minkowski_spacetime.coords}")

    # 2. Wrap the Spacetime in a LocObject
    # --------------------------------------
    print("\n2. Creating a LocObject from the spacetime...")
    loc_obj = LocObject(spacetime=minkowski_spacetime)

    print(f"   Created LocObject for manifold: {loc_obj.manifold}")
    print(f"   Metric is a {type(loc_obj.metric)}")
    print(f"   Orientation form: {loc_obj.orientation}")
    print(f"   Time-orientation form: {loc_obj.time_orientation}")
    
    # Check a placeholder method
    loc_obj.is_globally_hyperbolic()

    # 3. Define a functor for a free scalar field theory
    # ---------------------------------------------------
    print("\n3. Defining a functor for a free scalar field...")
    mass = Symbol('m')
    scalar_functor = AQFTFunctor(mass=mass)
    print(f"   Functor created for a scalar field with mass={mass}")

    # 4. Apply the functor to the LocObject to get the corresponding AlgObject
    # -----------------------------------------------------------------------
    minkowski_alg_obj = scalar_functor.map_object(loc_obj)

    print("\n--- Categorical Framework Example Complete ---")
    print("The basic structure is in place and accessible.")

if __name__ == "__main__":
    run_category_framework_example()
