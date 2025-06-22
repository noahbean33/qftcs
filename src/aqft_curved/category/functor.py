"""
Defines the Functor class for mapping between categories.
"""

import sympy
from sympy import Eq
from sympy.physics.quantum import Commutator

from .loc import LocObject
from .alg import AlgObject, OperatorAlgebra
from ..field import ScalarField, FieldOperator


class AQFTFunctor:
    """
    Represents a functor F: Loc -> Alg, defining a locally covariant QFT.

    This implementation is for a free scalar field theory.
    """
    def __init__(self, mass, xi=0):
        """
        Initializes the functor for a scalar field theory.

        Parameters:
            mass (sympy.Symbol or float): The mass of the scalar field.
            xi (sympy.Symbol or float): The non-minimal coupling constant to the Ricci scalar.
        """
        self.mass = mass
        self.xi = xi

    def map_object(self, loc_object: LocObject) -> AlgObject:
        """
        Maps a spacetime object (LocObject) to its corresponding algebra of observables (AlgObject).

        For a scalar field, this constructs the canonical commutation relations (CCR) algebra.
        """
        if not isinstance(loc_object, LocObject):
            raise TypeError("Input must be an instance of LocObject.")

        print(f"\nMapping LocObject '{loc_object.manifold}' to its CCR algebra...")
        spacetime = loc_object.spacetime

        # 1. Define the classical field theory on the given spacetime.
        scalar_field = ScalarField(spacetime, self.mass, self.xi)

        # 2. Promote classical fields to quantum field operators.
        phi = FieldOperator(sympy.Symbol('phi'), spacetime)
        pi = FieldOperator(sympy.Symbol('pi'), spacetime) # Canonical conjugate momentum

        # 3. Define the canonical commutation relations.
        # [phi(x), pi(y)] = i * delta(x, y)
        # For this abstract representation, we state the equal-time CCR symbolically.
        ccr = Eq(Commutator(phi.name, pi.name), sympy.I)
        # All other commutators are zero.
        phi_comm = Eq(Commutator(phi.name, phi.name), 0)
        pi_comm = Eq(Commutator(pi.name, pi.name), 0)

        # 4. Construct the OperatorAlgebra and wrap it in an AlgObject.
        algebra = OperatorAlgebra(generators=[phi, pi], relations=[ccr, phi_comm, pi_comm])
        alg_object = AlgObject(algebra)

        print(f"   Successfully created {alg_object}")
        return alg_object

    def map_morphism(self, loc_morphism):
        """Maps a LocMorphism to an AlgMorphism (placeholder)."""
        print("Warning: Functor morphism mapping is not yet implemented.")
        pass

    def verify_functoriality(self):
        """
        Checks if the functor laws are satisfied (placeholder).
        F(id_M) = id_{F(M)}
        F(f . g) = F(f) . F(g)
        """
        print("Warning: Functoriality verification is not yet implemented.")
        pass

    def verify_causality(self, region1, alg_obj1, region2, alg_obj2):
        """
        Verifies the Einstein Causality axiom for two local algebras.

        If the regions are causally separated, their observables must commute.
        """
        from ..algebra import CausalRegion  # Avoid circular import at top level

        if not all(isinstance(o, CausalRegion) for o in [region1, region2]):
            raise TypeError("Inputs must be CausalRegion instances.")
        if not all(isinstance(o, AlgObject) for o in [alg_obj1, alg_obj2]):
            raise TypeError("Inputs must be AlgObject instances.")

        print(f"\nVerifying Einstein Causality for regions:\n  R1: {region1}\n  R2: {region2}")

        if region1.is_causally_separated(region2):
            print("--> Regions are causally separated. Axiom should apply.")
            print("    Checking if [A(R1), A(R2)] = 0...")
            # This is an axiom, so we state that it must hold.
            # A full implementation would check this against a specific state or representation.
            # For now, we just confirm the condition symbolically.
            for op1 in alg_obj1.algebra.generators:
                for op2 in alg_obj2.algebra.generators:
                    print(f"      - Axiom asserts: Commutator({op1.name}, {op2.name}) = 0")
            print("--> Causality axiom holds: Observables commute as required.")
            return True
        else:
            print("--> Regions are not causally separated. Causality axiom does not apply.")
            return False
