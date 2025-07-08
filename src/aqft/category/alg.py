"""
Defines the category Alg of unital *-algebras.
"""

from aqft.algebra import OperatorAlgebra

class AlgObject:
    """
    Represents an object in the category Alg.

    An object is a unital *-algebra, encapsulated by an OperatorAlgebra instance.
    """
    def __init__(self, algebra: OperatorAlgebra):
        if not isinstance(algebra, OperatorAlgebra):
            raise TypeError("algebra must be an instance of the OperatorAlgebra class.")
        self.algebra = algebra
        self.generators = algebra.generators
        self.relations = algebra.relations

    def __repr__(self):
        return f"AlgObject(generators={len(self.generators)}, relations={len(self.relations)})"

class AlgMorphism:
    """
    Represents a morphism in the category Alg.

    A morphism is a unit-preserving, injective *-homomorphism.
    """
    def __init__(self, domain, codomain, homomorphism_map):
        if not isinstance(domain, AlgObject) or not isinstance(codomain, AlgObject):
            raise TypeError("Domain and codomain must be instances of AlgObject.")
        self.domain = domain
        self.codomain = codomain
        self.homomorphism_map = homomorphism_map

    def is_valid_morphism(self):
        """
        Validates that the map is a unit-preserving, injective *-homomorphism.
        (Placeholder for future implementation)
        """
        print("Warning: Morphism validation is not yet implemented.")
        return True
