import sympy
from sympy import Eq
from sympy.physics.quantum import Commutator
from functools import reduce
import operator

from .field import FieldOperator
from .spacetime import Spacetime
from .quantum_state import QuantumState, qeye


class OperatorAlgebra:
    """
    Manages a collection of symbolic operators and the algebraic relations
    (e.g., commutation relations) that define their behavior.
    """
    def __init__(self, generators=None, relations=None):
        """
        Initializes an OperatorAlgebra.

        Parameters:
            generators (list of FieldOperator): The initial set of operators.
            relations (list of sympy.Eq): The commutation relations defining the algebra.
                                           Example: [Eq(Commutator(phi, pi), sympy.I)]
        """
        self.generators = generators or []
        self.relations = relations or []

    def add_generator(self, operator):
        """Adds a new operator to the algebra's set of generators."""
        if not isinstance(operator, FieldOperator):
            raise TypeError("Generator must be an instance of FieldOperator.")
        self.generators.append(operator)

    def add_relation(self, relation):
        """Adds a new algebraic relation."""
        self.relations.append(relation)

    def commutator(self, op1, op2):
        """
        Computes the commutator [op1, op2] based on the defined relations.
        """
        comm = Commutator(op1.name, op2.name)
        neg_comm = Commutator(op2.name, op1.name)

        # Check for [op1, op2] = result
        for rel in self.relations:
            if rel.lhs == comm:
                return rel.rhs
        
        # Check for [op2, op1] = result  (implies [op1, op2] = -result)
        for rel in self.relations:
            if rel.lhs == neg_comm:
                return -rel.rhs

        # Default to zero if no relation is found
        return 0

class CausalRegion:
    """
    Represents a hyperrectangular region in spacetime, defined by coordinate ranges.
    This is essential for defining local algebras and enforcing locality.
    """
    def __init__(self, spacetime, ranges):
        """
        Initializes a CausalRegion.

        Parameters:
            spacetime (Spacetime): The spacetime containing the region.
            ranges (dict): A dictionary mapping coordinate symbols to (min, max) tuples.
                         Example: {t: (0, 1), x: (0, L)}
        """
        if not isinstance(spacetime, Spacetime):
            raise TypeError("spacetime must be an instance of the Spacetime class.")
        self.spacetime = spacetime
        self.ranges = ranges

        # Verify that all coordinates in ranges are part of the spacetime
        for coord in ranges:
            if coord not in spacetime.coords:
                raise ValueError(f"Coordinate {coord} not found in spacetime coordinates.")

    def __repr__(self):
        range_str = ', '.join([f"{k}: {v}" for k, v in self.ranges.items()])
        return f"CausalRegion({range_str})"

    def is_causally_separated(self, other_region):
        """
        Checks if this region is causally separated from another region.

        For Minkowski spacetime, this is true if all points in this region are
        spacelike separated from all points in the other region.

        Returns:
            bool: True if the regions are causally separated, False otherwise.
        """
        if self.spacetime.name != 'Minkowski':
            raise NotImplementedError("Causal separation check is only implemented for Minkowski spacetime.")

        if not isinstance(other_region, CausalRegion):
            raise TypeError("Can only check separation with another CausalRegion.")

        # The squared interval is ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
        # For separation, we need ds^2 > 0 for all pairs of points.
        # This is equivalent to min(dx^2 + dy^2 + dz^2) > max(dt^2).

        t_coord = self.spacetime.coords[0]
        spatial_coords = self.spacetime.coords[1:]

        # Calculate the maximum squared time difference
        r1_t_min, r1_t_max = self.ranges.get(t_coord, (0, 0))
        r2_t_min, r2_t_max = other_region.ranges.get(t_coord, (0, 0))
        max_dt_sq = max(r1_t_min - r2_t_max, r2_t_min - r1_t_max, 0)**2

        # Calculate the minimum squared spatial distance
        min_dist_sq = 0
        for coord in spatial_coords:
            r1_min, r1_max = self.ranges.get(coord, (0, 0))
            r2_min, r2_max = other_region.ranges.get(coord, (0, 0))
            # Distance between two intervals [a,b] and [c,d] is max(0, c-b, a-d)
            d = max(0, r1_min - r2_max, r2_min - r1_max)
            min_dist_sq += d**2

        return min_dist_sq > max_dt_sq

class AlgebraicProduct:
    """
    Represents a formal, non-commutative product of operators.
    """
    def __init__(self, operators):
        """
        Initializes an AlgebraicProduct.

        Parameters:
            operators (list of FieldOperator): A list of operators in the product.
        """
        if not all(isinstance(op, FieldOperator) for op in operators):
            raise TypeError("All items in the product must be FieldOperators.")
        self.operators = tuple(operators)

    def to_numerical(self, hilbert_dim):
        """
        Converts the product of operators to its numerical QuantumState representation.

        Parameters:
            hilbert_dim (int): The dimension of the Hilbert space.

        Returns:
            QuantumState: The numerical representation of the operator product.
        """
        if not self.operators:
            return qeye(hilbert_dim)

        if not all(hasattr(op, 'to_numerical') for op in self.operators):
            raise NotImplementedError(
                "Cannot create numerical representation: "
                "not all operators in the product have a 'to_numerical' method."
            )

        numerical_ops = [op.to_numerical(hilbert_dim) for op in self.operators]
        
        # Use reduce to multiply all operators in the list
        return reduce(operator.matmul, numerical_ops)

    def simplify(self, algebra):
        """
        Simplifies the product using the relations from a given algebra (placeholder).
        This would involve applying commutation relations to reorder operators.
        """
        print("Placeholder for simplifying algebraic product.")
        return self

    def __repr__(self):
        return ' * '.join(map(str, self.operators))

    def __mul__(self, other):
        if isinstance(other, AlgebraicProduct):
            return AlgebraicProduct(self.operators + other.operators)
        elif isinstance(other, FieldOperator):
            return AlgebraicProduct(self.operators + (other,))
        else:
            return NotImplemented
