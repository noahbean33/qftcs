import sympy
from sympy import Eq, Symbol
from sympy.physics.quantum import Commutator

from aqft.field import FieldOperator
from aqft.algebra import OperatorAlgebra, CausalRegion, AlgebraicProduct
from aqft.spacetime import PredefinedSpacetime

def test_operator_algebra_commutator():
    """Tests the commutator calculation in OperatorAlgebra."""
    # Define a spacetime
    st = PredefinedSpacetime('Minkowski')
    # Define two operators
    phi = FieldOperator(name=Symbol('phi', commutative=False), spacetime=st)
    pi = FieldOperator(name=Symbol('pi', commutative=False), spacetime=st)

    # Define the canonical commutation relation [phi, pi] = i
    relation = Eq(Commutator(phi.name, pi.name), sympy.I)

    # Create the algebra
    algebra = OperatorAlgebra(generators=[phi, pi], relations=[relation])

    # Test the commutator
    assert algebra.commutator(phi, pi) == sympy.I
    # Test the reverse commutator
    assert algebra.commutator(pi, phi) == -sympy.I
    # Test commutator with self
    assert algebra.commutator(phi, phi) == 0

def test_causal_region_separation():
    """Tests the causal separation logic for CausalRegion."""
    # Use Minkowski spacetime
    minkowski = PredefinedSpacetime('Minkowski')
    t, x, y, z = minkowski.coords

    # Define two spatially separated regions at the same time
    region1 = CausalRegion(minkowski, {t: (0, 1), x: (0, 1)})
    region2 = CausalRegion(minkowski, {t: (0, 1), x: (2, 3)})
    assert region1.is_causally_separated(region2)

    # Define two regions that are not causally separated (timelike)
    region3 = CausalRegion(minkowski, {t: (0, 1), x: (0, 1)})
    region4 = CausalRegion(minkowski, {t: (2, 3), x: (0, 1)})
    assert not region3.is_causally_separated(region4)

    # Define two regions that are lightlike separated at their boundaries
    # and thus not fully causally separated
    region5 = CausalRegion(minkowski, {t: (0, 1), x: (0, 1)})
    region6 = CausalRegion(minkowski, {t: (1, 2), x: (1, 2)})
    assert not region5.is_causally_separated(region6)

def test_algebraic_product():
    """Tests the creation and multiplication of AlgebraicProduct."""
    st = PredefinedSpacetime('Minkowski')
    op1 = FieldOperator(Symbol('A', commutative=False), spacetime=st)
    op2 = FieldOperator(Symbol('B', commutative=False), spacetime=st)
    op3 = FieldOperator(Symbol('C', commutative=False), spacetime=st)

    # Create a product
    product1 = AlgebraicProduct([op1, op2])
    assert str(product1) == "A * B"

    # Test multiplication
    product2 = product1 * op3
    assert isinstance(product2, AlgebraicProduct)
    assert product2.operators == (op1, op2, op3)
    assert str(product2) == "A * B * C"

    product3 = op3 * product1
    assert isinstance(product3, AlgebraicProduct)
    assert product3.operators == (op3, op1, op2)
    assert str(product3) == "C * A * B"
