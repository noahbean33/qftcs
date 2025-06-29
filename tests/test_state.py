import sympy
import pytest

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import FieldOperator, ScalarField
from aqft_curved.state import State, VacuumState
from aqft_curved.algebra import AlgebraicProduct

def test_state_instantiation():
    """Tests basic instantiation of the State class."""
    st = PredefinedSpacetime('Minkowski')
    field = FieldOperator(name='phi', spacetime=st)
    state = State(field=field)
    assert state.field == field
    assert not state.has_numerical

def test_vacuum_state_instantiation():
    """Tests instantiation of the VacuumState class."""
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')
    vac = VacuumState(field=field, state_type='Hadamard')
    assert vac.state_type == 'Hadamard'
    assert vac.field == field
    assert not vac.has_numerical

def test_numerical_vacuum_state():
    """Tests creation of a numerical vacuum state."""
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')
    vac = VacuumState(field=field, hilbert_dim=5)
    assert vac.has_numerical
    assert vac.hilbert_dim == 5
    # The numerical state should be a ket (column vector)
    assert vac.numerical_state.data.shape == (5, 1)
    # The vacuum state should be [1, 0, 0, 0, 0].T
    assert vac.numerical_state.data[0, 0] == 1
    assert vac.numerical_state.data[1, 0] == 0

def test_expectation_value_simple():
    """
    Tests the expectation value of simple operators in the vacuum state.
    <0|a|0> = 0 and <0|a_dag|0> = 0.
    """
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')
    vac = VacuumState(field=field, hilbert_dim=3)

    # Annihilation operator
    a = FieldOperator(name='a', spacetime=st, is_creation=False)
    exp_a = vac.expectation_value(a)
    assert exp_a == 0

    # Creation operator
    a_dag = FieldOperator(name='a_dag', spacetime=st, is_creation=True)
    exp_a_dag = vac.expectation_value(a_dag)
    assert exp_a_dag == 0

def test_expectation_value_number_operator():
    """
    Tests the expectation value of the number operator N = a_dag * a
    in the vacuum state, which should be 0.
    """
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')
    vac = VacuumState(field=field, hilbert_dim=3)

    a_dag = FieldOperator(name='a_dag', spacetime=st, is_creation=True)
    a = FieldOperator(name='a', spacetime=st, is_creation=False)

    # Number operator
    N = AlgebraicProduct([a_dag, a])
    
    # The State.expectation_value method supports objects with a `to_numerical` method.
    exp_N = vac.expectation_value(N)

    # The expectation value should be close to 0
    assert abs(exp_N) < 1e-9
