import pytest
import numpy as np

from aqft.spacetime import PredefinedSpacetime
from aqft.field import FieldOperator, ScalarField
from aqft.state import State, VacuumState
from aqft.algebra import AlgebraicProduct


@pytest.fixture
def setup_environment():
    """Sets up the necessary objects for the integration tests."""
    hilbert_dim = 5
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')
    a = FieldOperator(name='a', spacetime=st, is_creation=False)
    a_dag = FieldOperator(name='a_dag', spacetime=st, is_creation=True)
    number_operator = AlgebraicProduct([a_dag, a])
    vacuum = VacuumState(field=field, hilbert_dim=hilbert_dim)
    return {
        "hilbert_dim": hilbert_dim,
        "a_dag": a_dag,
        "number_operator": number_operator,
        "vacuum": vacuum,
        "field": field
    }


def test_numerical_expectation_values(setup_environment):
    """
    Tests the expectation values of the number operator for vacuum and one-particle states.
    This test verifies the bridge between symbolic operator definitions and numerical state calculations.
    """
    # Unpack fixtures
    hilbert_dim = setup_environment["hilbert_dim"]
    a_dag = setup_environment["a_dag"]
    number_operator = setup_environment["number_operator"]
    vacuum = setup_environment["vacuum"]
    field = setup_environment["field"]

    # 1. Test expectation value of N in the vacuum state: <0|N|0> == 0
    exp_N_vacuum = vacuum.expectation_value(number_operator)
    assert np.isclose(exp_N_vacuum, 0), f"Expected <0|N|0> to be 0, but got {exp_N_vacuum}"

    # 2. Create a one-particle state |1> = a_dag |0>
    a_dag_num = a_dag.to_numerical(hilbert_dim)
    one_particle_state_numerical = a_dag_num @ vacuum.numerical_state
    one_particle_state_numerical.normalize()

    one_particle_state = State(
        field=field,
        numerical_state=one_particle_state_numerical,
        hilbert_dim=hilbert_dim
    )

    # 3. Test expectation value of N in the one-particle state: <1|N|1> == 1
    exp_N_one_particle = one_particle_state.expectation_value(number_operator)
    assert np.isclose(exp_N_one_particle, 1), f"Expected <1|N|1> to be 1, but got {exp_N_one_particle}"
