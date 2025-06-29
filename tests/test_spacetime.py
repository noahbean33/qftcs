import sympy
from aqft_curved.spacetime import Spacetime, PredefinedSpacetime

def test_spacetime_instantiation():
    """Tests basic instantiation of the Spacetime class."""
    dim = 2
    coords = ('t', 'x')
    metric = sympy.diag(-1, 1)
    st = Spacetime(dimension=dim, coordinates=coords, metric=metric)
    assert st.dimension == dim
    assert st.coords == sympy.symbols(coords)
    assert st.metric == metric

def test_predefined_minkowski():
    """Tests the creation of Minkowski spacetime."""
    minkowski = PredefinedSpacetime('Minkowski')
    assert minkowski.name == 'Minkowski'
    assert minkowski.dimension == 4
    t, x, y, z = sympy.symbols('t x y z')
    expected_metric = sympy.diag(-1, 1, 1, 1)
    assert minkowski.metric == expected_metric

def test_schwarzschild_ricci_scalar():
    """
    Tests that the Ricci scalar for the Schwarzschild spacetime is zero,
    as it is a vacuum solution.
    """
    # The Schwarzschild metric is a solution to the vacuum Einstein equations,
    # so its Ricci tensor (and thus Ricci scalar) should be zero.
    schwarzschild = PredefinedSpacetime('Schwarzschild', M=sympy.Symbol('M'))
    ricci_scalar = schwarzschild.ricci_scalar()
    # The expression can be complex, so we simplify it before checking.
    assert sympy.simplify(ricci_scalar) == 0
