import sympy

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField

def test_scalar_field_eom_minkowski():
    """
    Tests the equation of motion for a massless scalar field in Minkowski spacetime.
    The result should be the standard wave equation: Box(phi) = 0.
    """
    # Setup spacetime and field
    minkowski = PredefinedSpacetime('Minkowski')
    # The field's name is a symbolic function, e.g., phi(t, x, y, z)
    phi = ScalarField(spacetime=minkowski, name='phi', mass=0)

    # Get the equation of motion
    eom = phi.equation_of_motion()

    # Define the expected result
    t, x, y, z = minkowski.coords
    phi_func = sympy.Function('phi')(*minkowski.coords)
    expected_eom = sympy.Eq(
        -sympy.diff(phi_func, t, 2) + \
        sympy.diff(phi_func, x, 2) + \
        sympy.diff(phi_func, y, 2) + \
        sympy.diff(phi_func, z, 2),
        0
    )

    # The equation of motion is (Box - m^2 - xi*R)phi = 0.
    # For this case, it simplifies to Box(phi) = 0.
    # We check that the left-hand side of the computed EOM simplifies to the
    # left-hand side of the expected EOM.
    assert sympy.simplify(eom.lhs - expected_eom.lhs) == 0
    assert eom.rhs == 0

def test_scalar_field_eom_massive():
    """
    Tests the EOM for a massive scalar field in Minkowski spacetime.
    The result should be (Box - m^2)phi = 0.
    """
    # Setup spacetime and field
    minkowski = PredefinedSpacetime('Minkowski')
    mass = sympy.Symbol('m')
    phi = ScalarField(spacetime=minkowski, name='phi', mass=mass)

    # Get the equation of motion
    eom = phi.equation_of_motion()

    # Define the expected result
    t, x, y, z = minkowski.coords
    phi_func = sympy.Function('phi')(*minkowski.coords)
    wave_op = (-sympy.diff(phi_func, t, 2) +
               sympy.diff(phi_func, x, 2) +
               sympy.diff(phi_func, y, 2) +
               sympy.diff(phi_func, z, 2))
    expected_eom = sympy.Eq(wave_op - mass**2 * phi_func, 0)

    # Check that the computed EOM matches the expected one
    assert sympy.simplify(eom.lhs - expected_eom.lhs) == 0
    assert eom.rhs == 0
