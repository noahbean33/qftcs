import sympy
import numpy as np

from aqft.spacetime import PredefinedSpacetime
from aqft.field import ScalarField

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

def test_solve_radial_equation():
    """
    Tests the numerical radial solver for a scalar field in Schwarzschild spacetime.
    This test ensures the solver runs without errors and returns arrays of the correct shape.
    """
    # 1. Setup spacetime and field
    schwarzschild = PredefinedSpacetime(name='Schwarzschild', M=1)
    scalar_field = ScalarField(spacetime=schwarzschild, name='phi', mass=0.0)

    # 2. Set parameters
    omega = 0.5
    l = 1
    r_start = 2.1
    r_end = 20.0
    num_points = 100
    initial_conditions = [1.0, 0.0]

    # 3. Solve the equation
    r_vals, R_vals = scalar_field.solve_radial_equation(
        omega=omega,
        l=l,
        r_start=r_start,
        r_end=r_end,
        initial_conditions=initial_conditions,
        num_points=num_points
    )

    # 4. Assertions
    assert isinstance(r_vals, np.ndarray)
    assert isinstance(R_vals, np.ndarray)
    assert r_vals.shape == (num_points,)
    assert R_vals.shape == (num_points,)
    assert np.isclose(r_vals[0], r_start)
    assert np.isclose(R_vals[0], initial_conditions[0])

