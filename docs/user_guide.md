# AQFT Python Package User Guide

## Introduction

Welcome to the AQFT (Quantum Field Theory in Curved Spacetime) Python package! This guide provides an overview of the package's core functionalities and examples to help you get started.

The `aqft` package is a tool designed for performing symbolic and numerical calculations in the framework of quantum field theory in curved spacetime. The current focus is the **Static Black Hole Explorer MVP**, which aims to compute the Renormalized Stress-Energy Tensor (RSET) for a scalar field in Schwarzschild spacetime.

## Core Concepts

The package is built around a few key abstractions:

-   **`Spacetime`**: Represents a pseudo-Riemannian manifold, defined by a coordinate system and a metric tensor. It provides methods for computing geometric quantities like Christoffel symbols, the Ricci tensor, and the Ricci scalar.
-   **`FieldOperator`**: The base class for all field types, representing operators on the spacetime.
-   **`ScalarField`**: A class representing a scalar quantum field. It includes methods to symbolically compute the Klein-Gordon equation and the classical stress-energy tensor.

## Key Features

### Symbolic Calculations

The package leverages `sympy` to perform symbolic computations. You can define spacetimes and fields with symbolic parameters and derive physical quantities in their symbolic form.

### Numerical Radial Wave Equation Solver

The `ScalarField.solve_radial_equation` method provides a numerical solver for the radial part of the Klein-Gordon equation in static, spherically symmetric spacetimes like Schwarzschild. This is a key tool for finding the mode solutions required for further calculations, such as computing the Renormalized Stress-Energy Tensor (RSET).

The following example demonstrates how to use this solver to find and plot a radial mode solution for a massless scalar field. An executable version can be found in `examples/solve_radial_equation_example.py`.

**Example Script (`examples/solve_radial_equation_example.py`):**

```python
import numpy as np
import matplotlib.pyplot as plt
from aqft.field import ScalarField
from aqft.spacetime import PredefinedSpacetime

def solve_and_plot_radial_equation():
    """
    This example demonstrates how to use the `solve_radial_equation` method
    of the ScalarField class to find and plot a numerical solution for the
    radial part of the Klein-Gordon equation in Schwarzschild spacetime.
    """
    print("--- Numerical Radial Equation Solver Example ---")

    # 1. Define Schwarzschild spacetime with M=1
    M = 1
    schwarzschild_spacetime = PredefinedSpacetime(name='Schwarzschild', M=M)
    print(f"Initialized Schwarzschild spacetime with M = {M}")

    # 2. Define a massless scalar field
    scalar_field = ScalarField(spacetime=schwarzschild_spacetime, name='phi', mass=0.0)
    print("Defined a massless scalar field 'phi'.")

    # 3. Set parameters for the radial equation
    omega = 0.5  # Frequency of the mode
    l = 1        # Angular momentum quantum number
    r_start = 2.1 * M  # Start integration just outside the event horizon
    r_end = 20 * M     # End integration at a larger radius
    num_points = 2000

    # Set initial conditions for R(r) and R'(r) at r_start
    # These are arbitrary for demonstration; physical solutions require specific boundary conditions.
    initial_conditions = [1.0, 0.0]  # R(r_start) = 1, R'(r_start) = 0

    print(f"Solving radial equation for omega={omega}, l={l} from r={r_start} to r={r_end}")

    # 4. Solve the radial equation
    try:
        r_vals, R_vals = scalar_field.solve_radial_equation(
            omega=omega,
            l=l,
            r_start=r_start,
            r_end=r_end,
            initial_conditions=initial_conditions,
            num_points=num_points
        )

        # 5. Plot the solution
        plt.figure(figsize=(10, 6))
        plt.plot(r_vals, R_vals, label=fr'$\omega={omega}, l={l}$')
        plt.title('Numerical Solution of the Radial Wave Equation $R(r)$')
        plt.xlabel('Radial Coordinate $r/M$')
        plt.ylabel('Radial Function $R(r)$')
        plt.grid(True)
        plt.legend()
        plt.axvline(x=2*M, color='r', linestyle='--', label='Event Horizon (2M)')
        plt.legend()
        print("Plot generated. Please close the plot window to exit.")
        plt.show()

    except Exception as e:
        print(f"An error occurred during calculation or plotting: {e}")

    print("--- Example Complete ---")

if __name__ == "__main__":
    solve_and_plot_radial_equation()
```

**Expected Output:**

The script will first print status messages to the console confirming the setup. It will then generate and display a plot showing the numerical solution of the radial function `R(r)` against the radial coordinate `r/M`. The plot will include a vertical dashed line indicating the position of the event horizon and show the behavior of the radial mode function outside the black hole.

### Example: Stress-Energy Tensor in Minkowski Spacetime

The following example demonstrates how to compute the classical stress-energy tensor for a scalar field. An executable version can be found in `examples/scalar_field_example.py`.

```python
import sympy
from aqft.spacetime import PredefinedSpacetime
from aqft.field import ScalarField

def run_stress_energy_tensor_example():
    """
    Demonstrates the calculation of the stress-energy tensor for a scalar field
    in Minkowski spacetime.
    """
    # 1. Initialize Minkowski Spacetime
    minkowski_spacetime = PredefinedSpacetime('Minkowski')

    # 2. Define a Scalar Field with symbolic mass and coupling
    m_sym = sympy.Symbol('m')
    xi_sym = sympy.Symbol('xi')
    scalar_field = ScalarField(
        spacetime=minkowski_spacetime, 
        mass=m_sym, 
        coupling_xi=xi_sym
    )

    # 3. Compute the symbolic stress-energy tensor
    T_munu = scalar_field.stress_energy_tensor()

    print("--- Stress-Energy Tensor (Scalar Field in Minkowski) ---")
    sympy.pprint(T_munu)

if __name__ == "__main__":
    run_stress_energy_tensor_example()
```

This script initializes a 4D Minkowski spacetime, defines a general scalar field, and then prints the symbolically computed stress-energy tensor.
if __name__ == "__main__":
    run_schwarzschild_scalar_field_tutorial()

```

**Expected Output:**

The script will print the Schwarzschild metric tensor and the computed Klein-Gordon equation (both with symbolic M and M=1) to the console. It will then display a 2D plot of the `g_tt` metric component. After closing the plot, the script will print "--- Schwarzschild Scalar Field Tutorial Complete ---".

```
--- Schwarzschild Scalar Field Tutorial ---

Initializing Schwarzschild spacetime with mass M = 1...
Schwarzschild Metric Tensor (symbolic M):
Matrix([[-1 + 2/r, 0, 0, 0], [0, 1/(1 - 2/r), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]])

Defining a massless scalar field on Schwarzschild spacetime...
Scalar field symbol: phi(t, r, theta, phi)

Computing the Klein-Gordon equation (Box phi = 0) for the massless scalar field...
Klein-Gordon Equation (Box phi) (symbolic M):
Eq(Piecewise((...complex expression with M...)))
Substituting M=1 for a more specific form:
Eq(Piecewise((...complex expression with M=1...)))
(Should be equal to 0 for a solution)

Plotting the g_tt metric component...
Plot for g_tt generated with M=1. Please close the plot window to continue.

--- Schwarzschild Scalar Field Tutorial Complete ---
```
(Note: The symbolic output for the Klein-Gordon equation can be quite long and complex, so it's abbreviated above.)

### De Sitter Spacetime Example

This example demonstrates how to initialize a de Sitter spacetime and verify one of its key properties: a constant, positive Ricci scalar `R = 12 / alpha^2`, where `alpha` is the de Sitter radius.

**Example Script (`examples/desitter_example.py`):**

```python
import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_desitter_example():
    """
    Demonstrates the properties of de Sitter spacetime.
    """
    print("--- de Sitter Spacetime Example ---")

    # 1. Initialize de Sitter Spacetime
    # -----------------------------------
    print("\nInitializing de Sitter spacetime...")
    # Use a symbolic radius 'alpha'
    alpha = sympy.Symbol('alpha', positive=True)
    desitter_spacetime = PredefinedSpacetime('deSitter', alpha=alpha)
    print(f"Spacetime Coordinates: {desitter_spacetime.coords}")
    print("Metric g_munu:")
    print(str(desitter_spacetime.metric))

    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For de Sitter spacetime, the Ricci scalar should be a positive constant:
    # R = 12 / alpha^2
    print("\nComputing the Ricci scalar R...")
    ricci_scalar = desitter_spacetime.ricci_scalar()
    print(f"Ricci Scalar: {ricci_scalar}")

    # Verification
    expected_ricci_scalar = 12 / alpha**2
    is_correct = sympy.simplify(ricci_scalar - expected_ricci_scalar) == 0
    print(f"\nIs the Ricci scalar correct (R = 12/alpha^2)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar does not match the expected value.")

    print("\n--- de Sitter Example Complete ---")

if __name__ == "__main__":
    run_desitter_example()

```

**Expected Output:**

```
--- de Sitter Spacetime Example ---

Initializing de Sitter spacetime...
Spacetime Coordinates: (t, r, theta, phi)
Metric g_munu:
Matrix([[-1 + r**2/alpha**2, 0, 0, 0], [0, 1/(1 - r**2/alpha**2), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]])

Computing the Ricci scalar R...
Ricci Scalar: 12/alpha**2

Is the Ricci scalar correct (R = 12/alpha^2)? True

--- de Sitter Example Complete ---
```

### Anti-de Sitter Spacetime Example

This example demonstrates how to initialize an Anti-de Sitter (AdS) spacetime and verify its characteristic property: a constant, negative Ricci scalar `R = -12 / alpha^2`, where `alpha` is the AdS radius.

**Example Script (`examples/antidesitter_example.py`):**

```python
import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_antidesitter_example():
    """
    Demonstrates the properties of Anti-de Sitter (AdS) spacetime.
    """
    print("--- Anti-de Sitter (AdS) Spacetime Example ---")

    # 1. Initialize Anti-de Sitter Spacetime
    # -----------------------------------------
    print("\nInitializing Anti-de Sitter spacetime...")
    # Use a symbolic radius 'alpha'
    alpha = sympy.Symbol('alpha', positive=True)
    ads_spacetime = PredefinedSpacetime('anti-desitter', alpha=alpha)
    print(f"Spacetime Coordinates: {ads_spacetime.coords}")
    print("Metric g_munu:")
    print(str(ads_spacetime.metric))

    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For AdS spacetime, the Ricci scalar should be a negative constant:
    # R = -12 / alpha^2
    print("\nComputing the Ricci scalar R...")
    ricci_scalar = ads_spacetime.ricci_scalar()
    print(f"Ricci Scalar: {ricci_scalar}")

    # Verification
    expected_ricci_scalar = -12 / alpha**2
    is_correct = sympy.simplify(ricci_scalar - expected_ricci_scalar) == 0
    print(f"\nIs the Ricci scalar correct (R = -12/alpha^2)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar does not match the expected value.")

    print("\n--- Anti-de Sitter Example Complete ---")

if __name__ == "__main__":
    run_antidesitter_example()

```

**Expected Output:**

```
--- Anti-de Sitter (AdS) Spacetime Example ---

Initializing Anti-de Sitter spacetime...
Spacetime Coordinates: (t, r, theta, phi)
Metric g_munu:
Matrix([[-1 - r**2/alpha**2, 0, 0, 0], [0, 1/(1 + r**2/alpha**2), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]])

Computing the Ricci scalar R...
Ricci Scalar: -12/alpha**2

Is the Ricci scalar correct (R = -12/alpha^2)? True

--- Anti-de Sitter Example Complete ---
```

### Reissner-Nordström Spacetime Example

This example demonstrates the initialization of a Reissner-Nordström spacetime, which describes a charged, non-rotating black hole. It verifies that the spacetime is Ricci-flat (`R=0`), as expected for a vacuum solution of the Einstein-Maxwell equations.

**Example Script (`examples/reissner_nordstrom_example.py`):**

```python
import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime

def run_reissner_nordstrom_example():
    """
    Demonstrates the properties of Reissner-Nordström spacetime.
    """
    print("--- Reissner-Nordström Spacetime Example ---")

    # 1. Initialize Reissner-Nordström Spacetime
    # -------------------------------------------
    print("\nInitializing Reissner-Nordström spacetime...")
    # Use symbolic mass 'M' and charge 'Q'
    M = sympy.Symbol('M', positive=True)
    Q = sympy.Symbol('Q', positive=True)
    rn_spacetime = PredefinedSpacetime('reissner-nordstrom', M=M, Q=Q)
    print(f"Spacetime Coordinates: {rn_spacetime.coords}")
    print("Metric g_munu:")
    print(str(rn_spacetime.metric))

    # 2. Compute the Ricci Scalar
    # ---------------------------
    # For Reissner-Nordström spacetime, the Ricci scalar should be zero.
    print("\nComputing the Ricci scalar R...")
    ricci_scalar = rn_spacetime.ricci_scalar()
    print(f"Ricci Scalar: {ricci_scalar}")

    # Verification
    # The calculation can be complex, so we simplify the result.
    simplified_ricci = sympy.simplify(ricci_scalar)
    is_correct = (simplified_ricci == 0)
    print(f"\nSimplified Ricci Scalar: {simplified_ricci}")
    print(f"Is the Ricci scalar correct (R = 0)? {is_correct}")
    if not is_correct:
        print("Warning: The Ricci scalar is non-zero, which is unexpected.")

    print("\n--- Reissner-Nordström Example Complete ---")

if __name__ == "__main__":
    run_reissner_nordstrom_example()

```

**Expected Output:**

```
--- Reissner-Nordström Spacetime Example ---

Initializing Reissner-Nordström spacetime...
Spacetime Coordinates: (t, r, theta, phi)
Metric g_munu:
Matrix([[2*M/r - Q**2/r**2 - 1, 0, 0, 0], [0, 1/(-2*M/r + Q**2/r**2 + 1), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]])

Computing the Ricci scalar R...
Ricci Scalar: 0

Simplified Ricci Scalar: 0
Is the Ricci scalar correct (R = 0)? True

--- Reissner-Nordström Example Complete ---
```

### Electromagnetic Field Example

This example introduces the `ElectromagneticField` class and demonstrates how to compute its fundamental properties: the field strength tensor `F_μν`, the source-free Maxwell's equations `∇_μ F^μν = 0`, and the stress-energy tensor `T_μν`.

**Example Script (`examples/em_field_example.py`):**

```python
import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ElectromagneticField

def run_em_field_example():
    """
    Demonstrates the basic properties of the ElectromagneticField class.
    """
    print("--- Electromagnetic Field Example ---")

    # 1. Initialize a Spacetime
    # --------------------------
    print("\nInitializing Minkowski spacetime...")
    minkowski = PredefinedSpacetime('Minkowski')
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # 2. Initialize the Electromagnetic Field
    # ---------------------------------------
    print("\nInitializing the electromagnetic field...")
    em_field = ElectromagneticField(minkowski)
    print("Symbolic 4-potential A_mu:")
    print(em_field.potential)

    # 3. Compute the Field Strength Tensor
    # ------------------------------------
    print("\nComputing the field strength tensor F_munu = d_mu(A_nu) - d_nu(A_mu)...")
    F_munu = em_field.field_strength_tensor()
    print("Field Strength Tensor F_munu:")
    print(F_munu)

    # The components of F_munu correspond to electric (E) and magnetic (B) fields.
    # For example, in Cartesian coordinates (t, x, y, z):
    # E_x = F_10, E_y = F_20, E_z = F_30
    # B_x = F_32, B_y = F_13, B_z = F_21
    t, x, y, z = minkowski.coords
    A_t, A_x, A_y, A_z = em_field.potential
    print("\nExample components:")
    print(f"E_x = F_10 = {F_munu[1, 0]}")
    print(f"B_x = F_32 = {F_munu[3, 2]}")

    # Verify anti-symmetry: F_munu = -F_numu
    is_antisymmetric = sympy.simplify(F_munu + F_munu.T) == sympy.zeros(4)
    print(f"\nIs the tensor anti-symmetric (F_munu = -F_numu)? {is_antisymmetric}")

    # 4. Compute the Equation of Motion
    # ---------------------------------
    print("\nComputing the equation of motion (nabla_mu F^munu = 0)...")
    eom = em_field.equation_of_motion()
    print("Maxwell's Equations in vector form (0 =):")
    print(eom.lhs)  # eom is Eq(lhs, 0), so we print the left-hand side

    # 5. Compute the Stress-Energy Tensor
    # -----------------------------------
    print("\nComputing the stress-energy tensor T_munu...")
    T_munu = em_field.stress_energy_tensor()
    print("Stress-Energy Tensor T_munu (showing T_00 component):")
    # The full tensor is very large, so we'll just display one component.
    # T_00 represents the energy density.
    print(f"T_00 = {T_munu[0, 0]}")

    print("\n--- Electromagnetic Field Example Complete ---")

if __name__ == "__main__":
    run_em_field_example()

```

**Expected Output:**

```
--- Electromagnetic Field Example ---

Initializing Minkowski spacetime...
Spacetime Coordinates: (t, x, y, z)

Initializing the electromagnetic field...
Symbolic 4-potential A_mu:
Matrix([[A_0(t, x, y, z)], [A_1(t, x, y, z)], [A_2(t, x, y, z)], [A_3(t, x, y, z)]])

Computing the field strength tensor F_munu = d_mu(A_nu) - d_nu(A_mu)...
Field Strength Tensor F_munu:
(Symbolic 4x4 anti-symmetric matrix is displayed here)

Example components:
E_x = F_10 = Derivative(A_0(t, x, y, z), x) - Derivative(A_1(t, x, y, z), t)
B_x = F_32 = Derivative(A_2(t, x, y, z), z) - Derivative(A_3(t, x, y, z), y)

Is the tensor anti-symmetric (F_munu = -F_numu)? True

Computing the equation of motion (nabla_mu F^munu = 0)...
Maxwell's Equations in vector form (0 =):
(Symbolic 4x1 vector of second-order partial differential equations is displayed here)

Computing the stress-energy tensor T_munu...
Stress-Energy Tensor T_munu (showing T_00 component):
T_00 = (Derivative(A_0(t, x, y, z), x) - Derivative(A_1(t, x, y, z), t))**2/2 + (Derivative(A_0(t, x, y, z), y) - Derivative(A_2(t, x, y, z), t))**2/2 + (Derivative(A_0(t, x, y, z), z) - Derivative(A_3(t, x, y, z), t))**2/2 + (Derivative(A_1(t, x, y, z), y) - Derivative(A_2(t, x, y, z), x))**2/2 + (Derivative(A_1(t, x, y, z), z) - Derivative(A_3(t, x, y, z), x))**2/2 + (Derivative(A_2(t, x, y, z), z) - Derivative(A_3(t, x, y, z), y))**2/2

--- Electromagnetic Field Example Complete ---
```

### Numerical Quantum States and Operators

This tutorial demonstrates how to work with numerical quantum states and operators. It shows how to create a vacuum state, define creation and annihilation operators, construct a one-particle state, and calculate expectation values. This example bridges the symbolic representation of operators with their numerical counterparts in a finite-dimensional Hilbert space (Fock space).

**Example Script (`examples/numerical_quantum_state_tutorial.py`):**

```python
"""
Numerical Quantum State Tutorial

This tutorial demonstrates how to work with numerical quantum states in the aqft_py library.
It covers:
1. Creating a numerical vacuum state.
2. Defining annihilation and creation operators.
3. Constructing a one-particle state.
4. Calculating expectation values of the number operator.
"""

import numpy as np
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import FieldOperator, ScalarField
from aqft_curved.state import VacuumState
from aqft_curved.algebra import AlgebraicProduct

def main():
    # Define a Hilbert space dimension
    # A larger dimension provides a better approximation of the infinite-dimensional Fock space.
    hilbert_dim = 5

    # 1. Setup the spacetime and a scalar field
    # Although the operators here are abstract, they are formally associated with a field.
    st = PredefinedSpacetime('Minkowski')
    field = ScalarField(spacetime=st, name='phi')

    # 2. Create a numerical vacuum state |0>
    vacuum = VacuumState(field=field, hilbert_dim=hilbert_dim)
    print(f"Created a numerical vacuum state in a Hilbert space of dimension {hilbert_dim}.")
    print("Vacuum state |0>:")
    print(vacuum.numerical_state)

    # 3. Define annihilation (a) and creation (a_dag) operators
    a = FieldOperator(name='a', spacetime=st, is_creation=False)
    a_dag = FieldOperator(name='a_dag', spacetime=st, is_creation=True)

    # Get their numerical representations
    a_num = a.to_numerical(hilbert_dim)
    a_dag_num = a_dag.to_numerical(hilbert_dim)

    print("\nAnnihilation operator a:")
    print(a_num)
    print("\nCreation operator a_dag:")
    print(a_dag_num)

    # 4. Construct the number operator N = a_dag * a
    number_operator = AlgebraicProduct([a_dag, a])

    # 5. Calculate the expectation value of N in the vacuum state
    # We expect <0|N|0> = 0
    exp_N_vacuum = vacuum.expectation_value(number_operator)
    print(f"\nExpectation value of Number Operator in vacuum <0|N|0> = {exp_N_vacuum.real:.4f}")

    # 6. Create a one-particle state |1> = a_dag |0>
    # We use the @ operator for matrix multiplication (operator application)
    one_particle_state_numerical = a_dag_num @ vacuum.numerical_state
    one_particle_state_numerical.normalize() # Normalize the state
    
    # Create a new State object for the one-particle state
    from aqft_curved.state import State
    one_particle_state = State(field=field, numerical_state=one_particle_state_numerical, hilbert_dim=hilbert_dim)

    print("\nOne-particle state |1> = a_dag|0>:")
    print(one_particle_state.numerical_state)

    # 7. Calculate the expectation value of N in the one-particle state
    # We expect <1|N|1> = 1
    exp_N_one_particle = one_particle_state.expectation_value(number_operator)
    print(f"\nExpectation value of Number Operator in one-particle state <1|N|1> = {exp_N_one_particle.real:.4f}")

if __name__ == "__main__":
    main()
```

**Expected Output:**

```
Created a numerical vacuum state in a Hilbert space of dimension 5.
Vacuum state |0>:
QuantumState: dims = [[5], [1]], shape = (5, 1), type = ket
Qobj data:
<Compressed Sparse Column sparse matrix of dtype 'complex128'
	with 1 stored elements and shape (5, 1)>
  Coords	Values
  (0, 0)	(1+0j)

Annihilation operator a:
QuantumState: dims = [[5], [5]], shape = (5, 5), type = oper
Qobj data:
<Compressed Sparse Column sparse matrix of dtype 'complex128'
	with 4 stored elements and shape (5, 5)>
  Coords	Values
  (0, 1)	(1+0j)
  (1, 2)	(1.4142135623730951+0j)
  (2, 3)	(1.7320508075688772+0j)
  (3, 4)	(2+0j)

Creation operator a_dag:
QuantumState: dims = [[5], [5]], shape = (5, 5), type = oper
Qobj data:
<Compressed Sparse Row sparse matrix of dtype 'complex128'
	with 4 stored elements and shape (5, 5)>
  Coords	Values
  (1, 0)	(1-0j)
  (2, 1)	(1.4142135623730951-0j)
  (3, 2)	(1.7320508075688772-0j)
  (4, 3)	(2-0j)

Expectation value of Number Operator in vacuum <0|N|0> = 0.0000

One-particle state |1> = a_dag|0>:
QuantumState: dims = [[5], [1]], shape = (5, 1), type = ket
Qobj data:
<Compressed Sparse Row sparse matrix of dtype 'complex128'
	with 1 stored elements and shape (5, 1)>
  Coords	Values
  (1, 0)	(1+0j)

Expectation value of Number Operator in one-particle state <1|N|1> = 1.0000
```

(More tutorials to be added here...)
