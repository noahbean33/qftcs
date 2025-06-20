# AQFT Python Package User Guide

## Introduction

Welcome to the AQFT (Algebraic Quantum Field Theory in Curved Spacetime) Python package! This guide provides an overview of the package, its core concepts, and examples to help you get started.

## Rust Backend (Numerical Engine)

Alongside the symbolic Python implementation, this project includes a high-performance numerical backend written in Rust, located in the `rust/aqft_core` directory. The Rust backend is designed for computationally intensive tasks, providing a fast and memory-safe engine for numerical field theory calculations.

**Note:** The Rust backend is currently developed and tested separately from the Python package.

### Architecture

The Rust backend is organized into two main modules:

-   **`spacetime`**: Provides the `Spacetime` struct, which handles all geometric calculations. It takes a function defining the metric tensor and numerically computes derivatives to find the Christoffel symbols, Ricci tensor, and Ricci scalar.
-   **`field`**: Contains implementations for various quantum fields. It uses the `spacetime` module to compute equations of motion and other physical quantities in a given gravitational background.

### Implemented Features

-   **Scalar Field (`ScalarField`)**:
    -   Klein-Gordon equation of motion.
    -   Stress-energy tensor, including non-minimal coupling to the Ricci scalar.
-   **Electromagnetic Field (`ElectromagneticField`)**:
    -   Representation via a 4-potential `A_μ`.
    -   Numerical computation of the field strength tensor `F_μν`.
-   **Charged Scalar Field (`ChargedScalarField`)**:
    -   Gauge-covariant Klein-Gordon equation of motion.
    -   Interaction with a background `ElectromagneticField`.

### Running Tests

To verify the correctness of the numerical implementations, you can run the comprehensive suite of unit tests. Navigate to the Rust directory and use Cargo:

```bash
cd rust/aqft_core
cargo test
```

This will compile and run all tests, confirming that the geometric and field calculations are correct (e.g., in Minkowski spacetime).

## Core Concepts

(Explanation of core concepts like Spacetime, FieldOperator, ScalarField, geometric quantities, etc., to be added here...)

## Core Module Features

This section details the key functionalities provided by the core Python modules of the AQFT package.

(More features to be detailed here...)

### Stress-Energy Tensor for Scalar Fields

The package can compute the classical stress-energy tensor `T_μν` for a `ScalarField`. This tensor is a fundamental quantity in general relativity and field theory, describing the distribution of energy and momentum in spacetime. The `ScalarField` class provides a method `stress_energy_tensor()` that symbolically calculates `T_μν` using the formula:

`T_μν = (∇_μ φ)(∇_ν φ) - g_μν [1/2 ((∇_α φ)(∇^α φ) + m² φ²) + 1/2 ξ R φ²]`
         `+ ξ [R_μν φ² - (∇_μ ∇_ν φ²) + g_μν (□ φ²)]`

where `φ` is the scalar field, `m` its mass, `ξ` the coupling to the Ricci scalar `R`, `g_μν` the metric tensor, `R_μν` the Ricci tensor, and `□` the d'Alembertian operator. The calculation includes terms for non-minimal coupling (`ξ ≠ 0`).

An example demonstrating this feature can be found in `examples/stress_energy_tensor_example.py`.

**Example Script (`examples/stress_energy_tensor_example.py`):**

```python
import sympy

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField

def run_stress_energy_tensor_example():
    """
    Demonstrates the calculation of the stress-energy tensor for a scalar field
    in Minkowski spacetime.
    """
    print("--- Stress-Energy Tensor Example (Scalar Field in Minkowski) ---")

    # 1. Initialize Minkowski Spacetime
    # ---------------------------------
    print("\nInitializing 4D Minkowski spacetime...")
    minkowski_spacetime = PredefinedSpacetime('Minkowski')
    print(f"Spacetime Coordinates: {minkowski_spacetime.coords}")
    print("Minkowski Metric Tensor:")
    print(str(minkowski_spacetime.metric))

    # 2. Define a Scalar Field
    # ------------------------
    # Define symbolic mass 'm' and coupling constant 'xi'
    m_sym = sympy.Symbol('m')
    xi_sym = sympy.Symbol('xi')

    print(f"\nDefining a scalar field with mass m={m_sym} and coupling xi={xi_sym}...")
    scalar_field = ScalarField(spacetime=minkowski_spacetime, mass=m_sym, coupling_xi=xi_sym)
    print(f"Scalar field symbol: {scalar_field.name}")

    # 3. Compute Stress-Energy Tensor
    # -------------------------------
    print("\nComputing the stress-energy tensor T_munu...")
    T_munu = scalar_field.stress_energy_tensor()
    
    print("Stress-Energy Tensor T_munu:")
    # Using str() to avoid potential Unicode issues on some consoles for the matrix
    print(str(T_munu))

    print("\n--- Stress-Energy Tensor Example Complete ---")

if __name__ == "__main__":
    run_stress_energy_tensor_example()

```

**Expected Output:**

The script will initialize a Minkowski spacetime, define a scalar field with symbolic mass `m` and coupling `xi`, and then compute and print the symbolic stress-energy tensor `T_μν`. The output will be a SymPy matrix representing the tensor components.

### Electromagnetic Field

The `ElectromagneticField` class represents the electromagnetic field, defined by a 4-potential `A_μ`. It provides methods to compute key physical quantities:

*   **Field Strength Tensor (`field_strength_tensor`)**: Computes `F_μν = ∂_μ A_ν - ∂_ν A_μ`.
*   **Maxwell's Equations (`equation_of_motion`)**: Computes the source-free equations `∇_μ F^μν = 0`.
*   **Stress-Energy Tensor (`stress_energy_tensor`)**: Computes the tensor `T_μν` for the EM field.

**Example Script (`examples/em_stress_energy_example.py`):**

```python
import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ElectromagneticField

def em_stress_energy_example():
    """
    An example to demonstrate the computation of the stress-energy tensor
    for the electromagnetic field in Minkowski spacetime.
    """
    print("--- Electromagnetic Field Stress-Energy Tensor Example ---")

    # Initialize a 4D Minkowski spacetime
    minkowski = PredefinedSpacetime("Minkowski")
    print("\nInitializing Minkowski spacetime and an EM field...")
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # Initialize the EM field. The 4-potential is created symbolically by default.
    em_field = ElectromagneticField(minkowski)
    print(f"EM Field 4-Potential: {em_field.name.T}")

    # Compute the stress-energy tensor
    print("\nComputing the stress-energy tensor T_munu...")
    T_munu = em_field.stress_energy_tensor()

    print("\nStress-Energy Tensor for the Electromagnetic Field (T_munu):")
    sympy.pprint(T_munu, use_unicode=False)

    print("\n--- Electromagnetic Field Stress-Energy Tensor Example Complete ---")

if __name__ == "__main__":
    em_stress_energy_example()
```

**Expected Output:**

The script will print the symbolic 4-potential and the full stress-energy tensor for the electromagnetic field. The output is a large symbolic matrix.

### Charged Scalar Field

The `ChargedScalarField` class models a scalar field that couples to an `ElectromagneticField`. This is achieved through the **gauge covariant derivative**, `D_μ = ∇_μ - iqA_μ`, where `q` is the field's charge and `A_μ` is the electromagnetic 4-potential.

Key features include:

*   **Equation of Motion**: The Klein-Gordon equation is modified to use the gauge covariant derivative: `(D_μ D^μ - m²)φ = 0`.
*   **Stress-Energy Tensor**: The stress-energy tensor is also modified to be gauge-invariant.

**Equation of Motion Example (`examples/charged_scalar_field_example.py`):**

```python
import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ChargedScalarField, ElectromagneticField

def charged_scalar_field_example():
    """
    An example to demonstrate the equation of motion for a charged scalar field
    coupled to an electromagnetic field.
    """
    print("--- Charged Scalar Field Example ---")

    # Initialize spacetime and fields
    minkowski = PredefinedSpacetime("Minkowski")
    em_field = ElectromagneticField(minkowski)
    charged_field = ChargedScalarField(minkowski, mass=sympy.Symbol('m'), charge=sympy.Symbol('q'), em_field=em_field)

    print("\nInitializing Minkowski spacetime, an EM field, and a charged scalar field...")
    print(f"Spacetime Coordinates: {minkowski.coords}")
    print(f"Scalar Field Symbol: {charged_field.name}")
    print(f"EM Field 4-Potential: {em_field.name.T}")

    # Compute the equation of motion
    print("\nComputing the equation of motion (D_mu D^mu - m^2) * phi = 0...")
    eom = charged_field.equation_of_motion()

    print("\nEquation of Motion for the Charged Scalar Field (0 =):")
    sympy.pprint(eom.lhs, use_unicode=False)

    print("\n--- Charged Scalar Field Example Complete ---")

if __name__ == "__main__":
    charged_scalar_field_example()
```

**Stress-Energy Tensor Example (`examples/charged_scalar_stress_energy_example.py`):**

```python
import sympy
from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ChargedScalarField, ElectromagneticField

def charged_scalar_stress_energy_example():
    """
    An example to demonstrate the computation of the stress-energy tensor
    for a charged scalar field coupled to an electromagnetic field.
    """
    print("--- Charged Scalar Field Stress-Energy Tensor Example ---")

    # Initialize a 4D Minkowski spacetime
    minkowski = PredefinedSpacetime("Minkowski")
    print("\nInitializing Minkowski spacetime, an EM field, and a charged scalar field...")
    print(f"Spacetime Coordinates: {minkowski.coords}")

    # Initialize the fields
    em_field = ElectromagneticField(minkowski)
    charged_field = ChargedScalarField(minkowski, mass=sympy.Symbol('m'), charge=sympy.Symbol('q'), em_field=em_field)
    print(f"Charged Scalar Field Symbol: {charged_field.name}")
    print(f"EM Field 4-Potential: {em_field.name.T}")

    # Compute the stress-energy tensor
    print("\nComputing the stress-energy tensor T_munu for the charged scalar field...")
    T_munu = charged_field.stress_energy_tensor()

    print("\nStress-Energy Tensor for the Charged Scalar Field (T_munu):")
    sympy.pprint(T_munu, use_unicode=False)

    print("\n--- Charged Scalar Field Stress-Energy Tensor Example Complete ---")

if __name__ == "__main__":
    charged_scalar_stress_energy_example()
```

```
--- Stress-Energy Tensor Example (Scalar Field in Minkowski) ---

Initializing 4D Minkowski spacetime...
Spacetime Coordinates: (t, x, y, z)
Minkowski Metric Tensor:
Matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

Defining a scalar field with mass m=m and coupling xi=xi...
Scalar field symbol: phi(t, x, y, z)

Computing the stress-energy tensor T_munu...
Stress-Energy Tensor T_munu:
Matrix([[...complex symbolic expression for T_00...], ... ],
        [...                                       ...], ... ]]) 
        
--- Stress-Energy Tensor Example Complete ---
```
(Note: The symbolic output for `T_μν` is very long and has been abbreviated.)

## Examples and Tutorials

### Scalar Field in FLRW Spacetime

This tutorial demonstrates how to work with a scalar field in a Friedmann-Lemaître-Robertson-Walker (FLRW) spacetime. It covers:

1.  Initializing an FLRW spacetime with a specific scale factor `a(t)` and curvature `k`.
2.  Defining a massless scalar field on this spacetime.
3.  Computing and displaying the Klein-Gordon equation for the scalar field.
4.  Visualizing a component of the FLRW metric tensor.

**Tutorial Script (`examples/flrw_scalar_field_tutorial.py`):**

```python
import sympy
import numpy as np
import matplotlib.pyplot as plt

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField
from aqft_curved.utils import plot_metric_component

def run_flrw_scalar_field_tutorial():
    """
    Runs the FLRW scalar field tutorial.
    """
    print("--- FLRW Scalar Field Tutorial ---")
    print("\nInitializing FLRW spacetime with scale factor a(t) = t and k=0...")

    # 1. Initialize FLRW Spacetime
    # -----------------------------
    # Define symbolic time 't' for the scale factor
    t_symbol = sympy.symbols('t')
    scale_factor_a = t_symbol  # Example: a(t) = t

    # Create an FLRW spacetime with k=0 (flat spatial sections)
    # The coordinates are (t, x, y, z) by default.
    # The scale factor 'a' is passed as a symbolic expression involving 't'.
    # The curvature 'k' is passed as a numerical value.
    flrw_spacetime = PredefinedSpacetime('FLRW', a=scale_factor_a, k=0)

    print("FLRW Metric Tensor:")
    print(str(flrw_spacetime.metric))

    # 2. Define a Massless Scalar Field
    # ---------------------------------
    print("\nDefining a massless scalar field on FLRW spacetime...")
    # The scalar field 'phi' will depend on all spacetime coordinates (t, x, y, z)
    scalar_field = ScalarField(name='phi', spacetime=flrw_spacetime, mass=0)
    print(f"Scalar field symbol: {scalar_field.name}")

    # 3. Compute Klein-Gordon Equation
    # --------------------------------
    print("\nComputing the Klein-Gordon equation (Box phi = 0) for the massless scalar field...")
    kg_equation = scalar_field.equation_of_motion()
    print("Klein-Gordon Equation (Box phi):")
    print(str(kg_equation))
    print("(Should be equal to 0 for a solution)")

    # 4. Plot a Metric Component (Optional)
    # -------------------------------------
    print("\nPlotting the g_xx metric component...")
    # We will plot the g_xx component (index (1,1) in (t,x,y,z) order)
    # as a function of 'x' and 't', keeping 'y' and 'z' fixed.

    # Define the coordinate ranges for the plot
    # x_coord is 'x', y_coord is 't' for the plot axes
    # The actual component g_xx = metric[1,1] in (t,x,y,z) order
    # The plot_metric_component function expects a 'ranges' dictionary where:
    # - keys are the coordinate symbols
    # - values for plot axes are (min, max) tuples
    # - values for fixed coordinates are numbers
    ranges_dict = {
        flrw_spacetime.coords[0]: (1, 5),  # t: range (1, 5) - y-axis of plot
        flrw_spacetime.coords[1]: (0, 5),  # x: range (0, 5) - x-axis of plot
        flrw_spacetime.coords[2]: 0,       # y: fixed at 0
        flrw_spacetime.coords[3]: 0        # z: fixed at 0
    }

    plot_metric_component(
        spacetime=flrw_spacetime,
        component=(1, 1),  # g_xx
        ranges=ranges_dict,
        # title=f"FLRW Metric Component g_xx(x, t) for a(t)={scale_factor_a}, k=0", # Title is now auto-generated
        n_points=50
    )
    print("Plot for g_xx generated. Please close the plot window to continue.")

    print("\n--- Tutorial Complete ---")

if __name__ == "__main__":
    run_flrw_scalar_field_tutorial()

```

**Expected Output:**

The script will print the FLRW metric tensor and the computed Klein-Gordon equation to the console. It will then display a 2D plot of the `g_xx` metric component. After closing the plot, the script will print "--- Tutorial Complete ---".

```
--- FLRW Scalar Field Tutorial ---

Initializing FLRW spacetime with scale factor a(t) = t and k=0...
FLRW Metric Tensor:
Matrix([[-1, 0, 0, 0], [0, t**2, 0, 0], [0, 0, t**2, 0], [0, 0, 0, t**2]])

Defining a massless scalar field on FLRW spacetime...
Scalar field symbol: phi(t, x, y, z)

Computing the Klein-Gordon equation (Box phi = 0) for the massless scalar field...
Klein-Gordon Equation (Box phi):
Eq(Piecewise(((-t**2*Derivative(phi(t, x, y, z), (t, 2)) + Derivative(phi(t, x, y, z), (x, 2)) + Derivative(phi(t, x, y, z), (y, 2)) + Derivative(phi(t, x, y, z), (z, 2)))/t**2, Eq(t**6, 0)), (...complex expression...), True)), 0)
(Should be equal to 0 for a solution)

Plotting the g_xx metric component...
Plot for g_xx generated. Please close the plot window to continue.

--- Tutorial Complete ---
```
(Note: The symbolic output for the Klein-Gordon equation can be quite long and complex, so it's abbreviated above.)

### Scalar Field in Schwarzschild Spacetime

This tutorial demonstrates how to work with a scalar field in a Schwarzschild spacetime. It covers:

1.  Initializing a Schwarzschild spacetime with a mass parameter `M`.
2.  Defining a massless scalar field on this spacetime.
3.  Computing and displaying the Klein-Gordon equation for the scalar field, both with a symbolic `M` and a numerical `M`.
4.  Visualizing a component (e.g., `g_tt`) of the Schwarzschild metric tensor.

**Tutorial Script (`examples/schwarzschild_scalar_field_tutorial.py`):**

```python
import sympy
import numpy as np
import matplotlib.pyplot as plt

# Ensure SymPy is using Unicode for pretty printing if available, otherwise use ASCII
try:
    sympy.init_printing(use_unicode=True)
except ImportError:
    sympy.init_printing(use_unicode=False)

from aqft_curved.spacetime import PredefinedSpacetime
from aqft_curved.field import ScalarField
from aqft_curved.utils import plot_metric_component

def run_schwarzschild_scalar_field_tutorial():
    """
    Runs the Schwarzschild scalar field tutorial.
    """
    print("--- Schwarzschild Scalar Field Tutorial ---")

    # Define Schwarzschild mass M (can be symbolic or numerical)
    M_symbol = sympy.symbols('M')
    # For actual calculations like plotting or numerical solutions, M needs a value.
    # We'll use M=1 for this tutorial when plotting.
    M_value = 1 

    print(f"\nInitializing Schwarzschild spacetime with mass M = {M_value}...")

    # 1. Initialize Schwarzschild Spacetime
    # -------------------------------------
    # The PredefinedSpacetime class handles symbolic M internally and substitutes it if a numerical value is given.
    # For symbolic calculations (like the EOM), M can remain symbolic.
    # For plotting, we need to ensure M is substituted.
    schwarzschild_spacetime_symbolic_M = PredefinedSpacetime('Schwarzschild', M=M_symbol)
    schwarzschild_spacetime_numeric_M = PredefinedSpacetime('Schwarzschild', M=M_value)

    print("Schwarzschild Metric Tensor (symbolic M):")
    # Using str() to avoid potential Unicode issues on some consoles
    print(str(schwarzschild_spacetime_symbolic_M.metric.subs(M_symbol, M_value))) # Display with M=1 for compactness

    # 2. Define a Massless Scalar Field
    # ---------------------------------
    print("\nDefining a massless scalar field on Schwarzschild spacetime...")
    # We use the spacetime with symbolic M for deriving the EOM, as it's more general.
    # The scalar field 'phi' will depend on (t, r, theta, phi)
    # The name 'phi' is assigned by default within the ScalarField constructor.
    scalar_field = ScalarField(spacetime=schwarzschild_spacetime_symbolic_M, mass=0)
    print(f"Scalar field symbol: {scalar_field.name}")

    # 3. Compute Klein-Gordon Equation
    # --------------------------------
    print("\nComputing the Klein-Gordon equation (Box phi = 0) for the massless scalar field...")
    # This will be symbolic in M
    kg_equation = scalar_field.equation_of_motion()
    print("Klein-Gordon Equation (Box phi) (symbolic M):")
    print(str(kg_equation))
    print("Substituting M=1 for a more specific form:")
    print(str(kg_equation.subs(M_symbol, M_value)))
    print("(Should be equal to 0 for a solution)")

    # 4. Plot a Metric Component (Optional)
    # -------------------------------------
    print("\nPlotting the g_tt metric component...")
    # We will plot the g_tt component (index (0,0))
    # as a function of 'r' and 'theta', keeping 't' and 'phi' fixed.
    # We use the spacetime instance where M has been substituted with a numerical value.
    
    # Coordinates are (t, r, theta, phi)
    # Let's plot g_tt(r, theta) at t=0, phi=0
    # Ranges for r: (2M+epsilon to 10M), theta: (0 to pi)
    # Ensure r_min > 2M to avoid singularity
    r_min = 2 * M_value + 0.1 
    r_max = 10 * M_value

    ranges_dict = {
        schwarzschild_spacetime_numeric_M.coords[0]: 0,              # t = 0 (fixed)
        schwarzschild_spacetime_numeric_M.coords[1]: (r_min, r_max), # r: range
        schwarzschild_spacetime_numeric_M.coords[2]: (0, np.pi),     # theta: range
        schwarzschild_spacetime_numeric_M.coords[3]: 0               # phi = 0 (fixed)
    }

    plot_metric_component(
        spacetime=schwarzschild_spacetime_numeric_M,
        component=(0, 0),  # g_tt
        ranges=ranges_dict,
        n_points=50
    )
    print(f"Plot for g_tt generated with M={M_value}. Please close the plot window to continue.")

    print("\n--- Schwarzschild Scalar Field Tutorial Complete ---")

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

(More tutorials to be added here...)
