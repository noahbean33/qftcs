# AQFT: A Library for Quantum Field Theory in Curved Spacetime

## Project Vision: The Static Black Hole Explorer

This library is a Python toolkit for performing calculations in Algebraic Quantum Field Theory (AQFT) in curved spacetimes. The project is currently focused on developing a Minimum Viable Product (MVP) called the **Static Black Hole Explorer**.

The primary goal of the MVP is to compute the **Renormalized Stress-Energy Tensor (RSET)** for a massless scalar field in Schwarzschild spacetime, providing a powerful tool for studying quantum effects near black holes.

### MVP Key Features

The Static Black Hole Explorer will include:

- **Symbolic Spacetime Geometry**: Automated calculation of geometric quantities like Christoffel symbols, Riemann and Ricci tensors, and the Ricci scalar using `sympy`.
- **Radial Mode Solver**: Numerical solution of the radial Klein-Gordon equation using `scipy`'s ODE solvers.
- **Hartle-Hawking Vacuum**: Implementation of the Hartle-Hawking quantum state, which is essential for describing the black hole's thermal atmosphere.
- **Mode-Sum Renormalization**: Regularization of the RSET using the Extended Coordinate Method to obtain physically meaningful, finite results.
- **Validation**: Verification of results against canonical papers, such as Howard and Candelas (1984).

## Current Status

The foundational symbolic modules are in place. The library can currently:
- Define predefined spacetimes like Minkowski and Schwarzschild.
- Symbolically compute all relevant geometric tensors.
- Define scalar fields and derive their equations of motion.

The next development phase focuses on implementing the numerical mode solver and the renormalization scheme.

## Installation

To install the library and its dependencies, clone the repository and run `pip` in editable mode from the project root:

```bash
git clone https://github.com/noahbean33/aqft_py.git
cd aqft_py
pip install -e .
```

## Usage Example

Here is a simple example of how to initialize Schwarzschild spacetime and compute its Ricci scalar:

```python
import sympy
from aqft.spacetime import PredefinedSpacetime

# 1. Initialize Schwarzschild spacetime with mass M
M = sympy.symbols('M')
schwarzschild = PredefinedSpacetime('Schwarzschild', M=M)

# 2. Get the coordinate system
t, r, theta, phi = schwarzschild.coords

# 3. Compute the Ricci scalar
ricci_scalar = schwarzschild.ricci_scalar()

print("Schwarzschild Spacetime")
print("Coordinates:", (t, r, theta, phi))
print("Ricci Scalar:", ricci_scalar)
# Expected output: 0
```

## Project Structure

The project has been streamlined to focus on a pure Python implementation.

```
aqft_py/
├── docs/
│   └── user_guide.md
├── examples/
│   ├── minkowski_example.py
│   └── schwarzschild_scalar_field_tutorial.py
├── src/
│   └── aqft/
│       ├── __init__.py
│       ├── spacetime.py
│       ├── field.py
│       ├── algebra.py
│       ├── state.py
│       └── utils.py
├── tests/
│   ├── test_spacetime.py
│   └── ...
├── pyproject.toml
└── README.md
```

## Long-Term Vision

Beyond the initial MVP, the long-term goal is to evolve this library into a modular, high-performance toolkit for QFT in curved spacetime, featuring:
- A layered, extensible architecture.
- Support for other fields (e.g., Dirac, Maxwell) and spacetimes.
- Potential integration with high-performance computing (HPC) libraries.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

