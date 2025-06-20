# Algebraic Quantum Field Theory in Curved Spacetime Library (aqft-cs)

## I. Current Project Status

This project is currently under active development with two parallel implementations:

1.  **Python (Symbolic) Frontend:** Located in `src/aqft_curved/`, this implementation uses `sympy` to provide a purely symbolic framework for defining spacetimes and fields. It is well-suited for theoretical analysis and deriving analytical expressions.
2.  **Rust (Numerical) Backend:** Located in `rust/aqft_core/`, this implementation provides a high-performance numerical engine for field theory calculations. It includes numerically implemented equations of motion and stress-energy tensors for scalar, charged scalar, and electromagnetic fields.

**Note:** The integration of the Rust backend into the Python package is currently on hold. The two components are developed and tested independently. You can run the Rust-specific tests by navigating to `rust/aqft_core` and running `cargo test`.
## II. Project Overview
A. Vision (Enhanced Precision)
To provide a robust, efficient, and user-friendly computational framework for Algebraic Quantum Field Theory (AQFT) in curved spacetimes (QFTCS), specifically addressing the current gap in dedicated tools. The library will empower physicists to define, manipulate, and analyze quantum fields on complex curved backgrounds, offering a seamless symbolic-to-numerical pipeline to bridge theoretical insights with practical computation and ensure scientific integrity within the semiclassical approximation.

B. Core Principles (Refined with Direct Citations from Review)
Simplicity & User-Friendliness (Python Frontend): Intuitive API design, clear documentation, and examples for quick adoption by physicists, integrating seamlessly with established scientific Python ecosystems (NumPy, SciPy, SymPy) for initial computations and high-level analysis.
Performance & Reliability (Rust Backend): High-performance computations, memory safety, and concurrency for demanding calculations, especially for large-scale tensor algebra, numerical solution of hyperbolic partial differential equations (PDEs), high-dimensional integrals, and rigorous renormalization procedures. Leveraging Rust's capabilities for memory safety and parallelism (Rayon) to avoid common pitfalls and ensure C/C++-level speed.
Mathematical Rigor & Fidelity (Central to AQFT): Strict adherence to the mathematical foundations of AQFT and General Relativity, including concepts like algebras of local observables (Weyl, CAR), Hadamard states, operator products, and the semi-classical Einstein equation (where applicable). Explicitly addresses the absence of a universally preferred vacuum by allowing selection/construction of relevant vacua.
Modularity & Extensibility: Decoupled components (geometry, algebra, QFT layers) allowing for future additions of new spacetimes, field types, and computational methods. This design promotes community contributions and allows for interoperability with specialized external libraries (e.g., FEniCS, Dedalus, or even calling into C/C++ libraries from Rust) for specific PDE solvers or advanced numerical methods.
Professionalism & Reproducibility: High-quality, well-tested, and maintainable code with clear error handling, aiming for reproducibility of results and robust error diagnostics to inform users when conditions approach or exceed physical limits (e.g., Planck scale).
C. Scope and Simplicity (Reinforcement)
The initial focus will be on free fields and linearized gravity, which are fundamental and provide a manageable starting point to ensure robustness and correctness before tackling more complex interacting theories. This directly addresses the observation that current research often involves solving specific instances of field equations (like Klein-Gordon) in defined backgrounds.

D. Target Audience
Theoretical physicists, mathematical physicists, and graduate students working on quantum field theory in curved spacetimes, cosmology, and quantum gravity.

E. Key Technologies and Dependencies
Python Frontend: NumPy (numerical arrays), SymPy (symbolic mathematics), SciPy (scientific computing), Matplotlib (visualization), Pandas (data handling).
Rust Backend: nalgebra (linear algebra), num-complex (complex numbers).
Interoperability: PyO3 or Maturin for efficient Python-Rust bindings.
## III. Architecture
A. Python Front-End (aqft_py)
Purpose: User-facing API, high-level abstractions, symbolic expression input and output, data manipulation, visualization, and scripting. Acts as the orchestrator for the symbolic-to-numerical workflow.
Key Libraries/Tools: PyO3 (for FFI), NumPy, SciPy, SymPy (for initial symbolic expression building and fallback), Matplotlib, Plotly, Pandas.
B. Rust Back-End (aqft_rs)
Purpose: Core computational engine, handling complex mathematical operations, advanced tensor algebra with index management, differential geometry, numerical methods, and non-commutative operator algebra.
Key Crates/Tools: nalgebra, ndarray, rayon, serde, symengine-rs (for core symbolic manipulation or as an option), or potentially a custom-built symbolic engine in Rust (e.g., inspired by Symbolica for performance in heavy algebraic expansions).
C. Interoperability (FFI Layer)
Mechanism: PyO3 (or Maturin) for seamless Rust-Python interoperability.
Data Exchange: Efficient transfer of numerical data (ndarray to NumPy arrays) and complex Rust structures to Python objects. (MVP) Crucially, implement zero-copy data transfer using Apache Arrow (via pyo3-arrow) for large datasets to minimize performance bottlenecks.
Error Handling: Robust error propagation from Rust to Python, providing clear and informative error messages (e.g., "Metric is singular at point X, cannot compute inverse" or "Numerical solver failed to converge, check input parameters").
D. Directory Structure
aqft_curved/
├── python/                     # Python front-end
│   ├── aqft_curved/           # Main Python package
│   │   ├── __init__.py
│   │   ├── spacetime.py       # Spacetime definitions
│   │   ├── field.py           # Quantum field operations
│   │   ├── algebra.py         # Operator algebras
│   │   ├── state.py           # States and correlation functions
│   │   └── utils.py           # Plotting, I/O, utilities
│   ├── tests/                 # Python unit tests
│   └── pyproject.toml         # Python project configuration (Maturin, Poetry, etc.)
├── rust/                      # Rust back-end
│   ├── src/                   # Rust source code
│   │   ├── lib.rs             # Rust library entry point
│   │   ├── metric.rs          # Metric and geometry computations
│   │   ├── algebra.rs         # Operator algebra implementations
│   │   ├── numerical.rs       # Numerical solvers
│   │   └── symbolic.rs        # Symbolic manipulations
│   ├── tests/                 # Rust unit tests
│   └── Cargo.toml             # Rust build configuration
├── docs/                      # Documentation
│   ├── api/                   # API reference (generated)
│   ├── tutorials/             # Jupyter notebooks for examples
│   └── user_guide.md          # User guide
├── examples/                  # Example scripts and notebooks
├── README.md                  # Project overview and installation
├── LICENSE                    # Licensing information
└── .gitignore                 # Standard Git ignore file
E. Numerical Precision and Complex Data Types
(MVP) Arbitrary Precision: Support for arbitrary-precision arithmetic (e.g., via Rust's rug crate leveraging GMP/MPFR/MPC) for calculations where standard floating-point precision (f64) may lead to significant errors or instabilities, especially in long-term simulations or near singularities. User-configurable precision levels.
(MVP) Complex Numbers: Robust and efficient handling of complex numbers throughout the Rust backend using dedicated crates (e.g., num-complex / crum), essential for quantum amplitudes and field theory. Ensure seamless mapping and transfer of complex data types across the FFI.
## IV. Core Modules (High-Level Design)
A. Python Frontend (aqft_py) Modules
1. aqft_curved.spacetime
Purpose: Define and manipulate curved spacetimes, allowing for symbolic or numerical metric definitions, and providing tools for differential geometry.
Key Classes/Methods:
(MVP) Spacetime: Attributes like dimension, metric (SymPy expression or callable), coordinates. Methods include set_metric(), christoffel_symbols(), riemann_tensor(), ricci_tensor(), ricci_scalar(), einstein_tensor(), and capabilities for geodesic computations and causal structure analysis (e.g., causal cones).
(MVP) PredefinedSpacetime: Subclass for common analytical spacetimes: Minkowski, Schwarzschild, and FLRW. These are fundamental for understanding the Unruh effect, Hawking radiation, and cosmological particle creation. (Kerr and de Sitter will be added in Phase 2/Long-term).
(MVP) CoordinateChart: For managing local coordinate systems and transformations.
2. aqft_curved.field
Purpose: Define quantum fields, their properties, and their behavior in curved backgrounds, with an emphasis on operator-valued fields.
Key Classes/Methods:
(MVP) ScalarField: Attributes like spacetime, mass, coupling. Methods include equation_of_motion() (e.g., Klein-Gordon operator ∇ 
μ
 ∇ 
μ
​
 +m 
2
 +ξR automatically constructed from metric), commutator(point1, point2) (via Rust).
(MVP) FieldOperator: Represents abstract field operators for AQFT; methods like apply(state), conjugate(), and support for Wick products/normal ordering and handling formal non-commutative algebra with user-defined commutation rules for smeared fields.
(Long-term) DiracField (Spinor fields requiring tetrads, spin connections, and rigorous Clifford algebra implementation), VectorField (Gauge fields).
3. aqft_curved.algebra
Purpose: Manage operator algebras for AQFT, enabling formal algebraic manipulation of quantum operators and local observables.
Key Classes/Methods:
(MVP) OperatorAlgebra: Attributes: generators, relations. Methods: add_generator(operator), commutator(op1, op2).
(MVP) CausalRegion: Represents spacetime regions for local algebras. Methods: is_causally_separated(region).
(Long-term) to_causal_net(): (Conceptual/Future) Construct causal net for AQFT locality, generalizing Haag-Kastler axioms.
(MVP) AlgebraicProduct: Class to represent non-commutative products of operators, supporting simplification using defined commutation relations and Wick's theorem (for free fields).
4. aqft_curved.state
Purpose: Define physically meaningful states and compute correlation functions, a critical missing piece in current computational tools, addressing the vacuum ambiguity challenge.
Key Classes/Methods:
(MVP) State: Attributes: spacetime, field. Methods: two_point_function(point1, point2), n_point_function(...) (initially for n=2), expectation_value(operator).
(MVP) VacuumState: Subclass for physically relevant vacuum states, with a particular focus on Hadamard states (essential for well-defined expectation values and renormalization) and explicit options for other vacua (e.g., Bunch-Davies for FLRW).
(Long-term) ThermalState: Subclass for thermal states (e.g., satisfying KMS condition).
5. aqft_curved.utils
Purpose: Visualization and I/O utilities.
Key Functions: plot_spacetime(spacetime) (Visualize metric, curvature invariants, geodesics), export_results(data, filename) (Save results, e.g., to CSV, HDF5), load_spacetime(filename) (Load predefined spacetime or from external data).
B. Rust Backend (aqft_rs) Core Modules
1. aqft_core::metric
Purpose: High-performance computation of geometric quantities for curved spacetimes, involving robust tensor calculus.
Key Structs/Methods:
(MVP) Metric: Fields: components (nalgebra matrix), coordinates. Methods: christoffel_symbols(), riemann_tensor(), ricci_tensor(), ricci_scalar(), einstein_tensor(), inverse().
(MVP) geodesic_integrator(metric, initial_position, initial_velocity): Solve geodesic equations numerically.
(Long-term) local_frame::Vielbein and local_frame::SpinConnection for future spinor field support.
(Long-term) differential_forms::ExteriorDerivative, differential_forms::HodgeStar for operations on differential forms, aligning with FEEC principles.
2. aqft_core::algebra
Purpose: Implement efficient operator algebra operations for AQFT, including sophisticated symbolic tensor algebra with robust index management and non-commutative operator handling.
Key Structs/Methods:
(MVP) TensorExpression: Representation of symbolic tensor expressions, including type-safe index labels (covariant/contravariant), symmetries (e.g., Riemann tensor symmetries), and implicit coordinate dependence.
(MVP) Operator: Fields: symbolic_form (Rust symbolic expression). Methods: commutator(other), wick_product(other), simplify_by_relations().
(MVP) IndexManager: Handles index raising/lowering, contraction, symmetrization/antisymmetrization, and canonicalization of indexed expressions (e.g., via Butler-Portugal algorithm for efficient simplification).
(Long-term) CliffordAlgebra: Dedicated module for Dirac gamma matrix manipulations, Fierz identities, and spinor algebra on curved backgrounds.
3. aqft_core::numerical
Purpose: High-performance numerical solvers for field equations, Green's functions, and integrals. This directly addresses the need for robust solvers for hyperbolic PDEs and efficient high-dimensional quadrature.
Key Functions:
(MVP) solve_klein_gordon(metric, mass, source): Numerically solve Klein-Gordon equation (e.g., using finite difference methods or ODE solvers for simpler cases like 1D radial equations in static spacetimes, or mode equations in FLRW).
(MVP) compute_green_function(metric, point1, point2): Numerical computation of two-point Green's functions/propagators for free scalar fields.
(MVP) integrate_expectation_value(...): Robust numerical integrators for computing expectation values (e.g., adaptive quadrature for lower dimensions).
(Long-term) Advanced PDE solvers (e.g., Finite Element Methods (FEM) via FEEC principles, or spectral methods on suitable backgrounds).
4. aqft_core::symbolic
Purpose: Core symbolic manipulations for metrics, operators, and general expressions, optimized for performance.
Key Functions: simplify_expression(expr), differentiate(expr, var), substitute_expression(expr, pattern, replacement) (for pattern matching).
Dependencies: (MVP) symengine-rs will be used as the core symbolic engine.
(Long-term) Exploration of a custom Rust symbolic engine (e.g., inspired by Symbolica) for highly specialized tensor/operator algebra if symengine-rs proves insufficient.
5. aqft_core::observables
Purpose: Calculation of key physical observables, emphasizing their derivation and intrinsic renormalization.
Key Functions/Structs:
(MVP) Functions for computing components of the renormalized Stress-Energy Tensor ⟨T 
μν
​
 ⟩ 
ren
​
  for scalar fields (using adiabatic regularization for FLRW, or Hadamard subtraction for static spacetimes).
(Long-term) Particle Number Operator, Conformal Anomalies.
6. aqft_core::pipeline
Purpose: Manages the seamless symbolic-to-numerical pipeline, translating derived equations into optimized forms for numerical computations.
Key Functions: (MVP) generate_numerical_function(symbolic_expr) (generates optimized Rust code or callable numerical function from a symbolic expression for a limited set of operations), prepare_for_solver(equation, solver_type) (formats equations for internal numerical solvers).
7. aqft_core::qinfo (New Module, Addressing Tensor Networks/Operator Algebra Numerics)
Purpose: Exploratory module for integrating concepts from quantum information theory and advanced numerical operator algebra.
(Long-term) Key Functions/Structs: Tools for numerical representation and manipulation of non-commutative operator algebras, potentially including probabilistic techniques for block diagonalization or transfer operators. Initial support for Tensor Network States (TNS) for specific QFTCS models (e.g., de Sitter spacetime), treating networks as discretized spacetime geometries.
IV. Example Use Cases (Python API)
A. Define a Curved Spacetime and Compute Curvature: (MVP)
Python

from aqft_curved.spacetime import Spacetime

M = Spacetime.predefined("Schwarzschild", mass=1.0)
M.set_coordinates("t r theta phi")
print(M.riemann_tensor())
B. Define a Scalar Field and Compute a Two-Point Function (with Renormalization focus): (MVP)
Python

from aqft_curved.field import ScalarField
from aqft_curcurved.state import VacuumState

# Define a massless scalar field on the Schwarzschild spacetime
field = ScalarField(M, mass=0.0, coupling="minimal")

# Compute two-point function for a Hadamard vacuum state
state = VacuumState(field, state_type="Hadamard") # Explicitly specify Hadamard
point1 = (0.0, 5.0, 0.0, 0.0) # (t, r, theta, phi)
point2 = (0.1, 5.1, 0.0, 0.0)
two_point = state.two_point_function(point1, point2, renormalized=True) # Request renormalized
print(f"Renormalized two-point function at specified points: {two_point}")
C. Derive and Numerically Solve a Field Equation (Simple Cases): (MVP)
Python

from aqft_curved.spacetime import Spacetime
from aqft_curved.field import ScalarField

M_FLRW = Spacetime.predefined("FLRW", scale_factor_a="a(t)") # Symbolic scale factor
phi = ScalarField(M_FLRW, mass=0.1, coupling=0.0)

# Derive the Klein-Gordon equation symbolically in FLRW background
kg_equation_symbolic = phi.equation_of_motion()
print(f"Klein-Gordon Equation (symbolic): {kg_equation_symbolic}")

# Numerically solve the derived equation for a specific initial condition
# (This would call into the Rust numerical backend via the pipeline)
# Assuming a function for setting up initial conditions and calling the solver
solution_numeric = M_FLRW.solve_field_equation(kg_equation_symbolic, initial_conditions={"phi": 1.0, "dphi_dt": 0.0}, num_points=100)
# Further analysis/plotting of solution_numeric
V. Development & Deployment
A. Build System (MVP)
Rust: Cargo (for build, test, and dependency management).
Python: Maturin (for building Python-Rust bindings and managing Python packaging via pyproject.toml).
B. Testing (MVP)
Unit Tests: pytest for Python, cargo test for Rust.
Integration Tests: For Python-Rust interactions and module interoperability.
End-to-End Tests: Higher-level tests (e.g., Jupyter notebooks) to validate physical results against known analytical solutions (e.g., Unruh effect, basic Hawking radiation for scalar fields, particle creation in simple FLRW models) or benchmark data, ensuring numerical accuracy and stability.
Property-Based Testing (Rust): For robustness and edge cases in symbolic manipulation and numerical routines.
C. Documentation (MVP)
User Guide (docs/user_guide.md): Comprehensive installation, quickstart, overview of key classes/methods, and FAQ.
API Reference (docs/api/): Auto-generated from Python docstrings (Sphinx) and Rust comments (rustdoc), including examples for each class/method.
Tutorials (docs/tutorials/): Jupyter notebooks demonstrating common workflows (e.g., "Scalar QFT on Minkowski spacetime," "Hadamard State Construction and Two-Point Functions in Schwarzschild," "Adiabatic Regularization of Stress-Energy Tensor in FLRW Universe," "Unruh Effect Simulation").
Theoretical Background: Brief but rigorous explanations of underlying AQFT and General Relativity concepts, referencing key theoretical literature (e.g., Wald, Hollands & Wald, Ford, Parker & Toms), to provide essential context for computational methods and design choices (e.g., why Hadamard states are crucial).
D. Packaging & Distribution (MVP)
Python: Distribute via PyPI using Maturin for robust Python-Rust binding compilation and wheel generation (pip install aqft-curved).
Rust: The Rust library will be compiled as a shared library bundled with the Python package.
Dependencies: Aim for a minimal set of direct external dependencies to ensure portability and ease of installation, addressing the "too many disparate tools" issue.
E. Versioning Strategy (MVP)
Adopt semantic versioning (MAJOR.MINOR.PATCH) to clearly communicate API changes and new features, ensuring predictability for users.
F. Performance and Scalability Considerations
(MVP) Parallelism: Support for shared-memory parallelism (Rayon) for key numerical computations.
(MVP) Memory Efficiency: Design with efficient memory management and data structures to handle large datasets, minimizing memory traffic. Leverage Rust's ownership model for safety and performance.
(Long-term) GPU Acceleration: The ability to offload computationally intensive tasks to GPUs (e.g., via wgpu or cuda-rust) will be explored for suitable algorithms.
(MVP) Numerical Precision: The library will allow user-configurable arbitrary-precision arithmetic where necessary, balancing accuracy with computational cost.
VI. Future Enhancements (Long-Term Goals)
Beyond Free Fields: Lay the groundwork for perturbative interacting QFTs (e.g., λϕ 
4
  theory in linearized gravity), potentially incorporating Feynman diagram structures and loop calculations for curved spacetime.
Higher-Spin Fields: Extend support to Dirac (Spinor) fields (requiring robust tetrad and spin connection formalism, and Clifford algebra implementation) and Electromagnetic/Gauge fields (vector fields), and eventually higher-spin fields.
Advanced Renormalization Techniques: Further explore and implement more sophisticated renormalization methods beyond adiabatic, such as point-splitting with specific choices of bidistributions, or heat kernel regularization for effective actions.
Integration with Numerical Relativity: Define clean, standardized interfaces (e.g., based on HDF5 or other common data formats used by existing Numerical Relativity codes like Einstein Toolkit or GRChombo) for importing dynamic metric data from numerical relativity simulations.
Machine Learning Integration: Explore applications of machine learning methods to improve the efficiency of QFTCS calculations, for example, in sparse linear algebra or Monte Carlo techniques.
Quantum Computing Interfaces: Develop interfaces for simulating QFTCS on emergent quantum computing platforms, particularly leveraging Tensor Network State methodologies for mapping field theory degrees of freedom onto qubits.
Category-Theoretic Structures: (Highly advanced) Explore how formal AQFT structures (e.g., functors from regions to algebras, nets of algebras beyond simple causal diamonds) could be represented and utilized.
VII. Licensing (MVP)
License: MIT or Apache-2.0 for open-source distribution. Include LICENSE file and specify in README.md.
VIII. Development Roadmap
Phase 1: MVP Core Functionality (e.g., 3-6 months)
Implement basic spacetime module in Python (Spacetime class, PredefinedSpacetime for Minkowski, Schwarzschild, FLRW).
Basic geometric computations in Rust (metric.rs for Christoffel, Riemann, Ricci for static/simple metrics).
Initial field module in Python for ScalarField.
Basic state module for VacuumState focusing on Hadamard states for scalar fields.
Core numerical.rs for solve_klein_gordon (1D radial equations in static spacetimes, mode equations in FLRW using finite difference or ODE solvers) and compute_green_function (two-point function for scalar fields).
Initial observables.rs for renormalized stress-energy tensor (scalar field in FLRW using adiabatic regularization, or static using Hadamard subtraction).
Set up aqft_core::algebra with TensorExpression and IndexManager for symbolic tensor manipulation (e.g., deriving Klein-Gordon operator from metric).
Establish Python-Rust FFI with PyO3/Maturin and basic data transfer, including Apache Arrow for critical data transfers.
Core pipeline.rs for converting symbolic equations to numerical functions (limited set of operations).
Implement utils.py for basic plotting and I/O.
Set up comprehensive unit tests (Python & Rust) and basic end-to-end tests for MVP features.
Initial user guide and tutorials (Jupyter notebooks) covering MVP examples.
Establish build system, packaging, and versioning.
Phase 2: Expanding & Refining MVP (e.g., 6-12 months)
Expand PredefinedSpacetime to include Kerr, de Sitter.
Improve numerical.rs with more robust PDE solvers (e.g., finite difference for 2D field equations) and more sophisticated integrators.
Refine aqft_core::algebra with more advanced operator handling and simplifications.
Add initial AlgebraicProduct and CausalRegion functionality.
Refine renormalization methods for diverse scalar field scenarios.
More comprehensive examples and tutorials.
Focus on performance optimization of core Rust routines.
Phase 3: Long-Term Goals & Enhancements (Ongoing)
Introduce DiracField and VectorField with associated geometric and algebraic complexities.
Develop qinfo module for tensor networks and numerical operator algebras.
Explore advanced renormalization techniques and quantum information measures.
Investigate GPU acceleration.
Formalize interfaces for Numerical Relativity integration.
Research and implement more advanced concepts from AQFT (e.g., full nets of algebras, operator product expansions).
IX. Key Decisions and Rationale
This section summarizes the key architectural and implementation decisions made, informed by the literature review, balancing the need for a focused MVP with ambitious long-term goals.

1. Core Symbolic Manipulation Engine (Rust Backend - aqft_core::symbolic)
Decision: For the MVP, symengine-rs will be used as the core symbolic engine.
Rationale: The literature highlights that building a full CAS is a massive undertaking. SymEngine (C++ core with Rust/Python wrappers) is a pragmatic choice for the MVP due to its maturity and significant speedups over pure Python (SymPy). This allows immediate focus on physics-specific features.
Long-term: Exploration of a custom Rust symbolic engine (e.g., inspired by Symbolica) for highly specialized tensor/operator algebra will occur if symengine-rs proves insufficient for complex calculations (e.g., very large algebraic expansions, advanced index canonicalization, or non-commutative operator algebra). Ideas from Cadabra's design for handling indexed variables and symmetries will inform the aqft_core::algebra module.
2. Numerical PDE Solver Strategy (Rust Backend - aqft_core::numerical)
Decision: For the MVP, numerical solutions for field equations (e.g., Klein-Gordon) will primarily use Rust-native implementations of basic finite difference methods or ODE solvers for simpler cases (e.g., 1D radial equations in static spacetimes, mode equations in FLRW).
Rationale: These methods are straightforward to implement and provide immediate utility for foundational QFTCS problems. This avoids the immediate complexity of integrating with larger external PDE frameworks.
Long-term: The strategic goal is to adopt Finite Element Methods (FEM), particularly FEEC, for its ability to handle arbitrary geometries and preserve symmetries, making it a more natural and potentially more accurate discretization method for QFTCS. Integration with (or contribution to) Rust-based FEM libraries (e.g., Fenris) will be pursued. Spectral methods will be considered as a complementary approach for suitable backgrounds.
3. Renormalization Scheme Integration (Rust Backend - aqft_core::observables)
Decision: For the MVP, renormalization will be intrinsically integrated into the computation of physical observables. Specifically, adiabatic regularization will be implemented for FLRW spacetimes, and Hadamard subtraction for static spacetimes, for two-point functions and stress-energy tensor.
Rationale: The literature unequivocally states that renormalization must be an intrinsic part of the algorithms, not a post-processing step, to yield finite, physically meaningful results in QFTCS. These methods are well-established and numerically convenient.
4. Operator Algebra Representation (Rust Backend - aqft_core::algebra)
Decision: For the MVP, the library will focus on formal algebraic manipulation of symbols representing smeared field operators, enforcing canonical commutation/anti-commutation relations for free fields and supporting Wick products. The AlgebraicProduct class in Python will call Rust for simplification based on defined rules.
Rationale: This approach provides a practical starting point for handling non-commutative algebra, analogous to existing physics-oriented symbolic modules, and is sufficient for free field theories in the MVP scope.
Long-term: Explore more sophisticated "numerical operator algebra" techniques (e.g., probabilistic block diagonalization, transfer operators) and direct support for CliffordAlgebra for Dirac fields.
5. Distributed-Memory Parallelism (Rust Backend - General)
Decision: The MVP will not include direct distributed-memory parallelism (MPI) support.
Rationale: Rust's native MPI ecosystem is still maturing, and direct FFI calls to C++/Fortran MPI libraries introduce significant complexity and unsafe code. Focusing on robust shared-memory parallelism via Rayon for within-node performance keeps the MVP "simple" and "straightforward" while still offering substantial speedups.
Long-term: Re-evaluate MPI integration if large-scale, multi-node simulations become essential for advanced features, acknowledging the associated development overhead.
6. Data Transfer for Large Structures (Python-Rust Bridge)
Decision: For the MVP, Apache Arrow (via pyo3-arrow) will be implemented for critical zero-copy data transfers of large numerical arrays/tensors between Python and Rust.
Rationale: The literature strongly advocates for Arrow to avoid performance bottlenecks from data copying, which is crucial for computationally intensive scientific applications.
7. Integration with External Numerical Libraries (Rust Backend - aqft_core::numerical & aqft_core::pipeline)
Decision: For the MVP, solve_klein_gordon and other numerical routines will primarily use Rust-native implementations of basic numerical methods (finite difference, ODE solvers).
Rationale: This avoids immediate FFI complexity with larger external C++/Fortran PDE frameworks.
Long-term: If native implementations prove insufficient for complex problems (e.g., highly complex geometries, non-linear problems), strategically explore FFI to established C/C++ libraries (e.g., PETSc, GSL) for specialized, highly optimized solvers, carefully managing the unsafe boundaries. The aqft_core::pipeline will be designed to facilitate this interoperability.
8. Predefined Spacetimes for MVP
Decision: For the MVP, the library will include Minkowski, Schwarzschild, and FLRW predefined spacetimes.
Rationale: These spacetimes are fundamental for understanding key QFTCS phenomena like the Unruh effect, Hawking radiation, and cosmological particle creation, directly aligning with the core problems the library aims to address first. Kerr and de Sitter spacetimes will be added in Phase 2.
