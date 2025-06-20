import sympy
from .spacetime import Spacetime

class FieldOperator:
    """
    Represents an abstract field operator in AQFT.
    """
    def __init__(self, name, spacetime):
        """
        Initializes a FieldOperator.

        Parameters:
            name (str): The symbolic name of the operator.
            spacetime (Spacetime): The spacetime on which the operator is defined.
        """
        self.name = name
        self.spacetime = spacetime

    def apply(self, state):
        """Applies the operator to a given state (placeholder)."""
        print(f"Applying operator {self.name} to state (not implemented).")
        return None

    def conjugate(self):
        """Returns the conjugate of the operator (placeholder)."""
        print(f"Returning conjugate of operator {self.name} (not implemented).")
        return FieldOperator(f"{self.name}^*", self.spacetime)

    def __repr__(self):
        return str(self.name)

class ScalarField(FieldOperator):
    """
    Represents a scalar quantum field and provides methods to derive its
    equation of motion in a given spacetime.
    """
    def __init__(self, spacetime, name, mass=0.0, coupling_xi=0.0):
        """
        Initializes a ScalarField.

        Parameters:
            spacetime (Spacetime): The spacetime on which the field exists.
            name (str): The name of the field, used to create its symbol.
            mass (float or sympy.Symbol): The mass of the field.
            coupling_xi (float or sympy.Symbol): The coupling constant to the Ricci scalar.
                                                 For minimal coupling, xi=0.
                                                 For conformal coupling in 4D, xi=1/6.
        """
        if not isinstance(spacetime, Spacetime):
            raise TypeError("spacetime must be an instance of the Spacetime class.")
        
        field_symbol = sympy.Function(name)(*spacetime.coords)
        super().__init__(field_symbol, spacetime)
        
        self.mass = sympy.Symbol(mass) if isinstance(mass, str) else mass
        self.coupling_xi = sympy.Symbol(coupling_xi) if isinstance(coupling_xi, str) else coupling_xi

    def d_alembertian(self):
        """
        Computes the d'Alembertian operator (Box) acting on the field.

        The d'Alembertian is defined as g^{mu,nu} nabla_mu nabla_nu phi.

        Returns:
            sympy.Expr: The symbolic expression for the d'Alembertian of the field.
        """
        if self.spacetime.metric is None:
            raise ValueError("Spacetime metric must be set before computing the d'Alembertian.")

        g_inv = self.spacetime.inverse_metric
        g_det_abs = abs(self.spacetime.metric.det())
        coords = self.spacetime.coords
        phi = self.name # The field function phi(t,x,y,z)

        box_phi = 0
        for mu in range(self.spacetime.dimension):
            for nu in range(self.spacetime.dimension):
                # Term: 1/sqrt(g) * d_mu(sqrt(g) * g^{mu,nu} * d_nu(phi))
                term = sympy.diff(sympy.sqrt(g_det_abs) * g_inv[mu, nu] * sympy.diff(phi, coords[nu]), coords[mu])
                box_phi += term

        return sympy.simplify(box_phi / sympy.sqrt(g_det_abs))

    def equation_of_motion(self):
        """
        Constructs the Klein-Gordon equation for the field in curved spacetime.

        The equation is: (Box - m^2 - xi*R) * phi = 0

        Returns:
            sympy.Eq: The symbolic Klein-Gordon equation.
        """
        box_phi = self.d_alembertian()
        R = self.spacetime.ricci_scalar()
        phi = self.name
        
        eom = box_phi - self.mass**2 * phi - self.coupling_xi * R * phi
        
        return sympy.Eq(eom, 0)

    def _second_covariant_derivative_scalar(self, scalar_expression):
        """
        Computes the second covariant derivative (nabla_mu nabla_nu f) of a scalar expression f.

        Parameters:
            scalar_expression (sympy.Expr): The scalar expression to differentiate.

        Returns:
            sympy.Matrix: A (0,2) tensor (Matrix) representing nabla_mu nabla_nu f.
        """
        dim = self.spacetime.dimension
        coords = self.spacetime.coords
        christoffels = self.spacetime.christoffel_symbols() # Gamma^k_ij = _christoffel[k,i,j]

        # First derivative: d_nu f (covector)
        df_covector = sympy.Matrix([sympy.diff(scalar_expression, c) for c in coords])

        nabla2_f = sympy.MutableDenseMatrix.zeros(dim, dim)
        for mu in range(dim):
            for nu in range(dim):
                # d_mu (d_nu f)
                term = sympy.diff(df_covector[nu], coords[mu])
                # - Gamma^lambda_{mu,nu} * d_lambda f
                sum_christoffel_term = sympy.S.Zero
                for lam in range(dim):
                    sum_christoffel_term += christoffels[lam, mu, nu] * df_covector[lam]
                nabla2_f[mu, nu] = sympy.simplify(term - sum_christoffel_term)
        return nabla2_f

    def _box_operator_scalar(self, scalar_expression):
        """
        Computes the d'Alembertian operator (Box f = g^{mu,nu} nabla_mu nabla_nu f) acting on a scalar expression f.

        Parameters:
            scalar_expression (sympy.Expr): The scalar expression.

        Returns:
            sympy.Expr: The symbolic expression for Box f.
        """
        dim = self.spacetime.dimension
        g_inv = self.spacetime.inverse_metric
        nabla2_f = self._second_covariant_derivative_scalar(scalar_expression)

        box_f = sympy.S.Zero
        for mu in range(dim):
            for nu in range(dim):
                box_f += g_inv[mu, nu] * nabla2_f[mu, nu]
        return sympy.simplify(box_f)

    def stress_energy_tensor(self):
        """
        Computes the stress-energy tensor T_munu for the scalar field.
        Formula used:
        T_μν = (∇_μ φ)(∇_ν φ) - g_μν [1/2 ((∇_α φ)(∇^α φ) + m² φ²) + 1/2 ξ R φ²]
                 + ξ [R_μν φ² - (∇_μ ∇_ν φ²) + g_μν (□ φ²)]
        where ∇_μ is the covariant derivative, φ is the scalar field, m is its mass,
        ξ is the coupling to the Ricci scalar R, g_μν is the metric tensor,
        R_μν is the Ricci tensor, and □ is the d'Alembertian operator.

        Returns:
            sympy.Matrix: The stress-energy tensor as a symbolic (0,2) tensor (Matrix).
        """
        phi = self.name
        m = self.mass
        xi = self.coupling_xi

        g = self.spacetime.metric
        g_inv = self.spacetime.inverse_metric
        R_munu = self.spacetime.ricci_tensor()
        R = self.spacetime.ricci_scalar()
        coords = self.spacetime.coords
        dim = self.spacetime.dimension

        T_munu = sympy.MutableDenseMatrix.zeros(dim, dim)

        # Term 1: (nabla_mu phi)(nabla_nu phi) = (d_mu phi)(d_nu phi) for scalar phi
        dphi = sympy.Matrix([sympy.diff(phi, c) for c in coords])
        term1 = sympy.MutableDenseMatrix.zeros(dim, dim)
        for mu in range(dim):
            for nu in range(dim):
                term1[mu, nu] = dphi[mu] * dphi[nu]

        # Term 2: - g_munu * [1/2 * ( (nabla_alpha phi)(nabla^alpha phi) + m^2 * phi^2 )]
        # nabla_alpha phi * nabla^alpha phi = g^{alpha,beta} (d_alpha phi)(d_beta phi)
        grad_phi_contracted = sympy.S.Zero
        for alpha in range(dim):
            for beta in range(dim):
                grad_phi_contracted += g_inv[alpha, beta] * dphi[alpha] * dphi[beta]
        
        lagrangian_minimal_part = sympy.S(1)/2 * (grad_phi_contracted + m**2 * phi**2)
        term2 = -g * lagrangian_minimal_part

        # Term 3: - g_munu * [1/2 * xi * R * phi^2]
        term3 = -g * (sympy.S(1)/2 * xi * R * phi**2)

        # Part for non-minimal coupling (terms proportional to xi)
        term4 = sympy.MutableDenseMatrix.zeros(dim, dim)
        term5 = sympy.MutableDenseMatrix.zeros(dim, dim)
        term6 = sympy.MutableDenseMatrix.zeros(dim, dim)

        if xi != 0:
            phi_sq = phi**2
            # Term 4: xi * R_munu * phi^2
            term4 = xi * R_munu * phi_sq

            # Term 5: - xi * (nabla_mu nabla_nu phi^2)
            nabla2_phi_sq = self._second_covariant_derivative_scalar(phi_sq)
            term5 = -xi * nabla2_phi_sq

            # Term 6: xi * g_munu * (Box phi^2)
            box_phi_sq = self._box_operator_scalar(phi_sq)
            term6 = xi * g * box_phi_sq
        
        T_munu = term1 + term2 + term3 + term4 + term5 + term6

        # Apply simplify to each component
        for i in range(dim):
            for j in range(dim):
                T_munu[i,j] = sympy.simplify(T_munu[i,j])
        
        return T_munu

    def stress_energy_tensor_contravariant(self):
        """
        Computes the contravariant stress-energy tensor T^munu for the scalar field.

        This is done by raising the indices of the covariant tensor: T^munu = g^mua * g^nub * T_ab.

        Returns:
            sympy.Matrix: The contravariant stress-energy tensor as a symbolic (2,0) tensor (Matrix).
        """
        T_munu = self.stress_energy_tensor()
        g_inv = self.spacetime.inverse_metric
        dim = self.spacetime.dimension

        # T^munu = g^{mu,alpha} * g^{nu,beta} * T_{alpha,beta}
        # In matrix form, this is g_inv * T_munu * g_inv (since g_inv is symmetric)
        T_con = g_inv * T_munu * g_inv

        # Apply simplify to each component
        for i in range(dim):
            for j in range(dim):
                T_con[i, j] = sympy.simplify(T_con[i, j])

        return T_con

    def commutator(self, point1, point2):
        """
        Computes the commutator of the field at two spacetime points (placeholder).
        This typically requires a state and is non-trivial.
        """
        print(f"Placeholder for commutator at {point1} and {point2}.")
        return None

class ChargedScalarField(ScalarField):
    """
    Represents a charged scalar quantum field, which couples to an electromagnetic field.
    The field is complex-valued.
    """
    def __init__(self, spacetime, name, mass=0.0, coupling_xi=0.0, charge=0.0, em_field=None):
        """
        Initializes a ChargedScalarField.

        Parameters:
            spacetime (Spacetime): The spacetime on which the field exists.
            name (str): The symbolic name of the field (e.g., 'phi').
            mass (float or sympy.Symbol): The mass of the field.
            coupling_xi (float or sympy.Symbol): The coupling constant to the Ricci scalar.
            charge (float or sympy.Symbol): The electric charge of the field.
            em_field (ElectromagneticField, optional): The electromagnetic field to which this field couples. Defaults to None.
        """
        super().__init__(spacetime, name, mass, coupling_xi)

        if em_field is not None and not isinstance(em_field, ElectromagneticField):
            raise TypeError("em_field must be an instance of the ElectromagneticField class or None.")

        self.charge = sympy.Symbol(charge) if isinstance(charge, str) else charge
        self.em_field = em_field

    def equation_of_motion(self):
        """
        Constructs the Klein-Gordon equation for the charged scalar field.

        The equation is: (D_mu D^mu - m^2 - xi*R) * phi = 0,
        where D_mu = nabla_mu - i*q*A_mu is the gauge covariant derivative.

        Returns:
            sympy.Eq: The symbolic Klein-Gordon equation for the charged field.
        """
        phi = self.name
        m = self.mass
        xi = self.coupling_xi
        q = self.charge
        R = self.spacetime.ricci_scalar()

        if self.em_field and q != 0:
            # Interacting field with gauge covariant derivative: D_mu = nabla_mu - i*q*A_mu
            A_mu = self.em_field.potential
            g_inv = self.spacetime.inverse_metric
            
            # D_nu phi = (d_nu - i*q*A_nu)phi
            D_nu_phi = sympy.Matrix([sympy.diff(phi, c) for c in self.spacetime.coords]) - sympy.I * q * A_mu * phi
            
            # D^mu phi = g^{mu,nu} D_nu phi
            D_sup_mu_phi = g_inv * D_nu_phi
            
            # D_mu D^mu phi = nabla_mu (D^mu phi)
            # We need the covariant divergence of a complex vector.
            box_D_phi = self.spacetime.covariant_divergence(D_sup_mu_phi, is_complex=True)
        else:
            # Free field, use standard d'Alembertian
            box_D_phi = self.d_alembertian()

        # Assemble the equation of motion
        eom = box_D_phi - (m**2 + xi * R) * phi
        
        return sympy.Eq(sympy.simplify(eom), 0)

    def stress_energy_tensor(self):
        """
        Computes the stress-energy tensor T_munu for the charged scalar field.
        
        For now, only the free field case (q=0) is implemented. The interacting case is not yet supported.
        """
        if self.em_field is not None and self.charge != 0:
            raise NotImplementedError("Stress-energy tensor for the interacting charged scalar field is not yet implemented.")

        # For the free complex scalar field, the stress tensor is the same as for the real scalar field.
        # A full treatment would require separate handling of phi and its complex conjugate.
        return super().stress_energy_tensor()


class ElectromagneticField(FieldOperator):
    """
    Represents the electromagnetic field (a U(1) gauge field).

    The field is fundamentally described by the 4-potential A_mu, from which
    physical observables like the electric and magnetic fields are derived.
    """
    def __init__(self, spacetime):
        """
        Initializes an ElectromagneticField.

        Parameters:
            spacetime (Spacetime): The spacetime on which the field exists.
        """
        if not isinstance(spacetime, Spacetime):
            raise TypeError("spacetime must be an instance of the Spacetime class.")

        # A_mu is a covector field (a (0,1)-tensor).
        potential_symbols = [sympy.Function(f'A_{i}')(*spacetime.coords) for i in range(spacetime.dimension)]
        self.potential = sympy.Matrix(potential_symbols)

        super().__init__(self.potential, spacetime)

    def field_strength_tensor(self):
        """
        Computes the electromagnetic field strength tensor F_munu (a (0,2)-tensor).

        F_munu = d_mu(A_nu) - d_nu(A_mu)
        """
        dim = self.spacetime.dimension
        coords = self.spacetime.coords
        A_mu = self.potential

        F_munu = sympy.MutableDenseMatrix.zeros(dim, dim)
        for mu in range(dim):
            for nu in range(dim):
                d_mu_A_nu = sympy.diff(A_mu[nu], coords[mu])
                d_nu_A_mu = sympy.diff(A_mu[mu], coords[nu])
                F_munu[mu, nu] = d_mu_A_nu - d_nu_A_mu
        
        return F_munu

    def field_strength_tensor_contravariant(self):
        """
        Computes the contravariant field strength tensor F^munu = g^mua * g^nub * F_ab.
        """
        F_munu = self.field_strength_tensor()
        g_inv = self.spacetime.inverse_metric
        
        F_con = g_inv * F_munu * g_inv.T

        for i in range(g_inv.rows):
            for j in range(g_inv.cols):
                F_con[i, j] = sympy.simplify(F_con[i, j])

        return F_con

    def equation_of_motion(self):
        """
        Constructs the source-free Maxwell's equations in curved spacetime: nabla_mu F^{mu,nu} = 0.
        """
        F_con = self.field_strength_tensor_contravariant()
        g_det_abs = abs(self.spacetime.metric.det())
        coords = self.spacetime.coords
        dim = self.spacetime.dimension

        eom_vector = sympy.MutableDenseMatrix.zeros(dim, 1)
        for nu in range(dim):
            divergence_term = sympy.S.Zero
            for mu in range(dim):
                term = sympy.sqrt(g_det_abs) * F_con[mu, nu]
                divergence_term += sympy.diff(term, coords[mu])
            
            eom_vector[nu] = sympy.simplify(divergence_term / sympy.sqrt(g_det_abs))

        return sympy.Eq(eom_vector, sympy.zeros(dim, 1))

    def stress_energy_tensor(self):
        """
        Computes the stress-energy tensor T_munu for the electromagnetic field.

        T_munu = F_{mu,lambda} * F_nu^lambda - 1/4 * g_munu * F_alphabeta * F^alphabeta
        """
        g = self.spacetime.metric
        g_inv = self.spacetime.inverse_metric
        dim = self.spacetime.dimension
        
        F_munu = self.field_strength_tensor()
        F_con = self.field_strength_tensor_contravariant()

        # F_nu^lambda = g^{lambda,beta} * F_{nu,beta}
        F_nu_sup_lambda = (g_inv * F_munu.T).T
        term1 = F_munu * F_nu_sup_lambda

        F_contracted = sympy.S.Zero
        for alpha in range(dim):
            for beta in range(dim):
                F_contracted += F_munu[alpha, beta] * F_con[alpha, beta]
        
        term2 = -sympy.S(1)/4 * g * F_contracted
        
        T_munu = term1 + term2
        
        for i in range(dim):
            for j in range(dim):
                T_munu[i, j] = sympy.simplify(T_munu[i, j])
                
        return T_munu
        g = self.spacetime.metric
        g_inv = self.spacetime.inverse_metric
        dim = self.spacetime.dimension

        # First, compute the invariant F_alphabeta * F^alphabeta
        F_invariant = sympy.S.Zero
        for alpha in range(dim):
            for beta in range(dim):
                F_invariant += F_munu[alpha, beta] * F_con[alpha, beta]

        # Second, compute the term F_{mu,lambda} * F_nu^lambda
        # F_nu^lambda = g^{lambda,sigma} * F_{nu,sigma}
        # So the term is F_{mu,lambda} * g^{lambda,sigma} * F_{nu,sigma}
        # In matrix form, this is F * g_inv * F.T (since F is anti-symmetric, F.T = -F)
        # Let's do it with indices to be safe.
        F_term = sympy.MutableDenseMatrix.zeros(dim, dim)
        for mu in range(dim):
            for nu in range(dim):
                sum_term = sympy.S.Zero
                for lam in range(dim):
                    for sigma in range(dim):
                        sum_term += F_munu[mu, lam] * g_inv[lam, sigma] * F_munu[nu, sigma]
                F_term[mu, nu] = sum_term

        # Assemble the full tensor
        T_munu = F_term - sympy.S(1)/4 * g * F_invariant

        # Apply simplify to each component
        for i in range(dim):
            for j in range(dim):
                T_munu[i, j] = sympy.simplify(T_munu[i, j])

        return T_munu
