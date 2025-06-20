import sympy
from sympy import conjugate
from sympy.physics.quantum import Commutator, Dagger
from .field import ScalarField, ChargedScalarField


class CanonicalQuantization:
    """
    Performs canonical quantization of a classical field theory.
    """

    def __init__(self, field):
        """
        Initializes the CanonicalQuantization object.

        Args:
            field (ScalarField or ChargedScalarField): The field to be quantized.
        """
        self.field = field
        self.spacetime = field.spacetime
        self.coords = self.spacetime.coords

        if isinstance(field, ChargedScalarField):
            self.is_charged = True
            self.phi = self.field.name
            self.phi_dag = conjugate(self.phi)
        elif isinstance(field, ScalarField):
            self.is_charged = False
            self.phi = self.field.name
        else:
            raise TypeError("Field type not supported for quantization.")

        # Define coordinates for two points (x and y) for commutation relations
        self.coords_x = self.coords
        if len(self.coords) == 4:
            self.coords_y = (self.coords[0],) + tuple(sympy.symbols("x' y' z'"))
        else:
            # Fallback for different dimensions
            self.coords_y = (self.coords[0],) + sympy.symbols('x_prime_1:' + str(len(self.coords)))

    def conjugate_momentum(self):
        """
        Computes the conjugate momentum for the field.
        For a real scalar field, pi = -d_t(phi).
        For a charged scalar field, returns a tuple (pi, pi_dag) where
        pi = (d_t phi_dag) and pi_dag = (d_t phi).
        This assumes a metric signature of (-, +, +, +) and a standard Lagrangian.

        Returns:
            sympy.Expr or tuple: The conjugate momentum(s).
        """
        if self.is_charged:
            # For L = g^munu (d_mu phi)† (d_nu phi) - ...
            # With Minkowski g, this is L = (d_t phi)†(d_t phi) - ...
            # pi = dL/d(d_t phi) = (d_t phi)† = d_t(phi_dag)
            pi = sympy.diff(self.phi_dag, self.coords[0])
            # pi_dag = dL/d(d_t phi_dag) = d_t phi
            pi_dag = sympy.diff(self.phi, self.coords[0])
            return (pi, pi_dag)
        else:
            # For L_real = 1/2 g^munu (d_mu phi)(d_nu phi) - ...
            # With Minkowski g, this is L = 1/2 (-(d_t phi)^2 + ...)
            # pi = dL/d(d_t phi) = -d_t phi
            pi = -sympy.diff(self.phi, self.coords[0])
            return pi

    def commutation_relations(self):
        """
        Defines the equal-time commutation relations for the field and its conjugate momentum.

        The relations are:
        [φ(t, x), π(t, y)] = i * δ(x - y)
        [φ(t, x), φ(t, y)] = 0
        [π(t, x), π(t, y)] = 0

        Returns:

        Returns:
            tuple: A tuple of sympy.Eq objects representing the commutation relations.
        """
        # Define the Dirac delta function for 3D space
        delta_3d = sympy.DiracDelta(self.coords_y[1] - self.coords_x[1]) * \
                   sympy.DiracDelta(self.coords_y[2] - self.coords_x[2]) * \
                   sympy.DiracDelta(self.coords_y[3] - self.coords_x[3])

        # Define the field operator(s)
        phi_op = sympy.Function(self.phi.func.__name__, is_commutative=False)
        phi_x = phi_op(*self.coords_x)
        phi_y = phi_op(*self.coords_y)

        if self.is_charged:
            # Define operators for the conjugated field and momenta
            phi_dag_x = Dagger(phi_x)
            phi_dag_y = Dagger(phi_y)

            pi_op = sympy.Function('pi', is_commutative=False)
            pi_dag_op = sympy.Function('pi_dag', is_commutative=False)
            pi_x = pi_op(*self.coords_x)
            pi_y = pi_op(*self.coords_y)
            pi_dag_x = pi_dag_op(*self.coords_x)
            pi_dag_y = pi_dag_op(*self.coords_y)

            # Commutation relations
            comm1 = sympy.Eq(Commutator(phi_x, pi_y), sympy.I * delta_3d)
            comm2 = sympy.Eq(Commutator(phi_dag_x, pi_dag_y), sympy.I * delta_3d)
            comm3 = sympy.Eq(Commutator(phi_x, phi_y), 0)
            comm4 = sympy.Eq(Commutator(pi_x, pi_y), 0)
            comm5 = sympy.Eq(Commutator(phi_x, phi_dag_y), 0)
            comm6 = sympy.Eq(Commutator(pi_x, pi_dag_y), 0)
            return (comm1, comm2, comm3, comm4, comm5, comm6)
        else:
            # Define the momentum operator
            pi_op = sympy.Function('pi', is_commutative=False)
            pi_x = pi_op(*self.coords_x)
            pi_y = pi_op(*self.coords_y)

            # Commutation relations
            comm1 = sympy.Eq(Commutator(phi_x, pi_y), sympy.I * delta_3d)
            comm2 = sympy.Eq(Commutator(phi_x, phi_y), 0)
            comm3 = sympy.Eq(Commutator(pi_x, pi_y), 0)

            return (comm1, comm2, comm3)

    def mode_expansion(self):
        """
        Constructs the mode expansion for the scalar field operator in Minkowski spacetime.

        The expansion is given by:
        φ(x) = ∫ d³k / ((2π)³ * 2ω_k) * [a_k * exp(-ikx) + a_k^† * exp(ikx)]

        Returns:
            sympy.Eq: An equation representing the field operator as its mode expansion.
        """
        if self.spacetime.name != "Minkowski":
            raise NotImplementedError("Mode expansion is currently only implemented for Minkowski spacetime.")

        m = self.field.mass
        t = self.coords[0]
        coords_spatial = sympy.Matrix(self.coords[1:])

        # Define 3-momentum vector k
        k_x, k_y, k_z = sympy.symbols('k_x k_y k_z')
        k_vec = sympy.Matrix([k_x, k_y, k_z])
        k_squared = k_vec.dot(k_vec)
        
        # Define energy omega_k
        omega_k = sympy.sqrt(k_squared + m**2)

        # Define creation and annihilation operators
        a = sympy.Function('a', is_commutative=False)
        a_k = a(*k_vec)
        a_dag_k = Dagger(a_k)

        # Minkowski dot product k.x
        k_dot_x = -omega_k * t + k_vec.dot(coords_spatial)

        # Integrand
        integrand = (a_k * sympy.exp(-sympy.I * k_dot_x) + 
                     a_dag_k * sympy.exp(sympy.I * k_dot_x))

        # Normalization factor
        norm_factor = (2 * sympy.pi)**3 * 2 * omega_k
        
        # Full integral expression
        mode_expansion_integral = sympy.Integral(integrand / norm_factor, (k_x, -sympy.oo, sympy.oo), (k_y, -sympy.oo, sympy.oo), (k_z, -sympy.oo, sympy.oo))

        phi_op = sympy.Function(self.phi.func.__name__, is_commutative=False)
        return sympy.Eq(phi_op(*self.coords), mode_expansion_integral)

    def creation_annihilation_commutation(self):
        """
        Defines the commutation relations for the creation and annihilation operators.

        The relations are:
        [a(k), a_dag(p)] = (2π)³ * 2ω_k * δ³(k - p)
        [a(k), a(p)] = 0
        [a_dag(k), a_dag(p)] = 0

        Returns:
            tuple: A tuple of three sympy.Eq objects representing the commutation relations.
        """
        m = self.field.mass

        # Define 3-momentum vectors k and p
        k_x, k_y, k_z = sympy.symbols('k_x k_y k_z')
        p_x, p_y, p_z = sympy.symbols('p_x p_y p_z')
        
        k_vec = sympy.Matrix([k_x, k_y, k_z])
        p_vec = sympy.Matrix([p_x, p_y, p_z])
        
        k_squared = k_vec.dot(k_vec)
        omega_k = sympy.sqrt(k_squared + m**2)

        # Define creation and annihilation operators
        a = sympy.Function('a', is_commutative=False)
        a_k = a(*k_vec)
        a_p = a(*p_vec)
        a_dag_p = Dagger(a_p)

        # Define the 3D Dirac delta function
        delta_k_p = sympy.DiracDelta(k_x - p_x) * sympy.DiracDelta(k_y - p_y) * sympy.DiracDelta(k_z - p_z)
        
        # Normalization factor
        norm_factor = (2 * sympy.pi)**3 * 2 * omega_k

        # Commutation relations
        comm1 = sympy.Eq(Commutator(a_k, a_dag_p), norm_factor * delta_k_p)
        comm2 = sympy.Eq(Commutator(a_k, a_p), 0)
        comm3 = sympy.Eq(Commutator(Dagger(a_k), a_dag_p), 0)

        return (comm1, comm2, comm3)

    def pi_mode_expansion(self):
        """
        Constructs the mode expansion for the conjugate momentum operator pi.

        The expansion is derived from pi = -d_t(phi) and is given by:
        pi(x) = ∫ d³k / ((2π)³ * 2ω_k) * [iω_k * (a_k * exp(-ikx) - a_k^† * exp(ikx))]

        Returns:
            sympy.Eq: An equation representing the momentum operator as its mode expansion.
        """
        if self.spacetime.name != "Minkowski":
            raise NotImplementedError("Mode expansion is currently only implemented for Minkowski spacetime.")

        m = self.field.mass
        t = self.coords[0]
        coords_spatial = sympy.Matrix(self.coords[1:])

        # Define 3-momentum vector k
        k_x, k_y, k_z = sympy.symbols('k_x k_y k_z')
        k_vec = sympy.Matrix([k_x, k_y, k_z])
        k_squared = k_vec.dot(k_vec)

        # Define energy omega_k
        omega_k = sympy.sqrt(k_squared + m**2)

        # Define creation and annihilation operators
        a = sympy.Function('a', is_commutative=False)
        a_k = a(*k_vec)
        a_dag_k = Dagger(a_k)

        # Minkowski dot product k.x
        k_dot_x = -omega_k * t + k_vec.dot(coords_spatial)

        # Integrand for pi
        integrand = sympy.I * omega_k * (a_dag_k * sympy.exp(sympy.I * k_dot_x) - a_k * sympy.exp(-sympy.I * k_dot_x))

        # Normalization factor
        norm_factor = (2 * sympy.pi)**3 * 2 * omega_k

        # Full integral expression
        pi_mode_expansion_integral = sympy.Integral(integrand / norm_factor, (k_x, -sympy.oo, sympy.oo), (k_y, -sympy.oo, sympy.oo), (k_z, -sympy.oo, sympy.oo))

        pi_op = sympy.Function('pi', is_commutative=False)
        return sympy.Eq(pi_op(*self.coords), pi_mode_expansion_integral)

    def hamiltonian(self):
        """
        Constructs the normal-ordered Hamiltonian operator H for the scalar field.

        The Hamiltonian is derived by integrating the classical Hamiltonian density
        H = 1/2 * (pi^2 + (nabla phi)^2 + m^2 * phi^2) over space and substituting
        the mode expansions for phi and pi. After normal ordering (which discards an
        infinite zero-point energy term), the Hamiltonian is:

        H = ∫ d³k ω_k * a_k^† * a_k

        This represents the total energy of the field as the sum of the energies of
        all field quanta (particles), where a_k^† a_k is the number operator for
        particles with momentum k.

        Returns:
            sympy.Eq: An equation representing the Hamiltonian operator.
        """
        if self.spacetime.name != "Minkowski":
            raise NotImplementedError("Hamiltonian is currently only implemented for Minkowski spacetime.")

        m = self.field.mass

        # Define 3-momentum vector k
        k_x, k_y, k_z = sympy.symbols('k_x k_y k_z')
        k_vec = sympy.Matrix([k_x, k_y, k_z])
        k_squared = k_vec.dot(k_vec)

        # Define energy omega_k
        omega_k = sympy.sqrt(k_squared + m**2)

        # Define creation and annihilation operators
        a = sympy.Function('a', is_commutative=False)
        a_k = a(*k_vec)
        a_dag_k = Dagger(a_k)

        # Number operator N_k = a_dag_k * a_k
        number_operator_k = a_dag_k * a_k

        # Integrand: omega_k * N_k
        integrand = omega_k * number_operator_k

        # Full integral expression
        hamiltonian_integral = sympy.Integral(integrand, (k_x, -sympy.oo, sympy.oo), (k_y, -sympy.oo, sympy.oo), (k_z, -sympy.oo, sympy.oo))

        H = sympy.Symbol('H')
        return sympy.Eq(H, hamiltonian_integral)
