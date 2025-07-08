import sys
import os
import sympy
from sympy import diff, Array

# Add vendored einsteinpy to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'vendor')))

try:
    from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor, RicciTensor, RicciScalar
except ImportError as e:
    print("Could not import EinsteinPy. Make sure it is installed or vendored correctly.")
    raise e

class CoordinateChart:
    """Manages local coordinate systems and transformations."""
    def __init__(self, coordinates, transformation_rules=None):
        """
        Initializes a CoordinateChart.

        Parameters:
            coordinates (tuple of sympy.Symbol): The coordinate symbols.
            transformation_rules (dict, optional): Rules for transforming to other charts.
        """
        self.coordinates = coordinates
        self.transformation_rules = transformation_rules

class Spacetime:
    """
    Represents a curved spacetime, defined by a metric tensor, and provides
    methods for computing standard geometric quantities.
    """
    def __init__(self, dimension, coordinates, metric=None, name=None):
        """
        Initializes a Spacetime object.

        Parameters:
            dimension (int): The number of spacetime dimensions.
            coordinates (tuple or list of str): The names of the coordinates (e.g., ('t', 'r')).
            metric (sympy.Matrix, optional): The metric tensor as a SymPy matrix.
            name (str, optional): The name of the spacetime. Defaults to None.
        """
        self.dimension = dimension
        self.coords = sympy.symbols(coordinates)
        self.metric = metric
        self.name = name
        self._inverse_metric = None
        self._christoffel = None

    def set_metric(self, metric_matrix):
        """Sets or updates the metric tensor for the spacetime."""
        if not isinstance(metric_matrix, sympy.Matrix) or metric_matrix.shape != (self.dimension, self.dimension):
            raise ValueError(f"Metric must be a {self.dimension}x{self.dimension} SymPy Matrix.")
        self.metric = metric_matrix
        # Invalidate cached properties
        self._inverse_metric = None
        self._christoffel = None
        print("Metric has been set.")

    @property
    def inverse_metric(self):
        """Computes and caches the inverse metric tensor."""
        if self._inverse_metric is None:
            if self.metric is None:
                raise ValueError("Metric is not set.")
            self._inverse_metric = self.metric.inv()
        return self._inverse_metric

    def _get_einsteinpy_metric(self):
        """Helper to create an einsteinpy MetricTensor instance."""
        if self.metric is None:
            raise ValueError("Metric is not set.")
        return MetricTensor(Array(self.metric), self.coords)

    def christoffel_symbols(self):
        """
        Computes the Christoffel symbols of the second kind (Gamma^k_ij) using EinsteinPy.

        Returns:
            sympy.Array: A 3D array containing the Christoffel symbols.
        """
        if self._christoffel is None:
            e_metric = self._get_einsteinpy_metric()
            # The result from EinsteinPy is an object, call .tensor() to get the array
            self._christoffel = ChristoffelSymbols.from_metric(e_metric).tensor()
        return self._christoffel

    def riemann_tensor(self):
        """
        Computes the Riemann curvature tensor (R^rho_sigma_mu_nu) using EinsteinPy.

        Returns:
            sympy.Array: A 4D array containing the Riemann tensor components.
        """
        e_metric = self._get_einsteinpy_metric()
        # The result from EinsteinPy is an object, call .tensor() to get the array
        # By default, it returns with config 'ulll', which matches our old implementation's indices
        return RiemannCurvatureTensor.from_metric(e_metric).tensor()

    def ricci_tensor(self):
        """
        Computes the Ricci curvature tensor (R_mu_nu) using EinsteinPy.

        Returns:
            sympy.Matrix: A 2D matrix containing the Ricci tensor components.
        """
        e_metric = self._get_einsteinpy_metric()
        # The result from EinsteinPy is an object, call .tensor() to get the array
        # Returns with 'll' config by default
        ricci_obj = RicciTensor.from_metric(e_metric)
        return sympy.Matrix(ricci_obj.tensor())

    def ricci_scalar(self):
        """
        Computes the Ricci scalar (R) using EinsteinPy.

        Returns:
            sympy.Expr: The symbolic expression for the Ricci scalar.
        """
        e_metric = self._get_einsteinpy_metric()
        # The result from EinsteinPy is an object, call .expr to get the expression
        return RicciScalar.from_metric(e_metric).expr

    def covariant_divergence(self, contravariant_vector):
        """
        Computes the covariant divergence of a contravariant vector field V^mu.

        The formula is: nabla_mu V^mu = 1/sqrt(|g|) * d_mu(sqrt(|g|) * V^mu)

        Parameters:
            contravariant_vector (sympy.Matrix): The vector field V^mu as a symbolic matrix.

        Returns:
            sympy.Expr: The symbolic expression for the covariant divergence.
        """
        if not (isinstance(contravariant_vector, sympy.Matrix) and contravariant_vector.shape == (self.dimension, 1)):
            raise ValueError(f"Input must be a contravariant vector of shape ({{self.dimension}}, 1).")

        g_det_abs = abs(self.metric.det())
        coords = self.coords
        dim = self.dimension

        divergence = sympy.S.Zero
        for mu in range(dim):
            term = sympy.sqrt(g_det_abs) * contravariant_vector[mu]
            divergence += sympy.diff(term, coords[mu])

        return sympy.simplify(divergence / sympy.sqrt(g_det_abs))

    def einstein_tensor(self):
        """
        Computes the Einstein tensor (G_munu).

        The Einstein tensor is defined as G_munu = R_munu - 1/2 * g_munu * R.

        Returns:
            sympy.Matrix: A 2D matrix containing the simplified Einstein tensor components.
        """
        if self.metric is None:
            raise ValueError("Metric is not set.")

        ricci = self.ricci_tensor()
        R = self.ricci_scalar()
        g = self.metric

        # The Einstein tensor G_munu is defined as R_munu - 1/2 * g_munu * R
        G = ricci - (sympy.S(1)/2) * g * R

        return sympy.simplify(G)

    def covariant_divergence_rank2_tensor(self, tensor_con):
        """
        Computes the covariant divergence of a rank-2 contravariant tensor (nabla_mu T^munu).

        Parameters:
            tensor_con (sympy.Matrix): The (2,0) tensor (contravariant) for which to compute the divergence.

        Returns:
            sympy.Matrix: A (1,0) tensor (vector) representing the covariant divergence.
        """
        if self.metric is None:
            raise ValueError("Metric is not set.")
        if not isinstance(tensor_con, sympy.Matrix) or tensor_con.shape != (self.dimension, self.dimension):
            raise ValueError(f"Tensor must be a {self.dimension}x{self.dimension} SymPy Matrix.")

        dim = self.dimension
        coords = self.coords
        chris = self.christoffel_symbols()

        div_T = sympy.MutableDenseMatrix.zeros(dim, 1)

        for nu in range(dim):
            s = sympy.S.Zero
            for mu in range(dim):
                # Partial derivative term: d_mu(T^munu)
                s += sympy.diff(tensor_con[mu, nu], coords[mu])
                
                # Christoffel terms
                for rho in range(dim):
                    # + Gamma^mu_murho * T^rhonu
                    s += chris[mu, mu, rho] * tensor_con[rho, nu]
                    # + Gamma^nu_murho * T^murho
                    s += chris[nu, mu, rho] * tensor_con[mu, rho]
            div_T[nu] = sympy.simplify(s)

        return div_T

class PredefinedSpacetime(Spacetime):
    """
    Provides common, analytical spacetimes.

    Available spacetimes: 'Minkowski', 'Schwarzschild', 'deSitter', 'Anti-deSitter', 'Reissner-Nordstrom', 'Kerr', 'FLRW'
    """
    def __init__(self, name, **params):
        if name.lower() == 'minkowski':
            coords = ('t', 'x', 'y', 'z')
            metric = sympy.diag(-1, 1, 1, 1)
            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)
        
        elif name.lower() == 'schwarzschild':
            coords = ('t', 'r', 'theta', 'phi')
            M_symbol = sympy.Symbol('M')
            t, r, theta, phi = sympy.symbols(coords)
            
            f = 1 - 2 * M_symbol / r
            metric = sympy.diag(
                -f,
                1 / f,
                r**2,
                r**2 * sympy.sin(theta)**2
            )

            # Substitute numerical value for M if provided and store it
            if 'M' in params:
                metric = metric.subs(M_symbol, params['M'])
                self.M = params['M']
            else:
                self.M = M_symbol

            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)

        elif name.lower() == 'desitter':
            coords = ('t', 'r', 'theta', 'phi')
            alpha_symbol = sympy.Symbol('alpha')  # de Sitter radius
            t, r, theta, phi = sympy.symbols(coords)

            f = 1 - (r**2 / alpha_symbol**2)
            metric = sympy.diag(
                -f,
                1 / f,
                r**2,
                r**2 * sympy.sin(theta)**2
            )

            if 'alpha' in params:
                metric = metric.subs(alpha_symbol, params['alpha'])
                self.alpha = params['alpha']
            else:
                self.alpha = alpha_symbol

            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)

        elif name.lower() == 'anti-desitter':
            coords = ('t', 'r', 'theta', 'phi')
            alpha_symbol = sympy.Symbol('alpha')  # AdS radius
            t, r, theta, phi = sympy.symbols(coords)

            f = 1 + (r**2 / alpha_symbol**2)
            metric = sympy.diag(
                -f,
                1 / f,
                r**2,
                r**2 * sympy.sin(theta)**2
            )

            if 'alpha' in params:
                metric = metric.subs(alpha_symbol, params['alpha'])
                self.alpha = params['alpha']
            else:
                self.alpha = alpha_symbol

            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)

        elif name.lower() == 'reissner-nordstrom':
            coords = ('t', 'r', 'theta', 'phi')
            M_symbol = sympy.Symbol('M')
            Q_symbol = sympy.Symbol('Q')
            t, r, theta, phi = sympy.symbols(coords)

            f = 1 - (2 * M_symbol / r) + (Q_symbol**2 / r**2)
            metric = sympy.diag(
                -f,
                1 / f,
                r**2,
                r**2 * sympy.sin(theta)**2
            )

            if 'M' in params:
                metric = metric.subs(M_symbol, params['M'])
                self.M = params['M']
            else:
                self.M = M_symbol
            if 'Q' in params:
                metric = metric.subs(Q_symbol, params['Q'])
                self.Q = params['Q']
            else:
                self.Q = Q_symbol

            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)

        elif name.lower() == 'kerr':
            coords = ('t', 'r', 'theta', 'phi')
            M_symbol = sympy.Symbol('M')
            a_symbol = sympy.Symbol('a')
            t, r, theta, phi = sympy.symbols(coords)

            rho_sq = r**2 + a_symbol**2 * sympy.cos(theta)**2
            delta = r**2 - 2 * M_symbol * r + a_symbol**2

            metric = sympy.zeros(4)
            metric[0, 0] = -(1 - 2 * M_symbol * r / rho_sq)
            metric[0, 3] = -(2 * M_symbol * r * a_symbol * sympy.sin(theta)**2 / rho_sq)
            metric[1, 1] = rho_sq / delta
            metric[2, 2] = rho_sq
            metric[3, 3] = (r**2 + a_symbol**2 + (2 * M_symbol * r * a_symbol**2 * sympy.sin(theta)**2) / rho_sq) * sympy.sin(theta)**2
            metric[3, 0] = metric[0, 3]

            if 'M' in params:
                metric = metric.subs(M_symbol, params['M'])
                self.M = params['M']
            else:
                self.M = M_symbol
            if 'a' in params:
                metric = metric.subs(a_symbol, params['a'])
                self.a = params['a']
            else:
                self.a = a_symbol

            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)

        elif name.lower() == 'flrw':
            coords = ('t', 'x', 'y', 'z')
            t_sym = sympy.symbols(coords[0])
            a_func = sympy.Function('a')(t_sym)
            k_symbol = sympy.Symbol('k')

            self.a = params.get('a', a_func)
            self.k = params.get('k', 0)
            
            _t, _x, _y, _z = sympy.symbols(coords)
            
            if self.k == 0:
                spatial_metric_part = sympy.diag(self.a**2, self.a**2, self.a**2)
            else:
                r_sq = _x**2 + _y**2 + _z**2
                spatial_factor = self.a**2 / (1 - self.k * r_sq)
                spatial_metric_part = sympy.diag(spatial_factor, spatial_factor, spatial_factor)

            metric_list = [-1] + [spatial_metric_part[i,i] for i in range(3)]
            metric = sympy.diag(*metric_list)

            if isinstance(self.k, sympy.Symbol) and 'k' in params:
                 metric = metric.subs(self.k, params['k'])
            if isinstance(self.a, sympy.Symbol) and 'a' in params:
                metric = metric.subs(self.a, params['a'])

            super().__init__(dimension=4, coordinates=coords, metric=metric, name=name)
            
        else:
            raise ValueError(f"Unknown predefined spacetime: {name}")
