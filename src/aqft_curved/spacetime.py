import sympy
from sympy import diff, Array

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
    def __init__(self, dimension, coordinates, metric=None):
        """
        Initializes a Spacetime object.

        Parameters:
            dimension (int): The number of spacetime dimensions.
            coordinates (tuple or list of str): The names of the coordinates (e.g., ('t', 'r')).
            metric (sympy.Matrix, optional): The metric tensor as a SymPy matrix.
        """
        self.dimension = dimension
        self.coords = sympy.symbols(coordinates)
        self.metric = metric
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

    def christoffel_symbols(self):
        """
        Computes the Christoffel symbols of the second kind (Gamma^k_ij).

        Returns:
            sympy.Array: A 3D array containing the Christoffel symbols.
        """
        if self._christoffel is None:
            if self.metric is None:
                raise ValueError("Metric is not set.")
            
            g_inv = self.inverse_metric
            g = self.metric
            n = self.dimension
            coords = self.coords
            
            chris = sympy.MutableDenseNDimArray.zeros(n, n, n)
            
            dg = Array([[[diff(g[i, j], coords[k]) for k in range(n)] for j in range(n)] for i in range(n)])

            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        s = 0
                        for l in range(n):
                            term1 = dg[l, j, i]
                            term2 = dg[i, l, j]
                            term3 = dg[i, j, l]
                            s += g_inv[k, l] * (term1 + term2 - term3) / 2
                        chris[k, i, j] = s
            self._christoffel = sympy.Array(chris)
        return self._christoffel

    def riemann_tensor(self):
        """
        Computes the Riemann curvature tensor (R^rho_sigma_mu_nu).

        Returns:
            sympy.Array: A 4D array containing the Riemann tensor components.
        """
        if self.metric is None:
            raise ValueError("Metric is not set.")
        
        n = self.dimension
        coords = self.coords
        chris = self.christoffel_symbols()
        
        riemann = sympy.MutableDenseNDimArray.zeros(n, n, n, n)
        
        d_chris = Array([[[[diff(chris[i, j, k], coords[l]) for l in range(n)] for k in range(n)] for j in range(n)] for i in range(n)])

        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        term1 = d_chris[rho, nu, sigma, mu]
                        term2 = d_chris[rho, mu, sigma, nu]
                        
                        term3 = 0
                        for lam in range(n):
                            term3 += chris[rho, lam, mu] * chris[lam, nu, sigma]
                        
                        term4 = 0
                        for lam in range(n):
                            term4 += chris[rho, lam, nu] * chris[lam, mu, sigma]
                            
                        riemann[rho, sigma, mu, nu] = term1 - term2 + term3 - term4
                        
        return sympy.Array(riemann)

    def ricci_tensor(self):
        """
        Computes the Ricci curvature tensor (R_mu_nu).

        Returns:
            sympy.Matrix: A 2D matrix containing the Ricci tensor components.
        """
        riemann = self.riemann_tensor()
        n = self.dimension
        ricci = sympy.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                s = 0
                for rho in range(n):
                    s += riemann[rho, mu, rho, nu]
                ricci[mu, nu] = s
        return sympy.Matrix(ricci)

    def ricci_scalar(self):
        """Computes the Ricci scalar (R)."""
        ricci = self.ricci_tensor()
        g_inv = self.inverse_metric
        R = (g_inv.T * ricci).trace()
        return sympy.simplify(R)

    def einstein_tensor(self):
        """Computes the Einstein tensor (G_mu_nu)."""
        ricci = self.ricci_tensor()
        R = self.ricci_scalar()
        g = self.metric
        return ricci - (R / 2) * g

class PredefinedSpacetime(Spacetime):
    """
    Provides common, analytical spacetimes.

    Available spacetimes: 'Minkowski', 'Schwarzschild', 'FLRW'
    """
    def __init__(self, name, **params):
        if name.lower() == 'minkowski':
            coords = ('t', 'x', 'y', 'z')
            metric = sympy.diag(-1, 1, 1, 1)
            super().__init__(dimension=4, coordinates=coords, metric=metric)
        
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

            # Substitute numerical value for M if provided
            if 'M' in params:
                metric = metric.subs(M_symbol, params['M'])

            super().__init__(dimension=4, coordinates=coords, metric=metric)

        elif name.lower() == 'flrw':
            coords = ('t', 'x', 'y', 'z')
            t_sym = sympy.symbols(coords[0])
            a_func = sympy.Function('a')(t_sym)
            k_symbol = sympy.Symbol('k')

            # Default to a symbolic function if not provided
            a = params.get('a', a_func)
            # Default to flat spatial curvature
            k = params.get('k', 0)

            # FLRW metric with curvature
            spatial_part = sympy.diag(1, 1, 1) / (1 - k * (sympy.symbols('x')**2 + sympy.symbols('y')**2 + sympy.symbols('z')**2))
            metric = sympy.diag(-1, *(a**2 * spatial_part))

            # Substitute numerical value for k if provided
            if 'k' in params:
                 metric = metric.subs(k_symbol, params['k'])

            super().__init__(dimension=4, coordinates=coords, metric=metric)
            
        else:
            raise ValueError(f"Unknown predefined spacetime: {name}")
