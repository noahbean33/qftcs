use nalgebra::{DMatrix, DVector};

/// Represents the geometry of a spacetime via a metric function.
///
/// This struct stores a function that defines the metric tensor at any given
/// set of coordinates. This allows for handling both flat and curved spacetimes.
pub struct Spacetime {
    /// The number of spacetime dimensions.
    pub dimension: usize,
    /// A function that takes a vector of coordinates and returns the metric
    /// tensor g_munu at that point.
    pub metric_fn: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>,
}

impl Spacetime {
    /// Creates a new Spacetime from a given metric function.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The number of spacetime dimensions.
    /// * `metric_fn` - A closure that takes coordinates `&DVector<f64>` and returns the metric `DMatrix<f64>`.
    pub fn new(dimension: usize, metric_fn: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>) -> Self {
        Spacetime { dimension, metric_fn }
    }

    /// Evaluates the metric tensor at a specific set of coordinates.
    pub fn metric_at(&self, coords: &DVector<f64>) -> DMatrix<f64> {
        let g = (self.metric_fn)(coords);
        assert!(g.is_square(), "Metric function must return a square matrix.");
        assert_eq!(g.nrows(), self.dimension, "Metric dimensions must match spacetime dimensions.");
        g
    }

    /// Evaluates the inverse metric tensor at a specific set of coordinates.
    pub fn inverse_metric_at(&self, coords: &DVector<f64>) -> DMatrix<f64> {
        self.metric_at(coords)
            .try_inverse()
            .expect("Metric must be invertible at the given coordinates.")
    }

    /// Computes the partial derivatives of the metric tensor using central differences.
    fn metric_derivative_at(&self, coords: &DVector<f64>) -> Vec<DMatrix<f64>> {
        let h = 1e-6; // Step size for finite differences
        let dim = self.dimension;
        let mut derivatives = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut coords_plus_h = coords.clone();
            coords_plus_h[i] += h;

            let mut coords_minus_h = coords.clone();
            coords_minus_h[i] -= h;

            let g_plus = self.metric_at(&coords_plus_h);
            let g_minus = self.metric_at(&coords_minus_h);

            derivatives.push((g_plus - g_minus) / (2.0 * h));
        }
        derivatives
    }

    /// Computes the Christoffel symbols of the second kind (Gamma^lambda_munu) at specific coordinates.
    ///
    /// Formula: Gamma^lambda_munu = 1/2 * g^lambdasigma * (d_mu g_nusigma + d_nu g_musigma - d_sigma g_munu)
    pub fn christoffel_symbols_at(&self, coords: &DVector<f64>) -> Vec<DMatrix<f64>> {
        let dim = self.dimension;
        let g_inv = self.inverse_metric_at(coords);
        let dg = self.metric_derivative_at(coords); // dg[sigma] is the matrix of d_sigma g_munu

        let mut christoffels = Vec::with_capacity(dim);

        for lambda in 0..dim {
            let mut christoffel_lambda = DMatrix::zeros(dim, dim);
            for mu in 0..dim {
                for nu in 0..dim {
                    let mut sum = 0.0;
                    for sigma in 0..dim {
                        let d_mu_g_nu_sigma = dg[mu][(nu, sigma)];
                        let d_nu_g_mu_sigma = dg[nu][(mu, sigma)];
                        let d_sigma_g_mu_nu = dg[sigma][(mu, nu)];
                        sum += g_inv[(lambda, sigma)] * (d_mu_g_nu_sigma + d_nu_g_mu_sigma - d_sigma_g_mu_nu);
                    }
                    christoffel_lambda[(mu, nu)] = 0.5 * sum;
                }
            }
            christoffels.push(christoffel_lambda);
        }
        christoffels
    }

    /// Computes partial derivatives of Christoffel symbols using central differences.
    fn christoffel_derivative_at(&self, coords: &DVector<f64>) -> Vec<Vec<DMatrix<f64>>> {
        let h = 1e-6; // Step size for finite differences
        let dim = self.dimension;
        let mut derivatives = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut coords_plus_h = coords.clone();
            coords_plus_h[i] += h;

            let mut coords_minus_h = coords.clone();
            coords_minus_h[i] -= h;

            let christoffels_plus = self.christoffel_symbols_at(&coords_plus_h);
            let christoffels_minus = self.christoffel_symbols_at(&coords_minus_h);

            let mut deriv_i = Vec::with_capacity(dim);
            for lambda in 0..dim {
                deriv_i.push((&christoffels_plus[lambda] - &christoffels_minus[lambda]) / (2.0 * h));
            }
            derivatives.push(deriv_i);
        }
        derivatives
    }

    /// Computes the Riemann curvature tensor R^rho_sigma_mu_nu at specific coordinates.
    pub fn riemann_tensor_at(&self, coords: &DVector<f64>) -> Vec<Vec<DMatrix<f64>>> {
        let dim = self.dimension;
        let christoffels = self.christoffel_symbols_at(coords);
        let d_christoffels = self.christoffel_derivative_at(coords); // d_christoffels[mu][rho] is matrix d_mu Gamma^rho

        let mut riemann_tensor = vec![vec![DMatrix::zeros(dim, dim); dim]; dim];

        for rho in 0..dim {
            for sigma in 0..dim {
                for mu in 0..dim {
                    for nu in 0..dim {
                        // Standard term: d_mu Gamma^rho_nu_sigma - d_nu Gamma^rho_mu_sigma
                        let mut val = d_christoffels[mu][rho][(nu, sigma)] - d_christoffels[nu][rho][(mu, sigma)];

                        // Product term: Gamma^rho_lambda_mu * Gamma^lambda_sigma_nu - Gamma^rho_lambda_nu * Gamma^lambda_sigma_mu
                        for lambda in 0..dim {
                            val += christoffels[rho][(lambda, mu)] * christoffels[lambda][(sigma, nu)]
                                 - christoffels[rho][(lambda, nu)] * christoffels[lambda][(sigma, mu)];
                        }
                        riemann_tensor[rho][sigma][(mu, nu)] = val;
                    }
                }
            }
        }
        riemann_tensor
    }

    /// Computes the Ricci tensor R_munu at specific coordinates.
    pub fn ricci_tensor_at(&self, coords: &DVector<f64>) -> DMatrix<f64> {
        let dim = self.dimension;
        let riemann = self.riemann_tensor_at(coords);
        let mut ricci_tensor = DMatrix::zeros(dim, dim);

        for mu in 0..dim {
            for nu in 0..dim {
                let mut sum = 0.0;
                for rho in 0..dim {
                    sum += riemann[rho][mu][(rho, nu)]; // R^rho_mu_rho_nu
                }
                ricci_tensor[(mu, nu)] = sum;
            }
        }
        ricci_tensor
    }

    /// Computes the Ricci scalar R at specific coordinates.
    pub fn ricci_scalar_at(&self, coords: &DVector<f64>) -> f64 {
        let g_inv = self.inverse_metric_at(coords);
        let ricci_tensor = self.ricci_tensor_at(coords);
        (g_inv.transpose() * ricci_tensor).trace()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_minkowski_spacetime() {
        // Define the Minkowski metric as a function
        let minkowski_fn = |coords: &DVector<f64>| {
            assert_eq!(coords.nrows(), 4);
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };

        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let test_coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let g = spacetime.metric_at(&test_coords);
        let g_inv = spacetime.inverse_metric_at(&test_coords);

        let mut expected_g = DMatrix::from_diagonal_element(4, 4, 1.0);
        expected_g[(0, 0)] = -1.0;

        assert_eq!(g, expected_g);
        // The Minkowski metric is its own inverse
        assert!((g_inv - expected_g).abs().max() < 1e-9);
    }

    #[test]
    fn test_christoffel_symbols_minkowski() {
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let test_coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let christoffels = spacetime.christoffel_symbols_at(&test_coords);

        // For Minkowski spacetime, all Christoffel symbols should be zero.
        for lambda in 0..4 {
            let christoffel_lambda = &christoffels[lambda];
            assert!(christoffel_lambda.abs().max() < 1e-9, "Christoffel symbols should be zero for Minkowski spacetime.");
        }
    }

    #[test]
    fn test_riemann_tensor_minkowski() {
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let test_coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let riemann = spacetime.riemann_tensor_at(&test_coords);

        // For Minkowski spacetime, the Riemann tensor should be zero.
        for rho in 0..4 {
            for sigma in 0..4 {
                assert!(riemann[rho][sigma].abs().max() < 1e-9, "Riemann tensor should be zero for Minkowski spacetime.");
            }
        }
    }

    #[test]
    fn test_ricci_tensor_and_scalar_minkowski() {
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let test_coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let ricci_tensor = spacetime.ricci_tensor_at(&test_coords);
        let ricci_scalar = spacetime.ricci_scalar_at(&test_coords);

        // For Minkowski spacetime, the Ricci tensor and scalar should be zero.
        assert!(ricci_tensor.abs().max() < 1e-9, "Ricci tensor should be zero for Minkowski spacetime.");
        assert!(ricci_scalar.abs() < 1e-9, "Ricci scalar should be zero for Minkowski spacetime.");
    }

    #[test]
    #[should_panic(expected = "Metric function must return a square matrix.")]
    fn test_non_square_metric_fn() {
        let non_square_fn = |_coords: &DVector<f64>| DMatrix::<f64>::from_element(2, 3, 0.0);
        let spacetime = Spacetime::new(2, Box::new(non_square_fn));
        let test_coords = DVector::from_vec(vec![0.0, 0.0]);
        spacetime.metric_at(&test_coords); // This should panic
    }

    #[test]
    #[should_panic(expected = "Metric must be invertible at the given coordinates.")]
    fn test_singular_metric_fn() {
        let singular_fn = |_coords: &DVector<f64>| DMatrix::<f64>::from_element(2, 2, 1.0);
        let spacetime = Spacetime::new(2, Box::new(singular_fn));
        let test_coords = DVector::from_vec(vec![0.0, 0.0]);
        spacetime.inverse_metric_at(&test_coords); // This should panic
    }
}

