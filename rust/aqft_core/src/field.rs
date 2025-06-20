//! Represents quantum fields and their properties.

use crate::spacetime::Spacetime;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::fmt;

// Represents the electromagnetic field via its 4-potential.
pub struct ElectromagneticField {
    /// The 4-potential A_mu as a function of spacetime coordinates.
    pub potential: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
}

impl fmt::Debug for ElectromagneticField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElectromagneticField")
            .field("potential", &"<function>")
            .finish()
    }
}

impl ElectromagneticField {
    /// Creates a new electromagnetic field from a given 4-potential function.
    pub fn new(potential: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>) -> Self {
        Self { potential }
    }

    /// Computes the partial derivatives of the 4-potential, d_mu(A_nu).
    fn derivatives_of_potential_at(
        &self,
        dim: usize,
        coords: &DVector<f64>,
        h: f64,
    ) -> DMatrix<f64> {
        let mut d_a = DMatrix::zeros(dim, dim);
        for mu in 0..dim { // The index of the derivative
            let mut coords_plus = coords.clone();
            coords_plus[mu] += h;
            let mut coords_minus = coords.clone();
            coords_minus[mu] -= h;

            let a_plus = (self.potential)(&coords_plus);
            let a_minus = (self.potential)(&coords_minus);

            for nu in 0..dim { // The index of the potential component
                d_a[(mu, nu)] = (a_plus[nu] - a_minus[nu]) / (2.0 * h);
            }
        }
        d_a
    }

    /// Computes the field strength tensor F_munu = d_mu(A_nu) - d_nu(A_mu).
    pub fn field_strength_tensor_at(&self, dim: usize, coords: &DVector<f64>) -> DMatrix<f64> {
        let h = 1e-5;
        let d_a = self.derivatives_of_potential_at(dim, coords, h);
        // F_munu = d_mu A_nu - d_nu A_mu, where d_a has components (d_mu, A_nu)
        &d_a - &d_a.transpose()
    }
}

/// Represents a charged scalar quantum field.
#[derive(Debug)]
pub struct ChargedScalarField<'a> {
    /// The mass of the field.
    pub mass: f64,
    /// The non-minimal coupling constant to the Ricci scalar.
    pub coupling_xi: f64,
    /// The charge of the field.
    pub charge: f64,
    /// A reference to the electromagnetic field it interacts with.
    pub em_field: &'a ElectromagneticField,
}

impl<'a> ChargedScalarField<'a> {
    /// Creates a new charged scalar field.
    pub fn new(
        mass: f64,
        coupling_xi: f64,
        charge: f64,
        em_field: &'a ElectromagneticField,
    ) -> Self {
        Self {
            mass,
            coupling_xi,
            charge,
            em_field,
        }
    }

    /// Computes the first partial derivatives of a complex scalar function.
    fn first_derivatives_complex(
        dim: usize,
        phi: &dyn Fn(&DVector<f64>) -> Complex<f64>,
        coords: &DVector<f64>,
        h: f64,
    ) -> DVector<Complex<f64>> {
        let mut d_phi = DVector::from_element(dim, Complex::new(0.0, 0.0));
        for i in 0..dim {
            let mut coords_plus = coords.clone();
            coords_plus[i] += h;
            let mut coords_minus = coords.clone();
            coords_minus[i] -= h;
            d_phi[i] = (phi(&coords_plus) - phi(&coords_minus)) / (2.0 * h);
        }
        d_phi
    }

    /// Computes the equation of motion for the charged scalar field.
    /// (Box_A + m^2 + xi*R)phi = 0, where Box_A is the gauge-covariant d'Alembertian.
    pub fn equation_of_motion(
        &self,
        spacetime: &Spacetime,
        phi: &dyn Fn(&DVector<f64>) -> Complex<f64>,
        coords: &DVector<f64>,
    ) -> Complex<f64> {
        let dim = spacetime.dimension;
        let h = 1e-5;
        let i = Complex::new(0.0, 1.0);

        let g_inv = spacetime.inverse_metric_at(coords);
        let christoffels = spacetime.christoffel_symbols_at(coords);

        // Define the gauge covariant derivative D_nu(phi)
        let d_nu_phi = |c: &DVector<f64>| {
            let d_phi = Self::first_derivatives_complex(dim, phi, c, h);
            let a_nu = (self.em_field.potential)(c);
            d_phi - a_nu.cast::<Complex<f64>>() * (i * self.charge * phi(c))
        };

        // Compute D_lambda(phi) at the current coordinates
        let d_lambda_phi_at_coords = d_nu_phi(coords);

        // Compute the partial derivative of D_nu(phi), which is d_mu(D_nu(phi))
        let mut d_mu_of_d_nu_phi = DMatrix::from_element(dim, dim, Complex::new(0.0, 0.0));
        for mu in 0..dim {
            let mut coords_plus = coords.clone();
            coords_plus[mu] += h;
            let mut coords_minus = coords.clone();
            coords_minus[mu] -= h;

            let d_nu_phi_plus = d_nu_phi(&coords_plus);
            let d_nu_phi_minus = d_nu_phi(&coords_minus);

            let deriv_row = (d_nu_phi_plus - d_nu_phi_minus) / Complex::new(2.0 * h, 0.0);
            d_mu_of_d_nu_phi.set_row(mu, &deriv_row.transpose());
        }

        // Compute the fully covariant derivative nabla_mu(D_nu(phi))
        let mut nabla_mu_d_nu_phi = d_mu_of_d_nu_phi;
        for mu in 0..dim {
            for nu in 0..dim {
                let mut correction = Complex::new(0.0, 0.0);
                for lambda in 0..dim {
                    correction += christoffels[lambda][(mu, nu)] * d_lambda_phi_at_coords[lambda];
                }
                nabla_mu_d_nu_phi[(mu, nu)] -= correction;
            }
        }

        // Contract with the inverse metric to get the Box operator
        let box_a_phi = g_inv.zip_map(&nabla_mu_d_nu_phi, |g, d| g * d).sum();

        let ricci_scalar = spacetime.ricci_scalar_at(coords);
        let phi_val = phi(coords);

        box_a_phi + (self.mass.powi(2) + self.coupling_xi * ricci_scalar) * phi_val
    }
}

/// Represents a scalar quantum field.
#[derive(Debug, PartialEq)]
pub struct ScalarField {
    /// The mass of the field.
    pub mass: f64,
    /// The non-minimal coupling constant to the Ricci scalar.
    pub coupling_xi: f64,
}

impl ScalarField {
    /// Creates a new scalar field with the given properties.
    pub fn new(mass: f64, coupling_xi: f64) -> Self {
        ScalarField { mass, coupling_xi }
    }

    /// Computes the first partial derivatives (gradient) of a scalar function.
    fn first_derivatives(
        dim: usize,
        phi: &dyn Fn(&DVector<f64>) -> f64,
        coords: &DVector<f64>,
        h: f64,
    ) -> DVector<f64> {
        let mut d_phi = DVector::zeros(dim);
        for i in 0..dim {
            let mut coords_plus = coords.clone();
            coords_plus[i] += h;
            let mut coords_minus = coords.clone();
            coords_minus[i] -= h;
            d_phi[i] = (phi(&coords_plus) - phi(&coords_minus)) / (2.0 * h);
        }
        d_phi
    }

    /// Computes the second partial derivatives (Hessian) of a scalar function.
    fn second_derivatives(
        dim: usize,
        phi: &dyn Fn(&DVector<f64>) -> f64,
        coords: &DVector<f64>,
        h: f64,
    ) -> DMatrix<f64> {
        let mut dd_phi = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    let mut coords_plus = coords.clone();
                    coords_plus[i] += h;
                    let mut coords_minus = coords.clone();
                    coords_minus[i] -= h;
                    dd_phi[(i, j)] =
                        (phi(&coords_plus) - 2.0 * phi(coords) + phi(&coords_minus)) / (h * h);
                } else {
                    let mut pp = coords.clone();
                    pp[i] += h;
                    pp[j] += h;
                    let mut pm = coords.clone();
                    pm[i] += h;
                    pm[j] -= h;
                    let mut mp = coords.clone();
                    mp[i] -= h;
                    mp[j] += h;
                    let mut mm = coords.clone();
                    mm[i] -= h;
                    mm[j] -= h;
                    dd_phi[(i, j)] = (phi(&pp) - phi(&pm) - phi(&mp) + phi(&mm)) / (4.0 * h * h);
                }
            }
        }
        dd_phi
    }

    /// Computes the d'Alembert (Box) operator acting on a scalar function.
    pub fn box_operator_at(
        &self,
        spacetime: &Spacetime,
        phi: &dyn Fn(&DVector<f64>) -> f64,
        coords: &DVector<f64>,
    ) -> f64 {
        let dim = spacetime.dimension;
        let h = 1e-5;

        let d_phi = Self::first_derivatives(dim, phi, coords, h);
        let dd_phi = Self::second_derivatives(dim, phi, coords, h);

        let g_inv = spacetime.inverse_metric_at(coords);
        let christoffels = spacetime.christoffel_symbols_at(coords);

        let mut box_phi = 0.0;
        for mu in 0..dim {
            for nu in 0..dim {
                let mut contracted_christoffel = 0.0;
                for lambda in 0..dim {
                    contracted_christoffel += christoffels[lambda][(mu, nu)] * d_phi[lambda];
                }
                box_phi += g_inv[(mu, nu)] * (dd_phi[(mu, nu)] - contracted_christoffel);
            }
        }
        box_phi
    }

    /// Computes the value of the Klein-Gordon operator acting on a field configuration.
    /// (Box + m^2 + xi * R) * phi
    pub fn equation_of_motion(
        &self,
        spacetime: &Spacetime,
        phi: &dyn Fn(&DVector<f64>) -> f64,
        coords: &DVector<f64>,
    ) -> f64 {
        let box_phi = self.box_operator_at(spacetime, phi, coords);
        let ricci_scalar = spacetime.ricci_scalar_at(coords);
        let phi_val = phi(coords);

        box_phi + self.mass.powi(2) * phi_val + self.coupling_xi * ricci_scalar * phi_val
    }

    /// Computes the second covariant derivative of a scalar function, nabla_mu nabla_nu phi.
    fn second_covariant_derivative_at(
        spacetime: &Spacetime,
        phi: &dyn Fn(&DVector<f64>) -> f64,
        coords: &DVector<f64>,
        h: f64,
    ) -> DMatrix<f64> {
        let dim = spacetime.dimension;
        let d_phi = Self::first_derivatives(dim, phi, coords, h);
        let dd_phi = Self::second_derivatives(dim, phi, coords, h);
        let christoffels = spacetime.christoffel_symbols_at(coords);

        let mut dd_cov_phi = DMatrix::zeros(dim, dim);
        for mu in 0..dim {
            for nu in 0..dim {
                let mut contracted_christoffel = 0.0;
                for lambda in 0..dim {
                    contracted_christoffel += christoffels[lambda][(mu, nu)] * d_phi[lambda];
                }
                dd_cov_phi[(mu, nu)] = dd_phi[(mu, nu)] - contracted_christoffel;
            }
        }
        dd_cov_phi
    }

    /// Computes the stress-energy tensor T_munu for the scalar field.
    pub fn stress_energy_tensor_at(
        &self,
        spacetime: &Spacetime,
        phi: &dyn Fn(&DVector<f64>) -> f64,
        coords: &DVector<f64>,
    ) -> DMatrix<f64> {
        let dim = spacetime.dimension;
        let h = 1e-5;

        let d_phi = Self::first_derivatives(dim, phi, coords, h);
        let phi_val = phi(coords);
        let g = spacetime.metric_at(coords);

        // Standard part of the tensor (minimal coupling)
        let term1 = &d_phi * d_phi.transpose();
        let g_inv = spacetime.inverse_metric_at(coords);
        let kinetic_term_contracted = (&d_phi.transpose() * &g_inv * &d_phi)[(0, 0)];
        let potential_term = self.mass.powi(2) * phi_val.powi(2);
        let lagrangian_term = -0.5 * (kinetic_term_contracted + potential_term);
        let mut t_munu = term1 + &g * lagrangian_term;

        // Non-minimal coupling part
        if self.coupling_xi.abs() > 1e-9 {
            let ricci_tensor = spacetime.ricci_tensor_at(coords);
            let box_phi = self.box_operator_at(spacetime, phi, coords);
            let dd_cov_phi = Self::second_covariant_derivative_at(spacetime, phi, coords, h);

            let term_g_box = &g * box_phi;
            let term_ricci = ricci_tensor * phi_val;

            t_munu += self.coupling_xi * (term_g_box - dd_cov_phi + term_ricci);
        }

        t_munu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spacetime::Spacetime;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_create_scalar_field() {
        let field = ScalarField::new(1.0, 0.1);
        assert_eq!(field.mass, 1.0);
        assert_eq!(field.coupling_xi, 0.1);
    }

    #[test]
    fn test_klein_gordon_minkowski() {
        // Spacetime setup
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]); // t=1, x=2

        // Field setup
        let field = ScalarField::new(1.0, 0.0); // m=1, xi=0

        // Test with phi = t^2. Box phi = -2. EOM = -2 + m^2 * t^2 = -2 + 1*1^2 = -1.
        let phi_t = |c: &DVector<f64>| c[0].powi(2);
        let eom_val_t = field.equation_of_motion(&spacetime, &phi_t, &coords);
        assert!((eom_val_t - -1.0).abs() < 1e-5, "EOM for t^2 failed");

        // Test with phi = x^2. Box phi = 2. EOM = 2 + m^2 * x^2 = 2 + 1*2^2 = 6.
        let phi_x = |c: &DVector<f64>| c[1].powi(2);
        let eom_val_x = field.equation_of_motion(&spacetime, &phi_x, &coords);
        assert!((eom_val_x - 6.0).abs() < 1e-5, "EOM for x^2 failed");
    }

    #[test]
    fn test_stress_energy_tensor_minkowski() {
        // Spacetime setup
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let coords = DVector::from_vec(vec![2.0, 1.0, 1.0, 1.0]);

        // Field setup (massless, minimal coupling)
        let field = ScalarField::new(0.0, 0.0);

        // Test with phi = t.
        // T_munu = d_mu phi d_nu phi - 1/2 g_munu (d_rho phi d^rho phi)
        // d_phi = (1, 0, 0, 0). d_rho phi d^rho phi = g^00 (d_0 phi)^2 = -1.
        // T = diag(1,0,0,0) - 1/2 * diag(-1,1,1,1) * (-1) = diag(0.5, 0.5, 0.5, 0.5)
        let phi = |c: &DVector<f64>| c[0]; // phi = t
        let t_munu = field.stress_energy_tensor_at(&spacetime, &phi, &coords);

        let expected = DMatrix::from_diagonal_element(4, 4, 0.5);

        assert!(
            (t_munu - expected).abs().max() < 1e-5,
            "T_munu for phi=t failed"
        );
    }

    #[test]
    fn test_stress_energy_tensor_non_minimal_minkowski() {
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Field with non-minimal coupling
        let field = ScalarField::new(1.0, 0.1); // m=1, xi=0.1

        // Test with phi = constant = 2.0. All derivatives are zero.
        // T_munu = -1/2 g_munu (m^2 phi^2) + xi (g_munu Box phi - nabla_mu_nu phi + R_munu phi^2)
        // In Minkowski, Box phi=0, nabla_mu_nu phi=0, R_munu=0.
        // T_munu = -1/2 g_munu m^2 phi^2 = -1/2 * diag(-1,1,1,1) * 1^2 * 2^2 = diag(2, -2, -2, -2)
        let phi = |_: &DVector<f64>| 2.0;
        let t_munu = field.stress_energy_tensor_at(&spacetime, &phi, &coords);

        let mut expected = DMatrix::from_diagonal_element(4, 4, -2.0);
        expected[(0, 0)] = 2.0;

        assert!((t_munu - expected).abs().max() < 1e-9, "T_munu for constant phi failed");
    }

    #[test]
    fn test_field_strength_tensor() {
        // Potential for a constant electric field E in the x-direction.
        // A_mu = (-E*x, 0, 0, 0). Let E=1, so A_0 = -x = -coords[1].
        let potential = |coords: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![-coords[1], 0.0, 0.0, 0.0])
        };
        let em_field = ElectromagneticField::new(Box::new(potential));
        let coords = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        // F_munu = d_mu A_nu - d_nu A_mu
        // F_01 = d_0 A_1 - d_1 A_0 = 0 - (-1) = 1 (E_x)
        // F_10 = d_1 A_0 - d_0 A_1 = -1 - 0 = -1 (-E_x)
        // All other components are zero.
        let f_munu = em_field.field_strength_tensor_at(4, &coords);

        let mut expected = DMatrix::zeros(4, 4);
        expected[(0, 1)] = 1.0;
        expected[(1, 0)] = -1.0;

        assert!(
            (f_munu - expected).abs().max() < 1e-9,
            "Field strength tensor calculation failed"
        );
    }

    #[test]
    fn test_charged_klein_gordon_minkowski() {
        // Spacetime setup
        let minkowski_fn = |_coords: &DVector<f64>| {
            let mut g = DMatrix::from_diagonal_element(4, 4, 1.0);
            g[(0, 0)] = -1.0;
            g
        };
        let spacetime = Spacetime::new(4, Box::new(minkowski_fn));
        let coords = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // EM field setup (zero potential)
        let zero_potential = |_: &DVector<f64>| DVector::zeros(4);
        let em_field = ElectromagneticField::new(Box::new(zero_potential));

        // Charged field setup
        let charged_field = ChargedScalarField::new(1.0, 0.0, 1.0, &em_field);

        // With A_mu=0, the equation should be the same as the uncharged KG equation.
        // Test with phi = t^2. Box phi = -2. EOM = -2 + m^2 * t^2 = -2 + 1*1^2 = -1.
        let phi_t = |c: &DVector<f64>| Complex::new(c[0].powi(2), 0.0);
        let eom_val_t = charged_field.equation_of_motion(&spacetime, &phi_t, &coords);
        let expected = Complex::new(-1.0, 0.0);

        assert!(
            (eom_val_t - expected).norm() < 1e-5,
            "Charged EOM for A_mu=0 failed"
        );
    }
}
