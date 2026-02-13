//! Linear system solvers.

pub mod cpu;
pub mod gpu;
pub mod gpu_shaders;

use crate::error::Result;
use crate::sparse::CsrMatrix;
use num_complex::Complex64;

/// A solver for linear systems Ax = b.
pub trait LinearSolver {
    /// Solve a real-valued system Ax = b.
    fn solve_real(&self, a: &CsrMatrix<f64>, b: &[f64]) -> Result<Vec<f64>>;

    /// Solve a complex-valued system Ax = b.
    fn solve_complex(
        &self,
        a: &CsrMatrix<Complex64>,
        b: &[Complex64],
    ) -> Result<Vec<Complex64>>;
}
