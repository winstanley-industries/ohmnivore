//! Linear system solvers.

pub mod backend;
pub mod bicgstab;
pub mod comm;
pub mod cpu;
pub mod distributed_bicgstab;
pub mod distributed_newton;
pub mod distributed_preconditioner;
pub mod ds_backend;
pub mod ds_shaders;
pub mod gpu;
pub mod gpu_shaders;
pub mod newton;
pub mod nonlinear;
pub mod partition;
pub mod preconditioner;

use crate::error::Result;
use crate::sparse::CsrMatrix;
use num_complex::Complex64;

/// A solver for linear systems Ax = b.
pub trait LinearSolver {
    /// Solve a real-valued system Ax = b.
    fn solve_real(&self, a: &CsrMatrix<f64>, b: &[f64]) -> Result<Vec<f64>>;

    /// Solve a complex-valued system Ax = b.
    fn solve_complex(&self, a: &CsrMatrix<Complex64>, b: &[Complex64]) -> Result<Vec<Complex64>>;
}
