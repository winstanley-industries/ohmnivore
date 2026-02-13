//! Generic BiCGSTAB iterative solver.
//!
//! Implements right-preconditioned BiCGSTAB written against the `SolverBackend`
//! trait, enabling the same algorithm to run on any backend (GPU, CPU, etc.).

use crate::error::{OhmnivoreError, Result};

use super::backend::{GpuCsrMatrix, SolverBackend};

const MAX_ITERATIONS: usize = 10_000;
const TOLERANCE: f64 = 1e-5;

/// Solve Ax = b using right-preconditioned BiCGSTAB.
///
/// On entry, `x` should be zero-initialized and `b` should contain the
/// right-hand side vector. On success, `x` holds the solution and the
/// iteration count is returned.
///
/// The `preconditioner_apply` closure computes `output = M^{-1} * input`.
pub fn bicgstab<B: SolverBackend>(
    backend: &B,
    a: &GpuCsrMatrix,
    b: &B::Buffer,
    x: &B::Buffer,
    preconditioner_apply: impl Fn(&B, &B::Buffer, &B::Buffer),
    n: usize,
) -> Result<usize> {
    // Scratch buffers
    let r = backend.new_buffer(n);
    let r_hat = backend.new_buffer(n);
    let p = backend.new_buffer(n);
    let v = backend.new_buffer(n);
    let s = backend.new_buffer(n);
    let t = backend.new_buffer(n);
    let p_hat = backend.new_buffer(n);
    let s_hat = backend.new_buffer(n);

    // r = b, r_hat = b (x starts at 0 so r = b - A*0 = b)
    backend.copy(b, &r);
    backend.copy(b, &r_hat);

    // Compute ||b|| for relative tolerance
    let b_norm = backend.dot(&r, &r).sqrt();
    if b_norm < 1e-30 {
        // b is zero, so x = 0 is the solution
        return Ok(0);
    }
    let abs_tol = TOLERANCE * b_norm;

    let mut rho: f64 = 1.0;
    let mut alpha: f64 = 1.0;
    let mut omega: f64 = 1.0;

    for iter in 0..MAX_ITERATIONS {
        // rho_new = r_hat . r
        let rho_new = backend.dot(&r_hat, &r);
        if rho_new.abs() < 1e-30 {
            return Err(OhmnivoreError::Solve("BiCGSTAB breakdown: rho ~ 0".into()));
        }

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        backend.axpy(-omega, &v, &p);
        backend.scale(beta, &p);
        backend.axpy(1.0, &r, &p);

        // Right-preconditioned: p_hat = M^{-1} * p, v = A * p_hat
        preconditioner_apply(backend, &p, &p_hat);
        backend.spmv(a, &p_hat, &v);

        // alpha = rho / (r_hat . v)
        let r_hat_dot_v = backend.dot(&r_hat, &v);
        if r_hat_dot_v.abs() < 1e-30 {
            return Err(OhmnivoreError::Solve(
                "BiCGSTAB breakdown: r_hat.v ~ 0".into(),
            ));
        }
        alpha = rho / r_hat_dot_v;

        // s = r - alpha * v
        backend.copy(&r, &s);
        backend.axpy(-alpha, &v, &s);

        // Check early convergence: ||s||
        let s_norm = backend.dot(&s, &s).sqrt();
        if s_norm < abs_tol {
            backend.axpy(alpha, &p_hat, x);
            return Ok(iter + 1);
        }

        // Right-preconditioned: s_hat = M^{-1} * s, t = A * s_hat
        preconditioner_apply(backend, &s, &s_hat);
        backend.spmv(a, &s_hat, &t);

        // omega = (t . s) / (t . t)
        let t_dot_s = backend.dot(&t, &s);
        let t_dot_t = backend.dot(&t, &t);
        if t_dot_t.abs() < 1e-30 {
            return Err(OhmnivoreError::Solve(
                "BiCGSTAB breakdown: ||t|| ~ 0".into(),
            ));
        }
        omega = t_dot_s / t_dot_t;

        // x = x + alpha * p_hat + omega * s_hat
        backend.axpy(alpha, &p_hat, x);
        backend.axpy(omega, &s_hat, x);

        // r = s - omega * t
        backend.copy(&s, &r);
        backend.axpy(-omega, &t, &r);

        // Check convergence: ||r||
        let r_norm = backend.dot(&r, &r).sqrt();
        if r_norm.is_nan() || r_norm.is_infinite() {
            return Err(OhmnivoreError::Solve(
                "BiCGSTAB diverged: NaN/Inf in residual".into(),
            ));
        }
        if r_norm < abs_tol {
            return Ok(iter + 1);
        }

        if omega.abs() < 1e-30 {
            return Err(OhmnivoreError::Solve(
                "BiCGSTAB breakdown: omega ~ 0".into(),
            ));
        }
    }

    Ok(MAX_ITERATIONS)
}
