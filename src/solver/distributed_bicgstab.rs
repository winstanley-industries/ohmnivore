//! Distributed BiCGSTAB solver with communication-aware reductions.
//!
//! Wraps the existing SolverBackend operations with a CommunicationBackend
//! for distributed dot products (all_reduce_sum) and halo exchange.
//! The non-distributed BiCGSTAB in `bicgstab.rs` remains for single-GPU
//! backward compatibility.

use crate::error::{OhmnivoreError, Result};

use super::backend::{GpuCsrMatrix, SolverBackend};
use super::comm::CommunicationBackend;

const MAX_ITERATIONS: usize = 10_000;
const TOLERANCE: f64 = 1e-5;

/// Solve Ax = b using distributed right-preconditioned BiCGSTAB.
///
/// `n_owned` is the number of owned nodes (dot products sum over 0..n_owned
/// to avoid double-counting in a multi-rank scenario). `n_total` is owned + halo
/// (buffer sizes).
///
/// `halo_sync` is called before each SpMV/preconditioner apply to exchange
/// boundary values with neighbor ranks.
///
/// When `comm` is `SingleProcessComm` and `n_owned == n_total`, this behaves
/// identically to the non-distributed BiCGSTAB.
#[allow(clippy::too_many_arguments)]
pub fn distributed_bicgstab<B: SolverBackend>(
    backend: &B,
    a: &GpuCsrMatrix,
    b: &B::Buffer,
    x: &B::Buffer,
    preconditioner_apply: impl Fn(&B, &B::Buffer, &B::Buffer),
    halo_sync: impl Fn(&B::Buffer),
    n_owned: usize,
    n_total: usize,
    comm: &dyn CommunicationBackend,
) -> Result<usize> {
    let _span = tracing::debug_span!("distributed_bicgstab", n_owned, n_total).entered();

    let r = backend.new_buffer(n_total);
    let r_hat = backend.new_buffer(n_total);
    let p = backend.new_buffer(n_total);
    let v = backend.new_buffer(n_total);
    let s = backend.new_buffer(n_total);
    let t = backend.new_buffer(n_total);
    let p_hat = backend.new_buffer(n_total);
    let s_hat = backend.new_buffer(n_total);

    // r = b, r_hat = b (x starts at 0 so r = b - A*0 = b)
    backend.copy(b, &r);
    backend.copy(b, &r_hat);

    // ||b|| via distributed dot (sum local partials over owned entries, all_reduce)
    let local_b_sq = backend.dot_n(&r, &r, n_owned);
    let b_norm = comm.all_reduce_sum(local_b_sq).sqrt();
    if b_norm < 1e-30 {
        return Ok(0);
    }
    let abs_tol = TOLERANCE * b_norm;

    let mut rho: f64 = 1.0;
    let mut alpha: f64 = 1.0;
    let mut omega: f64 = 1.0;

    for iter in 0..MAX_ITERATIONS {
        let rho_new = comm.all_reduce_sum(backend.dot_n(&r_hat, &r, n_owned));
        if rho_new.abs() < 1e-30 {
            return Err(OhmnivoreError::Solve("BiCGSTAB breakdown: rho ~ 0".into()));
        }

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        backend.axpy(-omega, &v, &p);
        backend.scale(beta, &p);
        backend.axpy(1.0, &r, &p);

        // Halo sync before preconditioner apply and SpMV
        halo_sync(&p);
        preconditioner_apply(backend, &p, &p_hat);
        halo_sync(&p_hat);
        backend.spmv(a, &p_hat, &v);

        let r_hat_dot_v = comm.all_reduce_sum(backend.dot_n(&r_hat, &v, n_owned));
        if r_hat_dot_v.abs() < 1e-30 {
            return Err(OhmnivoreError::Solve(
                "BiCGSTAB breakdown: r_hat.v ~ 0".into(),
            ));
        }
        alpha = rho / r_hat_dot_v;

        // s = r - alpha * v
        backend.copy(&r, &s);
        backend.axpy(-alpha, &v, &s);

        let s_norm = comm.all_reduce_sum(backend.dot_n(&s, &s, n_owned)).sqrt();
        if s_norm < abs_tol {
            backend.axpy(alpha, &p_hat, x);
            tracing::debug!(iterations = iter + 1, "distributed BiCGSTAB converged");
            return Ok(iter + 1);
        }

        halo_sync(&s);
        preconditioner_apply(backend, &s, &s_hat);
        halo_sync(&s_hat);
        backend.spmv(a, &s_hat, &t);

        let t_dot_s = comm.all_reduce_sum(backend.dot_n(&t, &s, n_owned));
        let t_dot_t = comm.all_reduce_sum(backend.dot_n(&t, &t, n_owned));
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

        let r_norm = comm.all_reduce_sum(backend.dot_n(&r, &r, n_owned)).sqrt();
        if r_norm.is_nan() || r_norm.is_infinite() {
            return Err(OhmnivoreError::Solve(
                "BiCGSTAB diverged: NaN/Inf in residual".into(),
            ));
        }
        if r_norm < abs_tol {
            tracing::debug!(iterations = iter + 1, "distributed BiCGSTAB converged");
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

/// Convenience wrapper: solve Ax = b using distributed BiCGSTAB on the GPU
/// with ISAI preconditioner and the given communication backend.
///
/// This mirrors `GpuSolver::solve_real` but routes through the distributed
/// path. For `SingleProcessComm`, behaviour is identical to the single-GPU path.
pub fn distributed_bicgstab_solve(
    a: &crate::sparse::CsrMatrix<f64>,
    b: &[f64],
    comm: &dyn CommunicationBackend,
) -> Result<Vec<f64>> {
    let n = a.nrows;
    if a.ncols != n || b.len() != n {
        return Err(OhmnivoreError::Solve(format!(
            "dimension mismatch: matrix is {}x{}, rhs length is {}",
            a.nrows,
            a.ncols,
            b.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let backend = super::backend::WgpuBackend::new()?;

    let values_f32: Vec<f32> = a.values.iter().map(|&v| v as f32).collect();
    let col_indices_u32: Vec<u32> = a.col_indices.iter().map(|&c| c as u32).collect();
    let row_ptrs_u32: Vec<u32> = a.row_pointers.iter().map(|&r| r as u32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();

    let gpu_matrix = backend.upload_matrix(&values_f32, &col_indices_u32, &row_ptrs_u32, n);
    let x_buf = backend.new_buffer(n);
    let b_buf = backend.new_buffer(n);
    backend.upload_vec(&b_f32, &b_buf);

    // Compute ISAI preconditioner
    let isai = super::preconditioner::compute_isai(a, 1);

    let ml_values: Vec<f32> = isai.m_l.values.iter().map(|&v| v as f32).collect();
    let ml_cols: Vec<u32> = isai.m_l.col_indices.iter().map(|&c| c as u32).collect();
    let ml_ptrs: Vec<u32> = isai.m_l.row_pointers.iter().map(|&r| r as u32).collect();
    let ml_gpu = backend.upload_matrix(&ml_values, &ml_cols, &ml_ptrs, n);

    let mu_values: Vec<f32> = isai.m_u.values.iter().map(|&v| v as f32).collect();
    let mu_cols: Vec<u32> = isai.m_u.col_indices.iter().map(|&c| c as u32).collect();
    let mu_ptrs: Vec<u32> = isai.m_u.row_pointers.iter().map(|&r| r as u32).collect();
    let mu_gpu = backend.upload_matrix(&mu_values, &mu_cols, &mu_ptrs, n);

    let tmp = backend.new_buffer(n);

    // Single-process: n_owned == n_total == n
    distributed_bicgstab(
        &backend,
        &gpu_matrix,
        &b_buf,
        &x_buf,
        |b: &super::backend::WgpuBackend,
         inp: &super::backend::WgpuBuffer,
         out: &super::backend::WgpuBuffer| {
            b.spmv(&ml_gpu, inp, &tmp);
            b.spmv(&mu_gpu, &tmp, out);
        },
        |_buf| {
            // No-op halo sync for single-process
        },
        n,
        n,
        comm,
    )?;

    let mut result_f32 = vec![0.0f32; n];
    backend.download_vec(&x_buf, &mut result_f32);
    Ok(result_f32.iter().map(|&v| v as f64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::comm::SingleProcessComm;
    use crate::sparse::CsrMatrix;

    /// Solve a small system with distributed BiCGSTAB using SingleProcessComm
    /// and compare against the known CPU direct solution.
    #[test]
    fn distributed_bicgstab_single_process_matches_direct() {
        // 3x3 SPD system: [[4,-1,0],[-1,4,-1],[0,-1,4]] x = [1,2,3]
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let b = vec![1.0, 2.0, 3.0];

        // CPU reference solution
        let cpu = crate::solver::cpu::CpuSolver::new();
        let reference = crate::solver::LinearSolver::solve_real(&cpu, &a, &b).unwrap();

        // Distributed BiCGSTAB with single process (should behave identically)
        let comm = SingleProcessComm;
        let result = match distributed_bicgstab_solve(&a, &b, &comm) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        for (r, e) in result.iter().zip(reference.iter()) {
            assert!((r - e).abs() < 1e-3, "mismatch: got {r}, expected {e}");
        }
    }
}
