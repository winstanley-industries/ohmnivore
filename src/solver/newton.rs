//! GPU-resident Newton-Raphson solver for nonlinear DC analysis.
//!
//! Iterates the Newton loop entirely on GPU using DS (double-single) precision:
//! evaluate nonlinear devices, assemble the system, solve the linear sub-problem
//! with BiCGSTAB, then check convergence with voltage limiting.

use crate::compiler::{GpuBjtDescriptor, GpuDiodeDescriptor, GpuMosfetDescriptor};
use crate::error::{OhmnivoreError, Result};
use crate::stats::Stats;
use std::time::Instant;

use super::backend::{SolverBackend, WgpuBuffer};
use super::bicgstab;
use super::ds_backend::WgpuDsBackend;
use super::nonlinear::{ConvergenceResult, NonlinearBackend};
use super::preconditioner;

/// Linear solve mode for each Newton iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum NewtonLinearMode {
    /// Solve Newton sub-problems with GPU BiCGSTAB.
    #[default]
    GpuBicgstab,
    /// Solve Newton sub-problems with CPU sparse LU.
    CpuSparseLu,
    /// Solve Newton sub-problems with CPU direct LU.
    CpuDirect,
}

/// Parameters for the Newton-Raphson iteration.
#[derive(Clone)]
pub struct NewtonParams {
    pub max_iterations: usize,
    pub abs_tol: f64,
    /// Maximum BiCGSTAB iterations for each Newton linear sub-solve.
    pub bicgstab_max_iterations: usize,
    /// Linear solver backend used inside each Newton iteration.
    pub linear_mode: NewtonLinearMode,
    /// Optional warm-start: upload this as the initial x instead of zeros.
    pub initial_guess: Option<Vec<f64>>,
}

impl Default for NewtonParams {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            abs_tol: 1e-4,
            bicgstab_max_iterations: bicgstab::DEFAULT_MAX_ITERATIONS,
            linear_mode: NewtonLinearMode::GpuBicgstab,
            initial_guess: None,
        }
    }
}

/// Run Newton-Raphson iteration on the GPU using DS (~f64) precision.
///
/// All device evaluation, assembly, voltage limiting, and convergence checking
/// operate in DS precision on GPU. The linear sub-problem is solved with
/// BiCGSTAB (also DS on GPU) or on CPU.
///
/// Returns the solution vector (f64) on success.
#[allow(clippy::too_many_arguments)]
pub fn newton_solve(
    ds_backend: &WgpuDsBackend,
    base_g_values_f64: &[f64],
    base_g_col_indices: &[u32],
    base_g_row_ptrs: &[u32],
    base_b_f64: &[f64],
    diode_descriptors: &[GpuDiodeDescriptor],
    bjt_descriptors: &[GpuBjtDescriptor],
    mosfet_descriptors: &[GpuMosfetDescriptor],
    system_size: usize,
    matrix_nnz: usize,
    params: &NewtonParams,
    mut stats: Option<&mut Stats>,
) -> Result<Vec<f64>> {
    let n_diodes = diode_descriptors.len() as u32;
    let n_bjts = bjt_descriptors.len() as u32;
    let n_mosfets = mosfet_descriptors.len() as u32;

    // Upload descriptors (u32 only, no lo buffer)
    let desc_buf = if n_diodes > 0 {
        ds_backend.upload_diode_descriptors(diode_descriptors)
    } else {
        ds_backend.new_buffer_flat(1)
    };
    let bjt_desc_buf = if n_bjts > 0 {
        ds_backend.upload_bjt_descriptors(bjt_descriptors)
    } else {
        ds_backend.new_buffer_flat(1)
    };
    let mosfet_desc_buf = if n_mosfets > 0 {
        ds_backend.upload_mosfet_descriptors(mosfet_descriptors)
    } else {
        ds_backend.new_buffer_flat(1)
    };

    // DS solution buffers (hi/lo pairs)
    let x_buf = ds_backend.new_buffer(system_size);
    if let Some(ref guess) = params.initial_guess {
        ds_backend.upload_vec_f64(guess, &x_buf);
    }
    let x_old_buf = ds_backend.new_buffer(system_size);

    // Flat eval output buffers (interleaved hi/lo per device)
    let diode_output_buf = ds_backend.new_buffer_flat(std::cmp::max(n_diodes as usize * 4, 1));
    let bjt_output_buf = ds_backend.new_buffer_flat(std::cmp::max(n_bjts as usize * 12, 1));
    let mosfet_output_buf = ds_backend.new_buffer_flat(std::cmp::max(n_mosfets as usize * 8, 1));

    // DS assembled output buffers (hi/lo pairs)
    let out_values_buf = ds_backend.new_buffer(matrix_nnz);
    let out_b_buf = ds_backend.new_buffer(system_size);
    let result_conv_buf = ds_backend.new_buffer(1); // unused by DS impl but required by trait

    // Upload base matrix values and RHS as DS
    let base_values_buf = ds_backend.upload_storage_buffer_f64(base_g_values_f64);
    let base_b_buf = ds_backend.new_buffer(system_size);
    ds_backend.upload_vec_f64(base_b_f64, &base_b_buf);

    for iter in 0..params.max_iterations {
        let _span = tracing::debug_span!("newton_iter", iter).entered();

        // Save current solution as x_old
        ds_backend.copy(&x_buf, &x_old_buf);

        // Step 1: Device evaluation (DS precision)
        let t = Instant::now();
        if n_diodes > 0 {
            ds_backend.evaluate_diodes(&desc_buf, &x_buf, &diode_output_buf, n_diodes);
        }
        if n_bjts > 0 {
            ds_backend.evaluate_bjts(&bjt_desc_buf, &x_buf, &bjt_output_buf, n_bjts);
        }
        if n_mosfets > 0 {
            ds_backend.evaluate_mosfets(&mosfet_desc_buf, &x_buf, &mosfet_output_buf, n_mosfets);
        }
        if let Some(ref mut s) = stats {
            s.device_eval += t.elapsed();
        }

        // Step 2: Assembly (DS precision)
        let t = Instant::now();
        ds_backend.assemble_nonlinear_system(
            &base_values_buf,
            &base_b_buf,
            &diode_output_buf,
            &desc_buf,
            &x_buf,
            &out_values_buf,
            &out_b_buf,
            n_diodes,
            matrix_nnz as u32,
            system_size as u32,
        );

        if n_bjts > 0 {
            ds_backend.assemble_bjt_stamps(
                &bjt_output_buf,
                &bjt_desc_buf,
                &x_buf,
                &out_values_buf,
                &out_b_buf,
                n_bjts,
            );
        }

        if n_mosfets > 0 {
            ds_backend.assemble_mosfet_stamps(
                &mosfet_output_buf,
                &mosfet_desc_buf,
                &x_buf,
                &out_values_buf,
                &out_b_buf,
                n_mosfets,
            );
        }
        if let Some(ref mut s) = stats {
            s.assembly += t.elapsed();
        }

        // Step 3: Download assembled matrix as f64 for NaN check + preconditioner
        let t = Instant::now();
        let mut assembled_f64 = vec![0.0f64; matrix_nnz];
        ds_backend.download_vec_f64(&out_values_buf, &mut assembled_f64);

        if assembled_f64
            .iter()
            .any(|v| v.is_nan() || v.is_infinite())
        {
            return Err(OhmnivoreError::NewtonNumericalError { iteration: iter });
        }

        if let Some(ref mut s) = stats {
            s.matrix_dl_ul += t.elapsed();
        }

        // Step 4: Solve the linear system
        let t = Instant::now();
        let col_indices_usize: Vec<usize> =
            base_g_col_indices.iter().map(|&c| c as usize).collect();
        let row_ptrs_usize: Vec<usize> = base_g_row_ptrs.iter().map(|&r| r as usize).collect();
        let csr_cpu = crate::sparse::CsrMatrix {
            nrows: system_size,
            ncols: system_size,
            values: assembled_f64.clone(),
            col_indices: col_indices_usize,
            row_pointers: row_ptrs_usize,
        };

        let bicgstab_iters = match params.linear_mode {
            NewtonLinearMode::CpuSparseLu => {
                let mut rhs_f64 = vec![0.0f64; system_size];
                ds_backend.download_vec_f64(&out_b_buf, &mut rhs_f64);

                let solve_result = match crate::solver::sparse_direct::solve_real_sparse_lu(
                    &csr_cpu, &rhs_f64,
                ) {
                    Ok(x) => {
                        let has_unsafe_scale = x
                            .iter()
                            .any(|v| !v.is_finite() || v.abs() > f32::MAX as f64);
                        if has_unsafe_scale {
                            Err(crate::error::OhmnivoreError::Solve(
                                "sparse LU produced out-of-range solution".into(),
                            ))
                        } else {
                            Ok(x)
                        }
                    }
                    Err(e) => Err(e),
                }
                .or_else(|sparse_err| {
                    tracing::debug!(
                        ?sparse_err,
                        "CPU sparse-LU failed, retrying dense CPU direct solve"
                    );
                    let cpu_solver = crate::solver::cpu::CpuSolver::new();
                    crate::solver::LinearSolver::solve_real(&cpu_solver, &csr_cpu, &rhs_f64)
                })?;

                ds_backend.upload_vec_f64(&solve_result, &x_buf);
                None
            }
            NewtonLinearMode::CpuDirect => {
                let mut rhs_f64 = vec![0.0f64; system_size];
                ds_backend.download_vec_f64(&out_b_buf, &mut rhs_f64);

                let cpu_solver = crate::solver::cpu::CpuSolver::new();
                let solve_result =
                    crate::solver::LinearSolver::solve_real(&cpu_solver, &csr_cpu, &rhs_f64)?;
                ds_backend.upload_vec_f64(&solve_result, &x_buf);
                None
            }
            NewtonLinearMode::GpuBicgstab => {
                let assembled_g_ds = ds_backend.upload_matrix_f64(
                    &assembled_f64,
                    base_g_col_indices,
                    base_g_row_ptrs,
                    system_size,
                );

                let x_solve_ds = ds_backend.new_buffer(system_size);

                // Compute preconditioner from assembled matrix (f64 precision)
                let mut has_zero_diag = false;
                let mut inv_diag_f64 = vec![0.0f64; system_size];
                for row in 0..system_size {
                    let start = base_g_row_ptrs[row] as usize;
                    let end = base_g_row_ptrs[row + 1] as usize;
                    let mut found = false;
                    for idx in start..end {
                        if base_g_col_indices[idx] as usize == row {
                            let val = assembled_f64[idx];
                            if val.abs() > 1e-30 {
                                inv_diag_f64[row] = 1.0 / val;
                            } else {
                                has_zero_diag = true;
                            }
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        has_zero_diag = true;
                    }
                }

                let linear_params = bicgstab::BiCgStabParams {
                    max_iterations: params.bicgstab_max_iterations,
                    tolerance: bicgstab::DEFAULT_TOLERANCE,
                };

                let iters = if has_zero_diag {
                    let isai = preconditioner::compute_isai(&csr_cpu, 1);

                    let ml_cols: Vec<u32> =
                        isai.m_l.col_indices.iter().map(|&c| c as u32).collect();
                    let ml_rows: Vec<u32> =
                        isai.m_l.row_pointers.iter().map(|&r| r as u32).collect();
                    let ml_gpu = ds_backend.upload_matrix_f64(
                        &isai.m_l.values,
                        &ml_cols,
                        &ml_rows,
                        system_size,
                    );

                    let mu_cols: Vec<u32> =
                        isai.m_u.col_indices.iter().map(|&c| c as u32).collect();
                    let mu_rows: Vec<u32> =
                        isai.m_u.row_pointers.iter().map(|&r| r as u32).collect();
                    let mu_gpu = ds_backend.upload_matrix_f64(
                        &isai.m_u.values,
                        &mu_cols,
                        &mu_rows,
                        system_size,
                    );

                    let tmp = ds_backend.new_buffer(system_size);

                    bicgstab::bicgstab_with_params(
                        ds_backend,
                        &assembled_g_ds,
                        &out_b_buf,
                        &x_solve_ds,
                        |b: &WgpuDsBackend, inp: &WgpuBuffer, out: &WgpuBuffer| {
                            b.spmv(&ml_gpu, inp, &tmp);
                            b.spmv(&mu_gpu, &tmp, out);
                        },
                        system_size,
                        linear_params,
                    )?
                } else {
                    let inv_diag_buf = ds_backend.upload_storage_buffer_f64(&inv_diag_f64);

                    bicgstab::bicgstab_with_params(
                        ds_backend,
                        &assembled_g_ds,
                        &out_b_buf,
                        &x_solve_ds,
                        |b: &WgpuDsBackend, inp: &WgpuBuffer, out: &WgpuBuffer| {
                            b.jacobi_apply(&inv_diag_buf, inp, out);
                        },
                        system_size,
                        linear_params,
                    )?
                };

                // Copy solution on GPU (no CPU round-trip)
                ds_backend.copy(&x_solve_ds, &x_buf);
                Some(iters)
            }
        };

        if let Some(ref mut s) = stats {
            s.linear_solve += t.elapsed();
            if let Some(iters) = bicgstab_iters {
                s.bicgstab_iters_per_newton.push(iters as u32);
            }
        }

        // Step 5a: BJT and MOSFET voltage limiting (before convergence check)
        let t = Instant::now();
        if n_bjts > 0 {
            ds_backend.limit_bjt_voltages(&x_old_buf, &x_buf, &bjt_desc_buf, n_bjts);
        }
        if n_mosfets > 0 {
            ds_backend.limit_mosfet_voltages(&x_old_buf, &x_buf, &mosfet_desc_buf, n_mosfets);
        }

        // Step 5b: Diode voltage limiting and convergence check
        let conv = ds_backend.limit_and_check_convergence(
            &x_old_buf,
            &x_buf,
            &desc_buf,
            &result_conv_buf,
            params.abs_tol,
            n_diodes,
            system_size as u32,
        );

        match conv {
            ConvergenceResult::Converged => {
                let mut result = vec![0.0f64; system_size];
                ds_backend.download_vec_f64(&x_buf, &mut result);

                // GPU NaN detection is unreliable (WGSL NaN comparison is
                // implementation-defined), so double-check on CPU.
                if result.iter().any(|v| v.is_nan() || v.is_infinite()) {
                    return Err(OhmnivoreError::NewtonNumericalError { iteration: iter });
                }

                if let Some(ref mut s) = stats {
                    s.convergence_check += t.elapsed();
                    s.newton_iterations = (iter + 1) as u32;
                    s.gpu_dispatches = ds_backend.dispatch_count();
                    s.gpu_readbacks = ds_backend.readback_count();
                }

                tracing::info!(iterations = iter + 1, "Newton converged");
                return Ok(result);
            }
            ConvergenceResult::NumericalError => {
                return Err(OhmnivoreError::NewtonNumericalError { iteration: iter });
            }
            ConvergenceResult::NotConverged { .. } => {
                // Continue iterating
            }
        }
        if let Some(ref mut s) = stats {
            s.convergence_check += t.elapsed();
        }
    }

    // Did not converge: download and compute final residual for error message
    let mut result = vec![0.0f64; system_size];
    ds_backend.download_vec_f64(&x_buf, &mut result);

    if result.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(OhmnivoreError::NewtonNumericalError {
            iteration: params.max_iterations,
        });
    }

    let mut x_old = vec![0.0f64; system_size];
    ds_backend.download_vec_f64(&x_old_buf, &mut x_old);

    let max_residual = result
        .iter()
        .zip(x_old.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    Err(OhmnivoreError::NewtonNotConverged {
        iterations: params.max_iterations,
        max_residual,
    })
}
