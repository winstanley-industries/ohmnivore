//! GPU-resident Newton-Raphson solver for nonlinear DC analysis.
//!
//! Iterates the Newton loop entirely on GPU: evaluate diodes, assemble the
//! nonlinear system (base G + diode stamps), solve the linear system with
//! BiCGSTAB, then check convergence with voltage limiting.

use crate::compiler::{GpuBjtDescriptor, GpuDiodeDescriptor, GpuMosfetDescriptor};
use crate::error::{OhmnivoreError, Result};
use crate::stats::Stats;
use std::time::Instant;

use super::backend::{SolverBackend, WgpuBackend, WgpuBuffer};
use super::bicgstab;
use super::ds_backend::WgpuDsBackend;
use super::nonlinear::{ConvergenceResult, NonlinearBackend};
use super::preconditioner;

/// Parameters for the Newton-Raphson iteration.
pub struct NewtonParams {
    pub max_iterations: usize,
    pub abs_tol: f64,
}

impl Default for NewtonParams {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            abs_tol: 1e-4,
        }
    }
}

/// Run Newton-Raphson iteration on the GPU to solve a nonlinear DC system.
///
/// The base conductance matrix values, column indices, and row pointers encode
/// the linear portion of the MNA system's CSR structure. `base_b` is the
/// excitation vector already on GPU. Diode contributions are evaluated and
/// stamped each iteration. The linear sub-problem is solved with BiCGSTAB.
///
/// Returns the solution vector (f64) on success.
#[allow(clippy::too_many_arguments)]
pub fn newton_solve(
    backend: &WgpuBackend,
    ds_backend: &WgpuDsBackend,
    base_b: &WgpuBuffer,
    base_g_values_f32: &[f32],
    base_g_col_indices: &[u32],
    base_g_row_ptrs: &[u32],
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

    // Upload diode descriptors to GPU (minimal buffer if empty to avoid zero-size)
    let desc_buf = if n_diodes > 0 {
        backend.upload_diode_descriptors(bytemuck::cast_slice(diode_descriptors))
    } else {
        backend.new_buffer(1)
    };

    // Upload BJT descriptors to GPU (minimal buffer if empty to avoid zero-size)
    let bjt_desc_buf = if n_bjts > 0 {
        backend.upload_bjt_descriptors(bjt_descriptors)
    } else {
        backend.new_buffer(1)
    };

    // Upload MOSFET descriptors to GPU (minimal buffer if empty to avoid zero-size)
    let mosfet_desc_buf = if n_mosfets > 0 {
        backend.upload_mosfet_descriptors(mosfet_descriptors)
    } else {
        backend.new_buffer(1)
    };

    // Create GPU buffers for Newton iteration
    let x_buf = backend.new_buffer(system_size); // current solution (zeros)
    let x_old_buf = backend.new_buffer(system_size); // previous iteration
    let diode_output_buf = backend.new_buffer(std::cmp::max(n_diodes as usize * 2, 1));
    let bjt_output_buf = backend.new_buffer(std::cmp::max(n_bjts as usize * 6, 1));
    let mosfet_output_buf = backend.new_buffer(std::cmp::max(n_mosfets as usize * 4, 1));
    let out_values_buf = backend.new_buffer(matrix_nnz); // assembled G values
    let out_b_buf = backend.new_buffer(system_size); // assembled RHS
    let result_conv_buf = backend.new_buffer(1); // convergence result scratch

    // Upload base G values as a storage buffer for the assembly shader
    let base_values_buf = backend.upload_storage_buffer(base_g_values_f32);

    let tolerance_f32 = params.abs_tol as f32;

    for iter in 0..params.max_iterations {
        let _span = tracing::debug_span!("newton_iter", iter).entered();
        // Save current solution as x_old
        backend.copy(&x_buf, &x_old_buf);

        // Step 1a-1c: Device evaluation
        let t = Instant::now();
        if n_diodes > 0 {
            backend.evaluate_diodes(&desc_buf, &x_buf, &diode_output_buf, n_diodes);
        }
        if n_bjts > 0 {
            backend.evaluate_bjts(&bjt_desc_buf, &x_buf, &bjt_output_buf, n_bjts);
        }
        if n_mosfets > 0 {
            backend.evaluate_mosfets(&mosfet_desc_buf, &x_buf, &mosfet_output_buf, n_mosfets);
        }
        if let Some(ref mut s) = stats { s.device_eval += t.elapsed(); }

        // Step 2a-2c: Assembly
        let t = Instant::now();
        backend.assemble_nonlinear_system(
            &base_values_buf,
            base_b,
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
            backend.assemble_bjt_stamps(
                &bjt_output_buf,
                &bjt_desc_buf,
                &x_buf,
                &out_values_buf,
                &out_b_buf,
                n_bjts,
            );
        }

        if n_mosfets > 0 {
            backend.assemble_mosfet_stamps(
                &mosfet_output_buf,
                &mosfet_desc_buf,
                &x_buf,
                &out_values_buf,
                &out_b_buf,
                n_mosfets,
            );
        }
        if let Some(ref mut s) = stats { s.assembly += t.elapsed(); }

        // Step 3: Upload assembled G as a new CSR matrix for BiCGSTAB
        // We need to download the assembled values, then re-upload as a proper CSR matrix.
        // The row_ptrs and col_indices don't change, only values do.
        let t = Instant::now();
        let mut assembled_values = vec![0.0f32; matrix_nnz];
        backend.download_vec(&out_values_buf, &mut assembled_values);

        // Early NaN/Inf detection — avoid expensive BiCGSTAB on corrupt matrices
        if assembled_values
            .iter()
            .any(|v| v.is_nan() || v.is_infinite())
        {
            return Err(OhmnivoreError::NewtonNumericalError { iteration: iter });
        }

        if let Some(ref mut s) = stats { s.matrix_dl_ul += t.elapsed(); }

        // Step 4: Solve the linear system with BiCGSTAB (on DS backend)
        let t = Instant::now();

        // Upload assembled f32 matrix to DS backend (lo = 0, assembly is f32)
        let assembled_g_ds = ds_backend.upload_matrix(
            &assembled_values,
            base_g_col_indices,
            base_g_row_ptrs,
            system_size,
        );

        // Upload assembled RHS to DS backend
        let mut rhs_f32 = vec![0.0f32; system_size];
        backend.download_vec(&out_b_buf, &mut rhs_f32);
        let x_solve_ds = ds_backend.new_buffer(system_size);
        let b_solve_ds = ds_backend.new_buffer(system_size);
        ds_backend.upload_vec(&rhs_f32, &b_solve_ds);

        // Compute Jacobi preconditioner from assembled matrix
        let mut has_zero_diag = false;
        let mut inv_diag_f64 = vec![0.0f64; system_size];
        for row in 0..system_size {
            let start = base_g_row_ptrs[row] as usize;
            let end = base_g_row_ptrs[row + 1] as usize;
            let mut found = false;
            for idx in start..end {
                if base_g_col_indices[idx] as usize == row {
                    let val = assembled_values[idx];
                    if val.abs() > 1e-30 {
                        inv_diag_f64[row] = 1.0 / (val as f64);
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

        let bicgstab_iters = if has_zero_diag {
            // Build a CPU-side CsrMatrix<f64> from the assembled values for ISAI
            let values_f64: Vec<f64> = assembled_values.iter().map(|&v| v as f64).collect();
            let col_indices_usize: Vec<usize> =
                base_g_col_indices.iter().map(|&c| c as usize).collect();
            let row_ptrs_usize: Vec<usize> = base_g_row_ptrs.iter().map(|&r| r as usize).collect();

            let csr_cpu = crate::sparse::CsrMatrix {
                nrows: system_size,
                ncols: system_size,
                values: values_f64,
                col_indices: col_indices_usize,
                row_pointers: row_ptrs_usize,
            };

            let isai = preconditioner::compute_isai(&csr_cpu, 0);

            // Upload ISAI factors to DS backend
            let ml_values: Vec<f32> = isai.m_l.values.iter().map(|&v| v as f32).collect();
            let ml_cols: Vec<u32> = isai.m_l.col_indices.iter().map(|&c| c as u32).collect();
            let ml_rows: Vec<u32> = isai.m_l.row_pointers.iter().map(|&r| r as u32).collect();
            let ml_gpu = ds_backend.upload_matrix(&ml_values, &ml_cols, &ml_rows, system_size);

            let mu_values: Vec<f32> = isai.m_u.values.iter().map(|&v| v as f32).collect();
            let mu_cols: Vec<u32> = isai.m_u.col_indices.iter().map(|&c| c as u32).collect();
            let mu_rows: Vec<u32> = isai.m_u.row_pointers.iter().map(|&r| r as u32).collect();
            let mu_gpu = ds_backend.upload_matrix(&mu_values, &mu_cols, &mu_rows, system_size);

            let tmp = ds_backend.new_buffer(system_size);

            bicgstab::bicgstab(
                ds_backend,
                &assembled_g_ds,
                &b_solve_ds,
                &x_solve_ds,
                |b: &WgpuDsBackend, inp: &WgpuBuffer, out: &WgpuBuffer| {
                    b.spmv(&ml_gpu, inp, &tmp);
                    b.spmv(&mu_gpu, &tmp, out);
                },
                system_size,
            )?
        } else {
            let inv_diag_buf = ds_backend.upload_storage_buffer_f64(&inv_diag_f64);

            bicgstab::bicgstab(
                ds_backend,
                &assembled_g_ds,
                &b_solve_ds,
                &x_solve_ds,
                |b: &WgpuDsBackend, inp: &WgpuBuffer, out: &WgpuBuffer| {
                    b.jacobi_apply(&inv_diag_buf, inp, out);
                },
                system_size,
            )?
        };
        if let Some(ref mut s) = stats {
            s.linear_solve += t.elapsed();
            s.bicgstab_iters_per_newton.push(bicgstab_iters as u32);
        }

        // Download DS result as f32 and upload to f32 backend's x_buf
        let mut solve_result_f32 = vec![0.0f32; system_size];
        ds_backend.download_vec(&x_solve_ds, &mut solve_result_f32);
        backend.upload_vec(&solve_result_f32, &x_buf);

        // Step 5a: BJT and MOSFET voltage limiting (before convergence check)
        let t = Instant::now();
        if n_bjts > 0 {
            backend.limit_bjt_voltages(&x_old_buf, &x_buf, &bjt_desc_buf, n_bjts);
        }
        if n_mosfets > 0 {
            backend.limit_mosfet_voltages(&x_old_buf, &x_buf, &mosfet_desc_buf, n_mosfets);
        }

        // Step 5b: Diode voltage limiting and convergence check
        let conv = backend.limit_and_check_convergence(
            &x_old_buf,
            &x_buf,
            &desc_buf,
            &result_conv_buf,
            tolerance_f32,
            n_diodes,
            system_size as u32,
        );

        match conv {
            ConvergenceResult::Converged => {
                // Download solution to CPU as f64
                let mut result_f32 = vec![0.0f32; system_size];
                backend.download_vec(&x_buf, &mut result_f32);

                // GPU NaN detection is unreliable (WGSL NaN comparison is
                // implementation-defined), so double-check on CPU.
                if result_f32.iter().any(|v| v.is_nan() || v.is_infinite()) {
                    return Err(OhmnivoreError::NewtonNumericalError { iteration: iter });
                }

                if let Some(ref mut s) = stats {
                    s.convergence_check += t.elapsed();
                    s.newton_iterations = (iter + 1) as u32;
                    s.gpu_dispatches = backend.dispatch_count();
                    s.gpu_readbacks = backend.readback_count();
                }

                tracing::info!(iterations = iter + 1, "Newton converged");
                return Ok(result_f32.iter().map(|&v| v as f64).collect());
            }
            ConvergenceResult::NumericalError => {
                return Err(OhmnivoreError::NewtonNumericalError { iteration: iter });
            }
            ConvergenceResult::NotConverged { .. } => {
                // Continue iterating
            }
        }
        if let Some(ref mut s) = stats { s.convergence_check += t.elapsed(); }
    }

    // Did not converge: download and compute final residual for error message
    let mut result_f32 = vec![0.0f32; system_size];
    backend.download_vec(&x_buf, &mut result_f32);

    // Check for NaN/Inf in the final solution
    if result_f32.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(OhmnivoreError::NewtonNumericalError {
            iteration: params.max_iterations,
        });
    }

    let mut x_old_f32 = vec![0.0f32; system_size];
    backend.download_vec(&x_old_buf, &mut x_old_f32);

    let max_residual = result_f32
        .iter()
        .zip(x_old_f32.iter())
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);

    Err(OhmnivoreError::NewtonNotConverged {
        iterations: params.max_iterations,
        max_residual,
    })
}
