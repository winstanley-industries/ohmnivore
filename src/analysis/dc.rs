//! DC operating point analysis.
//!
//! Solves Gx = b_dc for the DC operating point.
//! The solution vector x contains node voltages followed by branch currents.
//!
//! For linear circuits, uses the standard linear solver path.
//! For nonlinear circuits (with diodes, BJTs, or MOSFETs), uses GPU-accelerated Newton-Raphson.

use super::DcResult;
use crate::compiler::MnaSystem;
use crate::error::Result;
use crate::solver::LinearSolver;
use crate::stats::Stats;

/// Run DC operating point analysis.
///
/// 1. Check whether the circuit has nonlinear elements (diodes, BJTs, MOSFETs).
/// 2. If linear: solve Gx = b_dc with the provided linear solver.
/// 3. If nonlinear: use the GPU Newton-Raphson solver.
/// 4. Map solution indices back to node/branch names.
pub fn run(
    system: &MnaSystem,
    solver: &dyn LinearSolver,
    mut stats: Option<&mut Stats>,
) -> Result<DcResult> {
    let _span = tracing::info_span!("dc_analysis").entered();
    let x = if !system.diode_descriptors.is_empty()
        || !system.bjt_descriptors.is_empty()
        || !system.mosfet_descriptors.is_empty()
    {
        run_nonlinear(system, stats)?
    } else {
        if let Some(ref mut s) = stats {
            s.linear_solves += 1;
        }
        solver.solve_real(&system.g, &system.b_dc)?
    };

    map_solution(system, &x)
}

/// Find CSR value-array positions of diagonal entries for the first `n_nodes` rows.
fn find_node_diagonal_positions(system: &MnaSystem) -> Vec<usize> {
    let n_nodes = system.node_names.len();
    (0..n_nodes)
        .filter_map(|row| {
            let start = system.g.row_pointers[row];
            let end = system.g.row_pointers[row + 1];
            (start..end).find(|&idx| system.g.col_indices[idx] == row)
        })
        .collect()
}

/// Run the nonlinear DC path using GPU Newton-Raphson iteration.
///
/// First attempts a direct Newton solve. If that fails (typically due to poor
/// initial conditioning with MOSFETs in cutoff), falls back to GMIN stepping:
/// solve with artificially large GMIN, then progressively reduce it toward the
/// real value, using each converged solution as the initial guess for the next.
fn run_nonlinear(system: &MnaSystem, mut stats: Option<&mut Stats>) -> Result<Vec<f64>> {
    use crate::solver::backend::{SolverBackend, WgpuBackend};
    use crate::solver::ds_backend::WgpuDsBackend;
    use crate::solver::newton::{NewtonLinearMode, NewtonParams};

    let backend = WgpuBackend::new()?;
    let ds_backend = WgpuDsBackend::new()?;

    // Convert base G matrix CSR indices to u32 for GPU
    let col_indices_u32: Vec<u32> = system.g.col_indices.iter().map(|&c| c as u32).collect();
    let row_ptrs_u32: Vec<u32> = system.g.row_pointers.iter().map(|&r| r as u32).collect();
    let b_dc_f32: Vec<f32> = system.b_dc.iter().map(|&v| v as f32).collect();

    // Upload base RHS vector to GPU
    let base_b_buf = backend.new_buffer(system.size);
    backend.upload_vec(&b_dc_f32, &base_b_buf);

    let matrix_nnz = system.g.values.len();

    // Attempt 1: direct Newton solve with unmodified GMIN.
    // Works for simple/small circuits where the Jacobian is manageable.
    let direct_params = NewtonParams {
        // Avoid spending excessive time in a doomed direct attempt on large
        // ill-conditioned nonlinear systems; fallback stepping handles those.
        bicgstab_max_iterations: 1000,
        ..NewtonParams::default()
    };

    match gmin_newton_step(
        &backend,
        &ds_backend,
        &base_b_buf,
        &system.b_dc,
        system,
        &col_indices_u32,
        &row_ptrs_u32,
        matrix_nnz,
        &[],
        0.0,
        &direct_params,
        stats.as_deref_mut(),
    ) {
        Ok(x) => return Ok(x),
        Err(e) => {
            tracing::info!(?e, "Direct Newton failed, trying source stepping");
        }
    }

    // Source stepping: gradually ramp sources from 0 to full value.
    // Most robust for MOSFET-heavy circuits (inverter chains, etc.) because
    // it naturally walks through the threshold transition region.
    let diag_positions = find_node_diagonal_positions(system);

    match run_source_stepping(
        &backend,
        &ds_backend,
        system,
        &col_indices_u32,
        &row_ptrs_u32,
        matrix_nnz,
        &diag_positions,
        stats.as_deref_mut(),
    ) {
        Ok(x) => return Ok(x),
        Err(e) => {
            tracing::info!(?e, "Source stepping failed, falling back to GMIN stepping");
        }
    }

    // GMIN stepping: progressively reduce extra conductance to ground.
    // Large initial GMIN regularizes the Jacobian so BiCGSTAB can solve the
    // linear subproblem. Each subsequent step uses the previous converged
    // solution as initial guess.
    //
    // Adaptive schedule: try a 1000x jump; if it fails, subdivide with 10x jumps.
    // Intermediate steps use limited Newton iterations to fail fast.
    let mut x_prev: Option<Vec<f32>> = None;
    // Adaptive GMIN schedule: try 1000x jumps, subdivide on failure.
    // On success: record the GMIN level and attempt 1000x jump.
    // On failure: try progressively smaller steps (10x) from the last
    // successful level until we find one that works.
    let mut last_good_gmin = 1.0_f64; // above our starting point
    let mut target_gmin = 1e-1_f64;
    let mut subdivisions = 0_u32;

    while target_gmin > 1e-13 {
        let params = NewtonParams {
            max_iterations: 15,
            // Intermediate continuation steps should fail fast if the linear
            // subproblem is too ill-conditioned for the current GMIN level.
            bicgstab_max_iterations: 300,
            linear_mode: NewtonLinearMode::GpuBicgstab,
            initial_guess: x_prev.clone(),
            ..NewtonParams::default()
        };

        let result = gmin_newton_step(
            &backend,
            &ds_backend,
            &base_b_buf,
            &system.b_dc,
            system,
            &col_indices_u32,
            &row_ptrs_u32,
            matrix_nnz,
            &diag_positions,
            target_gmin,
            &params,
            stats.as_deref_mut(),
        )
        .or_else(|gpu_err| {
            // Fallback path: try CPU sparse-LU solve for this continuation step
            // before we subdivide further.
            let mut cpu_params = params.clone();
            cpu_params.linear_mode = NewtonLinearMode::CpuSparseLu;
            match gmin_newton_step(
                &backend,
                &ds_backend,
                &base_b_buf,
                &system.b_dc,
                system,
                &col_indices_u32,
                &row_ptrs_u32,
                matrix_nnz,
                &diag_positions,
                target_gmin,
                &cpu_params,
                stats.as_deref_mut(),
            ) {
                Ok(x) => {
                    tracing::debug!(
                        ?gpu_err,
                        target_gmin,
                        "GMIN GPU step failed, CPU sparse-LU fallback succeeded"
                    );
                    Ok(x)
                }
                Err(cpu_err) => {
                    tracing::debug!(
                        ?gpu_err,
                        ?cpu_err,
                        target_gmin,
                        "GMIN step failed on both GPU and CPU fallback"
                    );
                    Err(cpu_err)
                }
            }
        });

        match result {
            Ok(x) => {
                x_prev = Some(x.iter().map(|&v| v as f32).collect());
                last_good_gmin = target_gmin;
                subdivisions = 0;
                // Try 1000x jump
                target_gmin *= 1e-3;
            }
            Err(e) => {
                subdivisions += 1;
                if subdivisions > 13 {
                    tracing::error!("GMIN stepping exhausted after {subdivisions} subdivisions");
                    return Err(e);
                }
                // Step 10^(-subdivisions) from the last good point
                target_gmin = last_good_gmin * 10.0_f64.powi(-(subdivisions as i32));
                tracing::debug!(
                    ?e,
                    target_gmin,
                    subdivisions,
                    "GMIN step failed, subdividing"
                );
            }
        }
    }

    // Final step: solve with true GMIN (extra_gmin = 0), full Newton iterations
    let mut final_params = NewtonParams::default();
    final_params.initial_guess = x_prev.clone();

    match gmin_newton_step(
        &backend,
        &ds_backend,
        &base_b_buf,
        &system.b_dc,
        system,
        &col_indices_u32,
        &row_ptrs_u32,
        matrix_nnz,
        &diag_positions,
        0.0,
        &final_params,
        stats.as_deref_mut(),
    ) {
        Ok(x) => Ok(x),
        Err(final_err) => {
            if let Some(seed) = x_prev {
                tracing::info!(
                    ?final_err,
                    "Final true-GMIN Newton failed, trying pseudo-transient continuation"
                );
                run_pseudo_transient_continuation(
                    &backend,
                    &ds_backend,
                    system,
                    &col_indices_u32,
                    &row_ptrs_u32,
                    matrix_nnz,
                    &diag_positions,
                    seed,
                    stats.as_deref_mut(),
                )
            } else {
                Err(final_err)
            }
        }
    }
}

/// Run a single Newton solve with modified GMIN.
#[allow(clippy::too_many_arguments)]
fn gmin_newton_step(
    backend: &crate::solver::backend::WgpuBackend,
    ds_backend: &crate::solver::ds_backend::WgpuDsBackend,
    base_b_buf: &crate::solver::backend::WgpuBuffer,
    base_b_f64: &[f64],
    system: &MnaSystem,
    col_indices_u32: &[u32],
    row_ptrs_u32: &[u32],
    matrix_nnz: usize,
    diag_positions: &[usize],
    extra_gmin: f64,
    params: &crate::solver::newton::NewtonParams,
    stats: Option<&mut Stats>,
) -> Result<Vec<f64>> {
    use crate::solver::newton::newton_solve;

    let mut step_values_f64 = system.g.values.clone();
    for &pos in diag_positions {
        step_values_f64[pos] += extra_gmin;
    }
    let step_values_f32: Vec<f32> = step_values_f64.iter().map(|&v| v as f32).collect();

    tracing::debug!(extra_gmin, "GMIN stepping");

    newton_solve(
        backend,
        ds_backend,
        base_b_buf,
        &step_values_f32,
        &step_values_f64,
        col_indices_u32,
        row_ptrs_u32,
        base_b_f64,
        &system.diode_descriptors,
        &system.bjt_descriptors,
        &system.mosfet_descriptors,
        system.size,
        matrix_nnz,
        params,
        stats,
    )
}

/// Recover failed final DC solve with pseudo-transient continuation.
///
/// Adds an adaptive diagonal damping term `alpha * I` (node rows only) and
/// anchors the RHS as `b + alpha * x_prev`, then progressively reduces alpha
/// toward 0. This creates a homotopy path that is often more robust than a
/// direct final jump to true GMIN=0.
#[allow(clippy::too_many_arguments)]
fn run_pseudo_transient_continuation(
    backend: &crate::solver::backend::WgpuBackend,
    ds_backend: &crate::solver::ds_backend::WgpuDsBackend,
    system: &MnaSystem,
    col_indices_u32: &[u32],
    row_ptrs_u32: &[u32],
    matrix_nnz: usize,
    diag_positions: &[usize],
    seed_x: Vec<f32>,
    mut stats: Option<&mut Stats>,
) -> Result<Vec<f64>> {
    use crate::solver::backend::SolverBackend;
    use crate::solver::newton::{NewtonLinearMode, NewtonParams};

    let n_nodes = system.node_names.len();
    let damped_b_buf = backend.new_buffer(system.size);
    let mut x_prev = seed_x;

    // Adaptive alpha schedule: try 100x drops on success, 10x subdivisions on failure.
    let mut last_good_alpha = 1.0_f64;
    let mut target_alpha = 1e-1_f64;
    let mut subdivisions = 0_u32;

    while target_alpha > 1e-8 {
        let mut step_b_f64 = system.b_dc.clone();
        for i in 0..n_nodes {
            step_b_f64[i] += target_alpha * x_prev[i] as f64;
        }
        let step_b_f32: Vec<f32> = step_b_f64.iter().map(|&v| v as f32).collect();
        backend.upload_vec(&step_b_f32, &damped_b_buf);

        let params = NewtonParams {
            max_iterations: 20,
            bicgstab_max_iterations: 500,
            linear_mode: NewtonLinearMode::GpuBicgstab,
            initial_guess: Some(x_prev.clone()),
            ..NewtonParams::default()
        };

        let result = gmin_newton_step(
            backend,
            ds_backend,
            &damped_b_buf,
            &step_b_f64,
            system,
            col_indices_u32,
            row_ptrs_u32,
            matrix_nnz,
            diag_positions,
            target_alpha,
            &params,
            stats.as_deref_mut(),
        )
        .or_else(|gpu_err| {
            let mut cpu_params = params.clone();
            cpu_params.linear_mode = NewtonLinearMode::CpuSparseLu;
            match gmin_newton_step(
                backend,
                ds_backend,
                &damped_b_buf,
                &step_b_f64,
                system,
                col_indices_u32,
                row_ptrs_u32,
                matrix_nnz,
                diag_positions,
                target_alpha,
                &cpu_params,
                stats.as_deref_mut(),
            ) {
                Ok(x) => {
                    tracing::debug!(
                        ?gpu_err,
                        target_alpha,
                        "Pseudo-transient GPU step failed, CPU sparse-LU fallback succeeded"
                    );
                    Ok(x)
                }
                Err(cpu_err) => {
                    tracing::debug!(
                        ?gpu_err,
                        ?cpu_err,
                        target_alpha,
                        "Pseudo-transient step failed on both GPU and CPU fallback"
                    );
                    Err(cpu_err)
                }
            }
        });

        match result {
            Ok(x) => {
                x_prev = x.iter().map(|&v| v as f32).collect();
                last_good_alpha = target_alpha;
                subdivisions = 0;
                target_alpha *= 1e-2;
            }
            Err(e) => {
                subdivisions += 1;
                if subdivisions > 7 {
                    tracing::error!(
                        "Pseudo-transient continuation exhausted after {subdivisions} subdivisions"
                    );
                    return Err(e);
                }
                target_alpha = last_good_alpha * 10.0_f64.powi(-(subdivisions as i32));
                tracing::debug!(
                    ?e,
                    target_alpha,
                    subdivisions,
                    "Pseudo-transient step failed, subdividing"
                );
            }
        }
    }

    // Final true-DC solve (alpha = 0) from the latest damped solution.
    let base_b_f32: Vec<f32> = system.b_dc.iter().map(|&v| v as f32).collect();
    backend.upload_vec(&base_b_f32, &damped_b_buf);

    let final_params = NewtonParams {
        max_iterations: 25,
        bicgstab_max_iterations: 1500,
        initial_guess: Some(x_prev.clone()),
        ..NewtonParams::default()
    };

    gmin_newton_step(
        backend,
        ds_backend,
        &damped_b_buf,
        &system.b_dc,
        system,
        col_indices_u32,
        row_ptrs_u32,
        matrix_nnz,
        diag_positions,
        0.0,
        &final_params,
        stats.as_deref_mut(),
    )
    .or_else(|gpu_err| {
        let mut cpu_params = final_params.clone();
        cpu_params.linear_mode = NewtonLinearMode::CpuSparseLu;
        match gmin_newton_step(
            backend,
            ds_backend,
            &damped_b_buf,
            &system.b_dc,
            system,
            col_indices_u32,
            row_ptrs_u32,
            matrix_nnz,
            diag_positions,
            0.0,
            &cpu_params,
            stats.as_deref_mut(),
        ) {
            Ok(x) => {
                tracing::debug!(
                    ?gpu_err,
                    "Final pseudo-transient GPU step failed, CPU sparse-LU fallback succeeded"
                );
                Ok(x)
            }
            Err(cpu_err) => {
                tracing::debug!(
                    ?gpu_err,
                    ?cpu_err,
                    "Final pseudo-transient step failed on both GPU and CPU fallback"
                );
                Err(cpu_err)
            }
        }
    })
}

/// Source stepping: gradually ramp all independent sources from 0 to full value.
///
/// At each step, the RHS vector is scaled by `alpha` (0 → 1), effectively
/// reducing all voltage and current sources proportionally. This keeps the
/// circuit in a well-conditioned regime at each step:
/// - At low alpha: MOSFETs are in cutoff or weak inversion with small voltages
/// - Each step uses the previous converged solution as initial guess
/// - The Jacobian stays well-conditioned because operating points change gradually
///
/// This is the standard SPICE fallback when GMIN stepping fails, and is
/// particularly effective for long MOSFET chains where the gain amplifies
/// conditioning problems.
#[allow(clippy::too_many_arguments)]
fn run_source_stepping(
    backend: &crate::solver::backend::WgpuBackend,
    ds_backend: &crate::solver::ds_backend::WgpuDsBackend,
    system: &MnaSystem,
    col_indices_u32: &[u32],
    row_ptrs_u32: &[u32],
    matrix_nnz: usize,
    diag_positions: &[usize],
    mut stats: Option<&mut Stats>,
) -> Result<Vec<f64>> {
    use crate::solver::backend::SolverBackend;
    use crate::solver::newton::{NewtonLinearMode, NewtonParams};

    tracing::info!("Starting source stepping");

    let scaled_b_buf = backend.new_buffer(system.size);
    let mut x_prev: Option<Vec<f32>> = None;

    // Adaptive source stepping: try large jumps, subdivide on failure.
    let mut last_good_alpha = 0.0_f64;
    let mut target_alpha = 0.1_f64;
    let mut subdivisions = 0_u32;

    while target_alpha <= 1.0 {
        // Scale the RHS vector by alpha
        let b_scaled: Vec<f64> = system.b_dc.iter().map(|&v| target_alpha * v).collect();
        let b_scaled_f32: Vec<f32> = b_scaled.iter().map(|&v| v as f32).collect();
        backend.upload_vec(&b_scaled_f32, &scaled_b_buf);

        let params = NewtonParams {
            max_iterations: 50,
            bicgstab_max_iterations: 500,
            linear_mode: NewtonLinearMode::GpuBicgstab,
            initial_guess: x_prev.clone(),
            ..NewtonParams::default()
        };

        tracing::debug!(target_alpha, "Source stepping");

        let result = gmin_newton_step(
            backend,
            ds_backend,
            &scaled_b_buf,
            &b_scaled,
            system,
            col_indices_u32,
            row_ptrs_u32,
            matrix_nnz,
            diag_positions,
            0.0,
            &params,
            stats.as_deref_mut(),
        )
        .or_else(|gpu_err| {
            let mut cpu_params = params.clone();
            cpu_params.linear_mode = NewtonLinearMode::CpuSparseLu;
            match gmin_newton_step(
                backend,
                ds_backend,
                &scaled_b_buf,
                &b_scaled,
                system,
                col_indices_u32,
                row_ptrs_u32,
                matrix_nnz,
                diag_positions,
                0.0,
                &cpu_params,
                stats.as_deref_mut(),
            ) {
                Ok(x) => {
                    tracing::debug!(
                        ?gpu_err,
                        target_alpha,
                        "Source stepping GPU failed, CPU sparse-LU fallback succeeded"
                    );
                    Ok(x)
                }
                Err(cpu_err) => {
                    tracing::debug!(
                        ?gpu_err,
                        ?cpu_err,
                        target_alpha,
                        "Source stepping failed on both GPU and CPU fallback"
                    );
                    Err(cpu_err)
                }
            }
        });

        match result {
            Ok(x) => {
                x_prev = Some(x.iter().map(|&v| v as f32).collect());
                last_good_alpha = target_alpha;
                subdivisions = 0;
                if (target_alpha - 1.0).abs() < 1e-12 {
                    // We've reached alpha = 1.0, done!
                    tracing::info!("Source stepping converged");
                    return Ok(x);
                }
                // Adaptive: try a larger step next
                let step = (target_alpha - last_good_alpha).max(0.1);
                target_alpha = (target_alpha + step * 2.0).min(1.0);
            }
            Err(e) => {
                subdivisions += 1;
                if subdivisions > 20 {
                    tracing::error!(
                        "Source stepping exhausted after {subdivisions} subdivisions at alpha={target_alpha}"
                    );
                    return Err(e);
                }
                // Subdivide: try halfway between last good and target
                target_alpha = last_good_alpha + (target_alpha - last_good_alpha) / 2.0;
                if target_alpha - last_good_alpha < 1e-6 {
                    tracing::error!(
                        "Source stepping stalled at alpha={last_good_alpha}"
                    );
                    return Err(e);
                }
                tracing::debug!(
                    ?e,
                    target_alpha,
                    subdivisions,
                    "Source stepping failed, subdividing"
                );
            }
        }
    }

    // Should not reach here since alpha=1.0 is handled in the loop
    Err(crate::error::OhmnivoreError::Solve(
        "Source stepping completed without reaching alpha=1.0".into(),
    ))
}

/// Map a solution vector to named node voltages and branch currents.
fn map_solution(system: &MnaSystem, x: &[f64]) -> Result<DcResult> {
    let n_nodes = system.node_names.len();

    let node_voltages = system
        .node_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), x[i]))
        .collect();

    let branch_currents = system
        .branch_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), x[n_nodes + i]))
        .collect();

    Ok(DcResult {
        node_voltages,
        branch_currents,
    })
}
