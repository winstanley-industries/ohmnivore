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
            tracing::info!(?e, "Direct Newton failed, falling back to GMIN stepping");
        }
    }

    // GMIN stepping: progressively reduce extra conductance to ground.
    // Large initial GMIN regularizes the Jacobian so BiCGSTAB can solve the
    // linear subproblem. Each subsequent step uses the previous converged
    // solution as initial guess.
    //
    // Adaptive schedule: try a 1000x jump; if it fails, subdivide with 10x jumps.
    // Intermediate steps use limited Newton iterations to fail fast.
    let diag_positions = find_node_diagonal_positions(system);

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
            // Fallback path: try CPU direct solve for this continuation step
            // before we subdivide further.
            let mut cpu_params = params.clone();
            cpu_params.linear_mode = NewtonLinearMode::CpuDirect;
            match gmin_newton_step(
                &backend,
                &ds_backend,
                &base_b_buf,
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
                        "GMIN GPU step failed, CPU direct fallback succeeded"
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

    gmin_newton_step(
        &backend,
        &ds_backend,
        &base_b_buf,
        system,
        &col_indices_u32,
        &row_ptrs_u32,
        matrix_nnz,
        &diag_positions,
        0.0,
        &final_params,
        stats.as_deref_mut(),
    )
}

/// Run a single Newton solve with modified GMIN.
#[allow(clippy::too_many_arguments)]
fn gmin_newton_step(
    backend: &crate::solver::backend::WgpuBackend,
    ds_backend: &crate::solver::ds_backend::WgpuDsBackend,
    base_b_buf: &crate::solver::backend::WgpuBuffer,
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
        &system.b_dc,
        &system.diode_descriptors,
        &system.bjt_descriptors,
        &system.mosfet_descriptors,
        system.size,
        matrix_nnz,
        params,
        stats,
    )
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
