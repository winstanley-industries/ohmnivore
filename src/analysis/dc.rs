//! DC operating point analysis.
//!
//! Solves Gx = b_dc for the DC operating point.
//! The solution vector x contains node voltages followed by branch currents.
//!
//! For linear circuits (no diodes), uses the standard linear solver path.
//! For nonlinear circuits (with diodes), uses GPU-accelerated Newton-Raphson.

use crate::compiler::MnaSystem;
use crate::error::Result;
use crate::solver::LinearSolver;
use super::DcResult;

/// Run DC operating point analysis.
///
/// 1. Check whether the circuit has nonlinear elements (diode descriptors).
/// 2. If linear: solve Gx = b_dc with the provided linear solver.
/// 3. If nonlinear: use the GPU Newton-Raphson solver.
/// 4. Map solution indices back to node/branch names.
pub fn run(system: &MnaSystem, solver: &dyn LinearSolver) -> Result<DcResult> {
    let x = if !system.diode_descriptors.is_empty() {
        run_nonlinear(system)?
    } else {
        solver.solve_real(&system.g, &system.b_dc)?
    };

    map_solution(system, &x)
}

/// Run the nonlinear DC path using GPU Newton-Raphson iteration.
fn run_nonlinear(system: &MnaSystem) -> Result<Vec<f64>> {
    use crate::solver::backend::{SolverBackend, WgpuBackend};
    use crate::solver::newton::{newton_solve, NewtonParams};

    let backend = WgpuBackend::new()?;

    // Convert base G matrix to f32 CSR for GPU
    let values_f32: Vec<f32> = system.g.values.iter().map(|&v| v as f32).collect();
    let col_indices_u32: Vec<u32> = system.g.col_indices.iter().map(|&c| c as u32).collect();
    let row_ptrs_u32: Vec<u32> = system.g.row_pointers.iter().map(|&r| r as u32).collect();
    let b_dc_f32: Vec<f32> = system.b_dc.iter().map(|&v| v as f32).collect();

    // Upload base RHS vector to GPU
    let base_b_buf = backend.new_buffer(system.size);
    backend.upload_vec(&b_dc_f32, &base_b_buf);

    let matrix_nnz = system.g.values.len();

    newton_solve(
        &backend,
        &base_b_buf,
        &values_f32,
        &col_indices_u32,
        &row_ptrs_u32,
        &system.diode_descriptors,
        system.size,
        matrix_nnz,
        &NewtonParams::default(),
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
