//! DC operating point analysis.
//!
//! Solves Gx = b_dc for the DC operating point.
//! The solution vector x contains node voltages followed by branch currents.

use crate::compiler::MnaSystem;
use crate::error::Result;
use crate::solver::LinearSolver;
use super::DcResult;

/// Run DC operating point analysis.
///
/// 1. Extract G matrix and b_dc vector from MnaSystem.
/// 2. Solve Gx = b_dc using the provided solver.
/// 3. Map solution indices back to node/branch names.
pub fn run(system: &MnaSystem, solver: &dyn LinearSolver) -> Result<DcResult> {
    let x = solver.solve_real(&system.g, &system.b_dc)?;

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
