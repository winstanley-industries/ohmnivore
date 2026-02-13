//! AC frequency sweep analysis.
//!
//! For each frequency point:
//!   1. Form A = G + jωC where ω = 2πf
//!   2. Solve Ax = b_ac
//!   3. Collect complex node voltages and branch currents
//!
//! Frequency points are generated according to sweep type:
//!   - DEC: logarithmic, n_points per decade
//!   - OCT: logarithmic, n_points per octave
//!   - LIN: linear, n_points total

use super::AcResult;
use crate::compiler::MnaSystem;
use crate::error::Result;
use crate::ir::AcSweepType;
use crate::solver::LinearSolver;
use crate::stats::Stats;
use num_complex::Complex64;

/// Run AC frequency sweep analysis.
///
/// Requires a prior DC operating point (for bias, though for linear circuits
/// the AC analysis is independent of the DC point).
pub fn run(
    system: &MnaSystem,
    solver: &dyn LinearSolver,
    sweep_type: AcSweepType,
    n_points: usize,
    f_start: f64,
    f_stop: f64,
    mut stats: Option<&mut Stats>,
) -> Result<AcResult> {
    use crate::sparse::form_complex_matrix;

    // Generate frequency points
    let frequencies = generate_frequencies(sweep_type, n_points, f_start, f_stop);
    let _span = tracing::info_span!("ac_analysis", n_points = frequencies.len()).entered();
    let n_freqs = frequencies.len();
    let n_nodes = system.node_names.len();

    // Pre-allocate per-node and per-branch result vectors
    let mut node_voltages: Vec<(String, Vec<Complex64>)> = system
        .node_names
        .iter()
        .map(|name| (name.clone(), Vec::with_capacity(n_freqs)))
        .collect();

    let mut branch_currents: Vec<(String, Vec<Complex64>)> = system
        .branch_names
        .iter()
        .map(|name| (name.clone(), Vec::with_capacity(n_freqs)))
        .collect();

    // Sweep over each frequency
    for &f in &frequencies {
        let omega = 2.0 * std::f64::consts::PI * f;
        let a = form_complex_matrix(&system.g, &system.c, omega);
        let x = solver.solve_complex(&a, &system.b_ac)?;
        if let Some(ref mut s) = stats { s.linear_solves += 1; }

        for (i, (_, voltages)) in node_voltages.iter_mut().enumerate() {
            voltages.push(x[i]);
        }
        for (i, (_, currents)) in branch_currents.iter_mut().enumerate() {
            currents.push(x[n_nodes + i]);
        }
    }

    Ok(AcResult {
        frequencies,
        node_voltages,
        branch_currents,
    })
}

/// Generate frequency points for the given sweep type.
fn generate_frequencies(
    sweep_type: AcSweepType,
    n_points: usize,
    f_start: f64,
    f_stop: f64,
) -> Vec<f64> {
    match sweep_type {
        AcSweepType::Dec => {
            let decades = (f_stop / f_start).log10();
            let total = (n_points as f64 * decades).ceil() as usize;
            (0..total)
                .map(|i| f_start * 10.0_f64.powf(i as f64 / n_points as f64))
                .collect()
        }
        AcSweepType::Oct => {
            let octaves = (f_stop / f_start).log2();
            let total = (n_points as f64 * octaves).ceil() as usize;
            (0..total)
                .map(|i| f_start * 2.0_f64.powf(i as f64 / n_points as f64))
                .collect()
        }
        AcSweepType::Lin => {
            if n_points <= 1 {
                return vec![f_start];
            }
            let step = (f_stop - f_start) / (n_points - 1) as f64;
            (0..n_points).map(|i| f_start + step * i as f64).collect()
        }
    }
}
