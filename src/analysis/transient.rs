//! Transient (time-domain) analysis engine.
//!
//! Solves G*x(t) + C*dx(t)/dt = b(t) using Backward Euler (BE) and
//! Trapezoidal (TRAP) integration with adaptive timestep control.
//!
//! First step uses BE (unconditionally stable). Subsequent steps use TRAP
//! (second-order accurate). Adaptive control compares BE and TRAP solutions
//! to estimate local truncation error.

use super::transient_source;
use super::TranResult;
use crate::compiler::MnaSystem;
use crate::error::{OhmnivoreError, Result};
use crate::ir::{Component, TransientFunc};
use crate::solver::LinearSolver;
use crate::sparse::CsrMatrix;
use crate::stats::Stats;

/// Run transient analysis.
///
/// Solves the circuit from t=0 to t=tstop. Records output starting at t=tstart.
/// If `uic` is true, starts from zero initial conditions; otherwise computes
/// a DC operating point first.
#[allow(clippy::too_many_arguments)]
pub fn run(
    system: &MnaSystem,
    solver: &dyn LinearSolver,
    tstep: f64,
    tstop: f64,
    tstart: f64,
    uic: bool,
    components: &[Component],
    mut stats: Option<&mut Stats>,
) -> Result<TranResult> {
    let _span = tracing::info_span!("transient_analysis", tstop, tstep).entered();
    let n = system.size;
    let h_min = tstep / 10_000.0;
    let h_max = tstep;
    let lte_tol = 1e-3;
    let max_consecutive_failures = 10;

    // Initial conditions
    let mut x: Vec<f64> = if uic {
        compute_uic_initial_conditions(system, solver, components)?
    } else {
        let dc_result = super::dc::run(system, solver, stats.as_deref_mut())?;
        let mut x0 = vec![0.0; n];
        let n_nodes = system.node_names.len();
        for (i, (_, v)) in dc_result.node_voltages.iter().enumerate() {
            x0[i] = *v;
        }
        for (i, (_, c)) in dc_result.branch_currents.iter().enumerate() {
            x0[n_nodes + i] = *c;
        }
        x0
    };

    // Collect source info for b(t) evaluation
    let source_info = collect_source_info(system, components);

    // Output storage
    let mut times = Vec::new();
    let mut node_waveforms: Vec<Vec<f64>> = vec![Vec::new(); system.node_names.len()];
    let mut branch_waveforms: Vec<Vec<f64>> = vec![Vec::new(); system.branch_names.len()];

    // Record initial conditions if tstart == 0
    if tstart <= 0.0 {
        record_output(
            &x,
            system,
            &mut times,
            &mut node_waveforms,
            &mut branch_waveforms,
            0.0,
        );
    }

    let mut t = 0.0;
    let mut h = tstep.min(h_max);
    let mut b_prev = evaluate_b_at_time(&system.b_dc, &source_info, 0.0);
    let mut use_be = true; // first step always BE
    let mut consecutive_failures = 0;

    while t < tstop {
        // Clamp h so we don't overshoot tstop
        if t + h > tstop {
            h = tstop - t;
        }
        if h < h_min * 0.5 {
            break;
        }

        let t_next = t + h;
        let b_next = evaluate_b_at_time(&system.b_dc, &source_info, t_next);

        // Solve with TRAP (or BE if first step / fallback)
        let (x_trap, x_be) = if use_be {
            // BE only
            let a_be = form_companion_matrix(&system.g, &system.c, h, 1.0);
            let b_eff_be = form_companion_rhs_be(&system.c, &x, &b_next, h);
            let x_be = solver.solve_real(&a_be, &b_eff_be)?;
            if let Some(ref mut s) = stats { s.linear_solves += 1; }
            (x_be.clone(), x_be)
        } else {
            // Solve both TRAP and BE for LTE estimation
            let a_trap = form_companion_matrix(&system.g, &system.c, h, 2.0);
            let b_eff_trap = form_companion_rhs_trap(&system.g, &system.c, &x, &b_next, &b_prev, h);
            let x_trap = solver.solve_real(&a_trap, &b_eff_trap)?;
            if let Some(ref mut s) = stats { s.linear_solves += 1; }

            let a_be = form_companion_matrix(&system.g, &system.c, h, 1.0);
            let b_eff_be = form_companion_rhs_be(&system.c, &x, &b_next, h);
            let x_be = solver.solve_real(&a_be, &b_eff_be)?;
            if let Some(ref mut s) = stats { s.linear_solves += 1; }

            (x_trap, x_be)
        };

        // LTE estimation (skip on first step since BE-only)
        if !use_be {
            let lte = compute_lte(&x_trap, &x_be);

            if lte > lte_tol {
                // Reject step: reduce h and retry
                let h_new = h * (0.9 * (lte_tol / lte).sqrt()).clamp(0.5, 2.0);
                h = h_new.max(h_min);
                consecutive_failures += 1;

                if consecutive_failures >= max_consecutive_failures && h <= h_min * 1.01 {
                    return Err(OhmnivoreError::Timestep(format!(
                        "simulation failed: {} consecutive failures at minimum timestep h_min={:.2e} near t={:.2e}",
                        consecutive_failures, h_min, t
                    )));
                }

                if let Some(ref mut s) = stats { s.timesteps_rejected += 1; }
                // Switch to BE for recovery
                use_be = true;
                continue;
            }

            // Step accepted: compute next h
            let h_new = h * (0.9 * (lte_tol / lte).sqrt()).clamp(0.5, 2.0);
            h = h_new.clamp(h_min, h_max);
        } else {
            // After first BE step, switch to TRAP
            use_be = false;
        }

        // Accept step
        consecutive_failures = 0;
        if let Some(ref mut s) = stats { s.timesteps_accepted += 1; }
        x = x_trap;
        b_prev = b_next;
        t = t_next;

        // Record output if past tstart
        if t >= tstart {
            record_output(
                &x,
                system,
                &mut times,
                &mut node_waveforms,
                &mut branch_waveforms,
                t,
            );
        }
    }

    // Build result
    let node_voltages = system
        .node_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), node_waveforms[i].clone()))
        .collect();
    let branch_currents = system
        .branch_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), branch_waveforms[i].clone()))
        .collect();

    Ok(TranResult {
        times,
        node_voltages,
        branch_currents,
    })
}

/// Source descriptor for time-varying RHS evaluation.
struct SourceEntry {
    /// Index into b vector where this source contributes.
    b_indices: Vec<(usize, f64)>, // (index, sign)
    /// Transient function (if any).
    tran_func: Option<TransientFunc>,
    /// DC value (if any).
    dc_value: Option<f64>,
}

/// Collect source info needed for b(t) evaluation.
fn collect_source_info(system: &MnaSystem, components: &[Component]) -> Vec<SourceEntry> {
    let n_nodes = system.node_names.len();
    let mut entries = Vec::new();

    // Build a name -> node index lookup
    let node_idx = |name: &str| -> Option<usize> {
        if name == "0" || name.eq_ignore_ascii_case("GND") {
            None
        } else {
            system.node_names.iter().position(|n| n == name)
        }
    };

    let mut branch_idx = 0usize;
    for comp in components {
        match comp {
            Component::VSource { dc, tran, .. } => {
                let bk = n_nodes + branch_idx;
                entries.push(SourceEntry {
                    b_indices: vec![(bk, 1.0)],
                    tran_func: tran.clone(),
                    dc_value: *dc,
                });
                branch_idx += 1;
            }
            Component::ISource {
                nodes, dc, tran, ..
            } => {
                let ni = node_idx(&nodes.0);
                let nj = node_idx(&nodes.1);
                let mut indices = Vec::new();
                if let Some(i) = ni {
                    indices.push((i, -1.0)); // b(n+) -= I
                }
                if let Some(j) = nj {
                    indices.push((j, 1.0)); // b(n-) += I
                }
                entries.push(SourceEntry {
                    b_indices: indices,
                    tran_func: tran.clone(),
                    dc_value: *dc,
                });
            }
            Component::Inductor { .. } => {
                branch_idx += 1; // inductors have branch variables but no time-varying source
            }
            _ => {}
        }
    }
    entries
}

/// Evaluate the RHS vector b at time t, accounting for time-varying sources.
fn evaluate_b_at_time(b_dc: &[f64], sources: &[SourceEntry], t: f64) -> Vec<f64> {
    let mut b = vec![0.0; b_dc.len()];

    for source in sources {
        let value = if let Some(ref func) = source.tran_func {
            transient_source::evaluate(func, t)
        } else {
            source.dc_value.unwrap_or(0.0)
        };

        for &(idx, sign) in &source.b_indices {
            b[idx] += sign * value;
        }
    }
    b
}

/// Form companion matrix: A = G + alpha * C / h
///
/// alpha = 1.0 for BE, alpha = 2.0 for TRAP
fn form_companion_matrix(
    g: &CsrMatrix<f64>,
    c: &CsrMatrix<f64>,
    h: f64,
    alpha: f64,
) -> CsrMatrix<f64> {
    let factor = alpha / h;

    // G and C share sparsity pattern (compiler zero-fills G for C entries).
    // Build A by adding scaled C values to G values at matching positions.
    let mut a_values = g.values.clone();

    for row in 0..c.nrows {
        let c_start = c.row_pointers[row];
        let c_end = c.row_pointers[row + 1];

        for c_idx in c_start..c_end {
            let col = c.col_indices[c_idx];
            let c_val = c.values[c_idx];

            if c_val == 0.0 {
                continue;
            }

            // Find the matching position in G's CSR
            if let Some(g_idx) = g.value_index(row, col) {
                a_values[g_idx] += factor * c_val;
            }
        }
    }

    CsrMatrix {
        nrows: g.nrows,
        ncols: g.ncols,
        values: a_values,
        col_indices: g.col_indices.clone(),
        row_pointers: g.row_pointers.clone(),
    }
}

/// Form BE companion RHS: b_eff = b(t_n) + (C/h) * x_{n-1}
fn form_companion_rhs_be(c: &CsrMatrix<f64>, x_prev: &[f64], b_t: &[f64], h: f64) -> Vec<f64> {
    let c_x = c.spmv(x_prev);
    let factor = 1.0 / h;
    b_t.iter()
        .zip(c_x.iter())
        .map(|(&bt, &cx)| bt + factor * cx)
        .collect()
}

/// Form TRAP companion RHS: b_eff = b(t_n) + (2C/h - G) * x_{n-1} + b(t_{n-1})
fn form_companion_rhs_trap(
    g: &CsrMatrix<f64>,
    c: &CsrMatrix<f64>,
    x_prev: &[f64],
    b_t: &[f64],
    b_prev: &[f64],
    h: f64,
) -> Vec<f64> {
    let c_x = c.spmv(x_prev);
    let g_x = g.spmv(x_prev);
    let factor = 2.0 / h;

    b_t.iter()
        .enumerate()
        .map(|(i, &bt)| bt + factor * c_x[i] - g_x[i] + b_prev[i])
        .collect()
}

/// Compute LTE = (2/3) * ||x_trap - x_be||_inf
fn compute_lte(x_trap: &[f64], x_be: &[f64]) -> f64 {
    let max_diff = x_trap
        .iter()
        .zip(x_be.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    (2.0 / 3.0) * max_diff
}

/// Compute consistent initial conditions for UIC (Use Initial Conditions).
///
/// Reactive elements are constrained to their IC values (default 0):
/// - **Inductors** (IC=0 → I_L=0): Replace the branch equation V_i - V_j = 0
///   (short circuit) with I_L = 0 (open circuit).
/// - **Capacitors** (IC=0 → V_C=0): Replace the KCL equation at a capacitor
///   node with V(i) - V(j) = 0 (short circuit), fixing the capacitor voltage.
///
/// The remaining node voltages and branch currents are solved for consistency
/// with these constraints and the source values.
fn compute_uic_initial_conditions(
    system: &MnaSystem,
    solver: &dyn LinearSolver,
    components: &[Component],
) -> Result<Vec<f64>> {
    let n_nodes = system.node_names.len();

    // Build node name → matrix index lookup
    let node_idx = |name: &str| -> Option<usize> {
        if name == "0" || name.eq_ignore_ascii_case("GND") {
            None
        } else {
            system.node_names.iter().position(|n| n == name)
        }
    };

    // Clone G matrix and b_dc for modification
    let mut g_uic = system.g.clone();
    let mut b_uic = system.b_dc.clone();

    // Track which node rows have been replaced by IC constraints
    let mut constrained = vec![false; n_nodes];

    // Zero out all entries in a given row of the CSR matrix
    let zero_row = |g: &mut CsrMatrix<f64>, row: usize| {
        let start = g.row_pointers[row];
        let end = g.row_pointers[row + 1];
        for idx in start..end {
            g.values[idx] = 0.0;
        }
    };

    // --- Inductor ICs: replace branch equation with I_L = 0 ---
    let mut branch_idx = 0usize;
    for comp in components {
        match comp {
            Component::Inductor { .. } => {
                let bk = n_nodes + branch_idx;
                zero_row(&mut g_uic, bk);

                // Set diagonal (bk, bk) = 1.0 → equation becomes I_L = 0
                if let Some(diag_idx) = g_uic.value_index(bk, bk) {
                    g_uic.values[diag_idx] = 1.0;
                }
                b_uic[bk] = 0.0;

                branch_idx += 1;
            }
            Component::VSource { .. } => {
                branch_idx += 1;
            }
            _ => {}
        }
    }

    // --- Capacitor ICs: constrain V(i) - V(j) = 0 ---
    for comp in components {
        if let Component::Capacitor { nodes, .. } = comp {
            let ni = node_idx(&nodes.0);
            let nj = node_idx(&nodes.1);

            match (ni, nj) {
                (Some(i), None) if !constrained[i] => {
                    // Capacitor to ground: V(i) = 0
                    zero_row(&mut g_uic, i);
                    if let Some(diag) = g_uic.value_index(i, i) {
                        g_uic.values[diag] = 1.0;
                    }
                    b_uic[i] = 0.0;
                    constrained[i] = true;
                }
                (None, Some(j)) if !constrained[j] => {
                    // Capacitor from ground: V(j) = 0
                    zero_row(&mut g_uic, j);
                    if let Some(diag) = g_uic.value_index(j, j) {
                        g_uic.values[diag] = 1.0;
                    }
                    b_uic[j] = 0.0;
                    constrained[j] = true;
                }
                (Some(i), Some(j)) => {
                    // Capacitor between two non-ground nodes: V(i) - V(j) = 0
                    if !constrained[i] {
                        zero_row(&mut g_uic, i);
                        if let Some(diag) = g_uic.value_index(i, i) {
                            g_uic.values[diag] = 1.0;
                        }
                        if let Some(off) = g_uic.value_index(i, j) {
                            g_uic.values[off] = -1.0;
                        }
                        b_uic[i] = 0.0;
                        constrained[i] = true;
                    } else if !constrained[j] {
                        zero_row(&mut g_uic, j);
                        if let Some(diag) = g_uic.value_index(j, j) {
                            g_uic.values[diag] = 1.0;
                        }
                        if let Some(off) = g_uic.value_index(j, i) {
                            g_uic.values[off] = -1.0;
                        }
                        b_uic[j] = 0.0;
                        constrained[j] = true;
                    }
                }
                _ => {}
            }
        }
    }

    solver.solve_real(&g_uic, &b_uic)
}

/// Record the current solution into the output arrays.
fn record_output(
    x: &[f64],
    system: &MnaSystem,
    times: &mut Vec<f64>,
    node_waveforms: &mut [Vec<f64>],
    branch_waveforms: &mut [Vec<f64>],
    t: f64,
) {
    times.push(t);
    let n_nodes = system.node_names.len();
    for i in 0..n_nodes {
        node_waveforms[i].push(x[i]);
    }
    for i in 0..system.branch_names.len() {
        branch_waveforms[i].push(x[n_nodes + i]);
    }
}
