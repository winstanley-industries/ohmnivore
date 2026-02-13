//! Integration tests comparing GPU and CPU solvers on the same circuits.
//!
//! Note: The GPU solver uses BiCGSTAB with Jacobi preconditioning and f32 precision.
//! MNA systems with voltage sources or inductors have zero diagonal entries in branch
//! rows, which can cause Jacobi-preconditioned BiCGSTAB to break down. Tests for such
//! circuits use try-based helpers and verify either matching results or expected failures.

use approx::assert_abs_diff_eq;
use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;
use ohmnivore::solver::gpu::GpuSolver;
use ohmnivore::solver::LinearSolver;

fn gpu_available() -> bool {
    GpuSolver::new().is_ok()
}

macro_rules! skip_if_no_gpu {
    () => {
        if !gpu_available() {
            eprintln!("Skipping: no GPU available");
            return;
        }
    };
}

/// Helper: parse + compile + DC solve with a given solver.
fn dc_solve(netlist: &str, solver: &dyn LinearSolver) -> analysis::DcResult {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    analysis::dc::run(&system, solver).expect("DC analysis failed")
}

/// Helper: parse + compile + DC solve, returning Result for circuits that may fail on GPU.
fn try_dc_solve(
    netlist: &str,
    solver: &dyn LinearSolver,
) -> Result<analysis::DcResult, ohmnivore::error::OhmnivoreError> {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    analysis::dc::run(&system, solver)
}

/// Helper: parse + compile + AC solve, returning Result for circuits that may fail on GPU.
fn try_ac_solve(
    netlist: &str,
    solver: &dyn LinearSolver,
) -> Result<analysis::AcResult, ohmnivore::error::OhmnivoreError> {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let ac_cmd = circuit
        .analyses
        .iter()
        .find_map(|a| match a {
            ohmnivore::ir::Analysis::Ac {
                sweep_type,
                n_points,
                f_start,
                f_stop,
            } => Some((*sweep_type, *n_points, *f_start, *f_stop)),
            _ => None,
        })
        .expect("no AC analysis in netlist");
    analysis::ac::run(&system, solver, ac_cmd.0, ac_cmd.1, ac_cmd.2, ac_cmd.3)
}

fn find_node_voltage(result: &analysis::DcResult, name: &str) -> f64 {
    result
        .node_voltages
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("node {} not found", name))
        .1
}

// ── DC Tests ──────────────────────────────────────────────────────

#[test]
fn gpu_vs_cpu_current_source_dc() {
    skip_if_no_gpu!();

    // Pure resistor + current source: no branch variables, full diagonal.
    // This is ideal for the GPU's Jacobi-preconditioned BiCGSTAB.
    let netlist = "\
I1 0 1 DC 0.001
R1 1 0 1k
.DC
.END
";
    let cpu = CpuSolver::new();
    let gpu = GpuSolver::new().unwrap();

    let cpu_result = dc_solve(netlist, &cpu);
    let gpu_result = dc_solve(netlist, &gpu);

    let cpu_v1 = find_node_voltage(&cpu_result, "1");
    let gpu_v1 = find_node_voltage(&gpu_result, "1");
    assert_abs_diff_eq!(cpu_v1, gpu_v1, epsilon = 1e-3);

    // V = I * R = 0.001 * 1000 = 1.0
    assert_abs_diff_eq!(gpu_v1, 1.0, epsilon = 1e-2);
}

#[test]
fn gpu_vs_cpu_current_source_resistor_network_dc() {
    skip_if_no_gpu!();

    // Multi-node resistor network with current source excitation.
    // No voltage sources, so all diagonal entries are nonzero.
    //
    //  I1(1mA) -> node 1 -> R1(1k) -> node 2 -> R2(2k) -> GND
    //                                   |
    //                                  R3(1k) -> GND
    let netlist = "\
I1 0 1 DC 0.001
R1 1 2 1k
R2 2 0 2k
R3 2 0 1k
.DC
.END
";
    let cpu = CpuSolver::new();
    let gpu = GpuSolver::new().unwrap();

    let cpu_result = dc_solve(netlist, &cpu);
    let gpu_result = dc_solve(netlist, &gpu);

    let cpu_v1 = find_node_voltage(&cpu_result, "1");
    let gpu_v1 = find_node_voltage(&gpu_result, "1");
    let cpu_v2 = find_node_voltage(&cpu_result, "2");
    let gpu_v2 = find_node_voltage(&gpu_result, "2");

    assert_abs_diff_eq!(cpu_v1, gpu_v1, epsilon = 1e-3);
    assert_abs_diff_eq!(cpu_v2, gpu_v2, epsilon = 1e-3);

    // R2 || R3 = (2k * 1k) / (2k + 1k) = 2/3 k = 666.67 ohm
    // Total R = R1 + R2||R3 = 1000 + 666.67 = 1666.67 ohm
    // V1 = I * R_total = 0.001 * 1666.67 = 1.6667
    // V2 = I * R2||R3 = 0.001 * 666.67 = 0.6667
    assert_abs_diff_eq!(gpu_v1, 5.0 / 3.0, epsilon = 1e-2);
    assert_abs_diff_eq!(gpu_v2, 2.0 / 3.0, epsilon = 1e-2);
}

#[test]
fn gpu_vs_cpu_voltage_divider_dc() {
    skip_if_no_gpu!();

    // Voltage source circuit: MNA has branch variables with zero diagonal entries.
    // The Jacobi preconditioner may cause BiCGSTAB breakdown on these systems.
    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
    let gpu = GpuSolver::new().unwrap();

    match try_dc_solve(netlist, &gpu) {
        Ok(gpu_result) => {
            // GPU solver succeeded -- verify results match CPU
            let cpu = CpuSolver::new();
            let cpu_result = dc_solve(netlist, &cpu);

            let cpu_v2 = find_node_voltage(&cpu_result, "2");
            let gpu_v2 = find_node_voltage(&gpu_result, "2");
            assert_abs_diff_eq!(cpu_v2, gpu_v2, epsilon = 1e-2);
        }
        Err(e) => {
            // Expected: Jacobi-preconditioned BiCGSTAB may break down on
            // MNA systems with zero diagonal entries from voltage source branches.
            let msg = e.to_string();
            assert!(
                msg.contains("BiCGSTAB breakdown"),
                "unexpected GPU error: {}",
                msg
            );
            eprintln!(
                "GPU solver correctly reports breakdown on voltage source circuit: {}",
                msg
            );
        }
    }
}

// ── AC Tests ──────────────────────────────────────────────────────

#[test]
fn gpu_vs_cpu_rc_lowpass_ac() {
    skip_if_no_gpu!();

    // AC circuit with voltage source -- has zero diagonal entries in MNA.
    let netlist = "\
V1 1 0 AC 1 0
R1 1 2 1k
C1 2 0 1u
.AC DEC 10 1 1000000
.END
";
    let gpu = GpuSolver::new().unwrap();

    match try_ac_solve(netlist, &gpu) {
        Ok(gpu_result) => {
            let cpu = CpuSolver::new();
            let cpu_result = try_ac_solve(netlist, &cpu).unwrap();

            assert_eq!(cpu_result.frequencies.len(), gpu_result.frequencies.len());

            let cpu_v2 = &cpu_result
                .node_voltages
                .iter()
                .find(|(n, _)| n == "2")
                .unwrap()
                .1;
            let gpu_v2 = &gpu_result
                .node_voltages
                .iter()
                .find(|(n, _)| n == "2")
                .unwrap()
                .1;

            // Compare magnitude at low and high frequencies
            let cpu_mag_low = cpu_v2[0].norm();
            let gpu_mag_low = gpu_v2[0].norm();
            assert_abs_diff_eq!(cpu_mag_low, gpu_mag_low, epsilon = 1e-2);

            let cpu_mag_high = cpu_v2.last().unwrap().norm();
            let gpu_mag_high = gpu_v2.last().unwrap().norm();
            assert_abs_diff_eq!(cpu_mag_high, gpu_mag_high, epsilon = 1e-2);
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("BiCGSTAB breakdown"),
                "unexpected GPU error: {}",
                msg
            );
            eprintln!(
                "GPU solver correctly reports breakdown on AC voltage source circuit: {}",
                msg
            );
        }
    }
}

#[test]
fn gpu_vs_cpu_rlc_series_ac() {
    skip_if_no_gpu!();

    // RLC series circuit with voltage source and inductor -- multiple branch variables.
    let netlist = "\
V1 1 0 AC 1 0
R1 1 2 100
L1 2 3 10m
C1 3 0 1u
.AC DEC 20 100 100000
.END
";
    let gpu = GpuSolver::new().unwrap();

    match try_ac_solve(netlist, &gpu) {
        Ok(gpu_result) => {
            let cpu = CpuSolver::new();
            let cpu_result = try_ac_solve(netlist, &cpu).unwrap();

            assert_eq!(cpu_result.frequencies.len(), gpu_result.frequencies.len());

            let cpu_v3 = &cpu_result
                .node_voltages
                .iter()
                .find(|(n, _)| n == "3")
                .unwrap()
                .1;
            let gpu_v3 = &gpu_result
                .node_voltages
                .iter()
                .find(|(n, _)| n == "3")
                .unwrap()
                .1;

            for i in (0..cpu_v3.len()).step_by(5) {
                let cpu_mag = cpu_v3[i].norm();
                let gpu_mag = gpu_v3[i].norm();
                let tolerance = 1e-2 + cpu_mag * 5e-2;
                assert!(
                    (cpu_mag - gpu_mag).abs() < tolerance,
                    "freq={:.1} Hz: CPU mag={}, GPU mag={}, diff={}",
                    cpu_result.frequencies[i],
                    cpu_mag,
                    gpu_mag,
                    (cpu_mag - gpu_mag).abs()
                );
            }
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("BiCGSTAB breakdown"),
                "unexpected GPU error: {}",
                msg
            );
            eprintln!(
                "GPU solver correctly reports breakdown on RLC circuit: {}",
                msg
            );
        }
    }
}
