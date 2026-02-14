//! End-to-end integration tests for ohmnivore circuit simulator.

use approx::assert_abs_diff_eq;
use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::ir::AcSweepType;
use ohmnivore::output;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;
use ohmnivore::solver::LinearSolver;

/// Helper: parse + compile + DC solve
fn dc_solve(netlist: &str) -> ohmnivore::analysis::DcResult {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let solver = CpuSolver::new();
    analysis::dc::run(&system, &solver, None).expect("DC analysis failed")
}

/// Helper: parse + compile + AC solve
fn ac_solve(netlist: &str) -> ohmnivore::analysis::AcResult {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let solver = CpuSolver::new();

    // Find the AC analysis command
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

    analysis::ac::run(
        &system, &solver, ac_cmd.0, ac_cmd.1, ac_cmd.2, ac_cmd.3, None,
    )
    .expect("AC analysis failed")
}

// ── DC Tests ──────────────────────────────────────────────────────

#[test]
fn test_voltage_divider_dc() {
    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
    let result = dc_solve(netlist);

    // Find node voltages by name
    let v1 = result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "1")
        .unwrap()
        .1;
    let v2 = result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "2")
        .unwrap()
        .1;

    assert_abs_diff_eq!(v1, 10.0, epsilon = 1e-6);
    assert_abs_diff_eq!(v2, 5.0, epsilon = 1e-6);

    // Current through V1: I = V/R_total = 10/2000 = 0.005A
    // Convention: current flows into the positive terminal of the source, so I(V1) is negative
    let i_v1 = result
        .branch_currents
        .iter()
        .find(|(n, _)| n == "V1")
        .unwrap()
        .1;
    assert_abs_diff_eq!(i_v1, -0.005, epsilon = 1e-9);
}

#[test]
fn test_current_source_dc() {
    let netlist = "\
I1 0 1 DC 0.001
R1 1 0 1k
.DC
.END
";
    let result = dc_solve(netlist);

    let v1 = result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "1")
        .unwrap()
        .1;
    // V = I * R = 0.001 * 1000 = 1.0
    assert_abs_diff_eq!(v1, 1.0, epsilon = 1e-9);
}

#[test]
fn test_voltage_divider_unequal() {
    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 2k
.DC
.END
";
    let result = dc_solve(netlist);

    let v2 = result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "2")
        .unwrap()
        .1;
    // V2 = 10 * 2k / (1k + 2k) = 20/3 ≈ 6.667
    assert_abs_diff_eq!(v2, 20.0 / 3.0, epsilon = 1e-6);
}

// ── AC Tests ──────────────────────────────────────────────────────

#[test]
fn test_rc_lowpass_ac_sweep() {
    let netlist = "\
V1 1 0 AC 1 0
R1 1 2 1k
C1 2 0 1u
.AC DEC 10 1 1000000
.END
";
    let result = ac_solve(netlist);

    // Find the V(2) response (output node of the RC divider)
    let (_, v2_data) = result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "2")
        .expect("node 2 not found");

    // At very low frequency (close to f_start=1Hz), |V(2)| should be ~1.0
    let v2_low = v2_data[0].norm();
    assert_abs_diff_eq!(v2_low, 1.0, epsilon = 0.01);

    // At cutoff frequency f_c = 1/(2*pi*R*C) = 1/(2*pi*1000*1e-6) ≈ 159.15 Hz
    // Find the frequency point closest to 159.15 Hz
    let f_cutoff = 1.0 / (2.0 * std::f64::consts::PI * 1000.0 * 1e-6);
    let cutoff_idx = result
        .frequencies
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((**a - f_cutoff).abs())
                .partial_cmp(&((**b - f_cutoff).abs()))
                .unwrap()
        })
        .unwrap()
        .0;

    let v2_cutoff = v2_data[cutoff_idx].norm();
    // At cutoff, gain should be ~0.707 (-3dB)
    assert_abs_diff_eq!(v2_cutoff, 1.0 / std::f64::consts::SQRT_2, epsilon = 0.05);

    // At very high frequency (last point), |V(2)| should be small
    let v2_high = v2_data.last().unwrap().norm();
    assert!(
        v2_high < 0.01,
        "Expected V(2) < 0.01 at high freq, got {}",
        v2_high
    );
}

#[test]
fn test_ac_linear_sweep() {
    let netlist = "\
V1 1 0 AC 1 0
R1 1 2 1k
C1 2 0 1u
.AC LIN 100 1 1000
.END
";
    let result = ac_solve(netlist);

    // Linear sweep should have exactly n_points entries
    assert_eq!(result.frequencies.len(), 100);

    // First frequency should be f_start, last should be f_stop
    assert_abs_diff_eq!(result.frequencies[0], 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(*result.frequencies.last().unwrap(), 1000.0, epsilon = 1e-9);
}

#[test]
fn test_ac_octave_sweep() {
    let netlist = "\
V1 1 0 AC 1 0
R1 1 2 1k
C1 2 0 1u
.AC OCT 5 100 800
.END
";
    let result = ac_solve(netlist);

    // OCT: 5 points per octave, from 100 to 800 Hz = 3 octaves = 15 points
    let octaves = (800.0_f64 / 100.0).log2();
    let expected_total = (5.0 * octaves).ceil() as usize;
    assert_eq!(result.frequencies.len(), expected_total);

    // First frequency should be f_start
    assert_abs_diff_eq!(result.frequencies[0], 100.0, epsilon = 1e-9);
}

// ── CSV Output Tests ──────────────────────────────────────────────

#[test]
fn test_dc_csv_output() {
    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
    let result = dc_solve(netlist);

    let mut buf = Vec::new();
    output::write_dc_csv(&result, &mut buf).expect("CSV write failed");
    let csv = String::from_utf8(buf).unwrap();

    // Check header
    assert!(csv.starts_with("Variable,Value\n"));

    // Check that we have V(1), V(2), and I(V1) lines
    assert!(csv.contains("V(1),"));
    assert!(csv.contains("V(2),"));
    assert!(csv.contains("I(V1),"));
}

#[test]
fn test_ac_csv_output() {
    let netlist = "\
V1 1 0 AC 1 0
R1 1 2 1k
C1 2 0 1u
.AC LIN 3 100 1000
.END
";
    let result = ac_solve(netlist);

    let mut buf = Vec::new();
    output::write_ac_csv(&result, &mut buf).expect("CSV write failed");
    let csv = String::from_utf8(buf).unwrap();

    let lines: Vec<&str> = csv.lines().collect();

    // Header + 3 data rows
    assert_eq!(
        lines.len(),
        4,
        "Expected 4 lines (header + 3 data), got {}: {:?}",
        lines.len(),
        lines
    );

    // Header should contain Frequency and magnitude/phase columns
    assert!(lines[0].starts_with("Frequency"));
    assert!(lines[0].contains("_mag"));
    assert!(lines[0].contains("_phase_deg"));

    // Each data line should have the same number of commas as the header
    let header_commas = lines[0].matches(',').count();
    for line in &lines[1..] {
        assert_eq!(
            line.matches(',').count(),
            header_commas,
            "Data row has different number of columns than header"
        );
    }
}

// ── CLI / Fixture Test ────────────────────────────────────────────

#[test]
fn test_cli_with_fixture_file() {
    // Verify we can parse and solve the fixture file end-to-end
    let input =
        std::fs::read_to_string("tests/fixtures/voltage_divider.spice").expect("read fixture");
    let circuit = parser::parse(&input).expect("parse fixture");
    let system = compiler::compile(&circuit).expect("compile fixture");
    let solver = CpuSolver::new();

    let dc_result = analysis::dc::run(&system, &solver, None).expect("DC solve fixture");

    let v2 = dc_result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "2")
        .unwrap()
        .1;
    assert_abs_diff_eq!(v2, 5.0, epsilon = 1e-6);

    // Also verify CSV output doesn't error
    let mut buf = Vec::new();
    output::write_dc_csv(&dc_result, &mut buf).expect("CSV write");
    assert!(!buf.is_empty());
}
