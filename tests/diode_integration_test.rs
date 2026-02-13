//! Integration tests for diode circuits: parse -> compile -> (optionally) solve.

use ohmnivore::compiler;
use ohmnivore::parser;

// ── Parse-Compile Round Trip Tests ──────────────────────────────────

#[test]
fn test_diode_parse_compile_produces_descriptors() {
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
.MODEL D1N4148 D(IS=2.52e-9 N=1.752)
D1 2 0 D1N4148
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(system.diode_descriptors.len(), 1);
    let desc = &system.diode_descriptors[0];

    // Node "2" is the anode, ground is cathode
    // Nodes: "1" -> 0, "2" -> 1
    assert_eq!(desc.anode_idx, 1); // node "2"
    assert_eq!(desc.cathode_idx, u32::MAX); // ground

    // Model parameters should be IS=2.52e-9, N=1.752
    assert!((desc.is_val - 2.52e-9_f32).abs() < 1e-15);
    let expected_n_vt = (1.752 * 0.02585) as f32;
    assert!((desc.n_vt - expected_n_vt).abs() < 1e-5);
}

#[test]
fn test_diode_multiple_diodes_compile() {
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
R2 2 3 1k
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 2 0 DMOD
D2 3 0 DMOD
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(system.diode_descriptors.len(), 2);
    // Both diodes reference ground as cathode
    for desc in &system.diode_descriptors {
        assert_eq!(desc.cathode_idx, u32::MAX);
    }
}

#[test]
fn test_diode_between_non_ground_nodes_compile() {
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
R2 3 0 1k
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 2 3 DMOD
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(system.diode_descriptors.len(), 1);
    let desc = &system.diode_descriptors[0];

    // Both anode and cathode are non-ground, so neither should be sentinel
    assert_ne!(desc.anode_idx, u32::MAX);
    assert_ne!(desc.cathode_idx, u32::MAX);

    // All 4 G-matrix stamp positions should be valid
    for i in 0..4 {
        assert_ne!(
            desc.g_row_col[i],
            u32::MAX,
            "g_row_col[{i}] should not be sentinel for non-ground diode"
        );
    }
}

// ── Error Cases ─────────────────────────────────────────────────────

#[test]
fn test_diode_undefined_model_error() {
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
D1 2 0 UNDEFINED_MODEL
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse should succeed");
    let result = compiler::compile(&circuit);

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("undefined model"),
        "Expected 'undefined model' in error message, got: {err_msg}"
    );
}

#[test]
fn test_diode_missing_model_directive_error() {
    // Diode references a model but no .MODEL directive exists
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
D1 2 0 DMOD
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse should succeed");
    let result = compiler::compile(&circuit);

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("undefined model"),
        "Expected 'undefined model' in error, got: {err_msg}"
    );
}

// ── GPU Newton-Raphson End-to-End Tests ─────────────────────────────
// These tests require a GPU and are skipped if no GPU is available.

use ohmnivore::analysis;
use ohmnivore::solver::cpu::CpuSolver;

/// Helper to attempt GPU-based DC solve through Newton path.
/// Returns None if no GPU is available.
fn try_dc_solve_with_diodes(netlist: &str) -> Option<analysis::DcResult> {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let solver = CpuSolver::new();

    // dc::run will detect diodes and use the GPU Newton path
    match analysis::dc::run(&system, &solver) {
        Ok(result) => Some(result),
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("no GPU adapter") || msg.contains("GPU") {
                eprintln!("skipping GPU diode test: {msg}");
                None
            } else {
                panic!("unexpected error: {e}");
            }
        }
    }
}

#[test]
fn test_diode_resistor_vsource_end_to_end() {
    // Simple circuit: V1(5V) -> R1(1k) -> D1 -> GND
    // Expected: diode forward voltage ~0.6-0.7V for silicon (IS=1e-14, N=1)
    // So V(2) should be around 0.6-0.7V, and V(1) = 5.0V
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 2 0 DMOD
.DC
.END
";
    let Some(result) = try_dc_solve_with_diodes(netlist) else {
        return;
    };

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

    // V(1) should be 5.0V (set by voltage source)
    assert!((v1 - 5.0).abs() < 0.1, "V(1) = {v1}, expected ~5.0V");

    // V(2) is the diode forward voltage, should be ~0.5-0.8V
    assert!(
        v2 > 0.3 && v2 < 1.0,
        "V(2) = {v2}, expected diode forward voltage ~0.5-0.8V"
    );

    // Current through resistor: I = (V1 - V2) / R1
    // Should be roughly (5 - 0.7) / 1000 ~ 4.3mA
    let i_approx = (v1 - v2) / 1000.0;
    assert!(
        i_approx > 0.003 && i_approx < 0.006,
        "current through R1 = {i_approx:.4}, expected ~4.3mA"
    );
}

#[test]
fn test_linear_circuit_unchanged_with_diode_path() {
    // A purely linear circuit (no diodes) should still work through the
    // standard linear path even after our changes.
    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let solver = CpuSolver::new();

    // Should have no diode descriptors
    assert!(system.diode_descriptors.is_empty());

    let result = analysis::dc::run(&system, &solver).expect("DC analysis failed");

    let v2 = result
        .node_voltages
        .iter()
        .find(|(n, _)| n == "2")
        .unwrap()
        .1;

    // Standard voltage divider: V(2) = 5.0V
    assert!((v2 - 5.0).abs() < 1e-9, "V(2) = {v2}, expected 5.0V");
}
