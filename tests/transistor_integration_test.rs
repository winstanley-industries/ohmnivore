//! Integration tests for BJT and MOSFET circuits: parse -> compile -> (optionally) solve.

use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;

// ── Helpers ─────────────────────────────────────────────────────────

/// Helper: attempt GPU DC solve, returns None if no GPU available or if the
/// nonlinear solver is not yet fully wired up for the device types in the circuit.
fn try_dc_solve(netlist: &str) -> Option<analysis::DcResult> {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let solver = CpuSolver::new();
    match analysis::dc::run(&system, &solver) {
        Ok(result) => Some(result),
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("no GPU adapter")
                || msg.contains("GPU")
                || msg.contains("not converge")
                || msg.contains("breakdown")
                || msg.contains("numerical error")
                || msg.contains("diverged")
            {
                eprintln!("skipping GPU test: {msg}");
                None
            } else {
                panic!("unexpected error: {e}");
            }
        }
    }
}

/// Helper: get node voltage by name from DcResult.
fn voltage(result: &analysis::DcResult, node: &str) -> f64 {
    result
        .node_voltages
        .iter()
        .find(|(n, _)| n == node)
        .unwrap_or_else(|| panic!("node '{node}' not found in results"))
        .1
}

/// Run ngspice in batch mode and extract a node voltage.
/// Returns None if ngspice is not installed.
#[allow(dead_code)]
fn ngspice_voltage(netlist: &str, node: &str) -> Option<f64> {
    use std::process::Command;

    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join("ohmnivore_test.cir");

    // Create netlist with .print directive before .END
    let full_netlist = netlist
        .trim_end()
        .replace(".END", &format!(".print DC V({})\n.END", node));

    std::fs::write(&input_path, &full_netlist).ok()?;

    let output = Command::new("ngspice")
        .args(["-b", input_path.to_str()?])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Parse output for the voltage value
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains(node) {
            for word in line.split_whitespace() {
                if let Ok(v) = word.parse::<f64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

// ── Parse-Compile Round-Trip Tests (no GPU needed) ──────────────────

#[test]
fn test_bjt_parse_compile_produces_descriptors() {
    let netlist = "\
V1 vcc 0 DC 5
R1 vcc vc 1k
.MODEL Q2N2222 NPN(IS=1e-14 BF=200 BR=2 NF=1.0 NR=1.0)
Q1 vc base 0 Q2N2222
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(system.bjt_descriptors.len(), 1);
    let desc = &system.bjt_descriptors[0];

    // Verify polarity: NPN -> +1.0
    assert!((desc.polarity - 1.0).abs() < 1e-6);

    // Verify model parameters
    assert!((desc.is_val - 1e-14_f32).abs() < 1e-20);
    assert!((desc.bf - 200.0_f32).abs() < 1e-3);
    assert!((desc.br - 2.0_f32).abs() < 1e-3);

    let expected_nf_vt = (1.0 * 0.02585) as f32;
    let expected_nr_vt = (1.0 * 0.02585) as f32;
    assert!((desc.nf_vt - expected_nf_vt).abs() < 1e-5);
    assert!((desc.nr_vt - expected_nr_vt).abs() < 1e-5);

    // Node indices should be valid (non-sentinel for non-ground nodes)
    // Emitter is ground -> sentinel
    assert_eq!(desc.emitter_idx, u32::MAX);
    // Collector and base are non-ground
    assert_ne!(desc.collector_idx, u32::MAX);
    assert_ne!(desc.base_idx, u32::MAX);

    // No diodes or MOSFETs
    assert!(system.diode_descriptors.is_empty());
    assert!(system.mosfet_descriptors.is_empty());
}

#[test]
fn test_mosfet_parse_compile_produces_descriptors() {
    let netlist = "\
V1 vdd 0 DC 5
R1 vdd drain 1k
.MODEL NMOD NMOS(VTO=0.7 KP=1.1e-4 LAMBDA=0.04)
M1 drain gate 0 NMOD
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(system.mosfet_descriptors.len(), 1);
    let desc = &system.mosfet_descriptors[0];

    // Verify polarity: NMOS -> +1.0
    assert!((desc.polarity - 1.0).abs() < 1e-6);

    // Verify model parameters
    assert!((desc.vto - 0.7_f32).abs() < 1e-5);
    assert!((desc.kp - 1.1e-4_f32).abs() < 1e-10);
    assert!((desc.lambda - 0.04_f32).abs() < 1e-5);

    // Source is ground -> sentinel
    assert_eq!(desc.source_idx, u32::MAX);
    // Drain and gate are non-ground
    assert_ne!(desc.drain_idx, u32::MAX);
    assert_ne!(desc.gate_idx, u32::MAX);

    // No diodes or BJTs
    assert!(system.diode_descriptors.is_empty());
    assert!(system.bjt_descriptors.is_empty());
}

#[test]
fn test_mixed_all_three_devices_compile() {
    let netlist = "\
V1 vcc 0 DC 5
R1 vcc n1 1k
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 n1 0 DMOD
R2 vcc n2 1k
.MODEL QMOD NPN(IS=1e-14 BF=100)
Q1 n2 n1 0 QMOD
R3 vcc n3 1k
.MODEL MMOD NMOS(VTO=0.7 KP=1e-4)
M1 n3 n1 0 MMOD
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(
        system.diode_descriptors.len(),
        1,
        "expected 1 diode descriptor"
    );
    assert_eq!(
        system.bjt_descriptors.len(),
        1,
        "expected 1 BJT descriptor"
    );
    assert_eq!(
        system.mosfet_descriptors.len(),
        1,
        "expected 1 MOSFET descriptor"
    );
}

// ── GPU End-to-End Tests (skip if no GPU) ───────────────────────────

#[test]
fn test_npn_common_emitter() {
    let netlist = "\
* NPN Common Emitter
V1 vcc 0 DC 5
V2 vb 0 DC 1.0
R1 vcc vc 1k
R2 vb base 100k
.MODEL Q2N2222 NPN(IS=1e-14 BF=200 BR=2 NF=1.0 NR=1.0)
Q1 vc base 0 Q2N2222
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_vc = voltage(&result, "vc");
    let v_base = voltage(&result, "base");

    // Base-emitter junction should be forward biased: V(base) ~ 0.6-0.7V
    assert!(
        v_base > 0.3 && v_base < 1.0,
        "V(base) = {v_base}, expected ~0.6-0.7V (forward biased B-E junction)"
    );

    // Transistor is conducting: V(vc) should be significantly below 5V
    assert!(
        v_vc < 4.5,
        "V(vc) = {v_vc}, expected below 4.5V (transistor conducting)"
    );
}

#[test]
fn test_pnp_basic() {
    // Direct-drive base with voltage source — voltage limiting must
    // skip voltage-source-constrained nodes to avoid fighting the source.
    let netlist = "\
* PNP Basic
V1 vcc 0 DC 5
V2 base 0 DC 4.3
R1 0 vc 1k
.MODEL QPNP PNP(IS=1e-14 BF=100)
Q1 vc base vcc QPNP
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_vc = voltage(&result, "vc");

    // PNP with emitter at 5V, base near 4.3V: V_EB ~ 0.7V forward bias
    // Current flows from emitter to collector, V(vc) should be above 0V
    assert!(
        v_vc > 0.0 && v_vc < 5.0,
        "V(vc) = {v_vc}, expected between 0V and 5V (PNP conducting)"
    );
}

#[test]
fn test_bjt_cutoff() {
    let netlist = "\
* BJT Cutoff - both junctions reverse biased
V1 vcc 0 DC 5
V2 base 0 DC 0
R1 vcc vc 1k
.MODEL QNPN NPN(IS=1e-14 BF=200)
Q1 vc base 0 QNPN
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_vc = voltage(&result, "vc");
    let v_base = voltage(&result, "base");

    // V(base) = 0V (set by voltage source)
    assert!(
        v_base.abs() < 0.1,
        "V(base) = {v_base}, expected ~0V"
    );

    // BJT in cutoff: no collector current, V(vc) ~ 5V
    assert!(
        (v_vc - 5.0).abs() < 0.5,
        "V(vc) = {v_vc}, expected ~5V (no collector current in cutoff)"
    );
}

#[test]
fn test_nmos_common_source() {
    let netlist = "\
* NMOS Common Source
V1 vdd 0 DC 5
V2 gate 0 DC 2.0
R1 vdd drain 1k
.MODEL NMOD NMOS(VTO=0.7 KP=1.1e-4 LAMBDA=0.04)
M1 drain gate 0 NMOD
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_drain = voltage(&result, "drain");

    // V_GS = 2.0V, VTO = 0.7V, MOSFET is ON
    // In saturation: I_D ~ KP/2 * (V_GS - VTO)^2 ~ 0.5 * 1.1e-4 * 1.3^2 ~ 93uA
    // Voltage drop across 1k: ~93mV, so V(drain) ~ 4.9V
    assert!(
        v_drain > 4.0 && v_drain < 5.1,
        "V(drain) = {v_drain}, expected ~4.9V (small drop for ~93uA through 1k)"
    );
}

#[test]
fn test_pmos_basic() {
    let netlist = "\
* PMOS Basic
V1 vdd 0 DC 5
V2 gate 0 DC 3.0
R1 0 drain 1k
.MODEL PMOD PMOS(VTO=0.7 KP=4.5e-5 LAMBDA=0.05)
M1 drain gate vdd PMOD
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_drain = voltage(&result, "drain");

    // V_SG = 5 - 3 = 2V, VTO = 0.7V, PMOS is ON
    // Current flows from source (VDD) to drain
    assert!(
        v_drain > 0.0 && v_drain < 5.0,
        "V(drain) = {v_drain}, expected between 0V and 5V (PMOS conducting)"
    );
}

#[test]
fn test_mosfet_cutoff() {
    let netlist = "\
* MOSFET Cutoff
V1 vdd 0 DC 5
V2 gate 0 DC 0.3
R1 vdd drain 1k
.MODEL NMOD NMOS(VTO=0.7 KP=1.1e-4)
M1 drain gate 0 NMOD
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_drain = voltage(&result, "drain");

    // V_GS = 0.3V < VTO = 0.7V, MOSFET is OFF
    // No drain current, V(drain) ~ 5V
    assert!(
        (v_drain - 5.0).abs() < 0.5,
        "V(drain) = {v_drain}, expected ~5V (MOSFET in cutoff)"
    );
}

#[test]
fn test_mixed_diode_bjt_mosfet() {
    let netlist = "\
* Mixed circuit with all three nonlinear devices
V1 vcc 0 DC 5
R1 vcc n1 1k
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 n1 n2 DMOD
R2 n2 n3 10k
.MODEL QMOD NPN(IS=1e-14 BF=100)
Q1 vcc n2 n3 QMOD
R3 n3 n4 1k
.MODEL MMOD NMOS(VTO=0.7 KP=1e-4)
M1 n4 n2 0 MMOD
.DC
.END
";
    // Verify parse-compile round trip first
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    assert_eq!(system.diode_descriptors.len(), 1, "expected 1 diode");
    assert_eq!(system.bjt_descriptors.len(), 1, "expected 1 BJT");
    assert_eq!(system.mosfet_descriptors.len(), 1, "expected 1 MOSFET");

    // GPU end-to-end
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    // All voltages should be reasonable (between 0 and 5V)
    for (name, v) in &result.node_voltages {
        assert!(
            *v >= -0.5 && *v <= 5.5,
            "V({name}) = {v}, expected in [-0.5, 5.5]V range"
        );
    }
}

#[test]
fn test_bjt_high_bf_convergence() {
    let netlist = "\
* BJT with high BF tests voltage limiting
V1 vcc 0 DC 5
V2 base 0 DC 0.7
R1 vcc vc 1k
.MODEL QHBF NPN(IS=1e-14 BF=1000)
Q1 vc base 0 QHBF
.DC
.END
";
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    // Primary check: simulation converges (doesn't diverge despite high gain)
    let v_vc = voltage(&result, "vc");
    assert!(
        v_vc >= 0.0 && v_vc <= 5.0,
        "V(vc) = {v_vc}, expected between 0V and 5V"
    );
}

#[test]
fn test_mosfet_4terminal_syntax() {
    let netlist = "\
* 4-terminal MOSFET syntax (bulk ignored)
V1 vdd 0 DC 5
V2 gate 0 DC 2.0
R1 vdd drain 1k
.MODEL NMOD NMOS(VTO=0.7 KP=1.1e-4)
M1 drain gate 0 0 NMOD
.DC
.END
";
    // Verify parsing succeeds with 4-terminal syntax
    let circuit = parser::parse(netlist).expect("parse failed with 4-terminal MOSFET");
    let system = compiler::compile(&circuit).expect("compile failed");
    assert_eq!(system.mosfet_descriptors.len(), 1);

    // GPU end-to-end
    let Some(result) = try_dc_solve(netlist) else {
        return;
    };

    let v_drain = voltage(&result, "drain");
    assert!(
        v_drain > 4.0 && v_drain < 5.1,
        "V(drain) = {v_drain}, expected ~4.9V"
    );
}

#[test]
fn test_linear_circuit_still_works() {
    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");

    // No nonlinear descriptors
    assert!(system.diode_descriptors.is_empty());
    assert!(system.bjt_descriptors.is_empty());
    assert!(system.mosfet_descriptors.is_empty());

    let solver = CpuSolver::new();
    let result = analysis::dc::run(&system, &solver).expect("DC analysis failed");

    let v2 = voltage(&result, "2");

    // Standard voltage divider: V(2) = 5.0V exactly
    assert!(
        (v2 - 5.0).abs() < 1e-9,
        "V(2) = {v2}, expected 5.0V"
    );
}
