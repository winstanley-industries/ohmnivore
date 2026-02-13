//! Integration tests for transient analysis.

use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;

/// Helper: parse + compile + transient solve
fn tran_solve(netlist: &str) -> ohmnivore::analysis::TranResult {
    let circuit = parser::parse(netlist).expect("parse failed");
    let system = compiler::compile(&circuit).expect("compile failed");
    let solver = CpuSolver::new();

    let tran_cmd = circuit
        .analyses
        .iter()
        .find_map(|a| match a {
            ohmnivore::ir::Analysis::Tran {
                tstep,
                tstop,
                tstart,
                uic,
            } => Some((*tstep, *tstop, *tstart, *uic)),
            _ => None,
        })
        .expect("no TRAN analysis in netlist");

    analysis::transient::run(
        &system,
        &solver,
        tran_cmd.0,
        tran_cmd.1,
        tran_cmd.2,
        tran_cmd.3,
        &circuit.components,
        None,
    )
    .expect("transient analysis failed")
}

/// Find the voltage waveform for a node by name.
fn find_voltage<'a>(result: &'a ohmnivore::analysis::TranResult, node: &str) -> &'a Vec<f64> {
    &result
        .node_voltages
        .iter()
        .find(|(n, _)| n == node)
        .unwrap_or_else(|| panic!("node {} not found", node))
        .1
}

#[test]
fn test_rc_charging() {
    // RC circuit: V1(5V step) -> R(1k) -> C(1u) -> GND
    // Using UIC to start from zero initial conditions.
    // V(out) = V0 * (1 - exp(-t/RC))
    // RC = 1k * 1u = 1ms
    let netlist = "\
V1 1 0 DC 5
R1 1 2 1k
C1 2 0 1u
.TRAN 10u 5m UIC
.END
";
    let result = tran_solve(netlist);
    let v_out = find_voltage(&result, "2");

    let rc = 1e-3; // 1k * 1u = 1ms
    let v0 = 5.0;

    for (i, &t) in result.times.iter().enumerate() {
        if t > 0.0 {
            let expected = v0 * (1.0 - (-t / rc).exp());
            let actual = v_out[i];
            // Allow 2% error for numerical integration
            let tol = 0.02 * v0 + 1e-6;
            assert!(
                (actual - expected).abs() < tol,
                "RC charging: at t={:.2e}, expected={:.4}, actual={:.4}, diff={:.2e}",
                t,
                expected,
                actual,
                (actual - expected).abs()
            );
        }
    }
}

#[test]
fn test_rl_step_response() {
    // RL circuit: V1(5V) -> R(100) -> L(10m) -> GND
    // Using UIC to start from zero initial conditions.
    // I(L) = V/R * (1 - exp(-Rt/L))
    // Tau = L/R = 10m/100 = 100us
    let netlist = "\
V1 1 0 DC 5
R1 1 2 100
L1 2 0 10m
.TRAN 1u 500u UIC
.END
";
    let result = tran_solve(netlist);

    // Check the current through L1 (branch current)
    let i_l = &result
        .branch_currents
        .iter()
        .find(|(n, _)| n == "L1")
        .expect("L1 branch not found")
        .1;

    let r = 100.0;
    let l = 10e-3;
    let v = 5.0;
    let tau = l / r; // 100us

    for (i, &t) in result.times.iter().enumerate() {
        if t > 1e-6 {
            let expected = v / r * (1.0 - (-t / tau).exp());
            let actual = i_l[i];
            let tol = 0.02 * (v / r) + 1e-6;
            assert!(
                (actual - expected).abs() < tol,
                "RL step: at t={:.2e}, expected={:.6}, actual={:.6}, diff={:.2e}",
                t,
                expected,
                actual,
                (actual - expected).abs()
            );
        }
    }
}

#[test]
fn test_lc_oscillation() {
    // LC circuit: V1(5V) -> L(1m) -> C(1u) -> GND
    // At DC operating point, inductor is short so V(2) = 5V.
    // During transient, the LC node should oscillate.
    let netlist = "\
V1 1 0 DC 5
L1 1 2 1m
C1 2 0 1u
.TRAN 1u 200u
.END
";
    let result = tran_solve(netlist);
    let v_out = find_voltage(&result, "2");

    // Should see oscillation: max above 4.9V and we have multiple output points
    let max_v = v_out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_v > 4.9,
        "Expected oscillation peak above 4.9V, got {}",
        max_v
    );
    assert!(
        result.times.len() > 10,
        "Expected multiple output points, got {}",
        result.times.len()
    );
}

#[test]
fn test_pulse_source_rc() {
    // PULSE source driving an RC circuit
    let netlist = "\
V1 1 0 PULSE(0 5 0 0 0 50u 100u)
R1 1 2 1k
C1 2 0 100n
.TRAN 1u 200u
.END
";
    let result = tran_solve(netlist);
    let v_out = find_voltage(&result, "2");

    // Output should show RC charging/discharging following pulse
    // RC = 1k * 100n = 100us
    // Pulse width = 50us, period = 100us
    assert!(result.times.len() > 10);

    // During first pulse (0 to 50us), output should be rising toward 5V
    // Find a point near 25us
    let idx_25u = result.times.iter().position(|&t| t >= 25e-6).unwrap();
    assert!(
        v_out[idx_25u] > 0.5,
        "Expected output > 0.5V at 25us, got {}",
        v_out[idx_25u]
    );
    assert!(
        v_out[idx_25u] < 5.0,
        "Expected output < 5V at 25us, got {}",
        v_out[idx_25u]
    );
}

#[test]
fn test_sin_source() {
    let netlist = "\
V1 1 0 SIN(0 1 10k)
R1 1 0 1k
.TRAN 1u 200u
.END
";
    let result = tran_solve(netlist);
    let v_out = find_voltage(&result, "1");

    // V1 is a sine wave at 10kHz. Since R1 is in parallel with V1,
    // V(1) should follow the source exactly.
    for (i, &t) in result.times.iter().enumerate() {
        let expected = (2.0 * std::f64::consts::PI * 10e3 * t).sin();
        let tol = 0.05; // 5% tolerance for voltage source enforcement
        assert!(
            (v_out[i] - expected).abs() < tol,
            "SIN: at t={:.2e}, expected={:.4}, actual={:.4}",
            t,
            expected,
            v_out[i]
        );
    }
}
