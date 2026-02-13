pub mod compare;
pub mod ngspice;

use serde_derive::Deserialize;

use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::ir::Analysis;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;

use self::compare::{
    compare_ac, compare_dc, compare_tran, format_ac_report, format_dc_report, format_tran_report,
    Tolerances,
};
use self::ngspice::{NgspiceBatch, SpiceBackend};

// ── Manifest types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub defaults: Defaults,
    #[serde(rename = "circuit")]
    pub circuits: Vec<CircuitEntry>,
}

#[derive(Debug, Deserialize)]
pub struct Defaults {
    pub dc_voltage_rel_tol: f64,
    pub dc_voltage_abs_tol: f64,
    pub ac_mag_rel_tol: f64,
    pub ac_mag_abs_tol: f64,
    pub ac_phase_abs_tol: f64,
    pub tran_voltage_rel_tol: f64,
    pub tran_voltage_abs_tol: f64,
}

#[derive(Debug, Deserialize)]
pub struct CircuitEntry {
    pub name: String,
    pub file: String,
    pub analysis: String,
    pub compare_nodes: Vec<String>,
    pub dc_voltage_rel_tol: Option<f64>,
    pub dc_voltage_abs_tol: Option<f64>,
    pub ac_mag_rel_tol: Option<f64>,
    pub ac_mag_abs_tol: Option<f64>,
    pub ac_phase_abs_tol: Option<f64>,
    pub tran_voltage_rel_tol: Option<f64>,
    pub tran_voltage_abs_tol: Option<f64>,
}

// ── Manifest loading ────────────────────────────────────────────

pub fn load_manifest() -> Manifest {
    let content = std::fs::read_to_string("tests/regression/manifest.toml")
        .expect("failed to read tests/regression/manifest.toml");
    toml::from_str(&content).expect("failed to parse manifest.toml")
}

// ── Macro for test generation ───────────────────────────────────

macro_rules! regression_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                $crate::regression::run_regression_test(stringify!($name));
            }
        )*
    };
}

pub(crate) use regression_tests;

// ── Test runner ─────────────────────────────────────────────────

pub fn run_regression_test(name: &str) {
    let manifest = load_manifest();
    let entry = manifest
        .circuits
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("circuit '{}' not found in manifest", name));

    let spice_path = format!("tests/regression/circuits/{}", entry.file);
    let netlist_content = std::fs::read_to_string(&spice_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", spice_path, e));

    // Initialize ngspice backend
    let backend = NgspiceBatch::new().unwrap_or_else(|e| panic!("ngspice not available: {}", e));

    let spice_path = std::path::Path::new(&spice_path);

    match entry.analysis.as_str() {
        "dc" => run_dc_comparison(
            entry,
            &netlist_content,
            spice_path,
            &backend,
            &manifest.defaults,
        ),
        "ac" => run_ac_comparison(
            entry,
            &netlist_content,
            spice_path,
            &backend,
            &manifest.defaults,
        ),
        "tran" => run_tran_comparison(
            entry,
            &netlist_content,
            spice_path,
            &backend,
            &manifest.defaults,
        ),
        other => panic!("unknown analysis type: {}", other),
    }
}

fn run_dc_comparison(
    entry: &CircuitEntry,
    netlist: &str,
    spice_path: &std::path::Path,
    backend: &dyn SpiceBackend,
    defaults: &Defaults,
) {
    // Run Ohmnivore
    let circuit = parser::parse(netlist).expect("Ohmnivore parse failed");
    let system = compiler::compile(&circuit).expect("Ohmnivore compile failed");
    let solver = CpuSolver::new();
    let ohm_result = match analysis::dc::run(&system, &solver, None) {
        Ok(result) => result,
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("no GPU adapter") {
                eprintln!("Skipping '{}': {}", entry.name, msg);
                return;
            }
            panic!("Ohmnivore DC solve failed for '{}': {e}", entry.name);
        }
    };

    // Run ngspice
    let ng_result = backend
        .run_dc(spice_path, &entry.compare_nodes)
        .expect("ngspice DC run failed");

    // Compare with tolerances (use per-circuit overrides or defaults)
    let tol = Tolerances {
        rel_tol: entry
            .dc_voltage_rel_tol
            .unwrap_or(defaults.dc_voltage_rel_tol),
        abs_tol: entry
            .dc_voltage_abs_tol
            .unwrap_or(defaults.dc_voltage_abs_tol),
    };

    let results = compare_dc(&ohm_result, &ng_result, &entry.compare_nodes, &tol);
    let any_failed = results.iter().any(|r| !r.passed);

    if any_failed {
        let report = format_dc_report(&results);
        panic!(
            "\n\nRegression test '{}' FAILED:\n\n{}\n",
            entry.name, report
        );
    }
}

fn run_ac_comparison(
    entry: &CircuitEntry,
    netlist: &str,
    spice_path: &std::path::Path,
    backend: &dyn SpiceBackend,
    defaults: &Defaults,
) {
    // Run Ohmnivore
    let circuit = parser::parse(netlist).expect("Ohmnivore parse failed");
    let system = compiler::compile(&circuit).expect("Ohmnivore compile failed");
    let solver = CpuSolver::new();

    // Extract AC parameters from parsed circuit
    let (sweep_type, n_points, f_start, f_stop) = circuit
        .analyses
        .iter()
        .find_map(|a| match a {
            Analysis::Ac {
                sweep_type,
                n_points,
                f_start,
                f_stop,
            } => Some((*sweep_type, *n_points, *f_start, *f_stop)),
            _ => None,
        })
        .expect("no AC analysis found in netlist");

    let ohm_result = analysis::ac::run(&system, &solver, sweep_type, n_points, f_start, f_stop, None)
        .expect("Ohmnivore AC solve failed");

    // Run ngspice
    let ng_result = backend
        .run_ac(spice_path, &entry.compare_nodes)
        .expect("ngspice AC run failed");

    // Compare with tolerances
    let mag_tol = Tolerances {
        rel_tol: entry.ac_mag_rel_tol.unwrap_or(defaults.ac_mag_rel_tol),
        abs_tol: entry.ac_mag_abs_tol.unwrap_or(defaults.ac_mag_abs_tol),
    };
    let phase_abs_tol = entry.ac_phase_abs_tol.unwrap_or(defaults.ac_phase_abs_tol);

    let results = compare_ac(
        &ohm_result,
        &ng_result,
        &entry.compare_nodes,
        &mag_tol,
        phase_abs_tol,
    );
    let any_failed = results.iter().any(|r| !r.mag_passed || !r.phase_passed);

    if any_failed {
        let report = format_ac_report(&results);
        panic!(
            "\n\nRegression test '{}' FAILED:\n\n{}\n",
            entry.name, report
        );
    }
}

fn run_tran_comparison(
    entry: &CircuitEntry,
    netlist: &str,
    spice_path: &std::path::Path,
    backend: &dyn SpiceBackend,
    defaults: &Defaults,
) {
    // Run Ohmnivore
    let circuit = parser::parse(netlist).expect("Ohmnivore parse failed");
    let system = compiler::compile(&circuit).expect("Ohmnivore compile failed");
    let solver = CpuSolver::new();

    // Extract .TRAN parameters from parsed circuit
    let (tstep, tstop, tstart, uic) = circuit
        .analyses
        .iter()
        .find_map(|a| match a {
            Analysis::Tran {
                tstep,
                tstop,
                tstart,
                uic,
            } => Some((*tstep, *tstop, *tstart, *uic)),
            _ => None,
        })
        .expect("no TRAN analysis found in netlist");

    let ohm_result = analysis::transient::run(
        &system,
        &solver,
        tstep,
        tstop,
        tstart,
        uic,
        &circuit.components,
        None,
    )
    .expect("Ohmnivore transient solve failed");

    // Run ngspice
    let ng_result = backend
        .run_tran(spice_path, &entry.compare_nodes)
        .expect("ngspice tran run failed");

    // Compare with tolerances
    let tol = Tolerances {
        rel_tol: entry
            .tran_voltage_rel_tol
            .unwrap_or(defaults.tran_voltage_rel_tol),
        abs_tol: entry
            .tran_voltage_abs_tol
            .unwrap_or(defaults.tran_voltage_abs_tol),
    };

    let results = compare_tran(&ohm_result, &ng_result, &entry.compare_nodes, &tol);
    let any_failed = results.iter().any(|r| !r.passed);

    if any_failed {
        let report = format_tran_report(&results);
        panic!(
            "\n\nRegression test '{}' FAILED:\n\n{}\n",
            entry.name, report
        );
    }
}
