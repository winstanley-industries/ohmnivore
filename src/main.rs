use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::ir::Analysis;
use ohmnivore::output;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;
use std::io;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ohmnivore <netlist.spice>");
        std::process::exit(1);
    }

    let input = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", args[1], e);
        std::process::exit(1);
    });

    let circuit = parser::parse(&input).unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        std::process::exit(1);
    });

    let system = compiler::compile(&circuit).unwrap_or_else(|e| {
        eprintln!("Compile error: {}", e);
        std::process::exit(1);
    });

    let solver = CpuSolver::new();
    let mut stdout = io::stdout();

    for analysis_cmd in &circuit.analyses {
        match analysis_cmd {
            Analysis::Dc => {
                let dc_result = analysis::dc::run(&system, &solver).unwrap_or_else(|e| {
                    eprintln!("DC analysis error: {}", e);
                    std::process::exit(1);
                });
                output::write_dc_csv(&dc_result, &mut stdout).unwrap_or_else(|e| {
                    eprintln!("Output error: {}", e);
                    std::process::exit(1);
                });
            }
            Analysis::Ac {
                sweep_type,
                n_points,
                f_start,
                f_stop,
            } => {
                let ac_result = analysis::ac::run(
                    &system,
                    &solver,
                    *sweep_type,
                    *n_points,
                    *f_start,
                    *f_stop,
                )
                .unwrap_or_else(|e| {
                    eprintln!("AC analysis error: {}", e);
                    std::process::exit(1);
                });
                output::write_ac_csv(&ac_result, &mut stdout).unwrap_or_else(|e| {
                    eprintln!("Output error: {}", e);
                    std::process::exit(1);
                });
            }
        }
    }
}
