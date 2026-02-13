use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::ir::Analysis;
use ohmnivore::output;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;
use ohmnivore::solver::gpu::GpuSolver;
use ohmnivore::solver::LinearSolver;
use std::io;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ohmnivore <netlist.spice> [--gpu]");
        std::process::exit(1);
    }

    let use_gpu = args.iter().any(|a| a == "--gpu");

    let input_file = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| {
            eprintln!("Usage: ohmnivore <netlist.spice> [--gpu]");
            std::process::exit(1);
        });

    let input = std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", input_file, e);
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

    let solver: Box<dyn LinearSolver> = if use_gpu {
        Box::new(GpuSolver::new().unwrap_or_else(|e| {
            eprintln!("GPU solver error: {}", e);
            std::process::exit(1);
        }))
    } else {
        Box::new(CpuSolver::new())
    };
    let mut stdout = io::stdout();

    for analysis_cmd in &circuit.analyses {
        match analysis_cmd {
            Analysis::Dc => {
                let dc_result = analysis::dc::run(&system, solver.as_ref()).unwrap_or_else(|e| {
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
                    solver.as_ref(),
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
