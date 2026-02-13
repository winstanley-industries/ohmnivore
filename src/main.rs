use clap::Parser;
use ohmnivore::analysis;
use ohmnivore::compiler;
use ohmnivore::ir::Analysis;
use ohmnivore::output;
use ohmnivore::parser;
use ohmnivore::solver::cpu::CpuSolver;
use ohmnivore::solver::gpu::GpuSolver;
use ohmnivore::solver::LinearSolver;
use std::io;
use std::time::Instant;

/// GPU-accelerated circuit simulation solver
#[derive(Parser)]
#[command(name = "ohmnivore", version)]
struct Cli {
    /// SPICE netlist file to simulate
    netlist: String,

    /// Use CPU solver instead of GPU
    #[arg(long)]
    cpu: bool,

    /// Print performance stats to stderr
    #[arg(long)]
    stats: bool,
}

fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let mut stats = if cli.stats { Some(ohmnivore::stats::Stats::new()) } else { None };

    let input = std::fs::read_to_string(&cli.netlist).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", cli.netlist, e);
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

    let solver: Box<dyn LinearSolver> = if cli.cpu {
        Box::new(CpuSolver::new())
    } else {
        Box::new(GpuSolver::new().unwrap_or_else(|e| {
            eprintln!("GPU solver error: {}", e);
            std::process::exit(1);
        }))
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
            Analysis::Tran {
                tstep,
                tstop,
                tstart,
                uic,
            } => {
                let tran_result = analysis::transient::run(
                    &system,
                    solver.as_ref(),
                    *tstep,
                    *tstop,
                    *tstart,
                    *uic,
                    &circuit.components,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Transient analysis error: {}", e);
                    std::process::exit(1);
                });
                output::write_tran_csv(&tran_result, &mut stdout).unwrap_or_else(|e| {
                    eprintln!("Output error: {}", e);
                    std::process::exit(1);
                });
            }
        }
    }

    if let Some(ref stats) = stats {
        stats.display();
    }
}
