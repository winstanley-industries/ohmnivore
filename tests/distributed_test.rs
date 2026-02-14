//! Multi-process distributed solver tests.
//!
//! These tests require MPI and the `distributed` feature flag.
//! Run with: mpirun -n 1 cargo test --features distributed --test distributed_test
//!
//! Without MPI installed, these tests are excluded from the default build.

#![cfg(feature = "distributed")]

use ohmnivore::solver::comm::CommunicationBackend;
use ohmnivore::solver::comm_mpi::MpiComm;
use ohmnivore::solver::distributed_newton;

#[test]
fn distributed_voltage_divider_single_rank() {
    // Verify the MPI backend works in the degenerate single-process case.
    // Run with: mpirun -n 1 cargo test --features distributed --test distributed_test
    //
    // Multi-rank distributed linear solve is not yet implemented (the
    // distributed BiCGSTAB needs per-rank owned/halo vector partitioning
    // and real halo exchange). Skip if launched with more than 1 rank.
    let _universe = mpi::initialize().expect("MPI init failed");
    let comm = MpiComm::new();

    if comm.num_ranks() > 1 {
        eprintln!(
            "skipping: multi-rank distributed linear solve not yet implemented \
             (run with mpirun -n 1)"
        );
        return;
    }

    let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
    let circuit = ohmnivore::parser::parse(netlist).expect("parse failed");
    let system = ohmnivore::compiler::compile(&circuit).expect("compile failed");

    let result = distributed_newton::solve_dc(&system, &comm).expect("solve failed");

    assert_eq!(result.len(), system.size);

    // Node 2 should be 5V (voltage divider)
    let node_2_idx = system
        .node_names
        .iter()
        .position(|n| n == "2")
        .expect("node 2 not found");
    assert!(
        (result[node_2_idx] - 5.0).abs() < 0.1,
        "V(2)={}, expected 5.0",
        result[node_2_idx]
    );
}
