//! Distributed Newton-Raphson solver for nonlinear DC analysis.
//!
//! Wires the distributed BiCGSTAB solver with RAS ISAI(1) preconditioner
//! into the Newton-Raphson iteration loop. For single-GPU (SingleProcessComm),
//! this degenerates to a single partition with no halo exchange.
//!
//! - **Linear circuits**: Solved directly via distributed BiCGSTAB + RAS ISAI(1).
//! - **Nonlinear circuits (single rank)**: Delegates to the existing GPU
//!   `newton_solve` which handles device evaluation, assembly, and BiCGSTAB
//!   entirely on GPU.
//! - **Nonlinear circuits (multi rank)**: Not yet implemented; returns an error.

use crate::compiler::MnaSystem;
use crate::error::{OhmnivoreError, Result};

use super::comm::CommunicationBackend;

/// Solve a DC operating point using the distributed solver path.
///
/// Routes to the appropriate solver based on whether the circuit has nonlinear
/// elements and how many ranks are available:
///
/// 1. **Nonlinear + single rank**: GPU Newton-Raphson (existing `newton_solve`)
/// 2. **Nonlinear + multi rank**: Error (not yet implemented)
/// 3. **Linear**: Distributed BiCGSTAB + RAS ISAI(1) preconditioner
pub fn solve_dc(system: &MnaSystem, comm: &dyn CommunicationBackend) -> Result<Vec<f64>> {
    let _span = tracing::info_span!("distributed_newton_dc").entered();

    let has_nonlinear = !system.diode_descriptors.is_empty()
        || !system.bjt_descriptors.is_empty()
        || !system.mosfet_descriptors.is_empty();

    if has_nonlinear {
        if comm.num_ranks() > 1 {
            return Err(OhmnivoreError::Solve(
                "distributed Newton with nonlinear elements across multiple ranks \
                 is not yet implemented"
                    .into(),
            ));
        }
        solve_nonlinear_single_gpu(system)
    } else {
        solve_linear_distributed(system, comm)
    }
}

/// Nonlinear single-GPU path: delegates to the existing GPU Newton-Raphson.
fn solve_nonlinear_single_gpu(system: &MnaSystem) -> Result<Vec<f64>> {
    use super::backend::SolverBackend;
    use super::backend::WgpuBackend;
    use super::ds_backend::WgpuDsBackend;
    use super::newton::{newton_solve, NewtonParams};

    let backend = WgpuBackend::new()?;
    let ds_backend = WgpuDsBackend::new()?;

    let values_f32: Vec<f32> = system.g.values.iter().map(|&v| v as f32).collect();
    let col_indices_u32: Vec<u32> = system.g.col_indices.iter().map(|&c| c as u32).collect();
    let row_ptrs_u32: Vec<u32> = system.g.row_pointers.iter().map(|&r| r as u32).collect();
    let b_dc_f32: Vec<f32> = system.b_dc.iter().map(|&v| v as f32).collect();

    let base_b_buf = backend.new_buffer(system.size);
    backend.upload_vec(&b_dc_f32, &base_b_buf);

    let matrix_nnz = system.g.values.len();

    newton_solve(
        &backend,
        &ds_backend,
        &base_b_buf,
        &values_f32,
        &system.g.values,
        &col_indices_u32,
        &row_ptrs_u32,
        &system.b_dc,
        &system.diode_descriptors,
        &system.bjt_descriptors,
        &system.mosfet_descriptors,
        system.size,
        matrix_nnz,
        &NewtonParams::default(),
        None,
    )
}

/// Linear circuit path: distributed BiCGSTAB + RAS ISAI(1).
///
/// Each rank partitions the circuit graph, extracts its local submatrix and
/// subvector (owned + halo), solves cooperatively via distributed BiCGSTAB,
/// then reassembles the global solution via all-reduce.
fn solve_linear_distributed(
    system: &MnaSystem,
    comm: &dyn CommunicationBackend,
) -> Result<Vec<f64>> {
    use super::distributed_preconditioner::RasIsaiPreconditioner;
    use super::partition::{MetisPartitioner, Partitioner, SubdomainMap};

    let n = system.size;
    if n == 0 {
        return Ok(Vec::new());
    }

    let num_ranks = comm.num_ranks();
    let rank = comm.rank();

    // Build adjacency graph from MNA matrix sparsity pattern.
    let adj = build_adjacency(&system.g);

    // Partition into num_ranks parts.
    let partitioner = MetisPartitioner;
    let parts = partitioner.partition(&adj, num_ranks);

    // Build subdomain map for this rank.
    let map = SubdomainMap::build(&adj, &parts, rank, 1);

    // Extract local submatrix and subvector.
    let local_a = map.extract_submatrix(&system.g);
    let local_b = map.extract_subvector(&system.b_dc);

    // Build RAS ISAI(1) preconditioner on local submatrix.
    let ras = RasIsaiPreconditioner::new(&local_a, &map, 1);

    // Solve using distributed BiCGSTAB with DS backend for f64 precision.
    let local_x = solve_local_ds(&local_a, &local_b, &map, &ras, comm)?;

    // Scatter owned entries back to global solution.
    let mut global_x = vec![0.0; n];
    map.scatter_to_global(&local_x, &mut global_x);

    // For multi-rank: combine disjoint owned contributions from all ranks.
    // Each rank has zeros for non-owned positions, so element-wise sum yields
    // the full solution.
    if num_ranks > 1 {
        comm.all_reduce_sum_vec(&mut global_x);
    }

    Ok(global_x)
}

/// Solve a local system using the DS backend with distributed BiCGSTAB
/// and ISAI(1) preconditioning.
fn solve_local_ds(
    local_a: &crate::sparse::CsrMatrix<f64>,
    local_b: &[f64],
    map: &super::partition::SubdomainMap,
    ras: &super::distributed_preconditioner::RasIsaiPreconditioner,
    comm: &dyn CommunicationBackend,
) -> Result<Vec<f64>> {
    use super::backend::SolverBackend;
    use super::comm::HaloNeighbor;
    use super::ds_backend::WgpuDsBackend;

    let n_total = local_a.nrows;
    let n_owned = map.n_owned();

    let ds_backend = WgpuDsBackend::new()?;

    let col_indices_u32: Vec<u32> = local_a.col_indices.iter().map(|&c| c as u32).collect();
    let row_ptrs_u32: Vec<u32> = local_a.row_pointers.iter().map(|&r| r as u32).collect();

    let gpu_matrix =
        ds_backend.upload_matrix_f64(&local_a.values, &col_indices_u32, &row_ptrs_u32, n_total);
    let x_buf = ds_backend.new_buffer(n_total);
    let b_buf = ds_backend.new_buffer(n_total);
    ds_backend.upload_vec_f64(local_b, &b_buf);

    // Upload ISAI factors to DS backend.
    let ml = &ras.local_isai.m_l;
    let ml_cols: Vec<u32> = ml.col_indices.iter().map(|&c| c as u32).collect();
    let ml_rows: Vec<u32> = ml.row_pointers.iter().map(|&r| r as u32).collect();
    let ml_gpu = ds_backend.upload_matrix_f64(&ml.values, &ml_cols, &ml_rows, n_total);

    let mu = &ras.local_isai.m_u;
    let mu_cols: Vec<u32> = mu.col_indices.iter().map(|&c| c as u32).collect();
    let mu_rows: Vec<u32> = mu.row_pointers.iter().map(|&r| r as u32).collect();
    let mu_gpu = ds_backend.upload_matrix_f64(&mu.values, &mu_cols, &mu_rows, n_total);

    let tmp = ds_backend.new_buffer(n_total);

    // Pre-build neighbor list for halo exchange (used by halo_sync closure).
    let neighbors: Vec<HaloNeighbor> = map
        .neighbor_ranks
        .iter()
        .enumerate()
        .map(|(i, &rank)| HaloNeighbor {
            rank,
            send_indices: map.send_indices[i].clone(),
            recv_start: map.recv_indices[..i].iter().map(|v| v.len()).sum(),
            recv_count: map.recv_indices[i].len(),
        })
        .collect();
    let total_recv: usize = map.recv_indices.iter().map(|v| v.len()).sum();

    super::distributed_bicgstab::distributed_bicgstab(
        &ds_backend,
        &gpu_matrix,
        &b_buf,
        &x_buf,
        |b: &WgpuDsBackend,
         inp: &super::backend::WgpuBuffer,
         out: &super::backend::WgpuBuffer| {
            b.spmv(&ml_gpu, inp, &tmp);
            b.spmv(&mu_gpu, &tmp, out);
        },
        |buf| {
            if neighbors.is_empty() {
                return; // No neighbors, no halo exchange needed.
            }

            // Download buffer from GPU to CPU.
            let mut local_vec = vec![0.0f64; n_total];
            ds_backend.download_vec_f64(buf, &mut local_vec);

            // MPI halo exchange: send owned boundary values, receive halo values.
            let mut recv_buf = vec![0.0; total_recv];
            comm.halo_exchange(&neighbors, &local_vec, &mut recv_buf);

            // Scatter received values into halo positions.
            let mut recv_offset = 0;
            for recv_idxs in &map.recv_indices {
                for &idx in recv_idxs {
                    local_vec[idx] = recv_buf[recv_offset];
                    recv_offset += 1;
                }
            }

            // Upload back to GPU.
            ds_backend.upload_vec_f64(&local_vec, buf);
        },
        n_owned,
        n_total,
        comm,
    )?;

    // Download result as f64.
    let mut result = vec![0.0f64; n_total];
    ds_backend.download_vec_f64(&x_buf, &mut result);
    Ok(result)
}

/// Build a symmetric adjacency matrix from the sparsity pattern of an MNA matrix.
///
/// Removes self-loops (diagonal entries) since METIS adjacency format
/// represents only off-diagonal connections.
fn build_adjacency(a: &crate::sparse::CsrMatrix<f64>) -> crate::sparse::CsrMatrix<f64> {
    let n = a.nrows;
    let mut triplets = Vec::new();
    for i in 0..n {
        let start = a.row_pointers[i];
        let end = a.row_pointers[i + 1];
        for idx in start..end {
            let j = a.col_indices[idx];
            if i != j {
                // Only add (i,j) — from_triplets deduplicates symmetric pairs.
                triplets.push((i, j, 1.0));
            }
        }
    }
    crate::sparse::CsrMatrix::from_triplets(n, n, &triplets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::comm::SingleProcessComm;

    #[test]
    fn distributed_newton_linear_voltage_divider() {
        // V1=10V -> node 1 -> R1(1k) -> node 2 -> R2(1k) -> GND
        // Expected: V(1)=10V, V(2)=5V
        let netlist = "\
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
";
        let circuit = crate::parser::parse(netlist).expect("parse failed");
        let system = crate::compiler::compile(&circuit).expect("compile failed");

        let comm = SingleProcessComm;
        let result = match solve_dc(&system, &comm) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        assert_eq!(result.len(), system.size);

        let v1_idx = system
            .node_names
            .iter()
            .position(|n| n == "1")
            .expect("node 1 not found");
        let v2_idx = system
            .node_names
            .iter()
            .position(|n| n == "2")
            .expect("node 2 not found");

        assert!(
            (result[v1_idx] - 10.0).abs() < 0.1,
            "V(1)={}, expected 10.0",
            result[v1_idx]
        );
        assert!(
            (result[v2_idx] - 5.0).abs() < 0.1,
            "V(2)={}, expected 5.0",
            result[v2_idx]
        );
    }

    #[test]
    fn distributed_newton_nonlinear_diode() {
        // Simple diode circuit: V1=0.7V -> R1(100) -> D1 -> GND
        let netlist = "\
V1 1 0 DC 0.7
R1 1 2 100
D1 2 0 DMOD
.MODEL DMOD D IS=1e-14 N=1.0
.DC
.END
";
        let circuit = crate::parser::parse(netlist).expect("parse failed");
        let system = crate::compiler::compile(&circuit).expect("compile failed");

        let comm = SingleProcessComm;
        let result = match solve_dc(&system, &comm) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        assert_eq!(result.len(), system.size);

        // V(1) should be 0.7V (set by voltage source)
        let v1_idx = system
            .node_names
            .iter()
            .position(|n| n == "1")
            .expect("node 1 not found");
        assert!(
            (result[v1_idx] - 0.7).abs() < 0.1,
            "V(1)={}, expected ~0.7",
            result[v1_idx]
        );
    }

    #[test]
    fn distributed_newton_bench_circuit() {
        // Test with a bench circuit (5-stage inverter chain)
        let netlist =
            match std::fs::read_to_string("bench/circuits/inverter_chain_5_dc.spice") {
                Ok(s) => s,
                Err(_) => {
                    eprintln!("skipping: bench circuit not found");
                    return;
                }
            };
        let circuit = crate::parser::parse(&netlist).expect("parse failed");
        let system = crate::compiler::compile(&circuit).expect("compile failed");

        let comm = SingleProcessComm;
        let result = match solve_dc(&system, &comm) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        assert_eq!(result.len(), system.size);
    }

    #[test]
    fn distributed_newton_multi_rank_nonlinear_errors() {
        // Multi-rank nonlinear should return an error
        use crate::solver::comm::{CommunicationBackend, HaloNeighbor};

        struct TwoRankComm;
        impl CommunicationBackend for TwoRankComm {
            fn all_reduce_sum(&self, local: f64) -> f64 { local }
            fn all_reduce_max(&self, local: f64) -> f64 { local }
            fn halo_exchange(&self, _: &[HaloNeighbor], _: &[f64], _: &mut [f64]) {}
            fn all_reduce_sum_vec(&self, _: &mut [f64]) {}
            fn rank(&self) -> usize { 0 }
            fn num_ranks(&self) -> usize { 2 }
            fn barrier(&self) {}
        }

        let netlist = "\
V1 1 0 DC 0.7
R1 1 2 100
D1 2 0 DMOD
.MODEL DMOD D IS=1e-14 N=1.0
.DC
.END
";
        let circuit = crate::parser::parse(netlist).expect("parse failed");
        let system = crate::compiler::compile(&circuit).expect("compile failed");

        let comm = TwoRankComm;
        let result = solve_dc(&system, &comm);
        assert!(
            result.is_err(),
            "multi-rank nonlinear should error"
        );
    }
}
