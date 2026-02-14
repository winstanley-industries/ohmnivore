//! Distributed preconditioner implementations.
//!
//! RAS (Restricted Additive Schwarz) with local ISAI(1) per subdomain.

use crate::sparse::CsrMatrix;

use super::comm::{CommunicationBackend, HaloNeighbor};
use super::partition::SubdomainMap;
use super::preconditioner::{self, IsaiPreconditionerData};

/// Trait for distributed preconditioners.
pub trait DistributedPreconditioner: Send + Sync {
    /// Apply the preconditioner: output = M^{-1} * input.
    ///
    /// Handles halo exchange internally. Input/output are in local indexing
    /// (owned + halo). Only owned entries of output are meaningful.
    fn apply_cpu(
        &self,
        input: &[f64],
        map: &SubdomainMap,
        comm: &dyn CommunicationBackend,
    ) -> Vec<f64>;
}

/// RAS preconditioner with local ISAI(level) on each subdomain.
pub struct RasIsaiPreconditioner {
    /// Local ISAI factors for this subdomain's matrix.
    pub local_isai: IsaiPreconditionerData<f64>,
}

impl RasIsaiPreconditioner {
    /// Compute ISAI(level) on the local subdomain matrix.
    pub fn new(local_matrix: &CsrMatrix<f64>, _map: &SubdomainMap, level: usize) -> Self {
        let local_isai = preconditioner::compute_isai(local_matrix, level);
        Self { local_isai }
    }

    /// Recompute ISAI factors after matrix values change (e.g., Newton iteration).
    /// Sparsity pattern is assumed unchanged.
    pub fn update(&mut self, local_matrix: &CsrMatrix<f64>, level: usize) {
        self.local_isai = preconditioner::compute_isai(local_matrix, level);
    }
}

impl DistributedPreconditioner for RasIsaiPreconditioner {
    fn apply_cpu(
        &self,
        input: &[f64],
        map: &SubdomainMap,
        comm: &dyn CommunicationBackend,
    ) -> Vec<f64> {
        // Step 1: Halo exchange -- update halo entries of input from neighbors.
        let mut synced_input = input.to_vec();

        // Exchange with neighbors: halo_exchange gathers send values internally
        // using send_indices into the full local vector.
        let total_recv: usize = map.recv_indices.iter().map(|v| v.len()).sum();
        let mut recv_buf = vec![0.0; total_recv];

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

        comm.halo_exchange(&neighbors, &synced_input, &mut recv_buf);

        // Scatter received values into halo positions.
        let mut recv_offset = 0;
        for recv_idxs in &map.recv_indices {
            for &idx in recv_idxs {
                synced_input[idx] = recv_buf[recv_offset];
                recv_offset += 1;
            }
        }

        // Step 2: Apply local ISAI: output = M_U * (M_L * input)
        let tmp = self.local_isai.m_l.spmv(&synced_input);
        let full_output = self.local_isai.m_u.spmv(&tmp);

        // Step 3: Restrict to owned nodes (discard halo entries in output).
        full_output[..map.n_owned()].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::comm::SingleProcessComm;
    use crate::solver::partition::SubdomainMap;
    use crate::sparse::CsrMatrix;

    /// Build a symmetric adjacency matrix from edge pairs.
    fn adjacency_from_edges(n: usize, edges: &[(usize, usize)]) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for &(u, v) in edges {
            triplets.push((u, v, 1.0));
            triplets.push((v, u, 1.0));
        }
        CsrMatrix::from_triplets(n, n, &triplets)
    }

    #[test]
    fn ras_isai_single_partition_matches_global_isai() {
        // When there's only one partition, RAS ISAI should behave
        // identically to applying ISAI(1) on the full matrix.
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);

        // Build single-partition subdomain
        let parts = vec![0, 0, 0];
        let adj = adjacency_from_edges(3, &[(0, 1), (1, 2)]);
        let map = SubdomainMap::build(&adj, &parts, 0, 1);

        let ras = super::RasIsaiPreconditioner::new(&a, &map, 1);

        // Apply to a test vector
        let input = vec![1.0, 2.0, 3.0];
        let comm = SingleProcessComm;
        let output = ras.apply_cpu(&input, &map, &comm);

        // Compare against global ISAI(1)
        let global_isai = crate::solver::preconditioner::compute_isai(&a, 1);
        let tmp = global_isai.m_l.spmv(&input);
        let expected = global_isai.m_u.spmv(&tmp);

        for (o, e) in output.iter().zip(expected.iter()) {
            let diff: f64 = (o - e).abs();
            assert!(diff < 1e-10, "RAS output {o} != global ISAI {e}");
        }
    }
}
