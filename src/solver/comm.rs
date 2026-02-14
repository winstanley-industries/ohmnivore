//! Communication backend abstraction for distributed solvers.
//!
//! Provides a trait for cross-GPU coordination (dot-product reductions,
//! halo exchange) and a no-op single-process implementation.

/// Neighbor data for halo exchange: rank and the local indices to send/receive.
pub struct HaloNeighbor {
    pub rank: usize,
    pub send_indices: Vec<usize>,
    pub recv_start: usize,
    pub recv_count: usize,
}

/// Abstraction over inter-process communication for distributed solvers.
///
/// Implementations: `SingleProcessComm` (no-op), `MpiComm` (via mpi crate).
pub trait CommunicationBackend: Send + Sync {
    /// Sum a local scalar across all ranks.
    fn all_reduce_sum(&self, local: f64) -> f64;

    /// Max of a local scalar across all ranks.
    fn all_reduce_max(&self, local: f64) -> f64;

    /// Exchange halo boundary values with neighbor ranks.
    ///
    /// `neighbors` describes which ranks to exchange with and which local
    /// indices to send. `local_data` contains the full local vector.
    /// `recv_halo` is filled with received values in the order defined by
    /// the neighbor recv regions.
    fn halo_exchange(
        &self,
        neighbors: &[HaloNeighbor],
        local_data: &[f64],
        recv_halo: &mut [f64],
    );

    /// This process's rank (subdomain index).
    fn rank(&self) -> usize;

    /// Total number of ranks (subdomains).
    fn num_ranks(&self) -> usize;

    /// Element-wise sum of a vector across all ranks, in place.
    ///
    /// Each rank contributes its local values (zeros for non-owned positions).
    /// After the call, every rank holds the global sum. Used for reassembling
    /// a distributed solution vector.
    fn all_reduce_sum_vec(&self, local: &mut [f64]);

    /// Synchronization barrier.
    fn barrier(&self);
}

/// No-op communication backend for single-GPU execution.
///
/// All operations pass through unchanged. Halo exchange is a no-op
/// because there are no neighbor ranks.
pub struct SingleProcessComm;

impl CommunicationBackend for SingleProcessComm {
    fn all_reduce_sum(&self, local: f64) -> f64 {
        local
    }

    fn all_reduce_max(&self, local: f64) -> f64 {
        local
    }

    fn halo_exchange(
        &self,
        _neighbors: &[HaloNeighbor],
        _local_data: &[f64],
        _recv_halo: &mut [f64],
    ) {
        // Single process: no neighbors, nothing to exchange.
    }

    fn all_reduce_sum_vec(&self, _local: &mut [f64]) {
        // Single process: vector is already complete.
    }

    fn rank(&self) -> usize {
        0
    }

    fn num_ranks(&self) -> usize {
        1
    }

    fn barrier(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_process_all_reduce_sum() {
        let comm = SingleProcessComm;
        assert_eq!(comm.all_reduce_sum(42.0), 42.0);
        assert_eq!(comm.all_reduce_sum(-1.5), -1.5);
    }

    #[test]
    fn single_process_all_reduce_max() {
        let comm = SingleProcessComm;
        assert_eq!(comm.all_reduce_max(42.0), 42.0);
    }

    #[test]
    fn single_process_rank_and_size() {
        let comm = SingleProcessComm;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.num_ranks(), 1);
    }

    #[test]
    fn single_process_halo_exchange_is_noop() {
        let comm = SingleProcessComm;
        let send = vec![1.0, 2.0, 3.0];
        let mut recv = vec![0.0; 0];
        // Single process has no neighbors, so this should be a no-op.
        comm.halo_exchange(&[], &send, &mut recv);
        assert!(recv.is_empty());
    }
}
