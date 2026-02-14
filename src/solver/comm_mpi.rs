//! MPI communication backend for distributed solvers.
//!
//! Requires the `distributed` feature flag and an MPI installation.
//! Implements `CommunicationBackend` using `mpi::traits::*` for
//! inter-process communication (all-reduce, halo exchange).
//!
//! # Usage
//!
//! The caller must initialize MPI before constructing `MpiComm`:
//!
//! ```ignore
//! let universe = mpi::initialize().expect("MPI init failed");
//! let comm = MpiComm::new();
//! ```
//!
//! # Halo exchange
//!
//! The current implementation uses blocking send/recv for simplicity.
//! This can deadlock if both neighbors send simultaneously without
//! matching receives. A production implementation should use
//! `MPI_Isend`/`MPI_Irecv` (non-blocking).
//! TODO: Switch to non-blocking send/recv for the RoCE transport backend.

use super::comm::{CommunicationBackend, HaloNeighbor};
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

/// MPI-based communication backend for distributed solvers.
///
/// Wraps the MPI world communicator. Requires `mpi::initialize()` to have
/// been called before construction.
pub struct MpiComm;

impl MpiComm {
    /// Create a new MPI communication backend.
    ///
    /// Panics if MPI has not been initialized via `mpi::initialize()`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MpiComm {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunicationBackend for MpiComm {
    fn all_reduce_sum(&self, local: f64) -> f64 {
        let world = SimpleCommunicator::world();
        let mut global = 0.0f64;
        world.all_reduce_into(&local, &mut global, SystemOperation::sum());
        global
    }

    fn all_reduce_max(&self, local: f64) -> f64 {
        let world = SimpleCommunicator::world();
        let mut global = 0.0f64;
        world.all_reduce_into(&local, &mut global, SystemOperation::max());
        global
    }

    fn halo_exchange(
        &self,
        neighbors: &[HaloNeighbor],
        local_data: &[f64],
        recv_halo: &mut [f64],
    ) {
        let world = SimpleCommunicator::world();
        let my_rank = world.rank();

        // Use rank-based ordering to avoid deadlock: the lower-ranked process
        // sends first, the higher-ranked receives first.
        for nbr in neighbors {
            let send_data: Vec<f64> = nbr.send_indices.iter().map(|&i| local_data[i]).collect();
            let peer = world.process_at_rank(nbr.rank as i32);
            let recv_slice = &mut recv_halo[nbr.recv_start..nbr.recv_start + nbr.recv_count];

            if my_rank < nbr.rank as i32 {
                peer.send(&send_data[..]);
                peer.receive_into(recv_slice);
            } else {
                peer.receive_into(recv_slice);
                peer.send(&send_data[..]);
            }
        }
    }

    fn all_reduce_sum_vec(&self, local: &mut [f64]) {
        let world = SimpleCommunicator::world();
        let send = local.to_vec();
        world.all_reduce_into(&send[..], local, SystemOperation::sum());
    }

    fn rank(&self) -> usize {
        let world = SimpleCommunicator::world();
        world.rank() as usize
    }

    fn num_ranks(&self) -> usize {
        let world = SimpleCommunicator::world();
        world.size() as usize
    }

    fn barrier(&self) {
        let world = SimpleCommunicator::world();
        world.barrier();
    }
}
