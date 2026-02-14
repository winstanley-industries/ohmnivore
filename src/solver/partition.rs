//! Graph partitioning and subdomain mapping for distributed solvers.
//!
//! Uses METIS (via metis-rs) to partition the circuit graph, then builds
//! SubdomainMap structures with 1-layer overlap for RAS preconditioning.

use crate::sparse::CsrMatrix;
use std::collections::{HashMap, HashSet};

/// Graph partitioner trait. Returns a partition ID per node.
pub trait Partitioner {
    fn partition(&self, adjacency: &CsrMatrix<f64>, num_parts: usize) -> Vec<usize>;
}

/// METIS-based graph partitioner (pure Rust via metis-rs).
pub struct MetisPartitioner;

impl Partitioner for MetisPartitioner {
    fn partition(&self, adjacency: &CsrMatrix<f64>, num_parts: usize) -> Vec<usize> {
        let n = adjacency.nrows;
        if num_parts <= 1 {
            return vec![0; n];
        }

        // Convert CsrMatrix adjacency to metis-rs Graph format.
        let xadj: Vec<usize> = adjacency.row_pointers.clone();
        let adjncy: Vec<usize> = adjacency.col_indices.clone();

        let graph = metis_rs::Graph::new(n, xadj, adjncy);
        let (_edge_cut, parts) = metis_rs::partition(&graph, num_parts);
        parts
    }
}

/// Subdomain-to-global index mapping with overlap halo.
///
/// Local ordering: owned nodes first (0..n_owned), halo nodes after
/// (n_owned..n_owned+n_halo). This layout enables:
/// - Dot products summing over 0..n_owned (no double-counting)
/// - Restriction to owned nodes via simple slice
pub struct SubdomainMap {
    /// Global indices owned by this subdomain.
    pub owned_global: Vec<usize>,
    /// Global indices in the overlap halo (from neighboring partitions).
    pub halo_global: Vec<usize>,
    /// Global -> local index mapping (owned + halo).
    pub global_to_local: HashMap<usize, usize>,
    /// Ranks of neighboring subdomains (those sharing halo nodes).
    pub neighbor_ranks: Vec<usize>,
    /// Per-neighbor: local indices of owned nodes to send during halo exchange.
    pub send_indices: Vec<Vec<usize>>,
    /// Per-neighbor: local indices where received halo values are written.
    pub recv_indices: Vec<Vec<usize>>,
}

impl SubdomainMap {
    /// Build a subdomain map for the given rank from a partition assignment.
    ///
    /// `adjacency` is the circuit graph. `parts[i]` is the partition ID for
    /// node i. `rank` is which partition this map is for. `overlap_layers`
    /// is how many hops of overlap to add (1 for standard RAS).
    pub fn build(
        adjacency: &CsrMatrix<f64>,
        parts: &[usize],
        rank: usize,
        overlap_layers: usize,
    ) -> Self {
        let n = adjacency.nrows;

        // Collect owned nodes (those assigned to this rank).
        let owned_set: HashSet<usize> = (0..n).filter(|&i| parts[i] == rank).collect();
        let owned_global: Vec<usize> = {
            let mut v: Vec<usize> = owned_set.iter().copied().collect();
            v.sort();
            v
        };

        // Expand by overlap_layers hops to find halo nodes.
        let mut frontier: HashSet<usize> = owned_set.clone();
        let mut all_nodes: HashSet<usize> = owned_set.clone();

        for _ in 0..overlap_layers {
            let mut next_frontier = HashSet::new();
            for &node in &frontier {
                let start = adjacency.row_pointers[node];
                let end = adjacency.row_pointers[node + 1];
                for idx in start..end {
                    let neighbor = adjacency.col_indices[idx];
                    if !all_nodes.contains(&neighbor) {
                        next_frontier.insert(neighbor);
                    }
                }
            }
            all_nodes.extend(&next_frontier);
            frontier = next_frontier;
        }

        // Halo = expanded set minus owned.
        let mut halo_global: Vec<usize> =
            all_nodes.difference(&owned_set).copied().collect();
        halo_global.sort();

        // Build global->local mapping: owned first, halo after.
        let mut global_to_local = HashMap::new();
        for (local, &global) in owned_global.iter().enumerate() {
            global_to_local.insert(global, local);
        }
        let n_owned = owned_global.len();
        for (i, &global) in halo_global.iter().enumerate() {
            global_to_local.insert(global, n_owned + i);
        }

        // Determine neighbor ranks and build send/recv index lists.
        // A neighbor rank is any rank that owns a halo node.
        let mut neighbor_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for &halo_node in &halo_global {
            let halo_rank = parts[halo_node];
            neighbor_map
                .entry(halo_rank)
                .or_default()
                .push(halo_node);
        }

        let mut neighbor_ranks: Vec<usize> = neighbor_map.keys().copied().collect();
        neighbor_ranks.sort();

        // For each neighbor, determine:
        // - send_indices: local indices of our owned nodes adjacent to their partition
        // - recv_indices: local indices where their halo values land
        let mut send_indices = Vec::new();
        let mut recv_indices = Vec::new();

        for &nbr_rank in &neighbor_ranks {
            let halo_nodes = &neighbor_map[&nbr_rank];

            // recv_indices: where halo nodes from this neighbor sit in local indexing
            let recv: Vec<usize> = halo_nodes
                .iter()
                .map(|&g| global_to_local[&g])
                .collect();

            // send_indices: our owned boundary nodes adjacent to this neighbor's partition
            let nbr_owned: HashSet<usize> =
                (0..n).filter(|&i| parts[i] == nbr_rank).collect();
            let mut send_set = HashSet::new();
            for &owned_node in &owned_global {
                let start = adjacency.row_pointers[owned_node];
                let end = adjacency.row_pointers[owned_node + 1];
                for idx in start..end {
                    if nbr_owned.contains(&adjacency.col_indices[idx]) {
                        send_set.insert(global_to_local[&owned_node]);
                        break;
                    }
                }
            }
            let mut send: Vec<usize> = send_set.into_iter().collect();
            send.sort();

            send_indices.push(send);
            recv_indices.push(recv);
        }

        SubdomainMap {
            owned_global,
            halo_global,
            global_to_local,
            neighbor_ranks,
            send_indices,
            recv_indices,
        }
    }

    /// Number of owned nodes.
    pub fn n_owned(&self) -> usize {
        self.owned_global.len()
    }

    /// Number of halo nodes.
    pub fn n_halo(&self) -> usize {
        self.halo_global.len()
    }

    /// Total local size (owned + halo).
    pub fn local_size(&self) -> usize {
        self.n_owned() + self.n_halo()
    }
}

#[cfg(test)]
mod tests {
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
    fn partition_two_disconnected_components() {
        // Graph: 0-1, 2-3 (two disconnected edges)
        let adj = adjacency_from_edges(4, &[(0, 1), (2, 3)]);
        let partitioner = super::MetisPartitioner;
        let parts = super::Partitioner::partition(&partitioner, &adj, 2);
        assert_eq!(parts.len(), 4);
        // Nodes 0,1 should be in one partition; 2,3 in the other.
        assert_eq!(parts[0], parts[1]);
        assert_eq!(parts[2], parts[3]);
        assert_ne!(parts[0], parts[2]);
    }

    #[test]
    fn partition_returns_correct_count() {
        // Chain: 0-1-2-3-4-5
        let adj = adjacency_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let partitioner = super::MetisPartitioner;
        let parts = super::Partitioner::partition(&partitioner, &adj, 3);
        assert_eq!(parts.len(), 6);
        // All partition IDs should be in 0..3
        for &p in &parts {
            assert!(p < 3, "partition id {p} out of range");
        }
    }

    #[test]
    fn subdomain_map_single_partition() {
        // 4 nodes, 1 partition — everything is owned, no halo.
        let adj = adjacency_from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let parts = vec![0, 0, 0, 0];
        let map = super::SubdomainMap::build(&adj, &parts, 0, 1);
        assert_eq!(map.owned_global.len(), 4);
        assert_eq!(map.halo_global.len(), 0);
        assert_eq!(map.neighbor_ranks.len(), 0);
    }

    #[test]
    fn subdomain_map_two_partitions_with_overlap() {
        // Chain: 0-1-2-3, partitioned as [0,0,1,1]
        let adj = adjacency_from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let parts = vec![0, 0, 1, 1];

        let map0 = super::SubdomainMap::build(&adj, &parts, 0, 1);
        let map1 = super::SubdomainMap::build(&adj, &parts, 1, 1);

        // Partition 0 owns {0,1}, halo should include {2} (1-hop from node 1)
        assert_eq!(map0.owned_global.len(), 2);
        assert!(map0.owned_global.contains(&0));
        assert!(map0.owned_global.contains(&1));
        assert_eq!(map0.halo_global.len(), 1);
        assert!(map0.halo_global.contains(&2));

        // Partition 1 owns {2,3}, halo should include {1} (1-hop from node 2)
        assert_eq!(map1.owned_global.len(), 2);
        assert!(map1.owned_global.contains(&2));
        assert!(map1.owned_global.contains(&3));
        assert_eq!(map1.halo_global.len(), 1);
        assert!(map1.halo_global.contains(&1));

        // Both should list each other as neighbors
        assert_eq!(map0.neighbor_ranks, vec![1]);
        assert_eq!(map1.neighbor_ranks, vec![0]);
    }

    #[test]
    fn subdomain_map_global_to_local_mapping() {
        let adj = adjacency_from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let parts = vec![0, 0, 1, 1];
        let map0 = super::SubdomainMap::build(&adj, &parts, 0, 1);

        // Owned nodes come first in local ordering
        for (local_idx, &global_idx) in map0.owned_global.iter().enumerate() {
            assert_eq!(map0.global_to_local[&global_idx], local_idx);
        }
        // Halo nodes come after owned
        let n_owned = map0.owned_global.len();
        for (i, &global_idx) in map0.halo_global.iter().enumerate() {
            assert_eq!(map0.global_to_local[&global_idx], n_owned + i);
        }
    }
}
