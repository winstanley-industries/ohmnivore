//! MNA (Modified Nodal Analysis) compiler.
//!
//! Converts a Circuit IR into sparse MNA matrices ready for solving.
//!
//! # MNA System
//!
//! For n non-ground nodes and m branch variables (voltage sources + inductors),
//! the system is (n+m) x (n+m):
//!
//! ```text
//! [G] * [x] = [b]
//! ```
//!
//! where x = [node_voltages; branch_currents].
//!
//! # Stamps
//!
//! - **Resistor** R between nodes i,j: G(i,i) += 1/R, G(j,j) += 1/R, G(i,j) -= 1/R, G(j,i) -= 1/R
//! - **Capacitor** C between nodes i,j: same pattern into C matrix (used for AC: A = G + jωC)
//! - **Inductor** L between nodes i,j (branch k at row n+k):
//!   - G(i, n+k) += 1, G(n+k, i) += 1 (KCL coupling)
//!   - G(j, n+k) -= 1, G(n+k, j) -= 1
//!   - C_mat(n+k, n+k) -= L (gives -jωL term in AC)
//!   - At DC: V_i - V_j = 0 (short circuit)
//! - **Voltage source** V between nodes i,j (branch k at row n+k):
//!   - G(i, n+k) += 1, G(n+k, i) += 1
//!   - G(j, n+k) -= 1, G(n+k, j) -= 1
//!   - b_dc(n+k) = V_dc, b_ac(n+k) = V_ac
//! - **Current source** I from n+ to n- with value I0:
//!   - b_dc(n-) += I0, b_dc(n+) -= I0
//!   - b_ac(n-) += I_ac, b_ac(n+) -= I_ac

use crate::error::{OhmnivoreError, Result};
use crate::ir::{Circuit, Component};
use crate::sparse::CsrMatrix;
use num_complex::Complex64;
use std::collections::HashMap;

/// The compiled MNA system, ready for analysis.
#[derive(Debug)]
pub struct MnaSystem {
    /// Conductance matrix (frequency-independent stamps).
    pub g: CsrMatrix<f64>,
    /// Dynamic matrix (capacitance/inductance stamps, multiplied by jw for AC).
    pub c: CsrMatrix<f64>,
    /// DC excitation vector (source contributions).
    pub b_dc: Vec<f64>,
    /// AC excitation vector (complex magnitude+phase from AC sources).
    pub b_ac: Vec<Complex64>,
    /// Total system size (n_nodes + n_branches).
    pub size: usize,
    /// Node names in matrix-index order. Index i -> name of non-ground node i.
    pub node_names: Vec<String>,
    /// Branch variable names in order. Branch k -> component name (e.g., "V1", "L1").
    /// Branch k has matrix index n_nodes + k.
    pub branch_names: Vec<String>,
}

/// Returns true if the node identifier represents ground.
fn is_ground(node: &str) -> bool {
    node == "0" || node.eq_ignore_ascii_case("GND")
}

/// Convert AC (magnitude, phase_degrees) to Complex64.
fn ac_to_complex(mag: f64, phase_deg: f64) -> Complex64 {
    let phase_rad = phase_deg.to_radians();
    Complex64::new(mag * phase_rad.cos(), mag * phase_rad.sin())
}

/// Compile a Circuit into an MNA system.
///
/// 1. Collect all unique node names, assign integer indices (ground = excluded).
/// 2. Count branch variables (voltage sources + inductors).
/// 3. Stamp each component into G, C, b_dc, b_ac.
/// 4. Assemble CSR matrices from accumulated triplets.
pub fn compile(circuit: &Circuit) -> Result<MnaSystem> {
    // Step 1: Collect unique non-ground node names and assign indices.
    let mut node_map: HashMap<String, usize> = HashMap::new();
    let mut node_names: Vec<String> = Vec::new();

    let register_node = |node: &str, map: &mut HashMap<String, usize>, names: &mut Vec<String>| {
        if is_ground(node) {
            return;
        }
        if !map.contains_key(node) {
            let idx = names.len();
            map.insert(node.to_string(), idx);
            names.push(node.to_string());
        }
    };

    for component in &circuit.components {
        let (n_plus, n_minus) = match component {
            Component::Resistor { nodes, .. }
            | Component::Capacitor { nodes, .. }
            | Component::Inductor { nodes, .. }
            | Component::VSource { nodes, .. }
            | Component::ISource { nodes, .. } => (&nodes.0, &nodes.1),
        };
        register_node(n_plus, &mut node_map, &mut node_names);
        register_node(n_minus, &mut node_map, &mut node_names);
    }

    let n_nodes = node_names.len();

    // Step 2: Count branch variables (voltage sources + inductors) and assign branch indices.
    let mut branch_names: Vec<String> = Vec::new();
    let mut branch_map: HashMap<String, usize> = HashMap::new();

    for component in &circuit.components {
        match component {
            Component::VSource { name, .. } | Component::Inductor { name, .. } => {
                let branch_idx = branch_names.len();
                branch_map.insert(name.clone(), branch_idx);
                branch_names.push(name.clone());
            }
            _ => {}
        }
    }

    let n_branches = branch_names.len();
    let size = n_nodes + n_branches;

    // Step 3: Accumulate triplets for G and C matrices, and fill b_dc / b_ac.
    let mut g_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut c_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut b_dc = vec![0.0; size];
    let mut b_ac = vec![Complex64::new(0.0, 0.0); size];

    /// Helper to get the matrix index of a node, or None if ground.
    fn node_index(node: &str, map: &HashMap<String, usize>) -> Option<usize> {
        if is_ground(node) {
            None
        } else {
            map.get(node).copied()
        }
    }

    for component in &circuit.components {
        match component {
            Component::Resistor { name, nodes, value } => {
                let conductance = 1.0 / *value;
                if *value == 0.0 {
                    return Err(OhmnivoreError::Compile(format!(
                        "Resistor {} has zero resistance",
                        name
                    )));
                }
                let ni = node_index(&nodes.0, &node_map);
                let nj = node_index(&nodes.1, &node_map);

                if let Some(i) = ni {
                    g_triplets.push((i, i, conductance));
                }
                if let Some(j) = nj {
                    g_triplets.push((j, j, conductance));
                }
                if let (Some(i), Some(j)) = (ni, nj) {
                    g_triplets.push((i, j, -conductance));
                    g_triplets.push((j, i, -conductance));
                }
            }

            Component::Capacitor { name, nodes, value } => {
                let cap = *value;
                if cap < 0.0 {
                    return Err(OhmnivoreError::Compile(format!(
                        "Capacitor {} has negative capacitance",
                        name
                    )));
                }
                let ni = node_index(&nodes.0, &node_map);
                let nj = node_index(&nodes.1, &node_map);

                if let Some(i) = ni {
                    c_triplets.push((i, i, cap));
                }
                if let Some(j) = nj {
                    c_triplets.push((j, j, cap));
                }
                if let (Some(i), Some(j)) = (ni, nj) {
                    c_triplets.push((i, j, -cap));
                    c_triplets.push((j, i, -cap));
                }
            }

            Component::Inductor { name, nodes, value } => {
                let inductance = *value;
                let branch_idx = branch_map[name];
                let bk = n_nodes + branch_idx;

                let ni = node_index(&nodes.0, &node_map);
                let nj = node_index(&nodes.1, &node_map);

                // KCL coupling: G(i, bk) += 1, G(bk, i) += 1
                if let Some(i) = ni {
                    g_triplets.push((i, bk, 1.0));
                    g_triplets.push((bk, i, 1.0));
                }
                // G(j, bk) -= 1, G(bk, j) -= 1
                if let Some(j) = nj {
                    g_triplets.push((j, bk, -1.0));
                    g_triplets.push((bk, j, -1.0));
                }
                // C_mat(bk, bk) -= L
                c_triplets.push((bk, bk, -inductance));
            }

            Component::VSource {
                name,
                nodes,
                dc,
                ac,
            } => {
                let branch_idx = branch_map[name];
                let bk = n_nodes + branch_idx;

                let ni = node_index(&nodes.0, &node_map);
                let nj = node_index(&nodes.1, &node_map);

                // KCL coupling: G(i, bk) += 1, G(bk, i) += 1
                if let Some(i) = ni {
                    g_triplets.push((i, bk, 1.0));
                    g_triplets.push((bk, i, 1.0));
                }
                // G(j, bk) -= 1, G(bk, j) -= 1
                if let Some(j) = nj {
                    g_triplets.push((j, bk, -1.0));
                    g_triplets.push((bk, j, -1.0));
                }

                // b_dc(bk) = V_dc
                if let Some(v_dc) = dc {
                    b_dc[bk] = *v_dc;
                }
                // b_ac(bk) = V_ac as complex
                if let Some((mag, phase_deg)) = ac {
                    b_ac[bk] = ac_to_complex(*mag, *phase_deg);
                }
            }

            Component::ISource {
                name: _,
                nodes,
                dc,
                ac,
            } => {
                let ni = node_index(&nodes.0, &node_map);
                let nj = node_index(&nodes.1, &node_map);

                // Current flows from n+ to n-. Convention: current into n- is positive.
                // b_dc(n+) -= I_dc, b_dc(n-) += I_dc
                if let Some(i_dc) = dc {
                    if let Some(i) = ni {
                        b_dc[i] -= i_dc;
                    }
                    if let Some(j) = nj {
                        b_dc[j] += i_dc;
                    }
                }
                if let Some((mag, phase_deg)) = ac {
                    let i_ac = ac_to_complex(*mag, *phase_deg);
                    if let Some(i) = ni {
                        b_ac[i] -= i_ac;
                    }
                    if let Some(j) = nj {
                        b_ac[j] += i_ac;
                    }
                }
            }
        }
    }

    // Step 4: Assemble CSR matrices.
    let g = CsrMatrix::from_triplets(size, size, &g_triplets);
    let c = CsrMatrix::from_triplets(size, size, &c_triplets);

    Ok(MnaSystem {
        g,
        c,
        b_dc,
        b_ac,
        size,
        node_names,
        branch_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Circuit, Component};

    /// Helper: create a circuit with given components and no analyses.
    fn circuit_with(components: Vec<Component>) -> Circuit {
        Circuit {
            components,
            analyses: vec![],
        }
    }

    #[test]
    fn test_single_resistor_to_ground() {
        // R1 1k between node "1" and GND
        // Expected: 1x1 G matrix, G(0,0) = 1/1000 = 0.001
        let circuit = circuit_with(vec![Component::Resistor {
            name: "R1".into(),
            nodes: ("1".into(), "0".into()),
            value: 1000.0,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 1);
        assert_eq!(mna.node_names, vec!["1"]);
        assert!(mna.branch_names.is_empty());

        let g_dense = mna.g.to_dense();
        assert_eq!(g_dense.len(), 1);
        assert!((g_dense[0][0] - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_voltage_divider() {
        // Classic voltage divider: V1(5V) -> node 1 -> R1(1k) -> node 2 -> R2(1k) -> GND
        // V1 between node "1" and "0" (ground), 5V DC
        // R1 between "1" and "2", 1k
        // R2 between "2" and "0", 1k
        //
        // Nodes: "1" -> idx 0, "2" -> idx 1
        // Branches: V1 -> branch 0, matrix row 2
        //
        // Size = 3 (2 nodes + 1 branch)
        //
        // G matrix (3x3):
        //   Row 0 (node 1): G(0,0) = 1/1k = 0.001, G(0,1) = -1/1k = -0.001, G(0,2) = 1 (V1 coupling)
        //   Row 1 (node 2): G(1,0) = -1/1k, G(1,1) = 1/1k + 1/1k = 0.002, G(1,2) = 0
        //   Row 2 (V1 branch): G(2,0) = 1 (V1 KVL), G(2,1) = 0, G(2,2) = 0
        //
        // b_dc = [0, 0, 5.0]
        let circuit = circuit_with(vec![
            Component::VSource {
                name: "V1".into(),
                nodes: ("1".into(), "0".into()),
                dc: Some(5.0),
                ac: None,
            },
            Component::Resistor {
                name: "R1".into(),
                nodes: ("1".into(), "2".into()),
                value: 1000.0,
            },
            Component::Resistor {
                name: "R2".into(),
                nodes: ("2".into(), "0".into()),
                value: 1000.0,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 3);
        assert_eq!(mna.node_names, vec!["1", "2"]);
        assert_eq!(mna.branch_names, vec!["V1"]);

        let g = mna.g.to_dense();
        // Row 0: [0.001, -0.001, 1.0]
        assert!((g[0][0] - 0.001).abs() < 1e-12);
        assert!((g[0][1] - (-0.001)).abs() < 1e-12);
        assert!((g[0][2] - 1.0).abs() < 1e-12);

        // Row 1: [-0.001, 0.002, 0.0]
        assert!((g[1][0] - (-0.001)).abs() < 1e-12);
        assert!((g[1][1] - 0.002).abs() < 1e-12);
        assert!((g[1][2] - 0.0).abs() < 1e-12);

        // Row 2: [1.0, 0.0, 0.0]
        assert!((g[2][0] - 1.0).abs() < 1e-12);
        assert!((g[2][1] - 0.0).abs() < 1e-12);
        assert!((g[2][2] - 0.0).abs() < 1e-12);

        // b_dc
        assert!((mna.b_dc[0] - 0.0).abs() < 1e-12);
        assert!((mna.b_dc[1] - 0.0).abs() < 1e-12);
        assert!((mna.b_dc[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_rc_circuit() {
        // V1(1V AC) -> node "1" -> R(100) -> node "2" -> C(1e-6) -> GND
        // Nodes: "1" -> 0, "2" -> 1
        // Branches: V1 -> branch 0, matrix row 2
        // Size = 3
        //
        // G matrix (3x3):
        //   Row 0 (node 1): G(0,0)=1/100=0.01, G(0,1)=-0.01, G(0,2)=1
        //   Row 1 (node 2): G(1,0)=-0.01, G(1,1)=0.01
        //   Row 2 (V1): G(2,0)=1
        //
        // C matrix (3x3):
        //   Row 1 (node 2): C(1,1) = 1e-6
        //
        // b_ac = [0, 0, 1+0j]
        let circuit = circuit_with(vec![
            Component::VSource {
                name: "V1".into(),
                nodes: ("1".into(), "0".into()),
                dc: None,
                ac: Some((1.0, 0.0)),
            },
            Component::Resistor {
                name: "R1".into(),
                nodes: ("1".into(), "2".into()),
                value: 100.0,
            },
            Component::Capacitor {
                name: "C1".into(),
                nodes: ("2".into(), "0".into()),
                value: 1e-6,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 3);

        let g = mna.g.to_dense();
        assert!((g[0][0] - 0.01).abs() < 1e-12);
        assert!((g[0][1] - (-0.01)).abs() < 1e-12);
        assert!((g[0][2] - 1.0).abs() < 1e-12);
        assert!((g[1][0] - (-0.01)).abs() < 1e-12);
        assert!((g[1][1] - 0.01).abs() < 1e-12);
        assert!((g[2][0] - 1.0).abs() < 1e-12);

        let c = mna.c.to_dense();
        assert!((c[1][1] - 1e-6).abs() < 1e-18);
        // All other C entries should be zero
        assert!((c[0][0]).abs() < 1e-18);
        assert!((c[0][1]).abs() < 1e-18);

        // AC excitation
        assert!((mna.b_ac[2] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_inductor_stamps() {
        // L1 between node "1" and "2", 1mH
        // Nodes: "1" -> 0, "2" -> 1
        // Branch: L1 -> branch 0, matrix row 2
        // Size = 3
        //
        // G stamps: G(0,2)=1, G(2,0)=1, G(1,2)=-1, G(2,1)=-1
        // C stamps: C(2,2) = -0.001
        let circuit = circuit_with(vec![Component::Inductor {
            name: "L1".into(),
            nodes: ("1".into(), "2".into()),
            value: 0.001,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 3);
        assert_eq!(mna.branch_names, vec!["L1"]);

        let g = mna.g.to_dense();
        assert!((g[0][2] - 1.0).abs() < 1e-12);
        assert!((g[2][0] - 1.0).abs() < 1e-12);
        assert!((g[1][2] - (-1.0)).abs() < 1e-12);
        assert!((g[2][1] - (-1.0)).abs() < 1e-12);
        // Diagonal should be zero
        assert!((g[0][0]).abs() < 1e-12);
        assert!((g[1][1]).abs() < 1e-12);
        assert!((g[2][2]).abs() < 1e-12);

        let c = mna.c.to_dense();
        assert!((c[2][2] - (-0.001)).abs() < 1e-12);
    }

    #[test]
    fn test_current_source() {
        // I1: 2A DC from node "1" to GND (n+=1, n-=0)
        // Node "1" -> idx 0
        // Size = 1
        // b_dc(0) -= 2.0 => b_dc = [-2.0]
        let circuit = circuit_with(vec![Component::ISource {
            name: "I1".into(),
            nodes: ("1".into(), "0".into()),
            dc: Some(2.0),
            ac: None,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 1);
        assert!((mna.b_dc[0] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_current_source_between_non_ground_nodes() {
        // I1: 3A DC from node "1" to node "2"
        // Nodes: "1" -> 0, "2" -> 1
        // b_dc(0) -= 3.0, b_dc(1) += 3.0
        let circuit = circuit_with(vec![Component::ISource {
            name: "I1".into(),
            nodes: ("1".into(), "2".into()),
            dc: Some(3.0),
            ac: Some((1.0, 90.0)),
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 2);
        assert!((mna.b_dc[0] - (-3.0)).abs() < 1e-12);
        assert!((mna.b_dc[1] - 3.0).abs() < 1e-12);

        // AC: 1.0 at 90 degrees = j
        let expected_ac = ac_to_complex(1.0, 90.0);
        assert!((mna.b_ac[0] - (-expected_ac)).norm() < 1e-12);
        assert!((mna.b_ac[1] - expected_ac).norm() < 1e-12);
    }

    #[test]
    fn test_ground_variants() {
        // Test that "0", "GND", "gnd", "Gnd" are all ground
        let circuit = circuit_with(vec![
            Component::Resistor {
                name: "R1".into(),
                nodes: ("1".into(), "0".into()),
                value: 100.0,
            },
            Component::Resistor {
                name: "R2".into(),
                nodes: ("2".into(), "GND".into()),
                value: 100.0,
            },
            Component::Resistor {
                name: "R3".into(),
                nodes: ("3".into(), "gnd".into()),
                value: 100.0,
            },
            Component::Resistor {
                name: "R4".into(),
                nodes: ("4".into(), "Gnd".into()),
                value: 100.0,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        // All 4 non-ground nodes, no branches
        assert_eq!(mna.size, 4);
        assert_eq!(mna.node_names.len(), 4);
        // None of the ground variants should appear in node_names
        for name in &mna.node_names {
            assert!(!is_ground(name));
        }
    }

    #[test]
    fn test_two_resistors_in_series() {
        // R1(100) between "1" and "2", R2(200) between "2" and "0"
        // Nodes: "1" -> 0, "2" -> 1
        // Size = 2
        //
        // G(0,0) = 1/100 = 0.01
        // G(0,1) = -1/100 = -0.01
        // G(1,0) = -1/100 = -0.01
        // G(1,1) = 1/100 + 1/200 = 0.01 + 0.005 = 0.015
        let circuit = circuit_with(vec![
            Component::Resistor {
                name: "R1".into(),
                nodes: ("1".into(), "2".into()),
                value: 100.0,
            },
            Component::Resistor {
                name: "R2".into(),
                nodes: ("2".into(), "0".into()),
                value: 200.0,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 2);

        let g = mna.g.to_dense();
        assert!((g[0][0] - 0.01).abs() < 1e-12);
        assert!((g[0][1] - (-0.01)).abs() < 1e-12);
        assert!((g[1][0] - (-0.01)).abs() < 1e-12);
        assert!((g[1][1] - 0.015).abs() < 1e-12);
    }

    #[test]
    fn test_vsource_with_ac() {
        // V1 between "1" and "0", DC=10V, AC=(5.0, 45.0)
        let circuit = circuit_with(vec![Component::VSource {
            name: "V1".into(),
            nodes: ("1".into(), "0".into()),
            dc: Some(10.0),
            ac: Some((5.0, 45.0)),
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 2); // 1 node + 1 branch
        assert!((mna.b_dc[1] - 10.0).abs() < 1e-12);

        let expected = ac_to_complex(5.0, 45.0);
        assert!((mna.b_ac[1] - expected).norm() < 1e-12);
    }

    #[test]
    fn test_vsource_negative_terminal_non_ground() {
        // V1 between "1" (n+) and "2" (n-)
        // Both are non-ground, so both get coupling stamps.
        // G(0, 2) = 1, G(2, 0) = 1, G(1, 2) = -1, G(2, 1) = -1
        let circuit = circuit_with(vec![Component::VSource {
            name: "V1".into(),
            nodes: ("1".into(), "2".into()),
            dc: Some(3.0),
            ac: None,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 3); // 2 nodes + 1 branch

        let g = mna.g.to_dense();
        assert!((g[0][2] - 1.0).abs() < 1e-12);
        assert!((g[2][0] - 1.0).abs() < 1e-12);
        assert!((g[1][2] - (-1.0)).abs() < 1e-12);
        assert!((g[2][1] - (-1.0)).abs() < 1e-12);
        assert!((mna.b_dc[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = circuit_with(vec![]);
        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 0);
        assert!(mna.node_names.is_empty());
        assert!(mna.branch_names.is_empty());
        assert!(mna.b_dc.is_empty());
        assert!(mna.b_ac.is_empty());
    }

    #[test]
    fn test_zero_resistance_error() {
        let circuit = circuit_with(vec![Component::Resistor {
            name: "R1".into(),
            nodes: ("1".into(), "0".into()),
            value: 0.0,
        }]);

        let result = compile(&circuit);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_rlc_circuit() {
        // V1(1V DC, 1V AC@0deg) -> node "1" -> R(100) -> node "2" -> L(1mH) -> node "3" -> C(1uF) -> GND
        // Nodes: "1"->0, "2"->1, "3"->2
        // Branches: V1->branch 0 (row 3), L1->branch 1 (row 4)
        // Size = 5
        let circuit = circuit_with(vec![
            Component::VSource {
                name: "V1".into(),
                nodes: ("1".into(), "0".into()),
                dc: Some(1.0),
                ac: Some((1.0, 0.0)),
            },
            Component::Resistor {
                name: "R1".into(),
                nodes: ("1".into(), "2".into()),
                value: 100.0,
            },
            Component::Inductor {
                name: "L1".into(),
                nodes: ("2".into(), "3".into()),
                value: 0.001,
            },
            Component::Capacitor {
                name: "C1".into(),
                nodes: ("3".into(), "0".into()),
                value: 1e-6,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 5);
        assert_eq!(mna.node_names, vec!["1", "2", "3"]);
        assert_eq!(mna.branch_names, vec!["V1", "L1"]);

        let g = mna.g.to_dense();
        // Node 1 (row 0): R1 stamp + V1 coupling
        assert!((g[0][0] - 0.01).abs() < 1e-12); // 1/100
        assert!((g[0][1] - (-0.01)).abs() < 1e-12); // -1/100
        assert!((g[0][3] - 1.0).abs() < 1e-12); // V1 coupling

        // Node 2 (row 1): R1 stamp + L1 coupling
        assert!((g[1][0] - (-0.01)).abs() < 1e-12); // -1/100
        assert!((g[1][1] - 0.01).abs() < 1e-12); // 1/100
        assert!((g[1][4] - 1.0).abs() < 1e-12); // L1 coupling

        // Node 3 (row 2): L1 coupling
        assert!((g[2][4] - (-1.0)).abs() < 1e-12); // L1 coupling (negative terminal)

        // V1 branch (row 3)
        assert!((g[3][0] - 1.0).abs() < 1e-12);

        // L1 branch (row 4)
        assert!((g[4][1] - 1.0).abs() < 1e-12);
        assert!((g[4][2] - (-1.0)).abs() < 1e-12);

        // C matrix: capacitor stamp + inductor stamp
        let c = mna.c.to_dense();
        assert!((c[2][2] - 1e-6).abs() < 1e-18); // C1 at node 3
        assert!((c[4][4] - (-0.001)).abs() < 1e-12); // -L for inductor branch

        // Excitation vectors
        assert!((mna.b_dc[3] - 1.0).abs() < 1e-12);
        assert!((mna.b_ac[3] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_ac_phase_conversion() {
        // 45 degrees: mag * (cos(45) + j*sin(45)) = mag * (sqrt2/2 + j*sqrt2/2)
        let c = ac_to_complex(2.0, 45.0);
        let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((c.re - 2.0 * sqrt2_over_2).abs() < 1e-12);
        assert!((c.im - 2.0 * sqrt2_over_2).abs() < 1e-12);

        // 90 degrees: purely imaginary
        let c90 = ac_to_complex(1.0, 90.0);
        assert!((c90.re).abs() < 1e-12);
        assert!((c90.im - 1.0).abs() < 1e-12);

        // 0 degrees: purely real
        let c0 = ac_to_complex(3.0, 0.0);
        assert!((c0.re - 3.0).abs() < 1e-12);
        assert!((c0.im).abs() < 1e-12);
    }

    #[test]
    fn test_multiple_vsources() {
        // V1 between "1" and "0", V2 between "2" and "0"
        // Each gets its own branch variable
        let circuit = circuit_with(vec![
            Component::VSource {
                name: "V1".into(),
                nodes: ("1".into(), "0".into()),
                dc: Some(5.0),
                ac: None,
            },
            Component::VSource {
                name: "V2".into(),
                nodes: ("2".into(), "0".into()),
                dc: Some(3.0),
                ac: None,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 4); // 2 nodes + 2 branches
        assert_eq!(mna.branch_names, vec!["V1", "V2"]);

        // b_dc: V1 at row 2, V2 at row 3
        assert!((mna.b_dc[2] - 5.0).abs() < 1e-12);
        assert!((mna.b_dc[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_parallel_resistors() {
        // R1(100) and R2(200) both between "1" and "0"
        // G(0,0) = 1/100 + 1/200 = 0.015
        let circuit = circuit_with(vec![
            Component::Resistor {
                name: "R1".into(),
                nodes: ("1".into(), "0".into()),
                value: 100.0,
            },
            Component::Resistor {
                name: "R2".into(),
                nodes: ("1".into(), "0".into()),
                value: 200.0,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 1);

        let g = mna.g.to_dense();
        assert!((g[0][0] - 0.015).abs() < 1e-12);
    }
}
