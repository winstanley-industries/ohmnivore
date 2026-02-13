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

/// Thermal voltage at 300K: kT/q ~ 0.02585 V
const VT_300K: f64 = 0.02585;

/// Minimum conductance from every node to ground (standard SPICE GMIN).
/// Prevents singular matrices when nodes are only connected through off transistors.
const GMIN: f64 = 1e-12;

/// GPU-friendly descriptor for a single diode, matching the WGSL shader layout.
///
/// 12 words (u32) per diode, matching the flat buffer layout expected by the
/// diode evaluation and nonlinear assembly shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDiodeDescriptor {
    /// Index of the anode node in the solution vector.
    /// `u32::MAX` if the anode is ground.
    pub anode_idx: u32,
    /// Index of the cathode node in the solution vector.
    /// `u32::MAX` if the cathode is ground.
    pub cathode_idx: u32,
    /// Saturation current I_s (f32 bitcast).
    pub is_val: f32,
    /// n * V_T product (f32 bitcast).
    pub n_vt: f32,
    /// CSR value-array indices for the 4 G-matrix stamp positions:
    /// [0] = (anode, anode), [1] = (cathode, cathode),
    /// [2] = (anode, cathode), [3] = (cathode, anode).
    /// `u32::MAX` sentinel for ground-related entries.
    pub g_row_col: [u32; 4],
    /// RHS vector indices: [0] = anode, [1] = cathode.
    /// `u32::MAX` sentinel for ground.
    pub b_idx: [u32; 2],
    /// Padding to align to 12 u32 words (48 bytes).
    pub _padding: [u32; 2],
}

/// GPU-friendly descriptor for a single BJT, matching the WGSL shader layout.
///
/// 24 words (u32) per BJT = 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBjtDescriptor {
    /// Index of the collector node in the solution vector.
    /// `u32::MAX` if the collector is ground.
    pub collector_idx: u32,
    /// Index of the base node in the solution vector.
    /// `u32::MAX` if the base is ground.
    pub base_idx: u32,
    /// Index of the emitter node in the solution vector.
    /// `u32::MAX` if the emitter is ground.
    pub emitter_idx: u32,
    /// +1.0 for NPN, -1.0 for PNP (f32 bitcast).
    pub polarity: f32,
    /// Saturation current I_s (f32 bitcast).
    pub is_val: f32,
    /// Forward current gain (f32 bitcast).
    pub bf: f32,
    /// Reverse current gain (f32 bitcast).
    pub br: f32,
    /// NF * V_T product (f32 bitcast).
    pub nf_vt: f32,
    /// NR * V_T product (f32 bitcast).
    pub nr_vt: f32,
    /// Per-node voltage-source-constrained flags: [collector, base, emitter].
    /// 1 if the node is directly driven by a voltage source, 0 otherwise.
    /// The voltage limiting shader skips modifying constrained nodes.
    pub node_fixed: [u32; 3],
    /// CSR value-array indices for the 9 G-matrix stamp positions:
    /// [CC, CB, CE, BC, BB, BE, EC, EB, EE].
    /// `u32::MAX` sentinel for ground-related entries.
    pub g_row_col: [u32; 9],
    /// RHS vector indices: [C, B, E].
    /// `u32::MAX` sentinel for ground.
    pub b_idx: [u32; 3],
}

/// GPU-friendly descriptor for a single MOSFET, matching the WGSL shader layout.
///
/// 16 words (u32) per MOSFET = 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMosfetDescriptor {
    /// Index of the drain node in the solution vector.
    /// `u32::MAX` if the drain is ground.
    pub drain_idx: u32,
    /// Index of the gate node in the solution vector.
    /// `u32::MAX` if the gate is ground.
    pub gate_idx: u32,
    /// Index of the source node in the solution vector.
    /// `u32::MAX` if the source is ground.
    pub source_idx: u32,
    /// +1.0 for NMOS, -1.0 for PMOS (f32 bitcast).
    pub polarity: f32,
    /// Threshold voltage (f32 bitcast).
    pub vto: f32,
    /// Transconductance parameter (f32 bitcast).
    pub kp: f32,
    /// Channel-length modulation (f32 bitcast).
    pub lambda: f32,
    /// Packed per-node voltage-source-constrained flags.
    /// bit 0 = drain, bit 1 = gate, bit 2 = source.
    /// The voltage limiting shader skips modifying constrained nodes.
    pub node_fixed_packed: u32,
    /// CSR value-array indices for the 6 G-matrix stamp positions:
    /// [DD, DG, DS, SD, SG, SS].
    /// `u32::MAX` sentinel for ground-related entries.
    pub g_row_col: [u32; 6],
    /// RHS vector indices: [D, S].
    /// `u32::MAX` sentinel for ground.
    pub b_idx: [u32; 2],
}

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
    /// Diode descriptors for nonlinear Newton-Raphson solving on GPU.
    pub diode_descriptors: Vec<GpuDiodeDescriptor>,
    /// BJT descriptors for nonlinear Newton-Raphson solving on GPU.
    pub bjt_descriptors: Vec<GpuBjtDescriptor>,
    /// MOSFET descriptors for nonlinear Newton-Raphson solving on GPU.
    pub mosfet_descriptors: Vec<GpuMosfetDescriptor>,
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
        match component {
            Component::Bjt { nodes, .. } => {
                register_node(&nodes[0], &mut node_map, &mut node_names);
                register_node(&nodes[1], &mut node_map, &mut node_names);
                register_node(&nodes[2], &mut node_map, &mut node_names);
            }
            Component::Mosfet { nodes, .. } => {
                register_node(&nodes[0], &mut node_map, &mut node_names);
                register_node(&nodes[1], &mut node_map, &mut node_names);
                register_node(&nodes[2], &mut node_map, &mut node_names);
            }
            Component::Resistor { nodes, .. }
            | Component::Capacitor { nodes, .. }
            | Component::Inductor { nodes, .. }
            | Component::VSource { nodes, .. }
            | Component::ISource { nodes, .. }
            | Component::Diode { nodes, .. } => {
                register_node(&nodes.0, &mut node_map, &mut node_names);
                register_node(&nodes.1, &mut node_map, &mut node_names);
            }
        }
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

    /// Intermediate storage for diode info collected during stamping,
    /// resolved into GpuDiodeDescriptors after CSR construction.
    struct PendingDiode {
        anode: Option<usize>,
        cathode: Option<usize>,
        is_val: f32,
        n_vt: f32,
    }

    /// Intermediate storage for BJT info collected during stamping.
    struct PendingBjt {
        collector: Option<usize>,
        base: Option<usize>,
        emitter: Option<usize>,
        polarity: f32,
        is_val: f32,
        bf: f32,
        br: f32,
        nf_vt: f32,
        nr_vt: f32,
    }

    /// Intermediate storage for MOSFET info collected during stamping.
    struct PendingMosfet {
        drain: Option<usize>,
        gate: Option<usize>,
        source: Option<usize>,
        polarity: f32,
        vto: f32,
        kp: f32,
        lambda: f32,
    }

    let mut pending_diodes: Vec<PendingDiode> = Vec::new();
    let mut pending_bjts: Vec<PendingBjt> = Vec::new();
    let mut pending_mosfets: Vec<PendingMosfet> = Vec::new();

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
                tran: _,
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

            // Diodes are nonlinear — stamp placeholder zeros into G for sparsity
            // pattern so the GPU can write conductance stamps during Newton iteration.
            Component::Diode { name, nodes, model } => {
                // Look up the model by name
                let diode_model = circuit
                    .models
                    .iter()
                    .find(|m| m.name == *model)
                    .ok_or_else(|| {
                        OhmnivoreError::Compile(format!(
                            "Diode {} references undefined model '{}'",
                            name, model
                        ))
                    })?;

                let ni = node_index(&nodes.0, &node_map);
                let nj = node_index(&nodes.1, &node_map);
                let n_vt = (diode_model.n * VT_300K) as f32;

                // Add placeholder zeros for 4 stamp positions
                if let Some(i) = ni {
                    g_triplets.push((i, i, 0.0)); // (anode, anode)
                }
                if let Some(j) = nj {
                    g_triplets.push((j, j, 0.0)); // (cathode, cathode)
                }
                if let (Some(i), Some(j)) = (ni, nj) {
                    g_triplets.push((i, j, 0.0)); // (anode, cathode)
                    g_triplets.push((j, i, 0.0)); // (cathode, anode)
                }

                // Store pending info for post-CSR descriptor construction
                pending_diodes.push(PendingDiode {
                    anode: ni,
                    cathode: nj,
                    is_val: diode_model.is as f32,
                    n_vt,
                });
            }

            Component::ISource {
                name: _,
                nodes,
                dc,
                ac,
                tran: _,
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

            // BJTs are nonlinear — stamp 9 placeholder zeros (3x3 block) into G.
            Component::Bjt { name, nodes, model } => {
                let bjt_model = circuit
                    .bjt_models
                    .iter()
                    .find(|m| m.name == *model)
                    .ok_or_else(|| {
                        OhmnivoreError::Compile(format!(
                            "BJT {} references undefined model '{}'",
                            name, model
                        ))
                    })?;

                let nc = node_index(&nodes[0], &node_map);
                let nb = node_index(&nodes[1], &node_map);
                let ne = node_index(&nodes[2], &node_map);

                // Stamp 3x3 placeholder zeros for all (row, col) pairs among {C, B, E}
                let node_indices = [nc, nb, ne];
                for &row in &node_indices {
                    for &col in &node_indices {
                        if let (Some(r), Some(c)) = (row, col) {
                            g_triplets.push((r, c, 0.0));
                        }
                    }
                }

                let nf_vt = (bjt_model.nf * VT_300K) as f32;
                let nr_vt = (bjt_model.nr * VT_300K) as f32;

                pending_bjts.push(PendingBjt {
                    collector: nc,
                    base: nb,
                    emitter: ne,
                    polarity: if bjt_model.is_npn { 1.0f32 } else { -1.0f32 },
                    is_val: bjt_model.is as f32,
                    bf: bjt_model.bf as f32,
                    br: bjt_model.br as f32,
                    nf_vt,
                    nr_vt,
                });
            }

            // MOSFETs are nonlinear — stamp 6 placeholder zeros (drain & source rows,
            // drain/gate/source columns, no gate row) into G.
            Component::Mosfet { name, nodes, model } => {
                let mosfet_model = circuit
                    .mosfet_models
                    .iter()
                    .find(|m| m.name == *model)
                    .ok_or_else(|| {
                        OhmnivoreError::Compile(format!(
                            "MOSFET {} references undefined model '{}'",
                            name, model
                        ))
                    })?;

                let nd = node_index(&nodes[0], &node_map);
                let ng = node_index(&nodes[1], &node_map);
                let ns = node_index(&nodes[2], &node_map);

                // Stamp 6 placeholder zeros: drain row (DD, DG, DS) + source row (SD, SG, SS)
                let row_nodes = [nd, ns];
                let col_nodes = [nd, ng, ns];
                for &row in &row_nodes {
                    for &col in &col_nodes {
                        if let (Some(r), Some(c)) = (row, col) {
                            g_triplets.push((r, c, 0.0));
                        }
                    }
                }

                pending_mosfets.push(PendingMosfet {
                    drain: nd,
                    gate: ng,
                    source: ns,
                    polarity: if mosfet_model.is_nmos {
                        1.0f32
                    } else {
                        -1.0f32
                    },
                    vto: mosfet_model.vto.abs() as f32,
                    kp: mosfet_model.kp as f32,
                    lambda: mosfet_model.lambda as f32,
                });
            }
        }
    }

    // GMIN: add tiny conductance from every node to ground to prevent singular matrices.
    for i in 0..n_nodes {
        g_triplets.push((i, i, GMIN));
    }

    // Ensure G has entries at all positions where C has entries.
    // This guarantees the companion matrix A = G + alpha*C/h shares G's sparsity pattern.
    for &(row, col, _) in &c_triplets {
        g_triplets.push((row, col, 0.0));
    }

    // Step 4: Assemble CSR matrices.
    let g = CsrMatrix::from_triplets(size, size, &g_triplets);
    let c = CsrMatrix::from_triplets(size, size, &c_triplets);

    // Build set of voltage-source-constrained node indices.
    // The voltage limiting shaders must not modify these nodes.
    let mut vsource_constrained = std::collections::HashSet::new();
    for component in &circuit.components {
        if let Component::VSource { nodes, .. } = component {
            if let Some(i) = node_index(&nodes.0, &node_map) {
                vsource_constrained.insert(i);
            }
            if let Some(i) = node_index(&nodes.1, &node_map) {
                vsource_constrained.insert(i);
            }
        }
    }

    // Step 5: Resolve CSR value indices for diode descriptors.
    const SENTINEL: u32 = u32::MAX;

    let diode_descriptors: Vec<GpuDiodeDescriptor> = pending_diodes
        .iter()
        .map(|pd| {
            let anode_idx = pd.anode.map(|i| i as u32).unwrap_or(SENTINEL);
            let cathode_idx = pd.cathode.map(|j| j as u32).unwrap_or(SENTINEL);

            // Resolve g_row_col: [aa, cc, ac, ca]
            let g_aa = pd
                .anode
                .and_then(|i| g.value_index(i, i))
                .map(|v| v as u32)
                .unwrap_or(SENTINEL);
            let g_cc = pd
                .cathode
                .and_then(|j| g.value_index(j, j))
                .map(|v| v as u32)
                .unwrap_or(SENTINEL);
            let g_ac = pd
                .anode
                .zip(pd.cathode)
                .and_then(|(i, j)| g.value_index(i, j))
                .map(|v| v as u32)
                .unwrap_or(SENTINEL);
            let g_ca = pd
                .anode
                .zip(pd.cathode)
                .and_then(|(i, j)| g.value_index(j, i))
                .map(|v| v as u32)
                .unwrap_or(SENTINEL);

            GpuDiodeDescriptor {
                anode_idx,
                cathode_idx,
                is_val: pd.is_val,
                n_vt: pd.n_vt,
                g_row_col: [g_aa, g_cc, g_ac, g_ca],
                b_idx: [anode_idx, cathode_idx],
                _padding: [0; 2],
            }
        })
        .collect();

    // Step 6: Resolve CSR value indices for BJT descriptors.
    let bjt_descriptors: Vec<GpuBjtDescriptor> = pending_bjts
        .iter()
        .map(|pb| {
            let collector_idx = pb.collector.map(|i| i as u32).unwrap_or(SENTINEL);
            let base_idx = pb.base.map(|i| i as u32).unwrap_or(SENTINEL);
            let emitter_idx = pb.emitter.map(|i| i as u32).unwrap_or(SENTINEL);

            // Resolve 9 G-matrix positions: [CC, CB, CE, BC, BB, BE, EC, EB, EE]
            let nodes = [pb.collector, pb.base, pb.emitter];
            let mut g_row_col = [SENTINEL; 9];
            for (ri, &row) in nodes.iter().enumerate() {
                for (ci, &col) in nodes.iter().enumerate() {
                    let idx = ri * 3 + ci;
                    g_row_col[idx] = row
                        .zip(col)
                        .and_then(|(r, c)| g.value_index(r, c))
                        .map(|v| v as u32)
                        .unwrap_or(SENTINEL);
                }
            }

            let node_fixed = [
                pb.collector
                    .map_or(0, |i| vsource_constrained.contains(&i) as u32),
                pb.base
                    .map_or(0, |i| vsource_constrained.contains(&i) as u32),
                pb.emitter
                    .map_or(0, |i| vsource_constrained.contains(&i) as u32),
            ];

            GpuBjtDescriptor {
                collector_idx,
                base_idx,
                emitter_idx,
                polarity: pb.polarity,
                is_val: pb.is_val,
                bf: pb.bf,
                br: pb.br,
                nf_vt: pb.nf_vt,
                nr_vt: pb.nr_vt,
                node_fixed,
                g_row_col,
                b_idx: [collector_idx, base_idx, emitter_idx],
            }
        })
        .collect();

    // Step 7: Resolve CSR value indices for MOSFET descriptors.
    let mosfet_descriptors: Vec<GpuMosfetDescriptor> = pending_mosfets
        .iter()
        .map(|pm| {
            let drain_idx = pm.drain.map(|i| i as u32).unwrap_or(SENTINEL);
            let gate_idx = pm.gate.map(|i| i as u32).unwrap_or(SENTINEL);
            let source_idx = pm.source.map(|i| i as u32).unwrap_or(SENTINEL);

            // Resolve 6 G-matrix positions: [DD, DG, DS, SD, SG, SS]
            let rows = [pm.drain, pm.source];
            let cols = [pm.drain, pm.gate, pm.source];
            let mut g_row_col = [SENTINEL; 6];
            for (ri, &row) in rows.iter().enumerate() {
                for (ci, &col) in cols.iter().enumerate() {
                    let idx = ri * 3 + ci;
                    g_row_col[idx] = row
                        .zip(col)
                        .and_then(|(r, c)| g.value_index(r, c))
                        .map(|v| v as u32)
                        .unwrap_or(SENTINEL);
                }
            }

            let node_fixed_packed = (pm
                .drain
                .map_or(0, |i| vsource_constrained.contains(&i) as u32))
                | (pm
                    .gate
                    .map_or(0, |i| vsource_constrained.contains(&i) as u32)
                    << 1)
                | (pm
                    .source
                    .map_or(0, |i| vsource_constrained.contains(&i) as u32)
                    << 2);

            GpuMosfetDescriptor {
                drain_idx,
                gate_idx,
                source_idx,
                polarity: pm.polarity,
                vto: pm.vto,
                kp: pm.kp,
                lambda: pm.lambda,
                node_fixed_packed,
                g_row_col,
                b_idx: [drain_idx, source_idx],
            }
        })
        .collect();

    Ok(MnaSystem {
        g,
        c,
        b_dc,
        b_ac,
        size,
        node_names,
        branch_names,
        diode_descriptors,
        bjt_descriptors,
        mosfet_descriptors,
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
            models: vec![],
            bjt_models: vec![],
            mosfet_models: vec![],
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
        assert!((g_dense[0][0] - 0.001).abs() < 1e-9);
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
                tran: None,
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
        assert!((g[0][0] - 0.001).abs() < 1e-9);
        assert!((g[0][1] - (-0.001)).abs() < 1e-9);
        assert!((g[0][2] - 1.0).abs() < 1e-9);

        // Row 1: [-0.001, 0.002, 0.0]
        assert!((g[1][0] - (-0.001)).abs() < 1e-9);
        assert!((g[1][1] - 0.002).abs() < 1e-9);
        assert!((g[1][2] - 0.0).abs() < 1e-9);

        // Row 2: [1.0, 0.0, 0.0]
        assert!((g[2][0] - 1.0).abs() < 1e-9);
        assert!((g[2][1] - 0.0).abs() < 1e-9);
        assert!((g[2][2] - 0.0).abs() < 1e-9);

        // b_dc
        assert!((mna.b_dc[0] - 0.0).abs() < 1e-9);
        assert!((mna.b_dc[1] - 0.0).abs() < 1e-9);
        assert!((mna.b_dc[2] - 5.0).abs() < 1e-9);
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
                tran: None,
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
        assert!((g[0][0] - 0.01).abs() < 1e-9);
        assert!((g[0][1] - (-0.01)).abs() < 1e-9);
        assert!((g[0][2] - 1.0).abs() < 1e-9);
        assert!((g[1][0] - (-0.01)).abs() < 1e-9);
        assert!((g[1][1] - 0.01).abs() < 1e-9);
        assert!((g[2][0] - 1.0).abs() < 1e-9);

        let c = mna.c.to_dense();
        assert!((c[1][1] - 1e-6).abs() < 1e-18);
        // All other C entries should be zero
        assert!((c[0][0]).abs() < 1e-18);
        assert!((c[0][1]).abs() < 1e-18);

        // AC excitation
        assert!((mna.b_ac[2] - Complex64::new(1.0, 0.0)).norm() < 1e-9);
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
        assert!((g[0][2] - 1.0).abs() < 1e-9);
        assert!((g[2][0] - 1.0).abs() < 1e-9);
        assert!((g[1][2] - (-1.0)).abs() < 1e-9);
        assert!((g[2][1] - (-1.0)).abs() < 1e-9);
        // Diagonal should be zero
        assert!((g[0][0]).abs() < 1e-9);
        assert!((g[1][1]).abs() < 1e-9);
        assert!((g[2][2]).abs() < 1e-9);

        let c = mna.c.to_dense();
        assert!((c[2][2] - (-0.001)).abs() < 1e-9);
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
            tran: None,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 1);
        assert!((mna.b_dc[0] - (-2.0)).abs() < 1e-9);
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
            tran: None,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 2);
        assert!((mna.b_dc[0] - (-3.0)).abs() < 1e-9);
        assert!((mna.b_dc[1] - 3.0).abs() < 1e-9);

        // AC: 1.0 at 90 degrees = j
        let expected_ac = ac_to_complex(1.0, 90.0);
        assert!((mna.b_ac[0] - (-expected_ac)).norm() < 1e-9);
        assert!((mna.b_ac[1] - expected_ac).norm() < 1e-9);
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
        assert!((g[0][0] - 0.01).abs() < 1e-9);
        assert!((g[0][1] - (-0.01)).abs() < 1e-9);
        assert!((g[1][0] - (-0.01)).abs() < 1e-9);
        assert!((g[1][1] - 0.015).abs() < 1e-9);
    }

    #[test]
    fn test_vsource_with_ac() {
        // V1 between "1" and "0", DC=10V, AC=(5.0, 45.0)
        let circuit = circuit_with(vec![Component::VSource {
            name: "V1".into(),
            nodes: ("1".into(), "0".into()),
            dc: Some(10.0),
            ac: Some((5.0, 45.0)),
            tran: None,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 2); // 1 node + 1 branch
        assert!((mna.b_dc[1] - 10.0).abs() < 1e-9);

        let expected = ac_to_complex(5.0, 45.0);
        assert!((mna.b_ac[1] - expected).norm() < 1e-9);
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
            tran: None,
        }]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 3); // 2 nodes + 1 branch

        let g = mna.g.to_dense();
        assert!((g[0][2] - 1.0).abs() < 1e-9);
        assert!((g[2][0] - 1.0).abs() < 1e-9);
        assert!((g[1][2] - (-1.0)).abs() < 1e-9);
        assert!((g[2][1] - (-1.0)).abs() < 1e-9);
        assert!((mna.b_dc[2] - 3.0).abs() < 1e-9);
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
                tran: None,
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
        assert!((g[0][0] - 0.01).abs() < 1e-9); // 1/100
        assert!((g[0][1] - (-0.01)).abs() < 1e-9); // -1/100
        assert!((g[0][3] - 1.0).abs() < 1e-9); // V1 coupling

        // Node 2 (row 1): R1 stamp + L1 coupling
        assert!((g[1][0] - (-0.01)).abs() < 1e-9); // -1/100
        assert!((g[1][1] - 0.01).abs() < 1e-9); // 1/100
        assert!((g[1][4] - 1.0).abs() < 1e-9); // L1 coupling

        // Node 3 (row 2): L1 coupling
        assert!((g[2][4] - (-1.0)).abs() < 1e-9); // L1 coupling (negative terminal)

        // V1 branch (row 3)
        assert!((g[3][0] - 1.0).abs() < 1e-9);

        // L1 branch (row 4)
        assert!((g[4][1] - 1.0).abs() < 1e-9);
        assert!((g[4][2] - (-1.0)).abs() < 1e-9);

        // C matrix: capacitor stamp + inductor stamp
        let c = mna.c.to_dense();
        assert!((c[2][2] - 1e-6).abs() < 1e-18); // C1 at node 3
        assert!((c[4][4] - (-0.001)).abs() < 1e-9); // -L for inductor branch

        // Excitation vectors
        assert!((mna.b_dc[3] - 1.0).abs() < 1e-9);
        assert!((mna.b_ac[3] - Complex64::new(1.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn test_ac_phase_conversion() {
        // 45 degrees: mag * (cos(45) + j*sin(45)) = mag * (sqrt2/2 + j*sqrt2/2)
        let c = ac_to_complex(2.0, 45.0);
        let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((c.re - 2.0 * sqrt2_over_2).abs() < 1e-9);
        assert!((c.im - 2.0 * sqrt2_over_2).abs() < 1e-9);

        // 90 degrees: purely imaginary
        let c90 = ac_to_complex(1.0, 90.0);
        assert!((c90.re).abs() < 1e-9);
        assert!((c90.im - 1.0).abs() < 1e-9);

        // 0 degrees: purely real
        let c0 = ac_to_complex(3.0, 0.0);
        assert!((c0.re - 3.0).abs() < 1e-9);
        assert!((c0.im).abs() < 1e-9);
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
                tran: None,
            },
            Component::VSource {
                name: "V2".into(),
                nodes: ("2".into(), "0".into()),
                dc: Some(3.0),
                ac: None,
                tran: None,
            },
        ]);

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.size, 4); // 2 nodes + 2 branches
        assert_eq!(mna.branch_names, vec!["V1", "V2"]);

        // b_dc: V1 at row 2, V2 at row 3
        assert!((mna.b_dc[2] - 5.0).abs() < 1e-9);
        assert!((mna.b_dc[3] - 3.0).abs() < 1e-9);
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
        assert!((g[0][0] - 0.015).abs() < 1e-9);
    }

    // --- Diode tests ---

    use crate::ir::DiodeModel;

    /// Helper: create a circuit with components and models.
    fn circuit_with_models(components: Vec<Component>, models: Vec<DiodeModel>) -> Circuit {
        Circuit {
            components,
            analyses: vec![],
            models,
            bjt_models: vec![],
            mosfet_models: vec![],
        }
    }

    #[test]
    fn test_single_diode_descriptor() {
        // V1(1V) -> node "1", D1 from "1" to "0" using model DMOD
        // Nodes: "1" -> idx 0
        // Size = 2 (1 node + 1 branch for V1)
        let circuit = circuit_with_models(
            vec![
                Component::VSource {
                    name: "V1".into(),
                    nodes: ("1".into(), "0".into()),
                    dc: Some(1.0),
                    ac: None,
                    tran: None,
                },
                Component::Diode {
                    name: "D1".into(),
                    nodes: ("1".into(), "0".into()),
                    model: "DMOD".into(),
                },
            ],
            vec![DiodeModel {
                name: "DMOD".into(),
                is: 1e-14,
                n: 1.0,
            }],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.diode_descriptors.len(), 1);

        let desc = &mna.diode_descriptors[0];
        // Anode is node "1" -> idx 0
        assert_eq!(desc.anode_idx, 0);
        // Cathode is ground -> sentinel
        assert_eq!(desc.cathode_idx, u32::MAX);
        // IS
        assert!((desc.is_val - 1e-14_f32).abs() < 1e-20);
        // n_vt = 1.0 * 0.02585
        let expected_n_vt = (1.0 * 0.02585) as f32;
        assert!((desc.n_vt - expected_n_vt).abs() < 1e-6);

        // G matrix should have a placeholder zero at (0,0) from the diode
        // (summed with V1 stamps). The sparsity pattern includes (0,0).
        assert!(mna.g.value_index(0, 0).is_some());
        // g_row_col: aa should be valid, cc/ac/ca should be sentinels (ground)
        assert_ne!(desc.g_row_col[0], u32::MAX); // (anode, anode) exists
        assert_eq!(desc.g_row_col[1], u32::MAX); // (cathode, cathode) = ground
        assert_eq!(desc.g_row_col[2], u32::MAX); // (anode, cathode) = ground
        assert_eq!(desc.g_row_col[3], u32::MAX); // (cathode, anode) = ground

        // b_idx
        assert_eq!(desc.b_idx[0], 0); // anode
        assert_eq!(desc.b_idx[1], u32::MAX); // cathode = ground
    }

    #[test]
    fn test_diode_between_non_ground_nodes() {
        // D1 from node "1" (anode) to node "2" (cathode), both non-ground.
        // R1 to provide a path so nodes exist.
        let circuit = circuit_with_models(
            vec![
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("2".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Diode {
                    name: "D1".into(),
                    nodes: ("1".into(), "2".into()),
                    model: "DMOD".into(),
                },
            ],
            vec![DiodeModel {
                name: "DMOD".into(),
                is: 1e-14,
                n: 1.5,
            }],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.diode_descriptors.len(), 1);

        let desc = &mna.diode_descriptors[0];
        // Nodes: "1"->0, "2"->1
        assert_eq!(desc.anode_idx, 0);
        assert_eq!(desc.cathode_idx, 1);

        let expected_n_vt = (1.5 * 0.02585) as f32;
        assert!((desc.n_vt - expected_n_vt).abs() < 1e-6);

        // All 4 stamp positions should be valid (non-sentinel)
        for i in 0..4 {
            assert_ne!(
                desc.g_row_col[i],
                u32::MAX,
                "g_row_col[{i}] should not be sentinel"
            );
        }

        // G matrix should contain placeholders at all 4 positions
        let g = mna.g.to_dense();
        // The diode stamps are zeros; resistor stamps are non-zero on diagonals
        // (0,0) = 1/1000 + 0 = 0.001, (1,1) = 1/1000 + 0 = 0.001
        assert!((g[0][0] - 0.001).abs() < 1e-9);
        assert!((g[1][1] - 0.001).abs() < 1e-9);
        // Off-diagonals from diode placeholder only: (0,1) = 0, (1,0) = 0
        assert!((g[0][1]).abs() < 1e-9);
        assert!((g[1][0]).abs() < 1e-9);

        // b_idx
        assert_eq!(desc.b_idx[0], 0);
        assert_eq!(desc.b_idx[1], 1);
    }

    #[test]
    fn test_multiple_diodes_shared_model() {
        // Two diodes sharing the same model
        let circuit = circuit_with_models(
            vec![
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "0".into()),
                    value: 100.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("2".into(), "0".into()),
                    value: 100.0,
                },
                Component::Diode {
                    name: "D1".into(),
                    nodes: ("1".into(), "0".into()),
                    model: "DMOD".into(),
                },
                Component::Diode {
                    name: "D2".into(),
                    nodes: ("2".into(), "0".into()),
                    model: "DMOD".into(),
                },
            ],
            vec![DiodeModel {
                name: "DMOD".into(),
                is: 1e-12,
                n: 2.0,
            }],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.diode_descriptors.len(), 2);

        // Both should have the same model parameters
        let expected_n_vt = (2.0 * 0.02585) as f32;
        for desc in &mna.diode_descriptors {
            assert!((desc.is_val - 1e-12_f32).abs() < 1e-18);
            assert!((desc.n_vt - expected_n_vt).abs() < 1e-6);
            assert_eq!(desc.cathode_idx, u32::MAX); // ground
        }

        // Different anode nodes
        assert_eq!(mna.diode_descriptors[0].anode_idx, 0);
        assert_eq!(mna.diode_descriptors[1].anode_idx, 1);
    }

    #[test]
    fn test_diode_undefined_model_error() {
        let circuit = circuit_with_models(
            vec![Component::Diode {
                name: "D1".into(),
                nodes: ("1".into(), "0".into()),
                model: "NOMODEL".into(),
            }],
            vec![], // no models defined
        );

        let result = compile(&circuit);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("undefined model"),
            "Error should mention undefined model, got: {err_msg}"
        );
    }

    #[test]
    fn test_diode_ground_cathode_csr_indices() {
        // D1 from "1" to ground. Only (anode,anode) stamp should exist.
        let circuit = circuit_with_models(
            vec![
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Diode {
                    name: "D1".into(),
                    nodes: ("1".into(), "0".into()),
                    model: "DMOD".into(),
                },
            ],
            vec![DiodeModel {
                name: "DMOD".into(),
                is: 1e-14,
                n: 1.0,
            }],
        );

        let mna = compile(&circuit).unwrap();
        let desc = &mna.diode_descriptors[0];

        // Only (anode, anode) should have a valid CSR index
        assert_ne!(desc.g_row_col[0], u32::MAX);
        assert_eq!(desc.g_row_col[1], u32::MAX);
        assert_eq!(desc.g_row_col[2], u32::MAX);
        assert_eq!(desc.g_row_col[3], u32::MAX);

        // Verify the CSR index is correct by checking the value is 0.0
        // (diode placeholder) summed with the resistor's 0.001
        let val_idx = desc.g_row_col[0] as usize;
        // The value at that CSR position should be the resistor's conductance
        // (since the diode adds 0.0)
        assert!((mna.g.values[val_idx] - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_diode_ground_anode() {
        // D1 from ground (anode) to "1" (cathode).
        let circuit = circuit_with_models(
            vec![
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Diode {
                    name: "D1".into(),
                    nodes: ("0".into(), "1".into()),
                    model: "DMOD".into(),
                },
            ],
            vec![DiodeModel {
                name: "DMOD".into(),
                is: 1e-14,
                n: 1.0,
            }],
        );

        let mna = compile(&circuit).unwrap();
        let desc = &mna.diode_descriptors[0];

        assert_eq!(desc.anode_idx, u32::MAX); // ground
        assert_eq!(desc.cathode_idx, 0); // node "1" -> idx 0

        // Only (cathode, cathode) should have a valid CSR index
        assert_eq!(desc.g_row_col[0], u32::MAX); // (anode, anode) = ground
        assert_ne!(desc.g_row_col[1], u32::MAX); // (cathode, cathode) valid
        assert_eq!(desc.g_row_col[2], u32::MAX); // (anode, cathode) = ground
        assert_eq!(desc.g_row_col[3], u32::MAX); // (cathode, anode) = ground
    }

    #[test]
    fn test_diode_descriptor_size() {
        // Verify GpuDiodeDescriptor is exactly 48 bytes (12 u32 words)
        assert_eq!(
            std::mem::size_of::<super::GpuDiodeDescriptor>(),
            48,
            "GpuDiodeDescriptor should be 48 bytes (12 x u32)"
        );
    }

    // --- BJT and MOSFET tests ---

    use crate::ir::{BjtModel, MosfetModel};

    /// Helper: create a circuit with components and all model types.
    fn circuit_with_all_models(
        components: Vec<Component>,
        diode_models: Vec<DiodeModel>,
        bjt_models: Vec<BjtModel>,
        mosfet_models: Vec<MosfetModel>,
    ) -> Circuit {
        Circuit {
            components,
            analyses: vec![],
            models: diode_models,
            bjt_models,
            mosfet_models,
        }
    }

    #[test]
    fn test_bjt_descriptor_size() {
        assert_eq!(
            std::mem::size_of::<super::GpuBjtDescriptor>(),
            96,
            "GpuBjtDescriptor should be 96 bytes (24 x u32)"
        );
    }

    #[test]
    fn test_mosfet_descriptor_size() {
        assert_eq!(
            std::mem::size_of::<super::GpuMosfetDescriptor>(),
            64,
            "GpuMosfetDescriptor should be 64 bytes (16 x u32)"
        );
    }

    #[test]
    fn test_single_bjt_descriptor() {
        // V1(5V) -> node "1", R1(10k) from "1" to "2" (base),
        // Q1 NPN: collector="3", base="2", emitter="4",
        // R2(1k) from "3" to "1" (collector load), R3(100) from "4" to "0" (emitter)
        let circuit = circuit_with_all_models(
            vec![
                Component::VSource {
                    name: "V1".into(),
                    nodes: ("1".into(), "0".into()),
                    dc: Some(5.0),
                    ac: None,
                    tran: None,
                },
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "2".into()),
                    value: 10000.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("1".into(), "3".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R3".into(),
                    nodes: ("4".into(), "0".into()),
                    value: 100.0,
                },
                Component::Bjt {
                    name: "Q1".into(),
                    nodes: ["3".into(), "2".into(), "4".into()],
                    model: "2N2222".into(),
                },
            ],
            vec![],
            vec![BjtModel {
                name: "2N2222".into(),
                is: 1e-14,
                bf: 200.0,
                br: 2.0,
                nf: 1.0,
                nr: 1.0,
                is_npn: true,
            }],
            vec![],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.bjt_descriptors.len(), 1);

        let desc = &mna.bjt_descriptors[0];
        // Nodes: "1"->0, "2"->1, "3"->2, "4"->3
        assert_eq!(desc.collector_idx, 2); // node "3"
        assert_eq!(desc.base_idx, 1); // node "2"
        assert_eq!(desc.emitter_idx, 3); // node "4"
        assert_eq!(desc.polarity, 1.0); // NPN
        assert!((desc.is_val - 1e-14_f32).abs() < 1e-20);
        assert!((desc.bf - 200.0_f32).abs() < 1e-6);
        assert!((desc.br - 2.0_f32).abs() < 1e-6);

        let expected_nf_vt = (1.0 * 0.02585) as f32;
        let expected_nr_vt = (1.0 * 0.02585) as f32;
        assert!((desc.nf_vt - expected_nf_vt).abs() < 1e-6);
        assert!((desc.nr_vt - expected_nr_vt).abs() < 1e-6);

        // All 9 stamp positions should be valid (all nodes are non-ground)
        for i in 0..9 {
            assert_ne!(
                desc.g_row_col[i],
                u32::MAX,
                "g_row_col[{i}] should not be sentinel"
            );
        }

        // b_idx should match node indices
        assert_eq!(desc.b_idx[0], 2); // collector
        assert_eq!(desc.b_idx[1], 1); // base
        assert_eq!(desc.b_idx[2], 3); // emitter
    }

    #[test]
    fn test_single_mosfet_descriptor() {
        // V1(5V) -> node "1", M1 NMOS: drain="2", gate="1", source="3",
        // R1(1k) from "1" to "2" (drain load), R2(100) from "3" to "0"
        let circuit = circuit_with_all_models(
            vec![
                Component::VSource {
                    name: "V1".into(),
                    nodes: ("1".into(), "0".into()),
                    dc: Some(5.0),
                    ac: None,
                    tran: None,
                },
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "2".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("3".into(), "0".into()),
                    value: 100.0,
                },
                Component::Mosfet {
                    name: "M1".into(),
                    nodes: ["2".into(), "1".into(), "3".into()],
                    model: "NMOS1".into(),
                },
            ],
            vec![],
            vec![],
            vec![MosfetModel {
                name: "NMOS1".into(),
                vto: 0.7,
                kp: 1.1e-4,
                lambda: 0.04,
                is_nmos: true,
            }],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.mosfet_descriptors.len(), 1);

        let desc = &mna.mosfet_descriptors[0];
        // Nodes: "1"->0, "2"->1, "3"->2
        assert_eq!(desc.drain_idx, 1); // node "2"
        assert_eq!(desc.gate_idx, 0); // node "1"
        assert_eq!(desc.source_idx, 2); // node "3"
        assert_eq!(desc.polarity, 1.0); // NMOS
        assert!((desc.vto - 0.7_f32).abs() < 1e-6);
        assert!((desc.kp - 1.1e-4_f32).abs() < 1e-10);
        assert!((desc.lambda - 0.04_f32).abs() < 1e-6);

        // All 6 stamp positions should be valid (all nodes are non-ground)
        for i in 0..6 {
            assert_ne!(
                desc.g_row_col[i],
                u32::MAX,
                "g_row_col[{i}] should not be sentinel"
            );
        }

        // b_idx
        assert_eq!(desc.b_idx[0], 1); // drain
        assert_eq!(desc.b_idx[1], 2); // source
    }

    #[test]
    fn test_bjt_to_ground() {
        // Q1 NPN with emitter to ground
        // R1 to provide non-ground paths for collector and base
        let circuit = circuit_with_all_models(
            vec![
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("2".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Bjt {
                    name: "Q1".into(),
                    nodes: ["1".into(), "2".into(), "0".into()],
                    model: "QMOD".into(),
                },
            ],
            vec![],
            vec![BjtModel {
                name: "QMOD".into(),
                is: 1e-16,
                bf: 100.0,
                br: 1.0,
                nf: 1.0,
                nr: 1.0,
                is_npn: true,
            }],
            vec![],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.bjt_descriptors.len(), 1);

        let desc = &mna.bjt_descriptors[0];
        // Nodes: "1"->0, "2"->1
        assert_eq!(desc.collector_idx, 0);
        assert_eq!(desc.base_idx, 1);
        assert_eq!(desc.emitter_idx, u32::MAX); // ground

        // Ground-related G-matrix entries should be sentinel
        // Layout: [CC, CB, CE, BC, BB, BE, EC, EB, EE]
        assert_ne!(desc.g_row_col[0], u32::MAX); // CC
        assert_ne!(desc.g_row_col[1], u32::MAX); // CB
        assert_eq!(desc.g_row_col[2], u32::MAX); // CE (emitter=ground)
        assert_ne!(desc.g_row_col[3], u32::MAX); // BC
        assert_ne!(desc.g_row_col[4], u32::MAX); // BB
        assert_eq!(desc.g_row_col[5], u32::MAX); // BE (emitter=ground)
        assert_eq!(desc.g_row_col[6], u32::MAX); // EC (emitter=ground)
        assert_eq!(desc.g_row_col[7], u32::MAX); // EB (emitter=ground)
        assert_eq!(desc.g_row_col[8], u32::MAX); // EE (emitter=ground)

        // b_idx: emitter should be sentinel
        assert_eq!(desc.b_idx[0], 0); // collector
        assert_eq!(desc.b_idx[1], 1); // base
        assert_eq!(desc.b_idx[2], u32::MAX); // emitter = ground
    }

    #[test]
    fn test_mosfet_to_ground() {
        // M1 NMOS with source to ground
        let circuit = circuit_with_all_models(
            vec![
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("2".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Mosfet {
                    name: "M1".into(),
                    nodes: ["1".into(), "2".into(), "0".into()],
                    model: "MMOD".into(),
                },
            ],
            vec![],
            vec![],
            vec![MosfetModel {
                name: "MMOD".into(),
                vto: 1.0,
                kp: 2e-5,
                lambda: 0.0,
                is_nmos: true,
            }],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.mosfet_descriptors.len(), 1);

        let desc = &mna.mosfet_descriptors[0];
        // Nodes: "1"->0, "2"->1
        assert_eq!(desc.drain_idx, 0);
        assert_eq!(desc.gate_idx, 1);
        assert_eq!(desc.source_idx, u32::MAX); // ground

        // Layout: [DD, DG, DS, SD, SG, SS]
        assert_ne!(desc.g_row_col[0], u32::MAX); // DD
        assert_ne!(desc.g_row_col[1], u32::MAX); // DG
        assert_eq!(desc.g_row_col[2], u32::MAX); // DS (source=ground)
        assert_eq!(desc.g_row_col[3], u32::MAX); // SD (source=ground)
        assert_eq!(desc.g_row_col[4], u32::MAX); // SG (source=ground)
        assert_eq!(desc.g_row_col[5], u32::MAX); // SS (source=ground)

        // b_idx
        assert_eq!(desc.b_idx[0], 0); // drain
        assert_eq!(desc.b_idx[1], u32::MAX); // source = ground
    }

    #[test]
    fn test_bjt_undefined_model_error() {
        let circuit = circuit_with_all_models(
            vec![Component::Bjt {
                name: "Q1".into(),
                nodes: ["1".into(), "2".into(), "0".into()],
                model: "NOMODEL".into(),
            }],
            vec![],
            vec![], // no BJT models defined
            vec![],
        );

        let result = compile(&circuit);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("undefined model"),
            "Error should mention undefined model, got: {err_msg}"
        );
    }

    #[test]
    fn test_mosfet_undefined_model_error() {
        let circuit = circuit_with_all_models(
            vec![Component::Mosfet {
                name: "M1".into(),
                nodes: ["1".into(), "2".into(), "0".into()],
                model: "NOMODEL".into(),
            }],
            vec![],
            vec![],
            vec![], // no MOSFET models defined
        );

        let result = compile(&circuit);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("undefined model"),
            "Error should mention undefined model, got: {err_msg}"
        );
    }

    #[test]
    fn test_mixed_diode_bjt_mosfet() {
        // Circuit with all three nonlinear device types
        let circuit = circuit_with_all_models(
            vec![
                Component::VSource {
                    name: "V1".into(),
                    nodes: ("1".into(), "0".into()),
                    dc: Some(5.0),
                    ac: None,
                    tran: None,
                },
                Component::Resistor {
                    name: "R1".into(),
                    nodes: ("1".into(), "2".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R2".into(),
                    nodes: ("3".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R3".into(),
                    nodes: ("4".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Resistor {
                    name: "R4".into(),
                    nodes: ("5".into(), "0".into()),
                    value: 1000.0,
                },
                Component::Diode {
                    name: "D1".into(),
                    nodes: ("2".into(), "0".into()),
                    model: "DMOD".into(),
                },
                Component::Bjt {
                    name: "Q1".into(),
                    nodes: ["3".into(), "2".into(), "4".into()],
                    model: "QMOD".into(),
                },
                Component::Mosfet {
                    name: "M1".into(),
                    nodes: ["3".into(), "2".into(), "5".into()],
                    model: "MMOD".into(),
                },
            ],
            vec![DiodeModel {
                name: "DMOD".into(),
                is: 1e-14,
                n: 1.0,
            }],
            vec![BjtModel {
                name: "QMOD".into(),
                is: 1e-16,
                bf: 100.0,
                br: 1.0,
                nf: 1.0,
                nr: 1.0,
                is_npn: true,
            }],
            vec![MosfetModel {
                name: "MMOD".into(),
                vto: 0.7,
                kp: 1.1e-4,
                lambda: 0.04,
                is_nmos: true,
            }],
        );

        let mna = compile(&circuit).unwrap();
        assert_eq!(mna.diode_descriptors.len(), 1);
        assert_eq!(mna.bjt_descriptors.len(), 1);
        assert_eq!(mna.mosfet_descriptors.len(), 1);

        // Verify the sparsity pattern includes stamps from all devices
        // 5 non-ground nodes + 1 branch = size 6
        assert_eq!(mna.node_names.len(), 5);
        assert_eq!(mna.size, 6);
    }
}
