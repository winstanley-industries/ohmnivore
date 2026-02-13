//! Circuit intermediate representation.
//!
//! The parser produces a `Circuit` containing components and analysis commands.
//! The compiler consumes this IR to build MNA matrices.

/// Node identifier in the netlist (e.g., "0", "1", "GND", "out").
/// Ground is "0" or "GND" — the compiler maps these to the reference node.
pub type NodeId = String;

/// A circuit component parsed from the netlist.
#[derive(Debug, Clone)]
pub enum Component {
    Resistor {
        name: String,
        nodes: (NodeId, NodeId),
        value: f64,
    },
    Capacitor {
        name: String,
        nodes: (NodeId, NodeId),
        value: f64,
    },
    Inductor {
        name: String,
        nodes: (NodeId, NodeId),
        value: f64,
    },
    /// Independent voltage source. DC value and/or AC (magnitude, phase_degrees).
    VSource {
        name: String,
        nodes: (NodeId, NodeId),
        dc: Option<f64>,
        ac: Option<(f64, f64)>,
    },
    /// Independent current source. DC value and/or AC (magnitude, phase_degrees).
    ISource {
        name: String,
        nodes: (NodeId, NodeId),
        dc: Option<f64>,
        ac: Option<(f64, f64)>,
    },
}

/// AC sweep type matching SPICE syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcSweepType {
    /// Decade sweep — n_points is points per decade
    Dec,
    /// Octave sweep — n_points is points per octave
    Oct,
    /// Linear sweep — n_points is total points
    Lin,
}

/// An analysis command from the netlist.
#[derive(Debug, Clone)]
pub enum Analysis {
    /// DC operating point (.DC or .OP)
    Dc,
    /// AC frequency sweep (.AC)
    Ac {
        sweep_type: AcSweepType,
        n_points: usize,
        f_start: f64,
        f_stop: f64,
    },
}

/// A parsed circuit: components + analysis commands.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub components: Vec<Component>,
    pub analyses: Vec<Analysis>,
}
