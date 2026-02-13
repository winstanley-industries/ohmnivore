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
        tran: Option<TransientFunc>,
    },
    /// Independent current source. DC value and/or AC (magnitude, phase_degrees).
    ISource {
        name: String,
        nodes: (NodeId, NodeId),
        dc: Option<f64>,
        ac: Option<(f64, f64)>,
        tran: Option<TransientFunc>,
    },
    /// Diode element referencing a named model.
    Diode {
        name: String,
        nodes: (NodeId, NodeId),
        model: String,
    },
    /// BJT element: [collector, base, emitter] referencing a named model.
    Bjt {
        name: String,
        nodes: [NodeId; 3],
        model: String,
    },
    /// MOSFET element: [drain, gate, source] referencing a named model.
    Mosfet {
        name: String,
        nodes: [NodeId; 3],
        model: String,
    },
}

/// A diode model with Shockley equation parameters.
#[derive(Debug, Clone)]
pub struct DiodeModel {
    pub name: String,
    /// Saturation current (default: 1e-14)
    pub is: f64,
    /// Emission coefficient (default: 1.0)
    pub n: f64,
}

/// A BJT model with Ebers-Moll parameters.
#[derive(Debug, Clone)]
pub struct BjtModel {
    pub name: String,
    /// Saturation current (default: 1e-16)
    pub is: f64,
    /// Forward current gain (default: 100)
    pub bf: f64,
    /// Reverse current gain (default: 1)
    pub br: f64,
    /// Forward emission coefficient (default: 1.0)
    pub nf: f64,
    /// Reverse emission coefficient (default: 1.0)
    pub nr: f64,
    /// true = NPN, false = PNP
    pub is_npn: bool,
}

/// A Level-1 MOSFET model with Shichman-Hodges parameters.
#[derive(Debug, Clone)]
pub struct MosfetModel {
    pub name: String,
    /// Threshold voltage (default: 1.0)
    pub vto: f64,
    /// Transconductance parameter (default: 2e-5)
    pub kp: f64,
    /// Channel-length modulation (default: 0.0)
    pub lambda: f64,
    /// true = NMOS, false = PMOS
    pub is_nmos: bool,
}

/// Time-varying source waveform for transient analysis.
#[derive(Debug, Clone)]
pub enum TransientFunc {
    Pulse {
        v1: f64,
        v2: f64,
        td: f64,
        tr: f64,
        tf: f64,
        pw: f64,
        per: f64,
    },
    Sin {
        vo: f64,
        va: f64,
        freq: f64,
        td: f64,
        theta: f64,
    },
    Pwl {
        pairs: Vec<(f64, f64)>,
    },
    Exp {
        v1: f64,
        v2: f64,
        td1: f64,
        tau1: f64,
        td2: f64,
        tau2: f64,
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
    /// Transient analysis (.TRAN)
    Tran {
        tstep: f64,
        tstop: f64,
        tstart: f64,
        uic: bool,
    },
}

/// A parsed circuit: components + analysis commands.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub components: Vec<Component>,
    pub analyses: Vec<Analysis>,
    pub models: Vec<DiodeModel>,
    pub bjt_models: Vec<BjtModel>,
    pub mosfet_models: Vec<MosfetModel>,
}
