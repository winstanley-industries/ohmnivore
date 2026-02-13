//! Analysis engine — orchestrates DC and AC simulations.

pub mod ac;
pub mod dc;

use num_complex::Complex64;

/// DC operating point results.
#[derive(Debug)]
pub struct DcResult {
    /// (node_name, voltage) pairs.
    pub node_voltages: Vec<(String, f64)>,
    /// (branch_name, current) pairs (e.g., I(V1)).
    pub branch_currents: Vec<(String, f64)>,
}

/// AC sweep results.
#[derive(Debug)]
pub struct AcResult {
    /// Frequency points (Hz).
    pub frequencies: Vec<f64>,
    /// Per-node complex voltage at each frequency.
    /// Each entry: (node_name, vec_of_complex_voltage_per_freq).
    pub node_voltages: Vec<(String, Vec<Complex64>)>,
    /// Per-branch complex current at each frequency.
    pub branch_currents: Vec<(String, Vec<Complex64>)>,
}
