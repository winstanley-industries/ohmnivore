use ohmnivore::analysis::{AcResult, DcResult, TranResult};

/// Tolerance parameters for comparison.
pub struct Tolerances {
    pub rel_tol: f64,
    pub abs_tol: f64,
}

/// Result of comparing a single DC node voltage.
pub struct NodeComparison {
    pub node: String,
    pub ohmnivore: f64,
    pub ngspice: f64,
    pub abs_error: f64,
    pub rel_error: f64,
    pub passed: bool,
}

/// Result of comparing a single AC point (one node at one frequency).
pub struct AcPointComparison {
    pub node: String,
    pub frequency: f64,
    pub ohm_mag: f64,
    pub ng_mag: f64,
    pub mag_abs_error: f64,
    pub mag_passed: bool,
    pub ohm_phase_deg: f64,
    pub ng_phase_deg: f64,
    pub phase_error_deg: f64,
    pub phase_passed: bool,
}

/// Check if a value is within tolerance of reference.
/// Pass condition: abs(ohm - ref) <= max(rel_tol * abs(ref), abs_tol)
pub fn within_tolerance(ohm: f64, reference: f64, tol: &Tolerances) -> bool {
    let err = (ohm - reference).abs();
    let limit = (tol.rel_tol * reference.abs()).max(tol.abs_tol);
    err <= limit
}

/// Compute phase error in degrees, handling +/-180 wrapping.
pub fn phase_error_degrees(a: f64, b: f64) -> f64 {
    let diff = (a - b).rem_euclid(360.0);
    diff.min(360.0 - diff)
}

/// Compare DC results for specified nodes.
pub fn compare_dc(
    ohm: &DcResult,
    ng: &DcResult,
    compare_nodes: &[String],
    tol: &Tolerances,
) -> Vec<NodeComparison> {
    compare_nodes
        .iter()
        .map(|node_spec| {
            let name = extract_node_name(node_spec);
            let ohm_v = find_voltage(&ohm.node_voltages, &name)
                .unwrap_or_else(|| panic!("node '{}' not found in Ohmnivore results", name));
            let ng_v = find_voltage(&ng.node_voltages, &name)
                .unwrap_or_else(|| panic!("node '{}' not found in ngspice results", name));

            let abs_error = (ohm_v - ng_v).abs();
            let rel_error = if ng_v.abs() > 1e-15 {
                abs_error / ng_v.abs()
            } else {
                0.0
            };
            let passed = within_tolerance(ohm_v, ng_v, tol);

            NodeComparison {
                node: node_spec.clone(),
                ohmnivore: ohm_v,
                ngspice: ng_v,
                abs_error,
                rel_error,
                passed,
            }
        })
        .collect()
}

/// Compare AC results for specified nodes across all frequency points.
pub fn compare_ac(
    ohm: &AcResult,
    ng: &AcResult,
    compare_nodes: &[String],
    mag_tol: &Tolerances,
    phase_abs_tol: f64,
) -> Vec<AcPointComparison> {
    let mut results = Vec::new();

    for node_spec in compare_nodes {
        let name = extract_node_name(node_spec);

        let ohm_data = ohm
            .node_voltages
            .iter()
            .find(|(n, _)| n == &name)
            .unwrap_or_else(|| panic!("node '{}' not in Ohmnivore AC results", name));
        let ng_data = ng
            .node_voltages
            .iter()
            .find(|(n, _)| n == &name)
            .unwrap_or_else(|| panic!("node '{}' not in ngspice AC results", name));

        // Use the shorter frequency list (they should match but be safe)
        let n_points = ohm
            .frequencies
            .len()
            .min(ng.frequencies.len())
            .min(ohm_data.1.len())
            .min(ng_data.1.len());

        for i in 0..n_points {
            let ohm_c = ohm_data.1[i];
            let ng_c = ng_data.1[i];

            let ohm_mag = ohm_c.norm();
            let ng_mag = ng_c.norm();
            let ohm_phase = ohm_c.arg().to_degrees();
            let ng_phase = ng_c.arg().to_degrees();

            let mag_abs_error = (ohm_mag - ng_mag).abs();
            let mag_passed = within_tolerance(ohm_mag, ng_mag, mag_tol);
            let pe = phase_error_degrees(ohm_phase, ng_phase);
            let phase_passed = pe <= phase_abs_tol;

            results.push(AcPointComparison {
                node: node_spec.clone(),
                frequency: ohm.frequencies[i],
                ohm_mag,
                ng_mag,
                mag_abs_error,
                mag_passed,
                ohm_phase_deg: ohm_phase,
                ng_phase_deg: ng_phase,
                phase_error_deg: pe,
                phase_passed,
            });
        }
    }

    results
}

/// Extract node name from "V(name)" format.
fn extract_node_name(spec: &str) -> String {
    spec.strip_prefix("V(")
        .and_then(|s| s.strip_suffix(")"))
        .unwrap_or(spec)
        .to_string()
}

/// Find a voltage by node name (case-insensitive comparison for robustness).
fn find_voltage(voltages: &[(String, f64)], name: &str) -> Option<f64> {
    voltages
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| *v)
}

/// Format DC comparison results as a diagnostic table.
pub fn format_dc_report(results: &[NodeComparison]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:<12} {:>14} {:>14} {:>12} {:>10} {}\n",
        "Node", "Ohmnivore", "ngspice", "Abs Error", "Rel Error", "Status"
    ));
    s.push_str(&"-".repeat(76));
    s.push('\n');
    for r in results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        s.push_str(&format!(
            "{:<12} {:>14.6e} {:>14.6e} {:>12.6e} {:>9.4}% {}\n",
            r.node,
            r.ohmnivore,
            r.ngspice,
            r.abs_error,
            r.rel_error * 100.0,
            status
        ));
    }
    s
}

/// Format AC comparison failures as a diagnostic table (only show failures).
pub fn format_ac_report(results: &[AcPointComparison]) -> String {
    let failures: Vec<_> = results
        .iter()
        .filter(|r| !r.mag_passed || !r.phase_passed)
        .collect();
    if failures.is_empty() {
        return "All AC points passed.\n".to_string();
    }
    let mut s = String::new();
    s.push_str(&format!(
        "{:<10} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>8}\n",
        "Node", "Freq", "Ohm Mag", "NG Mag", "Mag Err", "Ohm Phase", "NG Phase", "Ph Err"
    ));
    s.push_str(&"-".repeat(96));
    s.push('\n');
    for r in &failures {
        s.push_str(&format!(
            "{:<10} {:>12.4e} {:>12.6e} {:>12.6e} {:>10.6e} {:>10.3} {:>10.3} {:>8.3}\n",
            r.node,
            r.frequency,
            r.ohm_mag,
            r.ng_mag,
            r.mag_abs_error,
            r.ohm_phase_deg,
            r.ng_phase_deg,
            r.phase_error_deg
        ));
    }
    s.push_str(&format!(
        "\n{} of {} points failed.\n",
        failures.len(),
        results.len()
    ));
    s
}

/// Result of comparing a single transient time point for one node.
pub struct TranPointComparison {
    pub node: String,
    pub time: f64,
    pub ohmnivore: f64,
    pub ngspice: f64,
    pub abs_error: f64,
    pub rel_error: f64,
    pub passed: bool,
}

/// Linear interpolation of a waveform at a specific time.
///
/// Binary-searches for the interval, then linearly interpolates.
/// Clamps to boundary values for times outside the data range.
fn interpolate_at(times: &[f64], values: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return values[0];
    }
    if t >= times[times.len() - 1] {
        return values[values.len() - 1];
    }

    // Binary search for the interval containing t
    let idx = match times.binary_search_by(|probe| probe.partial_cmp(&t).unwrap()) {
        Ok(i) => return values[i], // exact match
        Err(i) => i,               // t falls between times[i-1] and times[i]
    };

    let t0 = times[idx - 1];
    let t1 = times[idx];
    let v0 = values[idx - 1];
    let v1 = values[idx];

    let frac = (t - t0) / (t1 - t0);
    v0 + frac * (v1 - v0)
}

/// Compare transient results by interpolating Ohmnivore's adaptive-timestep data
/// at ngspice's regular time grid.
///
/// Skips comparison at t=0 (initial conditions may differ slightly).
pub fn compare_tran(
    ohm: &TranResult,
    ng: &TranResult,
    compare_nodes: &[String],
    tol: &Tolerances,
) -> Vec<TranPointComparison> {
    let mut results = Vec::new();

    for node_spec in compare_nodes {
        let name = extract_node_name(node_spec);

        let ohm_data = ohm
            .node_voltages
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(&name))
            .unwrap_or_else(|| panic!("node '{}' not in Ohmnivore tran results", name));
        let ng_data = ng
            .node_voltages
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(&name))
            .unwrap_or_else(|| panic!("node '{}' not in ngspice tran results", name));

        for (i, &t) in ng.times.iter().enumerate() {
            // Skip t=0 — initial conditions may differ
            if t == 0.0 {
                continue;
            }

            let ohm_v = interpolate_at(&ohm.times, &ohm_data.1, t);
            let ng_v = ng_data.1[i];

            let abs_error = (ohm_v - ng_v).abs();
            let rel_error = if ng_v.abs() > 1e-15 {
                abs_error / ng_v.abs()
            } else {
                0.0
            };
            let passed = within_tolerance(ohm_v, ng_v, tol);

            results.push(TranPointComparison {
                node: node_spec.clone(),
                time: t,
                ohmnivore: ohm_v,
                ngspice: ng_v,
                abs_error,
                rel_error,
                passed,
            });
        }
    }

    results
}

/// Format transient comparison failures as a diagnostic table (only show failures).
pub fn format_tran_report(results: &[TranPointComparison]) -> String {
    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if failures.is_empty() {
        return "All tran points passed.\n".to_string();
    }
    let mut s = String::new();
    s.push_str(&format!(
        "{:<10} {:>12} {:>14} {:>14} {:>12} {:>10}\n",
        "Node", "Time", "Ohmnivore", "ngspice", "Abs Error", "Rel Error"
    ));
    s.push_str(&"-".repeat(76));
    s.push('\n');
    for r in &failures {
        s.push_str(&format!(
            "{:<10} {:>12.4e} {:>14.6e} {:>14.6e} {:>12.6e} {:>9.4}%\n",
            r.node,
            r.time,
            r.ohmnivore,
            r.ngspice,
            r.abs_error,
            r.rel_error * 100.0
        ));
    }
    s.push_str(&format!(
        "\n{} of {} points failed.\n",
        failures.len(),
        results.len()
    ));
    s
}
