use num_complex::Complex64;
use ohmnivore::analysis::{AcResult, DcResult};
use std::path::Path;
use std::process::Command;

type AcParseResult = (Vec<f64>, Vec<(String, Vec<Complex64>)>);

/// Abstract backend for running SPICE simulations.
/// Designed to be swappable (e.g., text output vs raw binary parser).
pub trait SpiceBackend {
    fn run_dc(&self, netlist: &Path, nodes: &[String]) -> Result<DcResult, String>;
    fn run_ac(&self, netlist: &Path, nodes: &[String]) -> Result<AcResult, String>;
}

/// Runs ngspice in batch mode, parses text output.
pub struct NgspiceBatch;

impl NgspiceBatch {
    pub fn new() -> Result<Self, String> {
        Command::new("ngspice")
            .arg("--version")
            .output()
            .map_err(|e| format!("ngspice not found on PATH: {e}"))?;
        Ok(NgspiceBatch)
    }
}

/// Extract inner node name from "V(nodename)" format.
fn extract_node_name(node: &str) -> &str {
    node.strip_prefix("V(")
        .or_else(|| node.strip_prefix("v("))
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(node)
}

/// Build a temp file path unique to this process + circuit.
fn temp_netlist_path(netlist: &Path) -> std::path::PathBuf {
    let stem = netlist.file_stem().unwrap_or_default().to_string_lossy();
    std::env::temp_dir().join(format!(
        "ohmnivore_regression_{pid}_{stem}.cir",
        pid = std::process::id(),
    ))
}

/// Augment a DC netlist: replace .DC with .op, add .print dc lines.
/// The .print dc lines error with ".print: no dc analysis found" but their
/// presence triggers ngspice batch mode execution. The actual results come
/// from the .op operating point table.
fn augment_dc_netlist(content: &str, nodes: &[String]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim().to_uppercase();
        if trimmed == ".DC" {
            lines.push(".op".to_string());
        } else if trimmed == ".END" {
            for node in nodes {
                let inner = extract_node_name(node);
                lines.push(format!(".print dc v({})", inner));
            }
            lines.push(line.to_string());
        } else {
            lines.push(line.to_string());
        }
    }
    lines.join("\n") + "\n"
}

/// Augment an AC netlist: add .print ac vm(node) vp(node) lines for each node.
fn augment_ac_netlist(content: &str, nodes: &[String]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim().to_uppercase();
        if trimmed == ".END" {
            for node in nodes {
                let inner = extract_node_name(node);
                lines.push(format!(".print ac vm({}) vp({})", inner, inner));
            }
            lines.push(line.to_string());
        } else {
            lines.push(line.to_string());
        }
    }
    lines.join("\n") + "\n"
}

/// Parse DC output from ngspice batch mode.
///
/// Handles two formats:
/// - Format A (from .print dc): `v(2) = 5.000000e+00`
/// - Format B (operating point table): tabular node/voltage listing
fn parse_dc_output(output: &str, nodes: &[String]) -> Result<Vec<(String, f64)>, String> {
    let mut results: Vec<(String, f64)> = Vec::new();

    for node in nodes {
        let inner = extract_node_name(node);
        let inner_lower = inner.to_lowercase();
        let mut found = false;

        // Try Format A: `v(nodename) = value`
        for line in output.lines() {
            let trimmed = line.trim().to_lowercase();
            let pattern = format!("v({}) =", inner_lower);
            if let Some(pos) = trimmed.find(&pattern) {
                let after_eq = &trimmed[pos + pattern.len()..];
                let value_str = after_eq.trim();
                let value: f64 = value_str
                    .parse()
                    .map_err(|e| format!("Failed to parse DC voltage for {}: {}", node, e))?;
                results.push((inner.to_string(), value));
                found = true;
                break;
            }
        }
        if found {
            continue;
        }

        // Try Format B: operating point table
        // Lines look like: "V(2)   5.000000e+00" or "vcc   5.000000e+00"
        let mut in_table = false;
        let target_v = format!("v({})", inner_lower);
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("----") {
                in_table = true;
                continue;
            }
            if in_table && !trimmed.is_empty() {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let entry = parts[0].to_lowercase();
                    // Match bare name or V(name) format
                    if entry == inner_lower || entry == target_v {
                        let value: f64 = parts[1]
                            .parse()
                            .map_err(|e| format!("Failed to parse DC voltage for {}: {}", node, e))?;
                        results.push((inner.to_string(), value));
                        found = true;
                        break;
                    }
                }
                // If we hit a non-data line, stop searching this table
                if parts.len() < 2 {
                    in_table = false;
                }
            }
        }

        if !found {
            return Err(format!(
                "Node {} not found in ngspice DC output",
                node
            ));
        }
    }

    Ok(results)
}

/// Parse AC output from ngspice batch mode.
///
/// Expected format:
/// ```text
/// Index   frequency       vm(2)           vp(2)
/// --------------------------------------------------------------------------------
/// 0       1.000000e+00    9.999998e-01    -5.729578e-03
/// ```
fn parse_ac_output(
    output: &str,
    nodes: &[String],
) -> Result<AcParseResult, String> {
    let mut frequencies: Vec<f64> = Vec::new();
    // Map from inner node name (lowercase) -> (original_node_string, mag_col_index, phase_col_index)
    let mut node_col_map: Vec<(String, usize, usize)> = Vec::new();

    // Find the header line that contains "Index" and "frequency"
    let lines: Vec<&str> = output.lines().collect();
    let mut data_start = None;
    let mut header_cols: Vec<String> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim().to_lowercase();
        if trimmed.starts_with("index") && trimmed.contains("frequency") {
            header_cols = line.split_whitespace().map(|s| s.to_lowercase()).collect();
            // The data starts after the dashes line
            if i + 1 < lines.len() && lines[i + 1].trim().starts_with("---") {
                data_start = Some(i + 2);
            } else {
                data_start = Some(i + 1);
            }
            break;
        }
    }

    let data_start = data_start.ok_or("AC output header not found in ngspice output")?;

    // Map each requested node to its vm/vp column indices
    for node in nodes {
        let inner = extract_node_name(node);
        let inner_lower = inner.to_lowercase();
        let vm_name = format!("vm({})", inner_lower);
        let vp_name = format!("vp({})", inner_lower);

        let vm_idx = header_cols
            .iter()
            .position(|c| c == &vm_name)
            .ok_or_else(|| format!("Column {} not found in AC header", vm_name))?;
        let vp_idx = header_cols
            .iter()
            .position(|c| c == &vp_name)
            .ok_or_else(|| format!("Column {} not found in AC header", vp_name))?;

        node_col_map.push((inner.to_string(), vm_idx, vp_idx));
    }

    // Initialize per-node complex voltage vectors
    let mut node_voltages: Vec<Vec<Complex64>> = vec![Vec::new(); nodes.len()];

    // Parse data rows (ngspice may repeat the header mid-output for long tables)
    for line in &lines[data_start..] {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip repeated header/separator lines
        if trimmed.starts_with("---") || trimmed.starts_with("Index") || trimmed.starts_with("index") {
            continue;
        }
        // Data rows start with a digit (the index column)
        if !trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.len() < header_cols.len() {
            continue;
        }

        // Column 1 is frequency
        let freq: f64 = cols[1]
            .parse()
            .map_err(|e| format!("Failed to parse frequency: {}", e))?;
        frequencies.push(freq);

        for (node_idx, (_node_name, vm_idx, vp_idx)) in node_col_map.iter().enumerate() {
            let mag: f64 = cols[*vm_idx]
                .parse()
                .map_err(|e| format!("Failed to parse vm: {}", e))?;
            // ngspice vp() in batch mode returns phase in radians
            let phase_rad: f64 = cols[*vp_idx]
                .parse()
                .map_err(|e| format!("Failed to parse vp: {}", e))?;
            let complex = Complex64::from_polar(mag, phase_rad);
            node_voltages[node_idx].push(complex);
        }
    }

    if frequencies.is_empty() {
        return Err("No AC data rows found in ngspice output".to_string());
    }

    let result: Vec<(String, Vec<Complex64>)> = node_col_map
        .into_iter()
        .zip(node_voltages)
        .map(|((name, _, _), voltages)| (name, voltages))
        .collect();

    Ok((frequencies, result))
}

impl SpiceBackend for NgspiceBatch {
    fn run_dc(&self, netlist: &Path, nodes: &[String]) -> Result<DcResult, String> {
        let content =
            std::fs::read_to_string(netlist).map_err(|e| format!("Failed to read netlist: {e}"))?;

        let augmented = augment_dc_netlist(&content, nodes);
        let tmp_path = temp_netlist_path(netlist);
        std::fs::write(&tmp_path, &augmented)
            .map_err(|e| format!("Failed to write temp netlist: {e}"))?;

        let output = Command::new("ngspice")
            .args(["-b", tmp_path.to_str().unwrap()])
            .output()
            .map_err(|e| format!("Failed to run ngspice: {e}"))?;

        let _ = std::fs::remove_file(&tmp_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("ngspice exited with error:\n{stderr}"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let node_voltages = parse_dc_output(&stdout, nodes)?;

        Ok(DcResult {
            node_voltages,
            branch_currents: vec![],
        })
    }

    fn run_ac(&self, netlist: &Path, nodes: &[String]) -> Result<AcResult, String> {
        let content =
            std::fs::read_to_string(netlist).map_err(|e| format!("Failed to read netlist: {e}"))?;

        let augmented = augment_ac_netlist(&content, nodes);
        let tmp_path = temp_netlist_path(netlist);
        std::fs::write(&tmp_path, &augmented)
            .map_err(|e| format!("Failed to write temp netlist: {e}"))?;

        let output = Command::new("ngspice")
            .args(["-b", tmp_path.to_str().unwrap()])
            .output()
            .map_err(|e| format!("Failed to run ngspice: {e}"))?;

        let _ = std::fs::remove_file(&tmp_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("ngspice exited with error:\n{stderr}"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (frequencies, node_voltages) = parse_ac_output(&stdout, nodes)?;

        Ok(AcResult {
            frequencies,
            node_voltages,
            branch_currents: vec![],
        })
    }
}
