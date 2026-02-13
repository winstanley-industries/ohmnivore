//! SPICE netlist parser.
//!
//! Parses a subset of SPICE into the circuit IR.
//!
//! # Supported syntax
//!
//! ```text
//! * comment line
//! Rname n+ n- value       (resistor)
//! Cname n+ n- value       (capacitor)
//! Lname n+ n- value       (inductor)
//! Vname n+ n- [DC val] [AC mag [phase]]  (voltage source)
//! Iname n+ n- [DC val] [AC mag [phase]]  (current source)
//! .DC                     (request DC operating point)
//! .AC DEC|OCT|LIN np fstart fstop
//! .END
//! ```
//!
//! Values support engineering suffixes: T, G, MEG, K, M, U, N, P, F
//! (case-insensitive).

use nom::branch::alt;
use nom::bytes::complete::{tag_no_case, take_while1};
use nom::character::complete::{space0, space1};
use nom::combinator::{map, opt, recognize};
use nom::number::complete::double;
use nom::IResult;
use nom::Parser;

use crate::error::{OhmnivoreError, Result};
use crate::ir::{AcSweepType, Analysis, Circuit, Component, DiodeModel};

/// Parse a SPICE netlist string into a Circuit IR.
pub fn parse(input: &str) -> Result<Circuit> {
    let mut components = Vec::new();
    let mut analyses = Vec::new();
    let mut models = Vec::new();

    for (line_num, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();

        // Skip blank lines and comments
        if line.is_empty() || line.starts_with('*') {
            continue;
        }

        // Stop at .END
        let upper = line.to_uppercase();
        if upper == ".END" {
            break;
        }

        // Determine line type by first character
        let first = line.chars().next().unwrap();
        match first.to_ascii_uppercase() {
            'R' => {
                let comp = parse_rlc_line(line, 'R')
                    .map_err(|e| parse_err(line_num, raw_line, &e))?;
                components.push(comp);
            }
            'C' => {
                let comp = parse_rlc_line(line, 'C')
                    .map_err(|e| parse_err(line_num, raw_line, &e))?;
                components.push(comp);
            }
            'L' => {
                let comp = parse_rlc_line(line, 'L')
                    .map_err(|e| parse_err(line_num, raw_line, &e))?;
                components.push(comp);
            }
            'V' => {
                let comp = parse_source_line(line, true)
                    .map_err(|e| parse_err(line_num, raw_line, &e))?;
                components.push(comp);
            }
            'I' => {
                let comp = parse_source_line(line, false)
                    .map_err(|e| parse_err(line_num, raw_line, &e))?;
                components.push(comp);
            }
            'D' => {
                let comp = parse_diode_line(line)
                    .map_err(|e| parse_err(line_num, raw_line, &e))?;
                components.push(comp);
            }
            '.' => {
                if upper.starts_with(".MODEL") {
                    let model = parse_model_command(line)
                        .map_err(|e| parse_err(line_num, raw_line, &e))?;
                    models.push(model);
                } else {
                    let analysis = parse_dot_command(line)
                        .map_err(|e| parse_err(line_num, raw_line, &e))?;
                    if let Some(a) = analysis {
                        analyses.push(a);
                    }
                }
            }
            _ => {
                return Err(OhmnivoreError::Parse(format!(
                    "line {}: unknown element '{}': {}",
                    line_num + 1,
                    first,
                    raw_line
                )));
            }
        }
    }

    Ok(Circuit {
        components,
        analyses,
        models,
    })
}

fn parse_err(line_num: usize, raw_line: &str, detail: &str) -> OhmnivoreError {
    OhmnivoreError::Parse(format!("line {}: {} in: {}", line_num + 1, detail, raw_line))
}

// ---------------------------------------------------------------------------
// Engineering suffix value parser
// ---------------------------------------------------------------------------

/// Parse a numeric value with optional engineering suffix.
/// Handles: 10k, 100n, 4.7u, 1MEG, 1e3, -3.3, etc.
fn eng_value(input: &str) -> IResult<&str, f64> {
    let (rest, num) = double(input)?;
    let (rest, suffix) = opt(eng_suffix).parse(rest)?;
    let multiplier = suffix.unwrap_or(1.0);
    Ok((rest, num * multiplier))
}

/// Match an engineering suffix and return its multiplier.
fn eng_suffix(input: &str) -> IResult<&str, f64> {
    // Order matters: MEG must come before M
    alt((
        map(tag_no_case("MEG"), |_: &str| 1e6),
        map(tag_no_case("T"), |_: &str| 1e12),
        map(tag_no_case("G"), |_: &str| 1e9),
        map(tag_no_case("K"), |_: &str| 1e3),
        map(tag_no_case("M"), |_: &str| 1e-3),
        map(tag_no_case("U"), |_: &str| 1e-6),
        map(tag_no_case("N"), |_: &str| 1e-9),
        map(tag_no_case("P"), |_: &str| 1e-12),
        map(tag_no_case("F"), |_: &str| 1e-15),
    ))
    .parse(input)
}

// ---------------------------------------------------------------------------
// Token parsers
// ---------------------------------------------------------------------------

/// Parse a node identifier (alphanumeric string, e.g. "0", "GND", "out", "n1").
fn node_id(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

/// Parse a component/element name (everything up to the first whitespace).
fn element_name(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| !c.is_whitespace())(input)
}

// ---------------------------------------------------------------------------
// R, L, C parsing
// ---------------------------------------------------------------------------

/// Parse a line like: Rname n+ n- value
fn parse_rlc_line(line: &str, kind: char) -> std::result::Result<Component, String> {
    let (_, (name, _, n_plus, _, n_minus, _, value)) = (
        element_name,
        space1,
        node_id,
        space1,
        node_id,
        space1,
        eng_value,
    )
        .parse(line)
        .map_err(|_| format!("failed to parse {} component", kind))?;

    let name = name.to_string();
    let nodes = (n_plus.to_string(), n_minus.to_string());

    match kind.to_ascii_uppercase() {
        'R' => Ok(Component::Resistor { name, nodes, value }),
        'C' => Ok(Component::Capacitor { name, nodes, value }),
        'L' => Ok(Component::Inductor { name, nodes, value }),
        _ => Err(format!("unknown RLC kind: {}", kind)),
    }
}

// ---------------------------------------------------------------------------
// V/I source parsing
// ---------------------------------------------------------------------------

/// Parse a voltage or current source line:
///   Vname n+ n- [DC val] [AC mag [phase]]
fn parse_source_line(line: &str, is_voltage: bool) -> std::result::Result<Component, String> {
    // Parse: name node+ node-
    let (rest, (name, _, n_plus, _, n_minus, _)) = (
        element_name,
        space1,
        node_id,
        space1,
        node_id,
        space0,
    )
        .parse(line)
        .map_err(|_| "failed to parse source name/nodes".to_string())?;

    let rest = rest.trim();

    // Parse optional DC and AC specifications from remaining text
    let (dc, ac) = parse_source_specs(rest)?;

    let name = name.to_string();
    let nodes = (n_plus.to_string(), n_minus.to_string());

    if is_voltage {
        Ok(Component::VSource { name, nodes, dc, ac })
    } else {
        Ok(Component::ISource { name, nodes, dc, ac })
    }
}

/// Parse the DC/AC spec portion of a source line.
/// Could be:
///   (empty)
///   DC 5
///   AC 1 0
///   DC 5 AC 1 0
///   5          (bare number treated as DC value)
fn parse_source_specs(
    input: &str,
) -> std::result::Result<(Option<f64>, Option<(f64, f64)>), String> {
    if input.is_empty() {
        return Ok((None, None));
    }

    let mut rest = input;
    let mut dc: Option<f64> = None;
    let mut ac: Option<(f64, f64)> = None;

    // Try to parse DC spec
    let upper = rest.to_uppercase();
    if upper.starts_with("DC") {
        // skip "DC" keyword
        rest = rest[2..].trim_start();
        if !rest.is_empty() && !rest.to_uppercase().starts_with("AC") {
            let (r, val) = eng_value(rest)
                .map_err(|_| "failed to parse DC value".to_string())?;
            dc = Some(val);
            rest = r.trim_start();
        }
    } else if !upper.starts_with("AC") {
        // Bare number — treat as DC value
        if let Ok((r, val)) = eng_value(rest) {
            dc = Some(val);
            rest = r.trim_start();
        }
    }

    // Try to parse AC spec
    let upper = rest.to_uppercase();
    if upper.starts_with("AC") {
        rest = rest[2..].trim_start();
        let (r, mag) = eng_value(rest)
            .map_err(|_| "failed to parse AC magnitude".to_string())?;
        rest = r.trim_start();

        let phase = if !rest.is_empty() {
            let (_, p) = eng_value(rest)
                .map_err(|_| "failed to parse AC phase".to_string())?;
            p
        } else {
            0.0
        };
        ac = Some((mag, phase));
    }

    Ok((dc, ac))
}

// ---------------------------------------------------------------------------
// Diode parsing
// ---------------------------------------------------------------------------

/// Parse a diode line: Dname anode cathode MODELNAME
fn parse_diode_line(line: &str) -> std::result::Result<Component, String> {
    let (_, (name, _, anode, _, cathode, _, model_name)) = (
        element_name,
        space1,
        node_id,
        space1,
        node_id,
        space1,
        node_id,
    )
        .parse(line)
        .map_err(|_| "failed to parse diode element".to_string())?;

    Ok(Component::Diode {
        name: name.to_string(),
        nodes: (anode.to_string(), cathode.to_string()),
        model: model_name.to_string(),
    })
}

// ---------------------------------------------------------------------------
// .MODEL parsing
// ---------------------------------------------------------------------------

/// Parse: .MODEL name D(IS=val N=val)
fn parse_model_command(line: &str) -> std::result::Result<DiodeModel, String> {
    // Skip ".MODEL" keyword
    let rest = line.trim();
    let upper = rest.to_uppercase();
    if !upper.starts_with(".MODEL") {
        return Err("expected .MODEL directive".to_string());
    }
    let rest = rest[6..].trim_start();

    // Parse model name
    let (rest, model_name) = node_id(rest)
        .map_err(|_| "expected model name after .MODEL".to_string())?;
    let rest = rest.trim_start();

    // Parse type (only D supported)
    let (rest, model_type) = node_id(rest)
        .map_err(|_| "expected model type (D) after model name".to_string())?;
    if !model_type.eq_ignore_ascii_case("D") {
        return Err(format!("unsupported model type '{}', only D is supported", model_type));
    }
    let rest = rest.trim_start();

    // Default parameter values
    let mut is = 1e-14;
    let mut n = 1.0;

    // Parse optional parameters in parentheses
    if rest.starts_with('(') {
        let end = rest.find(')').ok_or("missing closing ')' in .MODEL parameters")?;
        let params_str = &rest[1..end];
        parse_model_params(params_str, &mut is, &mut n)?;
    }

    Ok(DiodeModel {
        name: model_name.to_string(),
        is,
        n,
    })
}

/// Parse key=value parameters from inside .MODEL parentheses.
/// Supports IS and N parameters in any order.
fn parse_model_params(params: &str, is: &mut f64, n: &mut f64) -> std::result::Result<(), String> {
    for token in params.split_whitespace() {
        let upper = token.to_uppercase();
        if upper.starts_with("IS=") {
            let original_val = &token[3..];
            let (_, val) = eng_value(original_val)
                .map_err(|_| format!("failed to parse IS value: {}", original_val))?;
            *is = val;
        } else if upper.starts_with("N=") {
            let original_val = &token[2..];
            let (_, val) = eng_value(original_val)
                .map_err(|_| format!("failed to parse N value: {}", original_val))?;
            *n = val;
        } else {
            return Err(format!("unknown model parameter: {}", token));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dot-command parsing
// ---------------------------------------------------------------------------

/// Parse a dot command (.DC, .AC, .OP, .END, etc.)
/// Returns None for .END (handled by caller) or unrecognized commands.
fn parse_dot_command(line: &str) -> std::result::Result<Option<Analysis>, String> {
    let upper = line.to_uppercase();
    let trimmed = upper.trim();

    if trimmed == ".DC" || trimmed == ".OP" {
        return Ok(Some(Analysis::Dc));
    }

    if trimmed.starts_with(".AC") {
        return parse_ac_command(line).map(Some);
    }

    // Ignore unrecognized dot commands (like .TITLE, .PARAM, etc.)
    Ok(None)
}

/// Parse: .AC DEC|OCT|LIN np fstart fstop
fn parse_ac_command(line: &str) -> std::result::Result<Analysis, String> {
    // Skip past ".AC"
    let rest = line[3..].trim_start();

    // Parse sweep type
    let (rest, sweep_type) = parse_sweep_type(rest)?;
    let rest = rest.trim_start();

    // Parse number of points
    let (rest, n_points) = parse_usize(rest)?;
    let rest = rest.trim_start();

    // Parse f_start
    let (rest, f_start) = eng_value(rest)
        .map_err(|_| "failed to parse AC f_start".to_string())?;
    let rest = rest.trim_start();

    // Parse f_stop
    let (_, f_stop) = eng_value(rest)
        .map_err(|_| "failed to parse AC f_stop".to_string())?;

    Ok(Analysis::Ac {
        sweep_type,
        n_points,
        f_start,
        f_stop,
    })
}

fn parse_sweep_type(input: &str) -> std::result::Result<(&str, AcSweepType), String> {
    let (rest, matched): (&str, &str) = alt((
        recognize(tag_no_case("DEC")),
        recognize(tag_no_case("OCT")),
        recognize(tag_no_case("LIN")),
    ))
    .parse(input)
    .map_err(|_: nom::Err<nom::error::Error<&str>>| {
        "expected DEC, OCT, or LIN for AC sweep type".to_string()
    })?;

    let sweep = match matched.to_uppercase().as_str() {
        "DEC" => AcSweepType::Dec,
        "OCT" => AcSweepType::Oct,
        "LIN" => AcSweepType::Lin,
        _ => unreachable!(),
    };

    Ok((rest, sweep))
}

fn parse_usize(input: &str) -> std::result::Result<(&str, usize), String> {
    let (rest, digits): (&str, &str) =
        take_while1(|c: char| c.is_ascii_digit())(input)
            .map_err(|_: nom::Err<nom::error::Error<&str>>| "expected integer".to_string())?;
    let n: usize = digits.parse().map_err(|_| "invalid integer".to_string())?;
    Ok((rest, n))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Engineering suffix tests ----

    #[test]
    fn test_eng_value_plain_number() {
        let (rest, val) = eng_value("100").unwrap();
        assert_eq!(rest, "");
        assert!((val - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_eng_value_with_decimal() {
        let (rest, val) = eng_value("4.7").unwrap();
        assert_eq!(rest, "");
        assert!((val - 4.7).abs() < 1e-12);
    }

    #[test]
    fn test_eng_value_kilo() {
        let (_, val) = eng_value("10k").unwrap();
        assert!((val - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_eng_value_kilo_uppercase() {
        let (_, val) = eng_value("10K").unwrap();
        assert!((val - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_eng_value_mega() {
        let (_, val) = eng_value("1MEG").unwrap();
        assert!((val - 1e6).abs() < 1e-3);
    }

    #[test]
    fn test_eng_value_mega_lowercase() {
        let (_, val) = eng_value("2.2meg").unwrap();
        assert!((val - 2.2e6).abs() < 1.0);
    }

    #[test]
    fn test_eng_value_milli() {
        let (_, val) = eng_value("100m").unwrap();
        assert!((val - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_eng_value_micro() {
        let (_, val) = eng_value("4.7u").unwrap();
        assert!((val - 4.7e-6).abs() < 1e-18);
    }

    #[test]
    fn test_eng_value_nano() {
        let (_, val) = eng_value("100n").unwrap();
        assert!((val - 100e-9).abs() < 1e-18);
    }

    #[test]
    fn test_eng_value_pico() {
        let (_, val) = eng_value("22p").unwrap();
        assert!((val - 22e-12).abs() < 1e-24);
    }

    #[test]
    fn test_eng_value_femto() {
        let (_, val) = eng_value("10f").unwrap();
        assert!((val - 10e-15).abs() < 1e-27);
    }

    #[test]
    fn test_eng_value_tera() {
        let (_, val) = eng_value("1T").unwrap();
        assert!((val - 1e12).abs() < 1.0);
    }

    #[test]
    fn test_eng_value_giga() {
        let (_, val) = eng_value("2.5G").unwrap();
        assert!((val - 2.5e9).abs() < 1.0);
    }

    #[test]
    fn test_eng_value_scientific_notation() {
        let (_, val) = eng_value("1e3").unwrap();
        assert!((val - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_eng_value_negative() {
        let (_, val) = eng_value("-3.3").unwrap();
        assert!((val - (-3.3)).abs() < 1e-12);
    }

    // ---- Resistor tests ----

    #[test]
    fn test_parse_resistor() {
        let circuit = parse("R1 1 0 10k").unwrap();
        assert_eq!(circuit.components.len(), 1);
        match &circuit.components[0] {
            Component::Resistor { name, nodes, value } => {
                assert_eq!(name, "R1");
                assert_eq!(nodes, &("1".into(), "0".into()));
                assert!((value - 10_000.0).abs() < 1e-6);
            }
            other => panic!("expected Resistor, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_resistor_meg() {
        let circuit = parse("R2 in out 1MEG").unwrap();
        match &circuit.components[0] {
            Component::Resistor { value, .. } => {
                assert!((value - 1e6).abs() < 1e-3);
            }
            other => panic!("expected Resistor, got {:?}", other),
        }
    }

    // ---- Capacitor tests ----

    #[test]
    fn test_parse_capacitor() {
        let circuit = parse("C1 2 0 100n").unwrap();
        assert_eq!(circuit.components.len(), 1);
        match &circuit.components[0] {
            Component::Capacitor { name, nodes, value } => {
                assert_eq!(name, "C1");
                assert_eq!(nodes, &("2".into(), "0".into()));
                assert!((value - 100e-9).abs() < 1e-18);
            }
            other => panic!("expected Capacitor, got {:?}", other),
        }
    }

    // ---- Inductor tests ----

    #[test]
    fn test_parse_inductor() {
        let circuit = parse("L1 3 4 4.7u").unwrap();
        match &circuit.components[0] {
            Component::Inductor { name, nodes, value } => {
                assert_eq!(name, "L1");
                assert_eq!(nodes, &("3".into(), "4".into()));
                assert!((value - 4.7e-6).abs() < 1e-18);
            }
            other => panic!("expected Inductor, got {:?}", other),
        }
    }

    // ---- Voltage source tests ----

    #[test]
    fn test_parse_vsource_dc() {
        let circuit = parse("V1 1 0 DC 5").unwrap();
        match &circuit.components[0] {
            Component::VSource { name, nodes, dc, ac } => {
                assert_eq!(name, "V1");
                assert_eq!(nodes, &("1".into(), "0".into()));
                assert_eq!(*dc, Some(5.0));
                assert_eq!(*ac, None);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vsource_bare_value() {
        let circuit = parse("V1 1 0 12").unwrap();
        match &circuit.components[0] {
            Component::VSource { dc, ac, .. } => {
                assert_eq!(*dc, Some(12.0));
                assert_eq!(*ac, None);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vsource_ac_only() {
        let circuit = parse("V1 1 0 AC 1 0").unwrap();
        match &circuit.components[0] {
            Component::VSource { dc, ac, .. } => {
                assert_eq!(*dc, None);
                assert!(ac.is_some());
                let (mag, phase) = ac.unwrap();
                assert!((mag - 1.0).abs() < 1e-12);
                assert!((phase - 0.0).abs() < 1e-12);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vsource_dc_and_ac() {
        let circuit = parse("V1 1 0 DC 5 AC 1 90").unwrap();
        match &circuit.components[0] {
            Component::VSource { dc, ac, .. } => {
                assert_eq!(*dc, Some(5.0));
                let (mag, phase) = ac.unwrap();
                assert!((mag - 1.0).abs() < 1e-12);
                assert!((phase - 90.0).abs() < 1e-12);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vsource_ac_no_phase() {
        let circuit = parse("V1 1 0 AC 1").unwrap();
        match &circuit.components[0] {
            Component::VSource { dc, ac, .. } => {
                assert_eq!(*dc, None);
                let (mag, phase) = ac.unwrap();
                assert!((mag - 1.0).abs() < 1e-12);
                assert!((phase - 0.0).abs() < 1e-12);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vsource_empty_spec() {
        let circuit = parse("V1 1 0").unwrap();
        match &circuit.components[0] {
            Component::VSource { dc, ac, .. } => {
                assert_eq!(*dc, None);
                assert_eq!(*ac, None);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    // ---- Current source tests ----

    #[test]
    fn test_parse_isource_dc() {
        let circuit = parse("I1 2 0 DC 1m").unwrap();
        match &circuit.components[0] {
            Component::ISource { name, dc, .. } => {
                assert_eq!(name, "I1");
                assert!((dc.unwrap() - 1e-3).abs() < 1e-15);
            }
            other => panic!("expected ISource, got {:?}", other),
        }
    }

    // ---- Analysis command tests ----

    #[test]
    fn test_parse_dc_analysis() {
        let circuit = parse(".DC").unwrap();
        assert_eq!(circuit.analyses.len(), 1);
        match &circuit.analyses[0] {
            Analysis::Dc => {}
            other => panic!("expected Dc, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_op_analysis() {
        let circuit = parse(".OP").unwrap();
        assert_eq!(circuit.analyses.len(), 1);
        match &circuit.analyses[0] {
            Analysis::Dc => {}
            other => panic!("expected Dc (from .OP), got {:?}", other),
        }
    }

    #[test]
    fn test_parse_ac_analysis_dec() {
        let circuit = parse(".AC DEC 10 1 100k").unwrap();
        assert_eq!(circuit.analyses.len(), 1);
        match &circuit.analyses[0] {
            Analysis::Ac {
                sweep_type,
                n_points,
                f_start,
                f_stop,
            } => {
                assert_eq!(*sweep_type, AcSweepType::Dec);
                assert_eq!(*n_points, 10);
                assert!((f_start - 1.0).abs() < 1e-12);
                assert!((f_stop - 100_000.0).abs() < 1e-6);
            }
            other => panic!("expected Ac, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_ac_analysis_lin() {
        let circuit = parse(".AC LIN 100 1k 10k").unwrap();
        match &circuit.analyses[0] {
            Analysis::Ac {
                sweep_type,
                n_points,
                f_start,
                f_stop,
            } => {
                assert_eq!(*sweep_type, AcSweepType::Lin);
                assert_eq!(*n_points, 100);
                assert!((f_start - 1e3).abs() < 1e-6);
                assert!((f_stop - 1e4).abs() < 1e-6);
            }
            other => panic!("expected Ac, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_ac_analysis_oct() {
        let circuit = parse(".AC OCT 20 100 10k").unwrap();
        match &circuit.analyses[0] {
            Analysis::Ac { sweep_type, .. } => {
                assert_eq!(*sweep_type, AcSweepType::Oct);
            }
            other => panic!("expected Ac, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_ac_case_insensitive() {
        let circuit = parse(".ac dec 10 1 100k").unwrap();
        assert_eq!(circuit.analyses.len(), 1);
        match &circuit.analyses[0] {
            Analysis::Ac { sweep_type, .. } => {
                assert_eq!(*sweep_type, AcSweepType::Dec);
            }
            other => panic!("expected Ac, got {:?}", other),
        }
    }

    // ---- Full netlist tests ----

    #[test]
    fn test_full_netlist() {
        let netlist = "\
* Simple RC circuit
V1 1 0 DC 5
R1 1 2 10k
C1 2 0 100n
.DC
.END
";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.components.len(), 3);
        assert_eq!(circuit.analyses.len(), 1);
    }

    #[test]
    fn test_full_netlist_with_ac() {
        let netlist = "\
* AC analysis example
V1 in 0 DC 0 AC 1 0
R1 in out 1k
C1 out 0 1u
.AC DEC 20 1 1MEG
.END
";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.components.len(), 3);
        assert_eq!(circuit.analyses.len(), 1);
        match &circuit.analyses[0] {
            Analysis::Ac {
                sweep_type,
                n_points,
                f_start,
                f_stop,
            } => {
                assert_eq!(*sweep_type, AcSweepType::Dec);
                assert_eq!(*n_points, 20);
                assert!((f_start - 1.0).abs() < 1e-12);
                assert!((f_stop - 1e6).abs() < 1.0);
            }
            other => panic!("expected Ac, got {:?}", other),
        }
    }

    #[test]
    fn test_comments_and_blank_lines() {
        let netlist = "\
* This is a comment

* Another comment
R1 1 0 1k

.DC
";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.components.len(), 1);
        assert_eq!(circuit.analyses.len(), 1);
    }

    #[test]
    fn test_end_stops_parsing() {
        let netlist = "\
R1 1 0 1k
.END
R2 2 0 2k
";
        let circuit = parse(netlist).unwrap();
        // R2 should not be parsed because .END was reached
        assert_eq!(circuit.components.len(), 1);
    }

    #[test]
    fn test_no_end_marker() {
        // Netlists without .END should still parse fine
        let netlist = "\
R1 1 0 1k
R2 2 0 2k
";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.components.len(), 2);
    }

    // ---- Error cases ----

    #[test]
    fn test_unknown_element() {
        let result = parse("X1 1 0 something");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("unknown element"));
    }

    #[test]
    fn test_malformed_resistor() {
        // Missing value
        let result = parse("R1 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_ac_command() {
        // Missing parameters
        let result = parse(".AC DEC");
        assert!(result.is_err());
    }

    // ---- Case insensitivity tests ----

    #[test]
    fn test_lowercase_component() {
        let circuit = parse("r1 1 0 10k").unwrap();
        assert_eq!(circuit.components.len(), 1);
        match &circuit.components[0] {
            Component::Resistor { name, .. } => {
                assert_eq!(name, "r1");
            }
            other => panic!("expected Resistor, got {:?}", other),
        }
    }

    #[test]
    fn test_mixed_case() {
        let circuit = parse("R1 Vdd GND 4.7K").unwrap();
        match &circuit.components[0] {
            Component::Resistor { nodes, value, .. } => {
                assert_eq!(nodes, &("Vdd".into(), "GND".into()));
                assert!((value - 4700.0).abs() < 1e-6);
            }
            other => panic!("expected Resistor, got {:?}", other),
        }
    }

    // ---- Edge cases ----

    #[test]
    fn test_empty_input() {
        let circuit = parse("").unwrap();
        assert!(circuit.components.is_empty());
        assert!(circuit.analyses.is_empty());
    }

    #[test]
    fn test_only_comments() {
        let circuit = parse("* just a comment\n* another one").unwrap();
        assert!(circuit.components.is_empty());
    }

    #[test]
    fn test_multiple_analyses() {
        let netlist = ".DC\n.AC DEC 10 1 100k";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.analyses.len(), 2);
    }

    #[test]
    fn test_inductor_with_scientific_notation() {
        let circuit = parse("L1 1 2 1e-6").unwrap();
        match &circuit.components[0] {
            Component::Inductor { value, .. } => {
                assert!((value - 1e-6).abs() < 1e-18);
            }
            other => panic!("expected Inductor, got {:?}", other),
        }
    }

    #[test]
    fn test_tabs_as_separators() {
        let circuit = parse("R1\t1\t0\t10k").unwrap();
        assert_eq!(circuit.components.len(), 1);
    }

    #[test]
    fn test_extra_whitespace() {
        let circuit = parse("  R1  1  0  10k  ").unwrap();
        assert_eq!(circuit.components.len(), 1);
    }

    #[test]
    fn test_negative_dc_source() {
        let circuit = parse("V1 1 0 DC -12").unwrap();
        match &circuit.components[0] {
            Component::VSource { dc, .. } => {
                assert!((dc.unwrap() - (-12.0)).abs() < 1e-12);
            }
            other => panic!("expected VSource, got {:?}", other),
        }
    }

    #[test]
    fn test_end_case_insensitive() {
        let netlist = "R1 1 0 1k\n.end\nR2 2 0 2k";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.components.len(), 1);
    }

    #[test]
    fn test_dot_command_case_insensitive() {
        let circuit = parse(".dc").unwrap();
        assert_eq!(circuit.analyses.len(), 1);
    }

    // ---- Diode element tests ----

    #[test]
    fn test_parse_diode_basic() {
        let circuit = parse("D1 1 2 MYDIODE").unwrap();
        assert_eq!(circuit.components.len(), 1);
        match &circuit.components[0] {
            Component::Diode { name, nodes, model } => {
                assert_eq!(name, "D1");
                assert_eq!(nodes, &("1".into(), "2".into()));
                assert_eq!(model, "MYDIODE");
            }
            other => panic!("expected Diode, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_diode_lowercase() {
        let circuit = parse("d2 anode cathode mymodel").unwrap();
        assert_eq!(circuit.components.len(), 1);
        match &circuit.components[0] {
            Component::Diode { name, nodes, model } => {
                assert_eq!(name, "d2");
                assert_eq!(nodes, &("anode".into(), "cathode".into()));
                assert_eq!(model, "mymodel");
            }
            other => panic!("expected Diode, got {:?}", other),
        }
    }

    // ---- .MODEL directive tests ----

    #[test]
    fn test_parse_model_all_params() {
        let circuit = parse(".MODEL MYDIODE D(IS=1e-14 N=1.0)").unwrap();
        assert_eq!(circuit.models.len(), 1);
        let m = &circuit.models[0];
        assert_eq!(m.name, "MYDIODE");
        assert!((m.is - 1e-14).abs() < 1e-26);
        assert!((m.n - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_model_defaults() {
        let circuit = parse(".MODEL D1N4148 D()").unwrap();
        assert_eq!(circuit.models.len(), 1);
        let m = &circuit.models[0];
        assert_eq!(m.name, "D1N4148");
        assert!((m.is - 1e-14).abs() < 1e-26);
        assert!((m.n - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_model_no_parens() {
        // .MODEL with no parameters — uses defaults
        let circuit = parse(".MODEL SIMPLE D").unwrap();
        assert_eq!(circuit.models.len(), 1);
        let m = &circuit.models[0];
        assert_eq!(m.name, "SIMPLE");
        assert!((m.is - 1e-14).abs() < 1e-26);
        assert!((m.n - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_model_case_insensitive() {
        let circuit = parse(".model mydiode d(is=2.52e-9 n=1.752)").unwrap();
        assert_eq!(circuit.models.len(), 1);
        let m = &circuit.models[0];
        assert_eq!(m.name, "mydiode");
        assert!((m.is - 2.52e-9).abs() < 1e-21);
        assert!((m.n - 1.752).abs() < 1e-12);
    }

    #[test]
    fn test_parse_model_param_order_reversed() {
        let circuit = parse(".MODEL DTEST D(N=1.5 IS=1e-12)").unwrap();
        let m = &circuit.models[0];
        assert!((m.is - 1e-12).abs() < 1e-24);
        assert!((m.n - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_parse_model_is_only() {
        let circuit = parse(".MODEL DTEST D(IS=5e-10)").unwrap();
        let m = &circuit.models[0];
        assert!((m.is - 5e-10).abs() < 1e-22);
        assert!((m.n - 1.0).abs() < 1e-12); // default
    }

    #[test]
    fn test_parse_model_n_only() {
        let circuit = parse(".MODEL DTEST D(N=2.0)").unwrap();
        let m = &circuit.models[0];
        assert!((m.is - 1e-14).abs() < 1e-26); // default
        assert!((m.n - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_multiple_diodes_one_model() {
        let netlist = "\
.MODEL DMOD D(IS=1e-14 N=1.0)
D1 1 2 DMOD
D2 3 4 DMOD
D3 5 0 DMOD
";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.models.len(), 1);
        assert_eq!(circuit.components.len(), 3);
        for comp in &circuit.components {
            match comp {
                Component::Diode { model, .. } => {
                    assert_eq!(model, "DMOD");
                }
                other => panic!("expected Diode, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_diode_in_full_netlist() {
        let netlist = "\
* Diode rectifier
V1 1 0 DC 5
R1 1 2 1k
.MODEL D1N4148 D(IS=2.52e-9 N=1.752)
D1 2 3 D1N4148
R2 3 0 1k
.DC
.END
";
        let circuit = parse(netlist).unwrap();
        assert_eq!(circuit.components.len(), 4); // V1, R1, D1, R2
        assert_eq!(circuit.models.len(), 1);
        assert_eq!(circuit.analyses.len(), 1);
        match &circuit.components[2] {
            Component::Diode { name, nodes, model } => {
                assert_eq!(name, "D1");
                assert_eq!(nodes, &("2".into(), "3".into()));
                assert_eq!(model, "D1N4148");
            }
            other => panic!("expected Diode, got {:?}", other),
        }
    }
}
