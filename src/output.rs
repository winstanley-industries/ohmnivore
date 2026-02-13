//! Results output formatting (CSV).

use crate::analysis::{AcResult, DcResult};
use crate::error::Result;
use std::io::Write;

/// Write DC operating point results as CSV.
///
/// Format:
/// ```csv
/// Variable,Value
/// V(1),5.0
/// V(2),3.3
/// I(V1),0.001
/// ```
pub fn write_dc_csv<W: Write>(result: &DcResult, writer: &mut W) -> Result<()> {
    writeln!(writer, "Variable,Value")?;
    for (name, voltage) in &result.node_voltages {
        writeln!(writer, "V({}),{}", name, voltage)?;
    }
    for (name, current) in &result.branch_currents {
        writeln!(writer, "I({}),{}", name, current)?;
    }
    Ok(())
}

/// Write AC sweep results as CSV.
///
/// Format:
/// ```csv
/// Frequency,V(1)_mag,V(1)_phase_deg,V(2)_mag,V(2)_phase_deg,...
/// 1.0,1.0,0.0,0.707,-45.0
/// 10.0,0.995,-5.7,0.701,-50.7
/// ```
pub fn write_ac_csv<W: Write>(result: &AcResult, writer: &mut W) -> Result<()> {
    // Header row
    write!(writer, "Frequency")?;
    for (name, _) in &result.node_voltages {
        write!(writer, ",V({})_mag,V({})_phase_deg", name, name)?;
    }
    for (name, _) in &result.branch_currents {
        write!(writer, ",I({})_mag,I({})_phase_deg", name, name)?;
    }
    writeln!(writer)?;

    // Data rows
    for (fi, freq) in result.frequencies.iter().enumerate() {
        write!(writer, "{}", freq)?;
        for (_, voltages) in &result.node_voltages {
            let v = voltages[fi];
            write!(writer, ",{},{}", v.norm(), v.arg().to_degrees())?;
        }
        for (_, currents) in &result.branch_currents {
            let c = currents[fi];
            write!(writer, ",{},{}", c.norm(), c.arg().to_degrees())?;
        }
        writeln!(writer)?;
    }
    Ok(())
}
