#[cfg(feature = "ngspice-compare")]
mod regression;

#[cfg(feature = "ngspice-compare")]
regression::regression_tests!(
    voltage_divider,
    current_source_resistor,
    rc_lowpass,
    diode_forward,
    npn_common_emitter,
    nmos_common_source,
);
