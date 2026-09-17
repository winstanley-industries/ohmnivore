"""Strict raw-output and numerical processing for the external EMI-01 study.

This module implements ADR-002's frozen reference measurement convention. It
does not participate in Ohmnivore's production simulator. Callers initialize
the hermetic NumPy runtime before importing it.
"""

import math
import re

import numpy as np


MAX_RAW_BYTES = 512 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_POINTS = 2_000_000
TIME_TOLERANCE = 1e-15


def _fail(kind, message):
    raise ValueError(f"{kind}: {message}")


def _finite_array(value, name, *, dimensions=1):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail("unsupported_input", f"{name} is not numeric")
    if result.ndim != dimensions or result.size == 0:
        _fail("unsupported_input", f"{name} must be a nonempty {dimensions}D array")
    if not np.isfinite(result).all():
        _fail("non_finite", name)
    return result


def parse_raw(data, expected_names, stop, max_step):
    """Decode exactly one little-endian ngspice real binary transient plot.

    Return an owned float64 matrix [point, vector], with time in column zero.
    Header counts and the complete payload length are checked before creating
    any arrays; concatenated plots, trailers and incomplete output are rejected.
    """
    if not isinstance(data, bytes):
        _fail("unsupported_input", "raw input must be bytes")
    if not data:
        _fail("missing_output", "empty raw output")
    if len(data) > MAX_RAW_BYTES:
        _fail("resource_limit", "raw byte budget exceeded")
    if (
        not math.isfinite(stop)
        or not math.isfinite(max_step)
        or stop <= 0
        or max_step <= 0
    ):
        _fail("unsupported_input", "positive finite stop and maximum step required")
    if (
        not expected_names
        or expected_names[0] != "time"
        or len(set(expected_names)) != len(expected_names)
    ):
        _fail(
            "unsupported_input", "unique expected vectors beginning with time required"
        )
    marker = b"\nBinary:\n"
    marker_start = data.find(marker, 0, MAX_HEADER_BYTES)
    if marker_start < 0:
        _fail("malformed_output", "missing bounded binary header")
    payload_start = marker_start + len(marker)
    try:
        lines = data[:marker_start].decode("ascii").splitlines()
    except UnicodeDecodeError:
        _fail("malformed_output", "header is not ASCII")
    if lines.count("Variables:") != 1:
        _fail("malformed_output", "exactly one Variables section required")
    variable_start = lines.index("Variables:")
    header = {}
    for line in lines[:variable_start]:
        if ":" not in line:
            _fail("malformed_output", "malformed header field")
        key, value = line.split(":", 1)
        if key in header:
            _fail("malformed_output", f"duplicate header field {key}")
        header[key] = value.strip()
    for name in ("Title", "Plotname", "Flags", "No. Variables", "No. Points"):
        if name not in header:
            _fail("malformed_output", f"missing header field {name}")
    if header["Flags"] != "real" or header["Plotname"] != "Transient Analysis":
        _fail("malformed_output", "expected a real transient plot")
    if not re.fullmatch(r"[0-9]{1,10}", header["No. Variables"]) or not re.fullmatch(
        r"[0-9]{1,10}", header["No. Points"]
    ):
        _fail("malformed_output", "invalid vector or point count")
    nvars, npoints = int(header["No. Variables"]), int(header["No. Points"])
    if npoints > MAX_POINTS:
        _fail("resource_limit", "raw point budget exceeded")
    if npoints < 2 or nvars != len(expected_names):
        _fail("malformed_output", "wrong vector or point count")
    variables = lines[variable_start + 1 :]
    if len(variables) != nvars:
        _fail("malformed_output", "variable schema length mismatch")
    names = []
    for index, line in enumerate(variables):
        fields = line.split()
        if len(fields) != 3 or fields[0] != str(index):
            _fail("malformed_output", "variable index or row malformed")
        name, unit = fields[1:]
        expected_unit = (
            "time"
            if index == 0
            else ("current" if name.startswith("i(") else "voltage")
        )
        if unit != expected_unit:
            _fail("malformed_output", f"wrong vector unit for {name}")
        names.append(name)
    if names != list(expected_names) or len(set(names)) != nvars:
        _fail("malformed_output", "missing, duplicated or reordered vectors")
    if len(data) - payload_start != npoints * nvars * 8:
        _fail("malformed_output", "truncated payload, trailer or point-count mismatch")
    values = np.frombuffer(
        data, dtype="<f8", count=npoints * nvars, offset=payload_start
    ).reshape(npoints, nvars)
    if not np.isfinite(values).all():
        _fail("non_finite", "raw waveform contains non-finite values")
    times = values[:, 0]
    if times[0] != 0.0 or abs(times[-1] - stop) > TIME_TOLERANCE:
        _fail("malformed_output", "waveform does not cover complete requested interval")
    gaps = np.diff(times)
    if np.any(gaps <= 0) or np.any(gaps > max_step + TIME_TOLERANCE):
        _fail("malformed_output", "non-monotonic time or excessive integration gap")
    return values.copy()


def cm_dm(a, b):
    """Average-current convention: Icm=(Ia+Ib)/2, Idm=(Ia-Ib)/2."""
    left, right = _finite_array(a, "conductor A"), _finite_array(b, "conductor B")
    if left.shape != right.shape:
        _fail("unsupported_input", "conductor shape mismatch")
    with np.errstate(over="ignore", invalid="ignore"):
        common, differential = (left + right) / 2, (left - right) / 2
    return _finite_array(common, "derived CM current"), _finite_array(
        differential, "derived DM current"
    )


def voltage_cm_dm(a, b):
    """Chassis-referenced voltage convention: Vcm=(Va+Vb)/2, Vdm=Va-Vb."""
    common, half_difference = cm_dm(a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        differential = 2 * half_difference
    return common, _finite_array(differential, "derived differential voltage")


def resample(t, x, start=100e-6, stop=200e-6, dt=1.25e-9):
    """Linearly interpolate a half-open uniform interval without extrapolation."""
    times, values = _finite_array(t, "time"), _finite_array(x, "waveform")
    if times.shape != values.shape or len(times) < 2 or np.any(np.diff(times) <= 0):
        _fail(
            "unsupported_input", "strictly increasing paired time and waveform required"
        )
    if (
        not all(math.isfinite(item) for item in (start, stop, dt))
        or dt <= 0
        or stop <= start
    ):
        _fail("unsupported_input", "invalid sampling interval")
    count_float = (stop - start) / dt
    count = round(count_float)
    if count < 4 or count > MAX_POINTS or abs(count_float - count) > 1e-7:
        _fail(
            "unsupported_input",
            "sampling interval must contain an integral bounded sample count",
        )
    grid = start + np.arange(count, dtype=np.float64) * dt
    if start < times[0] or grid[-1] > times[-1] or stop > times[-1] + TIME_TOLERANCE:
        _fail("missing_output", "sampling interval is not bracketed by waveform")
    return grid, np.interp(grid, times, values)


def spectrum(
    t,
    x,
    start=100e-6,
    stop=200e-6,
    dt=1.25e-9,
    *,
    frequency_min=150e3,
    frequency_max=10e6,
):
    """Return frequency Hz and one-sided Hann coherent sinusoidal RMS amplitude.

    This is a tone-amplitude spectrum, not a noise PSD. The contract band excludes
    DC and Nyquist; optional band arguments support independent analytic tests.
    """
    _, values = resample(t, x, start, stop, dt)
    if (
        not all(math.isfinite(item) for item in (frequency_min, frequency_max))
        or frequency_min <= 0
        or frequency_max < frequency_min
        or frequency_max >= 0.5 / dt
    ):
        _fail("unsupported_input", "spectral band must be positive and below Nyquist")
    count = len(values)
    window = (1 - np.cos(2 * np.pi * np.arange(count) / count)) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.fft.rfft(window * (values - np.mean(values)))
    frequency = np.fft.rfftfreq(count, dt)
    amplitude = _finite_array(
        np.sqrt(2) * np.abs(transformed) / np.sum(window), "derived spectrum"
    )
    # Floating construction of a 100-us interval can put a nominal endpoint a
    # few ULPs away from its decimal value. Use a sub-bin numerical tolerance.
    tolerance = np.finfo(np.float64).eps * max(frequency_max, 1.0) * 8
    retained = (frequency >= frequency_min - tolerance) & (
        frequency <= frequency_max + tolerance
    )
    if not np.any(retained):
        _fail("unsupported_input", "spectral band contains no bins")
    return frequency[retained], amplitude[retained]


def dbua(amplitude):
    values = _finite_array(amplitude, "spectrum")
    if np.any(values < 0):
        _fail("unsupported_input", "negative spectral amplitude")
    return 20 * np.log10(np.maximum(values, 1e-15) / 1e-6)


def spectrum_comparison(actual, reference):
    """Evaluate the ADR's 1-dB / 1-uA binwise adjacent-refinement gate."""
    left, right = (
        _finite_array(actual, "actual spectrum"),
        _finite_array(reference, "reference spectrum"),
    )
    if left.shape != right.shape or np.any(left < 0) or np.any(right < 0):
        _fail("unsupported_input", "invalid spectral comparison shapes or amplitude")
    high = np.maximum(left, right) > 1e-5  # 20 dBuA.
    db_difference = np.abs(dbua(left) - dbua(right))
    absolute_difference = np.abs(left - right)
    high_failed = high & (db_difference > 1)
    low_failed = ~high & (absolute_difference > 1e-6)
    return {
        "passed": not bool(np.any(high_failed | low_failed)),
        "bins": len(left),
        "high_bins": int(np.count_nonzero(high)),
        "failed_bins": int(np.count_nonzero(high_failed | low_failed)),
        "max_high_difference_db": float(np.max(db_difference[high], initial=0)),
        "max_low_difference_a": float(np.max(absolute_difference[~high], initial=0)),
    }


def waveform_comparison(actual, reference, absolute=0.02, relative=0.02):
    """Evaluate a waveform RMS difference against absolute + relative RMS."""
    left, right = (
        _finite_array(actual, "actual waveform"),
        _finite_array(reference, "reference waveform"),
    )
    if left.shape != right.shape or not all(
        math.isfinite(value) and value >= 0 for value in (absolute, relative)
    ):
        _fail("unsupported_input", "invalid waveform comparison")
    with np.errstate(over="ignore", invalid="ignore"):
        error = float(np.sqrt(np.mean((left - right) ** 2)))
        reference_rms = float(np.sqrt(np.mean(right**2)))
    if not math.isfinite(error) or not math.isfinite(reference_rms):
        _fail("non_finite", "derived RMS comparison")
    limit = absolute + relative * reference_rms
    return {
        "passed": error <= limit,
        "difference_rms": error,
        "reference_rms": reference_rms,
        "limit": limit,
    }
