"""Reference metrics; arrays first pass the strict raw waveform validator."""

import math

from third_party.emi_python.runtime import load_numpy

np = load_numpy()

from reference.emi01 import circuits, signals  # noqa: E402


def columns(raw, names):
    if raw.ndim != 2 or raw.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("malformed_output: waveform column schema mismatch")
    if not np.isfinite(raw).all():
        raise ValueError("non_finite: metric input waveform")
    return {name: raw[:, index] for index, name in enumerate(names)}


def finite_metrics(value):
    """Reject every non-finite derived quantity before feasibility or ranking."""
    if isinstance(value, dict):
        for item in value.values():
            finite_metrics(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_metrics(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError("non_finite: derived metric")


def rms(x):
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sqrt(np.mean(np.square(x))))
    finite_metrics(value)
    return value


def switch_voltages(d):
    return {
        "vds_ah": d["v(p)"] - d["v(a)"],
        "vds_al": d["v(a)"],
        "vds_bh": d["v(p)"] - d["v(b)"],
        "vds_bl": d["v(b)"],
    }


def evaluate(raw, candidate, corner, dt):
    d = columns(raw, circuits.STUDY_NAMES)
    t = d["time"]
    grid, _ = signals.resample(t, d["i(va)"], dt=dt)
    y = {
        name: np.interp(grid, t, values) for name, values in d.items() if name != "time"
    }
    ia, ib = y["i(va)"], y["i(vb)"]
    cm, dm = signals.cm_dm(ia, ib)
    vcm, vdm = signals.voltage_cm_dm(y["v(fa)"] - y["v(ch)"], y["v(fb)"] - y["v(ch)"])
    raw_cm, raw_dm = signals.cm_dm(d["i(va)"], d["i(vb)"])
    spectra = {}
    for name, wave in [
        ("a", d["i(va)"]),
        ("b", d["i(vb)"]),
        ("cm", raw_cm),
        ("dm", raw_dm),
    ]:
        frequency, amplitude = signals.spectrum(t, wave, dt=dt)
        spectra[name] = amplitude
    margins = {
        name: float(90 - np.max(signals.dbua(amplitude)))
        for name, amplitude in spectra.items()
    }
    period_n = round(20e-6 / dt)
    settled = {}
    for name, wave in [("a", ia), ("b", ib)]:
        error = rms(wave[-period_n:] - wave[-2 * period_n : -period_n])
        limit = 0.01 + 0.01 * rms(wave[-period_n:])
        settled[name] = {
            "rms_difference_a": error,
            "limit_a": limit,
            "pass": error <= limit,
        }
    physical = circuits.design(candidate)
    stress = {}
    stress["device_peak_v"] = max(
        float(np.max(np.abs(v))) for v in switch_voltages(d).values()
    )
    stress["device_peak_a"] = max(
        float(np.max(np.abs(d[f"i({n})"]))) for n in ["vdah", "vdal", "vdbh", "vdbl"]
    )
    stress["capacitor_peak_v"] = max(
        float(np.max(np.abs(v)))
        for v in [
            d["v(xcap)"] - d["v(fb)"],
            d["v(yca)"] - d["v(ch)"],
            d["v(ycb)"] - d["v(ch)"],
        ]
    )
    stress["winding_rms_a"] = max(
        rms(y[f"i({n})"]) for n in ["lda", "ldb", "lcma", "lcmb"]
    )
    # Multiplying NumPy scalars lets the common finite-result guard classify any
    # derived overflow, instead of leaking Python OverflowError as internal_failure.
    copper_loss = sum(
        physical["rdm_ohm"] * np.float64(rms(y[f"i(ld{leg})"])) ** 2
        + physical["rcm_ohm"] * np.float64(rms(y[f"i(lcm{leg})"])) ** 2
        for leg in ["a", "b"]
    )
    x_current = (y["v(fa)"] - y["v(xcap)"]) / 0.2
    y_currents = [(y["v(fa)"] - y["v(yca)"]) / 22, (y["v(fb)"] - y["v(ycb)"]) / 22]
    stress["damping_a_w"], stress["damping_b_w"] = [
        21.8 * np.float64(rms(current)) ** 2 for current in y_currents
    ]
    capacitor_loss = 0.2 * sum(
        np.float64(rms(current)) ** 2 for current in [x_current, *y_currents]
    )
    stress["loss_w"] = float(
        copper_loss + capacitor_loss + stress["damping_a_w"] + stress["damping_b_w"]
    )
    for kind in ["dm", "cm"]:
        geometry = candidate[f"{kind}_geometry"]
        scale = (
            candidate[f"l{kind}_h"]
            * corner["l_scale"]
            / (geometry[2] * geometry[0] * 1e-6)
        )
        if kind == "dm":
            currents = [d["i(lda)"], d["i(ldb)"]]
        else:
            a, b = d["i(lcma)"], d["i(lcmb)"]
            currents = [a + 0.995 * b, b + 0.995 * a]
        stress[f"{kind}_peak_t"] = max(
            float(np.max(np.abs(x * scale))) for x in currents
        )
    limits = {
        "device_peak_v": 960,
        "device_peak_a": 50,
        "capacitor_peak_v": 504,
        "winding_rms_a": 12,
        "loss_w": 25,
        "dm_peak_t": 0.2,
        "cm_peak_t": 0.2,
        "damping_a_w": 12,
        "damping_b_w": 12,
    }
    result = {
        "research_margin_db": margins,
        "stress": stress,
        "stress_limits": limits,
        "settling": settled,
        "mass_kg": physical["mass_kg"],
        "rms_cm_a": rms(cm),
        "rms_dm_a": rms(dm),
        "rms_vcm_v": rms(vcm),
        "rms_vdm_v": rms(vdm),
        "rms_chassis_return_a": rms(y["i(lch)"]),
        "load_series_resistor_w": float(
            np.float64(rms(y["v(la)"] - y["v(lr)"])) ** 2 / 20
        ),
        "load_winding_loss_surrogate_w": float(
            np.float64(rms(y["v(lr)"] - y["v(lb)"])) ** 2 / 100
        ),
        "rms_load_inductor_a": rms(y["i(lload)"]),
    }
    finite_metrics(result)
    violations = [key for key, limit in limits.items() if stress[key] > limit]
    violations += [
        "research_mask_" + key for key, margin in margins.items() if margin < 6
    ]
    result["violations"] = violations
    result["status"] = (
        "unsettled"
        if not all(x["pass"] for x in settled.values())
        else "predicted_infeasible"
        if violations
        else "predicted_feasible"
    )
    return result, np.column_stack(
        [frequency] + [spectra[k] for k in ["a", "b", "cm", "dm"]]
    )


def _cross(t, x, level, lo, hi, falling):
    mask = (t >= lo) & (t <= hi)
    tt, xx = t[mask], x[mask]
    pairs = np.where(
        (xx[:-1] >= level) & (xx[1:] < level)
        if falling
        else (xx[:-1] <= level) & (xx[1:] > level)
    )[0]
    if not len(pairs):
        raise ValueError("accuracy_failure: switching crossing missing")
    i = int(pairs[0])
    return float(tt[i] + (level - xx[i]) * (tt[i + 1] - tt[i]) / (xx[i + 1] - xx[i]))


def ringing(t, voltage):
    """First three same-polarity peaks after the frozen off-edge exclusion."""
    mask = (t > 14.04e-6) & (t < 14.4e-6)
    tt, residual = t[mask], voltage[mask] - 400
    candidates = []
    diagnostics = {}
    for polarity, sign in [("positive", 1), ("negative", -1)]:
        vv = sign * residual
        peaks = (
            np.where((vv[1:-1] > vv[:-2]) & (vv[1:-1] >= vv[2:]) & (vv[1:-1] > 0.2))[0]
            + 1
        )
        diagnostics[polarity + "_peaks_above_floor"] = int(len(peaks))
        if len(peaks) < 3:
            continue
        p = peaks[:3]
        if not (vv[p[0]] > vv[p[1]] > vv[p[2]]):
            continue
        candidates.append(
            {
                "status": "measured",
                "polarity": polarity,
                "residual_reference_v": 400.0,
                "peak_times_s": tt[p].tolist(),
                "peak_amplitudes_v": vv[p].tolist(),
                "frequency_hz": float(1 / np.mean(np.diff(tt[p]))),
                "log_decrement": float(np.mean(np.log(vv[p[:-1]] / vv[p[1:]]))),
            }
        )
    if not candidates:
        return {
            "status": "unresolved",
            "reason": "first three same-polarity peaks above 0.2 V are absent or not strictly decaying",
            **diagnostics,
        }
    return min(candidates, key=lambda item: item["peak_times_s"][0])


def dpt(raw, sample_step_s=0.25e-9):
    source = columns(raw, circuits.DPT_NAMES)
    t, _ = signals.resample(source["time"], source["v(d)"], 0, 16e-6, sample_step_s)
    t = np.append(t, 16e-6)  # Include the endpoint for bounded energy integration.
    d = {
        name: np.interp(t, source["time"], values)
        for name, values in source.items()
        if name != "time"
    }
    v, current = d["v(d)"], d["i(vdrain)"]
    result = {
        "sample_step_s": sample_step_s,
        "peak_v": float(np.max(v)),
        "min_v": float(np.min(v)),
        "peak_gate_v": float(np.max(d["v(g)"])),
        "min_gate_v": float(np.min(d["v(g)"])),
        "load_at_second_on_a": float(np.interp(12e-6, t, d["i(lload)"])),
    }
    for label, event, falling in [("on", 12e-6, True), ("off", 14e-6, False)]:
        lo, hi = event - 0.2e-6, event + 0.2e-6
        crossings = [_cross(t, v, level, lo, hi, falling) for level in [40, 200, 360]]
        result[f"{label}_edge_s"] = abs(crossings[2] - crossings[0])
        result[f"{label}_half_s"] = crossings[1]
        result[f"{label}_gate_at_vds_10_50_90_v"] = np.interp(
            crossings, t, d["v(g)"]
        ).tolist()
        mask = (t > lo) & (t < hi)
        tt = np.concatenate(([lo], t[mask], [hi]))
        result[f"{label}_energy_j"] = float(
            np.trapezoid(np.interp(tt, t, v) * np.interp(tt, t, current), tt)
        )
    gate_on = result["on_gate_at_vds_10_50_90_v"]
    gate_drop = gate_on[2] - gate_on[0]
    gate_mids = [result[f"{edge}_gate_at_vds_10_50_90_v"][1] for edge in ["on", "off"]]
    result["gate_dynamics"] = {
        "on_edge_gate_drop_v": gate_drop,
        "drop_limits_v": [0.2, 30.0],
        "half_crossing_gate_limits_v": [-3.0, 23.0],
        "pass": bool(
            0.2 <= gate_drop <= 30 and all(-3 <= gate <= 23 for gate in gate_mids)
        ),
    }
    result["ringing"] = ringing(t, v)
    finite_metrics(result)
    result["pass"] = bool(
        400 < result["peak_v"] < 960
        and result["load_at_second_on_a"] >= 10
        and result["gate_dynamics"]["pass"]
        and result["ringing"]["status"] == "measured"
        and all(
            0 < result[f"{edge}_energy_j"] < 0.002
            and 0 < result[f"{edge}_edge_s"] < 200e-9
            for edge in ["on", "off"]
        )
    )
    return result


def _waveform_check(actual, reference, voltage, bus):
    checked = signals.waveform_comparison(
        actual,
        reference,
        absolute=2 + 0.02 * bus if voltage else 0.02,
        relative=0 if voltage else 0.02,
    )
    return {
        "rms_error": checked["difference_rms"],
        "limit": checked["limit"],
        "pass": checked["passed"],
    }


def compare(a, b, names, dt, bus=400):
    """Compare fine b against a without shifting or optimizing alignment."""
    result = {"pass": True, "waveforms": {}, "spectra": {}}
    da, db = columns(a, names), columns(b, names)
    grid, _ = signals.resample(db["time"], db["i(va)"], dt=dt)
    wa, wb = (
        {name: da[name] for name in ["i(va)", "i(vb)"]},
        {name: db[name] for name in ["i(va)", "i(vb)"]},
    )
    wa.update(switch_voltages(da))
    wb.update(switch_voltages(db))
    for name in wa:
        x, y = (
            np.interp(grid, da["time"], wa[name]),
            np.interp(grid, db["time"], wb[name]),
        )
        check = _waveform_check(x, y, name.startswith("vds"), bus)
        result["waveforms"][name] = check
        result["pass"] &= check["pass"]
    waves = []
    for d in [da, db]:
        ia, ib = d["i(va)"], d["i(vb)"]
        waves.append([ia, ib, *signals.cm_dm(ia, ib)])
    for i, name in enumerate(["a", "b", "cm", "dm"]):
        _, x = signals.spectrum(da["time"], waves[0][i], dt=dt)
        _, y = signals.spectrum(db["time"], waves[1][i], dt=dt)
        check = spectral_check(x, y)
        result["spectra"][name] = check
        result["pass"] &= check["pass"]
    result["pass"] = bool(result["pass"])
    return result


def output_sampling_compare(raw, names, coarse_dt, fine_dt=1.25e-9, bus=400):
    """Separate waveform-grid error from integration error on one finest solve."""
    d = columns(raw, names)
    fine_grid, _ = signals.resample(d["time"], d["i(va)"], dt=fine_dt)
    coarse_grid, _ = signals.resample(d["time"], d["i(va)"], dt=coarse_dt)
    coarse_grid = np.append(coarse_grid, 200e-6)  # Bracket the final fine-grid point.
    result = {"pass": True, "waveforms": {}, "spectra": {}}
    waves = {name: d[name] for name in ["i(va)", "i(vb)"]}
    waves.update(switch_voltages(d))
    for name, wave in waves.items():
        coarse = np.interp(coarse_grid, d["time"], wave)
        reconstructed = np.interp(fine_grid, coarse_grid, coarse)
        fine = np.interp(fine_grid, d["time"], wave)
        check = _waveform_check(reconstructed, fine, name.startswith("vds"), bus)
        result["waveforms"][name] = check
        result["pass"] &= check["pass"]
    ia, ib = d["i(va)"], d["i(vb)"]
    for name, wave in zip(["a", "b", "cm", "dm"], [ia, ib, *signals.cm_dm(ia, ib)]):
        _, coarse = signals.spectrum(d["time"], wave, dt=coarse_dt)
        _, fine = signals.spectrum(d["time"], wave, dt=fine_dt)
        check = spectral_check(coarse, fine)
        result["spectra"][name] = check
        result["pass"] &= check["pass"]
    result["pass"] = bool(result["pass"])
    return result


def spectral_check(a, b):
    checked = signals.spectrum_comparison(a, b)
    return {
        "max_db_above_floor": checked["max_high_difference_db"],
        "max_a_below_floor": checked["max_low_difference_a"],
        "failed_bins": checked["failed_bins"],
        "pass": checked["passed"],
    }
