"""Independent EMI-01 v2 numerical review; run only outside benchmark timing.

This diagnostic uses the pinned NumPy runtime but does not import production
parsing, spectral, stress, qualification, or ranking helpers. It recomputes the
30 qualification jobs in a complete 30-job exploration or 102-job invocation.
It does not replace the harness's full terminal/provenance/accounting audit.
"""

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError(
        "Run this assertion-based diagnostic without Python optimization"
    )

from third_party.emi_python.runtime import load_numpy  # noqa: E402

np = load_numpy()

STRESS_LIMITS = {
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


def design_mass(candidate):
    """Independent SI evaluation of the declared hypothetical material volumes."""
    mass = 0.010  # Two 5 g damping resistors.
    for kind, cores, windings in [("dm", 2, 1), ("cm", 1, 2)]:
        area, length, turns, turn_length = candidate[kind + "_geometry"]
        core_volume = area * 1e-6 * length * 1e-3
        copper_volume = turns * turn_length * 1e-3 * 4e-6
        mass += 1.2 * cores * (4800 * core_volume + windings * 8960 * copper_volume)
    for capacitance in [candidate["cx_f"], candidate["cy_f"], candidate["cy_f"]]:
        energy_j = 0.5 * capacitance * 630**2
        volume_cm3 = 2 * energy_j / 0.2
        mass += 0.001 + volume_cm3 * 1.2e-3
    assert math.isfinite(mass) and mass > 0
    return mass


def contract(manifest):
    """Pin the reviewed v2 domain independently of production manifest helpers."""
    assert manifest["schema"] == "emi01-v2"
    assert manifest["model"] == "MSC040SMA120B"
    assert (
        manifest["model_archive_sha256"]
        == "de4a3acf222cbc7f6c1a4d1a2bc449dd31b5db2c6ed153920d797752c3831d51"
    )
    candidates = [
        ("light", 10e-6, 100e-6, 47e-9, 1e-9, [100, 60, 20, 45], [80, 60, 12, 40]),
        (
            "boundary",
            330e-6,
            1e-3,
            80e-9,
            47e-9,
            [1200, 160, 48, 135],
            [800, 140, 40, 110],
        ),
        (
            "reference",
            330e-6,
            1e-3,
            1e-6,
            47e-9,
            [1200, 160, 48, 135],
            [800, 140, 40, 110],
        ),
    ]
    keys = ["id", "ldm_h", "lcm_h", "cx_f", "cy_f", "dm_geometry", "cm_geometry"]
    assert manifest["candidates"] == [dict(zip(keys, row)) for row in candidates]
    corners = [
        ("nominal", 400, 27, 1, 1, 1, 1),
        ("fast_low_lc", 440, 27, 0.9, 0.9, 1.2, 0.8),
        ("hot_high_c", 360, 125, 1.1, 1.1, 1.2, 1.2),
    ]
    keys = ["id", "bus_v", "tj_c", "l_scale", "c_scale", "stray_scale", "rg_scale"]
    assert manifest["corners"] == [dict(zip(keys, row)) for row in corners]
    assert manifest["ensemble_max_steps_s"] == [2.5e-9, 1.25e-9, 0.625e-9]
    assert manifest["candidate_max_steps_s"] == {
        name: [0.625e-9, 0.3125e-9, 0.15625e-9] for name in ["boundary", "reference"]
    }
    assert manifest["ensemble_sample_steps_s"] == [5e-9, 2.5e-9, 1.25e-9]
    assert manifest["dpt_max_steps_s"] == [2e-9, 1e-9, 0.5e-9]
    assert manifest["dpt_sample_steps_s"] == [1e-9, 0.5e-9, 0.25e-9]
    assert manifest["observation_s"] == [100e-6, 200e-6]
    assert manifest["band_hz"] == [150e3, 10e6]
    assert manifest["research_mask_dbua"] == 90 and manifest["required_margin_db"] == 6
    assert manifest["spectral_accuracy"] == {
        "floor_a": 1e-5,
        "above_floor_db": 1,
        "below_floor_absolute_a": 1e-6,
    }
    assert manifest["reference_case_gates"] == {
        "passing_candidate": "reference",
        "boundary_candidate": "boundary",
        "boundary_margin_db": [5, 7],
        "boundary_requires_all_physical_screens": True,
        "failing_candidate": "light",
        "failing_requires_all_corners": True,
    }
    assert manifest["reference_case_gates"]["failing_requires_all_corners"] is True
    assert (
        manifest["reference_case_gates"]["boundary_requires_all_physical_screens"]
        is True
    )
    assert manifest["workers"] == [1, 4]
    assert manifest["warmups"] == 1 and manifest["samples"] == 3
    assert manifest["qualification_jobs"] == 30 and manifest["expected_jobs"] == 102
    assert manifest["y_series_damping_ohm"] == 21.8
    assert manifest["damping_resistor_mass_kg"] == 0.005
    assert manifest["damping_resistor_rated_w"] == 15
    assert manifest["damping_resistor_limit_w"] == 12
    assert manifest["load_winding_parallel_loss_ohm"] == 100
    assert manifest["numerical_options"] == {
        "reltol": 1e-5,
        "abstol_a": 1e-9,
        "vntol_v": 1e-7,
        "chgtol_c": 1e-16,
        "method": "gear",
        "maxord": 2,
        "itl1": 300,
        "itl4": 100,
    }


def raw(root, identity):
    folder = Path(root) / "jobs" / identity
    header = (folder / "raw.header").read_bytes()
    index = json.loads((folder / "raw.json").read_text())
    record = json.loads((folder / "result.json").read_text())
    assert index["raw_sha256"] == record["raw_sha256"]
    chunks = []
    for item in index["chunks"]:
        block = gzip.decompress(
            (Path(root) / "blobs" / (item["sha256"] + ".gz")).read_bytes()
        )
        assert len(block) == item["bytes"]
        assert hashlib.sha256(block).hexdigest() == item["sha256"]
        chunks.append(block)
    payload = b"".join(chunks)
    assert len(payload) == index["payload_bytes"]
    assert hashlib.sha256(header + payload).hexdigest() == index["raw_sha256"]
    names = [
        line.split()[1]
        for line in header.decode("ascii")
        .split("Variables:\n")[1]
        .split("Binary:\n")[0]
        .strip()
        .splitlines()
    ]
    data = np.frombuffer(payload, dtype="<f8").reshape(-1, len(names))
    assert np.isfinite(data).all()
    return {name: data[:, i] for i, name in enumerate(names)}


def spectrum(d, dt):
    n = round(100e-6 / dt)
    time = 100e-6 + np.arange(n) * dt
    window = np.array([0.5 - 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)])
    a = np.interp(time, d["time"], d["i(va)"])
    b = np.interp(time, d["time"], d["i(vb)"])
    frequency = np.arange(n // 2 + 1) / (n * dt)
    band = (frequency >= 150e3 - 1e-5) & (frequency <= 10e6 + 1e-5)
    waves = [a, b, 0.5 * (a + b), 0.5 * (a - b)]
    return frequency[band], {
        name: (
            np.sqrt(2) * abs(np.fft.rfft(window * (wave - np.mean(wave)))) / sum(window)
        )[band]
        for name, wave in zip(["a", "b", "cm", "dm"], waves)
    }


def compare(name, a, b, dt_a, dt_b):
    frequency, x = spectrum(a, dt_a)
    _, y = spectrum(b, dt_b)
    results = []
    for observable in x:
        left, right = x[observable], y[observable]
        absolute = abs(left - right)
        db = abs(20 * np.log10(np.maximum(left, 1e-15) / np.maximum(right, 1e-15)))
        above = np.maximum(left, right) > 10e-6
        failed = (above & (db > 1)) | (~above & (absolute > 1e-6))
        worst = int(np.argmax(absolute))
        results.append(
            {
                "id": name,
                "observable": observable,
                "failed_bins": int(sum(failed)),
                "below_floor_max_abs_uA": float(
                    np.max(absolute[~above], initial=0) * 1e6
                ),
                "above_floor_max_db": float(np.max(db[above], initial=0)),
                "all_max_abs_uA": float(absolute[worst] * 1e6),
                "worst_frequency_hz": float(frequency[worst]),
                "worst_amplitudes_uA": [
                    float(left[worst] * 1e6),
                    float(right[worst] * 1e6),
                ],
                "peak_detector_delta_db": float(
                    abs(20 * math.log10(max(left) / max(right)))
                ),
            }
        )
    return results


parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(args.root)
manifest = json.loads((root / "manifest.json").read_text())
terminal = json.loads((root / "terminal.json").read_text())
contract(manifest)
assert terminal["schema"] == "emi01-v2"
assert type(terminal["qualification_only"]) is bool
expected = [
    f"q{level}-dpt"
    if candidate is None
    else f"q{level}-ensemble-{candidate['id']}-{corner['id']}"
    for level in range(3)
    for candidate, corner in [(None, None)]
    + [(c, k) for c in manifest["candidates"] for k in manifest["corners"]]
]
all_expected = expected.copy()
if not terminal["qualification_only"]:
    for workers in [1, 4]:
        for phase in ["warmup1", "sample0", "sample1", "sample2"]:
            all_expected.extend(
                f"w{workers}-{phase}-ensemble-{candidate['id']}-{corner['id']}"
                for candidate in manifest["candidates"]
                for corner in manifest["corners"]
            )
assert terminal["job_ids"] == all_expected
assert len(expected) == 30 and len(set(all_expected)) == len(all_expected)
report = {
    "source": "Independent raw payload, Fourier, waveform, DPT and stress recomputation; diagnostic, not performance evidence",
    "manifest_sha256": hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest(),
    "terminal_sha256": hashlib.sha256(
        (root / "terminal.json").read_bytes()
    ).hexdigest(),
    "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "spectral_comparisons": [],
    "waveform_comparisons": [],
    "settling": [],
    "metric_errors": [],
    "dpt": [],
    "cases": [],
    "roles": [],
    "qualification_checks": [],
}


def sample(d, dt, start=100e-6, stop=200e-6, endpoint=False):
    n = round((stop - start) / dt)
    t = start + np.arange(n + int(endpoint)) * dt
    return t, {
        name: np.interp(t, d["time"], wave)
        for name, wave in d.items()
        if name != "time"
    }


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def waveforms(d):
    return [
        d["i(va)"],
        d["i(vb)"],
        d["v(p)"] - d["v(a)"],
        d["v(a)"],
        d["v(p)"] - d["v(b)"],
        d["v(b)"],
    ]


def waveform_checks(identity, a, b, dt, bus, coarse_sampling=None):
    grid = 100e-6 + np.arange(round(100e-6 / dt)) * dt
    for name, x, y in zip(
        ["ia", "ib", "vds_ah", "vds_al", "vds_bh", "vds_bl"], waveforms(a), waveforms(b)
    ):
        if coarse_sampling:
            coarse = (
                100e-6
                + np.arange(round(100e-6 / coarse_sampling) + 1) * coarse_sampling
            )
            xx = np.interp(grid, coarse, np.interp(coarse, a["time"], x))
        else:
            xx = np.interp(grid, a["time"], x)
        yy = np.interp(grid, b["time"], y)
        error = rms(xx - yy)
        limit = 0.02 + 0.02 * rms(yy) if name.startswith("i") else 2 + 0.02 * bus
        report["waveform_comparisons"].append(
            {
                "id": identity,
                "channel": name,
                "error": error,
                "limit": limit,
                "pass": error <= limit,
            }
        )


def near(label, actual, expected, atol=1e-10, rtol=1e-10):
    assert math.isfinite(float(actual)) and math.isfinite(float(expected)), label
    error = abs(float(actual) - float(expected))
    limit = atol + rtol * abs(float(expected))
    if error > limit:
        report["metric_errors"].append(
            {
                "id": label,
                "actual": float(actual),
                "expected": float(expected),
                "error": error,
                "limit": limit,
            }
        )


def role_check(name, level, rows):
    """A failing control must contain valid infeasibility at every corner."""
    assert name in {"light", "boundary", "reference"}
    assert [x["corner"] for x in rows] == ["nominal", "fast_low_lc", "hot_high_c"]
    worst = min(rows, key=lambda row: row["minimum_margin_db"])
    numerical = all(
        x["settled"] and x["status"] in {"predicted_feasible", "predicted_infeasible"}
        for x in rows
    )
    physical = numerical and all(not x["physical_violations"] for x in rows)
    feasible = numerical and all(x["status"] == "predicted_feasible" for x in rows)
    infeasible = numerical and all(x["status"] == "predicted_infeasible" for x in rows)
    if name == "light":
        passes = infeasible
    elif name == "boundary":
        passes = physical and 5 <= worst["minimum_margin_db"] <= 7
    else:
        passes = physical and feasible
    return {
        "candidate": name,
        "level": level,
        "statuses_by_corner": {x["corner"]: x["status"] for x in rows},
        "all_numerically_valid": bool(numerical),
        "all_physical_screens": bool(physical),
        "predicted_feasible_all_corners": bool(feasible),
        "predicted_infeasible_all_corners": bool(infeasible),
        "worst_corner": worst["corner"],
        "worst_channel": worst["worst_channel"],
        "worst_frequency_hz": worst["worst_frequency_hz"],
        "minimum_margin_db": worst["minimum_margin_db"],
        "distance_from_feasibility_db": worst["minimum_margin_db"] - 6,
        "mass_kg": worst["mass_kg"],
        "pass": bool(passes),
    }


for candidate in manifest["candidates"]:
    for corner in manifest["corners"]:
        identity = candidate["id"] + "-" + corner["id"]
        data = []
        for level in range(3):
            jid = f"q{level}-ensemble-{identity}"
            d = raw(root, jid)
            data.append(d)
            record = json.loads((root / "jobs" / jid / "result.json").read_text())
            assert record["candidate"] == candidate and record["corner"] == corner
            assert record["id"] == jid and record["level"] == level
            step = record["max_step_s"]
            dt = record["sample_step_s"]
            assert (
                step
                == manifest["candidate_max_steps_s"].get(
                    candidate["id"], manifest["ensemble_max_steps_s"]
                )[level]
            )
            assert dt == manifest["ensemble_sample_steps_s"][level]
            t = d["time"]
            assert (
                t[0] == 0
                and abs(t[-1] - 200e-6) < 1e-15
                and np.isfinite(np.column_stack(list(d.values()))).all()
            )
            assert np.all(np.diff(t) > 0) and float(np.max(np.diff(t))) <= step + 1e-15
            assert record["raw_points"] == len(t) and record["telemetry"][
                "accepted_steps"
            ] == len(t)
            _, y = sample(d, dt)
            n = round(20e-6 / dt)
            settled = True
            for name in ["i(va)", "i(vb)"]:
                error = rms(y[name][-n:] - y[name][-2 * n : -n])
                limit = 0.01 + 0.01 * rms(y[name][-n:])
                settled &= error <= limit
                report["settling"].append(
                    {
                        "id": jid,
                        "channel": name,
                        "error_a": error,
                        "limit_a": limit,
                        "pass": error <= limit,
                    }
                )
            f, s = spectrum(d, dt)
            saved = np.frombuffer(
                (root / "jobs" / jid / "spectra.f64").read_bytes(), dtype="<f8"
            ).reshape(-1, 5)
            assert np.allclose(saved[:, 0], f, rtol=1e-13, atol=1e-7)
            peaks = {}
            for i, name in enumerate(["a", "b", "cm", "dm"], 1):
                assert np.allclose(saved[:, i], s[name], rtol=1e-9, atol=1e-12)
                margin = 90 - 20 * math.log10(max(float(max(s[name])), 1e-15) / 1e-6)
                peak_index = int(np.argmax(s[name]))
                peaks[name] = {
                    "frequency_hz": float(f[peak_index]),
                    "amplitude_a": float(s[name][peak_index]),
                    "level_dbua": 90 - margin,
                    "margin_db": margin,
                }
                near(
                    jid + "-margin-" + name,
                    record["metrics"]["research_margin_db"][name],
                    margin,
                )
            rd = (
                1.724e-8
                * candidate["dm_geometry"][2]
                * candidate["dm_geometry"][3]
                * 1e-3
                / (4e-6)
            )
            rc = (
                1.724e-8
                * candidate["cm_geometry"][2]
                * candidate["cm_geometry"][3]
                * 1e-3
                / (4e-6)
            )
            x = (y["v(fa)"] - y["v(xcap)"]) / 0.2
            ya = (y["v(fa)"] - y["v(yca)"]) / 22
            yb = (y["v(fb)"] - y["v(ycb)"]) / 22
            stress = {
                "device_peak_v": max(float(np.max(abs(v))) for v in waveforms(d)[2:]),
                "device_peak_a": max(
                    float(np.max(abs(d[f"i({name})"])))
                    for name in ["vdah", "vdal", "vdbh", "vdbl"]
                ),
                "capacitor_peak_v": max(
                    float(np.max(abs(v)))
                    for v in [
                        d["v(xcap)"] - d["v(fb)"],
                        d["v(yca)"] - d["v(ch)"],
                        d["v(ycb)"] - d["v(ch)"],
                    ]
                ),
                "winding_rms_a": max(
                    rms(y[f"i({name})"]) for name in ["lda", "ldb", "lcma", "lcmb"]
                ),
                "loss_w": rd * (rms(y["i(lda)"]) ** 2 + rms(y["i(ldb)"]) ** 2)
                + rc * (rms(y["i(lcma)"]) ** 2 + rms(y["i(lcmb)"]) ** 2)
                + 0.2 * (rms(x) ** 2 + rms(ya) ** 2 + rms(yb) ** 2)
                + 21.8 * (rms(ya) ** 2 + rms(yb) ** 2),
                "damping_a_w": 21.8 * rms(ya) ** 2,
                "damping_b_w": 21.8 * rms(yb) ** 2,
            }
            for kind in ["dm", "cm"]:
                geometry = candidate[kind + "_geometry"]
                scale = (
                    candidate["l" + kind + "_h"]
                    * corner["l_scale"]
                    / (geometry[2] * geometry[0] * 1e-6)
                )
                currents = (
                    [d["i(lda)"], d["i(ldb)"]]
                    if kind == "dm"
                    else [
                        d["i(lcma)"] + 0.995 * d["i(lcmb)"],
                        d["i(lcmb)"] + 0.995 * d["i(lcma)"],
                    ]
                )
                stress[kind + "_peak_t"] = max(
                    float(np.max(abs(current * scale))) for current in currents
                )
            assert record["metrics"]["stress_limits"] == STRESS_LIMITS
            assert set(record["metrics"]["stress"]) == set(STRESS_LIMITS)
            for key, value in stress.items():
                near(jid + "-stress-" + key, record["metrics"]["stress"][key], value)
            expected_status = (
                "unsettled"
                if not settled
                else "predicted_infeasible"
                if any(stress[k] > STRESS_LIMITS[k] for k in stress)
                or any(peak["margin_db"] < 6 for peak in peaks.values())
                else "predicted_feasible"
            )
            assert record["status"] == record["metrics"]["status"] == expected_status
            mass = design_mass(candidate)
            near(jid + "-mass", record["metrics"]["mass_kg"], mass)
            physical_violations = [
                k for k in STRESS_LIMITS if stress[k] > STRESS_LIMITS[k]
            ]
            violations = physical_violations + [
                "research_mask_" + name
                for name, peak in peaks.items()
                if peak["margin_db"] < 6
            ]
            assert set(record["metrics"]["violations"]) == set(violations)
            worst = min(peaks, key=lambda name: peaks[name]["margin_db"])
            report["cases"].append(
                {
                    "id": jid,
                    "candidate": candidate["id"],
                    "corner": corner["id"],
                    "level": level,
                    "status": expected_status,
                    "settled": bool(settled),
                    "stress": stress,
                    "physical_violations": physical_violations,
                    "mass_kg": mass,
                    "peaks": peaks,
                    "worst_channel": worst,
                    "worst_frequency_hz": peaks[worst]["frequency_hz"],
                    "minimum_margin_db": peaks[worst]["margin_db"],
                }
            )
        for level in [1, 2]:
            dt = manifest["ensemble_sample_steps_s"][level]
            report["spectral_comparisons"] += compare(
                identity + f"-integration-{level}", data[level - 1], data[level], dt, dt
            )
            waveform_checks(
                identity + f"-integration-{level}",
                data[level - 1],
                data[level],
                dt,
                corner["bus_v"],
            )
        for dt in manifest["ensemble_sample_steps_s"][:2]:
            report["spectral_comparisons"] += compare(
                identity + f"-output-{dt}", data[2], data[2], dt, 1.25e-9
            )
            waveform_checks(
                identity + f"-output-{dt}",
                data[2],
                data[2],
                1.25e-9,
                corner["bus_v"],
                dt,
            )
        print(identity, "audited", flush=True)


def dpt_metrics(d, dt):
    t, y = sample(d, dt, 0, 16e-6, True)
    voltage = y["v(d)"]
    current = y["i(vdrain)"]
    m = {}
    for label, event, falling in [("on", 12e-6, True), ("off", 14e-6, False)]:
        crosses = []
        for threshold in [40, 200, 360]:
            ids = np.flatnonzero(
                (t[:-1] >= event - 0.2e-6)
                & (t[1:] <= event + 0.2e-6)
                & (
                    ((voltage[:-1] >= threshold) & (voltage[1:] < threshold))
                    if falling
                    else ((voltage[:-1] <= threshold) & (voltage[1:] > threshold))
                )
            )
            assert len(ids)
            i = int(ids[0])
            crosses.append(
                float(
                    t[i]
                    + (threshold - voltage[i])
                    * (t[i + 1] - t[i])
                    / (voltage[i + 1] - voltage[i])
                )
            )
        m[label + "_edge_s"] = abs(crosses[2] - crosses[0])
        m[label + "_half_s"] = crosses[1]
        lo, hi = event - 0.2e-6, event + 0.2e-6
        tt = np.r_[lo, t[(t > lo) & (t < hi)], hi]
        power = np.interp(tt, t, voltage) * np.interp(tt, t, current)
        m[label + "_energy_j"] = float(
            sum((power[:-1] + power[1:]) * 0.5 * np.diff(tt))
        )
        gates = np.interp(crosses, t, y["v(g)"])
        m[label + "_gate_crossings_v"] = gates.tolist()
    m["peak_v"] = float(max(voltage))
    m["min_v"] = float(min(voltage))
    m["peak_gate_v"] = float(max(y["v(g)"]))
    m["min_gate_v"] = float(min(y["v(g)"]))
    m["load_at_second_on_a"] = float(np.interp(12e-6, t, y["i(lload)"]))
    mask = (t > 14.04e-6) & (t < 14.4e-6)
    tt = t[mask]
    vv = voltage[mask] - 400
    candidates = []
    for polarity, sign in [("positive", 1), ("negative", -1)]:
        wave = sign * vv
        peaks = [
            i
            for i in range(1, len(wave) - 1)
            if wave[i] > wave[i - 1] and wave[i] >= wave[i + 1] and wave[i] > 0.2
        ]
        if len(peaks) >= 3 and wave[peaks[0]] > wave[peaks[1]] > wave[peaks[2]]:
            p = peaks[:3]
            candidates.append(
                {
                    "polarity": polarity,
                    "peak_times_s": tt[p].tolist(),
                    "peak_amplitudes_v": wave[p].tolist(),
                    "frequency_hz": 2 / (tt[p[2]] - tt[p[0]]),
                    "log_decrement": math.log(wave[p[0]] / wave[p[2]]) / 2,
                }
            )
    assert candidates
    m["ringing"] = min(candidates, key=lambda c: c["peak_times_s"][0])
    m["ringing"]["status"] = "measured"
    m["sample_step_s"] = dt
    gates = m["on_gate_crossings_v"]
    m["gate_drop_v"] = gates[2] - gates[0]
    m["gate_pass"] = bool(
        0.2 <= m["gate_drop_v"] <= 30
        and all(-3 <= m[edge + "_gate_crossings_v"][1] <= 23 for edge in ["on", "off"])
    )
    m["pass"] = bool(
        400 < m["peak_v"] < 960
        and m["load_at_second_on_a"] >= 10
        and m["gate_pass"]
        and all(
            0 < m[edge + "_energy_j"] < 0.002 and 0 < m[edge + "_edge_s"] < 200e-9
            for edge in ["on", "off"]
        )
    )
    assert all(math.isfinite(value) for value in m.values() if isinstance(value, float))
    return m


def dpt_refinement(label, a, b):
    metrics = {}
    for key in [
        "peak_v",
        "on_edge_s",
        "off_edge_s",
        "on_half_s",
        "off_half_s",
        "on_energy_j",
        "off_energy_j",
    ]:
        if key.endswith("energy_j"):
            limit = 2e-6 + 0.05 * abs(b[key])
        elif key == "peak_v":
            limit = 2 + 0.02 * abs(b[key])
        elif key.endswith("half_s"):
            limit = 2e-9 + 0.1 * b[key.replace("half", "edge")]
        else:
            limit = 2e-9 + 0.1 * abs(b[key])
        error = abs(a[key] - b[key])
        metrics[key] = {"difference": error, "limit": limit, "pass": error <= limit}
    for key, factor in [("frequency_hz", 0.1), ("log_decrement", 0.2)]:
        error = abs(a["ringing"][key] - b["ringing"][key])
        limit = factor * abs(b["ringing"][key])
        metrics[key] = {"difference": error, "limit": limit, "pass": error <= limit}
    return {
        "id": label,
        "metrics": metrics,
        "pass": bool(
            a["pass"]
            and b["pass"]
            and a["ringing"]["polarity"] == b["ringing"]["polarity"]
            and all(m["pass"] for m in metrics.values())
        ),
    }


dpt_data = []
for level in range(3):
    jid = f"q{level}-dpt"
    d = raw(root, jid)
    dpt_data.append(d)
    record = json.loads((root / "jobs" / jid / "result.json").read_text())
    dt = manifest["dpt_sample_steps_s"][level]
    assert record["id"] == jid and record["level"] == level
    assert record["sample_step_s"] == dt
    assert record["max_step_s"] == manifest["dpt_max_steps_s"][level]
    t = d["time"]
    assert t[0] == 0 and abs(t[-1] - 16e-6) < 1e-15
    assert np.all(np.diff(t) > 0) and max(np.diff(t)) <= record["max_step_s"] + 1e-15
    assert len(t) == record["raw_points"] == record["telemetry"]["accepted_steps"]
    m = dpt_metrics(d, dt)
    assert (
        m["pass"]
        and record["status"] == "qualified"
        and record["metrics"]["pass"] is True
    )
    for key, value in m.items():
        if key not in [
            "ringing",
            "on_gate_crossings_v",
            "off_gate_crossings_v",
            "gate_drop_v",
            "gate_pass",
            "pass",
        ]:
            near(jid + "-" + key, record["metrics"][key], value)
    for edge in ["on", "off"]:
        for index, value in enumerate(m[edge + "_gate_crossings_v"]):
            near(
                jid + f"-{edge}-gate-{index}",
                record["metrics"][edge + "_gate_at_vds_10_50_90_v"][index],
                value,
            )
    near(
        jid + "-gate-drop",
        record["metrics"]["gate_dynamics"]["on_edge_gate_drop_v"],
        m["gate_drop_v"],
    )
    assert record["metrics"]["gate_dynamics"]["pass"] == m["gate_pass"]
    for key in ["frequency_hz", "log_decrement"]:
        near(jid + "-" + key, record["metrics"]["ringing"][key], m["ringing"][key])
    for key in ["peak_times_s", "peak_amplitudes_v"]:
        for index, value in enumerate(m["ringing"][key]):
            near(
                jid + f"-{key}-{index}", record["metrics"]["ringing"][key][index], value
            )
    assert record["metrics"]["ringing"]["polarity"] == m["ringing"]["polarity"]
    report["dpt"].append({"id": jid, **m})

for level in [1, 2]:
    report["qualification_checks"].append(
        dpt_refinement(
            f"q{level - 1}-dpt-q{level}-dpt",
            report["dpt"][level - 1],
            report["dpt"][level],
        )
    )
for dt in manifest["dpt_sample_steps_s"][:2]:
    report["qualification_checks"].append(
        dpt_refinement(
            f"dpt-output-{dt}", dpt_metrics(dpt_data[-1], dt), report["dpt"][-1]
        )
    )

for candidate in manifest["candidates"]:
    for corner in manifest["corners"]:
        identity = candidate["id"] + "-" + corner["id"]
        for suffix in [
            "integration-1",
            "integration-2",
            "output-5e-09",
            "output-2.5e-09",
        ]:
            label = identity + "-" + suffix
            spectra = [x for x in report["spectral_comparisons"] if x["id"] == label]
            waves = [x for x in report["waveform_comparisons"] if x["id"] == label]
            assert len(spectra) == 4 and len(waves) == 6
            report["qualification_checks"].append(
                {
                    "id": label,
                    "pass": all(x["failed_bins"] == 0 for x in spectra)
                    and all(x["pass"] for x in waves),
                }
            )
assert len(report["qualification_checks"]) == 40
stored_checks = json.loads((root / "qualification.json").read_text())
assert stored_checks["expected_checks"] == 40
assert {x["id"]: x["pass"] for x in stored_checks["checks"]} == {
    x["id"]: x["pass"] for x in report["qualification_checks"]
}

for level in range(3):
    for name in ["light", "boundary", "reference"]:
        rows = [
            x for x in report["cases"] if x["level"] == level and x["candidate"] == name
        ]
        report["roles"].append(role_check(name, level, rows))

report["summary"] = {
    "jobs": len(expected),
    "role_checks": len(report["roles"]),
    "role_failures": sum(not x["pass"] for x in report["roles"]),
    "qualification_checks": len(report["qualification_checks"]),
    "qualification_failures": sum(
        not x["pass"] for x in report["qualification_checks"]
    ),
    "original_spectral_failed_bins": sum(
        x["failed_bins"] for x in report["spectral_comparisons"]
    ),
    "spectral_comparisons": len(report["spectral_comparisons"]),
    "waveform_failed": sum(not x["pass"] for x in report["waveform_comparisons"]),
    "waveform_comparisons": len(report["waveform_comparisons"]),
    "settling_failed": sum(not x["pass"] for x in report["settling"]),
    "metric_mismatches": len(report["metric_errors"]),
    "all_bin_max_delta_uA": max(
        x["all_max_abs_uA"] for x in report["spectral_comparisons"]
    ),
}
report["summary"]["pass"] = all(
    report["summary"][name] == 0
    for name in [
        "original_spectral_failed_bins",
        "waveform_failed",
        "settling_failed",
        "metric_mismatches",
        "role_failures",
        "qualification_failures",
    ]
)
output = args.output
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["summary"], indent=2), flush=True)

if not report["summary"]["pass"]:
    raise SystemExit(1)
