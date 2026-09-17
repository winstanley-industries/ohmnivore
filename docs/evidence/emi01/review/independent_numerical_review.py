"""Independent EMI-01 numerical review; run only outside benchmark timing.

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

from third_party.emi_python.runtime import load_numpy

np = load_numpy()

if sys.flags.optimize:
    raise RuntimeError(
        "Run this assertion-based diagnostic without Python optimization"
    )


def raw(root, identity):
    folder = Path(root) / "jobs" / identity
    header = (folder / "raw.header").read_bytes()
    index = json.loads((folder / "raw.json").read_text())
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
expected = [
    f"q{level}-dpt"
    if candidate is None
    else f"q{level}-ensemble-{candidate['id']}-{corner['id']}"
    for level in range(3)
    for candidate, corner in [(None, None)]
    + [(c, k) for c in manifest["candidates"] for k in manifest["corners"]]
]
assert terminal["job_ids"][:30] == expected and len(set(expected)) == 30
assert len(terminal["job_ids"]) in (30, 102) and len(set(terminal["job_ids"])) == len(
    terminal["job_ids"]
)
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


for candidate in manifest["candidates"]:
    for corner in manifest["corners"]:
        identity = candidate["id"] + "-" + corner["id"]
        data = []
        for level in range(3):
            jid = f"q{level}-ensemble-{identity}"
            d = raw(root, jid)
            data.append(d)
            record = json.loads((root / "jobs" / jid / "result.json").read_text())
            step = record["max_step_s"]
            dt = record["sample_step_s"]
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
            for i, name in enumerate(["a", "b", "cm", "dm"], 1):
                assert np.allclose(saved[:, i], s[name], rtol=1e-9, atol=1e-12)
                margin = 90 - 20 * math.log10(max(float(max(s[name])), 1e-15) / 1e-6)
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
            for key, value in stress.items():
                near(jid + "-stress-" + key, record["metrics"]["stress"][key], value)
            expected_status = (
                "unsettled"
                if not settled
                else "predicted_infeasible"
                if any(
                    stress[k] > record["metrics"]["stress_limits"][k] for k in stress
                )
                or any(v < 6 for v in record["metrics"]["research_margin_db"].values())
                else "predicted_feasible"
            )
            assert record["status"] == expected_status
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
                identity + f"-sampling-{dt}", data[2], data[2], dt, 1.25e-9
            )
            waveform_checks(
                identity + f"-sampling-{dt}",
                data[2],
                data[2],
                1.25e-9,
                corner["bus_v"],
                dt,
            )
        print(identity, "audited", flush=True)

for level in range(3):
    jid = f"q{level}-dpt"
    d = raw(root, jid)
    record = json.loads((root / "jobs" / jid / "result.json").read_text())
    dt = record["sample_step_s"]
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
    for key, value in m.items():
        if key not in ["ringing", "on_gate_crossings_v", "off_gate_crossings_v"]:
            near(jid + "-" + key, record["metrics"][key], value)
    for key in ["frequency_hz", "log_decrement"]:
        near(jid + "-" + key, record["metrics"]["ringing"][key], m["ringing"][key])
    assert record["metrics"]["ringing"]["polarity"] == m["ringing"]["polarity"]
    report["dpt"].append({"id": jid, **m})

report["summary"] = {
    "jobs": len(expected),
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
    ]
)
output = args.output
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["summary"], indent=2), flush=True)

if not report["summary"]["pass"]:
    raise SystemExit(1)
