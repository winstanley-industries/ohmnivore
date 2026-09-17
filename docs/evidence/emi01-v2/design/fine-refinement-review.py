"""Independent exploratory raw-waveform comparison, no production helpers."""

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


def raw(root, identity):
    folder = Path(root) / "jobs" / identity
    header = (folder / "raw.header").read_bytes()
    index = json.loads((folder / "raw.json").read_text())
    parts = []
    for item in index["chunks"]:
        part = gzip.decompress(
            (Path(root) / "blobs" / (item["sha256"] + ".gz")).read_bytes()
        )
        assert (
            len(part) == item["bytes"]
            and hashlib.sha256(part).hexdigest() == item["sha256"]
        )
        parts.append(part)
    data = b"".join(parts)
    assert len(data) == index["payload_bytes"]
    assert hashlib.sha256(header + data).hexdigest() == index["raw_sha256"]
    names = [
        line.split()[1]
        for line in header.decode("ascii")
        .split("Variables:\n")[1]
        .split("Binary:\n")[0]
        .strip()
        .splitlines()
    ]
    matrix = np.frombuffer(data, dtype="<f8").reshape(-1, len(names))
    assert np.isfinite(matrix).all()
    result = {name: matrix[:, i] for i, name in enumerate(names)}
    assert (
        result["time"][0] == 0
        and abs(result["time"][-1] - 200e-6) < 1e-15
        and np.all(np.diff(result["time"]) > 0)
    )
    return result


def spectrum(data, dt):
    n = round(100e-6 / dt)
    t = 100e-6 + np.arange(n) * dt
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    a = np.interp(t, data["time"], data["i(va)"])
    b = np.interp(t, data["time"], data["i(vb)"])
    f = np.arange(n // 2 + 1) / (n * dt)
    keep = (f >= 150e3 - 1e-5) & (f <= 10e6 + 1e-5)
    return f[keep], {
        name: (math.sqrt(2) * abs(np.fft.rfft((x - x.mean()) * window)) / window.sum())[
            keep
        ]
        for name, x in [("a", a), ("b", b), ("cm", (a + b) / 2), ("dm", (a - b) / 2)]
    }


def compare_spectra(label, a, b, dt_a, dt_b):
    f, x = spectrum(a, dt_a)
    ff, y = spectrum(b, dt_b)
    assert np.allclose(f, ff, rtol=1e-12, atol=1e-8)
    rows = []
    for name in x:
        left, right = x[name], y[name]
        above = np.maximum(left, right) > 10e-6
        delta = abs(left - right)
        db = abs(20 * np.log10(np.maximum(left, 1e-15) / np.maximum(right, 1e-15)))
        bad = np.where((above & (db > 1)) | (~above & (delta > 1e-6)))[0]
        imax = int(np.argmax(np.where(above, db, 0)))
        rows.append(
            {
                "id": label,
                "observable": name,
                "failed_bins": len(bad),
                "above_floor_max_db": float(np.max(db[above], initial=0)),
                "below_floor_max_abs_uA": float(np.max(delta[~above], initial=0) * 1e6),
                "worst_relative_frequency_hz": float(f[imax]),
                "worst_relative_amplitudes_uA": [
                    float(left[imax] * 1e6),
                    float(right[imax] * 1e6),
                ],
                "failed": [
                    {
                        "frequency_hz": float(f[i]),
                        "amplitudes_uA": [float(left[i] * 1e6), float(right[i] * 1e6)],
                        "db": float(db[i]),
                        "delta_uA": float(delta[i] * 1e6),
                    }
                    for i in bad[:25]
                ],
            }
        )
    return rows


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def waveforms(d):
    return {
        "ia": d["i(va)"],
        "ib": d["i(vb)"],
        "vds_ah": d["v(p)"] - d["v(a)"],
        "vds_al": d["v(a)"],
        "vds_bh": d["v(p)"] - d["v(b)"],
        "vds_bl": d["v(b)"],
    }


def compare_waveforms(label, a, b, bus):
    t = 100e-6 + np.arange(80000) * 1.25e-9
    ax, bx = waveforms(a), waveforms(b)
    rows = []
    for name in ax:
        u = np.interp(t, a["time"], ax[name])
        v = np.interp(t, b["time"], bx[name])
        error = rms(u - v)
        limit = 0.02 + 0.02 * rms(v) if name.startswith("i") else 2 + 0.02 * bus
        rows.append(
            {
                "id": label,
                "observable": name,
                "rms_error": error,
                "limit": limit,
                "pass": error <= limit,
            }
        )
    return rows


def sample_waveform_checks(label, d, bus, coarse_dt, fine_dt):
    grid = 100e-6 + np.arange(round(100e-6 / fine_dt)) * fine_dt
    coarse_grid = 100e-6 + np.arange(round(100e-6 / coarse_dt) + 1) * coarse_dt
    rows = []
    for name, x in waveforms(d).items():
        u = np.interp(grid, coarse_grid, np.interp(coarse_grid, d["time"], x))
        v = np.interp(grid, d["time"], x)
        error = rms(u - v)
        limit = 0.02 + 0.02 * rms(v) if name.startswith("i") else 2 + 0.02 * bus
        rows.append(
            {
                "id": label,
                "observable": name,
                "rms_error": error,
                "limit": limit,
                "pass": error <= limit,
            }
        )
    return rows


def physical(root, jid, d):
    record = json.loads((Path(root) / "jobs" / jid / "result.json").read_text())
    c, k = record["candidate"], record["corner"]
    dt = record["sample_step_s"]
    t = 100e-6 + np.arange(round(100e-6 / dt)) * dt
    y = {name: np.interp(t, d["time"], x) for name, x in d.items() if name != "time"}
    period = round(20e-6 / dt)
    settling = {
        name: {
            "error_a": rms(y[name][-period:] - y[name][-2 * period : -period]),
            "limit_a": 0.01 + 0.01 * rms(y[name][-period:]),
        }
        for name in ["i(va)", "i(vb)"]
    }
    rd = 1.724e-8 * c["dm_geometry"][2] * c["dm_geometry"][3] * 1e-3 / 4e-6
    rc = 1.724e-8 * c["cm_geometry"][2] * c["cm_geometry"][3] * 1e-3 / 4e-6
    xc = (y["v(fa)"] - y["v(xcap)"]) / 0.2
    ya = (y["v(fa)"] - y["v(yca)"]) / 22
    yb = (y["v(fb)"] - y["v(ycb)"]) / 22
    stress = {
        "device_peak_v": max(
            float(np.max(abs(v)))
            for name, v in waveforms(d).items()
            if name.startswith("vds")
        ),
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
        + 0.2 * (rms(xc) ** 2 + rms(ya) ** 2 + rms(yb) ** 2)
        + 21.8 * (rms(ya) ** 2 + rms(yb) ** 2),
        "damping_a_w": 21.8 * rms(ya) ** 2,
        "damping_b_w": 21.8 * rms(yb) ** 2,
    }
    for kind in ["dm", "cm"]:
        g = c[kind + "_geometry"]
        scale = c["l" + kind + "_h"] * k["l_scale"] / (g[2] * g[0] * 1e-6)
        currents = (
            [d["i(lda)"], d["i(ldb)"]]
            if kind == "dm"
            else [
                d["i(lcma)"] + 0.995 * d["i(lcmb)"],
                d["i(lcmb)"] + 0.995 * d["i(lcma)"],
            ]
        )
        stress[kind + "_peak_t"] = max(float(np.max(abs(x * scale))) for x in currents)
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
    assert limits == record["metrics"]["stress_limits"]
    f, s = spectrum(d, dt)
    peaks = {}
    for name, amps in s.items():
        i = int(np.argmax(amps))
        level = float(20 * math.log10(amps[i] / 1e-6))
        peaks[name] = {
            "frequency_hz": float(f[i]),
            "amplitude_a": float(amps[i]),
            "level_dbua": level,
            "margin_db": 90 - level,
        }
        assert math.isclose(
            peaks[name]["margin_db"],
            record["metrics"]["research_margin_db"][name],
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    for name, value in stress.items():
        assert math.isfinite(value) and math.isclose(
            value, record["metrics"]["stress"][name], rel_tol=1e-10, abs_tol=1e-10
        ), (jid, name, value, record["metrics"]["stress"][name])
    violations = [name for name, value in stress.items() if value > limits[name]] + [
        name for name, peak in peaks.items() if peak["margin_db"] < 6
    ]
    settled = all(v["error_a"] <= v["limit_a"] for v in settling.values())
    status = (
        "unsettled"
        if not settled
        else "predicted_infeasible"
        if violations
        else "predicted_feasible"
    )
    assert record["status"] == status
    worst = max(peaks, key=lambda key: peaks[key]["level_dbua"])
    return {
        "id": jid,
        "root": str(root),
        "raw_sha256": record["raw_sha256"],
        "record_sha256": hashlib.sha256(
            (Path(root) / "jobs" / jid / "result.json").read_bytes()
        ).hexdigest(),
        "max_step_s": record["max_step_s"],
        "status": status,
        "stress": stress,
        "stress_limits": limits,
        "settling": settling,
        "spectral_peaks": peaks,
        "worst_observable": worst,
        "min_margin_db": peaks[worst]["margin_db"],
        "physical_violations": [
            name for name, value in stress.items() if value > limits[name]
        ],
    }


parser = argparse.ArgumentParser()
parser.add_argument("candidate", choices=["anchor", "boundary"])
parser.add_argument("--output", type=Path)
args = parser.parse_args()
levels = (
    [
        ("/tmp/emi01-followup-probes3", "probe11"),
        ("/tmp/emi01-followup-probes5", "probe17"),
        ("/tmp/emi01-followup-probes5", "probe18"),
    ]
    if args.candidate == "anchor"
    else [("/tmp/emi01-followup-probes6", f"probe19_l{level}") for level in range(3)]
)
report = {
    "source": "exploratory independent adjacent refinement, output sampling and stress audit; not retained qualification",
    "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "candidate": args.candidate,
    "integration_spectra": [],
    "waveforms": [],
    "sampling_spectra": [],
    "sampling_waveforms": [],
    "physical": [],
}
for corner, bus in [("nominal", 400), ("fast_low_lc", 440), ("hot_high_c", 360)]:
    data = []
    for root, prefix in levels:
        jid = prefix + "-" + corner
        d = raw(root, jid)
        data.append(d)
        report["physical"].append(physical(root, jid, d))
    for level, dt in [(1, 2.5e-9), (2, 1.25e-9)]:
        label = corner + f"-integration-{level}"
        report["integration_spectra"] += compare_spectra(
            label, data[level - 1], data[level], dt, dt
        )
        report["waveforms"] += compare_waveforms(
            label, data[level - 1], data[level], bus
        )
    for da, db in [(5e-9, 2.5e-9), (2.5e-9, 1.25e-9), (5e-9, 1.25e-9)]:
        label = corner + f"-sampling-{da}-{db}"
        report["sampling_spectra"] += compare_spectra(label, data[2], data[2], da, db)
        report["sampling_waveforms"] += sample_waveform_checks(
            label, data[2], bus, da, db
        )
    print(corner, "done", flush=True)
report["summary"] = {
    "integration_failed_bins": sum(
        x["failed_bins"] for x in report["integration_spectra"]
    ),
    "waveform_failed": sum(not x["pass"] for x in report["waveforms"]),
    "sampling_failed_bins": sum(x["failed_bins"] for x in report["sampling_spectra"]),
    "sampling_waveform_failed": sum(
        not x["pass"] for x in report["sampling_waveforms"]
    ),
    "physical_failed": sum(bool(x["physical_violations"]) for x in report["physical"]),
    "all_predicted_feasible": all(
        x["status"] == "predicted_feasible" for x in report["physical"]
    ),
    "min_margin_db": min(x["min_margin_db"] for x in report["physical"]),
}
(args.output or Path("/tmp/emi01-" + args.candidate + "-fine-review.json")).write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report["summary"], indent=2))
for row in report["integration_spectra"]:
    print(
        row["id"],
        row["observable"],
        row["failed_bins"],
        row["above_floor_max_db"],
        row["below_floor_max_abs_uA"],
        row["worst_relative_frequency_hz"],
        row["worst_relative_amplitudes_uA"],
    )
for row in report["physical"]:
    print(
        row["id"],
        row["status"],
        row["min_margin_db"],
        row["worst_observable"],
        row["spectral_peaks"][row["worst_observable"]],
    )
