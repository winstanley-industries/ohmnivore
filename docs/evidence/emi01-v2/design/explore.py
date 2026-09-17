"""Exploratory design probes only; never retained performance evidence."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

if sys.flags.optimize:
    raise RuntimeError("Run this diagnostic without Python optimization")

from reference.emi01 import adapter, study  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True)
    runfiles = Path(
        "/home/adam/code/ohmnivore/bazel-bin/reference/emi01/study.runfiles"
    )
    archive = next((runfiles / "+http_file+emi01_microchip_model/file").iterdir())
    binary = Path(
        "/home/adam/code/ohmnivore/bazel-bin/third_party/ngspice/ngspice_46_build/bin/ngspice"
    ).resolve()
    m = json.loads(
        Path("/home/adam/code/ohmnivore/reference/emi01/manifest.json").read_text()
    )
    with tempfile.TemporaryDirectory(prefix="emi01-followup-model-") as temp:
        model = Path(temp) / "model.lib"
        provenance = adapter.adapt_archive(archive, model)
        study.write_json(
            args.out / "exploration.json",
            {
                "config": config,
                "model": provenance,
                "ngspice_sha256": study.check_elf(binary),
                "performance_evidence": False,
            },
        )
        specs = []
        for case in config:
            c = case["candidate"]
            for k in m["corners"]:
                if k["id"] not in case.get("corners", ["nominal"]):
                    continue
                specs.append(
                    {
                        "id": c["id"] + "-" + k["id"],
                        "fixture": "ensemble",
                        "candidate": c,
                        "corner": k,
                        "out": str(args.out),
                        "binary": str(binary),
                        "model": str(model),
                        "limits": m["limits"],
                        "level": 0,
                        "max_step_s": case.get("max_step_s", 2.5e-9),
                        "sample_step_s": 2.5e-9,
                        "manifest_sha256": study.sha(args.config.read_bytes()),
                    }
                )
        records, _ = study.batch(specs, 4, args.out, "exploratory-design")
        rows = []
        for r in records:
            v = r.get("metrics", {})
            row = {
                "id": r["id"],
                "status": r["status"],
                "margin_db": v.get("research_margin_db"),
                "stress": v.get("stress"),
                "settling": v.get("settling"),
                "violations": v.get("violations"),
                "error": r.get("error"),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        study.write_json(args.out / "outcomes.json", rows)


if __name__ == "__main__":
    main()
