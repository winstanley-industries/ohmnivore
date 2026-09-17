"""Compare retained runtime observations with current canonical Bazel artifacts.

Run outside retained timing, using the pinned interpreter. This checks artifact
identities; it is not execution attestation and does not replace the study audit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if sys.flags.optimize:
    raise RuntimeError("Run this assertion-based audit without Python optimization")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert sys.version_info[:3] == (3, 12, 13)
root = Path.cwd()
runfiles = root / "bazel-bin/reference/emi01/study.runfiles"
wheel = runfiles / (
    "rules_python++pip+emi01_pypi_312_numpy_cp312_cp312_"
    "manylinux_2_27_x86_64_e7dd01a4/site-packages"
)
gcc = runfiles / "gcc_toolchain++gcc_toolchains+ohmnivore_cuda_gcc15/lib64"
native = sorted(
    list((wheel / "numpy.libs").glob("*.so*"))
    + [
        wheel / "numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so",
        wheel / "numpy/linalg/_umath_linalg.cpython-312-x86_64-linux-gnu.so",
        runfiles / "_main/third_party/emi_python/libz.so.1",
        gcc / "libgcc_s.so.1",
        gcc / "libstdc++.so.6.0.34",
    ]
)
assert len(native) == 8
canonical = {
    "ngspice_sha256": digest(
        root / "bazel-bin/third_party/ngspice/ngspice_46_build/bin/ngspice"
    ),
    "python_executable_sha256": digest(Path(sys.executable)),
    "native_runtime_sha256": {str(p.resolve()): digest(p) for p in native},
}
frozen = "bf1165fe92fb5f527bbddfcb53374956d62929a5"
reports = []
for directory in args.run:
    metadata = json.loads((directory / "metadata.json").read_bytes())
    for key, value in canonical.items():
        assert metadata[key] == value, (directory, key, "runtime identity mismatch")
    sources = metadata["source_sha256"]
    assert len(sources) == 24
    for name, expected in sources.items():
        assert digest(root / name) == expected, (name, "working source mismatch")
        committed = subprocess.check_output(
            ["git", "show", f"{frozen}:{name}"], cwd=root
        )
        assert hashlib.sha256(committed).hexdigest() == expected, (
            name,
            "frozen commit mismatch",
        )
    reports.append(
        {
            "run": directory.name,
            "metadata_sha256": digest(directory / "metadata.json"),
            "terminal_sha256": digest(directory / "terminal.json"),
            "source_files": len(sources),
            "native_artifacts": len(native),
            "pass": True,
        }
    )
output = {
    "schema": "emi01-runtime-artifact-audit-v1",
    "boundary": "Current canonical artifacts and frozen source match retained observations; not execution attestation.",
    "frozen_implementation_commit": frozen,
    "implementation_sha256": digest(Path(__file__)),
    "canonical": canonical,
    "runs": reports,
    "pass": True,
}
args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
print(json.dumps({"pass": True, "runs": reports}, indent=2))
