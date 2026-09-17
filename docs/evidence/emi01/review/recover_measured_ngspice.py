"""Recover historical bytes only if the complete retained SHA256 matches."""

from pathlib import Path
import hashlib
import json
import re
import sys
import time

if sys.flags.optimize:
    raise SystemExit(
        "Run this review helper without -O or PYTHONOPTIMIZE; its acceptance checks require assertions."
    )

target = "b2574df80f88a02b3d0597206277b189686e8708dcf8449fec42f250e41d74a6"
inputs = [
    (
        Path(
            "/home/adam/code/ohmnivore/bazel-bin/third_party/ngspice/ngspice_46_build/bin/ngspice"
        ),
        range(1000, 1645),
    ),
    (Path("/tmp/emi01-ngspice-stable"), range(1, 1000)),
]
start = time.perf_counter()
attempts = 0
result = {"target_sha256": target, "searches": []}
for path, candidates in inputs:
    original = path.read_bytes()
    data = bytearray(original)
    pattern = rb"/sandbox/linux-sandbox/([0-9]+)/execroot/_main/bazel-out/k8-fastbuild/bin/third_party/ngspice/ngspice_46_build.build_tmpdir/ngspice_46_build/(?:bin|share/ngspice)\x00"
    hits = list(re.finditer(pattern, original))
    assert len(hits) == 2
    source_digits = hits[0].group(1)
    assert all(hit.group(1) == source_digits for hit in hits)
    result["searches"].append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(original).hexdigest(),
            "sandbox_id": source_digits.decode(),
            "offsets": [hit.start(1) for hit in hits],
            "range": [candidates.start, candidates.stop - 1],
        }
    )
    for candidate in candidates:
        digits = str(candidate).encode()
        if len(digits) != len(source_digits):
            continue
        for hit in hits:
            data[hit.start(1) : hit.end(1)] = digits
        attempts += 1
        if hashlib.sha256(data).hexdigest() == target:
            recovered = Path("/tmp/emi01-recovered-measured-ngspice")
            if recovered.exists():
                assert recovered.read_bytes() == data
            else:
                recovered.write_bytes(data)
                recovered.chmod(0o500)
            result.update(
                found=True,
                recovered_path=str(recovered),
                recovered_sandbox_id=candidate,
                attempts=attempts,
                elapsed_s=time.perf_counter() - start,
                bytes=len(data),
                changed_byte_offsets=[
                    i for i, (a, b) in enumerate(zip(original, data)) if a != b
                ],
            )
            Path("/tmp/emi01-ngspice-recovery.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result, indent=2))
            raise SystemExit(0)
result.update(found=False, attempts=attempts, elapsed_s=time.perf_counter() - start)
Path("/tmp/emi01-ngspice-recovery.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
