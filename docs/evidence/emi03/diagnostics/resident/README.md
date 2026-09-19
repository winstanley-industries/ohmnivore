# Resident CUDA follow-up: development evidence

This snapshot is **not EMI-03 completion or full qualification**. It preserves
one frozen q0 DPT differential and one full frozen q0 reference/nominal budget
probe. No 9/36-job timing mode is admitted and no speedup is established.

The separate resident executor runs the complete coupled circuit on the GPU in
bounded chunks, with native FP64 arithmetic, GPU numeric pivoting, original-system
residual checks, and no CPU numeric fallback. CPU execution remains authoritative.

| Probe | CPU wall | GPU wall | Result |
|---|---:|---:|---|
| Full 16 us q0 DPT, maximum step 2 ns | 0.597 s | 12.783 s | Complete waveform and switching-metric CPU differential passes |
| Full 200 us q0 reference/nominal, maximum step 625 ps | 33.279 s | 120.150 s including termination | GPU reaches the 120 s wall limit; no completed GPU output |

The second GPU process used 3.751 s of child CPU time, below the 110 s CPU limit.
The CPU reference used 33.213 s of child CPU time and emitted 461,672 points.
The probes impose one job's limits; they do not measure aggregate study resources,
external-oracle qualification, refinement, spectra, feasibility, or throughput.
The timed boundaries are standalone worker execution and termination, not the
frozen complete-study boundary. GPU telemetry missing after termination is
unavailable, not zero. Failed output is never published as a completed result.

A missing barrier between parallel output writes and the shared row-counter
increment was found with a multi-block prototype. The barrier is present in this
snapshot. The added regression checks 192 distinct voltage outputs and reactive
coordinates across more than two output chunks. The multi-block prototype passed
the corrected tests but was slower and is not included in this executor.

The seven resident tests cover 21 independent analytic transient fixtures, wide
output publication, changing pivots and singular zero responses, lazy branches,
observer/attempt limits, and 16 concurrent jobs with one isolated allocation
fault. The existing 20 CUDA cases also pass. These tests do not replace the frozen
thirty-job EMI qualification.

## Identity and reconstruction

`identity.json` binds the CPU/GPU binaries and 95 build/source files to the base
commit plus the exact decompressed `source.patch.gz`. The patch includes new
source files and contains no vendor model. Apply it to the recorded base to
reconstruct the measured implementation, then use the recorded Bazel command.
The implementation's full source identity is historical even when later work
changes the branch. Build the CPU reference and Python runfiles with
`bazel build -c opt //reference/emi03:ensemble` (without the CUDA configuration);
the identity records the separate CUDA worker build command. Switching build
configurations can move Bazel output links, so verify both binary hashes before
running the reconstruction audit.

`raw-index.json` binds every compressed raw file to its exact uncompressed bytes.
The complete CPU output for the budget probe is retained even though GPU execution
failed. `audit.json` checks source reconstruction, binary identity, lossless raw
reconstruction, raw point counts and terminal status. Its success validates this
negative development record; `acceptance_pass` remains false.

The compressed Python scripts preserve the exact captured probe/reconstruction
code. `scripts.json` records their decompressed hashes. They use the recorded host's
scratch paths and pinned Bazel runfiles; adjust scratch destinations when replaying
and always use a new output directory. The complete experiment must still run
through `//reference/emi03:ensemble` with fresh qualification in both invocations.

The canonical [validation records](validation/checks.json) pass lint, build, the
normal and lockfile test suites, ASAN, UBSAN, ngspice/prepared-AC checks, the required
AC replay benchmark, CUDA smoke, both executable linkage audits, and CUDA tests.
Explicit sanitizer/CUDA targets reject during analysis as required. Validation
adds only a linkage-test target relative to the measured source snapshot; see
[the source delta](validation/source-delta.json). [The scope audit](validation/scope.json)
confirms all 34 protected paths remain unchanged. [Host details](host.json) are a
post-run snapshot, not execution-clock or peak-resource evidence.
