# Resident hardware-counter investigation

EMI-03 remains incomplete. This evidence records a successful hardware-counter
capture after counter access was enabled, and six rejected optimization trials.
It supplies no complete-study qualification or accepted performance timing.

## Hardware and profile identity

`resident-full-counters84/profile.json` records the exact command, worker digest,
input digest and 89 source-file digests. Its worker and sources are the previously
retained [owner-pool checkpoint](../resident-owner-pool/README.md). The full
Nsight Compute set captured one `Emi03Advance` launch in 41 replay passes on the
1.03 us reference/nominal prefix. `resident.ncu-repz`, the exported metrics, CUDA/
SASS source counters and raw trajectory are retained. Clock and cache controls
were disabled. The current driver is 616.92, separately recorded in `device.csv`;
older timing observations used a different driver and are not current baselines.

The profiler is the checksum-pinned Nsight Compute archive 2026.3.1.2
(SHA-256 `a0039be47cfe551f89d82087256080a9f33c6257d99a72c1164a87466ab072aa`),
whose executable reports 2026.3.1.0 build 38829034. The separate pinned profiler
workspace and archive provenance are retained with the earlier
[profiling evidence](../resident-profiling/README.md). Historical permission
failures remain historical failures; hardware counters now work.

The kernel launches one 256-thread block for a complete circuit, with 106
registers per thread and no local or shared spilling requests. Scheduler counters
show 95.26% of cycles without an eligible warp. Average barrier stalls are 26.49
cycles per issued instruction out of 42.21 warp cycles per issued instruction.
That ratio is a warp-state observation, not a removable fraction of kernel wall
time. The dominant source locations are the completion of the warp-zero
triangular solve and factor-level barriers. Other warps wait for this work.
There is no evidence of a bandwidth or spilling bottleneck. Source paths refer to
the original Bazel sandbox; the exact matching source archive is retained.

## Rejected trials

Each trial has an immutable source patch, 89-file source identity, worker/runtime
digests, build log and ten passing resident tests. `variant-*/scope.json` explains
the variant; the generic freezer's original scope label is preserved in its
identity record. All six source patches reconstruct from the owner-pool source
archive. Binary files are identified by digest and not checked into Git.

| Trial | Change | Fresh ordinary diagnostic result |
|---|---|---|
| 85 | 128 threads, complete program/reduction coverage | 2.301 s vs 2.141 s baseline; 7.5% slower |
| 86 | 64 threads, complete program/reduction coverage | 2.598 s vs 2.141 s; 21.3% slower |
| 87 | 32 threads with warp synchronization | 3.682 s vs 2.141 s; 72.0% slower |
| 88 | AMD symbolic ordering | 1.868 s vs 2.087 s on the 1.03 us prefix; rejected after longer and DPT runs |
| 89 | AMD initially, COLAMD on numerical plan refresh | No improvement over AMD in the longer coupled run or DPT |
| 90 | Four-term operand prefetch with original ordered FP64 FMAs | 2.127 s vs 2.095 s; 1.5% slower, all waveform checks pass |

Short comparisons use one warmup and three alternating ordinary measurements per
lane, with separate phase-print executions where recorded. These are screening
medians, not the frozen five-measurement, two-invocation acceptance experiment.
No competing build, test or GPU work ran during the comparisons.

AMD reduces structural fill and solve depth, but its initial prefix benefit does
not generalize. In the 20 us coupled diagnostic, baseline/AMD/recovery-policy
request wall times are 30.598 / 33.680 / 33.663 s, versus 3.103 s CPU. GPU point
counts are 67,140 / 86,180 / 86,180. The latter two outputs are identical, and
neither requires a numerical plan refresh. All saved-observable waveform checks
pass. The 20 us records include phase printing and are not acceptance timings.

Both ordering trials also pass fresh DPT q0/q1/q2 CPU comparisons, all eight DPT
refinements per trial, and their actual sixteen-owner-pool resource audits. GPU
request times are 15.129 / 18.659 / 27.845 s for AMD and 14.704 / 18.310 / 26.679 s
for the recovery policy. These are slower than the historical owner-pool DPT
checkpoint; that older driver prevents an exact current speed ratio. The larger
factor/solve counts and the fresh longer coupled comparison reject both changes.
All six variants were removed from the implementation.

## Reconstruction

`audit.json` records successful reconstruction of six source variants, 56 raw
trajectories, 46 GPU waveform comparisons and 16 DPT refinement checks. DPT
metrics, telemetry, worker affinity and resource observations are re-audited.
The hardware-profile trajectory is also checked against a fresh CPU trajectory.
No assumption of identical internal point counts is used. `files.json` inventories
all retained artifacts; compressed scripts preserve the execution and audit steps.

These rejected trials have no new Compute Sanitizer coverage. The baseline's
previous sanitizer results and incomplete concurrent racecheck remain separately
identified in its checkpoint. Full thirty-job qualification, the 120 s coupled-job
limit, and both complete-study speedup/cold-time invocations remain outstanding.

DPT source snapshots are stored losslessly in `sources.tar.gz` so historical
BUILD files are not interpreted as live Bazel packages. Every original source
hash is rechecked after reconstruction. The archived-source audit script and
its passing log accompany this storage-only change.
