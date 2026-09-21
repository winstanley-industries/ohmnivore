# Explicit shared addresses in sparse factor and triangular work

Trial 100 is selected for continued qualification. Its sparse factor and
triangular scratch arrays already reside in shared memory; explicit 32-bit
shared addresses avoid generic address calculations. Ordered native-FP64 FMAs,
pivot checks, linear certification, mandatory correction, fresh nonlinear
acceptance checks and synchronization are preserved. Dense global-memory work
keeps its existing accesses. This is not a full EMI-03 acceptance result.

| Trial | Scope | Short ordinary median versus baseline |
|---|---|---:|
| 99 | Triangular scratch only | 1.988 versus 2.089 s |
| 100 | Also sparse factor and inverse scratch | 1.963 versus 2.086 s |
| 101 | Also expression scratch views | 1.961 versus 2.084 s |

Each short comparison has one warmup, three alternating observations and separate
phase-print runs. All three variants pass ten resident tests and every shortened
CPU waveform comparison. Trial 101 has no clear incremental short-run benefit,
so its expression wrapper is removed. Original identities and exact source
archives retain every variant, including the GPU harness environment in use.

In a 20 us repeat, trial 99 takes 28.677 versus 30.436 s with an identical raw
trajectory. Trial 100 takes 27.288 s, with 63,945 points versus 67,140 for that
baseline. Trial 101 takes 22.943 s with 49,050 points. An earlier trial-99 run
also produces 49,050 points and takes 23.344 s. These longer runs include phase
printing. Different internal trajectories from cuDSS initialization make the
larger timing differences unsuitable for isolating the code change's benefit;
step counts alone are not speedup evidence.

The broader `prefix-corpus100` diagnostic covers all nine distinct candidate and
corner inputs over 20 us through the real sixteen-owner pool. Nine fresh CPU
trajectories and both nine-job GPU batches pass waveform and resource checks.
The warm GPU request batch takes 46.313 s versus 7.017 s for the fresh CPU batch.
These request-only intervals omit full-window processing and the frozen
repetitions, and they do not establish any acceptance timing. The initial script
setup failure, before any workload request, is retained separately; the corrected
script uses the manifest's default grid for light and explicit overrides for
boundary/reference.

For boundary/nominal, CPU and GPU attempt counts are 52,049 and 52,048, and solve
counts are 249,167 and 256,857. Request times are 3.489 and 21.867 s. This points
to execution cost per attempt as the main gap in that case. All nine cases'
counts and times remain in `prefix-cost-comparison.json`; this diagnosis does
not redefine the full-study timing boundary or performance target.

Trial 100's fresh full DPT q0/q1/q2 CPU comparisons, all eight integration/output
refinements and owner-pool resource audits pass. GPU request wall times are
7.634 / 10.097 / 12.324 s. Seven targeted Compute Sanitizer checks pass with the
same explicit coverage as the owner-pool checkpoint: wide and independent
expression race checks cover only their first resident chunk; selected recovery,
pivot, memory, synchronization and initialization checks cover all launches.

The new full Nsight Compute capture succeeds in 41 passes and retains source
and instruction counters. No spilling or bandwidth bottleneck appears; dependency
stalls and low utilization remain. GPU clocks are observed rather than fixed,
so profiled durations are not ordinary performance measurements. The native
report and compressed exports are retained, including the launch-delay warning.

Three initial canonical attempts expose evidence-packaging issues: historical
unformatted source snapshots entered lint, their unstaged removals remained in
the lint file list, and historical BUILD files entered Bazel package discovery.
Their failed logs remain separate. Lossless source archives preserve every
original source hash while keeping historical packages out of the active build.
The canonical retry and reconstruction audit retain their own results.

All nineteen canonical retry stages pass. The rebuilt worker has a different
whole-binary hash; its resident instruction encodings match the frozen worker.
The source-identity difference is the accompanying ADR description. Both binary
identities and compressed disassemblies remain explicit; measured/profiled runs
are bound to the original frozen worker, not relabeled as rebuilt executions.

`audit.json` checks source identities, raw trajectories, CPU waveform comparisons,
DPT refinements, resources, sanitizer records and canonical records: 75 raw
trajectories, 58 GPU comparisons and eight DPT refinements reconstruct. `files.json`
inventories retained artifacts. Full thirty-job qualification, both 9/36-job
performance invocations and every frozen acceptance threshold remain required.
