# EMI-03 bounded transient ensemble experiment

`ensemble` implements the exact finite experiment in
[`docs/emi01-gpu-experiment-contract.md`](../../docs/emi01-gpu-experiment-contract.md).
It does not enable automatic dispatch. Each job contains the complete inverter,
filter, load and chassis, and every replay replica independently solves its
initial-value problem. All results remain research predictions.

Every invocation creates fresh thirty-job CPU and external ngspice references
at q0/q1/q2, then independently executes and validates all thirty GPU jobs.
Raw waveform comparisons, every spectrum bin, switching/stress/loss metrics,
classification, refinement checks and fixture-role checks use the unchanged
EMI-01/EMI-02 measurement functions. Failed qualification retains all terminal
records and prevents performance collection. Invalid raw output is retained
compressed; proprietary flattened model text and simulator diagnostics are not
published. Diagnostics are retained by hash and typed failure category.

Run the hermetic targets with absolute artifact paths. The caller supplies the
explicit opt-in CUDA worker path; the default CPU build does not need CUDA.

```sh
bazel test //reference/emi03:ensemble_test //reference/emi03:worker_process_test
bazel run -c opt //reference/emi03:ensemble -- \
  --gpu=/absolute/path/to/emi03_gpu_worker --out=/absolute/new/invocation-a
bazel run -c opt //reference/emi03:ensemble -- \
  --gpu=/absolute/path/to/emi03_gpu_worker --out=/absolute/new/invocation-b
```

The resident follow-up is available for continued development with
`bazel build -c opt --config=cuda --jobs=1 //cuda:emi03_resident_worker` and
`bazel test -c opt --config=cuda --jobs=1 --local_test_jobs=1
//cuda:emi03_resident_test //cuda:emi03_cuda_test`. Pass its absolute binary path
to `--gpu` explicitly. Its telemetry identifies
`transient_algorithm=resident-be-trap-fp64-v1`. It is **not qualified**: diagnostic
and unit-test passes cannot replace full EMI correctness, resource, cold-time,
median, and P95 gates. See ADR-008 for its ownership and numerical policy.

`--qualify-only` stops after complete qualification and cannot publish a speedup.
Use two independent invocations, and run no other build or benchmark during
evidence collection. Source identities are checked again before the summary is
published. A changed source invalidates the invocation.
`--audit=/absolute/invocation` replaces `--out` to independently reconstruct all
raw chunks, spectra, measurements, classifications and certification checks.
It also reconciles complete timing artifacts, worker request sequences, native
allocation peaks and recorded memory observations. Both passing and failed
qualification evidence are auditable.

After qualification, each CPU/GPU mode uses actual persistent worker processes
for one complete warmup and five measured complete studies at both nine and
thirty-six jobs. CPU worker counts are 1, 4 and 16; the GPU candidate uses four
persistent workers to bound CUDA context residency. Replicas appear in
replica-major order and retain their physical and numerical reference identity.
The persistent protocol is one line of three absolute paths separated by tabs:
`input.cir`, `output.raw`, and `statistics.json`. One JSON response echoes `input`
and has status `complete` or a typed failure. State is rebuilt for every request.
GPU workers additionally write `statistics.json.gpu.json` with execution,
fallback and allocation telemetry; common nonlinear statistics retain their
existing schema. Worker process IDs and request counts establish persistence.
The recorded Ryzen 9 9950X3D host uses one hardware thread from each physical
core. CPU worker affinity is the first 1, 4 or 16 entries of
`[4,6,20,22,0,2,8,10,12,14,16,18,24,26,28,30]`; GPU workers use its first four.
Fresh CPU/ngspice qualification shares the first four cores. The harness verifies
the host topology and available affinity before executing, and records actual
worker affinity.

The parent enforces per-request 120 s wall and 110 s CPU budgets without treating
cumulative worker CPU time as a per-job budget. CPU workers keep the 1 GiB address
space budget. CUDA virtual reservations do not count as resident host memory.
Whole-invocation host accounting samples the harness and every recursively
enumerated live descendant at 20 ms intervals, including fresh CPU/ngspice
qualification, refinement and persistent workers. It conservatively sums each
process's Linux resident high-water mark and applies the 16 GiB gate. Observed
excess or unavailable monitoring rejects the invocation and prevents throughput
publication. Process births/exits between samples remain a sampling limitation;
native CPU address-space limits independently constrain oracle jobs.

The native CUDA ledger enforces 256 MiB per worker across controlled expression
buffers and cuDSS device allocations, and the harness conservatively sums worker
allocation peaks against 4 GiB. Additional driver/context residency is observed
with `nvidia-smi` before worker creation, every 100 ms, and after each complete
study. The retained records include the raw preworker baseline, total observed
usage, and `max(0, sampled_peak - baseline)` incremental usage. The baseline is
subtracted once to exclude preexisting desktop allocations. An observed
increment above 4 GiB or unavailable telemetry terminates affected workers and
rejects the gate. These residency samples do not prove a bound on opaque driver
transients shorter than the sampling interval. Native ledger limits and sampled
residency are reported separately. No competing benchmark is permitted.

There are at most sixteen concurrent job states. Raw output remains limited to
512 MiB and two million points. Dead workers are never retried. Failed GPU
requests retain available native telemetry; missing telemetry is explicit and
cannot be treated as zero allocation or successful throughput.

Complete study timing charges scheduling, model expansion, execution, raw output
closure and lossless archiving, independent per-result CPU validation, spectral
processing and terminal job records. Setup is recorded separately and charged
to cold time. CPU cold time includes fresh CPU/external qualification; GPU cold
time additionally includes GPU qualification. Common qualification is charged
once, and initial source/model/binary setup is charged once to both cold studies.
The authoritative study clock is captured after required raw, spectral and
terminal output closes; its scalar timing record is retained for audit.
Empirical P95 is the nearest-rank statistic of the five measured studies.
For each ensemble size, the comparator is the fastest qualified CPU mode for
each reported statistic. The gate requires at least 2.0x median, 1.5x empirical
P95 and no cold regression, with no failed, missing or fallback jobs. The same
requirements must pass independently in both invocations.

These timings concern repeated replay of known inputs after charged
qualification. They establish neither novel-candidate throughput nor faster
design discovery. Retain failed gates as negative evidence; dispatch stays closed.
