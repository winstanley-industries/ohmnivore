# Explicit CUDA work queues

The GPU harness now requests 32 compute and copy work queues. The actual child
environment is checked by the sixteen-owner process integration test, recorded
in each invocation and checked by its reconstruction audit. CPU worker settings,
the resident kernel and all acceptance thresholds are unchanged.

NVIDIA documents [eight default work queues and possible stream serialization](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
The sixteen private owners use independent compute and allocation streams.
Explicitly increasing both queue settings improves this bounded scheduling
diagnostic; it does not establish complete-study acceleration.

| Ordinary run | Eight queues | 32 queues | Diagnostic ratio |
|---|---:|---:|---:|
| First order | 5.877 s | 2.741 s | 2.144x |
| Reversed order | 5.142 s | 2.762 s | 1.862x |

Each entry is the median of three execution-only batches following one warmup.
Each batch executes sixteen independent 1.03 us reference/nominal circuit
prefixes through the real persistent owner pool. Input preparation and numerical
comparison are outside these batch intervals. They are not the frozen 9/36-job
measurements and cannot substitute for validation, spectral processing, closed
output, cold qualification or complete-window timing.

Separate Nsight Systems hardware-metric captures also complete both queue modes.
Whole-capture SM-active P95 rises from 10% to 19%; these distributions include
startup, warmup, measurement and CPU validation. They are not kernel-only
occupancy, exact queue assignments or acceptance timings. Native reports and
SQLite export commands are retained; `audit_queues98.py.gz` reproduces the
reported metric distributions from those exports.

Two initial profiling attempts fail before workload execution because injected
profiling loads an ambient GCC runtime. Their idle reports and failure logs
remain present. The successful attempts explicitly provide the pinned runtime
library paths and pass the existing strict runtime check. No guard is bypassed.

The reconstruction audit verifies all 326 retained raw trajectories, 320 GPU
waveform comparisons, six owner/affinity/resource audits, the 89 baseline source
identities and both successful hardware summaries. The ordinary and profiled
runs use the exact worker from the adjacent `resident-owner-pool` source archive.
The original diagnostic script changes the old harness environment at runtime;
reproduction requires that retained harness version. `selected.patch.gz` and
`selected.json` identify the production harness change separately. Its ensemble
tests, actual sixteen-owner process test and lint pass; logs are retained.

Every summary and audit keeps `acceptance_pass: false`. Full thirty-job
qualification, all resource limits and both complete performance invocations
remain outstanding. `files.json` inventories the retained evidence.
