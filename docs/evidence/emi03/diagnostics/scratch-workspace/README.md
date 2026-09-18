# Incomplete EMI-03 development diagnostics

These are intermediate diagnostics, not complete qualification or performance evidence.
The source snapshot is commit `11c221f` plus `diagnostic-source.patch`; its kernel
hash is verified against `four-worker-dpt/invocation.json`. The subsequent source
formatting and added regression test are not relabeled into these identities.
The scripts retain their execution paths and require the pinned Bazel Python,
NumPy, CPU worker, model archive and the identified CUDA worker binary.

`four-worker-dpt` creates one fresh q0 CPU DPT reference and four independently
executed GPU copies with the existing limits. Sampled device residency remains
below 4 GiB, but per-job CPU-time exhaustion prevents completion and invalidates
the resource gate. Missing native end-of-job telemetry is not zero allocation.

`short-dpt` stops the DPT at 1.03 microseconds solely to obtain completed phase
telemetry. It is a different diagnostic workload and cannot satisfy any complete
EMI-03 accuracy, resource, cold-time or performance gate. No complete waveform
comparison is claimed for it. Phase telemetry has overlapping intervals and
must not be summed indiscriminately.
