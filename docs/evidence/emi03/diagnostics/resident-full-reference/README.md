# Complete resident trajectory: development evidence

**This is a failed wall-budget result, not EMI-03 completion.** The complete
frozen q0 reference/nominal circuit reaches 200 us and passes the CPU waveform,
spectral and physical-classification comparison. GPU execution takes 348.195 s,
exceeding the frozen 120 s limit. An explicitly extended 600 s diagnostic ceiling
allowed inspection of the complete trajectory; it does not change acceptance.

| Observation | Fresh CPU | GPU |
|---|---:|---:|
| Standalone worker wall time | 33.010 s | 348.195 s |
| Child CPU time | 32.948 s | 10.506 s |
| Output points | 461,672 | 483,937 |
| Physical classification | Predicted feasible | Predicted feasible |
| Frozen job status | Complete | Resource limit |

All four A/B/CM/DM spectral comparisons have zero failed bins. The largest
above-floor difference is 0.01824 dB. Complete raw output, spectral arrays,
stress/loss metrics, execution records and source identities are retained.
Explicit device-allocation telemetry reports zero CPU fallbacks, zero outstanding
device bytes and zero cleanup/allocation failures. This single-job diagnostic
does not establish aggregate host/device resource limits or ensemble performance.
It inherits the host affinity and is not a frozen CPU worker-mode comparison.

At 120.127 s the GPU had emitted 176,470 complete records and reached only
66.111 us. The partial-file observations remain diagnostic progress records;
they are never admitted as completed results. The GPU emits 22,265 more points
than CPU overall. Most additional points arise between about 1.03 and 8 us;
subsequent cumulative point differences remain approximately constant. Both
executions record 125 derivative-history fallback entries and 122 recoveries.
This localizes additional timestep work but does not establish its numerical
cause. Device cost per point remains the larger performance problem.

The measured implementation uses exact integer reductions of FP64 magnitudes,
explicit native-FP64 FMA in factor/triangular updates, and bounded compact metadata
cached in shared memory when it fits. Residual guards, GPU pivot recovery,
independent job state and accepted-Jacobian validation remain in force. Seven
resident tests and 20 existing CUDA cases pass for this snapshot. The separate
reduction target checks 1,024 groups against a host FP64 oracle, including low-word
ties, subnormals, infinities, and quiet/signaling NaNs. Its source is unchanged
between that test record and this snapshot.

`identity.json` identifies 97 numerical/build files and five comparison inputs.
Apply the decompressed `source.patch.gz` to its recorded base, and use the separate
CPU and CUDA build commands. The snapshot predates the later exact-state
expression cache and its additional boundary test; it cannot qualify later code.

`raw-index.json` binds compressed and uncompressed bytes. The fresh CPU output is
byte-identical to the previous resident CPU reference, so its existing compressed
blob is reused for storage only. CPU execution and timing were performed anew.
The compressed scripts record the actual diagnostic and reconstruction audit;
they use the recorded host's scratch paths and pinned Bazel runfiles. The audit
reconstructs source, validates binary identities and complete raw outputs,
recomputes every stored metric and spectral array, and confirms the wall-gate
failure. Its `audit_pass` validates this record; `acceptance_pass` remains false.

The full thirty-job qualification, refinement, aggregate resources, zero-failure
admission and both independent 9/36-job timing invocations still have to pass.
