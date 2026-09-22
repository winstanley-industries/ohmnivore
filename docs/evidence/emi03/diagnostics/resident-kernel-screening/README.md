# Rejected resident-kernel screening experiments

Both changes are rejected. Neither establishes a complete-workload speedup or
EMI-03 acceptance. The working implementation remains the corrected
[owner-pool checkpoint](../resident-owner-pool/README.md).

| Experiment | Baseline median | Candidate median | Candidate / baseline |
|---|---:|---:|---:|
| Serial sparse triangular solve | 2.285 s | 4.149 s | 1.816 |
| Outlined timestep routine | 2.285 s | 2.369 s | 1.036 |

Each experiment runs a fresh CPU reference, one GPU warmup per variant, then three
alternating ordinary request measurements per GPU variant. Input is the shortened
1.03 us coupled reference/nominal case. Complete raw output closes inside request
wall time. CPU comparison occurs afterward; these diagnostic medians are not the
frozen complete-study timing boundary. Two additional runs print internal kernel
phase clocks and are excluded from those medians. No other GPU workload or build
runs during the ordinary measurements.

The serial trial removes dependency-level scheduling in sparse forward/backward
solves, keeping each row's ordered FP64 FMA sequence and all certification checks.
It is substantially slower. The outlining trial keeps one timestep function out
of line and marks immutable kernel parameters `__grid_constant__`, reducing source
inlining without changing the numerical algorithm. It shows no useful gain and
is also discarded. Neither trial received full EMI qualification or promotion.

All 22 retained raw outputs parse completely and all 20 GPU outputs pass every
saved-observable CPU waveform check on the 0.5 ns diagnostic grid with the original
tolerance formulas. Every trajectory has 3,158 accepted points. This short window
cannot supply full-window spectra, stress/loss classification or refinement
qualification. Small raw-byte differences across otherwise identical baseline
runs remain visible; bitwise output determinism is not claimed.

`identity.json` in each trial records its binaries, source hashes and fresh input
identity. `candidate.patch.gz` applies to the exact source archive in the
owner-pool checkpoint; the reconstruction audit verifies every patched source
hash. Every raw trajectory is losslessly compressed and indexed. The private
adapted vendor deck is not published. Build logs, commands, ordinary/profile
records, native telemetry and the execution/audit scripts remain available.
