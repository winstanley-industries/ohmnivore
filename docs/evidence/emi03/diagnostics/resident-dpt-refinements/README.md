# Resident DPT refinement checkpoint

The v64 candidate passes three complete frozen DPT comparisons against fresh CPU
references and all eight CPU/GPU integration/output-refinement checks. GPU q0/q1/q2
wall times are 9.686 / 12.190 / 19.602 s; CPU times are 0.667 / 0.817 / 1.418 s.
The per-job wall, CPU, raw and point limits hold for these six individual runs.
Device cleanup and explicit allocation telemetry pass; aggregate residency was
not measured here. No speedup or complete EMI-03 acceptance is established.

This candidate retains exact-state expression caching and adds ordered row/source
lists, compact shared scratch with disjoint temporary lifetimes, optional shared
CSR indices and warp-aligned expression operators. Ten resident tests and all eighteen canonical validation stages pass.
`validation/lint-initial.log` retains a formatting failure in an earlier evidence
helper; the helper was reformatted and the full validation sweep rerun successfully.
The original coupled-reference trajectory and earlier DPT checkpoints retain
their own source identities and cannot qualify these changes.

`identity.json` and `source.patch.gz` reconstruct the tested sources. Six raw
trajectories are losslessly compressed. The audit reproduces every metric, all
three full waveform comparisons and eight refinement checks. The pinned model
archive is imported afresh; adapted vendor model text is not published.

Still required: fresh ngspice-backed complete thirty-job qualification, aggregate
resource accounting, zero failed/missing jobs and both independent 9/36-job
median/P95/cold-time gate invocations. EMI-03 remains incomplete.
