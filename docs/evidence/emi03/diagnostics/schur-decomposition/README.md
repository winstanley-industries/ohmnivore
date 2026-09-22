# Schur-complement decomposition screen

**Negative performance result; no EMI-03 acceptance pass.** See the
[experiment report](../../../../emi03-parallel-decomposition.md) for the mechanism,
measurements, comparison and limits. The selected resident implementation remains
unchanged.

## Retained material

- `snapshots/`: all nine input decks, CPU capture logs, 144 compressed matrix
  records, independent KLU solutions, source/binary/input identities and both
  offline numerical-feasibility reports.
- `structure-probe.tsv.gz`: original compiled graph, expression dependencies,
  linear/reactive stamps and variable names. `records/schur-structure107.json`
  and `records/schur-structure117.json` record the two exact partitions.
- `records/`: complete passing replay output, initial failures, build logs,
  sanitizer logs and structural screens. `schur-summary118.json` summarizes
  every completed timing variant; it does not omit the slower variants.
- `profiles/`: hardware-counter reports for the initial, sparse and final
  prototypes, including the final four-block mapping. Profiler and sanitizer
  timings are not ordinary performance measurements.
- `sources/`, `source-at-validation/`, `scripts/`: compressed exact experimental
  sources and orchestration scripts. Historical scripts retain their original
  workspace paths. Source archives are compressed to prevent historical copies
  from becoming Bazel packages or lint inputs.
- `selected-replay/`: the matched selected-solver adapter, build configuration,
  all comparison timing rows, numerical counters, identities and targeted
  sanitizer results. Its first launch receives memcheck/racecheck coverage;
  those logs must not be described as all-launch coverage.
- `manifest.json`: SHA-256 digests of retained artifacts. `validation/` records
  the final repository checks separately from the experiment's timing data.

The final GPU source is `cuda/emi03_schur_probe.cu`, a `testonly` manual executable
unconnected to the simulator. To replay its archived inputs from the repository:

```sh
mkdir -p /home/adam/emi03-work/tmp
gzip -dc docs/evidence/emi03/diagnostics/schur-decomposition/replay/schur-replay117.txt.gz > /home/adam/emi03-work/schur-replay117.txt
TMPDIR=/home/adam/emi03-work/tmp bazel run -c opt --config=cuda --jobs=1 //cuda:emi03_schur_probe -- /home/adam/emi03-work/schur-replay117.txt
```

Add `--validate-only` after the replay path to run the 576 differential/residual
checks and four hostile rejections without the timing screen. The CPU snapshot
extractor is `//cpp:emi03_schur_probe`; it takes one retained input deck and emits
the 16 selected matrices for that case. Build all tools through pinned Bazel
toolchains.

## Retained failures and caveats

The initial prototype issued pageable uploads on the default stream and kernels
on a nonblocking stream. Three ordinary validation attempts failed intermittently
with partition/nonfinite errors. The fix orders uploads and launches on the same
stream. The failing sources, stderr and partial rows remain; a sanitizer pass
on the earlier version is not used to dismiss those ordinary failures. This bug
was in the new probe, not the selected resident owner's pinned-copy path.

The first detailed-timing source transformation failed its text-match assertion;
the files named `schur114-*` consequently contain an unchanged repetition of
variant 113. Only `schur114b-*` contains the added per-interior counters. The
summary identifies the latter explicitly. The dependency-level variant 116 was
slower overall and is retained, including its test on partition 117.

The selected-solver adapter's initial build emitted a shared-symbol naming
warning; the private scratch symbol was renamed without suppressing the warning.
Its initial attempt to reassemble an already captured linear Jacobian was
rejected because the nonlinear assembler requires a nonlinear descriptor. The
corrected adapter feeds the captured Jacobian directly into the unchanged
ordering/pivot/factor plan. One sanitizer command used the wrong filter syntax
and exited before running; the corrected commands and successful logs remain.

Only accepted CPU states and manufactured RHS vectors were screened. The final
Schur prototype performs one original-residual correction; it is not the
production controller's complete bounded-retry policy. Ordered local factors
are exercised with repeated copies of each matrix, not a changing trajectory.
No full-study timing or qualification claim follows from this evidence.
