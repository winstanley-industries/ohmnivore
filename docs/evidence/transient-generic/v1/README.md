# Generic transient experiments

Baseline core: `0899911151c6a2b49b8d1d60a1103a0cd20031d2`. All builds use the
hermetic optimized Bazel toolchain. The final comparison uses identical corpus
and benchmark sources with the baseline and candidate core implementations.
`manifest.json` binds the artifacts and current corpus source hashes.

The final `matched-corpus-*.csv` files retain all 21 circuits, both estimator
policies, both input resolutions and 22 repetitions per lane. Two warmups per
process/lane are excluded from timing summaries, leaving 18 measured repetitions.
Process order is baseline, candidate, candidate, baseline, pinned to CPU 4.
Every recorded accuracy/error/work field is identical across both implementations
and all repetitions. All 42 fine-resolution lanes satisfy the analytic limits.

Final fine-lane candidate/baseline geometric-mean time ratio: **1.00287**.
The worst lane's median ratio is **1.02595**, with overlapping observed ranges.
These measurements support performance neutrality within this experiment's
variation, not a runtime speedup claim or universal absence of regressions.
Per-case medians and ranges are in `matched-corpus-summary.json`; inspect them
rather than relying only on the aggregate.

## Rejected experiments and initial failures

- `baseline.csv`: initial partial corpus run using minimum-step divisor 10,000;
  the high-amplitude RC case exhausted h_min. Its earlier rows also retain coarse
  global-error failures. This incomplete run is not a performance baseline.
- `baseline-admitted.csv`: complete baseline screening with divisor 1,000,000,
  Newton bound 100 and maximum-step divisors 1 and 2. It retains global-error
  failures rather than treating local tolerances as global bounds.
- `interval-screen.csv`: same coarse screening resolutions with candidate policy
  2 added. Endpoint scaling reduced points in several cases but increased RLC
  accumulated error. It was rejected; `rejected-interval-policy.patch.gz` records
  that exploratory implementation, not an exposed production API.
- `rejected-cache-benchmark/`: the matrix-scale cache variant, using the final
  fine resolution (divisor 32). Its geometric-mean slowdown was 2.6%; it was
  rejected. Its code is in `rejected-companion-cache.patch.gz`.
- An early short-EMI timing attempt used a nonoptimized baseline executable.
  Those timings were discarded and are excluded from all comparisons here.

The current solver keeps the existing error policy and only prepares numeric
updates for C-bearing companion entries. No scale cache or per-call cache counter
remains. Coarse accuracy failures persist unchanged in the final CSV files;
only the fine lane supplies validated performance measurements.

## Reproduction

Decompress and apply `corpus-only.patch.gz` to a detached baseline checkout, then build
`//cpp:transient_accuracy_benchmark` with `-c opt` in each checkout. The original
comparison script is archived as `benchmark_matched.py.txt` so it remains an
unchanged execution artifact; copy it to a private `.py` path and adjust the two
checkout paths when reproducing. The command/executable hashes, return codes,
CSV hashes and process times are retained in `matched-corpus-execution.json`.
Exploratory endpoint screening uses maximum-step divisors 1 and 2, whereas the
final accuracy lane uses 1 and 32; do not conflate their refinement indices.

See [the corpus guide](../../../transient-regression-corpus.md) and
[ADR-007](../../../adr/ADR-007-generic-transient-improvements.md) for numerical
scope, analytic limits, public-default preservation and acceptance requirements.
