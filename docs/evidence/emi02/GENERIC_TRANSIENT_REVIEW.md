# Generic transient follow-up review

This follow-up was implemented and reviewed locally without subagents. The earlier
independent EMI-02 reviews remain historical; this file does not claim another
independent human or agent review.

## Retained invariant

`PreparedTransientCompanion` owns validated immutable G/C copies. Its checked
factory initializes the complete canonical union matrix. Every C coordinate
appears exactly once in that union, in canonical C order. Preparing only those
coordinates preserves the association between the update loop index and C values;
the stored union index selects the destination, and the optional G index selects
the original C-only or G+C arithmetic. G-only entries cannot depend on timestep.

The metadata entry remains two indices wide. No scale cache, policy selector,
per-call counter or additional matrix copy remains. Positive/finite step and alpha
checks and scale overflow checks still precede numeric updates, including when C
is empty. Every changing value retains its finite check. Recomputing all changing
entries on the next successful form repairs partial numeric failure. Signed-zero,
cancellation, invalid-input precedence and recovery tests cover those boundaries.

There are no changes to physical equations, error norms, timestep decisions,
public default policies, KLU checks, native-device semantics, Rust or CUDA code.
The source diff contains no circuit, model or benchmark-specific dispatch.

## Cross-workload evidence

The 21 independent analytic circuits cover RC, RLC and nonlinear charge dynamics,
with both estimators. Forty-two fine lanes pass the frozen analytic error limits;
forty-two coarse lanes compare every prepared and fully checked trajectory bit,
step decision and sparse-solver work count. Twenty-six coarse analytic failures
are unchanged baseline limitations and remain visible in the benchmark CSVs.

The endpoint-scale policy was rejected for increased accumulated RLC error. The
scale-cache variant was rejected for a measured corpus slowdown. The retained
simple update plan has identical recorded numerical/work fields in all benchmark
lanes. Fine-lane aggregate time is 0.3% slower; the worst median is 2.6% slower with
overlapping observed ranges. This is not evidence of a runtime speedup or proof
that no untested workload could regress.

Run-3 completes every CPU/ngspice job once and passes all unchanged qualification
gates. Its retained-artifact audit passes. `run-3-parity.json` records identical
raw waveform hashes, measurements, classifications, importer identities and
numerical work counters for all thirty CPU jobs relative to run-2. Original failed
and passing invocations remain intact. `validation-v3/results.json` records the
canonical build, native-device/AC/acceptance, sanitizer and explicit compatibility
checks on this source.

## Reproducibility and publication

The generic artifact manifest and corpus source hashes were verified, including
uncompressed rejected-patch identities. The archived corpus-only patch applies to
baseline commit `0899911` and exactly reproduces the current test/benchmark files.
Run-3 source identities match the checkout. Retained deck/log files contain no
vendor subcircuit definitions, and no vendor model archive, library or flattened
model deck is included. The public qualification audit independently recomputes
all measurements and checks retained file, source, model and executable identities.
