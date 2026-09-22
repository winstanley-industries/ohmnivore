# Correction-equation diagnostics

These four source variants are unselected development experiments. None has full
EMI-03 qualification or an accepted performance result. Original FP64 linear and
nonlinear certification limits, accepted-Jacobian checks and refinement bounds
remain unchanged throughout the trials.

Trial 94 solves the Newton correction equation directly using a compensated
residual. Its ten resident tests pass, but the shortened coupled case fails
`solution-validation` and emits no accepted raw trajectory. Trial 95 adds failure
instrumentation: a nearly zero correction-equation row has componentwise error
one despite a tiny normwise residual. The instrumentation is not a performance
candidate, has no separately executed test suite, and also fails closed. Both
failures and their native cleanup telemetry are retained.

Trial 96 retries the original affine equation on GPU when correction-equation
certification fails. Its ten resident tests, all three fresh DPT CPU comparisons,
eight DPT refinements and actual-pool resource audit pass. DPT GPU request times
are 7.713 / 9.954 / 14.629 s. The short coupled median is slightly worse than the
baseline: 2.103 versus 2.090 s. Two 20 us comparisons give 26.631 versus 29.257 s,
then 29.532 versus 30.437 s. This inconsistent improvement does not select the
change or qualify the complete coupled waveform.

Trial 97 also compensates the companion-history accumulation. Its ten resident
tests and short waveform comparisons pass; the short median worsens to 2.149
versus 2.090 s. A 20 us run takes 28.023 s beside the same 30.437 s baseline.
There is no full DPT qualification of trial 97. Internal step counts vary between
these runs, so reduced counts alone are not treated as acceleration evidence.

The twenty-microsecond runs include phase printing and are diagnostics. None
includes full-window spectra, physical classification, complete-study output
processing or the frozen performance repetitions. Trial 94's failed partial
elapsed time is never included in a throughput comparison. The existing cuDSS
initialization can produce different internal FP64 trajectories across runs;
passing waveform comparisons do not imply bitwise identity.

Source patches and original source/binary identities reconstruct against the
adjacent `resident-owner-pool` archive. Generic freezer labels are preserved;
each variant's `scope.json` explains its actual change. Compressed scripts retain
execution and audit details. `audit.json` reproduces retained source identities,
raw trajectories, waveform/DPT comparisons, refinements and failure accounting.
`files.json` inventories the evidence. All acceptance flags remain false.

DPT source snapshots are stored losslessly in `sources.tar.gz` so historical
BUILD files are not interpreted as live Bazel packages. Every original source
hash is rechecked after reconstruction. The archived-source audit script and
its passing log accompany this storage-only change.
