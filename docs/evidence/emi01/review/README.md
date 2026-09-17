# Numerical review artifact fields

These exploratory JSON reports preserve their original bytes. Two reports contain historical
field names from an investigation of a broader spectral convergence floor. That proposal was
rejected. The governing gate remains **1 dB when either amplitude exceeds 10 uA (20 dBuA),
and 1 uA absolute difference below that floor**.

| Report | Accepted-gate fields and authoritative summary |
|---|---|
| `exploratory-before-heavy-refinement.json` | In `spectral_comparisons`, `old_failed_bins` counts failures against the governing 10 uA / 1 uA gate. `summary.original_spectral_failed_bins` totals those failures; `summary.pass` is false because the earlier coarse heavy policy failed. |
| `exploratory-heavy-waveform-sampling.json` | In `sampling_spectra`, `old_failed_bins` counts failures against the governing gate. `summary.sampling_spectral_failed_bins` is its authoritative accepted-gate total: zero. The independent waveform summary also reports zero failures. |
| `exploratory-heavy-integration.json` | Every row uses the governing gate directly: `failed_bins`, `max_db_above_floor`, and `max_uA_below_floor`. All 24 comparisons have zero failed bins. |

In the **first two reports only**, `failed_bins`, `above_floor_max_db`, and
`below_floor_max_abs_uA` describe the rejected **100 uA / 10 uA** proposal. Do not use those
fields to establish acceptance under the final contract. `old_failed_bins` means the unchanged
original gate; it does not mean that the final study adopted a replacement gate.

The heavy integration report's `q2-q3` labels compare 0.625 with 0.3125 ns, and `q3-q4` labels
compare 0.3125 with 0.15625 ns. These exploratory labels precede the final heavy manifest's
level numbering. Their shared output grids are 2.5 and 1.25 ns respectively.

The packaged `independent_numerical_review.py` uses only the governing 10 uA / 1 uA gate.
Its newly generated `failed_bins`, `above_floor_max_db`, `below_floor_max_abs_uA`, and
`summary.original_spectral_failed_bins` all refer to that accepted gate. Run it only outside
retained measurement intervals, following the reproduction command in
[`../review-numerical.md`](../review-numerical.md).

These files document exploratory review, not retained performance evidence. The earlier report
remains a failing historical diagnostic; the final heavy refinement reports document the
resolution. Retained invocations require their own numerical and terminal-accounting audits.
