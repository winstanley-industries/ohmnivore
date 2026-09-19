# Resident exact-state expression-cache checkpoint

This v53 development checkpoint passes the full frozen q0 DPT waveform, switching
and physical-classification comparison. GPU wall time is 9.624 s versus 0.618 s
on CPU. It does not pass the EMI-03 performance targets or complete qualification.

The private resident kernel adds exact IEEE-magnitude warp reductions, native
FP64 FMA in factor/triangular arithmetic, bounded packed expression/factor
metadata, and exact-state expression value/derivative reuse. Accepted-state
validation still freshly evaluates expression values. Eight resident tests,
twenty existing CUDA cases and the reduction regression pass. All eighteen
canonical validation stages pass, including expected sanitizer incompatibilities.

`identity.json` and `source.patch.gz` preserve exact source reconstruction. Raw
files are losslessly compressed; the audit reproduces the complete DPT comparison.
The older full reference/nominal trajectory in `../resident-full-reference`
predates this expression cache and cannot qualify this checkpoint. Full thirty-job
qualification and both independent frozen throughput invocations remain pending.
