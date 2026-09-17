# V2 design selection diagnostics

These are **exploratory diagnostics**, not retained qualification or performance evidence.
The [attempt table](attempts.json) preserves all 51 completed design/refinement probes,
including failures, parameter identities, exact model/binary identities, raw/deck fingerprints,
and derived outcomes. The full exploratory raw payloads remain local in the recorded `/tmp`
directories. Canonical retained v2 invocations independently rerun the selected cases.

The first cases increased L/C and magnetic geometry under the original network and requirements.
Increasing DM inductance alone moved sharp common-mode resonances and did not yield a monotonic
boundary. The selected 330 uH / 1 mH / 1 uF / 47 nF reference passes the physical screens and
research mask. Changing only its X capacitor to 80 nF produces a stable fast-corner rejection
about 0.483 dB below the required reserve; 85 nF was too close at about 0.016 dB below it.
All original inverter/load/corner, physical, mask, reserve, settling and accuracy settings remain
unchanged. The approximately 3.8 kg geometry is hypothetical and is not a mass optimization result.

The independent [coarse-grid review](coarse-anchor-review.json) rejects the initial coarse
integration policy. The selected [reference](anchor-fine-review.json) and
[boundary](boundary-fine-review.json) comparisons pass at 0.625/0.3125/0.15625 ns, including
waveforms, every spectral bin, output sampling and physical/status recomputation. These reports
support the pre-evidence choices; only complete canonical runs can establish retained acceptance.

The diagnostic scripts use the existing pinned runtime and local probe directories. They never
change production simulator behavior. The canonical commands in
[the reference guide](../../../../reference/emi01/README.md) reproduce the selected v2 workload
without requiring these local diagnostic directories.
