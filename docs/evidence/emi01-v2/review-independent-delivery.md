# EMI-01 v2 independent delivery review

**Disposition: PASS for the bounded external CPU reference follow-up. No remaining blocking
finding was identified in the reviewed delivery.** This review covers the completed retained
pair, accounting, provenance, numerical interpretation, final results/evidence documentation,
roadmap update, and whole-PR description. Publication and the final committed-head check are
separate actions. No production solver or GPU implementation is authorized by these results.

Reviewer: `independent_final_review`, separate from the implementation and numerical-diagnostic
authors. This reviewer authored the [source review](review-source.md) and the independent
accounting checker below, but did not author the harness, fixtures, numerical helper, or measured
runs. All delivery computations ran after both retained timing intervals. No additional
simulation, build, benchmark, or raw-waveform FFT was run by this reviewer.

## Review evidence and method

The implementation/source freeze is `8e45b4232741b82ec476fb6904b80f774eaf31d0`; the manifest SHA256
is `ec68ddb48866d8127fbdd0ecf411dfc326b6c7bc7929c10429e40edd9cd2c81f`.

The independently authored [accounting checker](review/independent_accounting_review.py) uses
only the Python standard library and no production scheduling, classification, mass, ranking,
or reporting helpers. Its [retained report](review/independent-accounting-review.json) binds its
source, manifest, frozen commit, invocation protocol, both metadata/terminal records,
[paired CPU report](cpu-budget.json), and [delivery supplement](delivery-supplement.json).
The final execution used the pinned Python 3.12.13 interpreter and succeeded; Python optimization
was separately rejected. [Execution identities](review/independent-accounting-execution.json)
record both checks. Pinned Ruff formatting and lint checks passed for the checker.

This review independently reconstructed the exact schedules, rehashed all 708 indexed small
job artifacts plus all 102 result records per invocation, and checked root artifact indexes.
It checked every candidate/corner identity, attempt/status, physical-limit decision, settling
decision, mass, stored spectral grid and limiting bin, ranking, and study summary. It compared
indexed raw chunk identities; decompression and numerical waveform recomputation belong to the
canonical full audits and separately authored [numerical review](review-numerical.md).

| Invocation | Terminal SHA256 |
|---|---|
| Run 1 | `82bde66cfc0d46a7fefb028f4a2063463d15a4c726c0de10bbbc0275118c37f2` |
| Run 2 | `ec4f149deabc21bd24b3d8052684a0162689c2dc4b017f2a38003c273478df9f` |

Both complete canonical audits succeeded before the paired report was produced. All 102
corresponding metrics, classifications, candidate/corner identities, and indexed raw payloads
match exactly across invocations. Within each invocation, all 72 warmup/measured jobs match
their respective finest qualification metrics, spectra, and indexed waveform payloads.
Distinct terminal records and command histories are execution evidence, not cryptographic
attestation of separate physical runs.

## Numerical qualification and useful case coverage

Each invocation has exactly 102 validated jobs, zero failures, and one attempt per job:
three qualified DPT results, 55 predicted-feasible ensemble results, and 44 predicted-infeasible
ensemble results. All 40 numerical qualification checks and all nine fixture-role groups pass.
Missing, failed, or unsettled simulations are not counted as either feasible designs or valid
failing controls.

The separate numerical reports bind their actual revised helper and the exact retained
terminal records. For each invocation they independently recompute 30 qualification jobs,
144 spectral comparisons over all 986 bins, 216 waveform comparisons, 54 settling checks,
40 qualification decisions, and nine role checks. They report zero failing bins, waveform or
settling failures, and metric discrepancies. The largest integration differences are
0.713180 dB above the 10 uA floor and 0.748781 uA below it. The original 1 dB/1 uA thresholds
were preserved. The diagnostic's post-measurement NumPy-to-JSON conversion correction changed
no numerical predicates or frozen source; original failure logs, corrected source identities,
successful reruns, and regression checks are retained.

`reference` passes every physical and settling screen and every corner, with worst margin
10.102245716701447 dB. `boundary` passes all physical and settling screens but rejects the
fast corner at 5.516619038826818 dB: 0.4833809611731823 dB below the unchanged 6 dB threshold.
The limiting bins are conductor A at 2.85 MHz and 150 kHz respectively. `light` is valid and
infeasible at all three corners. These classifications remain stable across all three
integration refinements. Names do not supply classifications; stored spectra and physical
measurements support them.

Mass is independently reproduced as 0.2285867686 / 3.8062284436 / 3.8084193316 kg for light,
boundary, and reference. The reference is the sole all-corner feasible member in this finite
set. Every measured ranking requires all corners and global numerical qualification. The
time-to-lightest field conservatively waits for all nine terminal records; qualification
already evaluated these inputs, so this does not measure discovery of a new design.

DPT qualification includes the required resolved, strictly decaying three-peak ringing
measurement, finite switching edges, loaded second turn-on, and signed energies. The observed
gate peak remains approximately 24.263 V. Passing the frozen crossing/edge observability gate
does not establish an absolute gate-rating, safe-operating-area, or hardware qualification.

## Accounting and performance interpretation

The checker independently reproduced summary/queue/completion quantiles, phase and telemetry
totals, all 12 FIFO counterfactual samples, nested factor/solve ceilings, and supplement values.
The supplement is bound to the exact paired terminal identities and each job hash.

| Invocation | Whole invocation s | Serial median study s | Four-worker median study s | Serial / four-worker jobs per hour |
|---|---:|---:|---:|---:|
| 1 | 3705.178348 | 611.736631 | 207.528919 | 52.963969 / 156.122820 |
| 2 | 3716.739840 | 614.216315 | 207.654594 | 52.750146 / 156.028332 |

Throughput is the median of three individual nine-job study rates. P95 is empirical nearest-rank
over 27 measured jobs per mode. Invocation time includes setup, qualification, warmups, complete
studies, validation, spectral processing, and closed required output; build/fetch, later audits,
and physical fsync are excluded. The roughly 0.30 s one-time setup is separately reported and
included in invocation time. Summed parallel service and nested native timers are not added to
study wall time. Memory observations are individual process high-water marks, not an aggregate
simultaneous host peak.

Simulator process service accounts for approximately 98.82–98.87% of measured job service.
Factor plus solve accounts for only about 11.38–11.45% of native analysis, giving an optimistic
analysis-only infinite-acceleration ceiling of about 1.13x. This cannot support a 2x complete-study
claim for factor/solve acceleration alone. Whole-simulator counterfactuals omit new GPU transfer,
setup, and certification costs and establish no measured GPU usefulness. Fixed-input replay
performance also establishes no unseen-candidate search throughput.

## Provenance, preserved scope, and resolved findings

All 25 frozen source identities match both the working tree and the freeze commit. Both runs
use the unchanged model archive/member/adapter identities and the same recorded interpreter,
eight native artifacts, limits, and numerical thread count. The retained
[runtime audit](review/paired-runtime-identities.json) matches current canonical artifacts.
This reviewer also independently rehashed the eight native artifacts and canonical ngspice.
The recorded ngspice SHA256 is
`fa5eaf1ed5ff33be4f250d6241f25afcacc9a5d7e56750d705816561f18b8eaa`.

The invocation-private mode-0500 executable snapshot prevents ordinary build-output replacement
from changing its workers' selected file. Artifact checks are not execution attestation, and
the existing ngspice sandbox-path-dependent build bytes remain a reproducibility limit. Future
rebuilds must record and qualify their own executable identity. Historical v1 records were not
relabelled; their full audits pass using detached `c621e5c` source.

All 12 canonical validation log hashes were checked, including the two expected explicit
CUDA/sanitizer analysis rejections. Reporting helper tests pass in normal and optimized Python;
the assertion-based numerical/accounting helpers reject optimization. Protected production
C++/CUDA, Rust/Cargo, acceptance and replay paths match `17519fc`; all 11 pre-existing AC
evidence files are byte-identical to that baseline. Historical `docs/evidence/emi01` is unchanged
from `c621e5c`. No EMI-02 production semantics, GPU implementation, or dispatch is introduced.

Resolved review findings were the missing failing-control role gate, the supplement's absent
terminal binding, missing explicit version checks in its runtime helper, premature completion
wording during measurement, and the final throughput-description mismatch. Corrections and
discriminating tests are present; the last correction changed wording to median per-study
throughput, with no numerical or measured-evidence change. The final results tables, evidence
README, roadmap, and whole-PR description were reviewed against the completed artifacts.

The approximately 3.8 kg designs remain predictions under a hypothetical geometry/mass model,
fixed parasitics, linear magnetics without core loss, and an unmeasured lumped load. Missing
DM self-capacitance, winding-fill and fringing validation, model recovery/self-heating limits,
and absent hardware correlation remain explicit. Neither the passing reference nor its stable
near-boundary comparison establishes manufacturability, aviation compliance, a statistical
tolerance guarantee, or a minimum-mass optimum.
