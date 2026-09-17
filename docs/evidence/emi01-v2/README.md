# EMI-01 v2 retained evidence

This version adds an all-corner passing reference and a near-boundary filter while preserving
all historical [v1 evidence](../emi01/README.md). The exact source freeze is
`8e45b4232741b82ec476fb6904b80f774eaf31d0`; the earlier contract commit is `9021ad4`.
The [invocation protocol](invocation-protocol.json) records the final manifest and simulator
identities before either complete measurement. The [results guide](../../emi01-v2-results.md)
contains canonical reproduction commands, decisions and interpretation limits.

The two fresh invocations run sequentially. Each includes 30 qualification jobs and 72
warmup/measured jobs: one warmup and three measured nine-job studies at each of one and four
workers. Builds, other benchmarks, full waveform audits, independent FFT recomputation and publication
run outside both retained timing intervals. Lightweight progress/result/spectrum reads and
documentation editing continue during measurement. Every simulation executes afresh; content-addressed lossless raw chunks can
share identical stored payloads. Required output includes closing files and checking/creating
chunks, without physical fsync. New designs need not share the same compression cost.

The [source review](review-source.md) preceded these measurements. The
[canonical validation index](validation/checks.json) records lint, build, CPU, ASan, UBSan,
lockfile, CUDA, acceptance and AC regression checks, including expected CUDA/sanitizer
analysis rejections. The [historical audit index](validation/historical-audits.json) records
fresh successful audits of both v1 invocations from their original source checkout.

The [design diagnostics](design/README.md) and
[intermediate qualification probe](validation/pre-evidence-qualification.json) are explicitly
outside retained evidence. That probe used intermediate source before the final failing-control
role gate and ran alongside build checks. Its timing is not a performance sample and its
identities do not certify final source. Final retained runs independently redo qualification.

Raw model bytes are fetched from the vendor through the existing checksum-pinned Bazel rule;
neither original nor adapted proprietary model bytes are redistributed. See
[model provenance](../../../reference/emi01/MODEL_PROVENANCE.md). Each invocation records its
exact source, model, Python/native runtime and ngspice identities. The invocation-private
read-only ngspice copy protects workers from build-output replacement; it does not repair the
historical ngspice build's sandbox-path-dependent executable bytes. Reproduction must qualify
its own recorded binary and must not rewrite historical identities.

Every job has a deck, simulator log, raw header/chunk index, complete result and (for ensemble
jobs) every one of the 986 bins for all four spectral observables. Numerical qualification,
valid predicted feasibility/infeasibility, simulation failures, and fixture-role coverage are
separate decisions. Missing or failed simulations satisfy neither passing nor failing controls.
Reported memory is per-process high-water RSS and enforced child limits, not a measured
simultaneous aggregate host peak.

## Artifact map

Both invocations are complete and accepted: **204/204 validated jobs, zero failures**. The
[paired CPU report](cpu-budget.json), [delivery supplement](delivery-supplement.json),
[independent numerical review](review-numerical.md), and
[final delivery review](review-independent-delivery.md) record closure. Numerical results and
raw chunk identities agree across the independently executed invocations.

| Invocation | Terminal and role accounting | Command boundary |
|---|---|---|
| Run 1 | [terminal](run-1/terminal.json), [reference cases](run-1/reference-cases.json) | [log](run-1-command.log), [outer time](run-1-command-time.txt) |
| Run 2 | [terminal](run-2/terminal.json), [reference cases](run-2/reference-cases.json) | [log](run-2-command.log), [outer time](run-2-command-time.txt) |

The canonical paired report audits both complete invocations before deriving any performance
budget. Reproduce it with the explicit `--reference-version=emi01-v2` selector; the command's
default remains v1 to preserve existing usage. Current source and runtime artifact checks are
separate from numerical recomputation and are not claims of execution attestation.

The independently authored [numerical reviewer](review/independent_numerical_review.py) pins the
v2 manifest and physical limits independently, reconstructs raw arrays without production parsing
helpers, and recomputes every qualification waveform/spectrum/stress/role decision. Its
[focused hostile tests](review/helper-checks.json) reject unsupported versions, altered contract
fields, nonfinite values, missing/failed/unsettled controls and contradictory classifications.
The canonical audit covers every warmup and measured job, including identities and terminal
completeness. The independent review is additional evidence, not a substitute for that audit.

The numerical reviewer initially encountered a NumPy-boolean JSON serialization error after
computation. The [final helper checks](review/helper-checks-final.json) bind the corrected
diagnostic-only serializer and regression tests; original pre-measurement helper identities and
initial error logs remain retained. Frozen harness/model/numerical predicates and measurements
were not changed. [Runtime checks](review/paired-runtime-identities.json) and the
[independent accounting report](review/independent-accounting-review.json) bind the completed pair.

The [final delivery-check index](validation/final-delivery-checks.json) binds the completed audit,
review, reporting-helper test artifacts and protected-scope checks after measurement.
