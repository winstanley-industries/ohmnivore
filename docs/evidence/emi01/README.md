# EMI-01 retained evidence

The [results and reproduction guide](../../emi01-results.md) interprets this evidence under
the exact [ADR-002 contract](../../adr/ADR-002-inverter-emi-design-study.md). The frozen harness
is commit `bf1165fe92fb5f527bbddfcb53374956d62929a5`, based on accepted roadmap commit `17519fc`.
No production simulator behavior or historical AC evidence is changed.

## Complete independent invocations

| Invocation | Terminal record | Outer command and resource accounting |
|---|---|---|
| Run 1 | [102 expected, 102 validated, zero failures](run-1/terminal.json) | [log](run-1-command.log), [time](run-1-command-time.txt) |
| Run 2 | [102 expected, 102 validated, zero failures](run-2/terminal.json) | [log](run-2-command.log), [time](run-2-command-time.txt) |

Both invocations independently execute 30 qualification jobs and 72 warmup/measured jobs.
Each contains one full warmup and three measured nine-job studies at each of one and four
workers. Every scheduled job has its own completion record. All 40 qualification checks pass
in each invocation. There are three qualified DPT records and 99 predicted-infeasible ensemble
records per invocation. Infeasible candidates are valid numerical results, not solver failures.

The [paired CPU budget](cpu-budget.json) calls the canonical full audit on both invocations
before deriving its results; its [execution log](paired-report.log) records completion.
The separate [Run 1 audit](run-1-audit.log) also checks all terminal/artifact identities.
Independent numerical recomputation is retained for
[Run 1](review/retained-run-1-numerical.json) and
[Run 2](review/retained-run-2-numerical.json), with
[canonical artifact identity checks](review/paired-runtime-identities.json).
The [independent delivery review](review-independent-delivery.md) reconciles final accounting,
numerical qualification, performance arithmetic, provenance and protected scope.

Those successful artifact checks precede the final sanitizer/default rebuilds. The existing
ngspice build embeds sandbox-specific installation paths; a later rebuild changed its binary
identity and correctly failed the [strict recheck](review/post-validation-runtime-recheck.log).
The [investigation](review/ngspice-reproducibility.md) records exact recovery of the historical
measured binary and a separate rebuilt-oracle qualification. All 30 diagnostic numerical
payloads, spectra, metrics and qualification checks match the retained reference exactly.
This establishes bounded numerical reproduction, not byte-identical builds or new performance
evidence. The original metadata and both retained timing streams remain unchanged.

`invocation_wall_s` starts inside the harness after interpreter/import startup and includes
verification, adaptation, qualification, warmups, all measured studies and completion records.
The outer `/usr/bin/time` records also include Bazel and interpreter startup. Required output
means closed files, without physical fsync. Maximum RSS values describe individual processes;
they are not simultaneous aggregate system memory. Build/fetch work is outside study timing.

## Raw data and interpretation

Each `run-*/jobs/<identity>/` directory retains decks, simulator log, validated result, raw
header/index and, for ensemble jobs, all 986 four-channel spectral bins. Raw binary payloads
are losslessly stored in bounded content-addressed gzip chunks under `run-*/blobs/`. Restore
and verify them through the canonical audit or the independently implemented review reader.
An absent/corrupt chunk is a failure; it is never replaced by a successful-looking spectrum.
All repeated jobs rerun simulation and validation. Compression can reuse identical chunks
within an invocation, and Git can deduplicate identical stored objects across invocations.

The proprietary vendor archive and adapted model bytes are **not redistributed**. Exact fetch,
member, adaptation and model identities are recorded in each metadata file and in the
[model provenance audit](../../../reference/emi01/MODEL_PROVENANCE.md).

The `review/` files beginning `exploratory-` are historical diagnostics collected before retained
timing. One is intentionally a failing coarse-grid result. Read their
[field-name sidecar](review/README.md); they do not replace the retained qualification records.
The [validation index](validation/checks.json) and
[final delivery checks](validation/delivery-checks.json) preserve canonical
repository regression results separately from these performance measurements.

To repeat the host artifact check, build the default study target, then run from the repository
root using the pinned interpreter (substitute the directories of the new invocations):

```sh
bazel build //reference/emi01:study
bazel-bin/reference/emi01/study.runfiles/rules_python++python+python_3_12_x86_64-unknown-linux-gnu/bin/python3 \
  docs/evidence/emi01/review/runtime_identity_audit.py \
  --run /absolute/new/run-1 --run /absolute/new/run-2 \
  --output /absolute/new/runtime-identities.json
```

That check compares current canonical paths and bytes with observations from runs on that
host. It is not execution attestation or a claim that absolute cache paths are portable.
The canonical full evidence audit and numerical-reader reproduction commands are documented
in the results guide and [numerical review](review-numerical.md).

The [delivery supplement](delivery-supplement.json) adds queue/completion quantiles and
counterfactual medians without changing the frozen report. Its
[small reporting script](review/delivery_supplement.py) regenerates those values from the
audited paired report and job records using the same pinned Python interpreter.
