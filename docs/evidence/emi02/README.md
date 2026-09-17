# EMI-02 retained evidence

Start with the [implementation and qualification report](../../emi02-results.md).

- [Independent dynamics review](INDEPENDENT_DYNAMICS_REVIEW.md): separately
  authored physical oracles, numerical findings and their regression checks.
- [Canonical validation](validation/results.json): exact commands, exit codes,
  durations and adjacent logs for all thirteen required stages.
- [run-1](run-1/qualification.json): one complete frozen CPU/reference qualification invocation, including
  every mandatory failure and numeric waveform artifacts for completed jobs.
- [Artifact audit](audit.json): independent recomputation passes while whole-study
  qualification remains false; the exact command and log are adjacent.

The run uses the [bounded bridge and audit](../../../reference/emi02/README.md).
Source contracts and simulator identities are frozen in its invocation record.
Vendor model text, flattened equations and model-bearing diagnostics are excluded.
Qualification failure must not be interpreted as predicted EMI infeasibility or
authorization for GPU work.
