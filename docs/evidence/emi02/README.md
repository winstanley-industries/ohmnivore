# EMI-02 retained evidence

Start with the [implementation and qualification report](../../emi02-results.md).

- [Run-2](run-2/qualification.json): the passing complete invocation, with all
  thirty CPU and thirty independent ngspice jobs, eighty refinement checks and
  thirty differential comparisons. Numeric waveform chunks and identities are retained.
- [Run-2 execution](run-2-execution.json): exact command, exit code, elapsed time,
  host topology and allowed CPU set under the unchanged per-job limits.
- [Canonical validation](validation-v2/results.json): all thirteen required
  stages, including 37 default/lockfile targets and 32 compatible sanitizer targets.
- [Resumed implementation review](INDEPENDENT_RESUMED_REVIEW.md): prepared CPU
  execution, guarded numerical policy, independent oracles and scope review.
- [Initial dynamics review](INDEPENDENT_DYNAMICS_REVIEW.md): physical oracles,
  numerical findings and their original regression checks.

The [independent run-2 artifact audit](audit-v2.json) passes, independently
recomputing measurements, refinement, differential comparisons and ranking. Its
[execution record](audit-v2-execution.json) and [log](audit-v2.log) are retained.

The original [run-1](run-1/qualification.json), [audit](audit.json) and
[validation](validation/results.json) remain unchanged historical records of the
failed initial qualification. Run-1 is not relabeled as current-source evidence.

The runs use the [bounded bridge and audit](../../../reference/emi02/README.md).
Source contracts and simulator identities are frozen in each invocation record.
Vendor model text, flattened equations and model-bearing diagnostics are excluded.
Passing CPU qualification does not authorize GPU work or establish hardware EMI
compliance. Predicted EMI infeasibility is distinct from simulator failure.
