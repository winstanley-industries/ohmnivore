# EMI-01 v2: passing and near-boundary filter references

This follow-up strengthens the external CPU reference with a passing filter and a distinct
near-boundary candidate. It preserves the [original v1 results](emi01-results.md) and raw evidence.
Production C++/CUDA, Rust/Cargo, CPU FP64/KLU authority and the proposed EMI-02 slice boundaries
remain unchanged.

## Frozen design and acceptance

The [ADR-002 follow-up](adr/ADR-002-inverter-emi-design-study.md) and
[v2 manifest](../reference/emi01/manifest-v2.json) bind the exact cases. The original `light`
candidate remains the failing control. The two new cases share 330 uH DM inductors, 1 mH CM
windings, 47 nF Y capacitors and the same enlarged hypothetical magnetic geometry. `reference`
uses a 1 uF X capacitor; `boundary` uses 80 nF. The component mass model predicts approximately
3.80842 kg and 3.80623 kg respectively. Their 2.19 g mass difference is not a mass-optimization
result; the capacitor change isolates useful acceptance-boundary behavior.

The inverter, dynamic SiC model, load/harness/chassis, source modulation, all three operating
corners, 90 dBuA research mask, 6 dB reserve and all physical screens remain unchanged. Both new
cases use 0.625/0.3125/0.15625 ns integration refinements. The original spectral floor/tolerances,
output sampling, observation interval, settling checks and resource limits remain unchanged.

The [51 exploratory attempts](evidence/emi01-v2/design/README.md) include rejected designs and
failed coarse-grid comparisons. They support design selection only. Complete retained v2
qualification and timing are recorded separately; exploratory timings are not evidence samples.

## Reproduction

```sh
bazel test //reference/emi01:signals_test //reference/emi01:metrics_test \
  //reference/emi01:adapter_test //reference/emi01:study_test //reference/emi01:report_test
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --out=/absolute/new/run-1
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --out=/absolute/new/run-2
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --audit=/absolute/new/run-1
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --audit=/absolute/new/run-2
bazel run //reference/emi01:report -- --reference-version=emi01-v2 --run=/absolute/new/run-1 \
  --run=/absolute/new/run-2 --out=/absolute/new/cpu-budget.json
```

Each invocation contains 30 qualification jobs and 72 warmup/measured jobs: one warmup plus
three measured nine-job studies at one and four workers. Each job simulates the entire coupled
network. The runner records numerical qualification separately from v2 fixture-role acceptance;
both must pass. Infeasible designs remain valid numerical results, while a failed or missing
simulation can never become a feasible candidate. Ranking explicitly orders complete candidates
by computed mass, with identity breaking exact ties.
Fixture coverage also requires the unchanged light control to be numerically valid and
predicted infeasible at all three corners; a solver failure cannot satisfy that role.

Historical v1 audits use the detached source commit documented in
[the reference guide](../reference/emi01/README.md), preserving strict source identities. Current
source hashes do not retroactively replace the original recorded hashes.

## Interpretation limits

A passing reference means predicted passing under the declared finite research model and
screens. It is not aviation compliance, laboratory correlation or a qualified hardware filter.
The enlarged geometry retains assumed CM winding capacitance/coupling, omits DM self-capacitance,
and has no winding-window/fill, gap-fringing or core-loss validation. The load remains an assumed
lumped lossy network. These limitations constrain the physical meaning of the mass and margin.

The [future GPU experiment](emi01-gpu-experiment-contract.md) uses this exact versioned workload
and retains its fresh-CPU-validation, complete-study timing and zero-failure gates. The next CPU
proposal remains [EMI-02A: disjoint linear coupled-inductor pairs](../reference/emi01/CPU_GAPS.md).
No EMI-02 or GPU implementation is included in this follow-up.
