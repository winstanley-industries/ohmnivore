# Proposed transient-ensemble GPU experiment contract

This is a **future experiment proposal**, frozen by EMI-01 before any GPU implementation.
It authorizes no implementation or dispatch. First complete the separately approved CPU slices
in [the capability-gap report](../reference/emi01/CPU_GAPS.md), preserving CPU FP64/KLU authority.
The [EMI-01 v2 results](emi01-v2-results.md) supply the current reference budget and case coverage;
[v1 results](emi01-results.md) remain historical evidence for the original candidates.
Historical linear-AC thresholds do not apply.

## Exact workload and electrical boundary

Use the exact [EMI-01 v2 manifest](../reference/emi01/manifest-v2.json), fetched/adapted model identities, corrected fixtures, complete
coupled network, initial state, observation band, detector, physical constraints and output
requirements in [ADR-002](adr/ADR-002-inverter-emi-design-study.md). One scheduled job is one
whole candidate/corner circuit. Do not split bridge legs, windings, load or chassis into solves.

Qualify all three DPT refinements and all nine candidate/corner refinements before measuring.
The performance domain is exactly two ordered ensemble sizes: the nine unique jobs once, and
four explicitly identified replay replicas of those same nine jobs (36 jobs, replica-major
order). Replicas test scheduling saturation; they are not additional physical designs or corners.
Input identity includes replica, candidate, corner, model, topology and numerical policy.
No random candidates, parameter search, mixed precision or distributed execution is included.
Preserve the failing control, stable near-boundary rejection and all-corner passing reference.
The original 6 dB reserve remains fixed; a candidate name cannot determine its classification.

The CPU comparator must be the resulting qualified Ohmnivore CPU implementation with persistent
workers, one-thread numerical libraries and private nonlinear/integration state per job. Measure
worker counts 1, 4 and 16 on the recorded 16-core reference host; compare against the fastest
qualified CPU mode separately for each ensemble size. ngspice is the external accuracy oracle,
not a claim about future KLU performance. Re-profile that CPU implementation before implementation
authorization if its breakdown invalidates the opportunity described by EMI-01.

## Accuracy, failures and required output

Native FP64 only. Generate fresh CPU reference trajectories for all nine exact inputs once per
complete invocation. Bind them to physical/numerical inputs, CPU implementation, model and source
identities. Each timed GPU result remains untrusted until fresh per-result CPU validation against
those immutable trajectories accepts the complete job. Every replica still independently solves
its complete initial-value problem; a stored trajectory cannot substitute for GPU execution.
Preserve nonlinear residual checks, accepted-step state, rejected-step rollback,
source breakpoints, initialization and typed failure meaning. Symbolic pattern reuse is allowed
only under exact pattern identity; no numerical state may leak between candidates or replicas.
CPU comparison must include raw waveform validation, every CM/DM/conductor spectral bin, switching
metrics, stress/loss constraints, and the final feasibility decision. Use ADR-002's original
absolute/relative waveform, 1 dB above 20 dBuA / 1 uA below-floor spectral, energy, edge and ringing
gates, plus its refinement checks. A changed integrator must independently demonstrate those
gates; equal internal step sequences are not required.

Reject non-finite values, missing/truncated output, crossed job identities, invalid timing grids,
unsupported models and resource exhaustion. Fault injection must independently expose each
failure class and prevent publication or feasible ranking. A fallback CPU run may produce a
valid result but cannot count as successful GPU throughput; retain its reason, cost and identity.
The experiment performance gate requires zero GPU fallbacks and zero failed or missing jobs.

Retain the same required raw information, spectra, provenance and terminal completion records as
the CPU comparator. Lossless storage encoding may differ only with independently verified exact
reconstruction. No performance claim may omit output or substitute kernel-only timing.

## Resource and timing gates

Reference accelerator: the recorded RTX 5080 class host, with approximately 16 GiB device memory;
record the actual driver, device, clocks/power policy and pinned toolchain at execution. Proposed
budgets: at most 4 GiB peak device allocation and 16 GiB aggregate host memory, with at most 16
simultaneous job states. Stream completed output within those budgets. Retain the per-job 120 s
wall / 110 s CPU fallback / 1 GiB oracle address-space / 512 MiB raw-file / two-million-point
limits and zero retry policy. Fail closed if a budget cannot be met.

Before collecting evidence, freeze one complete warmup plus five measured complete studies for
every CPU/GPU mode and ensemble size, in two independent invocations. Warmups also have complete
accounting. Record empirical nearest-rank P95 over the finite observations; do not claim a
population-tail estimate. No competing benchmark/build may run. Keep all candidates, corners,
replicas and failure records in throughput denominators.

Charge preparation, queueing/scheduling, context/model setup where cold, host/device transfers,
device execution and synchronization, per-result CPU validation, spectral processing,
and closed required output files. Report one-time setup separately and cold complete-study time
including it and fresh CPU qualification. Generate these CPU references independently in both
evidence invocations; report their measured cost separately and include it in whole-invocation
wall time. A warm measurement may reuse explicitly declared context/pattern/model/reference state;
both CPU and GPU modes receive the same permitted reuse. Physical fsync and dependency build/fetch
are separately excluded. Report complete invocation wall time as well as study and job latency.

The proposed usefulness threshold is **at least 2.0x median and 1.5x empirical P95 complete-study
speedup**, versus the best qualified persistent CPU mode, for both 9 and 36 jobs in each independent
invocation. Cold complete-study time, including fresh CPU qualification, setup and per-result
validation, must be no slower than the best cold CPU study. Memory, accuracy and zero-failure
gates are conjunctive. These thresholds
are a new engineering acceptance budget: a substantial reduction in design-study turnaround must
survive scheduling, certification and output costs. They are not inherited from AC or predicted
by a factorization microbenchmark. Failure is valid negative evidence and leaves dispatch closed.

Automatic dispatch, mixed precision, new physical models, general optimization and larger ensemble
claims require separate authorization even if this proposed experiment passes.
The warm 2.0x/1.5x gates apply only to repeated evaluations of this exact finite input set after
charged qualification. They do not establish throughput for previously unseen candidates.
Report cold and whole-invocation speedup separately, including qualification once in each timed
boundary rather than adding it twice. The cold CPU comparator includes its own same required
qualification and study output; identify common qualification cost explicitly. For novel inputs,
generating a full CPU oracle trajectory can remove the useful acceleration budget. No cache entry
is valid for changed physical/numerical inputs, and no broader certification or novel-candidate
throughput claim is authorized without a separate bounded contract.

After the CPU references exist, the three candidates' feasibility over all nine candidate/corner
inputs is already known. Faster replay
therefore does not demonstrate faster discovery or time to a new lightest feasible design. The
EMI-01 simulator-phase counterfactual excludes additional GPU certification/transfer costs and
is only an optimistic opportunity bound; actual experiment timings must charge those costs.
