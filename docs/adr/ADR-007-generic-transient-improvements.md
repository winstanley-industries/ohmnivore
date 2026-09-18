# ADR-007: Generic transient accuracy and work regression gates

Status: authorized investigation and implementation, 2026-09-18.

The EMI-02 timestep investigation identifies local error control on small reactive
states as the main source of extra accepted points. A change must be justified by
general numerical behavior, not a circuit name, vendor model, frequency band,
candidate identity or desired benchmark result. No device may be removed from
the error norm, and no physical tolerance or frozen external gate may be relaxed.

Before changing the qualified runner, compare the existing policy with an explicit
candidate policy on a shared independent analytic corpus. Cover stable RC dynamics,
underdamped and stiff RLC dynamics, zero crossings, amplitude and time rescaling,
large common-mode bias with small differential signals, nonlinear dynamics,
positive feedback, source corners and rejected-state isolation. Retain per-case
accuracy, accepted points, rejected attempts, linear solves and complete execution
wall times. Aggregate gains cannot hide a failing case. Existing default library,
native transient, AC, device and failure regressions remain mandatory.

The first candidate scales behavioral TRAP local error using both accepted start
and provisional end physical-coordinate magnitudes, instead of end estimates
alone. It keeps the same absolute/relative constants, full/two-half comparison,
history numerator, source restart, positivity audits, fallback, Newton/KLU checks
and controller. The accepted start prevents an instantaneous zero crossing from
collapsing the relative error scale. An explicit policy permits comparison against
the existing implementation and leaves established public defaults unchanged.
BE startup/recovery keeps its existing norm. Rejection cannot update any scale
history; there is no running peak, circuit-dependent scale or global signal bias.

Retain the candidate only if it passes the independent analytic limits and all
prior correctness tests, has useful complete-workload improvement, and shows no
material accuracy/work regression elsewhere. Every performance regression must be
reported rather than averaged away. Before selecting it in the model runner,
execute the full unchanged CPU/reference qualification and artifact audit with new
fingerprints. Existing evidence remains historical. A rejected candidate is a
negative result, not a reason to weaken these gates.

Audit-trigger changes, if investigated later, need a separate specified invariant
and independent adversarial fixtures. Merely observing frequent audits or a low
failure rate is insufficient justification to omit them. GPU work is excluded.

## Endpoint-scale experiment: rejected

The 21-case corpus exposed increased accumulated error in lightly damped RLC
circuits, for only modest step-count savings. Do not select or expose this
candidate in the shipped solver. Retain its patch and measurements as negative
development evidence. Coarse stress cases also expose existing global-error
limitations in both baseline estimators; local tolerance is not a global error
bound. Keep those failures visible, and require all 21 cases to pass the original
analytic error limits with a separately declared 32-times-smaller maximum step.
The finer lane changes input resolution, not tolerances or reference truth.

## Exact companion reuse

The prepared companion owns immutable, fully admitted G/C matrices. G-only union
entries are already finite and initialized by the checked factory and cannot
change with timestep. Prepare update entries only for positions present in C,
retaining original CSR order and the distinct C-only versus G+C arithmetic.
Every changing value keeps its original finite check. Do not revalidate or rewrite
constant G-only values on subsequent forms.

C-bearing entries remain in canonical C order, so the loop index directly names
its immutable C value. Store only the union value index and optional G index,
without a duplicate C index or wider per-entry metadata. Keep step/alpha and
scaling checks even for an empty C matrix. A partial numeric failure is repaired
by recomputing every C-bearing value on the next successful call. Independent
checked construction must verify bit-exact values, signed zeros, cancellation,
input failure precedence and recovery from partial failure.

An earlier variant also cached an identical scaling factor. The 42 validated
corpus lanes measured a 2.6% geometric-mean slowdown, so it was rejected. The
retained candidate has no cache lookup, per-call counters, extra matrix copies,
or expanded owner state; it only avoids work on immutable entries.

This optimization changes no physical equation, error estimate, timestep,
acceptance policy or public default. Require bit-exact states, timestep traces
and solver counters on the generic corpus and complete EMI workloads. Retain
per-case timing rather than interpreting fewer matrix updates as speedup.
