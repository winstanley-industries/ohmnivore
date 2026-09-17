# EMI-02 independent dynamics review

The review tests in `cpp/tests/emi02c_review_test.cc` were authored separately from
production expression binding, residual/Jacobian assembly, and transient integration.
Their oracles use explicit scalar/two-state physical equations, RK4 integration,
charge primitives, and analytic RLC frequency/damping. They do not consume production
stamps, expression derivatives, residuals, or identity helpers.

## Finding: companion-matrix scaling admitted a frozen physical state

The finite-p-well fixture drives a baseline capacitor through 10 ohms and multiplies
its sensed current by a voltage-controlled factor. Its capacitor voltage differs
from the controlling external voltage. At a 500 ns maximum timestep, the original
behavioral Newton path repeatedly accepted the previous state with zero iterations:
source current stayed at 138.559 uA while independent RK4 current decayed.

`RunNewtonAttempt` normalized its initial residual with row scales containing
`C/h*x` and companion-history terms. Their large individually cancelling values
could make a real capacitor-current residual smaller than the relative tolerance.
The zero-iteration acceptance path then bypassed the update gate. Refining the
maximum timestep increases these companion scales, so timestep refinement alone
cannot repair this defect.

The final fix preserves ordinary diode/BJT behavior and the existing KLU
validation thresholds. Every behavioral point must execute at least one
simultaneous Newton correction using the stable affine RHS, then satisfy both
update and freshly recomputed residual checks. ADR-005 records this requirement.

The final independent review also rejected the intermediate exception for an
exactly zero rounded residual. In the stiff two-row companion described below,
the initial state `[128.01,128.01]` has a 10 mV forward error, but FP64 row-product
cancellation produces exactly 128 in each row and thus a computed residual of
zero. The zero-iteration shortcut would bypass both stable RHS construction and
mandatory KLU correction. It is now restricted to legacy non-behavioral systems,
and the independent starting-guess regression includes this discriminating case.

## Finding: common-mode voltage could hide a differential Newton update

`Emi02Newton.CapacitorDifferentialConvergenceIsIndependentOfCommonMode` supplies
an independent nonlinear RC equation with a capacitor voltage of only a few
microvolts. It solves the same backward-Euler point at common-mode biases of zero
and 400 V. An explicit compensating current cancels the known GMIN contribution
from the bias shift, so the physical equation is unchanged. Its reference is the
stable positive root of a scalar quadratic, independent of production assembly.

The original node-to-ground update tolerance scales with each absolute node
voltage. At 400 V it could therefore accept a millivolt-scale Newton change even
though the small voltage across the capacitor was still inaccurate. Physical LTE
then compared nonlinear-solve error as though it were integration error; shrinking
the timestep alone did not establish convergence of that differential state.

Behavioral Newton now also checks each capacitor voltage difference and inductor
current against one percent of that coordinate's LTE tolerance. This is an
additional acceptance requirement: the original node/current update, nonlinear
residual, KLU and external waveform gates remain in force. The independent test
requires agreement with its quadratic root and between both common-mode biases
within 100 nV. Direct point APIs also range-check reactive indices before reading
state. These checks do not claim that a small Newton update alone proves forward
accuracy in every ill-conditioned system.

## Finding: forming the absolute-state RHS reintroduced cancellation

The absolute-state Newton equation was initially evaluated as `J*x-F(x)`.
Although algebraically correct, subtracting an already rounded full residual from
the matrix product can lose the small uncancelled source term when a companion
coefficient `C/h` is large. The resulting linear RHS can depend on an arbitrary
Newton starting guess even for a circuit with an exact linear solution.

`Emi02Newton.LinearCompanionSolutionDoesNotDependOnNewtonStartingGuess` isolates
this failure using two unit shunts, equal 128 A sources, a capacitor between the
nodes, and a zero behavioral current source. Its retained FP64 companion has
diagonals `2^40+1` and off-diagonals `-2^40`. Adding and subtracting the two
equations proves that its solution is exactly `[128,128]`, regardless of the
capacitive coupling. Multiple widely separated initial guesses must converge to
that answer within 100 nV and return identical final states.

The behavioral assembly now forms the equivalent affine RHS directly as
`b + d'(x)*x - d(x)`, with the source's actual row signs and continuation scale.
Linear G and extra-GMIN terms cancel algebraically before floating-point
evaluation. Nonlinear affine products and sums use extended precision, followed
by bounded FP64 conversion. Original residual evaluation and accepted-Jacobian
checks remain separate. Native diode/BJT mixtures are rejected on this path,
including direct MNA point entry, so their terms cannot be silently omitted from
the specialized affine RHS.

## Finding: a passing backward-error check did not imply forward accuracy

`Emi02Refinement.ImprovesWeakModeEvenWhenBackwardErrorAlreadyPasses` uses the
independently specified matrix with diagonals `1e8+1`, off-diagonals `-1e8`, and
RHS `[128,128]`. Summing its equations gives `x+y=256`; subtracting them gives
`(2e8+1)*(x-y)=0`. Thus `x=y=128` exactly. Its common mode is much less strongly
constrained than its differential mode, so a small error along that common mode
can be almost invisible relative to the large individual row products.

The ordinary KLU result passes both unchanged backward-error guards while its
forward error exceeds `1e-7`. A zero-budget refined call reproduces that ordinary
result. With one enabled correction, the test requires both states within `5e-9`
of the exact answer, exactly one correction solve, and both original residual
guards still passing. This discriminates refinement triggered only by a failed
backward-error check from the required correction of an already accepted solve.

The opt-in API now computes a residual against the original matrix and RHS using
extended-precision products and accumulation, and performs at least one nonzero
correction when its budget permits. An exactly zero correction RHS requires no
triangular solve. Further corrections are reserved for validation failures, with
at most four across an API invocation and any fresh-factor fallback. Ordinary
real solves, zero-budget calls and complex solves retain their original behavior.
Refinement is bounded accuracy recovery, not a universal forward-error guarantee;
the independent physical and complete-waveform gates remain necessary.

## Finding: algebraic current jumps could exhaust LTE retries

The periodic voltage-driven capacitor fixture uses continuous, finite PWL source
values. The slope changes after an initial plateau, so capacitor/sensing-source
current changes discontinuously even though its stored voltage is continuous.
The initial implementation included every algebraic branch current in the local
error norm. BE/TRAP differences in that non-state current persisted as the step
shrunk, exhausting `h_min` immediately after the 10 us source corner.

The behavioral error norm now uses the physical reactive coordinates: all native
capacitor voltage differences (including the model's internal sensing capacitors)
and all native inductor currents. Dependent source currents and control outputs
remain simultaneous nonlinear unknowns with residual/update validation, but they
are not independent integration states. Existing legacy LTE behavior is preserved.

A behavioral step immediately after a source hard point also restarts with BE.
For an ideal voltage-driven capacitor, otherwise TRAP gives
`i_next=2*C*slope-i_previous`: after a slope change it can alternate spurious
algebraic currents while the imposed capacitor voltage has zero local error.
One BE step initializes the right-side derivative current without changing the
physical capacitor voltage. The periodic test checks both current and charge.

ADR-005 records these bounded behavioral changes. No waveform, physical equation,
Newton/KLU validation threshold, or reference comparison tolerance is relaxed.

## Finding: rounding-sized terminal intervals made C/h ill-conditioned

After otherwise valid 50 ns steps, the p-well fixture reached
`0.00029999999999997455 s` with a requested stop of exactly `0.0003 s`.
Integrating the remaining roughly `2.55e-17 s` created an enormous companion
coefficient and exhausted Newton retries. The source and reactive state were
already well behaved; the artificial terminal interval came from accumulated
floating-point time additions.

For the behavioral path, when a proposed step would leave less than the adaptive
minimum before a hard point, the scheduler splits the remaining distance into
two intervals. This preserves the requested maximum step and exact terminal time.
The regression checks every recorded step against 50 ns and requires an exact
300 us final sample. Existing legacy scheduling remains unchanged.

## Finding: incomplete reactive metadata could bypass physical LTE

The public MNA structure permits callers to mutate its metadata. The first behavioral
validator checked individual capacitor terminals and inductor branch ranges, but
accepted deleting either complete constraint list while leaving C unchanged. Since the
new error norm uses those lists, this could remove genuine integration states from LTE.

Capacitor constraints now retain their capacitance. Validation independently reconstructs
their node stamps in declaration order and requires exact equality with the node block
of C. Inductor metadata must bind unique names and branches to negative C diagonals, and
all branch C coordinates must belong to those inductors. Mixed node/branch coordinates,
omitted, duplicated, misbound, or altered metadata fail before any output observation.
Nonfinite matrix and capacitance values retain their typed nonfinite failure. The added
hostile test changes one property at a time, including parallel capacitors at different
scales and a coupled pair, and checks both validation and the transient entry point.

A separate direct-IR check found that a branch's `V` name prefix alone previously
admitted it as an expression current sensor. The binding pass now collects identities
only from actual independent voltage-source variants before adding behavioral
placeholders. The regression rejects a falsely V-named inductor and behavioral voltage
source, and accepts the corresponding native voltage source. This review finding is
resolved in the implementation; its final validation is included in the canonical run.

## Process and evidence boundary review

The process review found that using the same final path for raw data and metadata could
overwrite the raw waveform with JSON and report success. Canonical path aliases and
cross-aliases between final and partial files now fail before publication. Every existing
final or partial path, including a dangling symlink, is preserved. Cleanup is limited to
files created by the running process; a filesystem exception during preflight cannot
delete another run's partial output.

Eight independent process tests exercise the actual Bazel-built runner: successful DC
header, selected-observable ordering and values; transient exact final time, maximum
step, RC voltage and statistics; canonical aliases; preservation of existing files and
dangling symlinks; preflight exceptions; singular solves; and late metadata write failure.
Failed solves and publication attempts must leave neither final nor temporary artifacts.

Two additional audit findings were handed to the qualification owner: completed CPU
statistics must satisfy `attempts-rejected_steps=points-1`, and external simulator logs
must not publish diagnostics that could quote model text. These are evidence validity
and publication checks, independent of whether the measured waveform passes.

## Independent coverage

- A continuous periodic PWL cycle checks the affine differential-capacitance
  primitive, signed source/sensor-current conservation, and return charge.
- A finite-p-well network checks the independently evolving auxiliary capacitor
  voltage and sensed current, including the late relaxation that exposed the
  zero-iteration defect.
- A package RLC network checks voltage/current trajectories, ringing frequency,
  and exponential damping against an independent physical state model.
- A nonlinear-capacitor current ramp uses an independently inverted cubic charge
  primitive on a refined ordinary-policy trajectory. A separate reduced-Newton
  run checks every accepted BE/two-half/TRAP state against independently solved
  scalar constitutive equations; rejected attempts never advance that oracle. Accepted-state observers must expose
  exactly the accepted prefix at Newton/LTE failure boundaries; repeated complete
  runs must reproduce state and time arrays exactly.

No accepted-snapshot resume API is added or claimed. Restart checks replay the
complete job after an injected failure. Complete SiC/DPT/inverter qualification
remains a separate gate.

## Review: behavioral TRAP step-doubling estimator

The second-order estimator was reviewed separately after its introduction. One full
TRAP step has leading local error proportional to `h^3`; two half TRAP steps have
one quarter of that leading error. Because the full step is accepted, `4/3` times
their difference estimates its local error, and the cubic-root timestep controller
matches that order. The first private half step receives accepted state and the
accepted/midpoint source values. The second receives the first half's private state
and midpoint/end source values. Each solve recomputes its own behavioral history;
neither half result replaces the accepted full-step result or enters the observer.

The new independent RC test uses a quadratic behavioral drive and a separate
behavioral conductance. Explicit scalar constitutive equations recompute every
attempt's full and half states, normalized error, acceptance and controller choice.
The quadratic source distinguishes correct midpoint forcing, while the conductance
distinguishes the previous behavioral residual. The test checks published full TRAP
states, the exact continuous RC response, cubic local-error scaling and the `4/3`
coefficient against an exact exponential. It observes controller choices that
distinguish the cube root from the prior square root. Existing charge, p-well,
package RLC, and forced Newton/LTE rejection isolation tests also still pass.

## Focused validation

`bazel test -c opt //cpp:emi02a_test //cpp:emi02c_review_test` passed after the
physical dynamics fixes. The subsequent metadata and process review passed
`bazel test -c opt //cpp:emi02c_review_test //cpp:emi02c_test //cpp:phase3b_test
//reference/emi02:runner_process_test`: five review tests and eight process tests,
plus the behavioral and legacy nonlinear transient regression targets.
The subsequent TRAP estimator review passed `bazel test -c opt
//cpp:emi02c_review_test //cpp:emi02c_test`: all six independent review tests and the
behavioral base target. The dynamics review target completed in 6.2 seconds in this
environment. This is focused
CPU correctness evidence, not a performance claim or complete EMI-02D qualification.

The final physical comparisons keep `1e-6 A + 1e-3*abs(reference)` and
`1e-5 V + 1e-3*abs(reference)` trajectory bounds; the charge loop has a 3 nC
absolute integral bound. P-well maximum step is 50 ns. Package RLC maximum step
is 2 ns over 900 us, resolving three positive ringing peaks. A 20 ns RLC maximum
remained controller-limited during the 1 us source ramp and missed the near-zero
current bound; the final refinement changes no circuit value or error threshold.
The ordinary nonlinear-charge trajectory uses a 100 ns maximum step. The separate
5 us, two-Newton-iteration fault run is compared to its independently recomputed
BE/two-half/TRAP equations rather than being labeled a converged physical trace.

Complete canonical, sanitizer, hermeticity, and full-model/reference evidence are
reported separately by the parent delivery. Earlier failing runs motivated the
fixes and do not count as passing qualification evidence.

## Qualification boundary at this review

The integration owner reports the focused numerical targets passing after the
common-mode update, stable affine RHS and mandatory residual-correction changes.
This narrative reviews their independent fixtures and implementation; it does not
replace the final canonical run or a fresh complete evidence audit.

The latest full-circuit diagnostic still failed the resource bound: it reached
roughly 40 us of a required 200 us waveform in about 110 seconds. That partial
trajectory establishes neither full-waveform accuracy nor resource qualification.
It cannot enter a successful-job denominator or justify an EMI feasibility,
throughput, or GPU claim. The subsequent [delivery report](../../emi02-results.md)
records the passing canonical checks and complete set of 30 CPU plus 30 external
reference jobs. All three CPU DPT cases pass their differential comparisons;
all 27 CPU inverter jobs reach the resource limit. The frozen circuit, accuracy
thresholds, resource gates and job set remain unchanged.
