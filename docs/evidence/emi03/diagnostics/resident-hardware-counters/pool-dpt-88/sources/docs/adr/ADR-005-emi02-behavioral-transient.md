# ADR-005: EMI-02C behavioral sources and transactional CPU transient state

- **Status:** Implementation contract under the authorized EMI-02 request
- **Date:** 2026-09-17
- **Prerequisites:** ADR-003 coupled elements, ADR-004 expressions, ADR-001 KLU/Newton

This slice executes the bounded stateless expression graph with the existing native RLC
states. It preserves the selected model's simultaneous capacitor-current multiplier
network; it does not replace differential capacitance by `Q=C(V)*V`, omit the baseline
capacitor, freeze a sensed current, or eliminate the finite p-well state. Expressions are
evaluated at the current Newton trial, including all source-current cross derivatives.

## Explicit experimental boundary

`ParseBehavioralNetlist` and `CompileBehavioralMna` are explicit library entry points, used
by `//cpp:emi02_runner`. Ordinary `ParseNetlist` remains closed to E/G/B/model imports.
The bridge admits `Ename p n VALUE={expression}`, `Gname p n VALUE={expression}`, and
`Bname p n I={expression}` only. Native RLC, independent sources and K pairs use their
existing parsers. Expressions are exactly the ADR-004 vocabulary. Native semiconductor
mixtures and behavioral AC/prepared AC are unsupported. Each input has at most 512 MNA
unknowns and 16,384 expression nodes; individual expressions retain ADR-004 budgets.

Each behavioral voltage source owns an ordinary voltage-source branch and incidence
stamp. It adds `-expression(x)` to that branch's KVL residual. A behavioral current source
adds `+expression(x)` at its positive node and the opposite at its negative node. Its
Jacobian is the expression derivative multiplied by the same row sign. Native placeholder
sources have zero independent RHS. All possible derivatives, including unselected lazy
branches, enter the immutable G union before KLU analysis. Descriptor identities, row
indices, signs and derivative coordinates are validated. No unknown name is inferred.

The nonlinear residual is `G*x+C*dx/dt-b+d(x)`. Existing BE/TRAP equations include the full
previous `d(x)` contribution on TRAP steps. Existing KLU backward-error and accepted-point
Jacobian factor/zero-solve checks remain mandatory. Final residual is freshly recomputed.

## Numerical policy

The explicitly behavioral graph uses a separate fixed policy: Newton voltage absolute
tolerance `1e-7 V`, current absolute tolerance `1e-9 A`, relative tolerance `1e-5`;
LTE voltage absolute `1e-7 V`, current absolute `1e-9 A`, relative `1e-4`.
Behavioral LTE compares every physical capacitor voltage difference and every inductor
branch current, including all auxiliary model capacitors. Algebraic voltage-source
currents and expression-output nodes are not independent integration states. Including
their discontinuous derivative responses at a continuous PWL corner would force
unbounded refinement. Native non-behavioral circuits retain the established all-variable
LTE policy. The frozen complete-waveform/spectrum gates still validate every observable.
Before integration, reactive metadata must account for the complete C matrix. Capacitor
constraints retain their positive finite capacitance and unique identity; reconstructing
their node stamps in declaration order must equal the node block exactly. Inductor
constraints have unique branch indices and identities matching the branch names, and
each has a negative C diagonal. Every branch coordinate in C belongs to those inductors;
node/branch mixed C coordinates are invalid. Missing, duplicated, or misbound reactive
metadata is `kInvalidStructure`, while nonfinite metadata or matrix values are
`kNonFinite`. Thus malformed metadata cannot silently remove a physical state from LTE.
Behavioral integration uses BE for the first step after each source hard point as well
as the landing step. This restarts algebraic displacement current after a source-slope
change and prevents the undamped TRAP derivative-current alternating mode. Reactive
state remains continuous and the source timestamps remain unchanged.
All expression/Newton intermediates remain finite and bounded by `1e100`.
Direct/DC continuation may take up to 300 iterations per point; transient and projection
points up to 100. Existing source scales `0,0.1,...,1` and extra-GMIN sequence remain
unchanged. Optional reduced iteration/attempt bounds support fault injection.

Behavioral direct attempts use simultaneous full Newton steps when finite and bounded.
This allows an auxiliary rational equation's trial states to cross a pole; requiring every
iteration to decrease its residual can trap a complete coupled solve on the wrong side.
For a nonfinite/over-bound direct trial, try scales `1/2,...,1/65536` until finite. Source
and GMIN continuation attempts instead require a residual merit decrease or an already
converged trial at the same 17 step lengths. Failed trials are private, and no expression
value is clipped. Exhaustion is `kNonConvergence`. Every returned solution still passes
both original update/residual criteria and original-matrix KLU guards.
Continuation line-search merit uses row scales frozen at the current Newton iterate:
divide each trial residual by the current row's existing absolute-plus-relative residual
tolerance, then take the maximum. Trial-dependent scales must not reward a larger
provisional state, and mixed-unit raw residuals must not let one high-scale internal
voltage equation stall unrelated equations. Separate, unchanged scaled residual and
update checks still govern final acceptance.
Legacy diode/BJT Newton and their default limits remain unchanged.

The behavioral Newton linear solve uses the algebraically equivalent absolute-state
form `J*x_trial = b + d'(x)*x - d(x)`, then computes `delta=x_trial-x`. The nonlinear
affine terms are accumulated in extended precision before conversion to FP64. Linear
G and extra-GMIN terms cancel algebraically; do not subtract a rounded full residual
from `J*x`, which can inject companion-matrix cancellation noise into the RHS.
Native diode/BJT mixtures are rejected even through direct MNA point APIs. This avoids artificial near-zero
delta constraint rows for ideal voltage sensors while retaining both original KLU
backward-error gates unchanged. The original nonlinear residual still determines
acceptance. Legacy devices keep their original correction-form solve.
Behavioral points must take at least one stable-RHS Newton correction, even when the
initial residual evaluates to zero in FP64. Cancellation in a stiff companion row can
round a nonzero exact residual to zero, so that observation cannot prove a root or bypass
the correction. A merely tolerance-small residual likewise cannot freeze the preceding
physical state: its `C/h` history scale grows as the step shrinks. Legacy non-behavioral
iteration-zero acceptance retains its existing residual and Jacobian checks.
Behavioral Newton additionally checks updates of every capacitor voltage difference
and inductor current against one percent of that coordinate's LTE tolerance. A
node-to-ground relative update check alone can accept millivolt changes on a 400 V
common-mode node while LTE needs sub-microvolt accuracy in a small differential
capacitor voltage. This stricter physical-coordinate update guard separates nonlinear
solve error from integration error; it changes no residual, KLU or external accuracy
gate. Direct point APIs range-check all reactive indices before reading state, and
transient admission separately requires the complete C/metadata correspondence.

Time integration retains BE/TRAP selection, hard-point landing, half-step LTE
isolation, halving on Newton nonconvergence, and bounded adaptive minimum. Behavioral
TRAP estimates its second-order local error by comparing one full TRAP step with two
private half TRAP steps from the same accepted state. The accepted state is the full
step; its error estimate is `4/3 * abs(full - two_half)`, since the leading local
errors scale as `h^3` and `h^3/4`. Each half step evaluates its own source and nonlinear
history, and all three solves must satisfy the same Newton/KLU gates. Its controller
uses `0.9 * error^(-1/3)`, clamped to `[0.5,2]`. BE keeps its first-order step-doubling
estimate and square-root controller. The non-behavioral TRAP-versus-BE estimator remains
unchanged. This avoids imposing a first-order error estimate on the second-order
behavioral TRAP solution; the independent external refinement gates remain mandatory.
The experimental
runner admits at most two million output points and four million attempts, no UIC, and
requires PWL/PULSE sources with explicitly matched DC/waveform initial source values.
Native library UIC and
discontinuous behavioral source projection remain unsupported until separately qualified;
finite reference edges are retained exactly. There is no silent source approximation.

If BE/TRAP cannot meet the frozen external refinement gates, record the failure; a new
integration method requires a further contract before implementation. Reference accuracy
thresholds and study inputs cannot be relaxed to make the port qualify.

## State, output, and failure atomicity

Full steps and both LTE half steps start from private copies of accepted state. Rejection
cannot commit capacitor voltage, inductor current, sensed source current, trial residual,
or another job's state. Factorization caches contain structural/numeric work only.

An optional accepted-state observer enables bounded streaming; default library output
retention is unchanged. Observer invocation occurs only for accepted states, in increasing
time order, after physical acceptance. Observer failure aborts the run. The experimental
runner writes to temporary output and publishes only after successful completion, closed
files, count/schema/finiteness checks, and terminal metadata. Failed/truncated output is
never a completed job. Maximum required raw output is 512 MiB. Files contain named public
observables and result data, never copied vendor equations.

Malformed input, unknown construct/name, expression domain/non-finite failure, dimension
or descriptor failure, KLU singularity/factorization/validation, Newton exhaustion, timestep
exhaustion and output failures retain distinct existing ErrorCode meanings. A failed
numerical solve is never an EMI pass. The external harness records every required job.

## Acceptance

Independently authored tests cover source polarity, complete residual/Jacobian entries,
sensed-current derivatives, affine nonlinear-capacitance positive/negative ramps and
charge integrals, periodic charge loops, finite-resistance auxiliary-state dependence,
package resonance, deterministic repeated solves and reduced-budget failures. Analytic
tests must discriminate omission of baseline current, current cross derivative, TRAP
history, and rejected-state isolation. All prior default behavior remains a regression
gate. Complete SiC model/DPT/inverter qualification belongs to ADR-006.

## Opt-in FP64 KLU refinement contract

The behavioral solver may call the separately named `FactorAndSolveRefined` real-solver
entry point. Existing `FactorAndSolve` and complex/AC paths retain their existing behavior.
When a correction budget is enabled, compute at least one nonzero residual correction
even if the initial backward-error guards pass. Those guards alone do not bound forward
error in small differential states of a mixed-scale companion system. An exactly zero
correction RHS needs no solve. Further corrections occur only on `kSolutionValidation`;
malformed structure, nonfinite inputs, singularity and factorization failures preserve
their existing typed failures.

There are at most four correction solves per API invocation, including any existing
refactor-to-fresh-factor fallback. A caller may request a smaller bound, including zero,
for discrimination and failure tests; a bound above four is `kUnsupportedSize`. Compute
the correction RHS `b-A*x` against the original unscaled CSR and RHS using long-double
products/accumulation as in existing residual validation, convert to finite FP64, and solve
the correction with the same existing FP64 KLU factors. Update the provisional FP64
solution and validate the original matrix/RHS after every correction. No correction RHS
or provisional solution is published. Nonfinite residual conversion/update fails
`kNonFinite`; exhausted correction budget preserves `kSolutionValidation`.

Both original guards remain mandatory without tolerance changes: row-equilibrated
normwise backward error <=1e-10 and maximum row componentwise backward error <=1e-5.
The new `iterative_refinement_solves` statistic counts correction triangular solves;
the existing `solves` count remains successful external solves. Refinement adds neither
a factorization backend nor a GPU path and never weakens validation of accepted results.
Independent constructed matrices must demonstrate rejection at zero refinement, recovery
under the bounded path, exact original residual validation and unchanged ordinary behavior.

Before a behavioral source hard point, if the proposed timestep would leave a positive
remainder smaller than the existing adaptive minimum, split the remaining distance in half
instead. This never increases the declared maximum timestep and avoids a rounding-sized
final interval with an artificial `C/h` conditioning spike. The solver still lands on the
exact hard point and retains all accepted-state and error-control checks.

The experimental EMI-02 runner sets the adaptive minimum to the declared maximum timestep
divided by 1,000,000. This permits bounded resolution of the initial finite gate edge's
small package-inductor current, whose physical step-doubling error remains second order.
The native default divisor, all LTE tolerances, accepted-step/attempt budgets, and exact
source timestamps remain unchanged.

Behavioral Newton may reuse a complete assembly only within one private attempt while
its system, source scale, GMIN scale, active rows and exact FP64 state are unchanged.
The initial assembly can serve the first iteration; an accepted line-search trial can
serve that iteration's convergence check and the next iteration. Every distinct trial
still evaluates all active expressions, analytic derivatives and intermediate bounds.
The returned point retains its separate freshly recomputed original-system residual
and accepted-Jacobian KLU factor/zero-solve check. Rejected trial assemblies are discarded. Reuse never crosses a point, source
change or integration step; legacy device paths retain their existing assembly calls. This removes duplicate pure evaluation without changing arithmetic,
iteration traces, tolerances, state ownership or acceptance policy.

## Prepared CPU execution within the unchanged qualification limits

The 110 s qualification CPU limit bounds an individual job, not implementation
work. A failed retained study does not complete EMI-02D. CPU overhead may be
removed while retaining every numerical method, arithmetic order, acceptance
threshold, finite/magnitude guard, retry limit and frozen qualification case.

A behavioral transient invocation may own an immutable snapshot of the admitted
MNA metadata and reuse one private companion workspace with its analyzed KLU
pattern. The factory validates all topology, descriptor coordinates and reactive
metadata; changing companion values, RHS and states retain their existing runtime
checks. No caller-controlled validation bypass is exposed. Public point, residual
and matrix APIs remain fully validating. Snapshot ownership prevents observer
aliases from changing the currently admitted circuit. Separate invocations and
failed attempts cannot share mutable workspaces or provisional physical state.
Companion construction preserves the G-only, C-only and combined-entry arithmetic,
including signed zeros and exact cancellation; the original DC initialization is
unchanged. Final original-system residual and accepted-Jacobian checks remain.

An analyzed sparse factorization may cache canonical CSR structure and reusable
numeric workspaces. Exact structure equality permits reuse; mismatches retain the
existing converter and typed-error precedence. Already validated structure need
not be converted again during original-matrix residual checks. Numeric factor,
refactor/fresh fallback, triangular solves, original residual arithmetic, both
backward-error guards, bounded refinement and statistics remain unchanged.
The same extended-precision row product may supply validation and refinement
residuals; its accumulation order and the FP64 correction rounding remain exact.
Within one immutable matrix/RHS solve, refinement validations may reuse the first
validation's exact row magnitude sums, row scales and normwise matrix/RHS terms.
Every new matrix/RHS call invalidates this numeric metadata; solution-dependent
products, componentwise denominators and both final guards are always recomputed.
After all structural and finite checks, an all-zero RHS and all-zero solution
have exactly zero backward error and correction. A private validator may return
that result directly; the numeric factorization and triangular solve that establish
the solution, including singular-Jacobian rejection, are still mandatory.

Expression execution may prebind gradient slots and avoid unnecessary derivative
allocation, or use caller-owned scratch storage. Value traversal, reverse-AD order,
lazy branches, inactive-dependency validation and all intermediate bounds remain
unchanged. A history-only caller must still perform derivative validity checks.
An identical validated history at the same immutable state may be reused within
one attempt; half-step history at a different state is always recomputed.

Acceptance requires bit-exact prepared-versus-checked point/companion comparisons,
typed-failure and isolation regressions, independent physical oracles, canonical
checks, and a fresh complete qualification invocation. Prior failed evidence is
historical and cannot be selectively repaired or relabeled as passing.

## Behavioral LTE retry method

A converged behavioral trial rejected only for excessive local error retries the
same integration method at the controller's reduced step. Switching a rejected
TRAP trial to first-order BE discards the error-order assumption used to choose
that step and can trigger repeated avoidable halvings. Keep BE when the rejected
trial was BE, including a shortened retry before a waveform corner. Startup,
waveform landing/restart and nonlinear-convergence recovery still use BE. Native
nonbehavioral rejection policy is unchanged. The full/two-half step formulas,
error normalizations, tolerances, controller factors and transactional state
boundaries remain unchanged. Tests must distinguish TRAP-to-TRAP LTE retries,
BE-to-BE LTE retries and BE recovery after a nonlinear failure.

## Opt-in derivative-history TRAP error estimator

The EMI-02 model runner may select a new behavioral-only local-error estimator.
The default public integration policy retains full/two-half step doubling. Native
nonbehavioral runs cannot select the new estimator. This is a numerical-policy
change, not a bit-exact execution optimization; it requires independent dynamic
oracles and every frozen external waveform/refinement gate before qualification.
BE/TRAP state equations, FP64 KLU checks, Newton convergence, source hard points,
physical LTE tolerances, controller and all resource bounds remain unchanged.

For each physical capacitor voltage difference or inductor current y, retain its
accepted derivative d and, within one smooth TRAP segment, the preceding accepted
derivative/time. Seed d from the final accepted half BE step,
`d = (y_accepted - y_first_half)/h_second_half`. After a TRAP trial of duration h,
infer `d_trial = 2*(y_trial-y_previous)/h - d_previous` from the same discrete
constitutive equation. With preceding accepted gap k, form
`D2d = ((d_trial-d_previous)/h - (d_previous-d_older)/k)/(h+k)` and signed local
error `e = h^3*D2d/6`. This is the trapezoidal defect `h^3*y'''/12`, estimated from
the divided difference of derivatives; unlike a naive third divided difference of
computed states, it reproduces quadratic derivative forcing on unequal steps.

Normalize abs(e) by the existing physical absolute tolerance plus the existing
relative tolerance times `max(abs(y_trial), abs(y_trial - 3*e/4))`. The latter is
an estimate of the refined coordinate, not a computed half-step solution. The
maximum physical-coordinate error drives the existing cube-root TRAP controller.
Derivative reconstruction and divided differences use checked extended-precision
intermediates; all states, derivatives, errors and proposed steps remain finite
and bounded. Exact accepted timestamps, not requested nominal steps, define h/k.

Until two accepted derivatives are available in a smooth segment, retain the
existing full/two-half TRAP estimator. Accepted BE recovery or a waveform hard
point resets older derivative history. Every rejected trial leaves derivative,
time and physical-state history unchanged. A new invocation owns fresh history.
No unavailable or invalid history permits unchecked acceptance. BE always retains
its current step-doubling estimator and returns the final half-step derivative.

Derivative reconstruction has an undamped alternating roundoff mode, and backward
stencils can lag rapid changes. Acceptance therefore requires unequal-grid cubic
and higher-order forcing, analytic RC, independently integrated stiff RLC,
common-mode/cancellation, nonlinear branch transition, source restart, rejection
rollback and bounds tests. The complete pinned DPT/inverter/reference qualification
and independent artifact audit remain mandatory; passing short tests is insufficient.

The estimator is not a universal conservative local-error bound. Three derivative
samples can alias smooth higher-order forcing, and a growing stiff mode can amplify
quadrature defect into a larger endpoint error. Guard the optional lane explicitly:

- Audit the first eligible history attempt and every 32 accepted TRAP steps after
  an audit with the original full/two-half estimator. Its actual error controls
  acceptance and adaptation on audited attempts.
- Use full/two-half estimation when the maximum history estimate is exactly zero.
- For a physical coordinate with `abs(delta_y) > 0.01*physical_LTE_tolerance`, use
  full/two-half estimation when `h*(d_trial-d_previous)/delta_y >= 0.5`. This catches
  the independently constructed growing-RC amplification case.
- If an audit's normalized error exceeds twice the history estimate and exceeds
  0.01, enter a step-doubling fallback. A rejected audit may activate fallback
  immediately, but cannot commit physical or derivative history. Return to history
  estimation only after 16 consecutive accepted full/two-half TRAP audits with
  valid derivative history, each satisfying actual error <= twice the history
  estimate or actual error <= 0.01. Every rejected trial (Newton or LTE), or any
  audit disagreement, resets this recovery streak. Missing-history initialization
  cannot count toward recovery. Restart periodic audit debt from the final accepted
  recovery audit; zero-estimate and positive-feedback guards always remain active.
  Accepted BE/hardpoint reset begins a new segment.

Recovery re-establishes recent measured agreement after a switching transient; it
is a bounded heuristic, not proof that the next unaudited step has conservative
error. Independent tests must distinguish 15 from 16 accepted agreements, reset on
rejection/disagreement, and preserve accepted-state/history transaction boundaries.

Periodic audits reduce, but do not eliminate, possible unsampled forcing between
audit points. Qualification claims are confined to the pinned workload and retained
external comparisons. The default full/two-half policy remains available for other
behavioral circuits, and no tolerance or resource-limit change follows from opting in.

The opt-in estimator uses the same representable timestamp gap for its companion
matrix and derivative reconstruction. After choosing a non-hardpoint endpoint,
set h to `next_time - time`; if rounding made that gap exceed the maximum step,
move the endpoint one representable value toward the accepted time first.
Hardpoint identity and the default integration policy remain unchanged. This
avoids injecting an alternating derivative error from mismatched nominal and
represented time increments.

Per-step diagnostics identify whether a valid-history audit occurred, whether it
met the recovery agreement criterion, and whether fallback remains active after
the trial. Aggregate counters report entries and completed recoveries; BE/source
resets are not recoveries. These diagnostics expose the policy transitions for
independent tests without exposing private model equations or state derivatives.

Prepared behavioral invocations may also retain the immutable G/C union mapping
and its CSR storage. Prepare lazily only after the first ordinary companion
construction succeeds, at its original validation point. Preserve the distinct
G-only, C-only and overlapping-entry arithmetic, including signed zero and exact
cancellations. New h/alpha, scaled entries and RHS intermediates remain checked.
Private G*x/C*x products may omit revalidation of the owned immutable CSR metadata;
state size/finiteness and every partial-sum check remain in original order. Public
helpers, native integration and the fully checked comparison path remain unchanged.
Successful accepted-state products may be reused within one attempt at that exact
state; the second half step always computes products for its different state.

A prepared nonlinear owner may retain at most four successful expression value/AD
sets, keyed by bit-exact full state and descriptor order under its immutable
compiled expressions. Publish an entry only after a fresh final original-system
assembly and residual validation succeeds. Initial Newton assembly and nonlinear
history may reuse these checked expression results at the exact same state;
every proposed Newton/backtracking state still evaluates complete value/AD results
afresh. Every final original-system check evaluates fresh values and domains; the
immediate accepted-trial derivative proof below is the sole exception to repeated
final AD evaluation. State dimensions and bounds, including unused
and inactive dependencies, remain checked before lookup. Rebuild all matrix/RHS,
stamps, affine terms, residuals and intermediate bounds for the current companion.
No failed or partially evaluated point enters the cache. The four-entry bound
covers the accepted full state and private half-step states without unbounded
retention. Bit-exact state identity distinguishes signed zero. Private diagnostic
counts must prove cache hits and fresh final checks in tests, alongside complete
prepared-versus-checked trajectory and failure parity.

Linear-row assembly may use its nonnegative accumulated magnitude scale to prove
the existing term and product bounds at the same accumulation point. Starting
from abs(source), after each `product += term; scale += abs(term)`, round-to-nearest
monotone FP64 arithmetic preserves abs(product) <= scale and abs(term)
<= scale. A nonfinite term/product also makes scale nonfinite. Checking the scale
therefore preserves all three guards' failures and messages without changing any
arithmetic. Final residual bounds and independent affine-RHS bounds remain explicit;
the proof does not permit reassociation, skipped partial sums or fast-math flags.

The private real-solve validator may reuse a row product term's magnitude in the
componentwise denominator: under the pinned round-to-nearest arithmetic,
`abs(coefficient * variable)` is bit-identical to
`abs(coefficient) * abs(variable)`, including signed zero and subnormals. Preserve
the denominator's addition order and all checks. Public validation retains the
original independently evaluated expression as the comparison oracle. Adversarial
normal/subnormal/extreme cases and bit-identical correction/solve results must
cover this specialization before qualification.

For a fixed finite positive magnitude bound, the ordered predicate
`abs(value) <= bound` alone is equivalent to `isfinite(value) && abs(value) <=
bound`: it rejects infinities and all NaNs. Bounded-value helpers may omit the
redundant finite predicate, retaining every call site and its existing failure.
Rejecting guards must use the negation of the ordered <= comparison; replacing
them with only `abs(value) > bound` would incorrectly admit NaN. This relies on
ordinary IEEE semantics; fast-math and finite-only compilation remain excluded.

For the opt-in refined real solve, the first nonzero residual correction is
mandatory even when the initial solution already meets the backward-error guards.
After checking the initial KLU output's finiteness, the private path may therefore
compute that exact original-matrix correction before the initial norm/denominator
validation. If the FP64 correction is zero, run the full original validator before
returning or considering fallback. Otherwise perform the same mandatory correction
and then the full original validator; every later correction still follows a full
validation. No returned result bypasses either backward-error guard. Preserve
residual accumulation/rounding, correction budgets/counts, fresh-factor fallback,
typed nonfinite failures and zero-correction behavior. Zero-budget and native solve
paths remain unchanged. Independent old-validator refinement oracles must verify
solutions, failure codes and statistics before accepting this deferred check.

Immediately after a successful prepared direct-Newton point, the owned system and
returned state are identical to the just-checked accepted trial: source scale is
one, extra GMIN is zero, no projection mask applies, and no callback intervenes.
Only this boundary may perform a fresh final residual assembly without rebuilding
unused Jacobian stamps or affine RHS terms. The preceding accepted trial already
checked their bounds and Jacobian rank at that identical point. Recompute every
original linear/behavioral residual and scale in its original order, with fresh
expression value/domain checks and the complete AD checks already performed by
that immediate accepted full trial, then apply the unchanged original residual
guard before publishing expression-cache data. This proof is local to the
successful call, never retained across matrix/RHS changes or inferred from a cache
hit. Public, unprepared, DC, native, projection and continuation checks retain full
assembly. Tests must reject initial/trial Jacobian and affine overflow even when
residual cancellation is exact, and preserve successful point/trace/KLU parity.

Within a Newton iteration, direct behavioral mode may omit the previous-residual
merit traversal that only continuation line search consumes. A fully assembled
behavioral trial may retain its normalized residual scalar through the pure
update-norm calculation and reuse it for that same accepted trial; no state,
assembly, scale or residual mutation may intervene. Preserve every guard and
comparison, including trial merit rejection and update convergence. For a
behavioral assembly, move its already-computed affine RHS into the solve instead
of copying/negating the residual only to overwrite it; native RHS construction
remains unchanged. These are local reuse of identical data and removal of unused
work, not changes to convergence policy. Prepared/public state, trace and KLU
statistics parity and failure regressions remain required.

The private real-solve validator may certify both backward-error guards before
computing row sums or inverse scales. Compute each original ordered row product,
componentwise denominator d, residual r, and requested FP64 correction unchanged.
Every row must satisfy `r <= round_long_double(d * (T / 2))`, where T is the
existing representable FP64 normwise tolerance; a zero denominator instead
requires exact zero residual. If every row qualifies, both original guards pass.
Any uncertain row restarts the complete existing private validator, including its
row metadata, original componentwise divisions, norms, and exact diagnostics.
This all-row certificate supersedes the earlier componentwise-only shortcut.
Public real and complex validation remain unchanged. Private callers consume only
admission status, never a successful error-estimate payload.

The certificate is enabled only for binary long double with at least FP64
precision, maximum exponent at least 2200, and minimum exponent at most -4300.
Finite FP64 inputs and the signed-32-bit CSR bound put every positive accumulated
d between 2^-2148 and 2^2080. Monotonic rounded accumulation gives abs(product)
<= d and abs(rhs) <= d, hence r <= 2*d. The half-tolerance product and nonzero
componentwise quotient are normal in that format. The original row scale is at
least every coefficient magnitude and abs(rhs); its positive inverse scale is
between approximately 2^-1024 and 2^1074. Consequently scaled row/RHS norms and
their products with the FP64 solution norm stay finite, and nonzero original
normwise quotients remain above 2^-4232. Other formats retain full validation.

For the rounding proof, let u <= 2^-53 be unit roundoff and let n <= 2^31 be the
number of nonnegative summands. The standard accumulation bound
`gamma_n = n*u/(1-n*u)` is below 2^-21. Write S for the computed absolute row sum,
X for the exact maximum solution magnitude, and b for the RHS magnitude.
Positive product/sum bounds give
`d <= (1+u)*(1+gamma_n)/(1-gamma_n) * (S*X+b)`.
For the same positive computed inverse scale s, the original global normwise
denominator is at least `(1-u)^3 * s*(S*X+b)`: it takes separate maxima of the
scaled matrix and RHS rows before its final product and sum. Certified residuals
and their scaling add at most three `(1+u)` factors before the final FP64 cast.
The resulting inflation, including that cast, is below 1.000002; it cannot bridge
the factor-of-two tolerance margin. This also proves the original componentwise
guard because T is smaller than its frozen 1e-5 tolerance. Zero rows satisfy both
guards exactly. No fast-math, reassociation, altered rounding mode, or skipped
original admission guard is permitted.

Preserve dimensions, finite-input checks, metadata allocation failure precedence,
residual/correction rounding, refinement budgets/statistics, Jacobian rank checks,
and fresh-factor fallback. A certified pass need not populate unused row metadata;
its validity flag must stay false until a full validation writes it. On fallback,
all partially written correction entries are overwritten by the complete original
pass before refinement consumes them. Independent public-validator comparisons
must cover certified nonzero residuals, uncertain passing and failing cases near
both frozen tolerances, exact diagnostics, signed zero, subnormal/extreme inputs,
changed matrix/RHS calls, and unchanged correction/solve statistics. Private
counter probes and before/after trajectory hashes must precede retaining this
optimization; the probe is not part of production or qualification evidence.

A prepared nonlinear point owner may retain at most two assembly storage buffers
under its immutable MNA snapshot. A buffer owns its CSR pattern/value storage,
residuals, scales and extended-precision affine-RHS scratch. Acquire it only at the
existing assembly allocation point, after the original argument/state admission
checks. Its first full use copies the original canonical pattern; later full uses
copy every current matrix value and reset every residual, scale and affine scratch
entry in the original order. Public/unprepared/native assembly keeps its existing
owning allocation path. No matrix values, residuals, scales, derivatives or
acceptance proof may be reused as computed results through this storage cache.

A lease belongs to exactly one active assembly and returns storage on every
normal or failed exit. Moves transfer that ownership exactly once; destruction
and recycling do not allocate. Retain no more than two idle buffers, with no
cross-owner or cross-invocation sharing. A failed or partial assembly can leave
scratch contents but cannot mark a partial pattern initialized, publish an
expression-cache entry, or bypass any reset/guard on reuse. The prepared final
residual specialization may leave its unused Jacobian storage allocated, while
retaining the already-governed fresh value/domain/residual checks and immediate
accepted-trial AD proof. The
existing allocation-error catch and validation order remain; capacity reuse need
not repeat allocations already satisfied by owned storage. Private diagnostics
and hostile failure/recovery tests must establish bounded ownership, actual
pattern reuse and exact prepared/checked solutions, errors, traces and KLU counts.

Within behavioral Newton only, the pre-backtracking proposal vector may be
initialized without copying the previous state, and the unused first
`previous+delta` traversal may be omitted. Every original delta finite/magnitude
check remains before any trial assembly, in the same order. Backtracking writes
every proposal entry as the original `previous+scale*delta`, beginning with scale
one, before any consumer can read it. Native proposal construction and limiting
remain unchanged. Neither change alters floating-point equations or any guard.
Retain these storage/local-work optimizations only after bit-exact workload and
counter comparisons and a favorable isolated timing measurement.

A prepared behavioral full-TRAP attempt may lazily retain the two physical
coordinate vectors extracted for its immutable accepted start and full trial
states. Only the original checked coordinate extraction populates either slot,
at its original derivative-reconstruction position: all state dimensions,
finite/magnitude bounds, capacitor terminal indices and voltage differences,
and inductor branch indices remain checked. Estimator and feedback evaluation
may reuse these identical coordinates within that attempt. Each slot is bound
to its state object, with a typed structural failure on an owner mismatch.
No state value is changed between extraction and reuse; private half steps,
rejected retries, later accepted steps and other invocations never receive these
slots. Derivative reconstruction, error normalization, feedback arithmetic,
failure ordering and every acceptance threshold remain unchanged. The checked
unprepared path recomputes coordinates at every original call. Existing exact
prepared/checked traces cover varying states, actual NR/LTE rejection, half-step
audits, source restarts, observer isolation and sixteen-agreement recovery;
hostile invocation admission remains independently checked before execution.


## Immediate accepted-trial derivative reuse

Only an owned prepared direct-Newton solve may avoid repeating reverse AD in its
final original-system residual check. Every full trial still evaluates complete
expression values and derivatives with all original intermediate, derivative,
affine and Jacobian bounds. Move the complete descriptor-ordered evaluations
into that trial's private assembly storage. A move-only successful attempt may
transfer this set together with its exact moved FP64 solution only after the
full trial, update/residual criteria and accepted-Jacobian KLU check have all
passed. Incomplete or rejected trials confer no proof. Idle workspace contents,
older expression-cache hits and equality of root values alone cannot establish
this boundary. No callback, state arithmetic, program, matrix or RHS mutation
intervenes between that accepted trial and its final check.

The final check freshly validates state shape and bounds, every structural
expression dependency (including inactive lazy branches), and the original lazy
value/domain traversal. Each new root value must match the accepted trial value
bit for bit, including signed zero; an incomplete set is kInvalidStructure and
a value mismatch is kSolutionValidation. This mismatch is a failed proof, never
a reason to publish reused derivatives. Rebuild every original linear row,
behavioral residual addition, scale and normalization in the existing order.
Only after the unchanged final residual guard succeeds may the complete accepted
value/AD set enter the four-entry expression cache. Its derivatives are the exact
results of the same immutable program and state whose fresh value/domain check
just succeeded. Public, unprepared, DC, projection, native and continuation paths
retain their full fresh final AD and assembly behavior.

Private diagnostics separately count fresh final values, reused accepted
derivatives and fresh full final evaluations. Tests must preserve exact points,
Newton traces and all KLU counters; distinguish derivative-only failures from
value failures before acceptance; exercise failed/backtracked trials and later
recovery without stale publication; and retain finite/bound/domain, bit-identity
and owner-isolation checks. Matched binaries must produce bit-identical model
outputs and work counters before a development timing comparison. This reuse
changes neither a numerical gate nor the complete frozen qualification workload.


A full trial retained as the next Newton iteration's working assembly was not
accepted, since an accepted trial returns immediately with its paired proof.
After taking that working assembly, release its retained expression evaluations
before the next solve/trial. Its checked Jacobian, affine RHS, residual and scales
already contain every numerical input needed by this iteration; no later read
uses those superseded evaluations. The newly checked trial alone may supply
accepted derivative proof. Keep descriptor-vector capacity and all numerical
operations, guards, traces, statistics and publication timing unchanged. Exact
point/workload parity and isolated timing govern retaining this storage change.

## Immediate prepared sparse-input admission reuse

Only the private prepared Newton workspace may invoke an admitted real sparse
solve after a successful full assembly. The complete Jacobian has just passed
its original finite/magnitude scan. Every behavioral affine-RHS intermediate
has passed the existing long-double magnitude bound (at most 1e100), so its
FP64 conversion is finite. The accepted-Jacobian rank check instead constructs
an all-zero RHS of the admitted row count. These matrix/RHS objects remain
unchanged through the synchronous solve; no callback or other owner intervenes.
The admission fact belongs to this completed assembly, never to idle storage,
an earlier point, an expression-cache hit, or an unvalidated caller assertion.

The real factorization exposes no public bypass flag or unsafe free function.
A private entry point is accessible only to the prepared Newton workspace. On
an exact match to the analyzed canonical CSR, it may omit the repeated finite
matrix-value and RHS-value scans. Matrix dimensions, value count, both CSR index
arrays and RHS length still receive their original checks. Every pattern
mismatch runs the complete original converter and numeric admission, retaining
its error order. The public, unprepared, native and complex entry points keep
all input validation. The fixed refinement-budget admission and allocation-error
handling remain. No CSC packing, numeric-value equality, KLU factor/refactor,
pivoting retry, mandatory correction, output finiteness, backward-error guard,
refinement budget or statistics behavior is changed.

Independent tests must show that nonfinite or over-bound assembled inputs fail
before KLU work, that failed/partial assemblies cannot authorize a later call,
and that valid recovery matches the fully checked point in solutions, Newton
traces, error identities and every KLU counter. Keep all public hostile CSR/RHS
and refinement regressions. Retain the optimization only after matched workload
outputs/counters remain bit-identical and isolated timing supports it.
