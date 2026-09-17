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
