# EMI-01 capability gaps and proposed bounded EMI-02 contracts

**Status: proposals only. No EMI-02 implementation is authorized or included.** Each slice
requires its own accepted ADR and a separate implementation/review gate. The recommended next
slice is **EMI-02A, disjoint pairs of linear coupled inductors**. It supplies an independently
testable prerequisite without making the production simulator claim SiC-switching support.

The evidence for this report is the actual [reference circuits](circuits.py),
[manifest](manifest.json), [model and adapter audit](MODEL_PROVENANCE.md), and inspection of the
current C++ [IR](../../cpp/include/ohmnivore/ir.h), [MNA compiler](../../cpp/src/compiler.cc),
[parser](../../cpp/src/parser.cc), [waveform evaluator](../../cpp/src/waveform.cc),
[nonlinear solver](../../cpp/src/nonlinear.cc), and [transient solver](../../cpp/src/transient.cc).
The authority remains [ADR-001](../../docs/adr/ADR-001-cpp-cuda-migration.md).

## Existing capability versus this workload

| Required by the chosen workload | Current C++ boundary | Decision |
|---|---|---|
| R, fixed C/L, independent V/I, PULSE/PWL, explicit harness/chassis RLC | Present; insertion-ordered MNA variables and canonical CSR | Reuse. The reference's equivalent load is already expressible once devices/coupling are available. No motor component is required. |
| CM choke and coupled harness | IR has individual inductors, no mutual-coupling descriptor | Add only disjoint two-winding linear pairs in EMI-02A. |
| Selected SiC model | No MOSFET descriptor or supported MOSFET model | A memoryless Level-1 DC MOSFET does not supply this model. Use only the selected fetched behavioral model's reachable semantics, in separate slices. |
| Model current law | Diode Shockley and bounded Ebers–Moll BJT DC are present | Neither reproduces the selected SiC channel/reverse-conduction equations. Do not repurpose those devices. The chosen SiC core itself has no native M or D card. |
| E/G/B behavioral sources and sensed source currents | Absent | Need bounded expression evaluation, exact derivatives and MNA branch-current dependencies. |
| Dynamic gate/drain storage | Fixed C is present; diode transient is memoryless; BJT transient explicitly unsupported | Preserve the model's fixed sensing capacitors and controlled displacement-current paths. Do not silently replace them with a memoryless channel or an arbitrary C(V) stamp. |
| `.SUBCKT`, X instances, local `.PARAM`, `PARAMS:` overrides | Absent | Admit exactly the selected package/core hierarchy through a bounded import layer; no general SPICE expansion. |
| Adaptive nonlinear transient | BE/TRAP with direct Newton, step-doubling/error estimation, bounded retry, KLU | Extend state/residual handling only after stateless behavioral qualification. The reference uses Gear-2; existing BE/TRAP may qualify by convergence. Identical internal timesteps are not a correctness requirement. |
| Initial conditions and gate drive | DC initialization followed by waveform-zero algebraic projection | Explicitly reconcile the different source defaults; see below. Do not change existing defaults globally. |
| ngspice `.TRAN tstep tstop tstart tmax`, `.options`, `.save`, `.control` | Production `.TRAN tstep tstop [tstart] [UIC]`; no generic command interpreter | Map max step through existing `time_step_seconds` and use named state extraction in the reference bridge. Do not add the ngspice control language. |
| EMI spectra, masks, mass, finite design enumeration | External EMI-01 harness only | Keep external. None is required in the production device/parser/transient slices. |
| Statistical switching/thermal/magnetic fidelity | Not supplied by either production or reference | Remains unqualified; no production semantics can repair missing reference physics by assertion. |

Every proposed slice preserves Rust/Cargo, ordinary existing C++ behavior, historical AC evidence
bytes and the opt-in CUDA contracts. CPU FP64 with pinned SuiteSparse KLU remains production
authority and the supported no-GPU implementation. ngspice's Sparse solver is an independent
external oracle, not a replacement for production KLU. GPU kernels, device residency, dispatch,
mixed precision, distributed solving and general optimization remain closed.

## EMI-02A — disjoint pairs of linear coupled inductors

**Recommended first CPU slice.** Support a `Kname Lfirst Lsecond k` declaration referring to two
existing, distinct linear inductors. A winding may belong to at most one pair. This admits the
reference's CM pair (`k=0.995`) and harness pair (`k=0.2`) and excludes general multiwinding
coupling graphs. Constant positive inductances and finite `-0.999 <= k <= 0.999` are proposed;
`|k|=1`, saturation, hysteresis, frequency-dependent permeability/loss, magnetizing branches,
transformer ratios and nonlinear magnetics are explicitly unsupported.

Let each existing inductor's positive terminal be its dot, voltage be positive-to-negative,
and current enter its positive terminal. Freeze:

```text
M = k * sqrt(L1) * sqrt(L2)
[v1]   [L1  M ] d[i1]/dt
[v2] = [M   L2]  [i2]
W = (L1*i1^2 + 2*M*i1*i2 + L2*i2^2)/2
```

Compute M with bounded intermediates and check representability. In the current compiler's
branch convention, each inductor has `C(branch,branch)=-L`; the two mutual entries are `-M`.
Keep both entries, their symmetric positions and deterministic declaration identities. Reversing
one winding's terminals changes its external sign; reversing k has the corresponding physical
effect. Positive equal outward currents reinforce the reference CM flux. Equal opposing
currents see leakage behavior. For equal L, the modal inductances are `L+M` and `L-M`, both
strictly positive in the admitted domain. The pair's stored-energy matrix must be positive
definite even though the MNA branch block uses negative entries.

DC coupling contributes no derivative term; the existing ideal-short DC semantics remain.
AC uses `j*omega*C` and transient uses the existing full C matrix, including off-diagonal
branch history terms. No ad hoc per-winding independent solve is permitted. Preserve current
initialization: no-UIC uses a complete coupled DC solve; UIC zero-current constraints apply to
both windings. Rejected trial steps cannot commit either winding current or flux linkage.

Proposed typed failure precedence: malformed K syntax → `kParse`; unsupported coupling form
or out-of-domain finite k → `kUnsupported`; duplicate K identity, missing/non-inductor target,
self-pair or repeated winding membership → `kCompile`; non-finite values or overflow →
`kNonFinite`; existing KLU singularity/validation failures retain their present types. Validate
the complete coupling set before any matrix mutation. A failed compile returns no partial MNA.

Required acceptance gates:

1. An independent exact-small 2×2 inductance oracle derives M, modal inductances, energy,
   voltage/current slopes and sign-reversal identities without compiler/stamp helpers.
   Cover unequal L, k=0, ±0.2, ±0.995 and the admitted endpoints.
2. Compare complete CSR G/C entries, branch signs, AC response and transient trajectories with
   that oracle and separate hermetic ngspice fixtures. Proposed analytic value tolerances are
   `1e-12 + 1e-10*|expected|` in each quantity's SI units; test sparse solve guards separately.
   For independently integrated trajectories use `1e-6 A + 1e-3*|Iref|` and
   `1e-5 V + 1e-3*|Vref|`, plus timestep refinement rather than a single coarse comparison.
3. Verify nonnegative energy for an independently enumerated signed-current grid and the
   eigenvalue bounds; forcing negative eigenvalues, one omitted mutual stamp, one reversed
   stamp, or omitted history coupling must make discriminating tests fail.
4. Include forward references, reversed declaration order, duplicate/unknown/self references,
   duplicate membership, NaN/infinity, `|k|>=1`, overflow, singular topology, UIC/no-UIC,
   source hard points, retry exhaustion and failed-step atomicity.
5. Existing canonical C++ checks and acceptance fixtures pass without changed expected bytes.
   Add a linearized version of the reference filter/harness driven by independent sources;
   label it a coupled-element test, never a realistic switching/performance benchmark.

EMI-02A does **not** enable the EMI-01 inverter, and must continue returning unsupported for its
SiC/model-language constructs.

## EMI-02B — selected model's stateless expression and derivative semantics

This proposal adds only a bounded evaluator and independently tested Jacobians, before transient
integration. The input equation authority is the **exact fetched model and adapted hash** in
[MODEL_PROVENANCE.md](MODEL_PROVENANCE.md), restricted to the reachable `MSC040SMA120B` package
and `MSCSICFET1200` core. This document does not redistribute their proprietary equation text.
The package's fixed R/L elements remain native R/L elements. There are no required `.FUNC`,
native MOSFET, native diode, table, delay, Laplace, stochastic, Verilog-A or thermal-network
constructs in that reachable subset.

The exact executable equation graph comprises the package R/L network and the core's named
`EfVDS`, `EfVGS`, `EfVGD`, `Ek2`, `Efb`, `Eremapvgs`, `Eremapvgd`, `Efcond1`, `Efcond2`,
`EIbd2`, `G13`, `G14`, `G41`, `Bgd` and `Bds` expressions, the declared parameter equations,
fixed Cgs, sensing capacitors/sources and passive resistors. Preserve every expression and
parameter from the pinned member; no device fit, simplification or algebraic elimination is
part of this slice. `Bgd/Bds` are evaluated as stateless functions of voltage and **current
unknowns** here; their capacitor state enters only in EMI-02C.

Freeze the minimum language vocabulary to finite numeric/engineering literals, identifiers,
parentheses/braces, unary signs, `+ - * / **`, `exp`, comparisons `<`/`>`, lazy `if`, terminal
voltage differences `v(a,b)`, node-to-ground `v(a)`, and currents `i(Vname)` through explicitly
identified voltage-sensing sources. Current orientation is positive source terminal to negative
source terminal. Bind names and parameter scope before evaluation. Local parameter dependency
resolution must allow the selected forward references, reject cycles, and evaluate an instance
override's right-hand side in its caller scope (including the selected `MDE=MDE` forwarding).
No unknown token may be silently ignored.

The compatibility authority is ngspice **46 in `ps` mode**, not an unspecified PSpice or LTspice
dialect. The pinned upstream `src/spicelib/parser/{ifeval.c,ptfuncs.c,inpptree.c}` exposes
load-bearing details: conditional evaluation chooses only one branch; its derivative chooses the
same branch; generic behavioral powers use absolute-base semantics outside LT/HSPICE modes;
division includes a signed `gmin*1e-20` denominator offset. The proposed evaluator must freeze
and independently test the actually reachable semantics, including equality boundaries and
that offset with fixed reference GMIN. It must not import C++ `pow` and `/` assumptions as
though they established oracle parity. Parameter-expression evaluation and runtime behavioral
evaluation are separate dialect paths; test both with their actual inputs.

Proposed budgets: expression depth ≤64, ≤512 syntax nodes per expression, ≤16,384 total
expression nodes, and at most the selected package/core definitions. Reject recursion,
parameter cycles, dynamic topology and all other functions. Each evaluation returns value,
derivatives with respect to its bound state unknowns, or a typed unsupported/non-finite/domain
failure. No clipped current/exponential may be reported as faithful evaluation unless the
original expression itself clips it. Newton trial damping can be separately bounded; it cannot
change an accepted model equation.

Required gates are independent analytic values/Jacobians for every operator, lazy conditional
domain traps, signs and threshold equalities; finite-difference and complex-step checks where
mathematically valid; source-current derivatives; parameter-forwarding/cycle/limit failures;
then DC I–V sweeps against the fetched-model ngspice oracle at both declared temperatures and
forward/reverse gate states. Complex-step is invalid across comparisons/absolute-value branch
boundaries: use one-sided analytic checks there. Require node-current conservation for each
two-terminal current source and validate the original nonlinear residual independently after
Newton. This slice alone exposes no new transient or CLI model support.

## EMI-02C — controlled displacement current and transactional transient state

Extend the bounded behavioral source graph to CPU nonlinear DC/transient solves with native
RLC/sensing-source states. Keep a complete MNA solve for the entire connected device/circuit;
no splitting device terminals, bridge legs, filter windings or chassis returns into independent
jobs. Reuse KLU symbolic structure only when the entire pattern identity matches; numerical
factorization and nonlinear state remain private to the circuit/job.

For a voltage-controlled charge element, the physical constitutive relation is
`i=dQ(v)/dt`, `C(v)=dQ/dv`. A discrete charge formulation has residual
`a0*Q(v_new)+history`, with Jacobian `a0*C(v_new)`. It is generally **wrong** to insert
`Q(v)=C(v)*v` when C is differential capacitance, because its derivative adds `v*dC/dv`.
An independent analytic ramp/closed-loop test must expose that error.

However, the selected vendor implementation is a capacitor-current multiplier network. Its
current is `i0*(1+f(control_voltages))`, with i0 an explicitly sensed baseline-capacitor current.
Its Newton derivatives include both `1+f` with respect to i0 and `i0*df/dx` with respect to
each controlling voltage. These terms must remain in the simultaneous Jacobian. Treating the
sensed current as a previous-iteration or previous-timestep constant changes the equation.

For its Miller branch, the control/sensing voltage share the drain-to-gate endpoints, admitting
a scalar charge integral. For the output branch, the controlling external Vds and capacitor
voltage differ because of the finite p-well resistance. There is no permission to replace that
network with a scalar `Q(Vds)` or to assert a global conservative terminal charge function from
a C(V) curve alone. Preserve the auxiliary states and two-terminal KCL. Charge-conservative
device improvements would define a different model and require another reference/ADR; they
are outside this port.

An attempted timestep owns separate trial voltage/current state, capacitor histories, inductor
flux/current histories, nonlinear guesses, LTE substep histories and diagnostics. Accepting a
step commits them together in deterministic order. Newton failure, local-error rejection,
unrepresentable time, resource exhaustion or validation failure commits none of them. The
full-step and two half-step LTE trajectories must start from the same accepted snapshot;
half-step state cannot contaminate the rejected full step or another job. Factorization caches
may retain valid structural work but cannot make a rejected physical history observable.

No new integration method is presumed necessary: first test existing BE/TRAP under the frozen
waveform/spectral refinement gates. If it cannot qualify, report that result and write a
separate Gear/BDF contract before adding one. Do not increase current default iteration/step
bounds for unrelated models, or reinterpret the external ngspice tolerance knobs as existing
Ohmnivore tolerances. A new scoped options object must specify voltage/current residual and
update scaling, local-error scaling, finite-value guards, exact iteration/retry/step budgets,
and traces before this slice is implemented.

Required gates: independent nonlinear-capacitor ramp and periodic-charge loops, the existing
adapter's signed affine-capacitance cases, current/charge conservation and full Jacobians,
source-current cross-coupling, package-RLC resonance, forced Newton/LTE failures at every commit
boundary, restart from accepted snapshots, reproducible typed failures and KLU residual guards.
Finally run the selected SiC device's standalone clamped-voltage, reverse-conduction and gate
charge fixtures against ngspice with independent sampling refinement. No full inverter claim is
made until EMI-02D passes.

## EMI-02D — bounded model import and coupled inverter qualification

Expose only the qualified two-definition model closure and at most four package instances in
one circuit. Proposed import budgets are ≤16 expanded subcircuit instances, depth ≤3,
≤512 total MNA unknowns, and the EMI-02B expression budgets. The reference archive/model must
remain externally fetched and hashed; do not embed proprietary model bytes or a transformed
copy in production source/distributions. Unknown model version, package, syntax or parameter
returns `kUnsupported`/provenance failure rather than substitution. The fixed `TJ_C` input takes
only the declared 27/125 °C values in this initial port; GM/VTO/MDE remain the pinned defaults.

The adapter supplies ordinary RLC, source waveforms, two K pairs and the selected model graph.
It maps named states to the unchanged external measurement pipeline. No general `.include`
filesystem search, shell/control command, general `.options` parser, optimizer, model catalogue,
arbitrary temperature sweep, `.MODEL` MOSFET family or ngspice installation dependency is added.

### Initialization and finite source edges

The current production compiler uses absent explicit source DC values as zero. Its no-UIC
initialization solves `b_dc`, then projects algebraic waveform values at t=0 while preserving
reactive state. The external reference's PULSE/PWL initialization uses their initial values.
Blindly importing its abbreviated source cards would therefore charge the gates differently.

The narrow bridge must emit explicit DC values equal to each waveform at t=0 before the initial
nonlinear solve: low-side gates initially +20 V for the pulse-train study; high-side gates
initially −3 V; both DPT gate sources initially −3 V. High-side drive is referenced to its
own switch/source terminal, never to global ground. Preserve the existing production defaults
and source-replacement rule. DC initializes every internal package capacitance/inductance and
the complete filter/load/chassis network; no UIC, arbitrary saved-state injection, or zeroing of
selected hidden device states is admitted.

Retain actual PWL/PULSE timestamps, finite 10 ns source transitions, widths and periods from
`circuits.py`. Do not translate the descriptive “200 ns deadtime” into different endpoint
timing: the precise source edges, internal gate threshold and package parasitics determine the
actual channel overlap. Land on source breakpoints, integrate the discarded settling interval,
and emit complete accepted-state coverage without extrapolation. Output sampling remains an
independent observation choice, not an instruction to force adaptive timesteps onto an FFT grid.

Required acceptance gates:

1. All EMI-02A–C independent tests pass, including negative syntax/model identity cases and
   simultaneous-invalid-input error precedence.
2. Qualify the same half-bridge double-pulse fixture at all frozen refinements: gate/drain
   waveforms, finite switching edge times, peak Vds, signed terminal energy, measurable ringing
   frequency/damping and explicitly unresolved ringing conditions. Preserve the terminal-current
   energy convention, including recoverable capacitive energy; do not relabel it intrinsic loss.
3. Qualify **all nine** complete bridge/filter/load/chassis jobs against external ngspice, with
   the same startup/settling, output sampling, CM/DM signs, spectrum, stress/flux and failure
   accounting contract. Use the frozen ADR-002 waveform/spectral gates; no selection of only
   convergent/pass-mask corners. Independently refine CPU and reference results and compare
   their converged outputs. A solver/model failure never becomes predicted EMI feasibility.
4. Verify missing/truncated/non-finite output, budget exhaustion, parser/source failures and
   one failed mandatory corner suppress whole-candidate feasible ranking. Record exact source,
   model, toolchain and job identities; rerunning a single failed job does not silently repair a
   complete-study evidence record.
5. Production canonical checks and separate ngspice checks pass. Obtain independent review of
   model equations, derivatives, dynamic state, coupling signs, initialization and all identities.
   No GPU runtime/performance claim is a gate or consequence of this CPU slice.

## Physics limitations cannot be resolved by a CPU port

The selected reference omits bipolar body-diode reverse recovery, self-heating and avalanche
qualification. Its numerical qualification cannot establish recovery-dominated switching loss,
hardware EMI accuracy or laboratory correlation. Retain that limitation across all comparisons;
a later device-model improvement needs a different reference and new evidence.

The finite positive-modulation segment is not a full machine fundamental cycle. The lumped
load/harness has no measured frequency-dependent impedance, and linear magnetics have no core
loss/saturation waveform or manufacturing qualification. The physical mass model and stress
screens do not establish total thermal feasibility, full stability or component availability.

The declared Vds/Id and external filter constraints do not themselves qualify internal/high-side
gate-voltage overshoot, parasitic turn-on, gate-oxide reliability, all capacitor AC/ripple ratings,
or complete transistor safe operating area. Keep those outside any predicted feasibility claim
until separately measured/modelled gates are supplied. A research-mask reserve is an assumption,
not a quantified bound on these missing effects. None of these gaps warrants silently adding
unvalidated physics to EMI-02A–D.

For the final frozen DPT fixture, unresolved ringing is an accuracy failure: three strictly
decaying same-polarity peaks above 0.2 V are required. Future CPU qualification must preserve
that requirement and the candidate-specific integration policy recorded in ADR-002.

The v2 passing/boundary reference changes only finite filter values, geometry and integration
refinements. It requires no additional production semantics beyond EMI-02A–D above. Bind parity
evidence to its exact versioned manifest, preserve the failing control and near-boundary rejection,
and reproduce the all-corner passing result without changing the 6 dB reserve. Enlarged geometry
does not validate the fixed CM parasitics, omitted DM self-capacitance, winding fill or gap fringing.
