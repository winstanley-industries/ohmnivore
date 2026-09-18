# EMI-02 resumed implementation review

This note records focused source review and regression coverage after the first
retained qualification. It does **not** claim that the resumed full study has
qualified: the separate `run-2` execution and artifact audit are pending at the
time of this review. The existing `run-1` evidence and
[`INDEPENDENT_DYNAMICS_REVIEW.md`](INDEPENDENT_DYNAMICS_REVIEW.md) remain historical
records and were not rewritten. No performance or GPU authorization follows from
these tests.

The governing contracts are [ADR-004](../../adr/ADR-004-emi02-behavioral-expressions.md),
[ADR-005](../../adr/ADR-005-emi02-behavioral-transient.md),
[ADR-006](../../adr/ADR-006-emi02-model-import-qualification.md). Independence
here means separately derived physical oracles and review across implementation
authors. The author of the derivative-history physical-oracle tests also
implemented the expression cache and its initial five regression cases; those
cache tests are authored regression coverage, not independent review of that
author's own implementation. The same author implemented the two-buffer assembly
workspace and its two hostile regression cases; its source received separate
review by the transient implementation author and coordinator. The final-residual specialization and its two added
regressions were authored separately and then independently reviewed. The
transient/companion implementation received separate source review; its author's
subsequent inspection is identified separately below.

## Exact prepared caches and solver validation

The source review checked these boundaries:

- The nonlinear owner retains at most four complete expression value/AD sets.
  Its compiled programs and descriptor order are owned and immutable. Full-state
  keys compare FP64 representations, including signed zero. Initial assembly and
  history validate dimensions and every state value before lookup; this also
  covers inactive and unused dependencies. New trial states evaluate complete
  value/AD results afresh. Final checks freshly evaluate values and domains and
  may reuse only the immediate accepted trial's derivatives as described below. Only a complete point that passes final original
  residual validation publishes an entry. A failed solve cannot evict entries by
  publishing partial results. Linear rows, current companion values, affine
  terms and stamps are rebuilt even on an expression hit.
- Four entries cover the previous accepted full state, private half-step states,
  and the next full solution before its first half reuses the previous state.
  Retention is bounded and per owner. Cache diagnostics distinguish reuse from
  fresh initial/trial/history evaluations, fresh final scalar values and reused
  immediate accepted-trial derivatives.
- Companion preparation first calls the original public companion constructor
  at the original lazy validation point. Its owned G/C mapping retains separate
  G-only, C-only and overlapping arithmetic, union order, signed zero and exact
  cancellation. Private multiplication retains state validation and every
  partial-sum check. Opaque products carry owner/shape checks and are used only
  for the same immutable accepted state in the full and first half of one
  attempt. The second half and a new attempt receive fresh products. The
  nonlinear owner copies companion values before the next companion update.
- Canonical CSR reuse retains the original converter as the fallback when a
  supplied pattern differs, preserving simultaneous-invalid-input precedence.
  Private validation reuses immutable row metadata and long-double products;
  the public sparse validator remains the independent checked path. An all-zero
  RHS/solution shortcut still follows actual KLU factorization and solve, so a
  singular changed Jacobian cannot pass by returning zero.
- The deferred first residual check applies only to the opt-in refined solve.
  Matrix/RHS validation and initial KLU-output finiteness checks remain before
  correction. Allocation timing/catching was restored before correction work is
  counted. The exact original long-double residual determines the mandatory
  first nonzero FP64 correction, after which both original backward-error guards
  run. If that correction rounds to zero, the complete validator runs before
  return or fallback. Later corrections and the shared remaining correction
  budget across fresh-factor fallback remain checked.

The separate solver review supplied the scalar discriminator `A=[1.5]`,
`b=denorm_min`: its FP64 correction rounds to zero while its componentwise
backward error is 0.2. It must fail validation and consume zero corrections,
including the fresh-factor fallback. The retained focused test log confirms that
case passes. Other cases compare the old validation/correction computation at
budgets 0, 1 and 4, corrections whose updates underflow, initial solution
overflow, signed zeros, changed matrices and repeated scratch use. Two further
cases retain coverage of zero-rich rows, unit scales, finite admission and mixed
exact/nonzero correction rows. The candidate zero-coefficient, exact-zero-error
and unit-inverse shortcuts were reverted after unfavorable private timing;
these tests pass against the restored implementation and do not imply that the
rejected shortcuts remain implemented.

The stronger private all-row admission certificate was reviewed independently.
It computes the original ordered product, denominator, residual and requested
FP64 correction for every row. Admission requires every row to lie below half
of the tighter normwise tolerance. Under the declared extended long-double
exponent range, finite FP64 products, positive accumulations, inverse scales and
nonzero quotients remain normal and finite. At most `2^31` nonnegative summands
give the documented `gamma_n < 2^-21` accumulation bound. The same positive row
scale relates the componentwise denominator to that row's scaled matrix/RHS
sum; taking separate global maxima only increases the original normwise
denominator. The combined rounding inflation remains below 1.000002, far inside
the factor-of-two margin. This proves both unchanged guards rather than
replacing them with a new tolerance.

Any uncertain row restarts the complete original private validator, overwriting
all partially filled corrections and preserving exact failure diagnostics.
Finite admission and metadata allocation retain their original position. A
certified pass leaves unpopulated metadata invalid. Public real/complex
validation remains unchanged, and private callers consume only admission status.
The 22 passing sparse tests include certified nonzero residuals, uncertain
passing/failing cases around the tighter normwise boundary, subnormal/extreme
values, zero denominators and exact public error messages. Three paired short
runs per case retained identical raw hashes and work counters. Their timing was
a development comparison; it does not establish full-workload resource
qualification or a concurrent-throughput claim.

The private prepared finite-input admission path received review across its two
implementation authors. The solver entry is private and accessible only to the
owned Newton workspace. The caller reaches it only after a complete full
assembly has checked every matrix value and affine-RHS accumulation against the
stronger magnitude bound; conversion to FP64 is then finite. The rank check
instead constructs a fresh zero RHS. No callback or mutation intervenes. This
proof belongs to that completed assembly, never to scratch capacity, an older
expression hit or a caller-set public flag.

Only the duplicate matrix/RHS finite scans are omitted on the exact analyzed
canonical pattern. Dimensions, value count, both index arrays and RHS length are
still checked; a mismatch restores the original converter and all finite checks.
Public/unprepared/native/complex paths retain admission. All KLU work, result
finiteness, original-matrix backward-error checks, refinement and statistics are
unchanged. Independent caller review strengthened the new test to isolate the
Jacobian guard: two zero-valued expressions have individually bounded slopes
whose sum exceeds the matrix bound, while values, affine terms and residual
remain bounded. No private admitted call or KLU count may advance. New NaN/Inf
inputs likewise fail before admission; subsequent changed-matrix/RHS recovery
must exactly match the public point, trace and every solver count.

Eight focused targets, including public sparse and refinement regressions, passed;
the strengthened discriminator passed in the final twelve-test cache target.
Twelve matched development records had identical waveform bits and work counts.
Median CPU changed approximately -2.02% for light and +0.48% for boundary, a
mixed timing result. The path was retained for the more expensive light workload,
subject to the unchanged full resource/accuracy qualification. No unrecorded
Newton-trace suppression was added.

The row-scale reduction was reviewed under the explicitly pinned
round-to-nearest arithmetic: after each unchanged addition, the nonnegative sum
of magnitudes bounds the current term and signed product. Only behavioral
assembly uses that proof to reduce redundant checks. Native assembly retains
the original three predicates. Ordered `abs(value) <= bound` checks still reject
NaN and infinity; rejecting guards use the negated ordered comparison. Residual,
affine and Jacobian bounds remain explicit where required.

The prepared assembly workspace retains at most two idle storage buffers. RAII
leases transfer their owner once on moves and recycle without allocating. A full
assembly always copies every current matrix value and resets all residuals,
scales and extended-precision affine scratch; only the immutable CSR pattern is
initialized once per buffer. An interrupted first pattern copy cannot mark it
initialized. Acquisition follows the original argument/state admission. A failed
assembly may leave scratch contents, but the next admitted use resets them before
reading. Final residual evaluation keeps unused Jacobian storage allocated while
retaining fresh value/domain/residual checks and the immediate accepted-trial
AD proof described below. The independent reviewers found no
cross-owner sharing, current/trial aliasing or failed-point publication path.

Two new tests compare points, complete Newton traces, errors and all KLU counts
against the public checked path after partial stamp overflow, NaN admission,
zero-iteration failure and repeated matrix/RHS changes. Counters require exactly
two pattern initializations, no more than two active/retained buffers, no lease
left active after errors, and actual repeated reuse. A second fixture alternates
different immutable owners with the same shape. The ten cache tests and five
other focused targets, including native nonlinear regressions, passed.

Review also found that moving from `Result::value()` had copied the behavioral
RHS because that accessor is const. Taking the successful assembly into a mutable
local now performs the intended move and returns RHS capacity after KLU finishes.
The behavioral pretrial vector no longer copies then overwrites the old state;
all original delta guards remain before the first complete backtracking fill.
Native proposal construction/limiting remains unchanged. Twelve isolated paired
development runs preserved raw hashes and work counters, with small favorable
median timing changes. This is not evidence of full resource qualification.

## Final original-system residual boundary

The final specialization is local to one successful prepared direct-Newton call.
Behavioral points cannot take the zero-iteration acceptance shortcut. The
returned state has just passed a fresh full trial assembly, all its affine and
Jacobian bounds, and the KLU accepted-Jacobian check. The state is moved without
arithmetic changes, and no callback or system mutation intervenes before the
final check. Source scale is one, extra GMIN is zero and no projection applies.

At this boundary the final check can omit constructing unused Jacobian and affine
RHS data. It still recomputes all original linear rows, fresh expression values
and domain checks, residual additions, scales and normalization in their original
order. Complete AD checks come from the immediately preceding accepted full
trial. That trial's complete evaluations move together with its exact solution
in a move-only successful result, only after every full-trial and KLU acceptance
check. Failed or partially captured trials never supply this proof; idle storage
and older expression-cache hits cannot supply it either. Each fresh scalar value
must bit-match its paired accepted value, including signed zero, before reuse;
a mismatch fails closed. The newly computed scalar drives residual stamping.
Publication of the complete accepted value/AD set follows the unchanged final
residual acceptance gate. Public, unprepared, DC, native, projection and
continuation entry points retain full fresh final AD and assembly.

This extension was implemented by the cache/workspace author and independently
reviewed by the transient author and coordinator, with an additional expression
author review. Its private scalar evaluator shares the full evaluator's exact
dimension/dependency admission and recursive lazy value/domain traversal, but
performs no reverse AD. It can therefore return a finite value for an expression
whose derivative fails; it is not an independent expression-admission API. The
new nonlinear fixture deliberately has an exact zero residual and finite root
value with a singular derivative. Initial and backtracked full trials still
reject it, including after one descriptor has already been captured. A later
nonsingular forcing recovers with the exact public state, trace, KLU counts and
history, proving the failed payload cannot be published. Existing repeat-state
checks require fresh full trials and fresh final scalar evaluation even when an
older exact-state cache entry exists. Counters now distinguish fresh final
values from the accepted derivatives, and successful prepared finals perform no
new reverse-AD evaluation. All eight focused targets passed with this change.
Twelve matched development runs retained identical raw hashes and complete work
counters; the light/boundary median CPU changes were approximately -1.82%/-0.29%.
These small gains do not establish the unchanged full-workload resource gate.

An additional storage refinement clears evaluations from the next iteration's
working assembly: a converged trial would already have returned, so these old
evaluations can never supply the current attempt's accepted proof. Jacobian,
affine RHS, residuals and scales are separate owning vectors already stamped;
only the new checked trial can be accepted. The transient author independently
reviewed this lifetime argument. Four focused targets and twelve exact paired
records passed; light/boundary median development CPU improved about 3.47%/1.28%.
This releases superseded derivative allocations before allocating the next
trial's derivative results and changes no numerical operation.

Review found and corrected a test-fixture issue: the first proposed Jacobian
overflow case also overflowed its affine RHS earlier. The corrected case uses a
zero root, where two individually bounded slopes produce an excessive combined
Jacobian while both value and affine RHS are zero. A separate centered expression
has zero value but an excessive derivative-times-state affine term. Both reject
unsafe initial states; trials aimed at the unsafe roots must backtrack and cannot
publish a converged result within a one-iteration budget. The tests compare
prepared and full errors, KLU counts and absence of final/cache publication.
Another case crosses a behavioral voltage-source branch with current sources
between two terminals and compares complete states, Newton traces, history and
all KLU statistics bit for bit.

The final local Newton review also checked three removals of unused work. Moving
the behavioral affine RHS into the solve leaves the residual/scales used by line
search intact; no later consumer needs the moved vector. Direct mode never
consumes the previous-merit traversal, while continuation still computes it.
The normalized residual of a successful full trial is retained only across the
pure update-norm calculation for that identical trial; zero is still an engaged
cached scalar. No state, scale or residual mutation intervenes. An independently
specified linear equation fixes the exact solution, two-iteration update/residual
trace, three KLU solves and fresh evaluator/publication counts. The eight cache
tests, six Newton tests, eleven history tests and fifteen prepared tests all
passed the focused rebuild containing these final local changes.

The separately authored simple-expression specialization also received independent
review. Only a root leaf or subtraction of two leaves qualifies, with at most two
structural dependencies. Dimension and full dependency checks retain their order;
subtraction reads/checks the first value before the second and retains exact value
arithmetic. Reverse accumulation visits the second leaf before the first, keeping
alias cancellation, signed zero and omission of zero derivatives identical. The
original immutable AST and metadata remain; no state-dependent result is retained.
The implementation author's initial 17 expression tests include roughly 28,900
bitwise comparisons against a wrapper that forces the generic evaluator, plus
wrong-dimension, domain, bound, ground, alias and parameter cases. The independent
ngspice fixture also passed. Paired development traces retained exact hashes and
work counters; those short timings are not qualification evidence.

The compiled reverse-AD traversal then removes only constants and comparison
visits that performed no outgoing derivative operation. Retained nodes keep the
original descending order, zero-adjoint test and all factor/product/accumulation
guards. Incoming adjoints to omitted comparisons are still checked, including an
overflow discriminator. Complete value recursion, lazy branches and structural
dependency admission remain unchanged. Separate reviewers examined that boundary;
the implementation author's 19-test log and independent ngspice result passed.
The full-scan test entry point bypasses both specializations and checks exact
values, ordered gradients and complete errors. Isolated paired workloads retained
exact traces/counters with favorable development timing.

## Derivative-history policy and independent physical oracles

The derivative-history estimator is an opt-in policy for the model runner. The
ordinary default retains full/two-half estimation. The actual BE/TRAP state
equations, physical tolerances and external qualification gates are unchanged.
The review does not treat a backward derivative stencil as a universal local
error bound: endpoint samples can miss forcing, derivative reconstruction has an
alternating roundoff mode, and an unstable stiff mode can amplify a small
quadrature defect.

Independent source inspection confirmed first-eligible and periodic 32-step
audits, zero-estimate and positive-feedback guards, and actual two-half error
controlling audited acceptance/adaptation. An underestimated audit can activate
fallback on a rejected trial, but cannot advance physical or derivative history.
Recovery requires exactly 16 consecutive accepted audits with valid history and
the documented agreement criterion. Missing-history initialization earns no
credit. Every Newton/LTE rejection or disagreement resets the streak; the final
accepted recovery audit restarts periodic audit debt. BE/source resets start a
new segment without claiming a measured recovery. History commits occur only
after acceptance and the accepted-step resource check. The implementation author
subsequently reinspected these same transitions; that inspection is additional
to the separate review, not a second independent oracle.

[`emi02_history_lte_test.cc`](../../../cpp/tests/emi02_history_lte_test.cc) defines
truth without production stamps, expression derivatives or LTE helpers:

- Forced `y'=3t^2` uses independent full/half quadrature on unequal timestamps;
  its exact full-TRAP defect is `h^3/2`. Higher-order forcing checks convergence
  to its analytic primitive. The trace oracle accounts for the documented
  representable half-tail split before a hardpoint.
- Analytic RC ramps compare both estimator policies. A small differential RC
  state is checked at zero and 400 V common mode, with the known GMIN bias
  contribution explicitly compensated.
- A stiff RLC has separately computed continuous roots and an independent
  long-double two-state BE/TRAP recurrence for every accepted point. The source
  restart's roughly 100 ns fast mode is resolved at 12.5 ns and 6.25 ns maximum
  steps; the continuous current bound remains 20 nA. Earlier coarser fixtures
  exceeded that bound through restart truncation, so the fixture was refined
  rather than its tolerance relaxed.
- A piecewise nonlinear load crosses a branch and is compared with its
  independent analytic RC trajectory. Hardpoint/BE reset, rejected-trial
  rollback, fresh-invocation behavior, native-policy rejection and derivative
  magnitude failure are exercised separately.
- A growing-RC example distinguishes a history estimate below one from a
  full/two-half error above one and requires rejection through the feedback
  guard. A cubic pulse invisible at the coarse endpoints is paired with a flat
  control: its zero history estimate must trigger an audit, reject the coarse
  trial and recover its independent charge integral.

The separately authored prepared-transient suite reconstructs policy transitions
from trace diagnostics, including 15 versus 16 accepted agreements and interrupted
streaks. It also compares prepared and fully checked complete trajectories under
both estimator policies, source corners, failures and mutation of external input
objects. These are bounded heuristic safeguards and regression evidence; only
the complete pinned external workload can establish its requested qualification.

The per-attempt coordinate reuse was independently reviewed. Its start/full-state
slots are first populated through the original checked extraction during
derivative reconstruction. Estimation and feedback consume those same immutable
objects before any move, callback or observer. Private halves and later retries
never receive the slots. The unprepared path still recomputes coordinates, and
the existing exact parity/physical tests passed. The implementation author's
twelve paired records retained waveform bits and work counters with a small
favorable median timing change; this does not qualify full-workload performance.

## Rejected development experiments

BDF2 and BDF3 were separately governed development experiments. Independent
polynomial-interpolation and actual unequal-half forcing oracles derived their
respective audit multipliers, with nine BDF2 and ten BDF3 tests passing. Physical
coverage included analytic RC, stiff/growing modes, low-damping and parasitic RLC,
nonlinear charge, signed coupled inductors, source restart and rejection rollback.
Despite those focused correctness results, both methods failed the unchanged
full-workload resource gates. Both production paths, public selectors, dedicated
tests and experimental ADRs were removed. Their development results neither
qualify them nor describe the delivered method, which remains BE/TRAP.

A PI timestep controller and bounded cubic Newton predictor received independent
mathematical, transaction and fallback tests during development. Full private
workload timings still failed the unchanged resource gates, so both experiments
were rejected for delivery. Their production paths, dedicated tests and predictor
counter admission were removed; the original eleven physical-history tests were
restored. Earlier passing development tests do not qualify those policies or
mean they remain in the delivered implementation. ThinLTO and the zero/unit sparse
shortcuts were also rejected after unfavorable timing. Removing expression
value-array initialization was likewise reverted after unfavorable paired timing;
its additional lazy-branch regression remains, with the original initialization
and full-AD operand loading restored.

A BDF2 implicit-defect-response prototype was independently tested against scalar
stable/growing resolvents, a dense coupled nonlinear Jacobian inverse, algebraic
rows with zero charge forcing, retained/fresh Jacobian parity and failed-point
invalidation. A subnormal auxiliary RHS had to fail the original sparse guard
and force an actual audit. All seven development tests passed. Full workload
timings nevertheless failed the unchanged resource limit, so its production
policy, Jacobian-retention API, counters and dedicated tests were rejected and
removed. A separate proposal to audit every rejected raw-history estimate was
also withdrawn after unfavorable timing. These passing development oracles do
not qualify either rejected estimator policy.

## Focused validation inspected

The optimized Bazel test logs inspected during this review report:

| Target | Tests passing | Main role |
| --- | ---: | --- |
| `//cpp:emi02_expression_cache_test` | 12 | Cache boundaries, fresh final values, accepted AD proof, overflow and point parity |
| `//cpp:emi02_history_lte_test` | 11 | Independent physical and forcing oracles |
| `//cpp:emi02_prepared_transient_test` | 15 | Complete checked/prepared parity and policy transitions |
| `//cpp:emi02_solver_reuse_test` | 22 | Exact sparse reuse, correction, certification and typed-failure cases |
| `//cpp:emi02_newton_test` | 6 | Physical convergence, stable RHS, backtracking and row bounds |
| `//cpp:emi02b_test` | 22 | Expression semantics, scalar/full value parity and exact simple/generic differential evaluation |

The physical-oracle target was also run directly during its development. Other
listed final focused results were produced by the coordinating implementation
authors and verified from their local Bazel logs. The final accepted-AD change
also ran eight focused targets directly: expressions and the independent ngspice
fixture, cache, Newton, prepared transient, physical history and native Phase 3A/3B;
all passed. This note does not replace the
coordinator's final canonical build/sanitizer/acceptance record or assert that
all of those checks have finished. Rejected BDF/controller/implicit-defect
reviews also ran their development tests directly; the sparse shortcut and
certification reviews inspected the implementation author's final test log.

## Qualification identity and audit review

The resumed runner statistics use `emi02-cpu-v2`. The selected BE/TRAP method has
explicit `trapezoidal` integration identity and `derivative-history-audited-v1`
estimator identity; focused validation passed and full qualification remains
pending. The audit requires that identity and
checks step, rejection, estimator, fallback and KLU counters in addition to
actual raw schema/count/byte identity. Full job identities, process-success
requirements, unchanged waveform/spectrum/refinement metrics and failed-job
ranking suppression remain in place. The four-worker spawn policy remains a
concurrent qualification policy, not a fair-baseline timing claim.

Counter review also corrected an unsupported bound: iterative refinement solves
can exceed successful solves, because a call can perform four corrections and
still fail. The valid upper bound is four times the sum of numeric factorization,
refactorization and reuse counts. Conversely, selected error-estimator counters
plus Newton rejections need not equal attempts: a legal BE step with no
representable midpoint can have no local-error estimate. The audit retains that
one-way inequality instead of inventing a false accounting identity.

The runner now writes each already-validated raw record through one preallocated
contiguous vector. Review confirmed original observable order and FP64 byte
layout, finite/state/output-budget checks, post-write I/O checks and publication
behavior. The coordinator's eight process tests and eighteen metadata/audit tests
passed, and twelve paired records retained exact output and work counters. This
is an output-overhead optimization, with no change to model equations or gates.

The production fingerprint includes every `cpp/src/*.cc`, `cpp/src/*.h` and
public header, plus the runner/importer/qualification sources, build files,
governing ADRs, model provenance, frozen manifests and reference implementation.
Thus the new private prepared header and final-residual implementation are bound.
Matching Bazel data globs ship those inputs into runfiles. Binary/model hashes and
source identities are checked at qualification/audit boundaries. This review
document is commentary, not a runtime fingerprint input.

The resumed source changes must receive a new complete run and audit; a historical
`run-1` record cannot be validated as current-source evidence by relabeling its
metadata. No changes to `reference/emi01`, historical `run-1` files or the earlier
review were present in the source diff inspected for this note. External logs
continue to publish diagnostic identities rather than model-bearing messages.
No unresolved source-review blocker was found for the pending qualification.

## Final resumed scope inspection

A final read-only inspection compared the complete resumed working tree with
`ec167e9`, including the new private headers and regression files. This inspection
ran no new simulator jobs or CPU-heavy tests. The reviewer authored the expression
and earlier sparse-validator optimizations, so inspection of those changes is
author review; the prepared sparse-input admission change and its caller wiring
were independently authored and reviewed here.

No change was found in the frozen EMI-01 manifests, source generators, waveform,
spectral, refinement or ranking functions; model/dependency/toolchain pins; Rust
reference; or CUDA paths. Qualification still constructs all 30 CPU and 30
external jobs, checks each lane against its frozen identity, requires both
backends' refinement gates and every differential/classification gate, and
suppresses feasible ranking when any required job fails. CPU and wall limits
remain the original 110/120 seconds supplied from the frozen manifest. Runner
point/output limits and the public numerical tolerances were not loosened by the
resumed changes. Additional statistics identity/accounting checks do not replace
any physical comparison. A recorded processor-affinity environment is not a
portable runtime guarantee.

Production integration remains BE/TRAP. Public behavioral execution defaults to
full/two-half estimation, while the model runner explicitly selects the separately
governed audited derivative-history policy. The latter retains first/periodic
audits, zero-estimate and positive-feedback checks, rejected-trial rollback,
sixteen-accepted-audit fallback recovery and BE/source restart boundaries.
Native nonbehavioral execution retains its original estimator and recovery lane.
No BDF2/BDF3, PI controller, cubic predictor, implicit-defect response, bytecode,
value-scratch zeroing or experimental compiler option remains in production.
References to rejected methods are historical review notes or negative metadata
tests, not reachable numerical paths. Value scratch remains initialized.

The final private sparse-input admission patch skips only repeated finite-input
scans for a just-completed full prepared assembly whose canonical CSR exactly
matches the analyzed pattern. The full assembly has already bounded every
Jacobian value and affine-RHS intermediate; the separate rank check constructs a
fresh zero RHS. Pattern mismatches restore the original converter and numeric
admission, and RHS dimensions are always checked. Public/complex paths and all
KLU packing, factor/refactor, fallback, correction, result and backward-error
checks remain unchanged. The zero-root combined-Jacobian overflow discriminator
fails before an admitted solve and then verifies valid recovery against the
public path. No persistent caller-controlled validation bypass was found.

No concrete source/scope blocker was found in this inspection. Complete final
qualification, artifact audit and canonical validation are separate pending
gates; this paragraph makes no claim that they have passed.

## Final run-2 evidence closure

The completed [run-2 qualification](run-2/qualification.json) and
[artifact audit](audit-v2.json) close the pending evidence gates described in the
earlier chronological sections. Both report success. The separate audit
[execution record](audit-v2-execution.json) exits zero and its
[log](audit-v2.log) records recomputation of the retained artifacts. The final
[canonical record](validation-v2/results.json) passes all thirteen command
stages: 37 default and lockfile tests, 32 compatible tests in each sanitizer
configuration, five declared sanitizer exclusions, and the two explicitly
requested CUDA/sanitizer analysis failures.

A separate read-only accounting and provenance inspection reconstructed the
expected schedule directly from `reference/emi01/manifest-v2.json`, without
calling the production specification or reconciliation helpers. For each of
`q0`, `q1` and `q2`, the schedule contains one DPT job followed by the Cartesian
product of candidates `light`, `boundary`, `reference` and corners `nominal`,
`fast_low_lc`, `hot_high_c` in that order. Both engines contain exactly these
thirty ordered records and no extra job directories. Every candidate/corner
object, integration limit, observation spacing and reference-version field
matches the manifest, including the finer boundary/reference integration grids.
All sixty completion-log identities occur exactly once, agree with their final
records and report one simulator attempt per job. No selective retry or omitted
case was found.

Both independently reconstructed forty-check refinement lists and the thirty
ordered differential comparisons pass. The finest CPU fixture roles also agree
with the frozen manifest: `reference` passes every corner with a worst research
margin of 10.104156 dB; `boundary` passes every physical screen with a worst
margin of 5.516628 dB; and `light` is predicted infeasible at every corner. The
external results agree. Only `reference` enters the whole-candidate feasible
ranking. Each engine reports three qualified DPT jobs, fifteen individually
predicted-feasible inverter jobs and twelve predicted-infeasible inverter jobs;
none of these latter classifications conceals a simulator failure.

All 62 invocation source hashes match the final checkout. Independent file
hashing matches CPU binary
`b603210d9dd7357f323c62dcbeada75f24a9b79d4b7992d945f5a3747c9e8234`
and ngspice binary
`f655c9bb8689a0cd00c9b1c81a525ccde306bd0cb73fa7d672d658938865b47d`.
Archive/member sizes and hashes match; an independent in-memory replay of the
documented mechanical adaptation reproduces the adapted size/hash with the
expected 109/1/1 replacements. All recorded per-job file hashes match. Raw
headers, point counts, variable counts, payload sizes and identities agree;
CPU attempts minus rejections equals accepted points minus one in every job.
All 1,330 CPU and 1,050 reference numeric chunks are referenced and present,
with no extra chunks. The successful artifact audit supplies the complete raw
reconstruction and measurement recomputation; this separate inspection checked
accounting and identities without another simulator run.

Published files obey the engine-specific allowlists. All thirty external logs
contain only the prescribed diagnostic hash/byte redaction. Published circuit
files contain the original test topology and no vendor model declarations;
CPU outputs contain import identities and numeric observations, with no expanded
deck or model-bearing process log. No proprietary model text publication was
found.

The [invocation](run-2/invocation.json) and
[run execution record](run-2-execution.json) record four spawned workers and the
inherited logical CPU mask 4, 6, 20, 22 on four distinct physical cores.
The original 110 s CPU, 120 s wall, 1 GiB address-space, 512 MiB output and
two-million-point limits remain in force. CPU inverter process wall times span
33.519–107.241 s. Resource qualification is specific to this recorded environment;
it establishes neither a portable runtime guarantee nor a fair parallel-CPU
performance baseline. The results report `gpu_authorized=false`, retain the
model/research-mask limitations and preserve failed run-1 as historical evidence.
No remaining acceptance, accounting, provenance or publication blocker was found
for this bounded CPU qualification.
