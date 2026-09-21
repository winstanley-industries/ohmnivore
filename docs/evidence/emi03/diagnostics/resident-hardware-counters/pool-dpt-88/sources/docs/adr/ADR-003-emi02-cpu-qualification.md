# ADR-003: EMI-02 CPU semantics and differential qualification

- **Status:** Implementation contract, authorized by the EMI-02 request
- **Date:** 2026-09-17
- **Baseline:** EMI-01 v2 on `82b7a6e`
- **Authority:** Preserves ADR-001 and the frozen ADR-002 reference; implements the
  staged contracts in [CPU_GAPS.md](../../reference/emi01/CPU_GAPS.md).

EMI-02 proceeds through independently validated CPU slices. A slice's implementation does
not claim completion of later slices. KLU FP64 remains the production authority, ngspice 46
the external oracle, and Rust remains reference material. No CUDA execution, dispatch,
mixed precision, new device physics, optimization, or altered EMI-01 evidence is included.

## EMI-02A: Disjoint linear coupled-inductor pairs

### Syntax and identity

Admit exactly `Kname Lfirst Lsecond k`, with forward references permitted. K declarations
are separate ordered IR records (`Circuit.inductor_couplings`); they create no node or
branch unknown and do not change existing component order. Names and winding references
are ASCII case-insensitive for coupling resolution. Ambiguous case-insensitive component
names are rejected when resolving a referenced winding. Each winding must identify one
existing linear inductor, the two windings must differ, and a winding may occur in only
one pair. K identities match `[Kk][A-Za-z0-9_]+` and must be unique, including case variants.

Inductances are finite and positive. The coefficient is finite in inclusive
`[-0.999, 0.999]`. More windings, nonlinear cores, perfect coupling, loss/dispersion,
saturation, hysteresis, initial flux, transformer-ratio syntax and model parameters are
unsupported. A zero coefficient is allowed and still reserves both winding identities.

### Equations, storage, and validation

The existing positive terminal is the winding dot, with branch current entering it.
`M = k * sqrt(L1) * sqrt(L2)` and the physical constitutive matrix is
`[[L1,M],[M,L2]]`. Its stored energy is
`(L1*i1*i1 + 2*M*i1*i2 + L2*i2*i2)/2`. The two MNA C off-diagonal entries are `-M`;
existing diagonal entries remain `-L1` and `-L2`. Compute using bounded intermediates,
reject non-finite or unrepresentable nonzero mutual inductance, and verify the represented
pair is positive definite with a scale-safe normalized coupling check. Retain both symmetric
mutual coordinates even at k=0. Validate the entire coupling set before matrix stamping.

DC remains ideal-short behavior. AC uses the complete `G+j*omega*C`. BE/TRAP matrix and
history construction use the full C matrix, including both mutual entries. No per-winding
solve is introduced. UIC preserves zero current in both windings; ordinary initialization
uses the complete DC state. Source projection preserves both winding currents. Trial/LTE
state stays private until the existing atomic acceptance point. Rejected attempts cannot
commit physical history. Existing step limits, retry policy, tolerances and KLU validation
remain unchanged.

### Typed failures and precedence

Malformed card arity/numeric spelling returns `kParse`; recognizable multiwinding or
model-based forms and finite out-of-range k return `kUnsupported`. Non-finite coefficient
or referenced inductance, overflow, underflow of a nonzero mutual value, or loss of positive
definiteness by representation returns `kNonFinite`. Duplicate/invalid K identities,
unknown/non-inductor/ambiguous targets, self-pairs and repeated winding membership return
`kCompile`. Parser syntax checks precede coefficient checks. Compile first validates all
coupling coefficients (non-finite before range), then identities/references/membership,
then referenced inductances and mutual arithmetic, in declaration order within each pass.
All non-finite coefficients precede any coefficient range failure across the whole set.
Unrelated invalid-component precedence remains as before. Failed compilation returns no
partial MNA. Existing singularity, structure, convergence and solution-validation failures
retain their types.

### Acceptance

Independent exact-small tests construct the 2-by-2 physical equations without production
stamping/identity helpers, covering unequal L, k=0, +/-0.2, +/-0.995 and +/-0.999;
full G/C matrices; common/differential modes; signed-current energy grids; winding/sign
reversal; analytic AC; UIC/no-UIC trajectories and timestep refinement. Analytic scalar
and matrix tolerance is `1e-12 + 1e-10*abs(expected)` in SI units. Independently integrated
trajectory comparison uses `1e-6 A + 1e-3*abs(reference)` and
`1e-5 V + 1e-3*abs(reference)`.

Hostile tests exercise malformed/unsupported syntax, forward and reordered declarations,
duplicate/ambiguous/missing/self references, duplicate membership, simultaneous-invalid
precedence, NaN/infinity, endpoints, overflow/underflow, singular topologies, hard points,
bounded failures, repeatability and trial-state isolation. Hermetic ngspice tests cover
both signs and a linearized source-driven CM/DM filter plus coupled harness. This last
fixture is solely a coupled-element test, not a switching or performance benchmark.

Canonical lint/build/test, ASan, UBSan, lockfile, hermeticity, ngspice and CUDA smoke checks
remain required, including explicit CUDA/sanitizer incompatibility checks. Existing fixture
expectations and historical AC/reference evidence remain unchanged.

## Later EMI-02 slices

EMI-02B (bounded expressions), EMI-02C (controlled displacement current and transient state),
and EMI-02D (exact model import and complete inverter qualification) retain the detailed
requirements in CPU_GAPS.md. Before editing their production paths, append exact interface,
numerical, resource and failure contracts here. Completion requires independent derivative,
dynamic-state and model-provenance checks followed by all frozen DPT and nine-job v2
differential/refinement gates. A coupled-inductor implementation alone does not enable the
EMI-01 model or qualify the inverter.
