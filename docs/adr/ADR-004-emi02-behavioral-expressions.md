# ADR-004: EMI-02B bounded behavioral expressions

- **Status:** Accepted for the explicitly authorized EMI-02 CPU implementation
- **Date:** 2026-09-17
- **Authority:** ADR-001 and the pinned EMI-01 model/provenance; no GPU authorization

## Contract frozen before implementation

Compile immutable, case-insensitive expressions after binding every parameter, node voltage
and voltage-source current. The only admitted vocabulary is finite engineering literals
(T/G/MEG/K/M/U/N/P/F), identifiers, parentheses/braces, unary signs, `+ - * / **`, `exp`,
`<`/`>`, three-argument `if`, `v(node[,node])`, and `i(source)`. Ground is `0`/`GND`.
Current bindings use the source's existing positive-to-negative branch orientation. All
names, including names in unselected branches, must bind before evaluation. Unknown functions,
operators and source kinds fail closed. No arbitrary subcircuit/function language is added.

Each expression is at most 512 syntax nodes, nesting/tree depth 64 and 65,536 input bytes.
The caller enforces a 16,384-node aggregate circuit budget using the exposed node count.
An expression binds at most 512 state unknowns. Compiled expressions expose sorted unique
structural dependencies, including both conditional arms, so the MNA pattern stays fixed.
Literals, bound parameters/state, evaluated value intermediates, reverse derivative
intermediates and accumulated derivatives have magnitude at most `1e100`, as frozen by
ADR-005. A result cannot hide an over-bound intermediate by subsequently cancelling it.
Evaluation returns a finite value and sorted sparse derivatives with respect to those state
unknowns. Conditions return 0 or 1 with zero derivatives; strict equality takes the false
arm, and only the selected arm is evaluated. Rejected evaluation produces no partial result.

The behavioral dialect follows the audited ngspice 46 `ps` runtime: powers have absolute
base; division adds `+1e-32` to a nonnegative denominator (including signed zero) and `-1e-32`
to a negative denominator. This is the pinned `gmin=1e-12` times `1e-20` offset. `exp(x)`
for x>14 is **the reference dialect's** linear continuation `1202604.284*(x-13)`; for x<=14
it is mathematical exp. The derivative above 14 is 1202604.284. This explicit source-audited
dialect behavior is not an added current limiter or an approximation of the vendor equation.
The same operator behavior applies to constant subexpressions after the actual included-library
PSpice preprocessing: the independent executable fixture verifies `(-2)**3` is +8, `1/0`
is 1e32, and `exp(20)` uses the linear continuation. Directly inferring constant-folding
behavior from `inpptree.c` without this frontend would give the wrong executable dialect.

Derivatives are analytic derivatives of the admitted value operation on its selected smooth
branch. In particular division differentiates the actual offset denominator. Upstream
ngspice instead constructs a symbolic quotient derivative whose denominator is `b*b+1e-32`;
the two are measurably different only near the artificial offset. This implementation does
not claim bitwise agreement with that upstream Jacobian regularization. Tests freeze both
value semantics and the exact local derivative here. At an absolute-power origin, exponent
>1 yields derivative zero; exponent 1 takes the nonnegative one-sided derivative and exponent
0 yields zero. A dependent base with 0<exponent<1 has no finite derivative and fails.

Parameter expressions use a separate dialect: ordinary division, absolute-base `**`, and
mathematical exp. They cannot reference state or use `if` (the selected parameter graph has
no conditionals, and general numparam conditional semantics are outside this port).
Parameter definitions permit forward
dependencies, are case-insensitive, and reject duplicate names and cycles. Instance override
right-hand sides are evaluated in caller scope before being installed in callee scope;
`MDE=MDE` therefore forwards the caller value. Overrides must name declared parameters.
Local definitions shadow caller values. Dependency resolution is bounded to 512 parameters,
64 dependency levels and 16,384 syntax nodes including override expressions.
The generic resolver gives no identifier an implicit simulator meaning. The importer must
explicitly lower the audited ngspice frontend reserved `vt` collision to its ambient-27-C
thermal-voltage constant; that reference compatibility behavior is independently tested
by `//acceptance:emi02b_expression_test` and documented with the model import.

Failure precedence is input/resource and supplied-binding shape (including parameter
finiteness), then the first syntax/vocabulary/binding failure in left-to-right parsing, then
evaluation. For example an unknown first identifier precedes a later missing operand;
a missing first operand precedes a later unknown identifier. Malformed syntax is `kParse`;
unknown vocabulary or names, cycles, unsupported
state access in parameter expressions and invalid override names are `kUnsupported`;
duplicate definitions/bindings and invalid state indices are `kCompile`; resource excess is
`kUnsupportedSize`; nonfinite input, domain/nonfinite results or nonfinite derivatives are
`kNonFinite`. Existing public error types are reused without silently replacing a value.

## Acceptance

Independently authored tests cover every operator and analytic Jacobian, strict equality,
signs, lazy domain traps, voltage difference and current cross derivatives, forward/caller
scope resolution, cycles, resource and malformed-input failures. Finite differences and
complex-step checks apply only on smooth branches, with explicit one-sided checks at absolute
value/condition boundaries. The complete selected-model DC sweep and original-residual
validation gates belong to the integrated EMI-02C/D acceptance, not this stateless library.
No vendor expression text is embedded or redistributed. Rust, ordinary existing simulator
paths, AC evidence, CUDA and GPU dispatch remain unchanged by this slice.
