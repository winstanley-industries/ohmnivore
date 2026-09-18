# Generic transient accuracy and work regression corpus

The corpus checks changes across 21 physical circuits rather than selecting an
inverter that happens to benefit. Fixtures and continuous closed-form reference
equations live in `cpp/tests/transient_accuracy_cases.cc`, outside production code.
The oracles do not call production matrix, integration or LTE helpers.

| Family | Cases | Variation |
|---|---:|---|
| Bipolar RC | 6 | Signal amplitude 0.1 mV to 100 V, time scale 1 ns to 1 s, 400 V common-mode bias |
| Bipolar RLC | 12 | Underdamped, lightly damped and stiff overdamped responses; amplitude, time scale and common-mode bias |
| Nonlinear charge storage | 3 | Independent cubic charge primitive; positive/negative current and voltage, amplitudes 1 mV to 100 V |

Each circuit runs with both full/two-half step doubling and audited derivative
history. The benchmark records every case at coarse input resolution and with
a 32-times-smaller maximum step. The fine lane must satisfy the original analytic
voltage/current limits. Those limits were chosen before the endpoint-scaling
experiment; they were not widened after observing errors. The independent physical
oracles' absolute budgets also cover the fixed 1e-12 S numerical GMIN shunts.

Coarse cases deliberately retain failures of those global-error limits. The
baseline already fails some coarse cases: local-error tolerance is not a bound
on accumulated phase or charge error. These rows remain in the CSV with
`pass=0`; they are diagnostic stress cases, not validated performance results.
Both policies use explicit minimum-step divisor 1,000,000 and Newton limit 100
to admit the signal dynamic range. An earlier run with the native default
minimum-step bound exhausted that bound; it is retained as negative evidence.
Neither setting changes the error tolerances or production defaults.

The regular Bazel test additionally compares every coarse prepared trajectory
against the fully checked point-by-point execution: all accepted state bits,
timestamps, local-error values, decisions, audit/fallback state and sparse-solver
work counts must agree. Existing growing-mode, hidden-pulse, source-corner,
nonlinear charge, isolation, hostile-input and native-device tests remain gates.

```sh
bazel test //cpp:transient_accuracy_test //cpp:emi02_prepared_transient_test
bazel run -c opt //cpp:transient_accuracy_benchmark -- 11 > /absolute/corpus.csv
```

The optional argument is the number of repetitions, 1 through 20 (default 3).
Policy 0 is full step doubling; policy 1 is audited derivative history.
Refinement 0 is coarse and 1 is the 32-times-finer lane. Every repetition includes
parse, compile, integration and independent accuracy measurement; build time is
excluded. Exit zero requires complete execution and all fine-lane accuracy checks;
inspect the per-row `pass` column for coarse failures. This is a test benchmark,
not a production simulator entry point.

For comparisons, use identical fixture/benchmark source and hermetic optimized
builds for both solver revisions. Record binary/source hashes, bind the process
to the same CPU, alternate baseline/candidate order, discard declared warmups,
and report per-case medians and ranges. Preserve all cases and failed runs;
never infer speedup from reduced internal operation counts. Measurements near
the timing spread do not support a performance claim.

The endpoint-scale experiment is rejected, and there is no production tolerance
change. The retained optimization only removes repeated writes/calculations of
immutable companion entries. The scale-cache variant was rejected after a
measured corpus regression. The retained plan uses no extra cache or per-call
bookkeeping. Its mathematical invariant and failure behavior are specified in
[ADR-007](adr/ADR-007-generic-transient-improvements.md).
