# Rejected factor, power and auxiliary-ordering screens

None of the four candidates is selected. EMI-03 remains incomplete: these shortened
1.03 us reference/nominal runs do not qualify the complete workload or its
performance gates. All four candidates start from selected checkpoint 100 at
`76148cc5576499c94f63f457f1726526fbc43b9f`.

| Candidate | Selected 100 median | Candidate median | Decision |
| --- | ---: | ---: | --- |
| 102: prefetch four immutable factor descriptors per thread | 1.972003435 s | 1.980660640 s | Rejected, 0.4% slower |
| 103: native FP64 products/square root for common power exponents | 1.962360259 s | 1.974234172 s | Rejected, 0.6% slower |
| 104: auxiliary drivers first, dependency order | 1.974157653 s | 2.031140642 s | Rejected, 2.9% slower |
| 105: auxiliary drivers first, reverse dependency order | 1.967633362 s | 2.054164711 s | Rejected, 4.4% slower |

Each screen has a fresh CPU trajectory, one GPU warmup per implementation,
three alternating ordinary measurements per implementation and one separate
phase-instrumented run per implementation. The phase runs are excluded from
medians. All 40 GPU waveform comparisons pass. The audit reconstructs all 44
raw trajectories, all comparisons, four source archives and four medians.
These are request timings, not complete-study measurements. No DPT, full
coupled, spectral, refinement or performance-acceptance claim is made for any
candidate. Variant 103's original freezer description is generic; its separate
scope record and exact source patch identify the actual power experiment.

All 11 resident tests pass for each candidate and for selected checkpoint 100
with the added dense resistor-coupling regression. That regression exceeds
1,024 sparse factor entries and compares every output against an analytic ramp-RC
solution. It is retained in the implementation even though all four optimizations
were removed.

The new regression passes all-launch memcheck and first-resident-chunk racecheck
on selected 100. The first memcheck attempt returned 99 for a CUDA selective-code-
recompilation warning, despite the test passing; its log and failed terminal
record are retained. An identical-command retry with the same binary and sources
reports zero errors. There are no warning suppressions. Race coverage is limited
to the recorded first chunk.

The ordering candidates preserve all 185 variables, original equations, numeric
GPU pivot selection and numerical guards. Both increase the dependency depth:
the profiled forward/backward/factor levels grow from 14/27/34 to 24/31/44
(forward) and 24/32/44 (reverse). Their exact phase logs remain diagnostic only.
Neither implements the algebraic reduction discussed below.

## Compiled auxiliary-equation probe

A separate read-only Bazel-built probe examines the exact compiled MNA system
and exported expression trees. All 40 grounded behavioral voltage drivers have
only their source and GMIN linear stamps at the auxiliary output, no reactive
coupling, no branch-current dependencies and an acyclic voltage dependency graph.
Eliminating both variables per driver would reduce 185 unknowns to 105, but no
such reduction is implemented or validated here. Each original auxiliary branch
current is `-1e-12 * v(auxiliary)`, not zero.

Naively expanding all original expression trees would grow 892 nodes to 5,464
(maximum 304 nodes per program, depth 15). Original eager domain checks, full
state/current reconstruction, original residual/update checks and nonsingularity
certification would still be required. The structural result is an opportunity
assessment, not a proof that a numerical implementation is equivalent or faster.
The exact input, compiled matrix/tree dump, probe source/build patch, analysis
script and reconstruction are retained under `structural-probe/`.

`audit.json` certifies this limited evidence only. Full EMI-03 acceptance is false.
Historical source snapshots and scripts are compressed to avoid creating Bazel
packages or lint inputs inside evidence directories.
