# Auxiliary-equation decomposition screens

These are **unqualified diagnostics**. EMI-03's complete-study gates remain
unpassed. Inputs are the 144 accepted-state matrices and independent KLU
solutions retained in `../schur-decomposition/`; they cover nine shortened
20 us CPU trajectories, not all transient Newton trials or qualification cases.

Forty isolated grounded voltage drivers form an acyclic algebraic subsystem.
Their 80 voltage/current variables can be eliminated from each linear system,
leaving 105 variables and 378 structural entries instead of 185 and 598.
Four dependency levels propagate 92 response coefficients without expanding
expression trees. Every original voltage and driver current is reconstructed.
In particular, the current is `b_aux - GMIN * v_aux`, not zero.

Variant 122 constructs the reduced matrix and RHS on the GPU, uses the selected
native-FP64 factor/triangular routines for the core, and applies up to four
corrections from the original 185-variable compensated residual. A first nonzero
residual correction is mandatory. Host checks use the canonical full original
validator and the captured independent KLU solution. All 288 manufactured/zero
RHS checks pass, and an actually singular reduced Jacobian with zero RHS is
rejected after preparing a valid symbolic plan. All four Compute Sanitizer tools
pass on the entire validation invocation, including preparation kernels.

| Nine-matrix batch, device-event median | Time |
| --- | ---: |
| Selected full solver replay (119b) | 106.455 us |
| Auxiliary elimination, COLAMD (122) | 85.111 us |
| Warp-only factor levels (123) | 86.254 us |
| Repeated COLAMD screen (127) | 85.527 us |
| AMD ordering (127) | 74.532 us |
| Interior-first ordering (127) | 93.426 us |

Each timing uses one warmup and nine measurements of 100 repeated matrices for
each batch size 1, 9 and 16. The original full residual, correction, auxiliary
reduction and recovery are inside the auxiliary timings; preparation, transfer,
CPU certification and transient work are outside. All 26,000 warmup/measured
solves per variant use fresh numeric factors and one correction, with no dense
fallbacks. These matrix timings do not establish an end-to-end improvement. The
sanitizer evidence for 122 must not be presented as coverage of later variants.

The further four-block Schur screens (126 and 128) operate on a CPU-prepared
105-variable replay. They optimistically reuse the factors and response columns
of eleven structurally linear interiors during repeated-matrix iterations;
four nonlinear interiors and the interface are refactored. They exclude the
auxiliary elimination and original-state recovery cost, yet still take about
102–105 us for nine matrices. They are rejected. Variant 128 also reuses the
interface pivot ordering with a bounded fresh-pivot retry. Neither implements a
changing-timestep cache or a transient executor. Their residual checks cover
the reduced matrix only, unlike 122/123/127.

An independent structural analysis (125) confines every possible nonlinear
entry of the reduced Jacobian to a 25-variable interface, leaving an 80-variable
linear interior. One ideal-source variable must be promoted to make the interior
structurally nonsingular. This is a different partition from 126/128. Its offline
NumPy proof passes all 144 full original residual and KLU checks with one
correction, maximum scaled KLU difference `2.419e-13`. No GPU implementation or
timing of this 25-variable interface is claimed.

## Transient projection experiment

Variant 129 also evaluates admitted auxiliary drivers in dependency order at
Newton trial states, reconstructing their GMIN currents before the unchanged
original residual/update/Jacobian checks. Admission is derived from the compiled
stamps and expression graph, including inactive dependencies; reactive coupling,
external current use, extra source stamps and cycles prevent projection. The
185-variable linear solve and CPU implementation remain intact in this variant.

The real 1.03 us reference/nominal screen rejects it: ordinary median request
time is 2.540698 s versus 1.979659 s for selected checkpoint 100, **28.3% slower**.
All ten GPU waveform comparisons against a fresh CPU trajectory pass. One warmup,
three alternating measurements and a separate phase run are retained for each
GPU implementation; phase runs are excluded from medians. The added expression
work outweighs the reduction in factor/solve work. This is not full qualification.
All eleven resident tests pass on the final experimental source. The experiment
is removed from the selected solver.

`projection129/` retains all eleven compressed raw trajectories, source identities,
successful and failed test logs, the exact experimental source and its patch.
Its audit verifies every raw trajectory hash. An initial workspace-layout edit
put new fields before `Progress`, violating the existing prefix readback protocol
and failing ten tests. The corrected layout retains `Progress` at offset zero
and adds a compile-time assertion in the experimental source. Both versions and
the failure remain visible; failed-test timings are not performance evidence.

## Failures and reproducibility

The initial structural script 124 incorrectly treated `(row, coefficient)`
records as scalar row indices, omitting four nonlinear stamp rows. Its
20/21-variable interface is invalid as a claim about constant interiors.
Version 124b fixes the parsing; version 125 checks confinement explicitly and
includes full original-state residual correction. The initial 124 solve also
had three KLU mismatches before correction. Those files remain as rejected
evidence, not qualification.

Exact diagnostic sources, generated structural recipes, replay-generation
scripts, all timing rows, build logs and failures are retained here. Sources and
large replay files are losslessly compressed to avoid adding historical Bazel
packages or lint inputs. `auxiliary122-BUILD.bazel.gz` records the temporary
manual target used to compile the archived probe; it is not a second production
solver. Use the pinned Bazel toolchains. `manifest.json` records artifact hashes;
the linked Schur archive contains the original input decks and matrix provenance.
