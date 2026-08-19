# GPU-02S persistent-session linear-AC evidence summary — 2026-08-16, review-refresh 2026-08-19

GPU-02S is a bounded evidence-only follow-up to GPU-02. It tests the current Ohmnivore use case of
large, repeated, compiler-derived linear-AC sweeps in a long-lived process. It is not a customer
workload, production speedup claim, acceptance-policy change, or automatic-dispatch authorization.
The frozen GPU-02 replay-v1 evidence and verdict remain unchanged.

Each invocation contains three warmup sessions and twenty recorded fresh-child sessions for every
case/backend/lane, plus one cold and twenty steady sessions in one persistent process for every
case/backend/lane. Both CPU and CUDA retain their owners across persistent sessions. CPU workers
retain per-worker KLU symbolic analysis. CUDA retains its primary context, stream, cuDSS objects,
allocations, canonical structure, and analysis while refreshing only complete complex-FP64 values
and right-hand sides.

Every recorded result passed the complete prepared envelope, association, dimension, finiteness,
residual/backward-error, fresh-KLU, and componentwise differential checks. The
`candidate_runtime` interval excludes fresh KLU certification only from timing; certification still
runs for every result and any failure invalidates the stream.

## Preserved evidence

| Invocation | UTC interval | File SHA-256 | Completion fingerprint |
|---|---|---|---|
| 1 | 2026-08-19T06:23:53Z to 2026-08-19T07:17:27Z | `2b6d4639933925c4e4569b608000ed9e8475e2049d53a72e57f26afeca212ecd` | `v1-ed8f9d3513c8e8fe4dab9c51d2bc44c1` |
| 2 | 2026-08-19T07:17:42Z to 2026-08-19T08:11:16Z | `406f77a3db052a46301129af3bbc20c0a5d01240ddff3d68cb99ff9efb978724` | `v1-824f0a6b3566c60917278d020942c7a4` |

Both files have 1,840 CSV records and bind all 35 declared source inputs: 320 unique fresh sessions,
1,280 fresh prefix samples, 336 persistent samples, all sixteen corner identities, and zero failures
or outstanding device bytes.
An independent audit parsed every row against its record header, reconciled every timing sum and
CPU/CUDA solve count, recomputed both terminal fingerprints, and verified identical source, binary,
manifest, structure, batch, and aggregate-member identities across invocations.

- Source fingerprint: `v1-3fd5bc5a942844e720117d9a43d50c6c`
- Binary fingerprint: `v1-ae9c40b2e86de2649b7f3958533d3839`
- Manifest fingerprint: `v1-de52b10506655ec327cab7e010fbe106`

## Compiler-derived workloads

| Class / case | Topology | Dimension / nnz | Frequencies per corner | Corners | Accepted solves per session | Eligibility |
|---|---|---:|---:|---:|---:|---|
| `session_control` / `tree_session_control` | binary tree | 257 / 768 | 64 | 4 | 256 | permanently ineligible |
| `session_grid_medium` / `grid_33_session` | 33-by-33 grid | 1,090 / 5,315 | 512 | 4 | 2,048 | eligible experiment |
| `session_grid_large` / `grid_65_session` | 65-by-65 grid | 4,226 / 20,867 | 256 | 4 | 1,024 | eligible experiment |
| `session_wide` / `ring_1024_session` | four-source ring | 1,028 / 3,080 | 2,048 | 4 | 8,192 | eligible experiment |

Each corner is constructed through public `Circuit`, `CompileMna`, and `PrepareLinearAcBatch`
interfaces. Component and source values change at every corner, while dimensions, canonical sparse
structure, and uniform-batch member count remain fixed.

## Fresh-session verdict

The technical speed bounds are `parallel_cpu / gpu >= 1.25` at the median and `>= 1.10` at P95.
Peak CUDA batch memory must be at most 2,147,483,648 bytes. Ratios below are exact final-prefix
session values; a fresh CUDA child includes primary-context initialization once.

| Class | Run | Lane | Median ratio | P95 ratio | Peak CUDA bytes | Verdict |
|---|---:|---|---:|---:|---:|---|
| `session_control` | 1 | inline certified | 0.18315041209913496 fail | 0.18345335539537241 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | inline certified | 0.18571008175169210 fail | 0.18697163107516979 fail | 33,554,432 pass | ineligible |
| `session_control` | 1 | candidate runtime | 0.13532433291573737 fail | 0.13220552382896231 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | candidate runtime | 0.13784923886452369 fail | 0.13873929489513234 fail | 33,554,432 pass | ineligible |
| `session_grid_medium` | 1 | inline certified | 0.90573984123588058 fail | 0.90525160098733415 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | inline certified | 0.90824429868829881 fail | 0.90156513697176399 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 1 | candidate runtime | 0.85246713051686429 fail | 0.85623668521281338 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | candidate runtime | 0.85059416614628935 fail | 0.84784522974184440 fail | 335,544,320 pass | fail |
| `session_grid_large` | 1 | inline certified | 0.94475026180635413 fail | 0.94196159516515332 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | inline certified | 0.94523710740839650 fail | 0.94635777052440850 fail | 738,197,504 pass | fail |
| `session_grid_large` | 1 | candidate runtime | 0.89765534447069817 fail | 0.90439032654183160 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | candidate runtime | 0.89379977898941754 fail | 0.90250130739775558 fail | 738,197,504 pass | fail |
| `session_wide` | 1 | inline certified | 0.93111013848810054 fail | 0.93326200861291020 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | inline certified | 0.92440273317226707 fail | 0.92297664819347536 fail | 469,762,048 pass | fail |
| `session_wide` | 1 | candidate runtime | 0.89303843992438192 fail | 0.89151617881761236 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | candidate runtime | 0.88151653634106120 fail | 0.88176510272763264 fail | 469,762,048 pass | fail |

## Persistent steady-state verdict

These ratios use only the twenty steady sessions after the cold session in the same process. Every
steady CUDA sample records zero context setup, zero library setup, zero structure upload, and zero
analysis; every corner still refreshes complete values/RHS and performs one uniform-batch refactor
and solve. This is the answer that excludes one-time CUDA initialization from the repeated-use case.

| Class | Run | Lane | Median ratio | P95 ratio | Peak CUDA bytes | Verdict |
|---|---:|---|---:|---:|---:|---|
| `session_control` | 1 | inline certified | 0.82737715443452353 fail | 0.83113321722702715 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | inline certified | 0.82340886528362267 fail | 0.81167822645396104 fail | 33,554,432 pass | ineligible |
| `session_control` | 1 | candidate runtime | 0.75924680445403903 fail | 0.76153951931427610 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | candidate runtime | 0.76148505812841039 fail | 0.77147967419576946 fail | 33,554,432 pass | ineligible |
| `session_grid_medium` | 1 | inline certified | 0.96409062381500443 fail | 0.94259154415886326 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | inline certified | 0.96290474181251240 fail | 0.94724104553722988 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 1 | candidate runtime | 0.92769847401166450 fail | 0.87045250965922560 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | candidate runtime | 0.93287779499767831 fail | 0.90151680012582969 fail | 335,544,320 pass | fail |
| `session_grid_large` | 1 | inline certified | 0.97689530987625850 fail | 0.98068027294187088 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | inline certified | 0.97846909315783293 fail | 0.97717411388030229 fail | 738,197,504 pass | fail |
| `session_grid_large` | 1 | candidate runtime | 0.90468916146525169 fail | 0.88107435708991089 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | candidate runtime | 0.91000731318737182 fail | 0.88563114645018270 fail | 738,197,504 pass | fail |
| `session_wide` | 1 | inline certified | 0.94626598884260493 fail | 0.92872170426487566 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | inline certified | 0.93979214147084844 fail | 0.92348769039481815 fail | 469,762,048 pass | fail |
| `session_wide` | 1 | candidate runtime | 0.90214725740156576 fail | 0.89989175760292395 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | candidate runtime | 0.89254685775644615 fail | 0.88846662635110074 fail | 469,762,048 pass | fail |

No eligible class passes any corner prefix in either lane or invocation. The control remains
permanently ineligible. Every memory bound passes. Automatic dispatch remains unauthorized.

## Why persistent CUDA is still slower

The table decomposes nearest-rank medians for persistent `candidate_runtime` steady sessions.
Times are milliseconds. `GPU device` is overlapping CUDA-event telemetry and is not added again to
the host interval. `Ideal zero-GPU ratio` is a deliberately impossible upper bound:
`CPU total / median(GPU preparation + residual validation)`, assuming all GPU execution takes zero.

| Case | Run | CPU total | CPU execute | GPU total | GPU execute | GPU pack | GPU upload | GPU factor/solve host | GPU factor/solve device | GPU execute remainder | Ideal zero-GPU ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 31.620 | 6.838 | 41.647 | 17.227 | 0.745 | 0.790 | 11.307 | 11.111 | 3.343 | 1.295750 |
| control | 2 | 31.711 | 6.752 | 41.643 | 17.215 | 0.748 | 0.772 | 11.333 | 11.142 | 3.362 | 1.296578 |
| 33-by-33 grid | 1 | 1,439.836 | 265.487 | 1,552.052 | 381.987 | 37.800 | 27.669 | 145.649 | 145.474 | 161.790 | 1.235284 |
| 33-by-33 grid | 2 | 1,446.002 | 264.221 | 1,550.044 | 381.830 | 37.699 | 27.149 | 145.467 | 145.264 | 161.431 | 1.241455 |
| 65-by-65 grid | 1 | 2,889.365 | 622.949 | 3,193.765 | 917.174 | 74.554 | 53.054 | 468.888 | 468.672 | 310.402 | 1.269649 |
| 65-by-65 grid | 2 | 2,903.844 | 631.460 | 3,191.012 | 923.720 | 73.911 | 53.007 | 475.044 | 474.847 | 309.108 | 1.281473 |
| four-source ring | 1 | 3,653.683 | 587.644 | 4,049.985 | 999.365 | 98.396 | 97.275 | 360.452 | 360.221 | 422.878 | 1.198161 |
| four-source ring | 2 | 3,628.072 | 586.307 | 4,064.853 | 1,021.198 | 98.012 | 97.031 | 382.579 | 382.351 | 423.190 | 1.192493 |

Primary-context initialization is real but not the steady-state blocker. Its fresh-session median is
roughly 191--204 ms across these cases; it is exactly zero in every persistent steady sample.
Removing it improves the eligible inline-certified median ratios to about 0.94--0.98, but CUDA is
still 2--6% slower. The more diagnostic candidate-runtime lane is 7--12% slower.

Three costs explain the result:

1. Compiler preparation plus authoritative residual validation is a large common floor. For the
   medium grid and ring, a zero-time GPU execution cannot reproduce the 1.25 median bound in either
   run. The large grid can reach it only theoretically, with almost no budget left for execution.
2. Native cuDSS batching removes repeated analysis and reduces library calls, but changed corners
   still require host packing and values/RHS upload. Those total about 65 ms for the medium grid,
   127 ms for the large grid, and 195 ms for the ring.
3. The CUDA executor remainder grows with batch bytes. It includes validated-envelope hashing and
   generation work not hidden under CUDA phase labels. Together with device factor/solve, it makes
   the complete CUDA execute interval slower than the parallel KLU execute interval.

This is consistent with NVIDIA's guidance: uniform batching and `REFACTORIZATION` are the intended
same-pattern mechanisms, while sparse-factor performance depends strongly on matrix/factor
structure. Hybrid execution is not supported when uniform batching is enabled or `batchCount > 1`.
Pinned host memory can improve transfers, and nested-dissection levels can materially change later
factor/solve performance, but neither is part of the fixed GPU-02 algorithm contract. See NVIDIA's
[cuDSS tips and tricks](https://docs.nvidia.com/cuda/archive/13.1.1/cudss/doc_output/tips_and_tricks.html),
[advanced features](https://docs.nvidia.com/cuda/cudss/advanced_features.html),
[uniform-batch configuration](https://docs.nvidia.com/cuda/cudss/types.html), and
[page-locked memory guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).

## Decision

GPU-02S succeeds as a correct, reproducible answer for the exact current synthetic session
envelope: CUDA primary-context startup is not why the persistent gate fails, and the present
FP64-cuDSS path does not accelerate these repeated linear-AC workloads end to end. A future bounded
experiment may separately test pinned compiler-to-device buffers, validated immutable-preparation
tokens that avoid repeated hashing, cuDSS ordering/factorization tuning, or a versioned corpus of
representative circuit matrices. Those changes require their own contract and evidence; none is
authorized here.

CPU KLU remains the correctness authority, supported no-GPU implementation, and explicit full-batch
fallback. Ordinary `SimulateAc` and CSV output remain unchanged. NL-04, mixed precision, automatic
dispatch, nonlinear/transient CUDA work, and downstream GPU work have not started.
