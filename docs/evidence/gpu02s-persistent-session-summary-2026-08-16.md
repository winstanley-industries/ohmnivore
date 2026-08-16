# GPU-02S persistent-session linear-AC evidence summary — 2026-08-16

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
| 1 | 2026-08-16T16:46:18Z to 2026-08-16T17:39:41Z | `58690cdd01288ccf7b05474da620fa3dcbfda0256434c702309f356a9c913fe0` | `v1-8f81e311c1dc0173f0a58e6f4b3c8bd6` |
| 2 | 2026-08-16T17:40:24Z to 2026-08-16T18:33:49Z | `0cc877199ce637531064a80089782bbb4607d46dfd4a00a8037ae1d5f42f0365` | `v1-9a55c9dae654cbdd9bddd251abf1afd0` |

Both files have 1,840 CSV records: 320 unique fresh sessions, 1,280 fresh prefix samples, 336
persistent samples, all sixteen corner identities, and zero failures or outstanding device bytes.
An independent audit parsed every row against its record header, reconciled every timing sum and
CPU/CUDA solve count, recomputed both terminal fingerprints, and verified identical source, binary,
manifest, structure, batch, and aggregate-member identities across invocations.

- Source fingerprint: `v1-d458c4fbe84bdfae28142ee8c74494d5`
- Binary fingerprint: `v1-528a81fd64bfcbabe807b3cc0582d1c6`
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
| `session_control` | 1 | inline certified | 0.18963009591496091 fail | 0.1892701307139909 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | inline certified | 0.19000128150350606 fail | 0.19008685270780493 fail | 33,554,432 pass | ineligible |
| `session_control` | 1 | candidate runtime | 0.13822028587905585 fail | 0.13817854107803046 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | candidate runtime | 0.13807220575684681 fail | 0.13975997734407974 fail | 33,554,432 pass | ineligible |
| `session_grid_medium` | 1 | inline certified | 0.91435329173174562 fail | 0.90814720127195259 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | inline certified | 0.90142159876095163 fail | 0.89136261580857812 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 1 | candidate runtime | 0.85087078859605947 fail | 0.84792736889358744 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | candidate runtime | 0.84618024558726401 fail | 0.84420241614871627 fail | 335,544,320 pass | fail |
| `session_grid_large` | 1 | inline certified | 0.95348370004190586 fail | 0.9540808830756502 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | inline certified | 0.94433078723703534 fail | 0.9382453861540756 fail | 738,197,504 pass | fail |
| `session_grid_large` | 1 | candidate runtime | 0.89814604723031077 fail | 0.90145433390691188 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | candidate runtime | 0.88824464899518929 fail | 0.8889503173720763 fail | 738,197,504 pass | fail |
| `session_wide` | 1 | inline certified | 0.92614216266837524 fail | 0.91882926773807716 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | inline certified | 0.92851510962348871 fail | 0.92869749730830931 fail | 469,762,048 pass | fail |
| `session_wide` | 1 | candidate runtime | 0.88406505023580861 fail | 0.88419427041710696 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | candidate runtime | 0.88885880923020999 fail | 0.88285159803049196 fail | 469,762,048 pass | fail |

## Persistent steady-state verdict

These ratios use only the twenty steady sessions after the cold session in the same process. Every
steady CUDA sample records zero context setup, zero library setup, zero structure upload, and zero
analysis; every corner still refreshes complete values/RHS and performs one uniform-batch refactor
and solve. This is the answer that excludes one-time CUDA initialization from the repeated-use case.

| Class | Run | Lane | Median ratio | P95 ratio | Peak CUDA bytes | Verdict |
|---|---:|---|---:|---:|---:|---|
| `session_control` | 1 | inline certified | 0.82524487046131711 fail | 0.82086804076688791 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | inline certified | 0.83278890190152832 fail | 0.82224171494911413 fail | 33,554,432 pass | ineligible |
| `session_control` | 1 | candidate runtime | 0.76427949528092631 fail | 0.75806238796889358 fail | 33,554,432 pass | ineligible |
| `session_control` | 2 | candidate runtime | 0.76257178984595519 fail | 0.76150073837545418 fail | 33,554,432 pass | ineligible |
| `session_grid_medium` | 1 | inline certified | 0.96648345059233476 fail | 0.95619064075858884 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | inline certified | 0.96563474631295698 fail | 0.94843743340170261 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 1 | candidate runtime | 0.94043686765494305 fail | 0.8739224265004758 fail | 335,544,320 pass | fail |
| `session_grid_medium` | 2 | candidate runtime | 0.93896025350220025 fail | 0.87668889721925192 fail | 335,544,320 pass | fail |
| `session_grid_large` | 1 | inline certified | 0.97114497694024449 fail | 0.97824488473817828 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | inline certified | 0.978703008194445 fail | 0.98414816148621143 fail | 738,197,504 pass | fail |
| `session_grid_large` | 1 | candidate runtime | 0.90361232784092516 fail | 0.87792323319506516 fail | 738,197,504 pass | fail |
| `session_grid_large` | 2 | candidate runtime | 0.90417890405291468 fail | 0.87794225218573296 fail | 738,197,504 pass | fail |
| `session_wide` | 1 | inline certified | 0.93921276461237946 fail | 0.92352114248846806 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | inline certified | 0.94520577787033111 fail | 0.93282774185603223 fail | 469,762,048 pass | fail |
| `session_wide` | 1 | candidate runtime | 0.88902390291204192 fail | 0.88445406153039619 fail | 469,762,048 pass | fail |
| `session_wide` | 2 | candidate runtime | 0.88472989067366781 fail | 0.88156602528064743 fail | 469,762,048 pass | fail |

No eligible class passes any corner prefix in either lane or invocation. The control remains
permanently ineligible. Every memory bound passes. Automatic dispatch remains unauthorized.

## Why persistent CUDA is still slower

The table decomposes nearest-rank medians for persistent `candidate_runtime` steady sessions.
Times are milliseconds. `GPU device` is overlapping CUDA-event telemetry and is not added again to
the host interval. `Ideal zero-GPU ratio` is a deliberately impossible upper bound:
`CPU total / median(GPU preparation + residual validation)`, assuming all GPU execution takes zero.

| Case | Run | CPU total | CPU execute | GPU total | GPU execute | GPU pack | GPU upload | GPU factor/solve host | GPU factor/solve device | GPU execute remainder | Ideal zero-GPU ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 31.600 | 6.944 | 41.346 | 17.131 | 0.735 | 0.780 | 11.300 | 11.088 | 3.323 | 1.304234 |
| control | 2 | 31.550 | 6.825 | 41.373 | 17.091 | 0.736 | 0.771 | 11.263 | 11.069 | 3.326 | 1.301736 |
| 33-by-33 grid | 1 | 1,443.756 | 265.286 | 1,535.198 | 379.909 | 37.556 | 27.087 | 147.146 | 146.962 | 159.297 | 1.250854 |
| 33-by-33 grid | 2 | 1,446.090 | 266.660 | 1,540.097 | 381.138 | 37.522 | 26.964 | 147.333 | 147.119 | 162.012 | 1.246906 |
| 65-by-65 grid | 1 | 2,883.993 | 626.718 | 3,191.626 | 928.703 | 73.974 | 52.055 | 483.191 | 483.008 | 308.784 | 1.273100 |
| 65-by-65 grid | 2 | 2,883.389 | 626.099 | 3,188.958 | 923.505 | 73.809 | 52.367 | 473.196 | 472.985 | 308.723 | 1.274248 |
| four-source ring | 1 | 3,618.676 | 581.281 | 4,070.392 | 1,041.605 | 97.202 | 95.188 | 409.246 | 409.049 | 417.145 | 1.194793 |
| four-source ring | 2 | 3,627.705 | 585.104 | 4,100.353 | 1,066.707 | 96.997 | 95.428 | 428.818 | 428.630 | 422.227 | 1.195400 |

Primary-context initialization is real but not the steady-state blocker. Its fresh-session median is
roughly 188--200 ms across these cases; it is exactly zero in every persistent steady sample.
Removing it improves the eligible inline-certified median ratios to about 0.94--0.98, but CUDA is
still 2--6% slower. The more diagnostic candidate-runtime lane is 6--13% slower.

Three costs explain the result:

1. Compiler preparation plus authoritative residual validation is a large common floor. For the
   medium grid and ring, a zero-time GPU execution cannot reproduce the 1.25 median bound in both
   runs. The large grid can reach it only theoretically, with almost no budget left for execution.
2. Native cuDSS batching removes repeated analysis and reduces library calls, but changed corners
   still require host packing and values/RHS upload. Those total about 65 ms for the medium grid,
   126 ms for the large grid, and 192 ms for the ring.
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
