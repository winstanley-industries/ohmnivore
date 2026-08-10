# GPU-02 native-FP64 CUDA uniform-batch evidence summary — 2026-08-08

This is the final reproducibility evidence for the opt-in GPU-02 experiment,
not a production speedup claim. Automatic CUDA dispatch remains unauthorized.
The earlier 2026-08-07 measurements are retained for before/after comparison
with the superseded per-member `UBATCH_SIZE=1` implementation; the exact
pre-fix source and binary are not retained.

Both final invocations used:

```sh
bazel run -c opt --config=cuda //cuda:gpu02_evidence_benchmark -- \
  --warmups=3 --repetitions=20
```

Each invocation contains all 715 frozen replay-v1 member identities, 480 raw
samples (4 workloads x 6 modes x 20 repetitions), 24 summaries, 4 gate
verdicts, and zero failures. Every CUDA execution used the full member count as
one native cuDSS uniform batch and recorded one factor/refactor call plus one
solve call per reuse. Each CUDA sample includes preparation, primary-context
and library setup, host packing, upload, matrix setup, device submission and
synchronization, status and memory queries, readback, result association, and
fresh CPU KLU validation.

## Preserved evidence

| Invocation | UTC interval | File SHA-256 | Completion fingerprint |
|---|---|---|---|
| 1 | 2026-08-09T04:34:26Z to 2026-08-09T04:36:31Z | `a1dff4f585eb994131196d9a95b8979df67cdb2046330cfec9a2e7ee55d141b6` | `v1-406214043417a80223b72e8ecb36f397` |
| 2 | 2026-08-09T04:36:40Z to 2026-08-09T04:38:45Z | `3232259b95bd94c7e25b41dd2c12be93b5ecf401b18fb6d311a32221b6fba45c` | `v1-048f21e6cc26275893e6663ace335bc9` |

The independent invocations have source fingerprint
`v1-404c729f9594460a6dcd5497e7ef8823` and binary fingerprint
`v1-08f2e8cfef866e2d63c2334550bf8cba`, but distinct completion and whole-file
fingerprints. An independent audit recomputed both source fingerprints and
terminal record fingerprints from the preserved bytes, reconciled every
sample's phase sum to its total, and checked every uniform-batch logical solve
and library-call count.

## Frozen workloads

| Class / case | Topology | Dimension / nnz | Batch | Prepared reuses | Purpose |
|---|---|---:|---:|---:|---|
| `small_control` / `ladder_s_65` | path ladder | 65 / 192 | 16 | 8 | launch-overhead control; permanently ineligible |
| `medium` / `tree_m_257` | binary tree | 257 / 768 | 61 | 4 | modest matrices and batch |
| `large` / `grid_l_1025` | 32x32 grid | 1025 / 4994 | 121 | 2 | largest and densest sparse factors |
| `medium_wide` / `ring_multi_m_260` | ring with four branches | 260 / 776 | 517 | 2 | widest independent batch |

Cold executes each batch once. Prepared timing constructs and uploads one
generation, analyzes its immutable structure once, and executes the declared
reuse count while retaining that generation. Every reuse is independently
CPU-certified.

## Performance diagnosis and correction

The initial implementation did not exercise the selected library's native
same-pattern batching: it set `UBATCH_SIZE=1` and issued factor, solve,
synchronization, status, and readback work member by member. GPU-02 now uses
contiguous member-major value/RHS/solution buffers with `UBATCH_SIZE` equal to
the complete batch and the default all-member `UBATCH_INDEX=-1`. The table
compares medians from diagnostic run 1 with final run 1; milliseconds are
prepared/reused totals or CUDA-event factor/solve telemetry as labeled. These
diagnostic medians use the same frozen nearest-rank rule as the CSV summaries.

| Case | Serial device factor/solve | Uniform device factor/solve | Device improvement | Serial end-to-end | Uniform end-to-end | End-to-end improvement |
|---|---:|---:|---:|---:|---:|---:|
| `ladder_s_65` | 50.874 | 20.013 | 2.542x | 278.398 | 272.290 | 1.022x |
| `tree_m_257` | 104.821 | 23.917 | 4.383x | 368.703 | 309.428 | 1.192x |
| `grid_l_1025` | 329.124 | 41.844 | 7.865x | 837.677 | 572.235 | 1.464x |
| `ring_multi_m_260` | 454.459 | 19.973 | 22.753x | 868.239 | 454.496 | 1.910x |

The corrected factor/solve work is competitive only for the widest case. In
final run 1, parallel-host KLU schedule/solve versus CUDA factor/solve device
telemetry was 9.577/20.013 ms (ladder), 15.178/23.917 ms (tree),
35.245/41.844 ms (grid), and 25.875/19.973 ms (ring). CUDA must additionally
pay one analysis (15.865--19.923 ms) and upload (7.603--8.930 ms) per prepared
generation.

The dominant remaining fixed cost is CUDA primary-context initialization:
214.566--220.091 ms at the per-case medians across the two final runs. It is
charged once in every fresh isolated raw sample by the frozen end-to-end
boundary. Moving or priming that work outside the sample would hide required
CUDA work and was not done.

Fresh CPU certification is the other limiting cost. It takes about 32.5 ms for
the prepared tree, 226.2--226.3 ms for the grid, and 141.7--141.9 ms for the ring.
The table below gives a deliberately generous upper bound that assumes every
GPU execution/setup/transfer/device/readback cost becomes zero and retains only
measured GPU preparation plus mandatory validation. Values are
`parallel_prepared / (gpu_prepare + gpu_validate)`.

| Case | Run 1 median upper bound | Run 2 median upper bound | Can reach 1.25 even with zero GPU cost? |
|---|---:|---:|---|
| `ladder_s_65` | 2.942 | 3.021 | irrelevant; permanently ineligible |
| `tree_m_257` | 1.410 | 1.421 | theoretically yes, but only about 4--5 ms remains for all GPU work |
| `grid_l_1025` | 1.141 | 1.140 | no |
| `ring_multi_m_260` | 1.154 | 1.153 | no |

Therefore the grid and ring prepared-median gate is structurally unreachable
under the frozen corpus plus mandatory fresh-KLU validation, even with a
zero-time GPU solver. This observation evaluates the frozen gate; it does not
redefine it or weaken certification.

## Frozen-gate verdict

Bounds are `parallel_cpu_median / gpu_median >= 1.25`,
`parallel_cpu_P95 / gpu_P95 >= 1.10`,
`gpu_cold_median / parallel_cpu_cold_median <= 1.10`, and peak batch memory
`<= 2,147,483,648` bytes. Ratios below are the exact recorded values.

| Workload class | Run | Prepared median ratio | Prepared P95 ratio | Cold ratio | GPU peak bytes | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `small_control` (`ladder_s_65`) | 1 | 0.052681733594836931 fail | 0.05951258501072134 fail | 116.89158180116951 fail | 33,554,432 pass | permanently ineligible |
| `small_control` (`ladder_s_65`) | 2 | 0.053764890760326416 fail | 0.055948207340879567 fail | 118.87482012877683 fail | 33,554,432 pass | permanently ineligible |
| `medium` (`tree_m_257`) | 1 | 0.16097472092349949 fail | 0.14903587891244494 fail | 18.182511093544928 fail | 33,554,432 pass | bounds fail |
| `medium` (`tree_m_257`) | 2 | 0.16112941053543081 fail | 0.15356046932408998 fail | 18.151525871237595 fail | 33,554,432 pass | bounds fail |
| `large` (`grid_l_1025`) | 1 | 0.5027960283141325 fail | 0.49741784578248388 fail | 2.7177545665174985 fail | 67,108,864 pass | bounds fail |
| `large` (`grid_l_1025`) | 2 | 0.50271318802228826 fail | 0.50923497110058347 fail | 2.739202031296331 fail | 67,108,864 pass | bounds fail |
| `medium_wide` (`ring_multi_m_260`) | 1 | 0.41336531273868182 fail | 0.41162440669145811 fail | 3.517391276605963 fail | 33,554,432 pass | bounds fail |
| `medium_wide` (`ring_multi_m_260`) | 2 | 0.41217570922502927 fail | 0.41183477029376669 fail | 3.5352568077543913 fail | 33,554,432 pass | bounds fail |

No eligible workload class reproduced the complete frozen gate. GPU-02 remains
a correct, reproducible negative experiment on this hardware after correcting
the material native-batching defect. CPU KLU remains the correctness authority,
supported no-GPU implementation, and explicit full-batch fallback. There is no
production speedup claim and no authorization for automatic dispatch.
