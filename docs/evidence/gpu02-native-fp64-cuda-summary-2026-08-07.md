# GPU-02 serial diagnostic evidence summary — 2026-08-07

This complete stream is retained as diagnostic evidence for the initial
per-member `UBATCH_SIZE=1` implementation. It is superseded by the native
uniform-batch evidence dated 2026-08-08 and is not the final GPU-02 performance
verdict. Keeping it preserves the measured before/after comparison, but the
exact pre-fix source and binary are not retained, so this is not a standalone
implementation-reproduction artifact. It was never a production speedup claim,
and automatic CUDA dispatch remains unauthorized.

Both invocations used:

```sh
bazel run -c opt --config=cuda //cuda:gpu02_evidence_benchmark -- \
  --warmups=3 --repetitions=20
```

Each invocation contains all 715 frozen replay-v1 member identities, 480 raw
samples (4 workloads x 6 modes x 20 repetitions), 24 summaries, 4 gate
verdicts, and zero failures. Each CUDA sample includes preparation, executor
creation, upload, device submission and synchronization, readback, and fresh
CPU KLU validation. The fair comparator is the deterministic parallel-host KLU
mode; deterministic single-thread KLU is also recorded in every invocation.

## Preserved evidence

| Invocation | UTC interval | File SHA-256 | Completion fingerprint |
|---|---|---|---|
| 1 | 2026-08-08T06:25:07Z to 2026-08-08T06:27:35Z | `d54fab5a23611211b1b9c2e9d48fd9d7c2ecace3250698e0500756e912a56285` | `v1-ab0d0b46a942e637e3d4c8ad996a555c` |
| 2 | 2026-08-08T06:28:07Z to 2026-08-08T06:30:35Z | `f669fb545919b5d37e1d9bbc5c38363b586c9081bdfed389a54397ec40578aae` | `v1-b110fef760e68468f05ec7049cbe3599` |

The independent invocations have the same source fingerprint
`v1-1263e756f58d08a82c64c6c005ea3763` and binary fingerprint
`v1-04fa83fef8832567c4802cabc571b90c`, but distinct completion and whole-file
fingerprints.

## Frozen-gate verdict

Bounds are `parallel_cpu_median / gpu_median >= 1.25`,
`parallel_cpu_P95 / gpu_P95 >= 1.10`,
`gpu_cold_median / parallel_cpu_cold_median <= 1.10`, and peak batch memory
`<= 2,147,483,648` bytes. Ratios below are the exact recorded values.

| Workload class | Run | Prepared median ratio | Prepared P95 ratio | Cold ratio | GPU peak bytes | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `small_control` (`ladder_s_65`) | 1 | 0.050957860285409905 fail | 0.0477778961918831 fail | 107.51762631139847 fail | 33,554,432 pass | permanently ineligible |
| `small_control` (`ladder_s_65`) | 2 | 0.050775269118587429 fail | 0.051956876390321621 fail | 107.68193151131962 fail | 33,554,432 pass | permanently ineligible |
| `medium` (`tree_m_257`) | 1 | 0.13337513982712929 fail | 0.13411864605163443 fail | 18.322381525657544 fail | 33,554,432 pass | bounds fail |
| `medium` (`tree_m_257`) | 2 | 0.13407662011757374 fail | 0.13826264578291 fail | 18.254897227289749 fail | 33,554,432 pass | bounds fail |
| `large` (`grid_l_1025`) | 1 | 0.34259075803367722 fail | 0.33656244520677331 fail | 3.4969023331298765 fail | 62,914,560 pass | bounds fail |
| `large` (`grid_l_1025`) | 2 | 0.3419125041453866 fail | 0.34518281552536584 fail | 3.527275462997046 fail | 62,914,560 pass | bounds fail |
| `medium_wide` (`ring_multi_m_260`) | 1 | 0.21513107388763011 fail | 0.21878089792773622 fail | 5.4308605698044605 fail | 33,554,432 pass | bounds fail |
| `medium_wide` (`ring_multi_m_260`) | 2 | 0.21587490490185329 fail | 0.21894577909842214 fail | 5.4034014676049109 fail | 33,554,432 pass | bounds fail |

No eligible workload class reproduced the complete frozen gate in this
diagnostic implementation. Its serial library-call shape was subsequently
identified as invalid for a GPU-02 performance conclusion and replaced without
changing the corpus, timing boundary, validation, or gate. CPU KLU remains the
correctness authority, supported no-GPU implementation, and explicit
full-batch fallback.
