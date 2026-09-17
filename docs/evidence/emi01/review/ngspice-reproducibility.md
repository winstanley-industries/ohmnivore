# ngspice rebuild identity and numerical reproducibility

The reference executable is **not byte reproducible across Bazel sandbox paths**.
The rebuilt canonical executable did reproduce all 30 frozen qualification jobs
exactly: binary waveform payloads, headers with only the Date line removed,
metrics, spectra, and all 40 qualification checks. This diagnostic supplies no
additional performance sample.

The [paired runtime audit](paired-runtime-identities.json) first verified both
retained runs against the then-current `b257...` executable. Rebuilding the normal
configuration after sanitizer checks produced `afb6...`; the subsequent artifact
check [failed on ngspice SHA256](post-validation-runtime-recheck.log). That failure
is preserved. The earlier successful audit remains historical. Neither retained
run metadata nor any frozen source input was modified to conceal the mismatch.

| Artifact | SHA256 | Sandbox number |
|---|---|---|
| Both retained invocations; exactly recovered executable | `b2574df80f88a02b3d0597206277b189686e8708dcf8449fec42f250e41d74a6` | `946` |
| Preserved exploratory executable | `9a2b946430233328237c7c36b55f9ffe2c9a0bd7c92463353f0458668b632169` | `590` |
| Rebuilt canonical executable used for the diagnostic | `afb6b34834e82ff0f4e075b3d72b9b7d507df5c34a55a63f35b21ba478ec2f6b` | `1645` |

All three are 18,616,040 bytes. No measured copy was found in the searched Bazel
outputs/cache paths or relevant `/tmp` copies. The [recovery helper](recover_measured_ngspice.py)
changed only the equal-length digits in two known installation-prefix strings,
accepting a candidate only when its complete SHA256 matched retained metadata.
Changing `590` to `946` recovered the exact historical executable through six
ASCII-byte changes. [ngspice-recovery.json](ngspice-recovery.json) records hashes,
offsets, and search bounds. The recovered executable remains local at
`/tmp/emi01-recovered-measured-ngspice`; it is not vendored.

The cause is `rules_foreign_cc` passing an absolute temporary installation
`--prefix` to configure. ngspice 46 `configure.ac:991-992` embeds it in
`NGSPICEBINDIR` and `NGSPICEDATADIR`, used by `src/conf.c:24-25`. The frozen
`SOURCE_DATE_EPOCH` and `-ffile-prefix-map` settings do not normalize those
configured string literals. This establishes a path-dependent build limitation.

The [ELF comparison](ngspice-elf-comparison.json) finds only six `.rodata` bytes
differ between the exploratory and recovered measured files; all other sections
and symbols match. The longer rebuilt prefix changes linked data placement:
242,188 bytes differ across eight sections, including `.text`; six data-object
symbols move and function symbols remain unchanged. In the
[disassembly comparison](ngspice-disassembly-comparison.json), all 35,574 changed
instruction lines refer to data, predominantly with address shifts of 8 or 16
bytes. Removing comments and hexadecimal operand literals leaves identical
mnemonic/register/addressing forms. This description is not a general semantic
equivalence proof.

The fresh diagnostic exited successfully: 30 expected, terminal, and validated
jobs, zero failures, and successful qualification. It recorded `afb6...` and the
same 24 frozen source identities, manifest, model, and job inputs. The
[independent comparison](rebuilt-reference-comparison.json) verifies every matched
numerical byte already present in retained invocation 1. Its
[helper](compare_rebuilt_reference.py) uses standard-library hashing and bounded
gzip decoding without calling harness scheduling, restoration, metrics, or
audit routines. Both assertion-based review helpers reject optimized Python.

The [diagnostic log](rebuilt-oracle-qualification.log),
[qualification result](rebuilt-oracle-qualification.json), and
[comparison log](rebuilt-reference-comparison.log) are retained.
[build-reproducibility.json](build-reproducibility.json) binds their hashes and
diagnostic terminal/metadata identities. Duplicate diagnostic raw files remain
local under `/tmp/emi01-rebuilt-oracle-qualification`; the repository retains
the already-matched original waveform payloads.

Reproduce with an unused output directory:

```sh
bazel run //reference/emi01:study -- \
  --qualification-only --out=/tmp/emi01-rebuilt-reproduction
RUNFILES_DIR="$PWD/bazel-bin/reference/emi01/study.runfiles" \
  bazel-bin/reference/emi01/study.runfiles/_main/reference/emi01/_study.venv/bin/python3 \
  docs/evidence/emi01/review/compare_rebuilt_reference.py \
  --retained=docs/evidence/emi01/run-1 \
  --diagnostic=/tmp/emi01-rebuilt-reproduction \
  --out=/tmp/emi01-rebuilt-comparison.json
```

The retained timings still describe `b257...`; they are not reassigned to
`afb6...`. Diagnostic timing and resource usage are not claimed identical or
added to the two retained performance invocations. No build fix or general
executable-equivalence claim is made here. A future packaging change must record
and qualify its resulting executable identity. Hashes and local observations
are not independent execution attestation.
