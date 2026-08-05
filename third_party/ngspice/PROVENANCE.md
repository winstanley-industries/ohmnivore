# Hermetic ngspice acceptance oracle

Ohmnivore Phase 2C uses ngspice only as an external acceptance oracle for six
bounded linear fixtures. It is not linked into the simulator and is not a
runtime dependency of Ohmnivore.

## Upstream source identity

- Upstream project: ngspice
- Version and upstream tag: `46`, `ngspice-46`
- Release date: 2026-03-29
- Official archive URL:
  `https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/46/ngspice-46.tar.gz`
- Archive size: 10,504,768 bytes
- SHA-256:
  `a0d1699af1940b06649276dcd6ff5a566c8c0cad01b2f7b5e99dedbb4d64c19b`
- SourceForge file id: `62408843`

The repository rule will not unpack different bytes: a missing archive,
redirect failure, or checksum mismatch fails the Bazel fetch. The acceptance
test additionally requires the built program to report `ngspice-46` before it
runs a fixture.

## Pinned build inputs and configuration

- Bazel 9.2.0 (`.bazelversion`)
- `rules_foreign_cc` 0.15.1
- GNU Make 4.4.1 source archive:
  `https://mirror.bazel.build/ftpmirror.gnu.org/gnu/make/make-4.4.1.tar.gz`,
  SHA-256
  `dd16fb1d67bfab79a72f5e8390735c49e3e8e70b4945a15ab1f81ddb78658fb3`
- GCC 15.2.0 x86-64 toolchain archive:
  `https://github.com/f0rmiga/gcc-builds/releases/download/03062026/gcc-toolchain-15.2.0-x86_64.tar.xz`,
  SHA-256
  `ed6a74810fe42979493f3b0ef188f1b7a388817f496f4c9cee3f7183415ab821`;
  it includes the glibc 2.28 sysroot, libstdc++, GNU BFD linker, and binutils
- BusyBox 1.35.0 x86-64 musl static POSIX tool binary:
  `https://busybox.net/downloads/binaries/1.35.0-x86_64-linux-musl/busybox`,
  SHA-256
  `6e123e7f3202a8c1e9b1f94d8941580a25135382b99e8d3e34fb858bba311348`
- execution-platform `/bin/bash`, accepted only when its SHA-256 is exactly
  `3efccc187bafa75ff1e37d246270ab3e7aa559f242c7a52bf3ec2a1b5450bdbd`
- `third_party/ngspice/rules_foreign_cc_make_llvm.patch`, SHA-256
  `3c2d0cd5c2c4d084bbed4018610645b777ec727e21dc4013a43127859f51cd5c`
- `third_party/ngspice/ngspice-static-hicum-link.patch`, SHA-256
  `4c031027f1bd5882b0c1babf80e53728ad32ab4219c20df6e98f7f1a3ecc48e9`
- `third_party/ngspice/hermetic_posix_tools.bzl`, SHA-256
  `7c4c832e414fefbf31e0b9354fbd9acbb5aa35d1c23abbd405a90d90bcf60b97`
- `third_party/ngspice/hermetic_tools_builder.cc`, SHA-256
  `58b41373cff530ab525ecf44093f238e9065df29e9047f61e78e3ee90aba5dff`

`MODULE.bazel.lock` records the resolved Bazel module and repository-rule
inputs. The non-module archive identities are stated explicitly above and in
`MODULE.bazel`; the generated GCC repository definition is inspectable with
`bazel mod show_repo`. Local C/C++ discovery remains disabled by `.bazelrc`.

The source release already contains `configure`, `Makefile.in`, and generated
Bison parser C/header pairs. Autoconf, Automake, M4, pkg-config, and `file`
have fail sentinels under their real command names on the build `PATH`, and
their corresponding environment variables are set to `false`. `BISON`,
`YACC`, and `LEX` are also set to `false`; the build fails if the release ever
requires parser regeneration. `SOURCE_DATE_EPOCH=1774742400` fixes ngspice's
embedded creation date to 2026-03-29T00:00:00Z, the upstream release date.

The ngspice patch changes only the libtool link driver for HICUM's uninstalled
static convenience archive from the C++ tag to the C tag. HICUM `.cpp` files
remain compiled by the pinned G++ driver. This prevents libtool from recording
its relocated shared-libstdc++ metadata in the convenience archive; the final
link instead consumes a Bazel-declared copy of the pinned `libstdc++.a` under a
distinct filename. The copy has identical bytes, but its name has no sibling
`.la` metadata for libtool to substitute. No simulator source or model behavior
is changed by the patch.

The generated configure/Make scripts execute inside Bazel's sandbox with a
`PATH` containing only the declared BusyBox tool tree and pinned GCC binutils.
`CONFIG_SHELL`, `SHELL`, `AWK`, `SED`, and `GREP` point directly into that
tree; `NM`, `OBJDUMP`, and `STRIP` point directly into GCC. A repository helper
creates the BusyBox aliases and copies the static libstdc++ archive without a
host `cp`. `rules_foreign_cc` requires Bash for its generated action wrapper;
the build exposes only `/bin/bash`, verifies the exact checksum above before
configure runs, and fails closed on any other bytes. No other `/bin`,
`/usr/bin`, or `/usr/local/bin` entry is present after the wrapper establishes
the foreign-build environment. This acceptance oracle is therefore specific
to Linux x86-64 and that exact execution shell.

Configure flags are:

```text
--disable-cider
--disable-dependency-tracking
--disable-klu
--disable-maintainer-mode
--disable-openmp
--disable-osdi
--disable-shared
--disable-xspice
--enable-static
--with-editline=no
--with-fftw3=no
--with-readline=no
--without-x
```

These flags remove all optional host libraries and dynamic model loading. The
ngspice executable is built only inside a Bazel sandbox with the exact pinned
GCC C/C++ drivers, assembler, archiver, GNU BFD linker, headers, and libraries.
`--enable-static` makes ngspice's convenience libraries static, and the build
also passes `-static -no-pie`, `-all-static`, deterministic `ar rcsD`, and
`make -j8`. Thus libstdc++, libgcc, libm, pthread, dl, and glibc are all
resolved from the pinned sysroot into one static executable; build scheduling
cannot affect archive member order or indexes.

## Runtime boundary and fail-closed checks

The test never searches `PATH` for ngspice. It receives the exact Bazel output
through runfiles, parses its ELF64 program-header table, and fails unless both
`PT_INTERP` and `PT_DYNAMIC` are absent. Only then does it execute that exact
static file directly and verify the exact version. This is stronger than a
runtime-library allowlist: there is no ELF interpreter and no dynamic library
resolution at all.

ngspice runs with `-n` (no user init file), an empty `PATH`, `LC_ALL=C`, and
temporary `HOME`/`TMPDIR` directories supplied by Bazel. Missing provenance,
unexpected linkage, a non-zero process result, malformed/non-finite output, or
missing comparison data is a hard test failure; there are no skips.

## Fixtures and comparison contract

The explicit target `//acceptance:ngspice_acceptance_test` compares only:

- a current-loaded resistive DC voltage divider;
- an AC RC low-pass filter;
- UIC RC charging from a DC source;
- a pulsed RC filter;
- a true series-RL voltage step.
- a non-UIC, current-driven series RLC response with nonzero `tstart`.

The harness asks ngspice for an ASCII raw file with 17 significant digits and
parses it strictly. Ohmnivore CSV headers, row widths, and insertion order are
validated against the exact fixture schemas; duplicate, extra, missing, or
reordered columns fail acceptance. DC variables are also required in exact
insertion order.

- DC: voltage values use `max(1e-9, 1e-3 * abs(reference))`.
- AC: the point count must match and every frequency uses
  `max(1e-12 Hz, 1e-12 * abs(reference))`. Magnitudes use
  `max(1e-12, 1e-2 * abs(reference))`; wrapped phase error is at most one
  degree.
- Transient: the deterministic comparison grid is the requested `.TRAN`
  output grid starting at `tstart` and ending at the exact `tstop`. Ohmnivore's
  strictly increasing accepted-step waveform is linearly interpolated only
  between bracketing samples onto ngspice's requested output times. Exact
  Ohmnivore samples are used directly; values within
  `32 * epsilon * max(1 second, abs(time))` are the same time for this lookup
  so decimal output-grid roundoff cannot force extrapolation. Clamping and
  extrapolation are forbidden. The harness independently reconstructs the
  complete requested grid, requires exact point count, strict monotonicity,
  `tstart`, regular `tstep` samples, and exact `tstop`, and rejects reduced,
  duplicated, or irregular ngspice output. The exact Ohmnivore start and stop
  points must also be present. Voltages use
  `max(1e-2 V, 2e-2 * abs(reference))`.

The tolerances intentionally match the legacy linear acceptance policy. They
are acceptance limits, not accuracy or performance claims.

The acceptance target is deliberately incompatible with the ASan and UBSan
build settings: sanitizer gates cover Ohmnivore's C++ CPU implementation and
tests, while this independently built upstream oracle is validated by the
separate normal-config `//acceptance:ngspice_acceptance_test` gate.
