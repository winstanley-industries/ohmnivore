# EMI-01 Python numerical runtime provenance

This dependency is used only by the external EMI-01 reference tooling. It does not add NumPy,
Python, or a new numerical backend to the production C++ simulator. The supported reference host
is Linux x86_64 with glibc 2.28 or newer. No ambient Python installation, Python package, C/C++
compiler, BLAS, Fortran runtime, libstdc++, libgcc, or zlib is selected.

## Exact inputs

Audited on 2026-09-16 against the official Bazel Central Registry, rules_python release manifest,
and PyPI release metadata. These are deliberately fixed versions, not moving latest-version
selectors.

| Input | Exact version / artifact | SHA-256 |
|---|---|---|
| rules_python | `2.3.3` source archive, BCR module and patch metadata in `MODULE.bazel.lock` | `f700c75859a827a2e3e3ba4c9c0ec2d796e191bf0438ac3fee0b7851d83a3d4c` |
| CPython standalone | `3.12.13+20260414-x86_64-unknown-linux-gnu-install_only` | `cdcf8724d46e4857f8db5ee9f4252dc2f5da34f7940294ec6b312389dd3f41e0` |
| NumPy | `numpy-2.4.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | `e7dd01a46700b1967487141a66ac1a3cf0dd8ebf1f08db37d46389401512ca97` |
| zlib | `1.3.1`, existing BCR module `1.3.1.bcr.5`; BCR build-file patches recorded in lockfile | `9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23` |
| Existing GCC runtime archive | `gcc-toolchain-15.2.0-x86_64.tar.xz`, release `03062026` | `ed6a74810fe42979493f3b0ef188f1b7a388817f496f4c9cee3f7183415ab821` |

Artifact and metadata URLs:

- [rules_python source](https://github.com/bazel-contrib/rules_python/releases/download/2.3.3/rules_python-2.3.3.tar.gz)
  and [BCR metadata](https://bcr.bazel.build/modules/rules_python/2.3.3/source.json).
- [CPython standalone archive](https://github.com/astral-sh/python-build-standalone/releases/download/20260414/cpython-3.12.13+20260414-x86_64-unknown-linux-gnu-install_only.tar.gz)
  and the [rules_python runtime manifest](https://github.com/bazel-contrib/rules_python/blob/2.3.3/python/private/runtimes_manifest_workspace.bzl).
- [NumPy wheel](https://files.pythonhosted.org/packages/bd/79/cc665495e4d57d0aa6fbcc0aa57aa82671dfc78fbf95fe733ed86d98f52a/numpy-2.4.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl)
  and [official release metadata](https://pypi.org/pypi/numpy/2.4.3/json).
- [zlib source](https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz)
  and [BCR metadata](https://bcr.bazel.build/modules/zlib/1.3.1.bcr.5/source.json).
- [Existing GCC runtime archive](https://github.com/f0rmiga/gcc-builds/releases/download/03062026/gcc-toolchain-15.2.0-x86_64.tar.xz).

`reference/emi01/requirements.txt` admits only the exact NumPy wheel hash. `pip.parse` requests
binary-only downloads and mandatory hashes; source builds and ambient package discovery are not
fallbacks. The Python toolchain is pinned to the full patch version `3.12.13`, not just `3.12`.
The rules_python-provided release manifest pins the interpreter archive hash. The complete Bazel
module graph and extension inputs are bound by `MODULE.bazel.lock`.

The module graph change adds direct reference-only declarations for rules_python and the already
resolved zlib. Compared with the accepted EMI roadmap baseline, resolved versions change only
from rules_python `1.7.0` to `2.3.3` and package_metadata `0.0.3` to `0.0.7`; toml.bzl `0.4.1` is
added and the now-unused stardoc `0.7.2` disappears. LLVM, GCC, rules_cc, CUDA, cuDSS, SuiteSparse,
ngspice, protobuf, and the existing lint tools retain their declared versions. rules_python's own
publishing-hub metadata may appear in its shared extension lock data; those packages are not
dependencies of the EMI-01 targets.

## Native linkage and execution

The unmodified manylinux wheel bundles OpenBLAS, libgfortran, and libquadmath, but requests host
libstdc++, libgcc_s, and zlib by ELF SONAME. Merely pinning the wheel therefore does not make its
native dependencies hermetic. `runtime.load_numpy()` preloads the following declared runfiles
before importing NumPy:

- `libgcc_s.so.1`, SHA-256
  `d913eb787c453b5b8952fee1fd40673328dd975dbf1864676ba769320c975657`;
- `libstdc++.so.6.0.34`, SHA-256
  `e17f019553d4f8f9f492d99edad3e33a831601b0a9a9eb74c6a7a1b979736c57`;
- `libz.so.1`, built from the pinned zlib source with the repository's pinned LLVM toolchain.

The first two files are from the existing GCC archive; using those runtime files does not enable
CUDA or change the CPU compiler. The loader checks `/proc/self/maps` and fails if these SONAMEs
have resolved to another copy. The runtime test also rejects mapped system shared libraries
outside the glibc ABI set: libc, libm, libdl, libpthread, librt, libutil, and the ELF loader. The
kernel, CPU instruction support, glibc ABI, and filesystem remain execution-environment inputs.

The helper fixes OpenBLAS, OpenMP, and MKL thread-count environment variables to one before the
first NumPy import. Reference parallelism belongs to the harness's bounded job scheduler. NumPy's
FFT uses the wheel's pocketfft implementation. No GPU package is present.

The reference zlib runfile uses an explicit transition disabling production LLVM ASan/UBSan
instrumentation: the prebuilt interpreter and NumPy wheel are not sanitizer-instrumented. This
does not alter sanitizer settings on production targets. Running the Python runtime test with a
sanitizer configuration checks dependency compatibility and linkage, not sanitizer coverage of
CPython or NumPy.

## Redistribution audit

No upstream source or binary bytes are copied into this repository. Bazel fetches the fixed
artifacts, retains their notices, and exposes them in the reference runfiles. Upstream notices
must accompany redistribution of a packaged runfiles tree:

- rules_python uses Apache-2.0; the downloaded archive retains its `LICENSE`.
- CPython uses the Python Software Foundation license and historical notices. The standalone
  distribution includes additional dependency notices; preserve its license files when packaging.
- NumPy uses BSD-3-Clause. Its wheel retains
  `numpy-2.4.3.dist-info/licenses/LICENSE.txt` and component notices, including pocketfft.
- The wheel's OpenBLAS and LAPACK notices specify BSD terms; bundled libgfortran has
  `GPL-3.0-or-later WITH GCC-exception-3.1`, and libquadmath has `LGPL-2.1-or-later`. The wheel
  license document includes these terms and upstream source locations. They remain dynamically
  linked and unmodified. Do not strip their notices or substitute a claim that the entire wheel
  is BSD-only.
- The existing GCC libstdc++ and libgcc runtime files use GPL terms with the GCC Runtime Library
  Exception; their source is GCC 15.2.0. Preserve the GCC archive's runtime notices and source
  availability when packaging.
- zlib uses the zlib license; the source archive and BCR license target retain `LICENSE`.

## Reproduction and observed checks

```sh
bazel test //third_party/emi_python:runtime_test
bazel test --lockfile_mode=error //third_party/emi_python:runtime_test
bazel test --config=asan //third_party/emi_python:runtime_test
bazel test --config=ubsan //third_party/emi_python:runtime_test
bazel mod graph --lockfile_mode=error
```

All four runtime-test configurations passed during integration. The test verifies the exact
Python/NumPy versions, an independently known constant-input FFT, and loaded native-library
paths. A temporary smoke binary additionally confirmed the interpreter version
`3.12.13 (main, Apr 14 2026, 14:29:00) [Clang 22.1.3]`; that exploratory target was removed.
