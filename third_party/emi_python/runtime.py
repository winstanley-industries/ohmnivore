"""Resolve NumPy's non-glibc runtime dependencies exclusively from Bazel."""

import ctypes
import importlib
import os
from pathlib import Path
import sys

from python.runfiles import runfiles


def load_numpy():
    """Load the pinned NumPy wheel with pinned native dependencies and one thread."""
    if (
        sys.version_info[:3] != (3, 12, 13)
        or sys.platform != "linux"
        or os.uname().machine != "x86_64"
    ):
        raise RuntimeError("EMI-01 requires Bazel Python 3.12.13 on Linux x86_64")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    resolver = runfiles.Create()
    if resolver is None:
        raise RuntimeError("Missing Bazel runfiles for hermetic NumPy")
    expected = []
    for name in (
        "ohmnivore_cuda_gcc15/lib64/libgcc_s.so.1",
        "ohmnivore_cuda_gcc15/lib64/libstdc++.so.6.0.34",
        "_main/third_party/emi_python/libz.so.1",
    ):
        path = resolver.Rlocation(name)
        if path is None or not Path(path).is_file():
            raise RuntimeError(f"Missing hermetic NumPy runtime: {name}")
        expected.append(Path(path).resolve())
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    numpy = importlib.import_module("numpy")
    if numpy.__version__ != "2.4.3":
        raise RuntimeError("EMI-01 requires the pinned NumPy 2.4.3 wheel")
    # Loading an ambient copy first defeats ELF SONAME preloading. Detect this
    # explicitly instead of silently accepting an unpinned numerical runtime.
    paths = {
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
    }
    for pinned in expected:
        prefix = pinned.name.split(".so", 1)[0] + ".so"
        actual = {
            Path(path).resolve() for path in paths if Path(path).name.startswith(prefix)
        }
        if actual != {pinned}:
            raise RuntimeError(
                f"Unpinned or missing NumPy runtime {prefix}: {sorted(map(str, actual))}"
            )
    return numpy
