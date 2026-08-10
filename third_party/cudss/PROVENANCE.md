# cuDSS GPU-02 provenance

GPU-02 uses exactly one NVIDIA cuDSS distribution and links exactly its static solver archive.
This input is an opt-in Linux x86-64 CUDA dependency; it is not part of ordinary CPU builds or
the no-GPU implementation.

## Pinned cuDSS input

- Product and version: NVIDIA cuDSS 0.8.0.10 for CUDA 13
- Platform: Linux x86-64
- Archive: `libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz`
- URL:
  `https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz`
- Size: 157058452 bytes
- SHA-256: `ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937`
- Extraction prefix: `libcudss-linux-x86_64-0.8.0.10_cuda13-archive`
- NVIDIA redistributable manifest:
  `https://developer.download.nvidia.com/compute/cudss/redist/redistrib_0.8.0.json`

`MODULE.bazel` supplies all of those fetch and extraction fields directly to `http_archive`.
Bazel verifies the complete archive before extraction. The external repository exports only
`include/*.h`, `lib/libcudss_static.a`, and the archived `LICENSE`; none of its dynamic cuDSS,
OpenMPI, NCCL, GNU OpenMP, source, example, or CMake artifacts are build inputs.

The archive bytes were independently downloaded on 2026-08-07 and verified with `sha256sum` and
`stat`; their digest and size matched the values above. NVIDIA's redistributable manifest is the
upstream provenance authority. The pinned byte digest in `MODULE.bazel`, not an unversioned URL
or local installation, is the build authority.

## Pinned CUDA dependency and linkage

The upstream static-target metadata declares cuBLAS as `libcudss_static.a`'s sole CUDA library
dependency. GPU-02 therefore adds only the `cublas` component to Ohmnivore's existing
checksum-pinned CUDA 13.0.2 redistributable manifest:

- CUDA manifest:
  `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json`
- Manifest SHA-256:
  `fce66717a81c510ffeb89ecc3e79849ab34af3b80139f750876d9033e31d71c2`
- cuBLAS package: `libcublas-linux-x86_64-13.1.0.3-archive.tar.xz`
- cuBLAS size: 838952932 bytes
- cuBLAS SHA-256:
  `88bc951efd906032a371153ca61975e0d9c4761e4012169169a6b3a47931606e`

The local `//third_party/cudss:cublas_static` target imports only the pinned
`libcublas_static.a` and `libcublasLt_static.a` archives and the pinned `culibos` archive from
that generated repository. Dynamic cuBLAS is deliberately excluded because its transitive
`libgcc_s.so.1` dependency would violate the hermetic static GCC runtime boundary. The CUDA runtime
is selected globally for opt-in CUDA builds by
`--@rules_cuda//cuda:runtime=@cuda//:cuda_runtime_static`; the executor also declares that target
directly. The supported device-code policy is the repository's `compute_120` PTX plus `sm_120`
cubin selection. The supported build host is Linux x86-64 using the pinned CUDA 13.0.2/nvcc and
GCC 15.2.0 CUDA toolchains. Execution requires a compatible NVIDIA GPU and driver, which are
runtime platform ABIs rather than build inputs.

The Bazel target `@cudss_0_8_0_10_cuda13//:cudss_static` exposes the headers and static archive;
the CUDA implementation combines it with `//third_party/cudss:cublas_static`. No `libcudss.so`,
dynamic cuBLAS, ambient CUDA toolkit,
ambient compiler, dynamic `libstdc++`, or dynamic `libgcc_s` is permitted. The host glibc and
NVIDIA kernel driver remain the declared Linux runtime ABIs. The final GPU-02 binary audit must
reject a dynamic cuDSS, `libstdc++`, or `libgcc_s` dependency and must prove that the static cuDSS
symbols are embedded.

## Licenses and redistribution boundary

The archive's `LICENSE` is the NVIDIA Math Libraries Software Development Kits license agreement,
version dated February 10, 2022, followed by third-party notices. It governs use and distribution;
cuDSS is not open source. The archived license is exposed as
`@cudss_0_8_0_10_cuda13//:license` and must accompany any redistributed binary or SDK material as
required by that agreement. The agreement permits distribution only of identified binary,
sample, and incorporated-header portions when its distribution requirements are met; it restricts
the SDK to systems with NVIDIA GPUs, forbids removing proprietary notices, and disclaims warranty.
This summary is operational documentation, not a substitute for reading the complete archived
license or obtaining legal advice.

GPU-02 does not copy vendor source or examples into Ohmnivore, does not modify vendor material,
and does not represent NVIDIA sponsorship or endorsement.

The separately pinned cuBLAS, CUDA runtime, and `culibos` packages each contain NVIDIA's CUDA
Toolkit End User License Agreement, last updated January 12, 2025. The generated CUDA repository
exports those exact archived license files; `//third_party/cudss:cuda_static_licenses` exposes all
three alongside the locally imported static archives. Their redistribution terms govern those
CUDA components independently of the cuDSS SDK license. Any redistributed GPU-02 binary must
satisfy every applicable archived NVIDIA license; this document is not legal advice.

## Scope and update policy

cuDSS is confined to the opt-in CUDA package. CPU KLU remains the correctness authority, the
supported no-GPU implementation, and the explicit whole-batch fallback. Ordinary simulator and
CSV paths never depend on this repository.

Any cuDSS, cuBLAS, CUDA, platform, architecture, or linkage change requires all of the following:

1. a reviewed ADR update before code changes;
2. a new versioned URL, byte size, extraction prefix, and independently verified SHA-256;
3. review of the new complete license and redistributable manifest;
4. regeneration and inspection of `MODULE.bazel.lock` through Bazel;
5. CUDA correctness, hostile-result, binary-linkage, sanitizer-incompatibility, and evidence gates.
