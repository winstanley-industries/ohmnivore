#include <cuda_runtime.h>
#include <link.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

struct DynamicRuntimeAudit {
  bool found_gcc_runtime = false;
};

int AuditDynamicRuntime(dl_phdr_info *info, std::size_t, void *opaque) {
  const std::string_view name =
      info->dlpi_name == nullptr ? "" : info->dlpi_name;
  if (name.find("libstdc++.so") != std::string_view::npos ||
      name.find("libgcc_s.so") != std::string_view::npos) {
    static_cast<DynamicRuntimeAudit *>(opaque)->found_gcc_runtime = true;
  }
  return 0;
}

__global__ void DeterministicTransform(const std::uint32_t *input,
                                       std::uint32_t *output,
                                       std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] =
        input[index] * 17U + static_cast<std::uint32_t>(index * index + 3U);
  }
}

[[nodiscard]] bool Check(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

} // namespace

int main() {
  DynamicRuntimeAudit runtime_audit;
  dl_iterate_phdr(AuditDynamicRuntime, &runtime_audit);
  if (runtime_audit.found_gcc_runtime) {
    std::cerr << "CUDA executable loaded a dynamic GCC runtime instead of the "
                 "pinned static "
                 "toolchain runtime\n";
    return 1;
  }

  int device_count = 0;
  if (!Check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") ||
      device_count < 1) {
    std::cerr << "No CUDA device is available\n";
    return 1;
  }

  cudaDeviceProp properties{};
  if (!Check(cudaGetDeviceProperties(&properties, 0),
             "cudaGetDeviceProperties") ||
      !Check(cudaSetDevice(0), "cudaSetDevice")) {
    return 1;
  }

  constexpr std::array<std::uint32_t, 8> kInput = {0U, 1U, 2U,  3U,
                                                   5U, 8U, 13U, 21U};
  std::array<std::uint32_t, kInput.size()> expected{};
  for (std::size_t index = 0; index < kInput.size(); ++index) {
    expected[index] =
        kInput[index] * 17U + static_cast<std::uint32_t>(index * index + 3U);
  }

  std::uint32_t *device_input = nullptr;
  std::uint32_t *device_output = nullptr;
  const std::size_t bytes = kInput.size() * sizeof(std::uint32_t);
  if (!Check(cudaMalloc(&device_input, bytes), "cudaMalloc(input)") ||
      !Check(cudaMalloc(&device_output, bytes), "cudaMalloc(output)")) {
    cudaFree(device_input);
    cudaFree(device_output);
    return 1;
  }

  bool okay = Check(
      cudaMemcpy(device_input, kInput.data(), bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(input)");
  if (okay) {
    DeterministicTransform<<<1, 32>>>(device_input, device_output,
                                      kInput.size());
    okay = Check(cudaGetLastError(), "DeterministicTransform launch") &&
           Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
  std::array<std::uint32_t, kInput.size()> actual{};
  if (okay) {
    okay = Check(
        cudaMemcpy(actual.data(), device_output, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(output)");
  }
  cudaFree(device_input);
  cudaFree(device_output);

  if (!okay || actual != expected) {
    std::cerr << "CUDA smoke result disagrees with deterministic CPU oracle\n";
    return 1;
  }

  int runtime_version = 0;
  int driver_version = 0;
  if (!Check(cudaRuntimeGetVersion(&runtime_version),
             "cudaRuntimeGetVersion") ||
      !Check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion")) {
    return 1;
  }
  std::cout << "backend=cuda device=\"" << properties.name
            << "\" compute_capability=" << properties.major << '.'
            << properties.minor << " runtime=" << runtime_version
            << " driver=" << driver_version
            << " global_memory_bytes=" << properties.totalGlobalMem
            << " deterministic_oracle=pass gcc_runtime=static\n";
  return 0;
}
