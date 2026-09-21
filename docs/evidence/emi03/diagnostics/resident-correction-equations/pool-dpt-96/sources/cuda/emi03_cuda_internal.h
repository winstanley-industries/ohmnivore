#ifndef OHMNIVORE_CUDA_EMI03_CUDA_INTERNAL_H_
#define OHMNIVORE_CUDA_EMI03_CUDA_INTERNAL_H_

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "cuda/emi03_real_solver.h"

namespace ohmnivore::emi03_cuda_internal {

using Clock = std::chrono::steady_clock;
[[nodiscard]] inline std::uint64_t Elapsed(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

class BackendError : public std::runtime_error {
public:
  BackendError(ErrorCode code, std::string message)
      : std::runtime_error(std::move(message)), code(code) {}
  ErrorCode code;
};

void CheckCuda(cudaError_t status, const char *operation);
void RequireJob();
[[nodiscard]] bool ConsumeFault(Emi03CudaFault fault);
[[nodiscard]] Emi03CudaStatistics &Statistics();
[[nodiscard]] void *AllocateDevice(std::size_t bytes);
void FreeDevice(void *pointer, std::size_t bytes) noexcept;
void Synchronize(cudaStream_t stream);
void ResetExpressionCache() noexcept;

} // namespace ohmnivore::emi03_cuda_internal

#endif // OHMNIVORE_CUDA_EMI03_CUDA_INTERNAL_H_
