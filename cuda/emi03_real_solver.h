#ifndef OHMNIVORE_CUDA_EMI03_REAL_SOLVER_H_
#define OHMNIVORE_CUDA_EMI03_REAL_SOLVER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "ohmnivore/status.h"

namespace ohmnivore {

inline constexpr std::size_t kEmi03WorkerDeviceAllocationLimit =
    256ULL * 1024 * 1024;

enum class Emi03CudaFault {
  kNone,
  kAllocationFailure,
  kCudssAllocationFailure,
  kCudssFailure,
  kCudaFailure,
  kDataInfoFailure,
  kNonFiniteSolve,
  kWrongSolve,
  kNonFiniteExpression,
  kWrongExpression,
};

struct Emi03CudaOptions {
  std::size_t maximum_device_bytes = kEmi03WorkerDeviceAllocationLimit;
  Emi03CudaFault fault = Emi03CudaFault::kNone;
};

struct Emi03CudaStatistics {
  std::string job_id;
  std::size_t current_device_bytes = 0;
  std::size_t outstanding_device_bytes = 0;
  std::size_t peak_device_bytes = 0;
  std::size_t controlled_peak_device_bytes = 0;
  std::size_t cudss_peak_device_bytes = 0;
  std::size_t cudss_estimated_peak_device_bytes = 0;
  std::size_t maximum_device_bytes = 0;
  std::size_t live_factorizations = 0;
  std::size_t analyses = 0;
  std::size_t factorizations = 0;
  std::size_t refactorizations = 0;
  std::size_t reuses = 0;
  std::size_t solves = 0;
  std::size_t successful_solves = 0;
  std::size_t refinements = 0;
  std::size_t rejected_pivots = 0;
  std::size_t allocation_failures = 0;
  std::size_t cleanup_failures = 0;
  std::size_t expression_batches = 0;
  std::size_t expression_full_ad = 0;
  std::size_t expression_value_only = 0;
  std::size_t expression_program_uploads = 0;
  std::size_t upload_bytes = 0;
  std::size_t readback_bytes = 0;
  std::uint64_t context_setup_ns = 0;
  std::uint64_t solver_setup_ns = 0;
  std::uint64_t analysis_ns = 0;
  std::uint64_t factor_ns = 0;
  std::uint64_t solve_ns = 0;
  std::uint64_t synchronization_ns = 0;
  std::uint64_t upload_ns = 0;
  std::uint64_t readback_ns = 0;
  std::uint64_t validation_ns = 0;
  std::uint64_t expression_prepare_ns = 0;
  std::uint64_t expression_evaluate_ns = 0;
};

// One synchronous job per process. A new job cannot inherit numeric state or
// program buffers. The CUDA primary context may persist in the worker.
[[nodiscard]] Result<bool>
BeginEmi03CudaJob(std::string job_id, const Emi03CudaOptions &options = {});
[[nodiscard]] Result<Emi03CudaStatistics> EndEmi03CudaJob();
[[nodiscard]] Emi03CudaStatistics SnapshotEmi03CudaJob();
[[nodiscard]] std::string
Emi03CudaStatisticsJson(const Emi03CudaStatistics &statistics);

} // namespace ohmnivore

#endif // OHMNIVORE_CUDA_EMI03_REAL_SOLVER_H_
