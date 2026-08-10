#ifndef OHMNIVORE_CUDA_PREPARED_AC_CUDA_H_
#define OHMNIVORE_CUDA_PREPARED_AC_CUDA_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "ohmnivore/prepared_ac.h"

namespace ohmnivore {

// Test-only hostile seams for the CUDA implementation. Allocation, CUDA, and
// cuDSS faults fire once after live preparation work has begun so cleanup and
// generation invalidation are exercised; partial output remains hostile on
// every execution. Production callers use kNone. No mode changes the
// backend-neutral acceptance boundary.
enum class CudaPreparedAcFault {
  kNone,
  kAllocationFailure,
  kCudaFailure,
  kCudssFailure,
  kDataInfoFailure,
  kPartialResult,
};

struct CudaPreparedAcOptions {
  CudaPreparedAcFault fault = CudaPreparedAcFault::kNone;
};

// Counts are cumulative for the executor lifetime. Phase timings and memory
// fields describe the most recent Execute call / currently prepared generation.
// CUDA event durations are telemetry only and are not part of the host interval
// sum. All memory figures are bytes.
struct CudaPreparedAcStatistics {
  std::uint64_t preparations = 0;
  std::uint64_t structure_uploads = 0;
  std::uint64_t analyses = 0;
  std::uint64_t executions = 0;
  // Logical member systems completed.
  std::uint64_t factorizations = 0;
  std::uint64_t solves = 0;
  // Native cuDSS uniform-batch calls completed. Each successful Execute adds
  // exactly one factor/refactor call and one solve call, independent of the
  // member count.
  std::uint64_t factorization_calls = 0;
  std::uint64_t solve_calls = 0;
  std::uint64_t releases = 0;
  std::size_t uniform_batch_size = 0;

  std::uint64_t last_context_setup_ns = 0;
  std::uint64_t last_library_setup_ns = 0;
  std::uint64_t last_host_pack_ns = 0;
  std::uint64_t last_upload_ns = 0;
  std::uint64_t last_matrix_setup_ns = 0;
  std::uint64_t last_analysis_submit_ns = 0;
  std::uint64_t last_analysis_sync_ns = 0;
  std::uint64_t last_factor_solve_submit_ns = 0;
  std::uint64_t last_factor_solve_sync_ns = 0;
  std::uint64_t last_readback_ns = 0;
  std::uint64_t last_phase_status_ns = 0;
  std::uint64_t last_memory_accounting_ns = 0;
  std::uint64_t last_result_assembly_ns = 0;
  std::uint64_t last_analysis_device_ns = 0;
  std::uint64_t last_factor_solve_device_ns = 0;

  // Incremental stream/event/library residency after CUDA primary-context
  // initialization and before the prepared generation's controlled
  // allocations. CUDA runtime memory queries cannot observe the pre-context
  // baseline, so this is not labeled as complete primary-context residency.
  std::size_t context_device_bytes = 0;
  // Structure, values, RHS, and solution buffers owned directly by the
  // executor for the current generation.
  std::size_t controlled_device_bytes = 0;
  std::size_t cudss_estimated_peak_device_bytes = 0;
  std::size_t cudss_observed_peak_device_bytes = 0;
  // Synchronized drop in free memory after the context baseline, excluding the
  // context_device_bytes baseline itself.
  std::size_t cuda_mem_info_peak_device_bytes = 0;
  // max(controlled + max(cuDSS estimate, observed internal allocation peak),
  //     synchronized cudaMemGetInfo batch delta).
  std::size_t peak_batch_device_bytes = 0;
  // Internal cuDSS bytes still owned by the current prepared generation.
  std::size_t cudss_outstanding_device_bytes = 0;
  // Internal cuDSS bytes remaining after the most recent explicit Release.
  std::size_t last_release_outstanding_device_bytes = 0;
};

// Explicit, opt-in GPU-02 implementation. CUDA and cuDSS declarations and all
// device ownership remain in the private implementation in the .cu file.
class CudaPreparedAcBatchBackend final : public PreparedAcBatchBackend {
public:
  explicit CudaPreparedAcBatchBackend(CudaPreparedAcOptions options = {});
  ~CudaPreparedAcBatchBackend() override;

  CudaPreparedAcBatchBackend(const CudaPreparedAcBatchBackend &) = delete;
  CudaPreparedAcBatchBackend &
  operator=(const CudaPreparedAcBatchBackend &) = delete;
  CudaPreparedAcBatchBackend(CudaPreparedAcBatchBackend &&) = delete;
  CudaPreparedAcBatchBackend &operator=(CudaPreparedAcBatchBackend &&) = delete;

  [[nodiscard]] Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &batch) override;
  // Observable owner-thread teardown. Evidence and tests call Release rather
  // than relying on the no-throw destructor, so any CUDA/cuDSS cleanup status
  // or allocation imbalance becomes kPreparedBackendFailure.
  [[nodiscard]] Result<bool> Release();
  [[nodiscard]] const CudaPreparedAcStatistics &statistics() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ohmnivore

#endif // OHMNIVORE_CUDA_PREPARED_AC_CUDA_H_
