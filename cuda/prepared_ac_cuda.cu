#include "cuda/prepared_ac_cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cudss.h>

namespace ohmnivore {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t ElapsedNanoseconds(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

[[noreturn]] void ThrowCuda(cudaError_t status, const char *operation) {
  throw std::runtime_error(std::string(operation) +
                           " failed with CUDA status " +
                           std::to_string(static_cast<int>(status)) + ": " +
                           cudaGetErrorString(status));
}

void CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    ThrowCuda(status, operation);
  }
}

[[noreturn]] void ThrowCudss(cudssStatus_t status, const char *operation) {
  throw std::runtime_error(std::string(operation) +
                           " failed with cuDSS status " +
                           std::to_string(static_cast<int>(status)));
}

void CheckCudss(cudssStatus_t status, const char *operation) {
  if (status != CUDSS_STATUS_SUCCESS) {
    ThrowCudss(status, operation);
  }
}

[[nodiscard]] std::string GenerationKey(const PreparedAcBatch &batch) {
  return std::to_string(batch.contract_version) + "\n" + batch.replay_id +
         "\n" + batch.structure.fingerprint + "\n" + batch.batch_fingerprint +
         "\n" + std::to_string(batch.members.size());
}

[[nodiscard]] std::size_t CheckedMultiply(std::size_t left, std::size_t right,
                                          const char *description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::runtime_error(std::string(description) + " size overflows");
  }
  return left * right;
}

class DeviceAllocationTracker {
public:
  static int Allocate(void *context, void **pointer, std::size_t size,
                      cudaStream_t stream) {
    auto *tracker = static_cast<DeviceAllocationTracker *>(context);
    if (tracker->reject_allocations_.load()) {
      *pointer = nullptr;
      return static_cast<int>(cudaErrorMemoryAllocation);
    }
    const cudaError_t status = stream == nullptr
                                   ? cudaMalloc(pointer, size)
                                   : cudaMallocAsync(pointer, size, stream);
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
    const std::size_t current = tracker->current_.fetch_add(size) + size;
    std::size_t peak = tracker->peak_.load();
    while (current > peak &&
           !tracker->peak_.compare_exchange_weak(peak, current)) {
    }
    return static_cast<int>(cudaSuccess);
  }

  static int Free(void *context, void *pointer, std::size_t size,
                  cudaStream_t stream) {
    auto *tracker = static_cast<DeviceAllocationTracker *>(context);
    const cudaError_t status =
        stream == nullptr ? cudaFree(pointer) : cudaFreeAsync(pointer, stream);
    if (status == cudaSuccess) {
      const std::size_t previous = tracker->current_.fetch_sub(size);
      if (previous < size) {
        tracker->current_.store(0);
        return static_cast<int>(cudaErrorInvalidValue);
      }
    }
    return static_cast<int>(status);
  }

  void Reset() {
    current_.store(0);
    peak_.store(0);
    reject_allocations_.store(false);
  }

  void RejectAllocations(bool reject) { reject_allocations_.store(reject); }

  [[nodiscard]] std::size_t current() const { return current_.load(); }
  [[nodiscard]] std::size_t peak() const { return peak_.load(); }

private:
  std::atomic<std::size_t> current_ = 0;
  std::atomic<std::size_t> peak_ = 0;
  std::atomic<bool> reject_allocations_ = false;
};

} // namespace

class CudaPreparedAcBatchBackend::Impl {
public:
  explicit Impl(CudaPreparedAcOptions options)
      : options_(options), owner_(std::this_thread::get_id()) {}

  ~Impl() { ResetNoThrow(); }

  [[nodiscard]] Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &batch) {
    auto valid = ValidatePreparedAcBatch(batch);
    if (!valid.ok()) {
      return Result<PreparedAcBatchResult>::Fail(valid.error().code,
                                                 valid.error().message);
    }
    if (std::this_thread::get_id() != owner_) {
      return BackendFailure(
          "CUDA prepared executor used from a non-owner host thread");
    }

    ClearLastTimings();
    try {
      const std::string generation = GenerationKey(batch);
      if (generation != generation_key_) {
        if (CanRefreshValuesAndRhs(batch)) {
          RefreshValuesAndRhs(batch, generation);
        } else {
          Prepare(batch, generation);
        }
      }
      CheckCuda(cudaEventRecord(factor_solve_start_, stream_),
                "cudaEventRecord(factor_solve_start)");
      const Clock::time_point factor_submit_start = Clock::now();
      const int factor_phase = numeric_factors_exist_
                                   ? CUDSS_PHASE_REFACTORIZATION
                                   : CUDSS_PHASE_FACTORIZATION;
      CheckCudss(cudssExecute(handle_, factor_phase, config_, data_, matrix_,
                              solution_, rhs_),
                 numeric_factors_exist_ ? "cudssExecute(batch refactorization)"
                                        : "cudssExecute(batch factorization)");
      statistics_.last_factor_solve_submit_ns +=
          ElapsedNanoseconds(factor_submit_start);
      CheckCuda(cudaEventRecord(factor_solve_end_, stream_),
                "cudaEventRecord(factor end)");

      Clock::time_point sync_start = Clock::now();
      CheckCuda(cudaStreamSynchronize(stream_),
                "cudaStreamSynchronize(batch factorization)");
      statistics_.last_factor_solve_sync_ns += ElapsedNanoseconds(sync_start);
      Clock::time_point status_start = Clock::now();
      CheckDataInfo("cuDSS uniform-batch factorization", true);
      statistics_.last_phase_status_ns += ElapsedNanoseconds(status_start);
      numeric_factors_exist_ = true;
      statistics_.last_factor_solve_device_ns =
          EventElapsedNanoseconds(factor_solve_start_, factor_solve_end_);

      CheckCuda(cudaEventRecord(factor_solve_start_, stream_),
                "cudaEventRecord(solve start)");
      const Clock::time_point solve_submit_start = Clock::now();
      cudssStatus_t solve_status = CUDSS_STATUS_SUCCESS;
      if (!fault_injected_ &&
          options_.fault == CudaPreparedAcFault::kCudssFailure) {
        fault_injected_ = true;
        solve_status = CUDSS_STATUS_INVALID_VALUE;
      } else {
        solve_status = cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_,
                                    matrix_, solution_, rhs_);
      }
      CheckCudss(solve_status, "cudssExecute(batch solve)");
      statistics_.last_factor_solve_submit_ns +=
          ElapsedNanoseconds(solve_submit_start);
      CheckCuda(cudaEventRecord(factor_solve_end_, stream_),
                "cudaEventRecord(solve end)");

      if (!fault_injected_ &&
          options_.fault == CudaPreparedAcFault::kCudaFailure) {
        fault_injected_ = true;
        CheckCuda(cudaErrorInvalidValue,
                  "injected post-solve CUDA synchronization");
      }

      sync_start = Clock::now();
      CheckCuda(cudaStreamSynchronize(stream_),
                "cudaStreamSynchronize(batch solve)");
      statistics_.last_factor_solve_sync_ns += ElapsedNanoseconds(sync_start);
      status_start = Clock::now();
      CheckDataInfo("cuDSS uniform-batch solve");
      statistics_.last_phase_status_ns += ElapsedNanoseconds(status_start);
      statistics_.last_factor_solve_device_ns +=
          EventElapsedNanoseconds(factor_solve_start_, factor_solve_end_);

      const Clock::time_point readback_start = Clock::now();
      CheckCuda(cudaMemcpyAsync(host_solution_.data(), device_solution_,
                                controlled_solution_bytes_,
                                cudaMemcpyDeviceToHost, stream_),
                "cudaMemcpyAsync(solution readback)");
      CheckCuda(cudaStreamSynchronize(stream_),
                "cudaStreamSynchronize(solution readback)");
      statistics_.last_readback_ns = ElapsedNanoseconds(readback_start);
      const Clock::time_point accounting_start = Clock::now();
      ObserveDeviceMemory();
      statistics_.last_memory_accounting_ns +=
          ElapsedNanoseconds(accounting_start);

      const Clock::time_point result_assembly_start = Clock::now();
      PreparedAcBatchResult result{
          .contract_version = batch.contract_version,
          .replay_id = batch.replay_id,
          .structure_fingerprint = batch.structure.fingerprint,
          .batch_fingerprint = batch.batch_fingerprint,
          .members = {},
      };
      result.members.reserve(batch.members.size());
      for (std::size_t member = 0; member < batch.members.size(); ++member) {
        std::vector<std::complex<double>> values;
        values.reserve(batch.structure.dimension);
        const std::size_t base = member * batch.structure.dimension;
        for (std::size_t row = 0; row < batch.structure.dimension; ++row) {
          const cuDoubleComplex value = host_solution_[base + row];
          values.emplace_back(cuCreal(value), cuCimag(value));
        }
        result.members.push_back(PreparedAcResultMember{
            .identity = batch.members[member].identity,
            .solution = std::move(values),
        });
      }
      statistics_.last_result_assembly_ns =
          ElapsedNanoseconds(result_assembly_start);

      ++statistics_.executions;
      statistics_.factorizations += batch.members.size();
      statistics_.solves += batch.members.size();
      ++statistics_.factorization_calls;
      ++statistics_.solve_calls;
      statistics_.cudss_outstanding_device_bytes = tracker_.current();
      if (options_.fault == CudaPreparedAcFault::kPartialResult &&
          !result.members.empty()) {
        result.members.pop_back();
        if (!result.members.empty() &&
            !result.members.front().solution.empty()) {
          result.members.front().solution.front() +=
              std::complex<double>{1.0, 0.0};
        }
      }
      return Result<PreparedAcBatchResult>::Ok(std::move(result));
    } catch (const std::bad_alloc &) {
      return FailureWithCleanup(
          "CUDA prepared executor host allocation failed");
    } catch (const std::exception &error) {
      const std::string message = error.what();
      return FailureWithCleanup(message);
    } catch (...) {
      return FailureWithCleanup(
          "CUDA prepared executor raised an unknown error");
    }
  }

  [[nodiscard]] Result<bool> Release() {
    if (std::this_thread::get_id() != owner_) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBackendFailure,
          "CUDA prepared executor released from a non-owner host thread");
    }
    try {
      const std::string cleanup_error = ResetAndCollectErrors();
      statistics_.last_release_outstanding_device_bytes = tracker_.current();
      if (!cleanup_error.empty()) {
        return Result<bool>::Fail(ErrorCode::kPreparedBackendFailure,
                                  cleanup_error);
      }
    } catch (...) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBackendFailure,
          "CUDA prepared executor cleanup raised an exception");
    }
    ++statistics_.releases;
    return Result<bool>::Ok(true);
  }

  [[nodiscard]] const CudaPreparedAcStatistics &statistics() const {
    return statistics_;
  }

private:
  [[nodiscard]] Result<PreparedAcBatchResult>
  BackendFailure(std::string message) const {
    return Result<PreparedAcBatchResult>::Fail(
        ErrorCode::kPreparedBackendFailure, std::move(message));
  }

  [[nodiscard]] Result<PreparedAcBatchResult>
  FailureWithCleanup(std::string message) {
    try {
      const std::string cleanup_error = ResetAndCollectErrors();
      if (!cleanup_error.empty()) {
        message += "; cleanup failure: " + cleanup_error;
      }
    } catch (...) {
      message += "; cleanup failure reporting raised an exception";
    }
    return BackendFailure(std::move(message));
  }

  void ClearLastTimings() {
    statistics_.last_context_setup_ns = 0;
    statistics_.last_library_setup_ns = 0;
    statistics_.last_host_pack_ns = 0;
    statistics_.last_upload_ns = 0;
    statistics_.last_matrix_setup_ns = 0;
    statistics_.last_analysis_submit_ns = 0;
    statistics_.last_analysis_sync_ns = 0;
    statistics_.last_factor_solve_submit_ns = 0;
    statistics_.last_factor_solve_sync_ns = 0;
    statistics_.last_readback_ns = 0;
    statistics_.last_phase_status_ns = 0;
    statistics_.last_memory_accounting_ns = 0;
    statistics_.last_result_assembly_ns = 0;
    statistics_.last_analysis_device_ns = 0;
    statistics_.last_factor_solve_device_ns = 0;
  }

  void Configure(cudssConfigParam_t parameter, const void *value,
                 std::size_t size, const char *description) {
    CheckCudss(cudssConfigSet(config_, parameter, value, size), description);
  }

  void CheckDataInfo(const char *phase, bool permit_fault = false) {
    int info = 0;
    std::size_t written = 0;
    CheckCudss(cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &info,
                            sizeof(info), &written),
               "cudssDataGet(CUDSS_DATA_INFO)");
    if (permit_fault && !fault_injected_ &&
        options_.fault == CudaPreparedAcFault::kDataInfoFailure) {
      fault_injected_ = true;
      info = 1;
    }
    if (written != sizeof(info) || info != 0) {
      throw std::runtime_error(std::string(phase) +
                               " reported asynchronous data-info failure " +
                               std::to_string(info));
    }
  }

  [[nodiscard]] std::uint64_t EventElapsedNanoseconds(cudaEvent_t start,
                                                      cudaEvent_t end) {
    float milliseconds = 0.0F;
    CheckCuda(cudaEventElapsedTime(&milliseconds, start, end),
              "cudaEventElapsedTime");
    return static_cast<std::uint64_t>(static_cast<double>(milliseconds) *
                                      1'000'000.0);
  }

  template <typename T>
  void AllocateControlled(T **pointer, std::size_t count,
                          const char *description) {
    const std::size_t bytes = CheckedMultiply(count, sizeof(T), description);
    CheckCuda(
        cudaMallocAsync(reinterpret_cast<void **>(pointer), bytes, stream_),
        description);
    controlled_bytes_ += bytes;
  }

  [[nodiscard]] bool
  CanRefreshValuesAndRhs(const PreparedAcBatch &batch) const {
    return !generation_key_.empty() && stream_ != nullptr && data_ != nullptr &&
           matrix_ != nullptr && rhs_ != nullptr && solution_ != nullptr &&
           batch.structure.fingerprint == prepared_structure_fingerprint_ &&
           batch.structure.dimension == prepared_dimension_ &&
           batch.structure.column_indices.size() == prepared_nonzeros_ &&
           batch.members.size() == statistics_.uniform_batch_size;
  }

  void PackValuesAndRhs(const PreparedAcBatch &batch) {
    const std::size_t member_values = CheckedMultiply(
        batch.members.size(), batch.structure.column_indices.size(),
        "GPU-02 member values");
    const std::size_t member_vectors =
        CheckedMultiply(batch.members.size(), batch.structure.dimension,
                        "GPU-02 member vectors");
    host_values_.clear();
    host_rhs_.clear();
    host_values_.reserve(member_values);
    host_rhs_.reserve(member_vectors);
    host_solution_.resize(member_vectors);
    for (const PreparedAcMember &member : batch.members) {
      for (const std::complex<double> value : member.matrix_values) {
        host_values_.push_back(
            make_cuDoubleComplex(value.real(), value.imag()));
      }
      for (const std::complex<double> value : member.rhs) {
        host_rhs_.push_back(make_cuDoubleComplex(value.real(), value.imag()));
      }
    }
  }

  void RefreshValuesAndRhs(const PreparedAcBatch &batch,
                           const std::string &generation) {
    const std::size_t expected_values = host_values_.size();
    const std::size_t expected_rhs = host_rhs_.size();
    const std::size_t expected_solution = host_solution_.size();
    const Clock::time_point host_pack_start = Clock::now();
    PackValuesAndRhs(batch);
    statistics_.last_host_pack_ns = ElapsedNanoseconds(host_pack_start);
    if (host_values_.size() != expected_values ||
        host_rhs_.size() != expected_rhs ||
        host_solution_.size() != expected_solution) {
      throw std::runtime_error(
          "GPU-02 same-structure refresh changed a prepared buffer size");
    }

    const Clock::time_point upload_start = Clock::now();
    CheckCuda(cudaMemcpyAsync(device_values_, host_values_.data(),
                              host_values_.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync(refreshed matrix values)");
    CheckCuda(cudaMemcpyAsync(device_rhs_, host_rhs_.data(),
                              host_rhs_.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync(refreshed right-hand sides)");
    CheckCuda(cudaMemsetAsync(device_solution_, 0, controlled_solution_bytes_,
                              stream_),
              "cudaMemsetAsync(refreshed uniform-batch solutions)");
    CheckCuda(cudaStreamSynchronize(stream_),
              "cudaStreamSynchronize(values/RHS refresh)");
    statistics_.last_upload_ns = ElapsedNanoseconds(upload_start);
    generation_key_ = generation;
    ++statistics_.values_rhs_uploads;
    ++statistics_.same_structure_refreshes;
  }

  void Prepare(const PreparedAcBatch &batch, const std::string &generation) {
    const std::string cleanup_error = ResetAndCollectErrors();
    if (!cleanup_error.empty()) {
      throw std::runtime_error("previous CUDA generation cleanup failed: " +
                               cleanup_error);
    }
    tracker_.Reset();

    if (batch.structure.dimension >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        batch.structure.column_indices.size() >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        batch.members.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "GPU-02 prepared dimensions exceed signed-32 limits");
    }

    const Clock::time_point context_setup_start = Clock::now();
    CheckCuda(cudaSetDevice(0), "cudaSetDevice(0)");
    CheckCuda(cudaFree(nullptr), "CUDA context initialization");
    std::size_t context_free_before = 0;
    std::size_t total = 0;
    CheckCuda(cudaMemGetInfo(&context_free_before, &total),
              "cudaMemGetInfo(context baseline)");
    statistics_.last_context_setup_ns = ElapsedNanoseconds(context_setup_start);

    const Clock::time_point library_setup_start = Clock::now();
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    CheckCuda(cudaEventCreate(&analysis_start_),
              "cudaEventCreate(analysis start)");
    CheckCuda(cudaEventCreate(&analysis_end_), "cudaEventCreate(analysis end)");
    CheckCuda(cudaEventCreate(&factor_solve_start_),
              "cudaEventCreate(factor/solve start)");
    CheckCuda(cudaEventCreate(&factor_solve_end_),
              "cudaEventCreate(factor/solve end)");
    CheckCudss(cudssCreate(&handle_), "cudssCreate");
    CheckCudss(cudssSetStream(handle_, stream_), "cudssSetStream");
    CheckCudss(cudssConfigCreate(&config_), "cudssConfigCreate");

    cudssDeviceMemHandler_t handler{};
    handler.ctx = &tracker_;
    handler.device_alloc = &DeviceAllocationTracker::Allocate;
    handler.device_free = &DeviceAllocationTracker::Free;
    std::snprintf(handler.name, sizeof(handler.name), "%s",
                  "ohmnivore-gpu02-accounted");
    CheckCudss(cudssSetDeviceMemHandler(handle_, &handler),
               "cudssSetDeviceMemHandler");
    CheckCudss(cudssDataCreate(handle_, &data_), "cudssDataCreate");

    const int deterministic = 0;
    const int iterative_refinement_steps = 1;
    const int hybrid_memory = 0;
    const int hybrid_execution = 0;
    const int uniform_batch_size = static_cast<int>(batch.members.size());
    const cudssReorderingAlg_t reordering = CUDSS_REORDERING_ALG_DEFAULT;
    const cudssFactorizationAlg_t factorization =
        CUDSS_FACTORIZATION_ALG_DEFAULT;
    const cudssPivotType_t pivot = CUDSS_PIVOT_AUTO;
    const cudssMatchingAlg_t matching = CUDSS_MATCHING_ALG_NONE;
    Configure(CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic,
              sizeof(deterministic), "configure deterministic mode");
    Configure(CUDSS_CONFIG_IR_N_STEPS, &iterative_refinement_steps,
              sizeof(iterative_refinement_steps),
              "configure iterative refinement steps");
    Configure(CUDSS_CONFIG_HYBRID_MEMORY_MODE, &hybrid_memory,
              sizeof(hybrid_memory), "disable hybrid memory mode");
    Configure(CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &hybrid_execution,
              sizeof(hybrid_execution), "disable hybrid execution mode");
    Configure(CUDSS_CONFIG_UBATCH_SIZE, &uniform_batch_size,
              sizeof(uniform_batch_size), "configure uniform batch size");
    Configure(CUDSS_CONFIG_REORDERING_ALG, &reordering, sizeof(reordering),
              "configure default reordering");
    Configure(CUDSS_CONFIG_FACTORIZATION_ALG, &factorization,
              sizeof(factorization), "configure factorization algorithm");
    Configure(CUDSS_CONFIG_PIVOT_TYPE, &pivot, sizeof(pivot),
              "configure pivot policy");
    Configure(CUDSS_CONFIG_MATCHING_ALG, &matching, sizeof(matching),
              "disable matching for uniform batch");

    std::size_t context_free_after = 0;
    CheckCuda(cudaMemGetInfo(&context_free_after, &total),
              "cudaMemGetInfo(context residency)");
    statistics_.context_device_bytes =
        context_free_before > context_free_after
            ? context_free_before - context_free_after
            : 0;
    batch_free_baseline_ = context_free_after;
    minimum_batch_free_ = context_free_after;
    statistics_.uniform_batch_size = batch.members.size();
    statistics_.last_library_setup_ns = ElapsedNanoseconds(library_setup_start);

    const Clock::time_point host_pack_start = Clock::now();
    host_row_offsets_.reserve(batch.structure.row_offsets.size());
    for (const std::size_t offset : batch.structure.row_offsets) {
      if (offset >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("GPU-02 CSR offset exceeds signed-32 limits");
      }
      host_row_offsets_.push_back(static_cast<std::int32_t>(offset));
    }
    host_column_indices_.reserve(batch.structure.column_indices.size());
    for (const std::size_t column : batch.structure.column_indices) {
      if (column >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("GPU-02 CSR column exceeds signed-32 limits");
      }
      host_column_indices_.push_back(static_cast<std::int32_t>(column));
    }

    PackValuesAndRhs(batch);
    statistics_.last_host_pack_ns = ElapsedNanoseconds(host_pack_start);

    const Clock::time_point upload_start = Clock::now();
    AllocateControlled(&device_row_offsets_, host_row_offsets_.size(),
                       "cudaMallocAsync(CSR offsets)");
    AllocateControlled(&device_column_indices_, host_column_indices_.size(),
                       "cudaMallocAsync(CSR columns)");
    AllocateControlled(&device_values_, host_values_.size(),
                       "cudaMallocAsync(matrix values)");
    AllocateControlled(&device_rhs_, host_rhs_.size(),
                       "cudaMallocAsync(right-hand sides)");
    AllocateControlled(&device_solution_, host_solution_.size(),
                       "cudaMallocAsync(uniform-batch solutions)");
    controlled_solution_bytes_ =
        CheckedMultiply(host_solution_.size(), sizeof(cuDoubleComplex),
                        "GPU-02 solution buffer");
    CheckCuda(cudaMemcpyAsync(device_row_offsets_, host_row_offsets_.data(),
                              host_row_offsets_.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync(CSR offsets)");
    CheckCuda(
        cudaMemcpyAsync(device_column_indices_, host_column_indices_.data(),
                        host_column_indices_.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(CSR columns)");
    CheckCuda(cudaMemcpyAsync(device_values_, host_values_.data(),
                              host_values_.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync(matrix values)");
    CheckCuda(cudaMemcpyAsync(device_rhs_, host_rhs_.data(),
                              host_rhs_.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync(right-hand sides)");
    CheckCuda(cudaMemsetAsync(device_solution_, 0, controlled_solution_bytes_,
                              stream_),
              "cudaMemsetAsync(uniform-batch solutions)");
    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(upload)");
    statistics_.last_upload_ns = ElapsedNanoseconds(upload_start);
    statistics_.controlled_device_bytes = controlled_bytes_;
    ObserveDeviceMemory();

    const std::int64_t dimension =
        static_cast<std::int64_t>(batch.structure.dimension);
    const std::int64_t nonzeros =
        static_cast<std::int64_t>(batch.structure.column_indices.size());
    const Clock::time_point matrix_setup_start = Clock::now();
    CheckCudss(cudssMatrixCreateCsr(
                   &matrix_, dimension, dimension, nonzeros,
                   device_row_offsets_, nullptr, device_column_indices_,
                   device_values_, CUDSS_R_32I, CUDSS_R_32I, CUDSS_C_64F,
                   CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
               "cudssMatrixCreateCsr");
    CheckCudss(cudssMatrixCreateDn(&rhs_, dimension, 1, dimension, device_rhs_,
                                   CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(rhs)");
    CheckCudss(cudssMatrixCreateDn(&solution_, dimension, 1, dimension,
                                   device_solution_, CUDSS_C_64F,
                                   CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(solution)");
    statistics_.last_matrix_setup_ns = ElapsedNanoseconds(matrix_setup_start);

    const Clock::time_point analysis_submit_start = Clock::now();
    CheckCuda(cudaEventRecord(analysis_start_, stream_),
              "cudaEventRecord(analysis start)");
    const bool inject_allocation_failure =
        !fault_injected_ &&
        options_.fault == CudaPreparedAcFault::kAllocationFailure;
    if (inject_allocation_failure) {
      fault_injected_ = true;
      tracker_.RejectAllocations(true);
    }
    const cudssStatus_t analysis_status =
        cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_, matrix_,
                     solution_, rhs_);
    if (analysis_status != CUDSS_STATUS_SUCCESS) {
      tracker_.RejectAllocations(false);
      CheckCudss(analysis_status, "cudssExecute(analysis)");
    }
    CheckCuda(cudaEventRecord(analysis_end_, stream_),
              "cudaEventRecord(analysis end)");
    statistics_.last_analysis_submit_ns =
        ElapsedNanoseconds(analysis_submit_start);
    const Clock::time_point analysis_sync_start = Clock::now();
    CheckCuda(cudaStreamSynchronize(stream_),
              "cudaStreamSynchronize(analysis)");
    statistics_.last_analysis_sync_ns = ElapsedNanoseconds(analysis_sync_start);
    const Clock::time_point analysis_status_start = Clock::now();
    CheckDataInfo("cuDSS uniform-batch analysis");
    statistics_.last_phase_status_ns +=
        ElapsedNanoseconds(analysis_status_start);
    tracker_.RejectAllocations(false);
    if (inject_allocation_failure) {
      throw std::runtime_error(
          "injected CUDA allocation failure was not reported by cuDSS");
    }
    statistics_.last_analysis_device_ns =
        EventElapsedNanoseconds(analysis_start_, analysis_end_);

    const Clock::time_point accounting_start = Clock::now();
    std::int64_t memory_estimates[16]{};
    std::size_t written = 0;
    CheckCudss(cudssDataGet(handle_, data_, CUDSS_DATA_MEMORY_ESTIMATES,
                            memory_estimates, sizeof(memory_estimates),
                            &written),
               "cudssDataGet(CUDSS_DATA_MEMORY_ESTIMATES)");
    if (written != sizeof(memory_estimates) || memory_estimates[1] < 0) {
      throw std::runtime_error(
          "cuDSS returned invalid device memory estimates");
    }
    statistics_.cudss_estimated_peak_device_bytes =
        static_cast<std::size_t>(memory_estimates[1]);
    statistics_.cudss_observed_peak_device_bytes = tracker_.peak();
    statistics_.cudss_outstanding_device_bytes = tracker_.current();
    ObserveDeviceMemory();
    UpdatePeakBatchMemory();
    statistics_.last_memory_accounting_ns +=
        ElapsedNanoseconds(accounting_start);

    generation_key_ = generation;
    prepared_structure_fingerprint_ = batch.structure.fingerprint;
    prepared_dimension_ = batch.structure.dimension;
    prepared_nonzeros_ = batch.structure.column_indices.size();
    ++statistics_.preparations;
    ++statistics_.structure_uploads;
    ++statistics_.values_rhs_uploads;
    ++statistics_.analyses;
  }

  void ObserveDeviceMemory() {
    std::size_t free = 0;
    std::size_t total = 0;
    CheckCuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo(batch peak)");
    minimum_batch_free_ = std::min(minimum_batch_free_, free);
    statistics_.cuda_mem_info_peak_device_bytes =
        batch_free_baseline_ > minimum_batch_free_
            ? batch_free_baseline_ - minimum_batch_free_
            : 0;
    statistics_.cudss_observed_peak_device_bytes = tracker_.peak();
    statistics_.cudss_outstanding_device_bytes = tracker_.current();
    UpdatePeakBatchMemory();
  }

  void UpdatePeakBatchMemory() {
    const std::size_t accounted =
        statistics_.controlled_device_bytes +
        std::max(statistics_.cudss_estimated_peak_device_bytes,
                 statistics_.cudss_observed_peak_device_bytes);
    statistics_.peak_batch_device_bytes =
        std::max(accounted, statistics_.cuda_mem_info_peak_device_bytes);
  }

  static void AppendCleanupError(std::string *errors,
                                 const std::string &message) {
    if (!errors->empty()) {
      *errors += "; ";
    }
    *errors += message;
  }

  static void RecordCudaCleanup(cudaError_t status, const char *operation,
                                std::string *errors) {
    if (status != cudaSuccess) {
      AppendCleanupError(errors, std::string(operation) +
                                     " failed with CUDA status " +
                                     std::to_string(static_cast<int>(status)) +
                                     ": " + cudaGetErrorString(status));
    }
  }

  static void RecordCudssCleanup(cudssStatus_t status, const char *operation,
                                 std::string *errors) {
    if (status != CUDSS_STATUS_SUCCESS) {
      AppendCleanupError(errors, std::string(operation) +
                                     " failed with cuDSS status " +
                                     std::to_string(static_cast<int>(status)));
    }
  }

  template <typename T>
  void FreeControlled(T **pointer, const char *description,
                      std::string *errors) {
    if (*pointer == nullptr) {
      return;
    }
    cudaError_t status = stream_ == nullptr ? cudaFree(*pointer)
                                            : cudaFreeAsync(*pointer, stream_);
    if (status != cudaSuccess && stream_ != nullptr) {
      RecordCudaCleanup(status, description, errors);
      status = cudaFree(*pointer);
      RecordCudaCleanup(status, "cudaFree cleanup fallback", errors);
    } else {
      RecordCudaCleanup(status, description, errors);
    }
    *pointer = nullptr;
  }

  [[nodiscard]] std::string ResetAndCollectErrors() {
    std::string errors;
    if (stream_ != nullptr) {
      RecordCudaCleanup(cudaStreamSynchronize(stream_),
                        "cudaStreamSynchronize(cleanup start)", &errors);
    }
    if (matrix_ != nullptr) {
      RecordCudssCleanup(cudssMatrixDestroy(matrix_),
                         "cudssMatrixDestroy(matrix)", &errors);
      matrix_ = nullptr;
    }
    if (rhs_ != nullptr) {
      RecordCudssCleanup(cudssMatrixDestroy(rhs_), "cudssMatrixDestroy(rhs)",
                         &errors);
      rhs_ = nullptr;
    }
    if (solution_ != nullptr) {
      RecordCudssCleanup(cudssMatrixDestroy(solution_),
                         "cudssMatrixDestroy(solution)", &errors);
      solution_ = nullptr;
    }
    if (data_ != nullptr && handle_ != nullptr) {
      RecordCudssCleanup(cudssDataDestroy(handle_, data_), "cudssDataDestroy",
                         &errors);
      data_ = nullptr;
    } else if (data_ != nullptr) {
      AppendCleanupError(&errors, "cuDSS data exists without a cleanup handle");
      data_ = nullptr;
    }

    FreeControlled(&device_solution_, "cudaFreeAsync(solution)", &errors);
    FreeControlled(&device_rhs_, "cudaFreeAsync(right-hand sides)", &errors);
    FreeControlled(&device_values_, "cudaFreeAsync(matrix values)", &errors);
    FreeControlled(&device_column_indices_, "cudaFreeAsync(CSR columns)",
                   &errors);
    FreeControlled(&device_row_offsets_, "cudaFreeAsync(CSR offsets)", &errors);

    if (config_ != nullptr) {
      RecordCudssCleanup(cudssConfigDestroy(config_), "cudssConfigDestroy",
                         &errors);
      config_ = nullptr;
    }
    if (handle_ != nullptr) {
      RecordCudssCleanup(cudssDestroy(handle_), "cudssDestroy", &errors);
      handle_ = nullptr;
    }
    if (stream_ != nullptr) {
      RecordCudaCleanup(cudaStreamSynchronize(stream_),
                        "cudaStreamSynchronize(cleanup end)", &errors);
    }

    const std::size_t outstanding = tracker_.current();
    statistics_.cudss_outstanding_device_bytes = outstanding;
    statistics_.last_release_outstanding_device_bytes = outstanding;
    if (outstanding != 0) {
      AppendCleanupError(&errors, "cuDSS custom allocator retained " +
                                      std::to_string(outstanding) +
                                      " device bytes");
    }

    if (analysis_start_ != nullptr) {
      RecordCudaCleanup(cudaEventDestroy(analysis_start_),
                        "cudaEventDestroy(analysis start)", &errors);
      analysis_start_ = nullptr;
    }
    if (analysis_end_ != nullptr) {
      RecordCudaCleanup(cudaEventDestroy(analysis_end_),
                        "cudaEventDestroy(analysis end)", &errors);
      analysis_end_ = nullptr;
    }
    if (factor_solve_start_ != nullptr) {
      RecordCudaCleanup(cudaEventDestroy(factor_solve_start_),
                        "cudaEventDestroy(factor/solve start)", &errors);
      factor_solve_start_ = nullptr;
    }
    if (factor_solve_end_ != nullptr) {
      RecordCudaCleanup(cudaEventDestroy(factor_solve_end_),
                        "cudaEventDestroy(factor/solve end)", &errors);
      factor_solve_end_ = nullptr;
    }
    if (stream_ != nullptr) {
      RecordCudaCleanup(cudaStreamDestroy(stream_), "cudaStreamDestroy",
                        &errors);
      stream_ = nullptr;
    }
    generation_key_.clear();
    prepared_structure_fingerprint_.clear();
    prepared_dimension_ = 0;
    prepared_nonzeros_ = 0;
    numeric_factors_exist_ = false;
    controlled_bytes_ = 0;
    controlled_solution_bytes_ = 0;
    batch_free_baseline_ = 0;
    minimum_batch_free_ = 0;
    host_row_offsets_.clear();
    host_column_indices_.clear();
    host_values_.clear();
    host_rhs_.clear();
    host_solution_.clear();
    statistics_.controlled_device_bytes = 0;
    statistics_.uniform_batch_size = 0;
    return errors;
  }

  void ResetNoThrow() noexcept {
    try {
      static_cast<void>(ResetAndCollectErrors());
    } catch (...) {
    }
  }

  CudaPreparedAcOptions options_;
  std::thread::id owner_;
  CudaPreparedAcStatistics statistics_;
  std::string generation_key_;
  std::string prepared_structure_fingerprint_;
  std::size_t prepared_dimension_ = 0;
  std::size_t prepared_nonzeros_ = 0;
  bool numeric_factors_exist_ = false;
  bool fault_injected_ = false;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t analysis_start_ = nullptr;
  cudaEvent_t analysis_end_ = nullptr;
  cudaEvent_t factor_solve_start_ = nullptr;
  cudaEvent_t factor_solve_end_ = nullptr;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t matrix_ = nullptr;
  cudssMatrix_t rhs_ = nullptr;
  cudssMatrix_t solution_ = nullptr;

  std::int32_t *device_row_offsets_ = nullptr;
  std::int32_t *device_column_indices_ = nullptr;
  cuDoubleComplex *device_values_ = nullptr;
  cuDoubleComplex *device_rhs_ = nullptr;
  cuDoubleComplex *device_solution_ = nullptr;
  std::size_t controlled_bytes_ = 0;
  std::size_t controlled_solution_bytes_ = 0;
  std::size_t batch_free_baseline_ = 0;
  std::size_t minimum_batch_free_ = 0;
  DeviceAllocationTracker tracker_;

  std::vector<std::int32_t> host_row_offsets_;
  std::vector<std::int32_t> host_column_indices_;
  std::vector<cuDoubleComplex> host_values_;
  std::vector<cuDoubleComplex> host_rhs_;
  std::vector<cuDoubleComplex> host_solution_;
};

CudaPreparedAcBatchBackend::CudaPreparedAcBatchBackend(
    CudaPreparedAcOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

CudaPreparedAcBatchBackend::~CudaPreparedAcBatchBackend() = default;

Result<PreparedAcBatchResult>
CudaPreparedAcBatchBackend::Execute(const PreparedAcBatch &batch) {
  return impl_->Execute(batch);
}

Result<bool> CudaPreparedAcBatchBackend::Release() { return impl_->Release(); }

const CudaPreparedAcStatistics &CudaPreparedAcBatchBackend::statistics() const {
  return impl_->statistics();
}

} // namespace ohmnivore
