#include "cuda/emi03_real_solver.h"

#include "cuda/emi03_cuda_internal.h"
#include "ohmnivore/solver.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <utility>
#include <vector>

#include <cudss.h>

namespace ohmnivore {
namespace {

using emi03_cuda_internal::BackendError;
using emi03_cuda_internal::CheckCuda;
using emi03_cuda_internal::Clock;
using emi03_cuda_internal::Elapsed;

struct JobState {
  cudaStream_t allocation_stream = nullptr;
  bool active = false;
  bool fault_consumed = false;
  bool library_allocation_failed = false;
  Emi03CudaOptions options;
  Emi03CudaStatistics statistics;
  std::atomic<std::size_t> current{0};
  std::atomic<std::size_t> peak{0};
  std::atomic<std::size_t> controlled{0};
  std::atomic<std::size_t> controlled_peak{0};
  std::atomic<std::size_t> cudss{0};
  std::atomic<std::size_t> cudss_peak{0};
};

thread_local JobState job;

void Maximum(std::atomic<std::size_t> &peak, std::size_t value) {
  std::size_t previous = peak.load();
  while (value > previous && !peak.compare_exchange_weak(previous, value)) {
  }
}

bool ConsumeFault(JobState &owner, Emi03CudaFault fault) {
  if (owner.active && !owner.fault_consumed && owner.options.fault == fault) {
    owner.fault_consumed = true;
    return true;
  }
  return false;
}

cudaError_t Allocate(JobState &job, void **pointer, std::size_t bytes,
                     bool library) {
  *pointer = nullptr;
  if (!job.active)
    return cudaErrorInvalidValue;
  if (bytes == 0)
    return cudaSuccess;
  if (ConsumeFault(job, Emi03CudaFault::kAllocationFailure)) {
    ++job.statistics.allocation_failures;
    job.library_allocation_failed = library;
    return cudaErrorMemoryAllocation;
  }
  std::size_t current = job.current.load();
  do {
    if (bytes > job.options.maximum_device_bytes ||
        current > job.options.maximum_device_bytes - bytes) {
      ++job.statistics.allocation_failures;
      job.library_allocation_failed = library;
      return cudaErrorMemoryAllocation;
    }
  } while (!job.current.compare_exchange_weak(current, current + bytes));
  // Complete allocation on a private nonblocking stream before exposing it to
  // an owner stream. Avoid the device-wide synchronization of legacy malloc.
  auto status = cudaMallocAsync(pointer, bytes, job.allocation_stream);
  if (status == cudaSuccess) {
    // Record ownership before synchronization: even an asynchronous error or
    // unsuccessful cleanup must retain the full allocation in the ledger.
    Maximum(job.peak, current + bytes);
    auto &owned = library ? job.cudss : job.controlled;
    auto &peak = library ? job.cudss_peak : job.controlled_peak;
    Maximum(peak, owned.fetch_add(bytes) + bytes);
    status = cudaStreamSynchronize(job.allocation_stream);
    if (status != cudaSuccess && *pointer) {
      const auto released = cudaFreeAsync(*pointer, job.allocation_stream);
      if (released != cudaSuccess ||
          cudaStreamSynchronize(job.allocation_stream) != cudaSuccess) {
        ++job.statistics.cleanup_failures;
        ++job.statistics.allocation_failures;
        job.library_allocation_failed = library;
        return status;
      }
      owned.fetch_sub(bytes);
      *pointer = nullptr;
    }
  }
  if (status != cudaSuccess) {
    job.current.fetch_sub(bytes);
    ++job.statistics.allocation_failures;
    job.library_allocation_failed = library;
    return status;
  }
  return cudaSuccess;
}

cudaError_t Free(JobState &job, void *pointer, std::size_t bytes,
                 bool library) {
  if (pointer == nullptr)
    return cudaSuccess;
  auto &owned = library ? job.cudss : job.controlled;
  if (owned.load() < bytes || job.current.load() < bytes) {
    ++job.statistics.cleanup_failures;
    return cudaErrorInvalidValue;
  }
  // Every controlled owner synchronizes its use stream before freeing. Library
  // callbacks additionally complete their supplied stream at the boundary
  // below.
  auto status = cudaFreeAsync(pointer, job.allocation_stream);
  if (status == cudaSuccess)
    status = cudaStreamSynchronize(job.allocation_stream);
  if (status != cudaSuccess) {
    ++job.statistics.cleanup_failures;
    return status;
  }
  owned.fetch_sub(bytes);
  job.current.fetch_sub(bytes);
  return cudaSuccess;
}

int LibraryAllocate(void *context, void **pointer, std::size_t bytes,
                    cudaStream_t) {
  auto &job = *static_cast<JobState *>(context);
  if (ConsumeFault(job, Emi03CudaFault::kCudssAllocationFailure)) {
    *pointer = nullptr;
    ++job.statistics.allocation_failures;
    job.library_allocation_failed = true;
    return static_cast<int>(cudaErrorMemoryAllocation);
  }
  return static_cast<int>(Allocate(job, pointer, bytes, true));
}

int LibraryFree(void *context, void *pointer, std::size_t bytes,
                cudaStream_t stream) {
  auto &owner = *static_cast<JobState *>(context);
  const auto status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    ++owner.statistics.cleanup_failures;
    return static_cast<int>(status);
  }
  return static_cast<int>(Free(owner, pointer, bytes, true));
}

void CheckCudss(cudssStatus_t status, const char *operation) {
  if (status != CUDSS_STATUS_SUCCESS)
    throw BackendError(status == CUDSS_STATUS_ALLOC_FAILED ||
                               job.library_allocation_failed
                           ? ErrorCode::kUnsupportedSize
                           : ErrorCode::kPreparedBackendFailure,
                       std::string(operation) + " failed: cuDSS status " +
                           std::to_string(static_cast<int>(status)));
}

std::string JsonString(const std::string &value) {
  std::ostringstream out;
  out << '"';
  constexpr char digits[] = "0123456789abcdef";
  for (unsigned char character : value) {
    if (character == '"' || character == '\\')
      out << '\\' << character;
    else if (character < 32)
      out << "\\u00" << digits[character >> 4] << digits[character & 15];
    else
      out << character;
  }
  out << '"';
  return out.str();
}

} // namespace

namespace emi03_cuda_internal {

void CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw BackendError(status == cudaErrorMemoryAllocation
                           ? ErrorCode::kUnsupportedSize
                           : ErrorCode::kPreparedBackendFailure,
                       std::string(operation) + ": " +
                           cudaGetErrorString(status));
}

void RequireJob() {
  if (!job.active)
    throw BackendError(ErrorCode::kUnsupported,
                       "EMI-03 CUDA execution requires an active explicit job");
}

bool ConsumeFault(Emi03CudaFault fault) {
  if (job.active && !job.fault_consumed && job.options.fault == fault) {
    job.fault_consumed = true;
    return true;
  }
  return false;
}

Emi03CudaStatistics &Statistics() { return job.statistics; }

void *AllocateDevice(std::size_t bytes) {
  RequireJob();
  void *pointer = nullptr;
  const auto status = Allocate(job, &pointer, bytes, false);
  if (status == cudaErrorMemoryAllocation)
    throw BackendError(
        ErrorCode::kUnsupportedSize,
        "EMI-03 CUDA tracked device allocation budget exhausted");
  CheckCuda(status, "EMI-03 cudaMalloc");
  return pointer;
}

void FreeDevice(void *pointer, std::size_t bytes) noexcept {
  static_cast<void>(Free(job, pointer, bytes, false));
}

void Synchronize(cudaStream_t stream) {
  const auto start = Clock::now();
  const auto status = cudaStreamSynchronize(stream);
  job.statistics.synchronization_ns += Elapsed(start);
  CheckCuda(status, "EMI-03 cudaStreamSynchronize");
}

} // namespace emi03_cuda_internal

Result<bool> BeginEmi03CudaJob(std::string job_id,
                               const Emi03CudaOptions &options) {
  if (job.active || job.current.load() != 0 ||
      job.statistics.live_factorizations != 0)
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "previous EMI-03 CUDA job is still active");
  if (job_id.empty() || job_id.size() > 1024 ||
      options.maximum_device_bytes == 0 ||
      options.maximum_device_bytes > kEmi03WorkerDeviceAllocationLimit)
    return Result<bool>::Fail(ErrorCode::kUnsupportedSize,
                              "invalid EMI-03 CUDA job identity or budget");
  job.options = options;
  job.statistics = {};
  job.statistics.job_id = std::move(job_id);
  job.statistics.maximum_device_bytes = options.maximum_device_bytes;
  job.peak = 0;
  job.controlled = 0;
  job.controlled_peak = 0;
  job.cudss = 0;
  job.cudss_peak = 0;
  job.fault_consumed = false;
  job.library_allocation_failed = false;
  const auto start = Clock::now();
  try {
    CheckCuda(cudaSetDevice(0), "EMI-03 cudaSetDevice(0)");
    static std::once_flag scheduling;
    std::call_once(scheduling, [] {
      CheckCuda(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),
                "EMI-03 blocking host synchronization");
    });
    CheckCuda(cudaStreamCreateWithFlags(&job.allocation_stream,
                                        cudaStreamNonBlocking),
              "EMI-03 private allocation stream");
    job.statistics.context_setup_ns = Elapsed(start);
    job.active = true;
    return Result<bool>::Ok(true);
  } catch (const BackendError &error) {
    return Result<bool>::Fail(error.code, error.what());
  }
}

Emi03CudaStatistics SnapshotEmi03CudaJob() {
  auto statistics = job.statistics;
  statistics.current_device_bytes = job.current.load();
  statistics.outstanding_device_bytes = statistics.current_device_bytes;
  statistics.peak_device_bytes = job.peak.load();
  statistics.controlled_peak_device_bytes = job.controlled_peak.load();
  statistics.cudss_peak_device_bytes = job.cudss_peak.load();
  return statistics;
}

Result<Emi03CudaStatistics> EndEmi03CudaJob() {
  if (!job.active)
    return Result<Emi03CudaStatistics>::Fail(ErrorCode::kInvalidStructure,
                                             "no EMI-03 CUDA job is active");
  emi03_cuda_internal::ResetExpressionCache();
  if (job.statistics.live_factorizations != 0 || job.current.load() != 0 ||
      job.statistics.cleanup_failures != 0)
    return Result<Emi03CudaStatistics>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "EMI-03 CUDA cleanup left live resources or reported an error");
  const auto destroyed = cudaStreamDestroy(job.allocation_stream);
  if (destroyed != cudaSuccess) {
    ++job.statistics.cleanup_failures;
    return Result<Emi03CudaStatistics>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "EMI-03 allocation stream cleanup failed");
  }
  job.allocation_stream = nullptr;
  job.active = false;
  return Result<Emi03CudaStatistics>::Ok(SnapshotEmi03CudaJob());
}

std::string Emi03CudaStatisticsJson(const Emi03CudaStatistics &statistics) {
  std::ostringstream out;
  out << "{\"schema\":\"emi03-cuda-v1\",\"backend\":\"cuda-fp64\","
         "\"gpu_fallbacks\":0,\"job_id\":"
      << JsonString(statistics.job_id) << ",\"transient_algorithm\":"
      << JsonString(statistics.transient_algorithm);
#define EMI03_JSON_FIELD(name) out << ",\"" #name "\":" << statistics.name
  EMI03_JSON_FIELD(current_device_bytes);
  EMI03_JSON_FIELD(outstanding_device_bytes);
  EMI03_JSON_FIELD(peak_device_bytes);
  EMI03_JSON_FIELD(controlled_peak_device_bytes);
  EMI03_JSON_FIELD(cudss_peak_device_bytes);
  EMI03_JSON_FIELD(cudss_estimated_peak_device_bytes);
  EMI03_JSON_FIELD(maximum_device_bytes);
  EMI03_JSON_FIELD(live_factorizations);
  EMI03_JSON_FIELD(analyses);
  EMI03_JSON_FIELD(factorizations);
  EMI03_JSON_FIELD(refactorizations);
  EMI03_JSON_FIELD(reuses);
  EMI03_JSON_FIELD(solves);
  EMI03_JSON_FIELD(successful_solves);
  EMI03_JSON_FIELD(refinements);
  EMI03_JSON_FIELD(rejected_pivots);
  EMI03_JSON_FIELD(allocation_failures);
  EMI03_JSON_FIELD(cleanup_failures);
  EMI03_JSON_FIELD(expression_batches);
  EMI03_JSON_FIELD(expression_full_ad);
  EMI03_JSON_FIELD(expression_value_only);
  EMI03_JSON_FIELD(expression_program_uploads);
  EMI03_JSON_FIELD(upload_bytes);
  EMI03_JSON_FIELD(readback_bytes);
  EMI03_JSON_FIELD(context_setup_ns);
  EMI03_JSON_FIELD(solver_setup_ns);
  EMI03_JSON_FIELD(analysis_ns);
  EMI03_JSON_FIELD(factor_ns);
  EMI03_JSON_FIELD(solve_ns);
  EMI03_JSON_FIELD(synchronization_ns);
  EMI03_JSON_FIELD(upload_ns);
  EMI03_JSON_FIELD(readback_ns);
  EMI03_JSON_FIELD(validation_ns);
  EMI03_JSON_FIELD(expression_prepare_ns);
  EMI03_JSON_FIELD(expression_evaluate_ns);
#undef EMI03_JSON_FIELD
  out << '}';
  return out.str();
}

class SparseRealFactorization::Impl {
public:
  explicit Impl(const CsrMatrix &matrix)
      : rows_(matrix.rows), row_offsets_(matrix.row_offsets),
        columns_(matrix.column_indices) {
    ++job.statistics.live_factorizations;
  }

  ~Impl() {
    // All submitted work is synchronized before its result is inspected.
    // Destruction also synchronizes partial preparation before releasing data.
    if (stream_ && cudaStreamSynchronize(stream_) != cudaSuccess)
      ++job.statistics.cleanup_failures;
    const auto destroy = [](cudssStatus_t status) {
      if (status != CUDSS_STATUS_SUCCESS)
        ++job.statistics.cleanup_failures;
    };
    if (a_)
      destroy(cudssMatrixDestroy(a_));
    if (b_)
      destroy(cudssMatrixDestroy(b_));
    if (x_)
      destroy(cudssMatrixDestroy(x_));
    if (data_)
      destroy(cudssDataDestroy(handle_, data_));
    if (config_)
      destroy(cudssConfigDestroy(config_));
    if (handle_)
      destroy(cudssDestroy(handle_));
    emi03_cuda_internal::FreeDevice(device_rows_, row_offsets_.size() * 4);
    emi03_cuda_internal::FreeDevice(device_columns_, columns_.size() * 4);
    emi03_cuda_internal::FreeDevice(device_values_, columns_.size() * 8);
    emi03_cuda_internal::FreeDevice(device_rhs_, rows_ * 8);
    emi03_cuda_internal::FreeDevice(device_solution_, rows_ * 8);
    if (stream_ && cudaStreamDestroy(stream_) != cudaSuccess)
      ++job.statistics.cleanup_failures;
    --job.statistics.live_factorizations;
  }

  void Prepare(const CsrMatrix &matrix) {
    if (rows_ == 0)
      return;
    const auto start = Clock::now();
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "EMI-03 create solver stream");
    CheckCudss(cudssCreate(&handle_), "EMI-03 cudssCreate");
    CheckCudss(cudssSetStream(handle_, stream_), "EMI-03 cudssSetStream");
    CheckCudss(cudssConfigCreate(&config_), "EMI-03 cudssConfigCreate");
    cudssDeviceMemHandler_t handler{};
    handler.ctx = &job;
    handler.device_alloc = LibraryAllocate;
    handler.device_free = LibraryFree;
    std::snprintf(handler.name, sizeof(handler.name), "%s", "emi03-budget");
    CheckCudss(cudssSetDeviceMemHandler(handle_, &handler),
               "EMI-03 cuDSS device allocator");
    CheckCudss(cudssDataCreate(handle_, &data_), "EMI-03 cudssDataCreate");
    const cudssReorderingAlg_t reordering = CUDSS_REORDERING_ALG_BTF_COLAMD;
    const cudssFactorizationAlg_t factor = CUDSS_FACTORIZATION_ALG_DEFAULT;
    const cudssPivotType_t pivot = CUDSS_PIVOT_GLOBAL_COL;
    const cudssPivotEpsilonAlg_t epsilon_algorithm =
        CUDSS_PIVOT_EPSILON_ALG_STATIC;
    const cudssMatchingAlg_t matching = CUDSS_MATCHING_ALG_NONE;
    Configure(CUDSS_CONFIG_REORDERING_ALG, reordering);
    Configure(CUDSS_CONFIG_FACTORIZATION_ALG, factor);
    Configure(CUDSS_CONFIG_PIVOT_TYPE, pivot);
    Configure(CUDSS_CONFIG_PIVOT_THRESHOLD, 1.0);
    Configure(CUDSS_CONFIG_PIVOT_EPSILON, std::numeric_limits<double>::min());
    Configure(CUDSS_CONFIG_PIVOT_EPSILON_ALG, epsilon_algorithm);
    Configure(CUDSS_CONFIG_MATCHING_ALG, matching);
    Configure(CUDSS_CONFIG_IR_N_STEPS, 0);
    Configure(CUDSS_CONFIG_DETERMINISTIC_MODE, 0);
    Configure(CUDSS_CONFIG_HYBRID_MEMORY_MODE, 0);
    Configure(CUDSS_CONFIG_HYBRID_EXECUTE_MODE, 0);
    Configure(CUDSS_CONFIG_HOST_NTHREADS, 1);
    Configure(CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY, 0);
    // A complete dense pair of factors bounds fill for this admitted n<=512.
    Configure(CUDSS_CONFIG_MAX_LU_NNZ,
              static_cast<std::int64_t>(2 * rows_ * rows_));

    host_rows_.assign(row_offsets_.begin(), row_offsets_.end());
    host_columns_.assign(columns_.begin(), columns_.end());
    device_rows_ = static_cast<std::int32_t *>(
        emi03_cuda_internal::AllocateDevice(host_rows_.size() * 4));
    device_columns_ = static_cast<std::int32_t *>(
        emi03_cuda_internal::AllocateDevice(host_columns_.size() * 4));
    device_values_ = static_cast<double *>(
        emi03_cuda_internal::AllocateDevice(columns_.size() * 8));
    device_rhs_ =
        static_cast<double *>(emi03_cuda_internal::AllocateDevice(rows_ * 8));
    device_solution_ =
        static_cast<double *>(emi03_cuda_internal::AllocateDevice(rows_ * 8));
    Upload(device_rows_, host_rows_.data(), host_rows_.size() * 4);
    Upload(device_columns_, host_columns_.data(), host_columns_.size() * 4);
    Upload(device_values_, matrix.values.data(), matrix.values.size() * 8);
    CheckCuda(cudaMemsetAsync(device_rhs_, 0, rows_ * 8, stream_),
              "EMI-03 initialize solver RHS");
    CheckCuda(cudaMemsetAsync(device_solution_, 0, rows_ * 8, stream_),
              "EMI-03 initialize solver result");
    emi03_cuda_internal::Synchronize(stream_);
    CheckCudss(cudssMatrixCreateCsr(&a_, static_cast<std::int64_t>(rows_),
                                    static_cast<std::int64_t>(rows_),
                                    static_cast<std::int64_t>(columns_.size()),
                                    device_rows_, nullptr, device_columns_,
                                    device_values_, CUDSS_R_32I, CUDSS_R_32I,
                                    CUDSS_R_64F, CUDSS_MTYPE_GENERAL,
                                    CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
               "EMI-03 cudssMatrixCreateCsr");
    CheckCudss(cudssMatrixCreateDn(&b_, rows_, 1, rows_, device_rhs_,
                                   CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
               "EMI-03 cudssMatrixCreateDn RHS");
    CheckCudss(cudssMatrixCreateDn(&x_, rows_, 1, rows_, device_solution_,
                                   CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
               "EMI-03 cudssMatrixCreateDn solution");
    job.statistics.solver_setup_ns += Elapsed(start);
    const auto analysis_start = Clock::now();
    Execute(CUDSS_PHASE_ANALYSIS);
    job.statistics.analysis_ns += Elapsed(analysis_start);
    ++statistics_.symbolic_analyses;
    ++job.statistics.analyses;
    std::int64_t estimates[16]{};
    std::size_t written = 0;
    CheckCudss(cudssDataGet(handle_, data_, CUDSS_DATA_MEMORY_ESTIMATES,
                            estimates, sizeof(estimates), &written),
               "EMI-03 cuDSS memory estimate");
    if (written != sizeof(estimates) || estimates[1] < 0)
      throw BackendError(ErrorCode::kPreparedBackendFailure,
                         "invalid EMI-03 cuDSS memory estimate");
    job.statistics.cudss_estimated_peak_device_bytes =
        std::max(job.statistics.cudss_estimated_peak_device_bytes,
                 static_cast<std::size_t>(estimates[1]));
    if (static_cast<std::uint64_t>(estimates[1]) >
        job.options.maximum_device_bytes)
      throw BackendError(ErrorCode::kUnsupportedSize,
                         "EMI-03 cuDSS memory estimate exceeds worker budget");
  }

  Result<std::vector<double>> Solve(const CsrMatrix &matrix,
                                    const std::vector<double> &rhs,
                                    std::size_t remaining_refinements) {
    // The alternate executable retains public hostile-input admission even
    // for prepared callers. Ordinary CPU admission optimization is unchanged.
    auto converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok())
      return Result<std::vector<double>>::Fail(converted.error().code,
                                               converted.error().message);
    if (matrix.rows != rows_ || matrix.row_offsets != row_offsets_ ||
        matrix.column_indices != columns_)
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          "numeric refactorization requires the analyzed CSR pattern");
    if (rhs.size() != rows_)
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          "matrix and right-hand-side dimensions disagree");
    for (double value : rhs)
      if (!std::isfinite(value))
        return Result<std::vector<double>>::Fail(
            ErrorCode::kNonFinite,
            "right-hand side contains a non-finite value");
    if (rows_ == 0) {
      ++statistics_.solves;
      ++job.statistics.successful_solves;
      return Result<std::vector<double>>::Ok({});
    }
    const bool refactor = numeric_exists_ && last_values_ != matrix.values;
    bool permit_fresh_retry = refactor;
    try {
      if (!numeric_exists_ || refactor)
        Factor(matrix, refactor);
      else {
        ++statistics_.numeric_reuses;
        ++job.statistics.reuses;
      }
      auto result = SolveAndValidate(matrix, rhs, &remaining_refinements);
      if (result.ok()) {
        ++statistics_.solves;
        ++job.statistics.successful_solves;
      } else if (refactor &&
                 result.error().code == ErrorCode::kSolutionValidation) {
        ++statistics_.numeric_refactorization_fallbacks;
        permit_fresh_retry = false;
        Factor(matrix, false);
        result = SolveAndValidate(matrix, rhs, &remaining_refinements);
        if (result.ok()) {
          ++statistics_.solves;
          ++job.statistics.successful_solves;
        }
      }
      return result;
    } catch (const BackendError &error) {
      if (!permit_fresh_retry || (error.code != ErrorCode::kSingular &&
                                  error.code != ErrorCode::kFactorization))
        throw;
      ++statistics_.numeric_refactorization_fallbacks;
      Factor(matrix, false);
      auto result = SolveAndValidate(matrix, rhs, &remaining_refinements);
      if (result.ok()) {
        ++statistics_.solves;
        ++job.statistics.successful_solves;
      }
      return result;
    }
  }

  SparseSolverStatistics statistics_;

private:
  template <typename T> void Configure(cudssConfigParam_t parameter, T value) {
    CheckCudss(cudssConfigSet(config_, parameter, &value, sizeof(value)),
               "EMI-03 cudssConfigSet");
  }

  void Upload(void *destination, const void *source, std::size_t bytes) {
    const auto start = Clock::now();
    CheckCuda(cudaMemcpyAsync(destination, source, bytes,
                              cudaMemcpyHostToDevice, stream_),
              "EMI-03 solver upload");
    job.statistics.upload_bytes += bytes;
    job.statistics.upload_ns += Elapsed(start);
  }

  void Execute(int phase) {
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kCudssFailure))
      throw BackendError(ErrorCode::kPreparedBackendFailure,
                         "injected EMI-03 cuDSS failure");
    CheckCudss(cudssExecute(handle_, phase, config_, data_, a_, x_, b_),
               "EMI-03 cudssExecute");
    emi03_cuda_internal::Synchronize(stream_);
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kCudaFailure))
      throw BackendError(ErrorCode::kPreparedBackendFailure,
                         "injected EMI-03 CUDA synchronization failure");
    int info = 0;
    std::size_t written = 0;
    CheckCudss(cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &info,
                            sizeof(info), &written),
               "EMI-03 cudssDataGet INFO");
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kDataInfoFailure))
      throw BackendError(ErrorCode::kPreparedBackendFailure,
                         "injected EMI-03 cuDSS asynchronous failure");
    if (written != sizeof(info) || info != 0)
      throw BackendError(ErrorCode::kFactorization,
                         "EMI-03 cuDSS asynchronous data-info failure " +
                             std::to_string(info));
  }

  void Factor(const CsrMatrix &matrix, bool refactor) {
    const auto start = Clock::now();
    // Device errors remain attached to cuDSS data until explicitly cleared.
    // Only an explicit fresh GPU factor retry reaches this after a failure.
    const int clear_info = 0;
    CheckCudss(cudssDataSet(handle_, data_, CUDSS_DATA_INFO, &clear_info,
                            sizeof(clear_info)),
               "EMI-03 reset cuDSS data info before factorization");
    row_scales_.assign(rows_, 1.0);
    scaled_values_.resize(matrix.values.size());
    for (std::size_t row = 0; row < rows_; ++row) {
      double maximum = 0;
      for (std::size_t entry = matrix.row_offsets[row];
           entry < matrix.row_offsets[row + 1]; ++entry)
        maximum = std::max(maximum, std::abs(matrix.values[entry]));
      if (maximum != 0)
        row_scales_[row] = maximum;
      for (std::size_t entry = matrix.row_offsets[row];
           entry < matrix.row_offsets[row + 1]; ++entry) {
        scaled_values_[entry] = matrix.values[entry] / row_scales_[row];
        if (matrix.values[entry] != 0 && scaled_values_[entry] == 0)
          throw BackendError(
              ErrorCode::kSolutionValidation,
              "EMI-03 row equilibration underflows matrix entry");
      }
    }
    Upload(device_values_, scaled_values_.data(), scaled_values_.size() * 8);
    numeric_exists_ = false;
    last_values_.clear();
    Execute(refactor ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION);
    std::int32_t pivots = 0;
    std::size_t written = 0;
    CheckCudss(cudssDataGet(handle_, data_, CUDSS_DATA_NPIVOTS, &pivots,
                            sizeof(pivots), &written),
               "EMI-03 cudssDataGet NPIVOTS");
    if (written != sizeof(pivots) || pivots < 0)
      throw BackendError(ErrorCode::kPreparedBackendFailure,
                         "invalid EMI-03 cuDSS pivot count");
    if (pivots != 0) {
      job.statistics.rejected_pivots += static_cast<std::size_t>(pivots);
      throw BackendError(ErrorCode::kSingular,
                         "EMI-03 refuses perturbed pivots, including zero-RHS "
                         "accepted-Jacobian checks");
    }
    if (refactor) {
      ++statistics_.numeric_refactorizations;
      ++job.statistics.refactorizations;
    } else {
      ++statistics_.numeric_factorizations;
      ++job.statistics.factorizations;
    }
    numeric_exists_ = true;
    last_values_ = matrix.values;
    job.statistics.factor_ns += Elapsed(start);
  }

  std::vector<double> TriangularSolve(const std::vector<double> &rhs) {
    const auto start = Clock::now();
    scaled_rhs_.resize(rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      scaled_rhs_[row] = rhs[row] / row_scales_[row];
      if (!std::isfinite(scaled_rhs_[row]) ||
          (rhs[row] != 0 && scaled_rhs_[row] == 0))
        throw BackendError(
            ErrorCode::kSolutionValidation,
            "EMI-03 row equilibration overflows or underflows RHS");
    }
    Upload(device_rhs_, scaled_rhs_.data(), scaled_rhs_.size() * 8);
    Execute(CUDSS_PHASE_SOLVE);
    std::vector<double> solution(rows_);
    const auto readback_start = Clock::now();
    CheckCuda(cudaMemcpyAsync(solution.data(), device_solution_, rows_ * 8,
                              cudaMemcpyDeviceToHost, stream_),
              "EMI-03 solution readback");
    emi03_cuda_internal::Synchronize(stream_);
    job.statistics.readback_bytes += rows_ * 8;
    job.statistics.readback_ns += Elapsed(readback_start);
    job.statistics.solve_ns += Elapsed(start);
    ++job.statistics.solves;
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kNonFiniteSolve))
      solution[0] = std::numeric_limits<double>::quiet_NaN();
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kWrongSolve))
      solution[0] += std::max(1.0, std::abs(solution[0]));
    return solution;
  }

  Result<std::vector<double>>
  SolveAndValidate(const CsrMatrix &matrix, const std::vector<double> &rhs,
                   std::size_t *remaining_refinements) {
    auto solution = TriangularSolve(rhs);
    const auto validate = [&]() {
      const auto start = Clock::now();
      auto result = ValidateSparseSolution(matrix, rhs, solution);
      job.statistics.validation_ns += Elapsed(start);
      return result;
    };
    auto validation = validate();
    bool corrected = false;
    while (*remaining_refinements > 0 &&
           ((validation.ok() && !corrected) ||
            (!validation.ok() &&
             validation.error().code == ErrorCode::kSolutionValidation))) {
      std::vector<double> correction(rows_);
      const auto residual_start = Clock::now();
      for (std::size_t row = 0; row < rows_; ++row) {
        long double product = 0;
        for (std::size_t entry = matrix.row_offsets[row];
             entry < matrix.row_offsets[row + 1]; ++entry)
          product +=
              static_cast<long double>(matrix.values[entry]) *
              static_cast<long double>(solution[matrix.column_indices[entry]]);
        correction[row] =
            static_cast<double>(static_cast<long double>(rhs[row]) - product);
        if (!std::isfinite(correction[row]))
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite,
              "iterative refinement residual is not finite FP64");
      }
      job.statistics.validation_ns += Elapsed(residual_start);
      if (std::all_of(correction.begin(), correction.end(),
                      [](double value) { return value == 0; }))
        break;
      corrected = true;
      --*remaining_refinements;
      ++statistics_.iterative_refinement_solves;
      ++job.statistics.refinements;
      correction = TriangularSolve(correction);
      for (std::size_t index = 0; index < rows_; ++index) {
        solution[index] += correction[index];
        if (!std::isfinite(correction[index]) ||
            !std::isfinite(solution[index]))
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite,
              "iterative refinement update is not finite FP64");
      }
      validation = validate();
    }
    if (!validation.ok())
      return Result<std::vector<double>>::Fail(validation.error().code,
                                               validation.error().message);
    return Result<std::vector<double>>::Ok(std::move(solution));
  }

  std::size_t rows_;
  std::vector<std::size_t> row_offsets_, columns_;
  std::vector<std::int32_t> host_rows_, host_columns_;
  std::vector<double> last_values_;
  std::vector<double> row_scales_, scaled_values_, scaled_rhs_;
  cudaStream_t stream_ = nullptr;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t a_ = nullptr, b_ = nullptr, x_ = nullptr;
  std::int32_t *device_rows_ = nullptr, *device_columns_ = nullptr;
  double *device_values_ = nullptr, *device_rhs_ = nullptr,
         *device_solution_ = nullptr;
  bool numeric_exists_ = false;
};

SparseRealFactorization::SparseRealFactorization(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}
SparseRealFactorization::~SparseRealFactorization() = default;
SparseRealFactorization::SparseRealFactorization(
    SparseRealFactorization &&) noexcept = default;
SparseRealFactorization &SparseRealFactorization::operator=(
    SparseRealFactorization &&) noexcept = default;

Result<std::unique_ptr<SparseRealFactorization>>
SparseRealFactorization::Analyze(const CsrMatrix &matrix) {
  using Outcome = Result<std::unique_ptr<SparseRealFactorization>>;
  try {
    emi03_cuda_internal::RequireJob();
    auto pattern = ConvertCsrToSolverCsc(matrix);
    if (!pattern.ok())
      return Outcome::Fail(pattern.error().code, pattern.error().message);
    if (matrix.rows > 512)
      return Outcome::Fail(ErrorCode::kUnsupportedSize,
                           "EMI-03 CUDA matrix exceeds 512 unknowns");
    for (std::size_t row = 0; row < matrix.rows; ++row)
      if (matrix.row_offsets[row] == matrix.row_offsets[row + 1] ||
          pattern.value().column_offsets[row] ==
              pattern.value().column_offsets[row + 1])
        return Outcome::Fail(ErrorCode::kSingular,
                             "EMI-03 structurally empty matrix row or column");
    auto implementation = std::make_unique<Impl>(matrix);
    implementation->Prepare(matrix);
    return Outcome::Ok(std::unique_ptr<SparseRealFactorization>(
        new SparseRealFactorization(std::move(implementation))));
  } catch (const BackendError &error) {
    return Outcome::Fail(error.code, error.what());
  } catch (const std::bad_alloc &) {
    return Outcome::Fail(ErrorCode::kFactorization,
                         "EMI-03 solver host allocation failed");
  }
}

Result<std::vector<double>>
SparseRealFactorization::FactorAndSolve(const CsrMatrix &matrix,
                                        const std::vector<double> &rhs) {
  return FactorAndSolveRefined(matrix, rhs, 0);
}

Result<std::vector<double>> SparseRealFactorization::FactorAndSolveRefined(
    const CsrMatrix &matrix, const std::vector<double> &rhs,
    std::size_t maximum_refinements) {
  if (maximum_refinements > 4)
    return Result<std::vector<double>>::Fail(
        ErrorCode::kUnsupportedSize,
        "iterative refinement allows at most four corrections");
  try {
    emi03_cuda_internal::RequireJob();
    return implementation_->Solve(matrix, rhs, maximum_refinements);
  } catch (const BackendError &error) {
    return Result<std::vector<double>>::Fail(error.code, error.what());
  } catch (const std::bad_alloc &) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kFactorization, "EMI-03 numeric host allocation failed");
  }
}

Result<std::vector<double>> SparseRealFactorization::FactorAndSolveAdmitted(
    const CsrMatrix &matrix, const std::vector<double> &rhs,
    std::size_t maximum_refinements) {
  return FactorAndSolveRefined(matrix, rhs, maximum_refinements);
}

const SparseSolverStatistics &SparseRealFactorization::statistics() const {
  return implementation_->statistics_;
}

} // namespace ohmnivore
