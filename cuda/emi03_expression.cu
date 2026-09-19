#include "cuda/emi03_expression.h"
#include "cuda/emi03_expression_device.cuh"

#include "cpp/src/expression_internal.h"
#include "cuda/emi03_cuda_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace ohmnivore {
namespace {

using emi03_cuda_internal::BackendError;
using emi03_cuda_internal::CheckCuda;
using emi03_cuda_internal::Clock;
using emi03_cuda_internal::Elapsed;
using internal::ExportedExpressionNode;
using internal::ExportedExpressionOp;

using namespace emi03_device;

__global__ void EvaluatePrograms(
    const DeviceProgram *programs, std::uint32_t count,
    const ExportedExpressionNode *all_nodes, const std::uint32_t *all_ad,
    const std::uint32_t *all_dependencies, const double *state,
    DeviceResult *results, double *all_gradients, double *all_values,
    double *all_adjoints, std::uint32_t *all_stacks, unsigned char *all_stages,
    bool derivatives) {
  EvaluateProgram(blockIdx.x * blockDim.x + threadIdx.x, programs, count,
                  all_nodes, all_ad, all_dependencies, state, results,
                  all_gradients, all_values, all_adjoints, all_stacks,
                  all_stages, derivatives);
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { emi03_cuda_internal::FreeDevice(pointer_, bytes_); }
  void Allocate(std::size_t bytes) {
    if (bytes == 0)
      return;
    pointer_ = emi03_cuda_internal::AllocateDevice(bytes);
    bytes_ = bytes;
  }
  template <typename T> T *as() const { return static_cast<T *>(pointer_); }
  void *data() const { return pointer_; }

private:
  void *pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class ProgramBatch {
public:
  ~ProgramBatch() {
    if (stream_ && cudaStreamSynchronize(stream_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    if (stream_ && cudaStreamDestroy(stream_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
  }

  bool Matches(std::span<const CompiledExpression> expressions) const {
    if (expressions.size() != owners_.size())
      return false;
    for (std::size_t i = 0; i < expressions.size(); ++i)
      if (internal::ExpressionProgramIdentity(expressions[i]) !=
          internal::ExpressionProgramIdentity(owners_[i]))
        return false;
    return true;
  }

  void Prepare(std::span<const CompiledExpression> expressions) {
    const auto start = Clock::now();
    std::vector<ExportedExpressionNode> nodes;
    std::vector<std::uint32_t> reverse_ad;
    for (const auto &expression : expressions) {
      auto exported = internal::ExportExpressionProgram(expression);
      if (!exported.ok())
        throw BackendError(exported.error().code, exported.error().message);
      const auto &program = exported.value();
      if (program.nodes.empty() || program.nodes.size() > kMaximumNodes ||
          program.root >= program.nodes.size() || program.state_size > 512 ||
          program.dependencies.size() > kMaximumNodes ||
          nodes.size() + program.nodes.size() > kMaximumTotalNodes)
        throw BackendError(ErrorCode::kUnsupportedSize,
                           "EMI-03 expression program resource limit");
      if (!programs_.empty() &&
          program.state_size != programs_.front().state_size)
        throw BackendError(ErrorCode::kInvalidStructure,
                           "EMI-03 expression batch state dimensions differ");
      programs_.push_back(DeviceProgram{
          static_cast<std::uint32_t>(nodes.size()),
          static_cast<std::uint32_t>(reverse_ad.size()),
          static_cast<std::uint32_t>(dependencies_.size()),
          static_cast<std::uint32_t>(program.nodes.size()),
          static_cast<std::uint32_t>(program.reverse_ad_indices.size()),
          static_cast<std::uint32_t>(program.dependencies.size()), program.root,
          program.state_size, program.dialect});
      nodes.insert(nodes.end(), program.nodes.begin(), program.nodes.end());
      reverse_ad.insert(reverse_ad.end(), program.reverse_ad_indices.begin(),
                        program.reverse_ad_indices.end());
      dependencies_.insert(dependencies_.end(), program.dependencies.begin(),
                           program.dependencies.end());
    }
    owners_.assign(expressions.begin(), expressions.end());
    host_results_.resize(expressions.size());
    host_gradients_.resize(dependencies_.size());
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "EMI-03 create expression stream");
    device_programs_.Allocate(programs_.size() * sizeof(DeviceProgram));
    device_nodes_.Allocate(nodes.size() * sizeof(ExportedExpressionNode));
    device_ad_.Allocate(std::max<std::size_t>(1, reverse_ad.size()) *
                        sizeof(std::uint32_t));
    device_dependencies_.Allocate(
        std::max<std::size_t>(1, dependencies_.size()) * sizeof(std::uint32_t));
    // Allocate one unused element for empty state/dependency arrays, avoiding
    // null-pointer arithmetic inside a constant-only expression kernel.
    device_state_.Allocate(std::max<std::size_t>(1, programs_[0].state_size) *
                           8);
    device_gradients_.Allocate(std::max<std::size_t>(1, dependencies_.size()) *
                               8);
    device_results_.Allocate(programs_.size() * sizeof(DeviceResult));
    device_values_.Allocate(nodes.size() * sizeof(double));
    device_adjoints_.Allocate(nodes.size() * sizeof(double));
    device_stacks_.Allocate(programs_.size() * 65 * sizeof(std::uint32_t));
    device_stages_.Allocate(programs_.size() * 65 * sizeof(unsigned char));
    Upload(device_programs_.data(), programs_.data(),
           programs_.size() * sizeof(DeviceProgram));
    Upload(device_nodes_.data(), nodes.data(),
           nodes.size() * sizeof(ExportedExpressionNode));
    Upload(device_ad_.data(), reverse_ad.data(),
           reverse_ad.size() * sizeof(std::uint32_t));
    Upload(device_dependencies_.data(), dependencies_.data(),
           dependencies_.size() * sizeof(std::uint32_t));
    emi03_cuda_internal::Synchronize(stream_);
    auto &statistics = emi03_cuda_internal::Statistics();
    ++statistics.expression_program_uploads;
    statistics.expression_prepare_ns += Elapsed(start);
  }

  Result<std::vector<ExpressionEvaluation>>
  Evaluate(std::span<const double> state, bool derivatives) {
    using Outcome = Result<std::vector<ExpressionEvaluation>>;
    if (state.size() != programs_.front().state_size)
      return Outcome::Fail(ErrorCode::kInvalidStructure,
                           "expression state size mismatch");
    const auto start = Clock::now();
    Upload(device_state_.data(), state.data(), state.size_bytes());
    const auto count = static_cast<std::uint32_t>(programs_.size());
    EvaluatePrograms<<<(count + 63) / 64, 64, 0, stream_>>>(
        device_programs_.as<DeviceProgram>(), count,
        device_nodes_.as<ExportedExpressionNode>(),
        device_ad_.as<std::uint32_t>(),
        device_dependencies_.as<std::uint32_t>(), device_state_.as<double>(),
        device_results_.as<DeviceResult>(), device_gradients_.as<double>(),
        device_values_.as<double>(), device_adjoints_.as<double>(),
        device_stacks_.as<std::uint32_t>(), device_stages_.as<unsigned char>(),
        derivatives);
    CheckCuda(cudaGetLastError(), "EMI-03 expression kernel launch");
    const auto readback_start = Clock::now();
    Readback(host_results_.data(), device_results_.data(),
             host_results_.size() * sizeof(DeviceResult));
    if (derivatives)
      Readback(host_gradients_.data(), device_gradients_.data(),
               host_gradients_.size() * sizeof(double));
    emi03_cuda_internal::Synchronize(stream_);
    auto &statistics = emi03_cuda_internal::Statistics();
    statistics.readback_ns += Elapsed(readback_start);
    ++statistics.expression_batches;
    (derivatives ? statistics.expression_full_ad
                 : statistics.expression_value_only) += programs_.size();
    statistics.expression_evaluate_ns += Elapsed(start);
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kNonFiniteExpression))
      host_results_[0].value = std::numeric_limits<double>::quiet_NaN();
    if (emi03_cuda_internal::ConsumeFault(Emi03CudaFault::kWrongExpression))
      host_results_[0].value += std::max(1.0, std::abs(host_results_[0].value));
    std::vector<ExpressionEvaluation> evaluations;
    evaluations.reserve(programs_.size());
    for (std::size_t i = 0; i < programs_.size(); ++i) {
      const auto &result = host_results_[i];
      if (result.status == 2)
        return Outcome::Fail(ErrorCode::kInvalidStructure,
                             "invalid EMI-03 CUDA expression traversal");
      if (result.status != 0 || !(std::abs(result.value) <= 1e100))
        return Outcome::Fail(ErrorCode::kNonFinite,
                             "nonfinite expression value or domain");
      ExpressionEvaluation evaluation;
      evaluation.value = result.value;
      if (derivatives) {
        const auto &program = programs_[i];
        for (std::size_t j = 0; j < program.dependency_count; ++j) {
          const auto offset = program.dependency_offset + j;
          const double gradient = host_gradients_[offset];
          if (!(std::abs(gradient) <= 1e100))
            return Outcome::Fail(ErrorCode::kNonFinite,
                                 "nonfinite expression derivative");
          if (gradient != 0)
            evaluation.derivatives.emplace_back(dependencies_[offset],
                                                gradient);
        }
      }
      evaluations.push_back(std::move(evaluation));
    }
    return Outcome::Ok(std::move(evaluations));
  }

private:
  void Upload(void *destination, const void *source, std::size_t bytes) {
    if (bytes == 0)
      return;
    const auto start = Clock::now();
    CheckCuda(cudaMemcpyAsync(destination, source, bytes,
                              cudaMemcpyHostToDevice, stream_),
              "EMI-03 expression upload");
    auto &statistics = emi03_cuda_internal::Statistics();
    statistics.upload_bytes += bytes;
    statistics.upload_ns += Elapsed(start);
  }

  void Readback(void *destination, const void *source, std::size_t bytes) {
    if (bytes == 0)
      return;
    CheckCuda(cudaMemcpyAsync(destination, source, bytes,
                              cudaMemcpyDeviceToHost, stream_),
              "EMI-03 expression readback");
    emi03_cuda_internal::Statistics().readback_bytes += bytes;
  }

  std::vector<CompiledExpression> owners_;
  std::vector<DeviceProgram> programs_;
  std::vector<std::uint32_t> dependencies_;
  std::vector<DeviceResult> host_results_;
  std::vector<double> host_gradients_;
  cudaStream_t stream_ = nullptr;
  DeviceBuffer device_programs_, device_nodes_, device_ad_,
      device_dependencies_;
  DeviceBuffer device_state_, device_results_, device_gradients_;
  DeviceBuffer device_values_, device_adjoints_, device_stacks_, device_stages_;
};

thread_local std::vector<std::unique_ptr<ProgramBatch>> program_cache;

} // namespace

namespace emi03_cuda_internal {
void ResetExpressionCache() noexcept { program_cache.clear(); }
} // namespace emi03_cuda_internal

Result<std::vector<ExpressionEvaluation>>
EvaluateEmi03CudaExpressions(std::span<const CompiledExpression> expressions,
                             std::span<const double> state, bool derivatives) {
  using Outcome = Result<std::vector<ExpressionEvaluation>>;
  try {
    emi03_cuda_internal::RequireJob();
    if (expressions.empty())
      return Outcome::Ok({});
    if (expressions.size() > kMaximumPrograms || state.size() > 512)
      return Outcome::Fail(ErrorCode::kUnsupportedSize,
                           "EMI-03 expression batch resource limit");
    for (auto &entry : program_cache)
      if (entry->Matches(expressions))
        return entry->Evaluate(state, derivatives);
    if (program_cache.size() >= 16)
      return Outcome::Fail(ErrorCode::kUnsupportedSize,
                           "EMI-03 expression program cache limit");
    auto batch = std::make_unique<ProgramBatch>();
    batch->Prepare(expressions);
    auto *entry = batch.get();
    program_cache.push_back(std::move(batch));
    return entry->Evaluate(state, derivatives);
  } catch (const BackendError &error) {
    return Outcome::Fail(error.code, error.what());
  } catch (const std::bad_alloc &) {
    return Outcome::Fail(ErrorCode::kFactorization,
                         "EMI-03 expression host allocation failed");
  }
}

} // namespace ohmnivore
