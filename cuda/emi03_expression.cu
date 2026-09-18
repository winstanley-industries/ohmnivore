#include "cuda/emi03_expression.h"

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

constexpr std::size_t kMaximumNodes = 512;
constexpr std::size_t kMaximumPrograms = 512;
constexpr std::size_t kMaximumTotalNodes = 16384;

struct DeviceProgram {
  std::uint32_t node_offset, ad_offset, dependency_offset;
  std::uint32_t nodes, ad_count, dependency_count;
  std::uint32_t root, state_size;
  ExpressionDialect dialect;
};

struct DeviceResult {
  double value;
  std::uint32_t status;
};

__device__ bool Bounded(double value) { return fabs(value) <= 1e100; }

__device__ double Denominator(double value) {
  return value + (value >= 0 ? 1e-32 : -1e-32);
}

__device__ void AddAdjoint(const ExportedExpressionNode *nodes,
                           double *adjoints, std::uint32_t child,
                           double adjoint, double factor, bool *error) {
  if (nodes[child].constant)
    return;
  const double term = adjoint * factor;
  adjoints[child] += term;
  if (!Bounded(factor) || !Bounded(term) || !Bounded(adjoints[child]))
    *error = true;
}

// Each thread evaluates one complete expression. The explicit DFS stack
// preserves CPU left-to-right evaluation and lazy IF domain checks. Inactive
// branches retain zero adjoints and never contribute to reverse AD.
__global__ void EvaluatePrograms(
    const DeviceProgram *programs, std::uint32_t count,
    const ExportedExpressionNode *all_nodes, const std::uint32_t *all_ad,
    const std::uint32_t *all_dependencies, const double *state,
    DeviceResult *results, double *all_gradients, double *all_values,
    double *all_adjoints, std::uint32_t *all_stacks, unsigned char *all_stages,
    bool derivatives) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const auto program = programs[index];
  const auto *nodes = all_nodes + program.node_offset;
  const auto *dependencies = all_dependencies + program.dependency_offset;
  auto *gradient = all_gradients + program.dependency_offset;
  DeviceResult result{0, 0};
  for (std::uint32_t i = 0; i < program.dependency_count; ++i) {
    if (!Bounded(state[dependencies[i]])) {
      result.status = 1;
      results[index] = result;
      return;
    }
    gradient[i] = 0;
  }
  // Dynamic indexing of large thread-local arrays creates opaque driver-backed
  // local-memory residency. Size this workspace to the actual program nodes
  // instead, and charge every byte to the shared device-allocation ledger.
  auto *values = all_values + program.node_offset;
  auto *adjoints = all_adjoints + program.node_offset;
  for (std::uint32_t i = 0; i < program.nodes; ++i) {
    values[i] = 0;
    adjoints[i] = 0;
  }
  auto *stack = all_stacks + index * 65;
  auto *stage = all_stages + index * 65;
  int depth = 0;
  stack[0] = program.root;
  stage[0] = 0;
  bool error = false;
  while (depth >= 0 && !error) {
    const auto node_index = stack[depth];
    const auto node = nodes[node_index];
    double value = 0;
    bool done = true;
    if (node.op == ExportedExpressionOp::kConstant)
      value = node.value;
    else if (node.op == ExportedExpressionOp::kState)
      value = state[node.first];
    else if (stage[depth] == 0) {
      stage[depth] = 1;
      ++depth;
      if (depth == 65) {
        result.status = 2;
        break;
      }
      stack[depth] = node.first;
      stage[depth] = 0;
      done = false;
    } else if (stage[depth] == 1) {
      const double a = values[node.first];
      if (node.op == ExportedExpressionOp::kNegate)
        value = -a;
      else if (node.op == ExportedExpressionOp::kExp)
        value = program.dialect == ExpressionDialect::kBehavioral && a > 14
                    ? 1202604.284 * (a - 13)
                    : exp(a);
      else {
        stage[depth] = node.op == ExportedExpressionOp::kIf ? 3 : 2;
        ++depth;
        if (depth == 65) {
          result.status = 2;
          break;
        }
        stack[depth] = node.op == ExportedExpressionOp::kIf
                           ? (a != 0 ? node.second : node.third)
                           : node.second;
        stage[depth] = 0;
        done = false;
      }
    } else if (stage[depth] == 3) {
      value = values[values[node.first] != 0 ? node.second : node.third];
    } else {
      const double a = values[node.first], b = values[node.second];
      switch (node.op) {
      case ExportedExpressionOp::kAdd:
        value = a + b;
        break;
      case ExportedExpressionOp::kSubtract:
        value = a - b;
        break;
      case ExportedExpressionOp::kMultiply:
        value = a * b;
        break;
      case ExportedExpressionOp::kDivide:
        value = a / (program.dialect == ExpressionDialect::kBehavioral
                         ? Denominator(b)
                         : b);
        break;
      case ExportedExpressionOp::kPower:
        value = pow(fabs(a), b);
        break;
      case ExportedExpressionOp::kLess:
        value = a < b ? 1.0 : 0.0;
        break;
      case ExportedExpressionOp::kGreater:
        value = a > b ? 1.0 : 0.0;
        break;
      default:
        result.status = 2;
        error = true;
        break;
      }
    }
    if (done) {
      if (!Bounded(value))
        error = true;
      values[node_index] = value;
      --depth;
    }
  }
  if (error || result.status != 0) {
    if (result.status == 0)
      result.status = 1;
    results[index] = result;
    return;
  }
  result.value = values[program.root];
  if (!derivatives) {
    results[index] = result;
    return;
  }
  adjoints[program.root] = 1;
  for (std::uint32_t position = 0; position < program.ad_count; ++position) {
    const auto i = all_ad[program.ad_offset + position];
    const auto node = nodes[i];
    const double adjoint = adjoints[i];
    if (adjoint == 0)
      continue;
    if (node.op == ExportedExpressionOp::kState) {
      gradient[node.second] += adjoint;
      if (!Bounded(gradient[node.second]))
        error = true;
      continue;
    }
    const double a = values[node.first], b = values[node.second];
    switch (node.op) {
    case ExportedExpressionOp::kConstant:
    case ExportedExpressionOp::kState:
      break;
    case ExportedExpressionOp::kNegate:
      AddAdjoint(nodes, adjoints, node.first, adjoint, -1, &error);
      break;
    case ExportedExpressionOp::kAdd:
      AddAdjoint(nodes, adjoints, node.first, adjoint, 1, &error);
      AddAdjoint(nodes, adjoints, node.second, adjoint, 1, &error);
      break;
    case ExportedExpressionOp::kSubtract:
      AddAdjoint(nodes, adjoints, node.first, adjoint, 1, &error);
      AddAdjoint(nodes, adjoints, node.second, adjoint, -1, &error);
      break;
    case ExportedExpressionOp::kMultiply:
      AddAdjoint(nodes, adjoints, node.first, adjoint, b, &error);
      AddAdjoint(nodes, adjoints, node.second, adjoint, a, &error);
      break;
    case ExportedExpressionOp::kDivide: {
      const double denominator =
          program.dialect == ExpressionDialect::kBehavioral ? Denominator(b)
                                                            : b;
      AddAdjoint(nodes, adjoints, node.first, adjoint, 1 / denominator, &error);
      AddAdjoint(nodes, adjoints, node.second, adjoint,
                 -(values[i] / denominator), &error);
      break;
    }
    case ExportedExpressionOp::kPower:
      if (!nodes[node.first].constant) {
        double slope;
        if (b == 0 || (a == 0 && b > 1))
          slope = 0;
        else if (a == 0 && b == 1)
          slope = 1;
        else
          slope = b * pow(fabs(a), b - 1) * (a < 0 ? -1.0 : 1.0);
        AddAdjoint(nodes, adjoints, node.first, adjoint, slope, &error);
      }
      if (!nodes[node.second].constant)
        AddAdjoint(nodes, adjoints, node.second, adjoint,
                   a == 0 && b > 0 ? 0 : values[i] * log(fabs(a)), &error);
      break;
    case ExportedExpressionOp::kExp:
      AddAdjoint(nodes, adjoints, node.first, adjoint,
                 program.dialect == ExpressionDialect::kBehavioral && a > 14
                     ? 1202604.284
                     : values[i],
                 &error);
      break;
    case ExportedExpressionOp::kLess:
    case ExportedExpressionOp::kGreater:
      break;
    case ExportedExpressionOp::kIf:
      AddAdjoint(nodes, adjoints, a != 0 ? node.second : node.third, adjoint, 1,
                 &error);
      break;
    }
  }
  if (error)
    result.status = 1;
  results[index] = result;
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

std::vector<std::unique_ptr<ProgramBatch>> program_cache;

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
