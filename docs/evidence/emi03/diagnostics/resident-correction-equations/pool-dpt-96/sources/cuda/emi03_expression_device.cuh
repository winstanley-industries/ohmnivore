#ifndef OHMNIVORE_CUDA_EMI03_EXPRESSION_DEVICE_CUH_
#define OHMNIVORE_CUDA_EMI03_EXPRESSION_DEVICE_CUH_

#include "cpp/src/expression_internal.h"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ohmnivore::emi03_device {
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

// Positive IEEE-754 magnitude bits have the same ordering as finite doubles.
// Inspecting the representation preserves the exact finite/boundary predicate
// without consuming the device's scarce FP64 arithmetic issue slots.
__device__ inline bool Bounded(double value) {
  const auto magnitude =
      static_cast<unsigned long long>(__double_as_longlong(value)) &
      0x7fffffffffffffffULL;
  return magnitude <=
         static_cast<unsigned long long>(__double_as_longlong(1e100));
}

__device__ inline double Denominator(double value) {
  return value + (value >= 0 ? 1e-32 : -1e-32);
}

template <typename Node>
__device__ inline void AddAdjoint(const Node *nodes, double *adjoints,
                                  std::uint32_t child, double adjoint,
                                  double factor, bool *error) {
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
__device__ inline void EvaluateProgram(
    std::uint32_t index, const DeviceProgram *programs, std::uint32_t count,
    const ExportedExpressionNode *all_nodes, const std::uint32_t *all_ad,
    const std::uint32_t *all_dependencies, const double *state,
    DeviceResult *results, double *all_gradients, double *all_values,
    double *all_adjoints, std::uint32_t *all_stacks, unsigned char *all_stages,
    bool derivatives) {
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

} // namespace ohmnivore::emi03_device
#endif
