#ifndef OHMNIVORE_CPP_SRC_EXPRESSION_INTERNAL_H_
#define OHMNIVORE_CPP_SRC_EXPRESSION_INTERNAL_H_

#include <cstdint>

#include "ohmnivore/expression.h"

namespace ohmnivore::internal {

// Immutable, backend-neutral snapshot for the opt-in EMI-03 executor. The
// parser remains the only source of admitted programs; no model text is
// exposed.
enum class ExportedExpressionOp : std::uint32_t {
  kConstant,
  kState,
  kNegate,
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kPower,
  kExp,
  kLess,
  kGreater,
  kIf
};

struct ExportedExpressionNode {
  ExportedExpressionOp op;
  std::uint32_t first = 0, second = 0, third = 0;
  double value = 0.0;
  std::uint32_t constant = 1;
};

struct ExportedExpressionProgram {
  std::vector<ExportedExpressionNode> nodes;
  std::vector<std::uint32_t> reverse_ad_indices;
  std::vector<std::uint32_t> dependencies;
  std::uint32_t root = 0, state_size = 0;
  ExpressionDialect dialect = ExpressionDialect::kBehavioral;
};

[[nodiscard]] Result<ExportedExpressionProgram>
ExportExpressionProgram(const CompiledExpression &expression);
// The caller must retain a CompiledExpression copy while using this identity.
[[nodiscard]] const void *
ExpressionProgramIdentity(const CompiledExpression &expression) noexcept;

// Scalar value/domain checks only. Does not validate analytic derivatives;
// callers that need those guarantees must separately own a full evaluation.
[[nodiscard]] Result<double>
EvaluateExpressionValue(const CompiledExpression &expression,
                        std::span<const double> state);

// Test reference: original recursive values and full descending reverse-AD
// node scan. Bypasses simple-expression specialization and compiled AD order.
[[nodiscard]] Result<ExpressionEvaluation>
EvaluateExpressionOriginalAdForTesting(const CompiledExpression &expression,
                                       std::span<const double> state);

} // namespace ohmnivore::internal

#endif // OHMNIVORE_CPP_SRC_EXPRESSION_INTERNAL_H_
