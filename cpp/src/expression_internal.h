#ifndef OHMNIVORE_CPP_SRC_EXPRESSION_INTERNAL_H_
#define OHMNIVORE_CPP_SRC_EXPRESSION_INTERNAL_H_

#include "ohmnivore/expression.h"

namespace ohmnivore::internal {

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
