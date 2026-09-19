#ifndef OHMNIVORE_CUDA_EMI03_EXPRESSION_H_
#define OHMNIVORE_CUDA_EMI03_EXPRESSION_H_

#include <span>
#include <vector>

#include "ohmnivore/expression.h"

namespace ohmnivore {

// Values and derivatives have descriptor order. An inactive IF arm is never
// evaluated. The job owns immutable program uploads until EndEmi03CudaJob.
[[nodiscard]] Result<std::vector<ExpressionEvaluation>>
EvaluateEmi03CudaExpressions(std::span<const CompiledExpression> expressions,
                             std::span<const double> state,
                             bool derivatives = true);

} // namespace ohmnivore

#endif // OHMNIVORE_CUDA_EMI03_EXPRESSION_H_
