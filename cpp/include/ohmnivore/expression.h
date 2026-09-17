#ifndef OHMNIVORE_EXPRESSION_H_
#define OHMNIVORE_EXPRESSION_H_

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ohmnivore/status.h"

namespace ohmnivore {

enum class ExpressionDialect { kBehavioral, kParameter };
using ParameterValues = std::map<std::string, double>;

struct ExpressionBindings {
  std::size_t state_size = 0;
  // nullopt denotes ground. 0/GND are always recognized as ground.
  std::map<std::string, std::optional<std::size_t>> node_indices;
  std::map<std::string, std::size_t> current_indices;
  ParameterValues parameters;
};

struct ExpressionEvaluation {
  double value = 0.0;
  std::vector<std::pair<std::size_t, double>> derivatives;
};

struct ExpressionProgram;

class CompiledExpression {
public:
  CompiledExpression() = default;
  [[nodiscard]] std::size_t node_count() const;
  [[nodiscard]] std::span<const std::size_t> dependencies() const;

private:
  std::shared_ptr<const ExpressionProgram> program_;
  friend Result<CompiledExpression>
  CompileExpression(std::string_view, const ExpressionBindings &,
                    ExpressionDialect);
  friend Result<ExpressionEvaluation>
  EvaluateExpression(const CompiledExpression &, std::span<const double>);
};

[[nodiscard]] Result<CompiledExpression>
CompileExpression(std::string_view text, const ExpressionBindings &bindings,
                  ExpressionDialect dialect = ExpressionDialect::kBehavioral);

[[nodiscard]] Result<ExpressionEvaluation>
EvaluateExpression(const CompiledExpression &expression,
                   std::span<const double> state);

struct ParameterDefinition {
  std::string name;
  std::string expression;
};

[[nodiscard]] Result<ParameterValues>
ResolveParameters(const std::vector<ParameterDefinition> &definitions,
                  const ParameterValues &caller = {},
                  const std::vector<ParameterDefinition> &overrides = {});

} // namespace ohmnivore

#endif // OHMNIVORE_EXPRESSION_H_
