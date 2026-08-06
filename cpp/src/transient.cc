#include "ohmnivore/transient.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ohmnivore/solver.h"
#include "ohmnivore/waveform.h"

namespace ohmnivore {
namespace {

[[nodiscard]] std::optional<std::string> ValidateCsr(const CsrMatrix &matrix) {
  if (matrix.row_offsets.size() != matrix.rows + 1 ||
      matrix.column_indices.size() != matrix.values.size()) {
    return "invalid CSR structure";
  }
  if (matrix.row_offsets.empty() || matrix.row_offsets.front() != 0 ||
      matrix.row_offsets.back() != matrix.values.size()) {
    return "invalid CSR row offsets";
  }
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
        matrix.row_offsets[row + 1] > matrix.values.size()) {
      return "invalid CSR row offsets";
    }
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      if (matrix.column_indices[index] >= matrix.columns) {
        return "CSR column index is outside the matrix";
      }
      if (index > matrix.row_offsets[row] &&
          matrix.column_indices[index - 1] >= matrix.column_indices[index]) {
        return "CSR columns must be strictly increasing within each row";
      }
      if (!std::isfinite(matrix.values[index])) {
        return "CSR matrix contains a non-finite value";
      }
    }
  }
  return std::nullopt;
}

[[nodiscard]] Result<std::vector<double>>
Multiply(const CsrMatrix &matrix, const std::vector<double> &vector) {
  if (matrix.columns != vector.size()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "matrix-vector dimensions disagree");
  }
  if (const auto error = ValidateCsr(matrix); error.has_value()) {
    return Result<std::vector<double>>::Fail(ErrorCode::kSolve, *error);
  }
  for (double value : vector) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "matrix-vector input contains a non-finite value");
    }
  }

  std::vector<double> product(matrix.rows, 0.0);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      product[row] +=
          matrix.values[index] * vector[matrix.column_indices[index]];
      if (!std::isfinite(product[row])) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kSolve,
            "matrix-vector multiplication produced a non-finite value");
      }
    }
  }
  return Result<std::vector<double>>::Ok(std::move(product));
}

[[nodiscard]] Result<CsrMatrix> DenseToCsr(const std::vector<double> &dense,
                                           std::size_t size) {
  if (size != 0 && size > std::numeric_limits<std::size_t>::max() / size) {
    return Result<CsrMatrix>::Fail(
        ErrorCode::kUnsupportedSize,
        "dense projection matrix dimensions are not representable");
  }
  if (dense.size() != size * size) {
    return Result<CsrMatrix>::Fail(ErrorCode::kInvalidStructure,
                                   "invalid dense matrix dimensions");
  }
  CsrMatrix matrix;
  matrix.rows = size;
  matrix.columns = size;
  matrix.row_offsets.reserve(size + 1);
  matrix.row_offsets.push_back(0);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column < size; ++column) {
      const double value = dense[row * size + column];
      if (!std::isfinite(value)) {
        return Result<CsrMatrix>::Fail(
            ErrorCode::kNonFinite, "dense matrix contains a non-finite value");
      }
      if (value != 0.0) {
        matrix.column_indices.push_back(column);
        matrix.values.push_back(value);
      }
    }
    matrix.row_offsets.push_back(matrix.values.size());
  }
  return Result<CsrMatrix>::Ok(std::move(matrix));
}

class SparseRealFactorizationCache {
public:
  [[nodiscard]] Result<std::vector<double>>
  Solve(const CsrMatrix &matrix, const std::vector<double> &rhs) {
    Result<SolverCscPattern> converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok()) {
      return Result<std::vector<double>>::Fail(converted.error().code,
                                               converted.error().message);
    }
    for (Entry &entry : entries_) {
      if (entry.pattern.size == converted.value().size &&
          entry.pattern.column_offsets == converted.value().column_offsets &&
          entry.pattern.row_indices == converted.value().row_indices &&
          entry.pattern.csr_value_indices ==
              converted.value().csr_value_indices) {
        return entry.factorization->FactorAndSolve(matrix, rhs);
      }
    }
    Result<std::unique_ptr<SparseRealFactorization>> analyzed =
        SparseRealFactorization::Analyze(matrix);
    if (!analyzed.ok()) {
      return Result<std::vector<double>>::Fail(analyzed.error().code,
                                               analyzed.error().message);
    }
    entries_.push_back(Entry{.pattern = converted.TakeValue(),
                             .factorization = analyzed.TakeValue()});
    return entries_.back().factorization->FactorAndSolve(matrix, rhs);
  }

private:
  struct Entry {
    SolverCscPattern pattern;
    std::unique_ptr<SparseRealFactorization> factorization;
  };
  std::vector<Entry> entries_;
};

struct ReactiveConstraintEquation {
  std::vector<double> coefficients;
  double right_hand_side = 0.0;
  std::string description;
};

struct RankBasisRow {
  std::vector<double> coefficients;
  std::size_t pivot = 0;
};

[[nodiscard]] bool AppendIfIndependent(const std::vector<double> &candidate,
                                       std::vector<RankBasisRow> *basis) {
  double scale = 0.0;
  for (double value : candidate) {
    scale = std::max(scale, std::abs(value));
  }
  if (!std::isfinite(scale) || scale == 0.0) {
    return false;
  }

  std::vector<double> row = candidate;
  for (double &value : row) {
    value /= scale;
  }
  for (const RankBasisRow &existing : *basis) {
    const double factor = row[existing.pivot];
    if (factor == 0.0) {
      continue;
    }
    for (std::size_t column = existing.pivot; column < row.size(); ++column) {
      row[column] -= factor * existing.coefficients[column];
    }
  }

  const double tolerance =
      128.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<std::size_t>(1, row.size()));
  const auto pivot = std::find_if(row.begin(), row.end(), [&](double value) {
    return std::abs(value) > tolerance;
  });
  if (pivot == row.end()) {
    return false;
  }
  const std::size_t pivot_index = static_cast<std::size_t>(pivot - row.begin());
  const double pivot_value = row[pivot_index];
  for (std::size_t column = pivot_index; column < row.size(); ++column) {
    row[column] /= pivot_value;
  }
  basis->push_back(
      RankBasisRow{.coefficients = std::move(row), .pivot = pivot_index});
  return true;
}

[[nodiscard]] bool EquationIsSatisfied(const std::vector<double> &coefficients,
                                       double right_hand_side,
                                       const std::vector<double> &state) {
  double actual = 0.0;
  double scale = std::abs(right_hand_side);
  for (std::size_t index = 0; index < state.size(); ++index) {
    actual += coefficients[index] * state[index];
    scale += std::abs(coefficients[index]) * std::abs(state[index]);
  }
  const double tolerance =
      256.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
  return std::isfinite(actual) &&
         std::abs(actual - right_hand_side) <= tolerance;
}

[[nodiscard]] Result<std::vector<double>>
ProjectReactiveState(const MnaSystem &system,
                     const std::vector<double> &target_state,
                     const std::vector<double> &source_rhs, std::string context,
                     SparseRealFactorizationCache *factorization_cache) {
  const std::size_t size = system.g.rows;
  const std::size_t node_count = system.node_names.size();
  if (system.g.rows != system.g.columns || target_state.size() != size ||
      source_rhs.size() != size) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kInvalidStructure,
        context + " has invalid projection dimensions");
  }
  Result<SolverCscPattern> structure = ConvertCsrToSolverCsc(system.g);
  if (!structure.ok()) {
    return Result<std::vector<double>>::Fail(
        structure.error().code,
        context + " has invalid G matrix: " + structure.error().message);
  }
  for (double value : target_state) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kNonFinite, context + " target state is non-finite");
    }
  }
  for (double value : source_rhs) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kNonFinite, context + " source RHS is non-finite");
    }
  }

  std::vector<bool> replaceable_row(size, false);
  std::vector<ReactiveConstraintEquation> constraints;
  for (const CapacitorInitialConstraint &constraint :
       system.capacitor_initial_constraints) {
    if ((constraint.positive_node_index.has_value() &&
         *constraint.positive_node_index >= node_count) ||
        (constraint.negative_node_index.has_value() &&
         *constraint.negative_node_index >= node_count)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          context + ": capacitor '" + constraint.name +
              "' has invalid constraint metadata");
    }
    ReactiveConstraintEquation equation{
        .coefficients = std::vector<double>(size, 0.0),
        .right_hand_side = 0.0,
        .description = "capacitor '" + constraint.name + "' voltage",
    };
    if (constraint.positive_node_index.has_value()) {
      const std::size_t node = *constraint.positive_node_index;
      equation.coefficients[node] += 1.0;
      equation.right_hand_side += target_state[node];
      replaceable_row[node] = true;
    }
    if (constraint.negative_node_index.has_value()) {
      const std::size_t node = *constraint.negative_node_index;
      equation.coefficients[node] -= 1.0;
      equation.right_hand_side -= target_state[node];
      replaceable_row[node] = true;
    }
    if (std::all_of(equation.coefficients.begin(), equation.coefficients.end(),
                    [](double value) { return value == 0.0; })) {
      continue;
    }
    constraints.push_back(std::move(equation));
  }

  std::vector<bool> constrained_branch(size, false);
  for (const InductorInitialConstraint &constraint :
       system.inductor_initial_constraints) {
    if (constraint.branch_index < node_count ||
        constraint.branch_index >= size ||
        constrained_branch[constraint.branch_index]) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          context + ": inductor '" + constraint.name +
              "' has invalid constraint metadata");
    }
    constrained_branch[constraint.branch_index] = true;
    replaceable_row[constraint.branch_index] = true;
    ReactiveConstraintEquation equation{
        .coefficients = std::vector<double>(size, 0.0),
        .right_hand_side = target_state[constraint.branch_index],
        .description = "inductor '" + constraint.name + "' current",
    };
    equation.coefficients[constraint.branch_index] = 1.0;
    constraints.push_back(std::move(equation));
  }

  const std::vector<double> dense_g = system.g.ToDense();
  std::vector<double> selected_dense;
  std::vector<double> selected_rhs;
  selected_dense.reserve(size * size);
  selected_rhs.reserve(size);
  std::vector<RankBasisRow> basis;
  const auto select = [&](const std::vector<double> &row, double rhs) {
    if (!AppendIfIndependent(row, &basis)) {
      return;
    }
    selected_dense.insert(selected_dense.end(), row.begin(), row.end());
    selected_rhs.push_back(rhs);
  };

  // Reactive state constraints are mandatory.  Preserve every algebraic row
  // that cannot carry a capacitor impulse or replace an inductor constitutive
  // equation before using the remaining dynamic rows to complete the rank.
  for (const ReactiveConstraintEquation &constraint : constraints) {
    select(constraint.coefficients, constraint.right_hand_side);
  }
  for (std::size_t row = 0; row < size; ++row) {
    if (!replaceable_row[row]) {
      select(
          std::vector<double>(
              dense_g.begin() + static_cast<std::ptrdiff_t>(row * size),
              dense_g.begin() + static_cast<std::ptrdiff_t>((row + 1) * size)),
          source_rhs[row]);
    }
  }
  for (std::size_t row = 0; row < size && selected_rhs.size() < size; ++row) {
    if (replaceable_row[row]) {
      select(
          std::vector<double>(
              dense_g.begin() + static_cast<std::ptrdiff_t>(row * size),
              dense_g.begin() + static_cast<std::ptrdiff_t>((row + 1) * size)),
          source_rhs[row]);
    }
  }
  if (selected_rhs.size() != size) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSingular,
        context + " constraints do not define a unique algebraic state");
  }

  Result<CsrMatrix> matrix = DenseToCsr(selected_dense, size);
  if (!matrix.ok()) {
    return Result<std::vector<double>>::Fail(
        matrix.error().code,
        context + " matrix construction failed: " + matrix.error().message);
  }
  Result<std::vector<double>> solved =
      factorization_cache->Solve(matrix.value(), selected_rhs);
  if (!solved.ok()) {
    return Result<std::vector<double>>::Fail(
        solved.error().code,
        context + " solve failed: " + solved.error().message);
  }

  for (const ReactiveConstraintEquation &constraint : constraints) {
    if (!EquationIsSatisfied(constraint.coefficients,
                             constraint.right_hand_side, solved.value())) {
      return Result<std::vector<double>>::Fail(ErrorCode::kSolutionValidation,
                                               context + " has inconsistent " +
                                                   constraint.description);
    }
  }
  for (std::size_t row = 0; row < size; ++row) {
    if (replaceable_row[row]) {
      continue;
    }
    const std::vector<double> equation(
        dense_g.begin() + static_cast<std::ptrdiff_t>(row * size),
        dense_g.begin() + static_cast<std::ptrdiff_t>((row + 1) * size));
    if (!EquationIsSatisfied(equation, source_rhs[row], solved.value())) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolutionValidation,
          context + " is inconsistent with an algebraic circuit constraint");
    }
  }
  return solved;
}

[[nodiscard]] double ConstraintTolerance(double first, double second = 0.0) {
  return 64.0 * std::numeric_limits<double>::epsilon() *
         std::max({1.0, std::abs(first), std::abs(second)});
}

void SortUnique(std::vector<double> *values) {
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
}

[[nodiscard]] Result<double>
ComputeNormalizedLocalError(const std::vector<double> &higher_accuracy,
                            const std::vector<double> &lower_accuracy,
                            double estimate_multiplier) {
  if (higher_accuracy.size() != lower_accuracy.size() ||
      !std::isfinite(estimate_multiplier) || estimate_multiplier <= 0.0) {
    return Result<double>::Fail(ErrorCode::kSolve,
                                "LTE solution dimensions disagree");
  }
  double normalized_error = 0.0;
  for (std::size_t index = 0; index < higher_accuracy.size(); ++index) {
    if (!std::isfinite(higher_accuracy[index]) ||
        !std::isfinite(lower_accuracy[index])) {
      return Result<double>::Fail(
          ErrorCode::kSolve, "LTE input contains a non-finite solution value");
    }
    const double difference = higher_accuracy[index] - lower_accuracy[index];
    const double estimate = estimate_multiplier * std::abs(difference);
    const double scale =
        kTransientAbsoluteTolerance +
        kTransientRelativeTolerance * std::max(std::abs(higher_accuracy[index]),
                                               std::abs(lower_accuracy[index]));
    const double component_error = estimate / scale;
    if (!std::isfinite(estimate) || !std::isfinite(scale) || scale <= 0.0 ||
        !std::isfinite(component_error)) {
      return Result<double>::Fail(
          ErrorCode::kSolve,
          "local-error estimation produced a non-finite value");
    }
    normalized_error = std::max(normalized_error, component_error);
  }
  return Result<double>::Ok(normalized_error);
}

[[nodiscard]] double AdaptationFactor(double normalized_error) {
  if (normalized_error == 0.0) {
    return 2.0;
  }
  return std::clamp(0.9 * std::sqrt(1.0 / normalized_error), 0.5, 2.0);
}

[[nodiscard]] bool ContainsExact(const std::vector<double> &values,
                                 double value) {
  return std::binary_search(values.begin(), values.end(), value);
}

} // namespace

Result<std::vector<double>> BuildTransientRhs(const MnaSystem &system,
                                              double time_seconds) {
  if (!std::isfinite(time_seconds) || time_seconds < 0.0) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "transient time must be finite and nonnegative");
  }
  if (system.b_dc.size() != system.g.rows) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "transient DC right-hand-side dimension is invalid");
  }
  std::vector<double> rhs = system.b_dc;
  for (double value : rhs) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "transient DC right-hand side contains a non-finite value");
    }
  }

  for (const TransientSourceStamp &source : system.transient_sources) {
    if (!std::isfinite(source.dc_value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "transient source '" + source.name +
                                 "' has a non-finite DC replacement value");
    }
    Result<double> evaluated =
        EvaluateTransientWaveform(source.waveform, time_seconds);
    if (!evaluated.ok()) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "transient source '" + source.name +
              "' evaluation failed: " + evaluated.error().message);
    }
    const double delta = evaluated.value() - source.dc_value;
    if (!std::isfinite(delta)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "transient source '" + source.name +
                                 "' produced a non-finite RHS delta");
    }
    for (const TransientRhsStamp &stamp : source.rhs_stamps) {
      if (stamp.index >= rhs.size() || !std::isfinite(stamp.coefficient)) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kSolve, "transient source '" + source.name +
                                   "' has invalid RHS stamp metadata");
      }
      rhs[stamp.index] += stamp.coefficient * delta;
      if (!std::isfinite(rhs[stamp.index])) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kSolve,
            "transient RHS accumulation produced a non-finite value");
      }
    }
  }
  return Result<std::vector<double>>::Ok(std::move(rhs));
}

Result<CsrMatrix> FormTransientCompanionMatrix(const CsrMatrix &g,
                                               const CsrMatrix &c,
                                               double step_size_seconds,
                                               double alpha) {
  if (g.rows != g.columns || c.rows != c.columns || g.rows != c.rows ||
      g.columns != c.columns) {
    return Result<CsrMatrix>::Fail(
        ErrorCode::kSolve, "G and C must be square matrices of equal size");
  }
  if (!std::isfinite(step_size_seconds) || step_size_seconds <= 0.0 ||
      !std::isfinite(alpha) || alpha <= 0.0) {
    return Result<CsrMatrix>::Fail(
        ErrorCode::kSolve,
        "transient companion step and alpha must be finite and positive");
  }
  if (const auto error = ValidateCsr(g); error.has_value()) {
    return Result<CsrMatrix>::Fail(ErrorCode::kSolve,
                                   "invalid G matrix: " + *error);
  }
  if (const auto error = ValidateCsr(c); error.has_value()) {
    return Result<CsrMatrix>::Fail(ErrorCode::kSolve,
                                   "invalid C matrix: " + *error);
  }
  const double factor = alpha / step_size_seconds;
  if (!std::isfinite(factor)) {
    return Result<CsrMatrix>::Fail(
        ErrorCode::kSolve,
        "transient companion scaling produced a non-finite value");
  }

  CsrMatrix result;
  result.rows = g.rows;
  result.columns = g.columns;
  result.row_offsets.reserve(g.rows + 1);
  result.row_offsets.push_back(0);
  result.values.reserve(g.values.size() + c.values.size());
  result.column_indices.reserve(g.values.size() + c.values.size());

  for (std::size_t row = 0; row < g.rows; ++row) {
    std::size_t g_index = g.row_offsets[row];
    std::size_t c_index = c.row_offsets[row];
    const std::size_t g_end = g.row_offsets[row + 1];
    const std::size_t c_end = c.row_offsets[row + 1];
    while (g_index < g_end || c_index < c_end) {
      const bool take_g = c_index == c_end ||
                          (g_index < g_end && g.column_indices[g_index] <
                                                  c.column_indices[c_index]);
      const bool take_c = g_index == g_end ||
                          (c_index < c_end && c.column_indices[c_index] <
                                                  g.column_indices[g_index]);
      std::size_t column = 0;
      double value = 0.0;
      if (take_g) {
        column = g.column_indices[g_index];
        value = g.values[g_index];
        ++g_index;
      } else if (take_c) {
        column = c.column_indices[c_index];
        value = factor * c.values[c_index];
        ++c_index;
      } else {
        column = g.column_indices[g_index];
        value = g.values[g_index] + factor * c.values[c_index];
        ++g_index;
        ++c_index;
      }
      if (!std::isfinite(value)) {
        return Result<CsrMatrix>::Fail(
            ErrorCode::kSolve,
            "transient companion matrix produced a non-finite value");
      }
      // Retain the deterministic G/C union pattern across methods and step
      // sizes, including exact numerical cancellations, for symbolic reuse.
      result.column_indices.push_back(column);
      result.values.push_back(value);
    }
    result.row_offsets.push_back(result.values.size());
  }
  return Result<CsrMatrix>::Ok(std::move(result));
}

Result<std::vector<double>> BuildBackwardEulerRhs(
    const CsrMatrix &c, const std::vector<double> &previous_state,
    const std::vector<double> &current_rhs, double step_size_seconds) {
  if (!std::isfinite(step_size_seconds) || step_size_seconds <= 0.0 ||
      c.rows != current_rhs.size()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "invalid backward-Euler RHS dimensions or step");
  }
  Result<std::vector<double>> multiplied = Multiply(c, previous_state);
  if (!multiplied.ok()) {
    return multiplied;
  }
  const double factor = 1.0 / step_size_seconds;
  if (!std::isfinite(factor)) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve,
        "backward-Euler RHS scaling produced a non-finite value");
  }
  std::vector<double> result(current_rhs.size(), 0.0);
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (!std::isfinite(current_rhs[index])) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "backward-Euler source RHS contains a non-finite value");
    }
    result[index] = current_rhs[index] + factor * multiplied.value()[index];
    if (!std::isfinite(result[index])) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "backward-Euler RHS formation produced a non-finite value");
    }
  }
  return Result<std::vector<double>>::Ok(std::move(result));
}

Result<std::vector<double>>
BuildTrapezoidalRhs(const CsrMatrix &g, const CsrMatrix &c,
                    const std::vector<double> &previous_state,
                    const std::vector<double> &previous_rhs,
                    const std::vector<double> &current_rhs,
                    double step_size_seconds) {
  if (!std::isfinite(step_size_seconds) || step_size_seconds <= 0.0 ||
      g.rows != c.rows || g.columns != c.columns ||
      g.rows != previous_rhs.size() || g.rows != current_rhs.size()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "invalid trapezoidal RHS dimensions or step");
  }
  Result<std::vector<double>> g_product = Multiply(g, previous_state);
  if (!g_product.ok()) {
    return g_product;
  }
  Result<std::vector<double>> c_product = Multiply(c, previous_state);
  if (!c_product.ok()) {
    return c_product;
  }
  const double factor = 2.0 / step_size_seconds;
  if (!std::isfinite(factor)) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve,
        "trapezoidal RHS scaling produced a non-finite value");
  }
  std::vector<double> result(current_rhs.size(), 0.0);
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (!std::isfinite(previous_rhs[index]) ||
        !std::isfinite(current_rhs[index])) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "trapezoidal source RHS contains a non-finite value");
    }
    result[index] = current_rhs[index] + previous_rhs[index] +
                    factor * c_product.value()[index] -
                    g_product.value()[index];
    if (!std::isfinite(result[index])) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "trapezoidal RHS formation produced a non-finite value");
    }
  }
  return Result<std::vector<double>>::Ok(std::move(result));
}

namespace {

[[nodiscard]] Result<std::vector<double>> SolveBackwardEulerStep(
    const MnaSystem &system, const std::vector<double> &previous_state,
    const std::vector<double> &current_rhs, double step_size_seconds,
    std::string context, SparseRealFactorizationCache *factorization_cache) {
  Result<CsrMatrix> matrix =
      FormTransientCompanionMatrix(system.g, system.c, step_size_seconds, 1.0);
  if (!matrix.ok()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, context + ": " + matrix.error().message);
  }
  Result<std::vector<double>> rhs = BuildBackwardEulerRhs(
      system.c, previous_state, current_rhs, step_size_seconds);
  if (!rhs.ok()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, context + ": " + rhs.error().message);
  }
  Result<std::vector<double>> solved =
      factorization_cache->Solve(matrix.value(), rhs.value());
  if (!solved.ok()) {
    return Result<std::vector<double>>::Fail(
        solved.error().code, context + ": " + solved.error().message);
  }
  return solved;
}

[[nodiscard]] bool EqualVectors(const std::vector<double> &first,
                                const std::vector<double> &second) {
  return first.size() == second.size() &&
         std::equal(first.begin(), first.end(), second.begin());
}

} // namespace

namespace {

[[nodiscard]] Result<std::vector<double>> BuildTransientInitialStateImpl(
    const MnaSystem &system, bool use_initial_conditions,
    SparseRealFactorizationCache *factorization_cache) {
  if (system.g.rows != system.g.columns ||
      system.node_names.size() + system.branch_names.size() != system.g.rows ||
      system.b_dc.size() != system.g.rows) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kInvalidStructure,
        "invalid MNA dimensions for transient initialization");
  }
  if (!use_initial_conditions) {
    Result<std::vector<double>> solved =
        factorization_cache->Solve(system.g, system.b_dc);
    if (!solved.ok()) {
      return Result<std::vector<double>>::Fail(
          solved.error().code,
          "DC transient initialization failed: " + solved.error().message);
    }
    return solved;
  }

  const std::size_t size = system.g.rows;
  const std::vector<double> zero_reactive_state(size, 0.0);
  Result<std::vector<double>> solved =
      ProjectReactiveState(system, zero_reactive_state, system.b_dc,
                           "UIC initialization", factorization_cache);
  if (!solved.ok()) {
    return Result<std::vector<double>>::Fail(
        solved.error().code,
        "UIC initialization constraints are inconsistent: " +
            solved.error().message);
  }

  for (const CapacitorInitialConstraint &constraint :
       system.capacitor_initial_constraints) {
    const double positive =
        constraint.positive_node_index.has_value()
            ? solved.value()[*constraint.positive_node_index]
            : 0.0;
    const double negative =
        constraint.negative_node_index.has_value()
            ? solved.value()[*constraint.negative_node_index]
            : 0.0;
    if (std::abs(positive - negative) >
        ConstraintTolerance(positive, negative)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolutionValidation,
          "capacitor '" + constraint.name +
              "' has an inconsistent zero-voltage UIC");
    }
  }
  for (const InductorInitialConstraint &constraint :
       system.inductor_initial_constraints) {
    const double current = solved.value()[constraint.branch_index];
    if (std::abs(current) > ConstraintTolerance(current)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolutionValidation,
          "inductor '" + constraint.name +
              "' has an inconsistent zero-current UIC");
    }
  }
  return solved;
}

} // namespace

Result<std::vector<double>>
BuildTransientInitialState(const MnaSystem &system,
                           bool use_initial_conditions) {
  if (!system.diode_descriptors.empty()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kUnsupported,
        "phase 3A does not support nonlinear transient analysis");
  }
  SparseRealFactorizationCache factorization_cache;
  return BuildTransientInitialStateImpl(system, use_initial_conditions,
                                        &factorization_cache);
}

Result<TransientResult>
RunTransientAnalysis(const MnaSystem &system, const TranAnalysis &analysis,
                     const TransientExecutionLimits &limits) {
  if (!system.diode_descriptors.empty()) {
    return Result<TransientResult>::Fail(
        ErrorCode::kUnsupported,
        "phase 3A does not support nonlinear transient analysis");
  }
  SparseRealFactorizationCache factorization_cache;
  if (!std::isfinite(analysis.time_step_seconds) ||
      analysis.time_step_seconds <= 0.0 ||
      !std::isfinite(analysis.stop_time_seconds) ||
      analysis.stop_time_seconds <= 0.0 ||
      !std::isfinite(analysis.start_time_seconds) ||
      analysis.start_time_seconds < 0.0 ||
      analysis.start_time_seconds > analysis.stop_time_seconds) {
    return Result<TransientResult>::Fail(
        ErrorCode::kSolve, "invalid transient analysis time bounds");
  }
  if (limits.maximum_accepted_steps == 0 || limits.maximum_step_attempts == 0 ||
      limits.maximum_step_attempts < limits.maximum_accepted_steps ||
      !std::isfinite(limits.minimum_step_divisor) ||
      limits.minimum_step_divisor < 1.0) {
    return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                         "invalid transient execution limits");
  }
  const double maximum_step = analysis.time_step_seconds;
  const double minimum_step = maximum_step / limits.minimum_step_divisor;
  if (!std::isfinite(minimum_step) || minimum_step <= 0.0) {
    return Result<TransientResult>::Fail(
        ErrorCode::kSolve,
        "transient minimum timestep is not positively representable");
  }

  std::vector<double> waveform_points;
  for (const TransientSourceStamp &source : system.transient_sources) {
    Result<std::vector<double>> points = CollectTransientWaveformBreakpoints(
        source.waveform, analysis.stop_time_seconds);
    if (!points.ok()) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve,
          "transient source '" + source.name +
              "' breakpoint collection failed: " + points.error().message);
    }
    for (double point : points.value()) {
      if (!std::isfinite(point) || point < 0.0 ||
          point > analysis.stop_time_seconds) {
        return Result<TransientResult>::Fail(
            ErrorCode::kSolve, "transient source '" + source.name +
                                   "' produced an invalid breakpoint");
      }
      waveform_points.push_back(point);
    }
  }
  SortUnique(&waveform_points);

  std::vector<double> hard_points = waveform_points;
  hard_points.push_back(analysis.start_time_seconds);
  hard_points.push_back(analysis.stop_time_seconds);
  hard_points.erase(std::remove_if(hard_points.begin(), hard_points.end(),
                                   [](double value) { return value <= 0.0; }),
                    hard_points.end());
  SortUnique(&hard_points);

  Result<std::vector<double>> initial = BuildTransientInitialStateImpl(
      system, analysis.use_initial_conditions, &factorization_cache);
  if (!initial.ok()) {
    return Result<TransientResult>::Fail(initial.error().code,
                                         initial.error().message);
  }
  Result<std::vector<double>> initial_rhs = BuildTransientRhs(system, 0.0);
  if (!initial_rhs.ok()) {
    return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                         initial_rhs.error().message);
  }
  Result<std::vector<double>> projected_initial = ProjectReactiveState(
      system, initial.value(), initial_rhs.value(),
      "initial transient source projection", &factorization_cache);
  if (!projected_initial.ok()) {
    return Result<TransientResult>::Fail(projected_initial.error().code,
                                         projected_initial.error().message);
  }

  TransientResult result;
  std::vector<double> state = projected_initial.TakeValue();
  std::vector<double> previous_rhs = initial_rhs.TakeValue();
  double time = 0.0;
  double proposed_step = maximum_step;
  bool recovery_step = false;
  std::size_t hard_index = 0;
  std::size_t attempts = 0;
  std::size_t accepted_steps = 0;
  if (analysis.start_time_seconds == 0.0) {
    result.times_seconds.push_back(0.0);
    result.states.push_back(state);
  }

  while (time < analysis.stop_time_seconds) {
    if (attempts >= limits.maximum_step_attempts) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve, "transient analysis exceeded " +
                                 std::to_string(limits.maximum_step_attempts) +
                                 " timestep attempts");
    }
    ++attempts;
    while (hard_index < hard_points.size() && hard_points[hard_index] <= time) {
      ++hard_index;
    }
    if (hard_index == hard_points.size()) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve,
          "transient hard-point schedule ended before tstop");
    }

    const double next_hard_point = hard_points[hard_index];
    const double hard_distance = next_hard_point - time;
    if (!std::isfinite(hard_distance) || hard_distance <= 0.0) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve, "transient hard point is not strictly increasing");
    }
    double step = std::min(proposed_step, maximum_step);
    bool lands_on_hard_point = false;
    double next_time = time + step;
    if (step >= hard_distance || next_time >= next_hard_point) {
      step = hard_distance;
      next_time = next_hard_point;
      lands_on_hard_point = true;
    }
    if (!std::isfinite(step) || step <= 0.0 || !std::isfinite(next_time) ||
        next_time <= time) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve,
          "transient timestep is not strictly increasing and representable");
    }
    if (step < minimum_step && !lands_on_hard_point) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve, "adaptive transient timestep fell below h_min");
    }

    const bool waveform_landing =
        lands_on_hard_point && ContainsExact(waveform_points, next_time);
    const bool use_backward_euler =
        accepted_steps == 0 || recovery_step || waveform_landing;
    const TransientIntegrationMethod method =
        use_backward_euler ? TransientIntegrationMethod::kBackwardEuler
                           : TransientIntegrationMethod::kTrapezoidal;

    Result<std::vector<double>> current_rhs =
        BuildTransientRhs(system, next_time);
    if (!current_rhs.ok()) {
      return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                           current_rhs.error().message);
    }

    // Integrate to a waveform breakpoint with its left-limit forcing, then
    // project algebraic variables to the right-limit forcing while preserving
    // capacitor voltages and inductor currents. This prevents a discontinuous
    // source from acting over the interval before its scheduled edge.
    std::vector<double> integration_rhs = current_rhs.value();
    if (waveform_landing) {
      const double left_time = std::nextafter(next_time, time);
      if (left_time > time && left_time < next_time) {
        Result<std::vector<double>> left_rhs =
            BuildTransientRhs(system, left_time);
        if (!left_rhs.ok()) {
          return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                               left_rhs.error().message);
        }
        integration_rhs = left_rhs.TakeValue();
      } else {
        integration_rhs = previous_rhs;
      }
    }

    std::vector<double> accepted_state;
    double normalized_error = 0.0;
    if (use_backward_euler) {
      Result<std::vector<double>> full_step = SolveBackwardEulerStep(
          system, state, integration_rhs, step,
          "backward-Euler transient solve failed", &factorization_cache);
      if (!full_step.ok()) {
        return Result<TransientResult>::Fail(full_step.error().code,
                                             full_step.error().message);
      }

      if (system.c.values.empty()) {
        accepted_state = full_step.TakeValue();
      } else {
        const double midpoint = time + step * 0.5;
        if (!std::isfinite(midpoint) || midpoint <= time ||
            midpoint >= next_time) {
          if (lands_on_hard_point &&
              next_time == std::nextafter(
                               time, std::numeric_limits<double>::infinity())) {
            accepted_state = full_step.TakeValue();
          } else {
            return Result<TransientResult>::Fail(
                ErrorCode::kSolve, "backward-Euler error-estimation midpoint "
                                   "is not representable");
          }
        } else {
          Result<std::vector<double>> midpoint_rhs =
              BuildTransientRhs(system, midpoint);
          if (!midpoint_rhs.ok()) {
            return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                                 midpoint_rhs.error().message);
          }
          const double first_half = midpoint - time;
          const double second_half = next_time - midpoint;
          Result<std::vector<double>> first_half_state = SolveBackwardEulerStep(
              system, state, midpoint_rhs.value(), first_half,
              "first half backward-Euler LTE solve failed",
              &factorization_cache);
          if (!first_half_state.ok()) {
            return Result<TransientResult>::Fail(
                first_half_state.error().code,
                first_half_state.error().message);
          }
          Result<std::vector<double>> refined = SolveBackwardEulerStep(
              system, first_half_state.value(), integration_rhs, second_half,
              "second half backward-Euler LTE solve failed",
              &factorization_cache);
          if (!refined.ok()) {
            return Result<TransientResult>::Fail(refined.error().code,
                                                 refined.error().message);
          }
          Result<double> error = ComputeNormalizedLocalError(
              refined.value(), full_step.value(), 1.0);
          if (!error.ok()) {
            return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                                 error.error().message);
          }
          normalized_error = error.value();
          accepted_state = refined.TakeValue();
        }
      }

      const double adapted_step = step * AdaptationFactor(normalized_error);
      if (!std::isfinite(adapted_step) || adapted_step <= 0.0) {
        return Result<TransientResult>::Fail(
            ErrorCode::kSolve,
            "adaptive backward-Euler timestep produced a non-finite value");
      }
      if (normalized_error > 1.0) {
        result.step_trace.push_back(TransientStepRecord{
            .start_time_seconds = time,
            .end_time_seconds = next_time,
            .step_size_seconds = step,
            .method = method,
            .accepted = false,
            .normalized_local_error = normalized_error,
            .landed_on_hard_point = lands_on_hard_point,
        });
        if (adapted_step < minimum_step) {
          return Result<TransientResult>::Fail(
              ErrorCode::kSolve,
              "adaptive transient timestep fell below h_min");
        }
        proposed_step = adapted_step;
        recovery_step = true;
        continue;
      }
      proposed_step = std::clamp(adapted_step, minimum_step, maximum_step);
      recovery_step = false;
    } else {
      Result<CsrMatrix> trap_matrix =
          FormTransientCompanionMatrix(system.g, system.c, step, 2.0);
      if (!trap_matrix.ok()) {
        return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                             trap_matrix.error().message);
      }
      Result<std::vector<double>> trap_rhs = BuildTrapezoidalRhs(
          system.g, system.c, state, previous_rhs, current_rhs.value(), step);
      if (!trap_rhs.ok()) {
        return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                             trap_rhs.error().message);
      }
      Result<std::vector<double>> trap_solution =
          factorization_cache.Solve(trap_matrix.value(), trap_rhs.value());
      if (!trap_solution.ok()) {
        return Result<TransientResult>::Fail(
            trap_solution.error().code, "trapezoidal transient solve failed: " +
                                            trap_solution.error().message);
      }

      Result<CsrMatrix> be_matrix =
          FormTransientCompanionMatrix(system.g, system.c, step, 1.0);
      if (!be_matrix.ok()) {
        return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                             be_matrix.error().message);
      }
      Result<std::vector<double>> be_rhs =
          BuildBackwardEulerRhs(system.c, state, integration_rhs, step);
      if (!be_rhs.ok()) {
        return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                             be_rhs.error().message);
      }
      Result<std::vector<double>> be_solution =
          factorization_cache.Solve(be_matrix.value(), be_rhs.value());
      if (!be_solution.ok()) {
        return Result<TransientResult>::Fail(
            be_solution.error().code,
            "LTE backward-Euler solve failed: " + be_solution.error().message);
      }
      Result<double> error = ComputeNormalizedLocalError(
          trap_solution.value(), be_solution.value(), 2.0 / 3.0);
      if (!error.ok()) {
        return Result<TransientResult>::Fail(ErrorCode::kSolve,
                                             error.error().message);
      }
      normalized_error = error.value();
      const double factor = AdaptationFactor(normalized_error);
      const double adapted_step = step * factor;
      if (!std::isfinite(adapted_step) || adapted_step <= 0.0) {
        return Result<TransientResult>::Fail(
            ErrorCode::kSolve,
            "adaptive transient timestep produced a non-finite value");
      }
      if (normalized_error > 1.0) {
        result.step_trace.push_back(TransientStepRecord{
            .start_time_seconds = time,
            .end_time_seconds = next_time,
            .step_size_seconds = step,
            .method = method,
            .accepted = false,
            .normalized_local_error = normalized_error,
            .landed_on_hard_point = lands_on_hard_point,
        });
        if (adapted_step < minimum_step) {
          return Result<TransientResult>::Fail(
              ErrorCode::kSolve,
              "adaptive transient timestep fell below h_min");
        }
        proposed_step = adapted_step;
        recovery_step = true;
        continue;
      }
      accepted_state = trap_solution.TakeValue();
      proposed_step = std::clamp(adapted_step, minimum_step, maximum_step);
    }

    if (waveform_landing &&
        !EqualVectors(integration_rhs, current_rhs.value())) {
      Result<std::vector<double>> projected = ProjectReactiveState(
          system, accepted_state, current_rhs.value(),
          "waveform discontinuity projection", &factorization_cache);
      if (!projected.ok()) {
        return Result<TransientResult>::Fail(projected.error().code,
                                             projected.error().message);
      }
      accepted_state = projected.TakeValue();
    }

    if (accepted_steps >= limits.maximum_accepted_steps) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve, "transient analysis exceeded " +
                                 std::to_string(limits.maximum_accepted_steps) +
                                 " accepted timesteps");
    }
    ++accepted_steps;
    result.step_trace.push_back(TransientStepRecord{
        .start_time_seconds = time,
        .end_time_seconds = next_time,
        .step_size_seconds = step,
        .method = method,
        .accepted = true,
        .normalized_local_error = normalized_error,
        .landed_on_hard_point = lands_on_hard_point,
    });
    state = std::move(accepted_state);
    previous_rhs = current_rhs.TakeValue();
    time = next_time;

    if (time >= analysis.start_time_seconds) {
      result.times_seconds.push_back(time);
      result.states.push_back(state);
    }
  }

  if (time != analysis.stop_time_seconds || result.times_seconds.empty() ||
      result.times_seconds.back() != analysis.stop_time_seconds) {
    return Result<TransientResult>::Fail(
        ErrorCode::kSolve,
        "transient analysis did not include the exact requested stop time");
  }
  if (result.times_seconds.front() != analysis.start_time_seconds) {
    return Result<TransientResult>::Fail(
        ErrorCode::kSolve,
        "transient analysis did not include the exact output start time");
  }
  return Result<TransientResult>::Ok(std::move(result));
}

} // namespace ohmnivore
