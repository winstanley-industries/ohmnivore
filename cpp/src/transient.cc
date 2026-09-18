#include "ohmnivore/transient.h"
#include "ohmnivore/behavioral.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/waveform.h"

#include "cpp/src/nonlinear_internal.h"

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

// Only invocation-owned G/C matrices admitted by the ordinary companion builder
// reach this path. Numeric state and partial sums keep the checked-path order.
[[nodiscard]] Result<bool>
ValidateImmutableMultiplyState(const CsrMatrix &matrix,
                               const std::vector<double> &state) {
  if (matrix.columns != state.size())
    return Result<bool>::Fail(ErrorCode::kSolve,
                              "matrix-vector dimensions disagree");
  for (double value : state) {
    if (!std::isfinite(value))
      return Result<bool>::Fail(
          ErrorCode::kSolve, "matrix-vector input contains a non-finite value");
  }
  return Result<bool>::Ok(true);
}

[[nodiscard]] Result<std::vector<double>>
MultiplyImmutable(const CsrMatrix &matrix, const std::vector<double> &state) {
  auto valid = ValidateImmutableMultiplyState(matrix, state);
  if (!valid.ok())
    return Result<std::vector<double>>::Fail(valid.error().code,
                                             valid.error().message);
  std::vector<double> product(matrix.rows, 0.0);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      product[row] +=
          matrix.values[index] * state[matrix.column_indices[index]];
      if (!std::isfinite(product[row]))
        return Result<std::vector<double>>::Fail(
            ErrorCode::kSolve,
            "matrix-vector multiplication produced a non-finite value");
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

[[nodiscard]] std::optional<std::size_t>
FindValueIndex(const CsrMatrix &matrix, std::size_t row, std::size_t column) {
  if (row >= matrix.rows || matrix.row_offsets.size() != matrix.rows + 1) {
    return std::nullopt;
  }
  for (std::size_t index = matrix.row_offsets[row];
       index < matrix.row_offsets[row + 1]; ++index) {
    if (matrix.column_indices[index] == column) {
      return index;
    }
    if (matrix.column_indices[index] > column) {
      break;
    }
  }
  return std::nullopt;
}

[[nodiscard]] Result<MnaSystem>
BuildNonlinearSystemForMatrix(const MnaSystem &source,
                              const CsrMatrix &base_matrix,
                              const std::vector<double> &right_hand_side) {
  const std::size_t node_count = source.node_names.size();
  const std::size_t size = node_count + source.branch_names.size();
  if ((source.diode_descriptors.empty() &&
       source.behavioral_descriptors.empty()) ||
      base_matrix.rows != size || base_matrix.columns != size ||
      right_hand_side.size() != size) {
    return Result<MnaSystem>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear transient matrix dimensions or diode metadata are invalid");
  }
  if (const auto error = ValidateCsr(base_matrix); error.has_value()) {
    return Result<MnaSystem>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear transient base matrix is invalid: " + *error);
  }

  using Coordinate = std::pair<std::size_t, std::size_t>;
  std::map<Coordinate, double> entries;
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t index = base_matrix.row_offsets[row];
         index < base_matrix.row_offsets[row + 1]; ++index) {
      entries.emplace(Coordinate{row, base_matrix.column_indices[index]},
                      base_matrix.values[index]);
    }
  }

  std::vector<DiodeDescriptor> descriptors = source.diode_descriptors;
  for (DiodeDescriptor &descriptor : descriptors) {
    if ((descriptor.anode_node_index.has_value() &&
         *descriptor.anode_node_index >= node_count) ||
        (descriptor.cathode_node_index.has_value() &&
         *descriptor.cathode_node_index >= node_count)) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kInvalidStructure,
          "nonlinear transient diode descriptor references an invalid node");
    }
    const auto retain = [&](std::optional<std::size_t> row,
                            std::optional<std::size_t> column) {
      if (row.has_value() && column.has_value()) {
        entries.try_emplace(Coordinate{*row, *column}, 0.0);
      }
    };
    retain(descriptor.anode_node_index, descriptor.anode_node_index);
    retain(descriptor.anode_node_index, descriptor.cathode_node_index);
    retain(descriptor.cathode_node_index, descriptor.anode_node_index);
    retain(descriptor.cathode_node_index, descriptor.cathode_node_index);
  }

  for (const auto &descriptor : source.behavioral_descriptors) {
    for (const auto &row : descriptor.rows) {
      for (const auto column : descriptor.expression.dependencies()) {
        entries.try_emplace(Coordinate{row.row, column}, 0.0);
      }
    }
  }
  CsrMatrix union_matrix;
  union_matrix.rows = size;
  union_matrix.columns = size;
  union_matrix.row_offsets.reserve(size + 1);
  union_matrix.row_offsets.push_back(0);
  auto entry = entries.begin();
  for (std::size_t row = 0; row < size; ++row) {
    while (entry != entries.end() && entry->first.first == row) {
      union_matrix.column_indices.push_back(entry->first.second);
      union_matrix.values.push_back(entry->second);
      ++entry;
    }
    union_matrix.row_offsets.push_back(union_matrix.values.size());
  }

  for (DiodeDescriptor &descriptor : descriptors) {
    const auto position = [&](std::optional<std::size_t> row,
                              std::optional<std::size_t> column) {
      return row.has_value() && column.has_value()
                 ? FindValueIndex(union_matrix, *row, *column)
                 : std::nullopt;
    };
    descriptor.anode_anode_value_index =
        position(descriptor.anode_node_index, descriptor.anode_node_index);
    descriptor.anode_cathode_value_index =
        position(descriptor.anode_node_index, descriptor.cathode_node_index);
    descriptor.cathode_anode_value_index =
        position(descriptor.cathode_node_index, descriptor.anode_node_index);
    descriptor.cathode_cathode_value_index =
        position(descriptor.cathode_node_index, descriptor.cathode_node_index);
  }

  MnaSystem transformed = source;
  transformed.g = std::move(union_matrix);
  transformed.b_dc = right_hand_side;
  transformed.diode_descriptors = std::move(descriptors);
  auto remapped = RemapBehavioralDescriptors(&transformed);
  if (!remapped.ok())
    return Result<MnaSystem>::Fail(remapped.error().code,
                                   remapped.error().message);
  return Result<MnaSystem>::Ok(std::move(transformed));
}

class SparseRealFactorizationCache {
public:
  explicit SparseRealFactorizationCache(bool prepare_behavioral = true)
      : prepare_behavioral_(prepare_behavioral) {}

  [[nodiscard]] bool prepares_behavioral() const { return prepare_behavioral_; }

  [[nodiscard]] Result<const CsrMatrix *>
  CompanionMatrix(const MnaSystem &source, double step, double alpha,
                  CsrMatrix *ordinary_matrix) {
    if (!prepare_behavioral_ || source.behavioral_descriptors.empty()) {
      auto formed =
          FormTransientCompanionMatrix(source.g, source.c, step, alpha);
      if (!formed.ok())
        return Result<const CsrMatrix *>::Fail(formed.error().code,
                                               formed.error().message);
      *ordinary_matrix = formed.TakeValue();
      return Result<const CsrMatrix *>::Ok(ordinary_matrix);
    }
    if (!companion_) {
      auto prepared = internal::PreparedTransientCompanion::Create(
          source.g, source.c, step, alpha);
      if (!prepared.ok())
        return Result<const CsrMatrix *>::Fail(prepared.error().code,
                                               prepared.error().message);
      companion_ = prepared.TakeValue();
      return Result<const CsrMatrix *>::Ok(&companion_->matrix());
    }
    return companion_->Form(step, alpha);
  }

  [[nodiscard]] Result<std::vector<double>>
  BackwardEulerRhs(const MnaSystem &source, const std::vector<double> &state,
                   const std::vector<double> &rhs, double step) const {
    return companion_ ? companion_->BackwardEulerRhs(state, rhs, step)
                      : BuildBackwardEulerRhs(source.c, state, rhs, step);
  }

  [[nodiscard]] Result<std::vector<double>> TrapezoidalRhs(
      const MnaSystem &source, const std::vector<double> &state,
      const std::vector<double> &previous_rhs,
      const std::vector<double> &current_rhs, double step,
      std::optional<internal::PreparedTransientCompanion::StateProducts>
          *same_state_products) const {
    return companion_
               ? companion_->TrapezoidalRhs(state, previous_rhs, current_rhs,
                                            step, same_state_products)
               : BuildTrapezoidalRhs(source.g, source.c, state, previous_rhs,
                                     current_rhs, step);
  }

  [[nodiscard]] Result<NonlinearPointResult>
  SolveBehavioral(const MnaSystem &source, const CsrMatrix &matrix,
                  const std::vector<double> &rhs,
                  const std::vector<double> &initial_guess,
                  std::size_t maximum_iterations) {
    if (!behavioral_solver_) {
      // Only the first actual companion matrix establishes the immutable
      // pattern. Preparation never invents a timestep or caches a trial state.
      auto nonlinear = BuildNonlinearSystemForMatrix(source, matrix, rhs);
      if (!nonlinear.ok())
        return Result<NonlinearPointResult>::Fail(nonlinear.error().code,
                                                  nonlinear.error().message);
      auto prepared =
          internal::PreparedNonlinearPointSolver::Create(nonlinear.value());
      if (!prepared.ok())
        return Result<NonlinearPointResult>::Fail(prepared.error().code,
                                                  prepared.error().message);
      behavioral_solver_ = prepared.TakeValue();
    }
    return behavioral_solver_->Solve(matrix, rhs, initial_guess,
                                     maximum_iterations);
  }

  [[nodiscard]] Result<std::vector<double>>
  BehavioralHistory(const MnaSystem &source,
                    const std::vector<double> &state) const {
    return behavioral_solver_ ? behavioral_solver_->History(state)
                              : BuildDiodeResidualContribution(source, state);
  }

  [[nodiscard]] Result<SparseRealFactorization *> Get(const CsrMatrix &matrix) {
    Result<SolverCscPattern> converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok()) {
      return Result<SparseRealFactorization *>::Fail(converted.error().code,
                                                     converted.error().message);
    }
    for (Entry &entry : entries_) {
      if (entry.pattern.size == converted.value().size &&
          entry.pattern.column_offsets == converted.value().column_offsets &&
          entry.pattern.row_indices == converted.value().row_indices &&
          entry.pattern.csr_value_indices ==
              converted.value().csr_value_indices) {
        return Result<SparseRealFactorization *>::Ok(entry.factorization.get());
      }
    }
    Result<std::unique_ptr<SparseRealFactorization>> analyzed =
        SparseRealFactorization::Analyze(matrix);
    if (!analyzed.ok()) {
      return Result<SparseRealFactorization *>::Fail(analyzed.error().code,
                                                     analyzed.error().message);
    }
    entries_.push_back(Entry{.pattern = converted.TakeValue(),
                             .factorization = analyzed.TakeValue()});
    return Result<SparseRealFactorization *>::Ok(
        entries_.back().factorization.get());
  }

  [[nodiscard]] Result<std::vector<double>>
  Solve(const CsrMatrix &matrix, const std::vector<double> &rhs) {
    Result<SparseRealFactorization *> factorization = Get(matrix);
    if (!factorization.ok()) {
      return Result<std::vector<double>>::Fail(factorization.error().code,
                                               factorization.error().message);
    }
    return factorization.value()->FactorAndSolve(matrix, rhs);
  }

  [[nodiscard]] SparseSolverStatistics statistics() const {
    SparseSolverStatistics total;
    if (behavioral_solver_)
      total = behavioral_solver_->statistics();
    for (const Entry &entry : entries_) {
      const SparseSolverStatistics &current = entry.factorization->statistics();
      total.symbolic_analyses += current.symbolic_analyses;
      total.numeric_factorizations += current.numeric_factorizations;
      total.numeric_refactorizations += current.numeric_refactorizations;
      total.numeric_refactorization_fallbacks +=
          current.numeric_refactorization_fallbacks;
      total.numeric_reuses += current.numeric_reuses;
      total.iterative_refinement_solves += current.iterative_refinement_solves;
      total.solves += current.solves;
    }
    return total;
  }

private:
  struct Entry {
    SolverCscPattern pattern;
    std::unique_ptr<SparseRealFactorization> factorization;
  };
  std::vector<Entry> entries_;
  bool prepare_behavioral_;
  std::unique_ptr<internal::PreparedTransientCompanion> companion_;
  std::unique_ptr<internal::PreparedNonlinearPointSolver> behavioral_solver_;
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
  struct SelectedEquation {
    std::vector<double> coefficients;
    double right_hand_side;
    std::optional<std::size_t> physical_row;
  };
  std::vector<SelectedEquation> selected_equations;
  selected_equations.reserve(size);
  std::vector<RankBasisRow> basis;
  const auto select = [&](const std::vector<double> &row, double rhs,
                          std::optional<std::size_t> physical_row) {
    if (!AppendIfIndependent(row, &basis)) {
      return;
    }
    selected_equations.push_back(SelectedEquation{
        .coefficients = row,
        .right_hand_side = rhs,
        .physical_row = physical_row,
    });
  };

  // Reactive state constraints are mandatory.  Preserve every algebraic row
  // that cannot carry a capacitor impulse or replace an inductor constitutive
  // equation before using the remaining dynamic rows to complete the rank.
  for (const ReactiveConstraintEquation &constraint : constraints) {
    select(constraint.coefficients, constraint.right_hand_side, std::nullopt);
  }
  for (std::size_t row = 0; row < size; ++row) {
    if (!replaceable_row[row]) {
      select(
          std::vector<double>(
              dense_g.begin() + static_cast<std::ptrdiff_t>(row * size),
              dense_g.begin() + static_cast<std::ptrdiff_t>((row + 1) * size)),
          source_rhs[row], row);
    }
  }
  for (std::size_t row = 0; row < size && selected_equations.size() < size;
       ++row) {
    if (replaceable_row[row]) {
      select(
          std::vector<double>(
              dense_g.begin() + static_cast<std::ptrdiff_t>(row * size),
              dense_g.begin() + static_cast<std::ptrdiff_t>((row + 1) * size)),
          source_rhs[row], row);
    }
  }
  if (selected_equations.size() != size) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSingular,
        context + " constraints do not define a unique algebraic state");
  }

  std::vector<double> selected_dense;
  std::vector<double> selected_rhs;
  std::vector<bool> active_node_equations(node_count, false);
  std::vector<bool> retained_physical_row(size, false);
  if ((system.diode_descriptors.empty() &&
       system.behavioral_descriptors.empty())) {
    selected_dense.reserve(size * size);
    selected_rhs.reserve(size);
    for (const SelectedEquation &equation : selected_equations) {
      selected_dense.insert(selected_dense.end(), equation.coefficients.begin(),
                            equation.coefficients.end());
      selected_rhs.push_back(equation.right_hand_side);
    }
  } else {
    selected_dense.assign(size * size, 0.0);
    selected_rhs.assign(size, 0.0);
    std::vector<bool> occupied(size, false);
    for (const SelectedEquation &equation : selected_equations) {
      if (!equation.physical_row.has_value()) {
        continue;
      }
      const std::size_t row = *equation.physical_row;
      std::copy(equation.coefficients.begin(), equation.coefficients.end(),
                selected_dense.begin() +
                    static_cast<std::ptrdiff_t>(row * size));
      selected_rhs[row] = equation.right_hand_side;
      occupied[row] = true;
      retained_physical_row[row] = true;
      if (row < node_count) {
        active_node_equations[row] = true;
      }
    }
    for (const SelectedEquation &equation : selected_equations) {
      if (equation.physical_row.has_value()) {
        continue;
      }
      std::optional<std::size_t> slot;
      for (std::size_t row = 0; row < size; ++row) {
        if (!occupied[row] && replaceable_row[row]) {
          slot = row;
          break;
        }
      }
      // An independent physical equation can be redundant with a mandatory
      // reactive constraint, leaving only its original row available.  Keep
      // the established replaceable-row preference, then deterministically
      // use the first remaining row; active diode equations are still keyed
      // only to retained physical node rows above.
      if (!slot.has_value()) {
        for (std::size_t row = 0; row < size; ++row) {
          if (!occupied[row]) {
            slot = row;
            break;
          }
        }
      }
      if (!slot.has_value()) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kInvalidStructure,
            context + " cannot place an independent reactive constraint");
      }
      const std::size_t row = *slot;
      std::copy(equation.coefficients.begin(), equation.coefficients.end(),
                selected_dense.begin() +
                    static_cast<std::ptrdiff_t>(row * size));
      selected_rhs[row] = equation.right_hand_side;
      occupied[row] = true;
    }
  }

  Result<CsrMatrix> matrix = DenseToCsr(selected_dense, size);
  if (!matrix.ok()) {
    return Result<std::vector<double>>::Fail(
        matrix.error().code,
        context + " matrix construction failed: " + matrix.error().message);
  }
  Result<std::vector<double>> solved = [&]() {
    if ((system.diode_descriptors.empty() &&
         system.behavioral_descriptors.empty())) {
      return factorization_cache->Solve(matrix.value(), selected_rhs);
    }
    Result<MnaSystem> nonlinear =
        BuildNonlinearSystemForMatrix(system, matrix.value(), selected_rhs);
    if (!nonlinear.ok()) {
      return Result<std::vector<double>>::Fail(nonlinear.error().code,
                                               nonlinear.error().message);
    }
    Result<SparseRealFactorization *> factorization =
        factorization_cache->Get(nonlinear.value().g);
    if (!factorization.ok()) {
      return Result<std::vector<double>>::Fail(factorization.error().code,
                                               factorization.error().message);
    }
    Result<NonlinearPointResult> point =
        internal::RunNonlinearPointForProjection(
            nonlinear.value(), target_state, active_node_equations,
            factorization.value(), kDirectNewtonMaximumIterations);
    if (!point.ok()) {
      return Result<std::vector<double>>::Fail(point.error().code,
                                               point.error().message);
    }
    return Result<std::vector<double>>::Ok(point.TakeValue().solution);
  }();
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
  std::vector<double> diode_residual(size, 0.0);
  if (!(system.diode_descriptors.empty() &&
        system.behavioral_descriptors.empty())) {
    Result<std::vector<double>> evaluated =
        BuildDiodeResidualContribution(system, solved.value());
    if (!evaluated.ok()) {
      return Result<std::vector<double>>::Fail(
          evaluated.error().code, context + " residual validation failed: " +
                                      evaluated.error().message);
    }
    diode_residual = evaluated.TakeValue();
  }
  for (std::size_t row = 0; row < size; ++row) {
    if ((system.diode_descriptors.empty() &&
         system.behavioral_descriptors.empty())
            ? replaceable_row[row]
            : replaceable_row[row] && !retained_physical_row[row]) {
      continue;
    }
    const std::vector<double> equation(
        dense_g.begin() + static_cast<std::ptrdiff_t>(row * size),
        dense_g.begin() + static_cast<std::ptrdiff_t>((row + 1) * size));
    if (!EquationIsSatisfied(equation, source_rhs[row] - diode_residual[row],
                             solved.value())) {
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

[[nodiscard]] Result<double> ComputeNormalizedLocalError(
    const MnaSystem &system, const std::vector<double> &higher_accuracy,
    const std::vector<double> &lower_accuracy, double estimate_multiplier) {
  if (higher_accuracy.size() != lower_accuracy.size() ||
      !std::isfinite(estimate_multiplier) || estimate_multiplier <= 0.0) {
    return Result<double>::Fail(ErrorCode::kSolve,
                                "LTE solution dimensions disagree");
  }
  double normalized_error = 0.0;
  if (!system.behavioral_descriptors.empty()) {
    const auto compare = [&](double high, double low,
                             double absolute) -> Result<bool> {
      const double scale =
          absolute + BehavioralNumericalPolicy::lte_relative_tolerance *
                         std::max(std::abs(high), std::abs(low));
      const double error = estimate_multiplier * std::abs(high - low) / scale;
      if (!std::isfinite(high) || !std::isfinite(low) ||
          !std::isfinite(error) || scale <= 0) {
        return Result<bool>::Fail(
            ErrorCode::kNonFinite,
            "behavioral reactive-state LTE is non-finite");
      }
      normalized_error = std::max(normalized_error, error);
      return Result<bool>::Ok(true);
    };
    for (const auto &capacitor : system.capacitor_initial_constraints) {
      const auto voltage = [&](const std::vector<double> &state) {
        return (capacitor.positive_node_index
                    ? state[*capacitor.positive_node_index]
                    : 0.0) -
               (capacitor.negative_node_index
                    ? state[*capacitor.negative_node_index]
                    : 0.0);
      };
      auto valid =
          compare(voltage(higher_accuracy), voltage(lower_accuracy),
                  BehavioralNumericalPolicy::voltage_absolute_tolerance);
      if (!valid.ok())
        return Result<double>::Fail(valid.error().code, valid.error().message);
    }
    for (const auto &inductor : system.inductor_initial_constraints) {
      auto valid =
          compare(higher_accuracy[inductor.branch_index],
                  lower_accuracy[inductor.branch_index],
                  BehavioralNumericalPolicy::current_absolute_tolerance);
      if (!valid.ok())
        return Result<double>::Fail(valid.error().code, valid.error().message);
    }
    return Result<double>::Ok(normalized_error);
  }
  for (std::size_t index = 0; index < higher_accuracy.size(); ++index) {
    if (!std::isfinite(higher_accuracy[index]) ||
        !std::isfinite(lower_accuracy[index])) {
      return Result<double>::Fail(
          ErrorCode::kSolve, "LTE input contains a non-finite solution value");
    }
    const double difference = higher_accuracy[index] - lower_accuracy[index];
    const double estimate = estimate_multiplier * std::abs(difference);
    const bool behavioral = !system.behavioral_descriptors.empty();
    const double absolute =
        behavioral
            ? (index < system.node_names.size()
                   ? BehavioralNumericalPolicy::voltage_absolute_tolerance
                   : BehavioralNumericalPolicy::current_absolute_tolerance)
            : kTransientAbsoluteTolerance;
    const double relative =
        behavioral ? BehavioralNumericalPolicy::lte_relative_tolerance
                   : kTransientRelativeTolerance;
    const double scale =
        absolute + relative * std::max(std::abs(higher_accuracy[index]),
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

namespace internal {

Result<std::unique_ptr<PreparedTransientCompanion>>
PreparedTransientCompanion::Create(const CsrMatrix &g, const CsrMatrix &c,
                                   double step, double alpha) {
  auto first = FormTransientCompanionMatrix(g, c, step, alpha);
  if (!first.ok())
    return Result<std::unique_ptr<PreparedTransientCompanion>>::Fail(
        first.error().code, first.error().message);
  constexpr auto missing = std::numeric_limits<std::size_t>::max();
  std::vector<UnionEntry> entries;
  entries.reserve(first.value().values.size());
  for (std::size_t row = 0; row < g.rows; ++row) {
    std::size_t gi = g.row_offsets[row];
    std::size_t ci = c.row_offsets[row];
    for (std::size_t index = first.value().row_offsets[row];
         index < first.value().row_offsets[row + 1]; ++index) {
      const auto column = first.value().column_indices[index];
      const bool has_g =
          gi < g.row_offsets[row + 1] && g.column_indices[gi] == column;
      const bool has_c =
          ci < c.row_offsets[row + 1] && c.column_indices[ci] == column;
      entries.push_back({has_g ? gi++ : missing, has_c ? ci++ : missing});
    }
  }
  return Result<std::unique_ptr<PreparedTransientCompanion>>::Ok(
      std::unique_ptr<PreparedTransientCompanion>(
          new PreparedTransientCompanion(g, c, first.TakeValue(),
                                         std::move(entries))));
}

Result<const CsrMatrix *> PreparedTransientCompanion::Form(double step,
                                                           double alpha) {
  if (!std::isfinite(step) || step <= 0.0 || !std::isfinite(alpha) ||
      alpha <= 0.0)
    return Result<const CsrMatrix *>::Fail(
        ErrorCode::kSolve,
        "transient companion step and alpha must be finite and positive");
  const double factor = alpha / step;
  if (!std::isfinite(factor))
    return Result<const CsrMatrix *>::Fail(
        ErrorCode::kSolve,
        "transient companion scaling produced a non-finite value");
  constexpr auto missing = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0; index < entries_.size(); ++index) {
    const auto entry = entries_[index];
    const double value =
        entry.c_index == missing ? g_.values[entry.g_index]
        : entry.g_index == missing
            ? factor * c_.values[entry.c_index]
            : g_.values[entry.g_index] + factor * c_.values[entry.c_index];
    if (!std::isfinite(value))
      return Result<const CsrMatrix *>::Fail(
          ErrorCode::kSolve,
          "transient companion matrix produced a non-finite value");
    matrix_.values[index] = value;
  }
  return Result<const CsrMatrix *>::Ok(&matrix_);
}

Result<std::vector<double>>
PreparedTransientCompanion::BackwardEulerRhs(const std::vector<double> &state,
                                             const std::vector<double> &rhs,
                                             double step) const {
  if (!std::isfinite(step) || step <= 0.0 || c_.rows != rhs.size())
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "invalid backward-Euler RHS dimensions or step");
  auto product = MultiplyImmutable(c_, state);
  if (!product.ok())
    return product;
  const double factor = 1.0 / step;
  if (!std::isfinite(factor))
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve,
        "backward-Euler RHS scaling produced a non-finite value");
  std::vector<double> result(rhs.size(), 0.0);
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (!std::isfinite(rhs[index]))
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "backward-Euler source RHS contains a non-finite value");
    result[index] = rhs[index] + factor * product.value()[index];
    if (!std::isfinite(result[index]))
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "backward-Euler RHS formation produced a non-finite value");
  }
  return Result<std::vector<double>>::Ok(std::move(result));
}

Result<std::vector<double>> PreparedTransientCompanion::TrapezoidalRhs(
    const std::vector<double> &state, const std::vector<double> &previous_rhs,
    const std::vector<double> &current_rhs, double step,
    std::optional<StateProducts> *same_state_products) const {
  if (!std::isfinite(step) || step <= 0.0 || g_.rows != c_.rows ||
      g_.columns != c_.columns || g_.rows != previous_rhs.size() ||
      g_.rows != current_rhs.size())
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "invalid trapezoidal RHS dimensions or step");
  StateProducts private_products;
  const StateProducts *products;
  if (same_state_products && same_state_products->has_value()) {
    auto valid = ValidateImmutableMultiplyState(g_, state);
    if (!valid.ok())
      return Result<std::vector<double>>::Fail(valid.error().code,
                                               valid.error().message);
    valid = ValidateImmutableMultiplyState(c_, state);
    if (!valid.ok())
      return Result<std::vector<double>>::Fail(valid.error().code,
                                               valid.error().message);
    products = &same_state_products->value();
    if (products->owner != this || products->g.size() != g_.rows ||
        products->c.size() != c_.rows)
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          "prepared transient products have a different owner or invalid "
          "dimensions");
  } else {
    auto g_product = MultiplyImmutable(g_, state);
    if (!g_product.ok())
      return g_product;
    auto c_product = MultiplyImmutable(c_, state);
    if (!c_product.ok())
      return c_product;
    private_products = {this, g_product.TakeValue(), c_product.TakeValue()};
    if (same_state_products) {
      *same_state_products = std::move(private_products);
      products = &same_state_products->value();
    } else {
      products = &private_products;
    }
  }
  const double factor = 2.0 / step;
  if (!std::isfinite(factor))
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve,
        "trapezoidal RHS scaling produced a non-finite value");
  std::vector<double> result(current_rhs.size(), 0.0);
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (!std::isfinite(previous_rhs[index]) ||
        !std::isfinite(current_rhs[index]))
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "trapezoidal source RHS contains a non-finite value");
    result[index] = current_rhs[index] + previous_rhs[index] +
                    factor * products->c[index] - products->g[index];
    if (!std::isfinite(result[index]))
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve,
          "trapezoidal RHS formation produced a non-finite value");
  }
  return Result<std::vector<double>>::Ok(std::move(result));
}

} // namespace internal

namespace {

[[nodiscard]] Result<std::vector<double>>
SolveTransientSystem(const MnaSystem &system, const CsrMatrix &matrix,
                     const std::vector<double> &right_hand_side,
                     const std::vector<double> &initial_guess,
                     std::size_t nonlinear_maximum_iterations,
                     const std::string &context,
                     SparseRealFactorizationCache *factorization_cache) {
  if ((system.diode_descriptors.empty() &&
       system.behavioral_descriptors.empty())) {
    Result<std::vector<double>> solved =
        factorization_cache->Solve(matrix, right_hand_side);
    if (!solved.ok()) {
      return Result<std::vector<double>>::Fail(
          solved.error().code, context + ": " + solved.error().message);
    }
    return solved;
  }
  if (!system.behavioral_descriptors.empty() &&
      factorization_cache->prepares_behavioral()) {
    auto solved = factorization_cache->SolveBehavioral(
        system, matrix, right_hand_side, initial_guess,
        nonlinear_maximum_iterations);
    if (!solved.ok())
      return Result<std::vector<double>>::Fail(
          solved.error().code, context + ": " + solved.error().message);
    return Result<std::vector<double>>::Ok(solved.TakeValue().solution);
  }
  Result<MnaSystem> nonlinear =
      BuildNonlinearSystemForMatrix(system, matrix, right_hand_side);
  if (!nonlinear.ok()) {
    return Result<std::vector<double>>::Fail(
        nonlinear.error().code, context + ": " + nonlinear.error().message);
  }
  Result<SparseRealFactorization *> factorization =
      factorization_cache->Get(nonlinear.value().g);
  if (!factorization.ok()) {
    return Result<std::vector<double>>::Fail(factorization.error().code,
                                             context + ": " +
                                                 factorization.error().message);
  }
  Result<NonlinearPointResult> solved =
      RunNonlinearPoint(nonlinear.value(), initial_guess, factorization.value(),
                        nonlinear_maximum_iterations);
  if (!solved.ok()) {
    return Result<std::vector<double>>::Fail(
        solved.error().code, context + ": " + solved.error().message);
  }
  return Result<std::vector<double>>::Ok(solved.TakeValue().solution);
}

[[nodiscard]] Result<std::vector<double>> SolveBackwardEulerStep(
    const MnaSystem &system, const std::vector<double> &previous_state,
    const std::vector<double> &current_rhs, double step_size_seconds,
    std::size_t nonlinear_maximum_iterations, std::string context,
    SparseRealFactorizationCache *factorization_cache) {
  CsrMatrix ordinary_matrix;
  auto matrix = factorization_cache->CompanionMatrix(system, step_size_seconds,
                                                     1.0, &ordinary_matrix);
  if (!matrix.ok()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, context + ": " + matrix.error().message);
  }
  Result<std::vector<double>> rhs = factorization_cache->BackwardEulerRhs(
      system, previous_state, current_rhs, step_size_seconds);
  if (!rhs.ok()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, context + ": " + rhs.error().message);
  }
  return SolveTransientSystem(system, *matrix.value(), rhs.value(),
                              previous_state, nonlinear_maximum_iterations,
                              context, factorization_cache);
}

[[nodiscard]] bool EqualVectors(const std::vector<double> &first,
                                const std::vector<double> &second) {
  return first.size() == second.size() &&
         std::equal(first.begin(), first.end(), second.begin());
}

[[nodiscard]] Result<std::vector<double>> SolveTrapezoidalStep(
    const MnaSystem &system, const std::vector<double> &state,
    const std::vector<double> &previous_rhs,
    const std::vector<double> &current_rhs, double step,
    std::size_t nonlinear_maximum_iterations,
    SparseRealFactorizationCache *factorization_cache,
    std::optional<std::vector<double>> *same_state_history = nullptr,
    std::optional<internal::PreparedTransientCompanion::StateProducts>
        *same_state_products = nullptr) {
  CsrMatrix ordinary_matrix;
  auto trap_matrix =
      factorization_cache->CompanionMatrix(system, step, 2.0, &ordinary_matrix);
  if (!trap_matrix.ok()) {
    return Result<std::vector<double>>::Fail(ErrorCode::kSolve,
                                             trap_matrix.error().message);
  }
  Result<std::vector<double>> trap_rhs = factorization_cache->TrapezoidalRhs(
      system, state, previous_rhs, current_rhs, step, same_state_products);
  if (!trap_rhs.ok()) {
    return Result<std::vector<double>>::Fail(ErrorCode::kSolve,
                                             trap_rhs.error().message);
  }
  std::vector<double> trap_rhs_values = trap_rhs.TakeValue();
  if (!(system.diode_descriptors.empty() &&
        system.behavioral_descriptors.empty())) {
    std::vector<double> private_history;
    const std::vector<double> *history;
    if (same_state_history != nullptr && same_state_history->has_value()) {
      history = &same_state_history->value();
    } else {
      auto evaluated =
          system.behavioral_descriptors.empty()
              ? BuildDiodeResidualContribution(system, state)
              : factorization_cache->BehavioralHistory(system, state);
      if (!evaluated.ok()) {
        return Result<std::vector<double>>::Fail(evaluated.error().code,
                                                 evaluated.error().message);
      }
      // Fill lazily at the original evaluation point, after matrix/RHS
      // construction. Domain/derivative guards and failure ordering stay
      // intact.
      if (same_state_history != nullptr) {
        *same_state_history = evaluated.TakeValue();
        history = &same_state_history->value();
      } else {
        private_history = evaluated.TakeValue();
        history = &private_history;
      }
    }
    for (std::size_t row = 0; row < trap_rhs_values.size(); ++row) {
      trap_rhs_values[row] -= (*history)[row];
      if (!std::isfinite(trap_rhs_values[row]) ||
          std::abs(trap_rhs_values[row]) > kNonlinearMaximumMagnitude) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kNonFinite,
            "trapezoidal diode history produced a non-finite or "
            "over-bound value");
      }
    }
  }
  Result<std::vector<double>> trap_solution = SolveTransientSystem(
      system, *trap_matrix.value(), trap_rhs_values, state,
      nonlinear_maximum_iterations, "trapezoidal transient solve failed",
      factorization_cache);
  if (!trap_solution.ok()) {
    return Result<std::vector<double>>::Fail(trap_solution.error().code,
                                             trap_solution.error().message);
  }

  return trap_solution;
}

struct BehavioralDerivativeSample {
  double time;
  std::vector<long double> derivatives;
};

struct BehavioralDerivativeHistory {
  std::optional<BehavioralDerivativeSample> current;
  std::optional<BehavioralDerivativeSample> older;
};

inline constexpr std::size_t kHistoryAuditAcceptedTrapInterval = 32;
inline constexpr std::size_t kHistoryRecoveryAcceptedAudits = 16;
inline constexpr double kHistoryAuditUnderestimateRatio = 2.0;
inline constexpr double kHistoryAuditMinimumError = .01;
inline constexpr long double kHistoryPositiveFeedbackLimit = .5L;
inline constexpr long double kHistoryFeedbackActivityFraction = .01L;

struct BehavioralDerivativeControl {
  bool first_audit_pending = true;
  std::size_t accepted_trap_since_audit = 0;
  bool fallback_active = false;
  std::size_t consecutive_accepted_agreements = 0;
};

[[nodiscard]] bool HistoryValueIsBounded(long double value) {
  return std::abs(value) <= kNonlinearMaximumMagnitude;
}

[[nodiscard]] Result<std::vector<long double>>
PhysicalHistoryCoordinates(const MnaSystem &system,
                           const std::vector<double> &state) {
  if (state.size() != system.g.rows)
    return Result<std::vector<long double>>::Fail(
        ErrorCode::kInvalidStructure,
        "derivative-history state dimensions disagree");
  for (double value : state) {
    if (!HistoryValueIsBounded(value))
      return Result<std::vector<long double>>::Fail(
          ErrorCode::kNonFinite,
          "derivative-history state is non-finite or over-bound");
  }
  std::vector<long double> coordinates;
  coordinates.reserve(system.capacitor_initial_constraints.size() +
                      system.inductor_initial_constraints.size());
  for (const auto &capacitor : system.capacitor_initial_constraints) {
    if ((capacitor.positive_node_index &&
         *capacitor.positive_node_index >= system.node_names.size()) ||
        (capacitor.negative_node_index &&
         *capacitor.negative_node_index >= system.node_names.size()))
      return Result<std::vector<long double>>::Fail(
          ErrorCode::kInvalidStructure,
          "derivative-history capacitor index is invalid");
    const long double positive = capacitor.positive_node_index
                                     ? state[*capacitor.positive_node_index]
                                     : 0.0L;
    const long double negative = capacitor.negative_node_index
                                     ? state[*capacitor.negative_node_index]
                                     : 0.0L;
    const long double voltage = positive - negative;
    if (!HistoryValueIsBounded(voltage))
      return Result<std::vector<long double>>::Fail(
          ErrorCode::kNonFinite,
          "derivative-history capacitor voltage is over-bound");
    coordinates.push_back(voltage);
  }
  for (const auto &inductor : system.inductor_initial_constraints) {
    if (inductor.branch_index < system.node_names.size() ||
        inductor.branch_index >= state.size())
      return Result<std::vector<long double>>::Fail(
          ErrorCode::kInvalidStructure,
          "derivative-history inductor index is invalid");
    coordinates.push_back(state[inductor.branch_index]);
  }
  return Result<std::vector<long double>>::Ok(std::move(coordinates));
}

// These two state objects remain immutable until this one full-step attempt
// ends. Only the original checked coordinate extraction can populate a slot;
// no trial from a private half step or later retry receives this cache.
struct BehavioralAttemptCoordinates {
  const std::vector<double> *before_state;
  const std::vector<double> *after_state;
  std::optional<std::vector<long double>> before;
  std::optional<std::vector<long double>> after;
};

[[nodiscard]] Result<const std::vector<long double> *>
PhysicalHistoryCoordinatesForAttempt(
    const MnaSystem &system, const std::vector<double> &state,
    std::optional<std::vector<long double>> *coordinates,
    const std::vector<double> *expected_state) {
  using Coordinates = Result<const std::vector<long double> *>;
  if (expected_state != &state)
    return Coordinates::Fail(ErrorCode::kInvalidStructure,
                             "derivative-history coordinate owner mismatch");
  if (!coordinates->has_value()) {
    auto checked = PhysicalHistoryCoordinates(system, state);
    if (!checked.ok())
      return Coordinates::Fail(checked.error().code, checked.error().message);
    *coordinates = checked.TakeValue();
  }
  return Coordinates::Ok(&coordinates->value());
}

// A null preceding derivative requests the final half-BE derivative. A TRAP
// derivative is reconstructed from its own full-step states and the derivative
// at that step's accepted start; private half steps never enter this history.
[[nodiscard]] Result<std::vector<long double>> ReconstructHistoryDerivative(
    const MnaSystem &system, const std::vector<double> &before,
    const std::vector<double> &after, double before_time, double after_time,
    const std::vector<long double> *preceding_derivative = nullptr,
    BehavioralAttemptCoordinates *cached_coordinates = nullptr) {
  const long double h = static_cast<long double>(after_time) - before_time;
  if (!HistoryValueIsBounded(h) || h <= 0)
    return Result<std::vector<long double>>::Fail(
        ErrorCode::kNonFinite, "derivative-history time gap is invalid");
  std::optional<std::vector<long double>> local_before, local_after;
  auto first = PhysicalHistoryCoordinatesForAttempt(
      system, before,
      cached_coordinates ? &cached_coordinates->before : &local_before,
      cached_coordinates ? cached_coordinates->before_state : &before);
  if (!first.ok())
    return Result<std::vector<long double>>::Fail(first.error().code,
                                                  first.error().message);
  auto last = PhysicalHistoryCoordinatesForAttempt(
      system, after,
      cached_coordinates ? &cached_coordinates->after : &local_after,
      cached_coordinates ? cached_coordinates->after_state : &after);
  if (!last.ok())
    return Result<std::vector<long double>>::Fail(last.error().code,
                                                  last.error().message);
  if (preceding_derivative &&
      preceding_derivative->size() != first.value()->size())
    return Result<std::vector<long double>>::Fail(
        ErrorCode::kInvalidStructure,
        "derivative-history coordinate counts disagree");
  std::vector<long double> derivative(first.value()->size(), 0.0L);
  for (std::size_t i = 0; i < derivative.size(); ++i) {
    const long double difference = (*last.value())[i] - (*first.value())[i];
    const long double numerator =
        (preceding_derivative ? 2.0L : 1.0L) * difference;
    const long double secant = numerator / h;
    const long double previous =
        preceding_derivative ? (*preceding_derivative)[i] : 0.0L;
    derivative[i] = secant - previous;
    if (!HistoryValueIsBounded(difference) ||
        !HistoryValueIsBounded(numerator) || !HistoryValueIsBounded(secant) ||
        !HistoryValueIsBounded(previous) ||
        !HistoryValueIsBounded(derivative[i]))
      return Result<std::vector<long double>>::Fail(
          ErrorCode::kNonFinite, "derivative-history reconstruction overflow");
  }
  return Result<std::vector<long double>>::Ok(std::move(derivative));
}

[[nodiscard]] Result<double> EstimateDerivativeHistoryError(
    const MnaSystem &system, const std::vector<double> &trial_state,
    const std::vector<long double> &trial_derivatives, double time,
    double next_time, const BehavioralDerivativeHistory &history,
    BehavioralAttemptCoordinates *cached_coordinates = nullptr) {
  if (!history.current || !history.older || history.current->time != time)
    return Result<double>::Fail(
        ErrorCode::kInvalidStructure,
        "derivative-history estimator lacks accepted history");
  std::optional<std::vector<long double>> local_coordinates;
  auto coordinates = PhysicalHistoryCoordinatesForAttempt(
      system, trial_state,
      cached_coordinates ? &cached_coordinates->after : &local_coordinates,
      cached_coordinates ? cached_coordinates->after_state : &trial_state);
  if (!coordinates.ok())
    return Result<double>::Fail(coordinates.error().code,
                                coordinates.error().message);
  const std::size_t size = coordinates.value()->size();
  if (trial_derivatives.size() != size ||
      history.current->derivatives.size() != size ||
      history.older->derivatives.size() != size)
    return Result<double>::Fail(
        ErrorCode::kInvalidStructure,
        "derivative-history estimator dimensions disagree");
  const long double h = static_cast<long double>(next_time) - time;
  const long double k = static_cast<long double>(time) - history.older->time;
  const long double span = h + k;
  const long double h_squared = h * h;
  const long double h_cubed = h_squared * h;
  if (!HistoryValueIsBounded(h) || h <= 0 || !HistoryValueIsBounded(k) ||
      k <= 0 || !HistoryValueIsBounded(span) || span <= 0 ||
      !HistoryValueIsBounded(h_squared) || !HistoryValueIsBounded(h_cubed))
    return Result<double>::Fail(
        ErrorCode::kNonFinite,
        "derivative-history divided-difference time overflow");
  long double maximum = 0.0L;
  for (std::size_t i = 0; i < size; ++i) {
    const long double first =
        trial_derivatives[i] - history.current->derivatives[i];
    const long double previous =
        history.current->derivatives[i] - history.older->derivatives[i];
    const long double first_slope = first / h;
    const long double previous_slope = previous / k;
    const long double slope_change = first_slope - previous_slope;
    const long double second = slope_change / span;
    const long double numerator = h_cubed * second;
    const long double error = numerator / 6.0L;
    const long double refined = (*coordinates.value())[i] - .75L * error;
    const long double absolute =
        i < system.capacitor_initial_constraints.size()
            ? BehavioralNumericalPolicy::voltage_absolute_tolerance
            : BehavioralNumericalPolicy::current_absolute_tolerance;
    const long double scale =
        absolute +
        BehavioralNumericalPolicy::lte_relative_tolerance *
            std::max(std::abs((*coordinates.value())[i]), std::abs(refined));
    const long double normalized = std::abs(error) / scale;
    for (long double value :
         {trial_derivatives[i], history.current->derivatives[i],
          history.older->derivatives[i], first, previous, first_slope,
          previous_slope, slope_change, second, numerator, error, refined,
          scale, normalized}) {
      if (!HistoryValueIsBounded(value))
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "derivative-history error-estimation overflow");
    }
    if (scale <= 0)
      return Result<double>::Fail(
          ErrorCode::kNonFinite,
          "derivative-history error scale is nonpositive");
    maximum = std::max(maximum, normalized);
  }
  const double result = static_cast<double>(maximum);
  if (!HistoryValueIsBounded(result))
    return Result<double>::Fail(
        ErrorCode::kNonFinite,
        "derivative-history normalized error is over-bound");
  return Result<double>::Ok(result);
}

[[nodiscard]] Result<bool> HasPositiveDerivativeHistoryFeedback(
    const MnaSystem &system, const std::vector<double> &before,
    const std::vector<double> &after,
    const std::vector<long double> &derivative,
    const std::vector<long double> &previous_derivative, double time,
    double next_time,
    BehavioralAttemptCoordinates *cached_coordinates = nullptr) {
  std::optional<std::vector<long double>> local_before, local_after;
  auto first = PhysicalHistoryCoordinatesForAttempt(
      system, before,
      cached_coordinates ? &cached_coordinates->before : &local_before,
      cached_coordinates ? cached_coordinates->before_state : &before);
  if (!first.ok())
    return Result<bool>::Fail(first.error().code, first.error().message);
  auto last = PhysicalHistoryCoordinatesForAttempt(
      system, after,
      cached_coordinates ? &cached_coordinates->after : &local_after,
      cached_coordinates ? cached_coordinates->after_state : &after);
  if (!last.ok())
    return Result<bool>::Fail(last.error().code, last.error().message);
  if (derivative.size() != first.value()->size() ||
      previous_derivative.size() != first.value()->size())
    return Result<bool>::Fail(
        ErrorCode::kInvalidStructure,
        "derivative-history feedback dimensions disagree");
  const long double h = static_cast<long double>(next_time) - time;
  if (!HistoryValueIsBounded(h) || h <= 0)
    return Result<bool>::Fail(
        ErrorCode::kNonFinite,
        "derivative-history feedback time gap is invalid");
  bool positive_feedback = false;
  for (std::size_t i = 0; i < derivative.size(); ++i) {
    const long double change = (*last.value())[i] - (*first.value())[i];
    const long double absolute =
        i < system.capacitor_initial_constraints.size()
            ? BehavioralNumericalPolicy::voltage_absolute_tolerance
            : BehavioralNumericalPolicy::current_absolute_tolerance;
    const long double tolerance =
        absolute + BehavioralNumericalPolicy::lte_relative_tolerance *
                       std::max(std::abs((*first.value())[i]),
                                std::abs((*last.value())[i]));
    if (!HistoryValueIsBounded(change) || !HistoryValueIsBounded(tolerance) ||
        tolerance <= 0)
      return Result<bool>::Fail(ErrorCode::kNonFinite,
                                "derivative-history feedback scale overflow");
    if (std::abs(change) <= kHistoryFeedbackActivityFraction * tolerance)
      continue;
    const long double derivative_change =
        derivative[i] - previous_derivative[i];
    const long double numerator = h * derivative_change;
    const long double gain = numerator / change;
    if (!HistoryValueIsBounded(derivative_change) ||
        !HistoryValueIsBounded(numerator) || !HistoryValueIsBounded(gain))
      return Result<bool>::Fail(
          ErrorCode::kNonFinite,
          "derivative-history positive-feedback overflow");
    positive_feedback =
        positive_feedback || gain >= kHistoryPositiveFeedbackLimit;
  }
  return Result<bool>::Ok(positive_feedback);
}

struct IntegratedStepAttempt {
  std::vector<double> accepted_state;
  double normalized_error;
  double adapted_step;
  std::optional<std::vector<long double>> proposed_derivatives;
  bool used_step_doubling = false;
  bool used_derivative_history = false;
  bool checked_derivative_history = false;
  bool audited_derivative_history = false;
  bool disable_derivative_history = false;
  bool derivative_history_audit_agreed = false;
};

[[nodiscard]] Result<IntegratedStepAttempt> IntegrateTransientStepAttempt(
    const MnaSystem &system, const std::vector<double> &state,
    const std::vector<double> &previous_rhs,
    const std::vector<double> &current_rhs,
    const std::vector<double> &integration_rhs, double time, double next_time,
    double step, bool use_backward_euler, bool lands_on_hard_point,
    std::size_t nonlinear_maximum_iterations,
    SparseRealFactorizationCache *factorization_cache,
    const BehavioralDerivativeHistory *derivative_history = nullptr,
    const BehavioralDerivativeControl *derivative_control = nullptr) {
  std::vector<double> accepted_state;
  double normalized_error = 0.0;
  std::optional<std::vector<long double>> proposed_derivatives;
  bool used_step_doubling = false;
  bool used_derivative_history = false;
  bool checked_derivative_history = false;
  bool audited_derivative_history = false;
  bool disable_derivative_history = false;
  bool derivative_history_audit_agreed = false;
  if ((derivative_history == nullptr) != (derivative_control == nullptr))
    return Result<IntegratedStepAttempt>::Fail(
        ErrorCode::kInvalidStructure,
        "derivative-history execution policy is incomplete");
  if (use_backward_euler) {
    Result<std::vector<double>> full_step = SolveBackwardEulerStep(
        system, state, integration_rhs, step, nonlinear_maximum_iterations,
        "backward-Euler transient solve failed", factorization_cache);
    if (!full_step.ok()) {
      return Result<IntegratedStepAttempt>::Fail(full_step.error().code,
                                                 full_step.error().message);
    }

    if (system.c.values.empty()) {
      accepted_state = full_step.TakeValue();
    } else {
      const double midpoint = time + step * 0.5;
      if (!std::isfinite(midpoint) || midpoint <= time ||
          midpoint >= next_time) {
        if (lands_on_hard_point &&
            next_time ==
                std::nextafter(time, std::numeric_limits<double>::infinity())) {
          accepted_state = full_step.TakeValue();
        } else {
          return Result<IntegratedStepAttempt>::Fail(
              ErrorCode::kSolve,
              "backward-Euler error-estimation midpoint is not representable");
        }
      } else {
        Result<std::vector<double>> midpoint_rhs =
            BuildTransientRhs(system, midpoint);
        if (!midpoint_rhs.ok()) {
          return Result<IntegratedStepAttempt>::Fail(
              ErrorCode::kSolve, midpoint_rhs.error().message);
        }
        const double first_half = midpoint - time;
        const double second_half = next_time - midpoint;
        Result<std::vector<double>> first_half_state = SolveBackwardEulerStep(
            system, state, midpoint_rhs.value(), first_half,
            nonlinear_maximum_iterations,
            "first half backward-Euler LTE solve failed", factorization_cache);
        if (!first_half_state.ok()) {
          return Result<IntegratedStepAttempt>::Fail(
              first_half_state.error().code, first_half_state.error().message);
        }
        Result<std::vector<double>> refined = SolveBackwardEulerStep(
            system, first_half_state.value(), integration_rhs, second_half,
            nonlinear_maximum_iterations,
            "second half backward-Euler LTE solve failed", factorization_cache);
        if (!refined.ok()) {
          return Result<IntegratedStepAttempt>::Fail(refined.error().code,
                                                     refined.error().message);
        }
        Result<double> error = ComputeNormalizedLocalError(
            system, refined.value(), full_step.value(), 1.0);
        if (!error.ok()) {
          return Result<IntegratedStepAttempt>::Fail(ErrorCode::kSolve,
                                                     error.error().message);
        }
        normalized_error = error.value();
        used_step_doubling = true;
        if (derivative_history) {
          auto derivative = ReconstructHistoryDerivative(
              system, first_half_state.value(), refined.value(), midpoint,
              next_time);
          if (!derivative.ok())
            return Result<IntegratedStepAttempt>::Fail(
                derivative.error().code, derivative.error().message);
          proposed_derivatives = derivative.TakeValue();
        }
        accepted_state = refined.TakeValue();
      }
    }
  } else {
    // Full TRAP and its first half start from exactly the same immutable
    // accepted state. This cache belongs only to this attempt and is never
    // passed to the second half, which has its own provisional state.
    std::optional<std::vector<double>> accepted_history;
    std::optional<internal::PreparedTransientCompanion::StateProducts>
        accepted_products;
    auto *same_state_history =
        !system.behavioral_descriptors.empty() &&
                factorization_cache->prepares_behavioral()
            ? &accepted_history
            : nullptr;
    auto *same_state_products =
        same_state_history ? &accepted_products : nullptr;
    auto trap_solution =
        SolveTrapezoidalStep(system, state, previous_rhs, current_rhs, step,
                             nonlinear_maximum_iterations, factorization_cache,
                             same_state_history, same_state_products);
    if (!trap_solution.ok())
      return Result<IntegratedStepAttempt>::Fail(trap_solution.error().code,
                                                 trap_solution.error().message);
    if (!system.behavioral_descriptors.empty()) {
      BehavioralAttemptCoordinates coordinates{&state, &trap_solution.value(),
                                               std::nullopt, std::nullopt};
      auto *cached_coordinates =
          factorization_cache->prepares_behavioral() ? &coordinates : nullptr;
      if (derivative_history && derivative_history->current) {
        if (derivative_history->current->time != time)
          return Result<IntegratedStepAttempt>::Fail(
              ErrorCode::kInvalidStructure,
              "derivative history does not end at the accepted state");
        auto derivative = ReconstructHistoryDerivative(
            system, state, trap_solution.value(), time, next_time,
            &derivative_history->current->derivatives, cached_coordinates);
        if (!derivative.ok())
          return Result<IntegratedStepAttempt>::Fail(
              derivative.error().code, derivative.error().message);
        proposed_derivatives = derivative.TakeValue();
      }
      std::optional<double> history_error;
      bool needs_step_doubling = true;
      if (derivative_history && derivative_history->older) {
        if (!proposed_derivatives)
          return Result<IntegratedStepAttempt>::Fail(
              ErrorCode::kInvalidStructure,
              "derivative-history estimator has no current derivative");
        auto error = EstimateDerivativeHistoryError(
            system, trap_solution.value(), *proposed_derivatives, time,
            next_time, *derivative_history, cached_coordinates);
        if (!error.ok())
          return Result<IntegratedStepAttempt>::Fail(error.error().code,
                                                     error.error().message);
        history_error = error.value();
        auto positive_feedback = HasPositiveDerivativeHistoryFeedback(
            system, state, trap_solution.value(), *proposed_derivatives,
            derivative_history->current->derivatives, time, next_time,
            cached_coordinates);
        if (!positive_feedback.ok())
          return Result<IntegratedStepAttempt>::Fail(
              positive_feedback.error().code,
              positive_feedback.error().message);
        needs_step_doubling = derivative_control->fallback_active ||
                              derivative_control->first_audit_pending ||
                              derivative_control->accepted_trap_since_audit >=
                                  kHistoryAuditAcceptedTrapInterval - 1 ||
                              *history_error == 0.0 ||
                              positive_feedback.value();
        if (!needs_step_doubling) {
          normalized_error = *history_error;
          used_derivative_history = true;
        }
      }
      if (needs_step_doubling) {
        const double midpoint = time + step * 0.5;
        if (!std::isfinite(midpoint) || midpoint <= time ||
            midpoint >= next_time) {
          return Result<IntegratedStepAttempt>::Fail(
              ErrorCode::kSolve,
              "trapezoidal LTE midpoint is not representable");
        }
        auto midpoint_rhs = BuildTransientRhs(system, midpoint);
        if (!midpoint_rhs.ok())
          return Result<IntegratedStepAttempt>::Fail(
              midpoint_rhs.error().code, midpoint_rhs.error().message);
        auto first = SolveTrapezoidalStep(
            system, state, previous_rhs, midpoint_rhs.value(), midpoint - time,
            nonlinear_maximum_iterations, factorization_cache,
            same_state_history, same_state_products);
        if (!first.ok())
          return Result<IntegratedStepAttempt>::Fail(first.error().code,
                                                     first.error().message);
        auto second = SolveTrapezoidalStep(
            system, first.value(), midpoint_rhs.value(), current_rhs,
            next_time - midpoint, nonlinear_maximum_iterations,
            factorization_cache);
        if (!second.ok())
          return Result<IntegratedStepAttempt>::Fail(second.error().code,
                                                     second.error().message);
        auto error = ComputeNormalizedLocalError(system, trap_solution.value(),
                                                 second.value(), 4.0 / 3.0);
        if (!error.ok())
          return Result<IntegratedStepAttempt>::Fail(error.error().code,
                                                     error.error().message);
        normalized_error = error.value();
        used_step_doubling = true;
        checked_derivative_history = derivative_history != nullptr;
        if (history_error) {
          audited_derivative_history = true;
          disable_derivative_history =
              normalized_error > kHistoryAuditMinimumError &&
              normalized_error >
                  kHistoryAuditUnderestimateRatio * *history_error;
          derivative_history_audit_agreed = !disable_derivative_history;
        }
      }
    } else {

      Result<CsrMatrix> be_matrix =
          FormTransientCompanionMatrix(system.g, system.c, step, 1.0);
      if (!be_matrix.ok()) {
        return Result<IntegratedStepAttempt>::Fail(ErrorCode::kSolve,
                                                   be_matrix.error().message);
      }
      Result<std::vector<double>> be_rhs =
          BuildBackwardEulerRhs(system.c, state, integration_rhs, step);
      if (!be_rhs.ok()) {
        return Result<IntegratedStepAttempt>::Fail(ErrorCode::kSolve,
                                                   be_rhs.error().message);
      }
      Result<std::vector<double>> be_solution = SolveTransientSystem(
          system, be_matrix.value(), be_rhs.value(), state,
          nonlinear_maximum_iterations, "LTE backward-Euler solve failed",
          factorization_cache);
      if (!be_solution.ok()) {
        return Result<IntegratedStepAttempt>::Fail(be_solution.error().code,
                                                   be_solution.error().message);
      }
      Result<double> error = ComputeNormalizedLocalError(
          system, trap_solution.value(), be_solution.value(), 2.0 / 3.0);
      if (!error.ok()) {
        return Result<IntegratedStepAttempt>::Fail(ErrorCode::kSolve,
                                                   error.error().message);
      }
      normalized_error = error.value();
    }
    accepted_state = trap_solution.TakeValue();
  }

  const double factor =
      !system.behavioral_descriptors.empty() && !use_backward_euler
          ? (normalized_error == 0.0
                 ? 2.0
                 : std::clamp(0.9 * std::cbrt(1.0 / normalized_error), 0.5,
                              2.0))
          : AdaptationFactor(normalized_error);
  const double adapted_step = step * factor;
  if (!std::isfinite(adapted_step) || adapted_step <= 0.0) {
    return Result<IntegratedStepAttempt>::Fail(
        ErrorCode::kSolve,
        "adaptive transient timestep produced a non-finite value");
  }
  return Result<IntegratedStepAttempt>::Ok(IntegratedStepAttempt{
      .accepted_state = std::move(accepted_state),
      .normalized_error = normalized_error,
      .adapted_step = adapted_step,
      .proposed_derivatives = std::move(proposed_derivatives),
      .used_step_doubling = used_step_doubling,
      .used_derivative_history = used_derivative_history,
      .checked_derivative_history = checked_derivative_history,
      .audited_derivative_history = audited_derivative_history,
      .disable_derivative_history = disable_derivative_history,
      .derivative_history_audit_agreed = derivative_history_audit_agreed,
  });
}

} // namespace

namespace {

[[nodiscard]] Result<std::vector<double>> BuildTransientInitialStateImpl(
    const MnaSystem &system, bool use_initial_conditions,
    SparseRealFactorizationCache *factorization_cache) {
  if (!system.bjt_descriptors.empty()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kUnsupported,
        "phase 3C does not support BJT transient analysis or charge storage");
  }
  if (system.g.rows != system.g.columns ||
      system.node_names.size() + system.branch_names.size() != system.g.rows ||
      system.b_dc.size() != system.g.rows) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kInvalidStructure,
        "invalid MNA dimensions for transient initialization");
  }
  if (!use_initial_conditions) {
    if ((system.diode_descriptors.empty() &&
         system.behavioral_descriptors.empty())) {
      Result<std::vector<double>> solved =
          factorization_cache->Solve(system.g, system.b_dc);
      if (!solved.ok()) {
        return Result<std::vector<double>>::Fail(
            solved.error().code,
            "DC transient initialization failed: " + solved.error().message);
      }
      return solved;
    }
    NonlinearDcOptions options;
    if (!system.behavioral_descriptors.empty()) {
      options.direct_maximum_iterations =
          BehavioralNumericalPolicy::dc_maximum_iterations;
      options.source_step_maximum_iterations =
          BehavioralNumericalPolicy::dc_maximum_iterations;
      options.gmin_step_maximum_iterations =
          BehavioralNumericalPolicy::dc_maximum_iterations;
      options.final_gmin_maximum_iterations =
          BehavioralNumericalPolicy::dc_maximum_iterations;
    }
    Result<NonlinearDcResult> solved = RunNonlinearDc(system, options);
    if (!solved.ok()) {
      return Result<std::vector<double>>::Fail(
          solved.error().code,
          "nonlinear DC transient initialization failed: " +
              solved.error().message);
    }
    return Result<std::vector<double>>::Ok(solved.TakeValue().solution);
  }

  if (!system.behavioral_descriptors.empty())
    return Result<std::vector<double>>::Fail(
        ErrorCode::kUnsupported,
        "behavioral UIC is outside the qualified initialization contract");
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
  SparseRealFactorizationCache factorization_cache;
  return BuildTransientInitialStateImpl(system, use_initial_conditions,
                                        &factorization_cache);
}

namespace {

Result<TransientResult> RunTransientAnalysisImpl(
    const MnaSystem &input_system, const TranAnalysis &analysis,
    const TransientExecutionLimits &limits, bool prepare_behavioral) {
  try {
    // A callback can retain a mutable alias to the caller's circuit. All
    // behavioral validation, source evaluation, history and integration use
    // this invocation's owned snapshot rather than that external object.
    std::optional<MnaSystem> behavioral_snapshot;
    if (!input_system.behavioral_descriptors.empty())
      behavioral_snapshot = input_system;
    const MnaSystem &system =
        behavioral_snapshot ? *behavioral_snapshot : input_system;
    switch (limits.behavioral_error_estimator) {
    case BehavioralErrorEstimator::kStepDoubling:
      break;
    case BehavioralErrorEstimator::kDerivativeHistory:
      if (system.behavioral_descriptors.empty())
        return Result<TransientResult>::Fail(
            ErrorCode::kUnsupported,
            "derivative-history LTE requires behavioral sources");
      break;
    default:
      return Result<TransientResult>::Fail(
          ErrorCode::kInvalidStructure, "unknown behavioral error estimator");
    }
    if (!system.bjt_descriptors.empty()) {
      return Result<TransientResult>::Fail(
          ErrorCode::kUnsupported,
          "phase 3C does not support BJT transient analysis or charge storage");
    }
    if (!system.behavioral_descriptors.empty()) {
      auto valid = ValidateBehavioralTransient(system);
      if (!valid.ok())
        return Result<TransientResult>::Fail(valid.error().code,
                                             valid.error().message);
      if (analysis.use_initial_conditions)
        return Result<TransientResult>::Fail(ErrorCode::kUnsupported,
                                             "behavioral UIC is unsupported");
    }
    SparseRealFactorizationCache factorization_cache(prepare_behavioral);
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
    if (limits.maximum_accepted_steps == 0 ||
        limits.maximum_step_attempts == 0 ||
        limits.maximum_step_attempts < limits.maximum_accepted_steps ||
        !std::isfinite(limits.minimum_step_divisor) ||
        limits.minimum_step_divisor < 1.0 ||
        limits.nonlinear_maximum_iterations >
            (system.behavioral_descriptors.empty()
                 ? kDirectNewtonMaximumIterations
                 : BehavioralNumericalPolicy::transient_maximum_iterations)) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve, "invalid transient execution limits");
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
    Result<std::vector<double>> projected_initial =
        system.behavioral_descriptors.empty()
            ? ProjectReactiveState(system, initial.value(), initial_rhs.value(),
                                   "initial transient source projection",
                                   &factorization_cache)
            : Result<std::vector<double>>::Ok(initial.TakeValue());
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
    bool nonlinear_retry_pending = false;
    BehavioralDerivativeHistory derivative_history;
    BehavioralDerivativeControl derivative_control;
    const bool use_derivative_history =
        limits.behavioral_error_estimator ==
        BehavioralErrorEstimator::kDerivativeHistory;
    std::size_t hard_index = 0;
    std::size_t attempts = 0;
    std::size_t accepted_steps = 0;
    double first_output_time = -1.0;
    double last_output_time = -1.0;
    const auto emit =
        [&](double output_time,
            const std::vector<double> &output_state) -> Result<bool> {
      if (limits.accepted_state_observer) {
        auto emitted =
            limits.accepted_state_observer(output_time, output_state);
        if (!emitted.ok())
          return emitted;
        if (!emitted.value())
          return Result<bool>::Fail(ErrorCode::kIo,
                                    "accepted-state observer rejected output");
      }
      if (limits.retain_output_states) {
        result.times_seconds.push_back(output_time);
        result.states.push_back(output_state);
      }
      if (result.emitted_points == 0)
        first_output_time = output_time;
      last_output_time = output_time;
      ++result.emitted_points;
      return Result<bool>::Ok(true);
    };
    if (analysis.start_time_seconds == 0.0) {
      auto emitted = emit(0.0, state);
      if (!emitted.ok())
        return Result<TransientResult>::Fail(emitted.error().code,
                                             emitted.error().message);
    }

    while (time < analysis.stop_time_seconds) {
      if (attempts >= limits.maximum_step_attempts) {
        return Result<TransientResult>::Fail(
            nonlinear_retry_pending ? ErrorCode::kNonConvergence
                                    : ErrorCode::kSolve,
            "transient analysis exceeded " +
                std::to_string(limits.maximum_step_attempts) +
                " timestep attempts");
      }
      ++attempts;
      while (hard_index < hard_points.size() &&
             hard_points[hard_index] <= time) {
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
            ErrorCode::kSolve,
            "transient hard point is not strictly increasing");
      }
      double step = std::min(proposed_step, maximum_step);
      bool lands_on_hard_point = false;
      double next_time = time + step;
      if (step >= hard_distance || next_time >= next_hard_point) {
        step = hard_distance;
        next_time = next_hard_point;
        lands_on_hard_point = true;
      } else if (!system.behavioral_descriptors.empty() &&
                 next_hard_point - next_time < minimum_step) {
        // Avoid a rounding-sized final interval with an enormous C/h. Split
        // the remaining distance without exceeding the requested maximum step.
        next_time = time + hard_distance * 0.5;
        step = next_time - time;
      }
      if (use_derivative_history) {
        // The history recurrence measures accepted timestamps. Use that same
        // duration in the companion, rather than the pre-addition proposal.
        // Rounding time+h upward must not exceed the caller's maximum step.
        step = next_time - time;
        if (step > maximum_step) {
          next_time = std::nextafter(next_time, time);
          step = next_time - time;
          lands_on_hard_point = next_time == next_hard_point;
        }
      }
      if (!std::isfinite(step) || step <= 0.0 || !std::isfinite(next_time) ||
          next_time <= time ||
          (use_derivative_history && step > maximum_step)) {
        return Result<TransientResult>::Fail(
            nonlinear_retry_pending ? ErrorCode::kNonConvergence
                                    : ErrorCode::kSolve,
            "transient timestep is not strictly increasing and representable");
      }
      if (step < minimum_step && !lands_on_hard_point) {
        return Result<TransientResult>::Fail(
            nonlinear_retry_pending ? ErrorCode::kNonConvergence
                                    : ErrorCode::kSolve,
            "adaptive transient timestep fell below h_min at t_ns=" +
                std::to_string(time * 1e9) +
                " step_fs=" + std::to_string(step * 1e15));
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

      Result<IntegratedStepAttempt> integrated = IntegrateTransientStepAttempt(
          system, state, previous_rhs, current_rhs.value(), integration_rhs,
          time, next_time, step, use_backward_euler, lands_on_hard_point,
          limits.nonlinear_maximum_iterations, &factorization_cache,
          use_derivative_history ? &derivative_history : nullptr,
          use_derivative_history ? &derivative_control : nullptr);
      if (!integrated.ok()) {
        if ((system.diode_descriptors.empty() &&
             system.behavioral_descriptors.empty()) ||
            integrated.error().code != ErrorCode::kNonConvergence) {
          return Result<TransientResult>::Fail(integrated.error().code,
                                               integrated.error().message);
        }
        result.step_trace.push_back(TransientStepRecord{
            .start_time_seconds = time,
            .end_time_seconds = next_time,
            .step_size_seconds = step,
            .method = method,
            .accepted = false,
            .normalized_local_error = 0.0,
            .landed_on_hard_point = lands_on_hard_point,
            .derivative_history_fallback_active =
                derivative_control.fallback_active,
            .rejection_reason =
                TransientStepRejectionReason::kNonlinearConvergence,
        });
        const double retry_step = step * 0.5;
        if (!std::isfinite(retry_step) || retry_step <= 0.0 ||
            retry_step < minimum_step) {
          return Result<TransientResult>::Fail(
              ErrorCode::kNonConvergence,
              "nonlinear transient timestep retry fell below h_min or became "
              "unrepresentable at t_ns=" +
                  std::to_string(time * 1e9) +
                  " step_fs=" + std::to_string(step * 1e15) + ": " +
                  integrated.error().message);
        }
        proposed_step = retry_step;
        recovery_step = true;
        nonlinear_retry_pending = true;
        derivative_control.consecutive_accepted_agreements = 0;
        continue;
      }

      const double normalized_error = integrated.value().normalized_error;
      const double adapted_step = integrated.value().adapted_step;
      result.step_doubling_error_estimates +=
          integrated.value().used_step_doubling;
      result.derivative_history_error_estimates +=
          integrated.value().used_derivative_history;
      result.derivative_history_step_doubling_checks +=
          integrated.value().checked_derivative_history;
      // An audited underestimate changes only the retry policy, never accepted
      // derivatives or timestamps. A rejected audit retains its audit debt.
      if (integrated.value().disable_derivative_history &&
          !derivative_control.fallback_active) {
        derivative_control.fallback_active = true;
        ++result.derivative_history_fallback_entries;
      }
      if (normalized_error > 1.0) {
        derivative_control.consecutive_accepted_agreements = 0;
        result.step_trace.push_back(TransientStepRecord{
            .start_time_seconds = time,
            .end_time_seconds = next_time,
            .step_size_seconds = step,
            .method = method,
            .accepted = false,
            .normalized_local_error = normalized_error,
            .landed_on_hard_point = lands_on_hard_point,
            .derivative_history_audited =
                integrated.value().audited_derivative_history,
            .derivative_history_audit_agreed =
                integrated.value().derivative_history_audit_agreed,
            .derivative_history_fallback_active =
                derivative_control.fallback_active,
            .rejection_reason = TransientStepRejectionReason::kLocalError,
        });
        if (adapted_step < minimum_step) {
          return Result<TransientResult>::Fail(
              ErrorCode::kSolve,
              "adaptive transient timestep fell below h_min at t_ns=" +
                  std::to_string(time * 1e9) +
                  " step_fs=" + std::to_string(step * 1e15));
        }
        proposed_step = adapted_step;
        recovery_step =
            system.behavioral_descriptors.empty() || use_backward_euler;
        nonlinear_retry_pending = false;
        continue;
      }
      auto accepted_attempt = integrated.TakeValue();
      std::vector<double> accepted_state =
          std::move(accepted_attempt.accepted_state);
      proposed_step = std::clamp(adapted_step, minimum_step, maximum_step);
      recovery_step =
          !system.behavioral_descriptors.empty() && waveform_landing;
      nonlinear_retry_pending = false;

      if (system.behavioral_descriptors.empty() && waveform_landing &&
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
            ErrorCode::kSolve,
            "transient analysis exceeded " +
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
          .derivative_history_audited =
              accepted_attempt.audited_derivative_history,
          .derivative_history_audit_agreed =
              accepted_attempt.derivative_history_audit_agreed,
      });
      if (use_derivative_history) {
        // This is the only derivative-history mutation. Rejected solves/LTE,
        // half-step candidates and exhausted accepted-step budgets never reach
        // this transaction. BE and waveform restarts begin a fresh segment.
        if (use_backward_euler || waveform_landing ||
            !accepted_attempt.proposed_derivatives) {
          derivative_history.older.reset();
        } else {
          derivative_history.older = std::move(derivative_history.current);
        }
        if (accepted_attempt.proposed_derivatives)
          derivative_history.current = BehavioralDerivativeSample{
              .time = next_time,
              .derivatives = std::move(*accepted_attempt.proposed_derivatives)};
        else
          derivative_history.current.reset();
        if (use_backward_euler || waveform_landing) {
          derivative_control = {};
        } else if (accepted_attempt.audited_derivative_history) {
          derivative_control.first_audit_pending = false;
          derivative_control.accepted_trap_since_audit = 0;
          if (derivative_control.fallback_active &&
              accepted_attempt.derivative_history_audit_agreed) {
            ++derivative_control.consecutive_accepted_agreements;
            if (derivative_control.consecutive_accepted_agreements ==
                kHistoryRecoveryAcceptedAudits) {
              derivative_control.fallback_active = false;
              derivative_control.consecutive_accepted_agreements = 0;
              ++result.derivative_history_fallback_recoveries;
            }
          } else {
            derivative_control.consecutive_accepted_agreements = 0;
          }
        } else if (!derivative_control.first_audit_pending) {
          ++derivative_control.accepted_trap_since_audit;
          derivative_control.consecutive_accepted_agreements = 0;
        }
        result.step_trace.back().derivative_history_fallback_active =
            derivative_control.fallback_active;
      }
      state = std::move(accepted_state);
      previous_rhs = current_rhs.TakeValue();
      time = next_time;

      if (time >= analysis.start_time_seconds) {
        auto emitted = emit(time, state);
        if (!emitted.ok())
          return Result<TransientResult>::Fail(emitted.error().code,
                                               emitted.error().message);
      }
    }

    if (time != analysis.stop_time_seconds || result.emitted_points == 0 ||
        last_output_time != analysis.stop_time_seconds) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve,
          "transient analysis did not include the exact requested stop time");
    }
    if (first_output_time != analysis.start_time_seconds) {
      return Result<TransientResult>::Fail(
          ErrorCode::kSolve,
          "transient analysis did not include the exact output start time");
    }
    result.solver_statistics = factorization_cache.statistics();
    return Result<TransientResult>::Ok(std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<TransientResult>::Fail(
        ErrorCode::kFactorization, "transient analysis allocation failed");
  }
}

} // namespace

Result<TransientResult>
RunTransientAnalysis(const MnaSystem &system, const TranAnalysis &analysis,
                     const TransientExecutionLimits &limits) {
  return RunTransientAnalysisImpl(system, analysis, limits, true);
}

Result<TransientResult> internal::RunTransientAnalysisUnpreparedForTest(
    const MnaSystem &system, const TranAnalysis &analysis,
    const TransientExecutionLimits &limits) {
  return RunTransientAnalysisImpl(system, analysis, limits, false);
}

} // namespace ohmnivore
