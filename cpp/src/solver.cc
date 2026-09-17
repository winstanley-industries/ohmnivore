#include "ohmnivore/solver.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "klu.h"

namespace ohmnivore {
namespace {

template <typename T> [[nodiscard]] bool IsFinite(T value) {
  if constexpr (std::is_same_v<T, double>) {
    return std::isfinite(value);
  } else {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
  }
}

[[nodiscard]] std::string FormatDouble(double value) {
  std::array<char, 64> buffer{};
  const auto converted = std::to_chars(
      buffer.data(), buffer.data() + buffer.size(), value,
      std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (converted.ec != std::errc{}) {
    return "unrepresentable";
  }
  return std::string(buffer.data(), converted.ptr);
}

template <typename T>
[[nodiscard]] Result<SolverCscPattern>
ConvertCsrToSolverCscImpl(const CsrMatrixBase<T> &matrix) {
  constexpr std::size_t kMaximumKluIndex =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (matrix.rows > kMaximumKluIndex || matrix.columns > kMaximumKluIndex ||
      matrix.values.size() > kMaximumKluIndex) {
    return Result<SolverCscPattern>::Fail(
        ErrorCode::kUnsupportedSize,
        "sparse solver requires dimensions and nonzero count representable "
        "as signed 32-bit indexes");
  }
  if (matrix.rows != matrix.columns) {
    return Result<SolverCscPattern>::Fail(
        ErrorCode::kInvalidStructure, "sparse solver matrix must be square");
  }
  if (matrix.row_offsets.size() != matrix.rows + 1 ||
      matrix.column_indices.size() != matrix.values.size()) {
    return Result<SolverCscPattern>::Fail(ErrorCode::kInvalidStructure,
                                          "invalid CSR structure");
  }
  if (matrix.row_offsets.empty() || matrix.row_offsets.front() != 0 ||
      matrix.row_offsets.back() != matrix.values.size()) {
    return Result<SolverCscPattern>::Fail(
        ErrorCode::kInvalidStructure,
        "CSR row offsets must start at zero and end at the value count");
  }

  std::vector<std::size_t> counts(matrix.columns, 0);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
        matrix.row_offsets[row + 1] > matrix.values.size()) {
      return Result<SolverCscPattern>::Fail(ErrorCode::kInvalidStructure,
                                            "invalid CSR row offsets");
    }
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      const std::size_t column = matrix.column_indices[index];
      if (column >= matrix.columns) {
        return Result<SolverCscPattern>::Fail(
            ErrorCode::kInvalidStructure,
            "CSR column index is outside the matrix");
      }
      if (index > matrix.row_offsets[row] &&
          matrix.column_indices[index - 1] >= column) {
        return Result<SolverCscPattern>::Fail(
            ErrorCode::kInvalidStructure,
            "CSR columns must be strictly increasing within each row");
      }
      if (!IsFinite(matrix.values[index])) {
        return Result<SolverCscPattern>::Fail(
            ErrorCode::kNonFinite, "matrix contains a non-finite value");
      }
      ++counts[column];
    }
  }

  SolverCscPattern converted;
  converted.size = matrix.rows;
  converted.column_offsets.assign(matrix.columns + 1, 0);
  for (std::size_t column = 0; column < matrix.columns; ++column) {
    const std::size_t next =
        static_cast<std::size_t>(converted.column_offsets[column]) +
        counts[column];
    if (next > kMaximumKluIndex) {
      return Result<SolverCscPattern>::Fail(
          ErrorCode::kUnsupportedSize,
          "CSC column offset is not representable as a signed 32-bit index");
    }
    converted.column_offsets[column + 1] = static_cast<std::int32_t>(next);
  }
  converted.row_indices.resize(matrix.values.size());
  converted.csr_value_indices.resize(matrix.values.size());
  std::vector<std::int32_t> next = converted.column_offsets;
  next.pop_back();
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      const std::size_t column = matrix.column_indices[index];
      const std::size_t destination = static_cast<std::size_t>(next[column]++);
      converted.row_indices[destination] = static_cast<std::int32_t>(row);
      converted.csr_value_indices[destination] = index;
    }
  }
  return Result<SolverCscPattern>::Ok(std::move(converted));
}

[[nodiscard]] bool SamePattern(const SolverCscPattern &first,
                               const SolverCscPattern &second) {
  return first.size == second.size &&
         first.column_offsets == second.column_offsets &&
         first.row_indices == second.row_indices &&
         first.csr_value_indices == second.csr_value_indices;
}

template <typename T>
[[nodiscard]] bool HasEmptyStructuralLine(const CsrMatrixBase<T> &matrix,
                                          const SolverCscPattern &pattern) {
  for (std::size_t index = 0; index < matrix.rows; ++index) {
    if (matrix.row_offsets[index] == matrix.row_offsets[index + 1] ||
        pattern.column_offsets[index] == pattern.column_offsets[index + 1]) {
      return true;
    }
  }
  return false;
}

template <typename T>
[[nodiscard]] Result<double>
ValidateSparseSolutionImpl(const CsrMatrixBase<T> &matrix,
                           const std::vector<T> &rhs,
                           const std::vector<T> &solution) {
  Result<SolverCscPattern> structure = ConvertCsrToSolverCscImpl(matrix);
  if (!structure.ok()) {
    return Result<double>::Fail(structure.error().code,
                                structure.error().message);
  }
  if (rhs.size() != matrix.rows || solution.size() != matrix.columns) {
    return Result<double>::Fail(ErrorCode::kSolutionValidation,
                                "solution validation dimensions disagree");
  }
  for (const T value : rhs) {
    if (!IsFinite(value)) {
      return Result<double>::Fail(
          ErrorCode::kNonFinite,
          "solution validation right-hand side is non-finite");
    }
  }
  for (const T value : solution) {
    if (!IsFinite(value)) {
      return Result<double>::Fail(ErrorCode::kNonFinite,
                                  "sparse solve produced a non-finite value");
    }
  }

  long double solution_norm = 0.0L;
  for (const T value : solution) {
    solution_norm =
        std::max(solution_norm, static_cast<long double>(std::abs(value)));
  }
  long double scaled_residual_norm = 0.0L;
  long double scaled_matrix_norm = 0.0L;
  long double scaled_rhs_norm = 0.0L;
  long double maximum_componentwise_error = 0.0L;
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if constexpr (std::is_same_v<T, double>) {
      long double product = 0.0L;
      long double denominator = std::abs(static_cast<long double>(rhs[row]));
      long double row_sum = 0.0L;
      long double row_scale = denominator;
      for (std::size_t index = matrix.row_offsets[row];
           index < matrix.row_offsets[row + 1]; ++index) {
        const long double coefficient = matrix.values[index];
        const long double variable = solution[matrix.column_indices[index]];
        product += coefficient * variable;
        denominator += std::abs(coefficient) * std::abs(variable);
        row_sum += std::abs(coefficient);
        row_scale = std::max(row_scale, std::abs(coefficient));
      }
      const long double residual =
          std::abs(product - static_cast<long double>(rhs[row]));
      const long double row_error =
          denominator == 0.0L
              ? (residual == 0.0L
                     ? 0.0L
                     : std::numeric_limits<long double>::infinity())
              : residual / denominator;
      maximum_componentwise_error =
          std::max(maximum_componentwise_error, row_error);
      const long double inverse_scale =
          row_scale == 0.0L ? 1.0L : 1.0L / row_scale;
      scaled_residual_norm =
          std::max(scaled_residual_norm, residual * inverse_scale);
      scaled_matrix_norm =
          std::max(scaled_matrix_norm, row_sum * inverse_scale);
      scaled_rhs_norm = std::max(scaled_rhs_norm,
                                 std::abs(static_cast<long double>(rhs[row])) *
                                     inverse_scale);
    } else {
      std::complex<long double> product{0.0L, 0.0L};
      const std::complex<long double> right_hand_side{rhs[row].real(),
                                                      rhs[row].imag()};
      long double denominator = std::abs(right_hand_side);
      long double row_sum = 0.0L;
      long double row_scale = denominator;
      for (std::size_t index = matrix.row_offsets[row];
           index < matrix.row_offsets[row + 1]; ++index) {
        const std::complex<long double> coefficient{
            matrix.values[index].real(), matrix.values[index].imag()};
        const T original_variable = solution[matrix.column_indices[index]];
        const std::complex<long double> variable{original_variable.real(),
                                                 original_variable.imag()};
        product += coefficient * variable;
        denominator += std::abs(coefficient) * std::abs(variable);
        row_sum += std::abs(coefficient);
        row_scale = std::max(row_scale, std::abs(coefficient));
      }
      const long double residual = std::abs(product - right_hand_side);
      const long double row_error =
          denominator == 0.0L
              ? (residual == 0.0L
                     ? 0.0L
                     : std::numeric_limits<long double>::infinity())
              : residual / denominator;
      maximum_componentwise_error =
          std::max(maximum_componentwise_error, row_error);
      const long double inverse_scale =
          row_scale == 0.0L ? 1.0L : 1.0L / row_scale;
      scaled_residual_norm =
          std::max(scaled_residual_norm, residual * inverse_scale);
      scaled_matrix_norm =
          std::max(scaled_matrix_norm, row_sum * inverse_scale);
      scaled_rhs_norm =
          std::max(scaled_rhs_norm, std::abs(right_hand_side) * inverse_scale);
    }
  }
  const long double normwise_denominator =
      scaled_matrix_norm * solution_norm + scaled_rhs_norm;
  const long double normwise_backward_error =
      normwise_denominator == 0.0L
          ? (scaled_residual_norm == 0.0L
                 ? 0.0L
                 : std::numeric_limits<long double>::infinity())
          : scaled_residual_norm / normwise_denominator;
  const double reported_normwise = static_cast<double>(normwise_backward_error);
  const double reported_componentwise =
      static_cast<double>(maximum_componentwise_error);
  if (!std::isfinite(reported_normwise) ||
      reported_normwise > kSparseBackwardErrorTolerance ||
      !std::isfinite(reported_componentwise) ||
      reported_componentwise > kSparseComponentwiseBackwardErrorTolerance) {
    return Result<double>::Fail(
        ErrorCode::kSolutionValidation,
        "sparse solution failed backward-error validation: normwise_error=" +
            FormatDouble(reported_normwise) + " normwise_tolerance=" +
            FormatDouble(kSparseBackwardErrorTolerance) +
            " componentwise_error=" + FormatDouble(reported_componentwise) +
            " componentwise_tolerance=" +
            FormatDouble(kSparseComponentwiseBackwardErrorTolerance));
  }
  return Result<double>::Ok(reported_componentwise);
}

void ConfigureKlu(klu_common *common) {
  static_cast<void>(klu_defaults(common));
  common->btf = 1;
  common->ordering = 0;
  common->scale = 2;
  common->tol = 1.0;
  common->halt_if_singular = 1;
}

[[nodiscard]] ErrorCode KluErrorCode(const klu_common &common,
                                     std::size_t size) {
  if (common.status == KLU_TOO_LARGE) {
    return ErrorCode::kUnsupportedSize;
  }
  if (common.status == KLU_SINGULAR ||
      (common.structural_rank >= 0 &&
       static_cast<std::size_t>(common.structural_rank) < size) ||
      (common.numerical_rank >= 0 &&
       static_cast<std::size_t>(common.numerical_rank) < size)) {
    return ErrorCode::kSingular;
  }
  return ErrorCode::kFactorization;
}

[[nodiscard]] std::string KluFailureMessage(const std::string &operation,
                                            const klu_common &common) {
  return operation + " failed (KLU status=" + std::to_string(common.status) +
         ", structural_rank=" + std::to_string(common.structural_rank) +
         ", numerical_rank=" + std::to_string(common.numerical_rank) +
         ", singular_column=" + std::to_string(common.singular_col) + ")";
}

template <typename T>
[[nodiscard]] std::vector<T> GatherCscValues(const CsrMatrixBase<T> &matrix,
                                             const SolverCscPattern &pattern) {
  std::vector<T> values;
  values.reserve(pattern.csr_value_indices.size());
  for (const std::size_t index : pattern.csr_value_indices) {
    values.push_back(matrix.values[index]);
  }
  return values;
}

[[nodiscard]] std::vector<double>
InterleaveComplex(const std::vector<std::complex<double>> &values) {
  std::vector<double> interleaved;
  interleaved.reserve(values.size() * 2);
  for (const std::complex<double> value : values) {
    interleaved.push_back(value.real());
    interleaved.push_back(value.imag());
  }
  return interleaved;
}

} // namespace

Result<SolverCscPattern> ConvertCsrToSolverCsc(const CsrMatrix &matrix) {
  try {
    return ConvertCsrToSolverCscImpl(matrix);
  } catch (const std::bad_alloc &) {
    return Result<SolverCscPattern>::Fail(
        ErrorCode::kFactorization, "sparse CSC conversion allocation failed");
  }
}

Result<SolverCscPattern> ConvertCsrToSolverCsc(const ComplexCsrMatrix &matrix) {
  try {
    return ConvertCsrToSolverCscImpl(matrix);
  } catch (const std::bad_alloc &) {
    return Result<SolverCscPattern>::Fail(
        ErrorCode::kFactorization,
        "complex sparse CSC conversion allocation failed");
  }
}

Result<double> ValidateSparseSolution(const CsrMatrix &matrix,
                                      const std::vector<double> &rhs,
                                      const std::vector<double> &solution) {
  try {
    return ValidateSparseSolutionImpl(matrix, rhs, solution);
  } catch (const std::bad_alloc &) {
    return Result<double>::Fail(ErrorCode::kFactorization,
                                "sparse solution validation allocation failed");
  }
}

Result<double>
ValidateSparseSolution(const ComplexCsrMatrix &matrix,
                       const std::vector<std::complex<double>> &rhs,
                       const std::vector<std::complex<double>> &solution) {
  try {
    return ValidateSparseSolutionImpl(matrix, rhs, solution);
  } catch (const std::bad_alloc &) {
    return Result<double>::Fail(
        ErrorCode::kFactorization,
        "complex sparse solution validation allocation failed");
  }
}

class SparseRealFactorization::Impl {
public:
  explicit Impl(SolverCscPattern pattern) : pattern_(std::move(pattern)) {
    ConfigureKlu(&common_);
  }

  ~Impl() {
    if (numeric_ != nullptr) {
      static_cast<void>(klu_free_numeric(&numeric_, &common_));
    }
    if (symbolic_ != nullptr) {
      static_cast<void>(klu_free_symbolic(&symbolic_, &common_));
    }
  }

  [[nodiscard]] Result<bool> FactorFresh(std::vector<double> &values,
                                         std::string operation) {
    if (numeric_ != nullptr) {
      static_cast<void>(klu_free_numeric(&numeric_, &common_));
    }
    last_values_.clear();
    numeric_ =
        klu_factor(pattern_.column_offsets.data(), pattern_.row_indices.data(),
                   values.data(), symbolic_, &common_);
    if (numeric_ == nullptr) {
      return Result<bool>::Fail(KluErrorCode(common_, pattern_.size),
                                KluFailureMessage(operation, common_));
    }
    ++statistics_.numeric_factorizations;
    last_values_ = values;
    return Result<bool>::Ok(true);
  }

  [[nodiscard]] Result<std::vector<double>>
  SolveAndValidate(const CsrMatrix &matrix, const std::vector<double> &rhs,
                   std::size_t *remaining_refinements) {
    std::vector<double> solution = rhs;
    if (klu_solve(symbolic_, numeric_, static_cast<std::int32_t>(pattern_.size),
                  1, solution.data(), &common_) == 0) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kFactorization,
          KluFailureMessage("KLU triangular solve", common_));
    }
    Result<double> validation = ValidateSparseSolution(matrix, rhs, solution);
    bool corrected = false;
    while (*remaining_refinements > 0 &&
           ((validation.ok() && !corrected) ||
            (!validation.ok() &&
             validation.error().code == ErrorCode::kSolutionValidation))) {
      std::vector<double> correction(matrix.rows);
      for (std::size_t row = 0; row < matrix.rows; ++row) {
        long double product = 0.0L;
        for (std::size_t index = matrix.row_offsets[row];
             index < matrix.row_offsets[row + 1]; ++index) {
          product +=
              static_cast<long double>(matrix.values[index]) *
              static_cast<long double>(solution[matrix.column_indices[index]]);
        }
        correction[row] =
            static_cast<double>(static_cast<long double>(rhs[row]) - product);
        if (!std::isfinite(correction[row])) {
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite,
              "iterative refinement residual is not finite FP64");
        }
      }
      if (std::all_of(correction.begin(), correction.end(),
                      [](double value) { return value == 0.0; }))
        break;
      corrected = true;
      --*remaining_refinements;
      ++statistics_.iterative_refinement_solves;
      if (klu_solve(symbolic_, numeric_,
                    static_cast<std::int32_t>(pattern_.size), 1,
                    correction.data(), &common_) == 0) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kFactorization,
            KluFailureMessage("KLU refinement triangular solve", common_));
      }
      for (std::size_t index = 0; index < solution.size(); ++index) {
        solution[index] += correction[index];
        if (!std::isfinite(correction[index]) ||
            !std::isfinite(solution[index])) {
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite,
              "iterative refinement update is not finite FP64");
        }
      }
      validation = ValidateSparseSolution(matrix, rhs, solution);
    }
    if (!validation.ok()) {
      return Result<std::vector<double>>::Fail(validation.error().code,
                                               validation.error().message);
    }
    return Result<std::vector<double>>::Ok(std::move(solution));
  }

  [[nodiscard]] Result<std::vector<double>>
  FactorAndSolve(const CsrMatrix &matrix, const std::vector<double> &rhs,
                 std::size_t remaining_refinements = 0) {
    Result<SolverCscPattern> converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok()) {
      return Result<std::vector<double>>::Fail(converted.error().code,
                                               converted.error().message);
    }
    if (!SamePattern(pattern_, converted.value())) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          "numeric refactorization requires the analyzed CSR pattern");
    }
    if (rhs.size() != pattern_.size) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          "matrix and right-hand-side dimensions disagree");
    }
    for (const double value : rhs) {
      if (!std::isfinite(value)) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kNonFinite,
            "right-hand side contains a non-finite value");
      }
    }
    if (pattern_.size == 0) {
      ++statistics_.solves;
      return Result<std::vector<double>>::Ok({});
    }

    std::vector<double> values = GatherCscValues(matrix, pattern_);
    bool refactored_without_pivoting = false;
    if (numeric_ == nullptr) {
      Result<bool> factored = FactorFresh(values, "KLU numeric factorization");
      if (!factored.ok()) {
        return Result<std::vector<double>>::Fail(factored.error().code,
                                                 factored.error().message);
      }
    } else if (values == last_values_) {
      ++statistics_.numeric_reuses;
    } else {
      if (klu_refactor(pattern_.column_offsets.data(),
                       pattern_.row_indices.data(), values.data(), symbolic_,
                       numeric_, &common_) == 0) {
        ++statistics_.numeric_refactorization_fallbacks;
        Result<bool> factored = FactorFresh(
            values, "KLU pivoting factorization after refactorization failure");
        if (!factored.ok()) {
          return Result<std::vector<double>>::Fail(factored.error().code,
                                                   factored.error().message);
        }
      } else {
        ++statistics_.numeric_refactorizations;
        last_values_ = values;
        refactored_without_pivoting = true;
      }
    }

    Result<std::vector<double>> solved =
        SolveAndValidate(matrix, rhs, &remaining_refinements);
    if (!solved.ok() && refactored_without_pivoting) {
      // KLU refactor deliberately preserves the first numeric pivot order.
      // If changed values make that order unstable, retain the symbolic
      // analysis but deterministically rebuild the numeric factors with full
      // threshold partial pivoting and revalidate. This is a sparse KLU retry,
      // never a dispatch or fallback to the dense test oracle.
      ++statistics_.numeric_refactorization_fallbacks;
      Result<bool> factored = FactorFresh(
          values,
          "KLU pivoting factorization after refactorized solve failure");
      if (!factored.ok()) {
        return Result<std::vector<double>>::Fail(factored.error().code,
                                                 factored.error().message);
      }
      solved = SolveAndValidate(matrix, rhs, &remaining_refinements);
    }
    if (!solved.ok()) {
      return solved;
    }
    ++statistics_.solves;
    return solved;
  }

  SolverCscPattern pattern_;
  klu_common common_{};
  klu_symbolic *symbolic_ = nullptr;
  klu_numeric *numeric_ = nullptr;
  std::vector<double> last_values_;
  SparseSolverStatistics statistics_;
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
  try {
    Result<SolverCscPattern> converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok()) {
      return Result<std::unique_ptr<SparseRealFactorization>>::Fail(
          converted.error().code, converted.error().message);
    }
    auto implementation = std::make_unique<Impl>(converted.TakeValue());
    if (implementation->pattern_.size != 0) {
      if (HasEmptyStructuralLine(matrix, implementation->pattern_)) {
        return Result<std::unique_ptr<SparseRealFactorization>>::Fail(
            ErrorCode::kSingular,
            "sparse matrix is structurally singular: an empty row or column "
            "prevents a full matching");
      }
      implementation->symbolic_ =
          klu_analyze(static_cast<std::int32_t>(implementation->pattern_.size),
                      implementation->pattern_.column_offsets.data(),
                      implementation->pattern_.row_indices.data(),
                      &implementation->common_);
      if (implementation->symbolic_ == nullptr) {
        return Result<std::unique_ptr<SparseRealFactorization>>::Fail(
            KluErrorCode(implementation->common_,
                         implementation->pattern_.size),
            KluFailureMessage("KLU symbolic analysis",
                              implementation->common_));
      }
      if (implementation->symbolic_->structural_rank >= 0 &&
          static_cast<std::size_t>(implementation->symbolic_->structural_rank) <
              implementation->pattern_.size) {
        return Result<std::unique_ptr<SparseRealFactorization>>::Fail(
            ErrorCode::kSingular,
            "KLU symbolic analysis found a structurally rank-deficient "
            "matrix: rank=" +
                std::to_string(implementation->symbolic_->structural_rank) +
                " size=" + std::to_string(implementation->pattern_.size));
      }
      ++implementation->statistics_.symbolic_analyses;
    }
    return Result<std::unique_ptr<SparseRealFactorization>>::Ok(
        std::unique_ptr<SparseRealFactorization>(
            new SparseRealFactorization(std::move(implementation))));
  } catch (const std::bad_alloc &) {
    return Result<std::unique_ptr<SparseRealFactorization>>::Fail(
        ErrorCode::kFactorization,
        "sparse symbolic analysis allocation failed");
  }
}

Result<std::vector<double>>
SparseRealFactorization::FactorAndSolve(const CsrMatrix &matrix,
                                        const std::vector<double> &rhs) {
  try {
    return implementation_->FactorAndSolve(matrix, rhs);
  } catch (const std::bad_alloc &) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kFactorization,
        "sparse numeric factorization allocation failed");
  }
}

const SparseSolverStatistics &SparseRealFactorization::statistics() const {
  return implementation_->statistics_;
}

Result<std::vector<double>> SparseRealFactorization::FactorAndSolveRefined(
    const CsrMatrix &matrix, const std::vector<double> &rhs,
    std::size_t maximum_refinements) {
  if (maximum_refinements > 4) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kUnsupportedSize,
        "iterative refinement allows at most four corrections");
  }
  try {
    return implementation_->FactorAndSolve(matrix, rhs, maximum_refinements);
  } catch (const std::bad_alloc &) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kFactorization, "iterative refinement allocation failed");
  }
}

class SparseComplexFactorization::Impl {
public:
  explicit Impl(SolverCscPattern pattern) : pattern_(std::move(pattern)) {
    ConfigureKlu(&common_);
  }

  ~Impl() {
    if (numeric_ != nullptr) {
      static_cast<void>(klu_z_free_numeric(&numeric_, &common_));
    }
    if (symbolic_ != nullptr) {
      static_cast<void>(klu_free_symbolic(&symbolic_, &common_));
    }
  }

  [[nodiscard]] Result<bool>
  FactorFresh(const std::vector<std::complex<double>> &values,
              std::vector<double> *interleaved, std::string operation) {
    if (numeric_ != nullptr) {
      static_cast<void>(klu_z_free_numeric(&numeric_, &common_));
    }
    last_values_.clear();
    numeric_ = klu_z_factor(pattern_.column_offsets.data(),
                            pattern_.row_indices.data(), interleaved->data(),
                            symbolic_, &common_);
    if (numeric_ == nullptr) {
      return Result<bool>::Fail(KluErrorCode(common_, pattern_.size),
                                KluFailureMessage(operation, common_));
    }
    ++statistics_.numeric_factorizations;
    last_values_ = values;
    return Result<bool>::Ok(true);
  }

  [[nodiscard]] Result<std::vector<std::complex<double>>>
  SolveAndValidate(const ComplexCsrMatrix &matrix,
                   const std::vector<std::complex<double>> &rhs) {
    std::vector<double> solution_interleaved = InterleaveComplex(rhs);
    if (klu_z_solve(symbolic_, numeric_,
                    static_cast<std::int32_t>(pattern_.size), 1,
                    solution_interleaved.data(), &common_) == 0) {
      return Result<std::vector<std::complex<double>>>::Fail(
          ErrorCode::kFactorization,
          KluFailureMessage("KLU complex triangular solve", common_));
    }
    std::vector<std::complex<double>> solution;
    solution.reserve(pattern_.size);
    for (std::size_t index = 0; index < pattern_.size; ++index) {
      solution.emplace_back(solution_interleaved[2 * index],
                            solution_interleaved[2 * index + 1]);
    }
    Result<double> validation = ValidateSparseSolution(matrix, rhs, solution);
    if (!validation.ok()) {
      return Result<std::vector<std::complex<double>>>::Fail(
          validation.error().code, validation.error().message);
    }
    return Result<std::vector<std::complex<double>>>::Ok(std::move(solution));
  }

  [[nodiscard]] Result<std::vector<std::complex<double>>>
  FactorAndSolve(const ComplexCsrMatrix &matrix,
                 const std::vector<std::complex<double>> &rhs) {
    Result<SolverCscPattern> converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok()) {
      return Result<std::vector<std::complex<double>>>::Fail(
          converted.error().code, converted.error().message);
    }
    if (!SamePattern(pattern_, converted.value())) {
      return Result<std::vector<std::complex<double>>>::Fail(
          ErrorCode::kInvalidStructure,
          "numeric refactorization requires the analyzed CSR pattern");
    }
    if (rhs.size() != pattern_.size) {
      return Result<std::vector<std::complex<double>>>::Fail(
          ErrorCode::kInvalidStructure,
          "matrix and right-hand-side dimensions disagree");
    }
    for (const std::complex<double> value : rhs) {
      if (!IsFinite(value)) {
        return Result<std::vector<std::complex<double>>>::Fail(
            ErrorCode::kNonFinite,
            "right-hand side contains a non-finite value");
      }
    }
    if (pattern_.size == 0) {
      ++statistics_.solves;
      return Result<std::vector<std::complex<double>>>::Ok({});
    }

    std::vector<std::complex<double>> values =
        GatherCscValues(matrix, pattern_);
    std::vector<double> interleaved = InterleaveComplex(values);
    bool refactored_without_pivoting = false;
    if (numeric_ == nullptr) {
      Result<bool> factored = FactorFresh(values, &interleaved,
                                          "KLU complex numeric factorization");
      if (!factored.ok()) {
        return Result<std::vector<std::complex<double>>>::Fail(
            factored.error().code, factored.error().message);
      }
    } else if (values == last_values_) {
      ++statistics_.numeric_reuses;
    } else {
      if (klu_z_refactor(pattern_.column_offsets.data(),
                         pattern_.row_indices.data(), interleaved.data(),
                         symbolic_, numeric_, &common_) == 0) {
        ++statistics_.numeric_refactorization_fallbacks;
        Result<bool> factored = FactorFresh(
            values, &interleaved,
            "KLU complex pivoting factorization after refactorization failure");
        if (!factored.ok()) {
          return Result<std::vector<std::complex<double>>>::Fail(
              factored.error().code, factored.error().message);
        }
      } else {
        ++statistics_.numeric_refactorizations;
        last_values_ = values;
        refactored_without_pivoting = true;
      }
    }

    Result<std::vector<std::complex<double>>> solved =
        SolveAndValidate(matrix, rhs);
    if (!solved.ok() && refactored_without_pivoting) {
      ++statistics_.numeric_refactorization_fallbacks;
      Result<bool> factored = FactorFresh(values, &interleaved,
                                          "KLU complex pivoting factorization "
                                          "after refactorized solve failure");
      if (!factored.ok()) {
        return Result<std::vector<std::complex<double>>>::Fail(
            factored.error().code, factored.error().message);
      }
      solved = SolveAndValidate(matrix, rhs);
    }
    if (!solved.ok()) {
      return solved;
    }
    ++statistics_.solves;
    return solved;
  }

  SolverCscPattern pattern_;
  klu_common common_{};
  klu_symbolic *symbolic_ = nullptr;
  klu_numeric *numeric_ = nullptr;
  std::vector<std::complex<double>> last_values_;
  SparseSolverStatistics statistics_;
};

SparseComplexFactorization::SparseComplexFactorization(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}
SparseComplexFactorization::~SparseComplexFactorization() = default;
SparseComplexFactorization::SparseComplexFactorization(
    SparseComplexFactorization &&) noexcept = default;
SparseComplexFactorization &SparseComplexFactorization::operator=(
    SparseComplexFactorization &&) noexcept = default;

Result<std::unique_ptr<SparseComplexFactorization>>
SparseComplexFactorization::Analyze(const ComplexCsrMatrix &matrix) {
  try {
    Result<SolverCscPattern> converted = ConvertCsrToSolverCsc(matrix);
    if (!converted.ok()) {
      return Result<std::unique_ptr<SparseComplexFactorization>>::Fail(
          converted.error().code, converted.error().message);
    }
    auto implementation = std::make_unique<Impl>(converted.TakeValue());
    if (implementation->pattern_.size != 0) {
      if (HasEmptyStructuralLine(matrix, implementation->pattern_)) {
        return Result<std::unique_ptr<SparseComplexFactorization>>::Fail(
            ErrorCode::kSingular,
            "complex sparse matrix is structurally singular: an empty row or "
            "column prevents a full matching");
      }
      implementation->symbolic_ =
          klu_analyze(static_cast<std::int32_t>(implementation->pattern_.size),
                      implementation->pattern_.column_offsets.data(),
                      implementation->pattern_.row_indices.data(),
                      &implementation->common_);
      if (implementation->symbolic_ == nullptr) {
        return Result<std::unique_ptr<SparseComplexFactorization>>::Fail(
            KluErrorCode(implementation->common_,
                         implementation->pattern_.size),
            KluFailureMessage("KLU complex symbolic analysis",
                              implementation->common_));
      }
      if (implementation->symbolic_->structural_rank >= 0 &&
          static_cast<std::size_t>(implementation->symbolic_->structural_rank) <
              implementation->pattern_.size) {
        return Result<std::unique_ptr<SparseComplexFactorization>>::Fail(
            ErrorCode::kSingular,
            "KLU complex symbolic analysis found a structurally "
            "rank-deficient matrix: rank=" +
                std::to_string(implementation->symbolic_->structural_rank) +
                " size=" + std::to_string(implementation->pattern_.size));
      }
      ++implementation->statistics_.symbolic_analyses;
    }
    return Result<std::unique_ptr<SparseComplexFactorization>>::Ok(
        std::unique_ptr<SparseComplexFactorization>(
            new SparseComplexFactorization(std::move(implementation))));
  } catch (const std::bad_alloc &) {
    return Result<std::unique_ptr<SparseComplexFactorization>>::Fail(
        ErrorCode::kFactorization,
        "complex sparse symbolic analysis allocation failed");
  }
}

Result<std::vector<std::complex<double>>>
SparseComplexFactorization::FactorAndSolve(
    const ComplexCsrMatrix &matrix,
    const std::vector<std::complex<double>> &rhs) {
  try {
    return implementation_->FactorAndSolve(matrix, rhs);
  } catch (const std::bad_alloc &) {
    return Result<std::vector<std::complex<double>>>::Fail(
        ErrorCode::kFactorization,
        "complex sparse numeric factorization allocation failed");
  }
}

const SparseSolverStatistics &SparseComplexFactorization::statistics() const {
  return implementation_->statistics_;
}

Result<std::vector<double>> SolveSparseReal(const CsrMatrix &matrix,
                                            const std::vector<double> &rhs) {
  Result<std::unique_ptr<SparseRealFactorization>> factorization =
      SparseRealFactorization::Analyze(matrix);
  if (!factorization.ok()) {
    return Result<std::vector<double>>::Fail(factorization.error().code,
                                             factorization.error().message);
  }
  return factorization.value()->FactorAndSolve(matrix, rhs);
}

Result<std::vector<std::complex<double>>>
SolveSparseComplex(const ComplexCsrMatrix &matrix,
                   const std::vector<std::complex<double>> &rhs) {
  Result<std::unique_ptr<SparseComplexFactorization>> factorization =
      SparseComplexFactorization::Analyze(matrix);
  if (!factorization.ok()) {
    return Result<std::vector<std::complex<double>>>::Fail(
        factorization.error().code, factorization.error().message);
  }
  return factorization.value()->FactorAndSolve(matrix, rhs);
}

} // namespace ohmnivore
