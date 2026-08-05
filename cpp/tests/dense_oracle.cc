#include "cpp/tests/dense_oracle.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ohmnivore {

extern "C" const char ohmnivore_dense_oracle_test_only_marker[] =
    "OHMNIVORE_DENSE_ORACLE_TEST_ONLY_v1";

Result<std::vector<double>>
SolveDenseOracleReal(const CsrMatrix &matrix, const std::vector<double> &rhs) {
  if (matrix.rows != matrix.columns || rhs.size() != matrix.rows) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "matrix and right-hand-side dimensions disagree");
  }
  if (matrix.row_offsets.size() != matrix.rows + 1 ||
      matrix.column_indices.size() != matrix.values.size() ||
      matrix.row_offsets.back() != matrix.values.size()) {
    return Result<std::vector<double>>::Fail(ErrorCode::kSolve,
                                             "invalid CSR structure");
  }
  if (matrix.row_offsets.front() != 0) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve, "CSR row offsets must start at zero");
  }
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
        matrix.row_offsets[row + 1] > matrix.values.size()) {
      return Result<std::vector<double>>::Fail(ErrorCode::kSolve,
                                               "invalid CSR row offsets");
    }
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      if (matrix.column_indices[index] >= matrix.columns) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kSolve, "CSR column index is outside the matrix");
      }
      if (index > matrix.row_offsets[row] &&
          matrix.column_indices[index - 1] >= matrix.column_indices[index]) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kSolve,
            "CSR columns must be strictly increasing within each row");
      }
    }
  }

  const std::size_t size = matrix.rows;
  std::vector<double> dense = matrix.ToDense();
  std::vector<double> b = rhs;
  for (double value : dense) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "matrix contains a non-finite value");
    }
  }
  for (double value : b) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "right-hand side contains a non-finite value");
    }
  }

  for (std::size_t column = 0; column < size; ++column) {
    std::size_t pivot_row = column;
    double pivot_magnitude = std::abs(dense[column * size + column]);
    for (std::size_t row = column + 1; row < size; ++row) {
      const double candidate = std::abs(dense[row * size + column]);
      if (candidate > pivot_magnitude) {
        pivot_magnitude = candidate;
        pivot_row = row;
      }
    }
    // A single global tolerance is invalid for MNA matrices because their
    // rows mix conductance, incidence, and dynamic units across many orders of
    // magnitude. Partial pivoting already selects the largest entry in this
    // column; only an exact zero proves that no pivot exists.
    if (pivot_magnitude == 0.0) {
      return Result<std::vector<double>>::Fail(ErrorCode::kSolve,
                                               "matrix is singular at column " +
                                                   std::to_string(column));
    }

    if (pivot_row != column) {
      for (std::size_t entry = column; entry < size; ++entry) {
        std::swap(dense[column * size + entry],
                  dense[pivot_row * size + entry]);
      }
      std::swap(b[column], b[pivot_row]);
    }

    const double pivot = dense[column * size + column];
    for (std::size_t row = column + 1; row < size; ++row) {
      const double factor = dense[row * size + column] / pivot;
      dense[row * size + column] = 0.0;
      for (std::size_t entry = column + 1; entry < size; ++entry) {
        dense[row * size + entry] -= factor * dense[column * size + entry];
      }
      b[row] -= factor * b[column];
    }
  }

  std::vector<double> solution(size, 0.0);
  for (std::size_t reverse = 0; reverse < size; ++reverse) {
    const std::size_t row = size - reverse - 1;
    double residual = b[row];
    for (std::size_t column = row + 1; column < size; ++column) {
      residual -= dense[row * size + column] * solution[column];
    }
    solution[row] = residual / dense[row * size + row];
    if (!std::isfinite(solution[row])) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "solve produced a non-finite value");
    }
  }
  return Result<std::vector<double>>::Ok(std::move(solution));
}

Result<std::vector<std::complex<double>>>
SolveDenseOracleComplex(const ComplexCsrMatrix &matrix,
                        const std::vector<std::complex<double>> &rhs) {
  using Complex = std::complex<double>;
  if (matrix.rows != matrix.columns || rhs.size() != matrix.rows) {
    return Result<std::vector<Complex>>::Fail(
        ErrorCode::kSolve, "matrix and right-hand-side dimensions disagree");
  }
  if (matrix.row_offsets.size() != matrix.rows + 1 ||
      matrix.column_indices.size() != matrix.values.size() ||
      matrix.row_offsets.back() != matrix.values.size()) {
    return Result<std::vector<Complex>>::Fail(ErrorCode::kSolve,
                                              "invalid CSR structure");
  }
  if (matrix.row_offsets.front() != 0) {
    return Result<std::vector<Complex>>::Fail(
        ErrorCode::kSolve, "CSR row offsets must start at zero");
  }
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
        matrix.row_offsets[row + 1] > matrix.values.size()) {
      return Result<std::vector<Complex>>::Fail(ErrorCode::kSolve,
                                                "invalid CSR row offsets");
    }
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      if (matrix.column_indices[index] >= matrix.columns) {
        return Result<std::vector<Complex>>::Fail(
            ErrorCode::kSolve, "CSR column index is outside the matrix");
      }
      if (index > matrix.row_offsets[row] &&
          matrix.column_indices[index - 1] >= matrix.column_indices[index]) {
        return Result<std::vector<Complex>>::Fail(
            ErrorCode::kSolve,
            "CSR columns must be strictly increasing within each row");
      }
    }
  }

  const auto is_finite = [](Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
  };
  const std::size_t size = matrix.rows;
  std::vector<Complex> dense = matrix.ToDense();
  std::vector<Complex> b = rhs;
  for (Complex value : dense) {
    if (!is_finite(value)) {
      return Result<std::vector<Complex>>::Fail(
          ErrorCode::kSolve, "matrix contains a non-finite value");
    }
  }
  for (Complex value : b) {
    if (!is_finite(value)) {
      return Result<std::vector<Complex>>::Fail(
          ErrorCode::kSolve, "right-hand side contains a non-finite value");
    }
  }

  for (std::size_t column = 0; column < size; ++column) {
    std::size_t pivot_row = column;
    double pivot_magnitude = std::abs(dense[column * size + column]);
    for (std::size_t row = column + 1; row < size; ++row) {
      const double candidate = std::abs(dense[row * size + column]);
      if (candidate > pivot_magnitude) {
        pivot_magnitude = candidate;
        pivot_row = row;
      }
    }
    // See the real solver above: mixed-unit MNA scaling makes a global
    // absolute pivot threshold reject valid nonsingular AC systems.
    if (pivot_magnitude == 0.0) {
      return Result<std::vector<Complex>>::Fail(
          ErrorCode::kSolve,
          "matrix is singular at column " + std::to_string(column));
    }

    if (pivot_row != column) {
      for (std::size_t entry = column; entry < size; ++entry) {
        std::swap(dense[column * size + entry],
                  dense[pivot_row * size + entry]);
      }
      std::swap(b[column], b[pivot_row]);
    }

    const Complex pivot = dense[column * size + column];
    for (std::size_t row = column + 1; row < size; ++row) {
      const Complex factor = dense[row * size + column] / pivot;
      dense[row * size + column] = {0.0, 0.0};
      for (std::size_t entry = column + 1; entry < size; ++entry) {
        dense[row * size + entry] -= factor * dense[column * size + entry];
      }
      b[row] -= factor * b[column];
    }
  }

  std::vector<Complex> solution(size, {0.0, 0.0});
  for (std::size_t reverse = 0; reverse < size; ++reverse) {
    const std::size_t row = size - reverse - 1;
    Complex residual = b[row];
    for (std::size_t column = row + 1; column < size; ++column) {
      residual -= dense[row * size + column] * solution[column];
    }
    solution[row] = residual / dense[row * size + row];
    if (!is_finite(solution[row])) {
      return Result<std::vector<Complex>>::Fail(
          ErrorCode::kSolve, "solve produced a non-finite value");
    }
  }
  return Result<std::vector<Complex>>::Ok(std::move(solution));
}

} // namespace ohmnivore
