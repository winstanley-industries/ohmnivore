#include "ohmnivore/solver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ohmnivore {

Result<std::vector<double>> SolveCpuReference(const CsrMatrix &matrix,
                                              const std::vector<double> &rhs) {
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
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
        matrix.row_offsets[row + 1] > matrix.values.size()) {
      return Result<std::vector<double>>::Fail(ErrorCode::kSolve,
                                               "invalid CSR row offsets");
    }
  }
  for (std::size_t column : matrix.column_indices) {
    if (column >= matrix.columns) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "CSR column index is outside the matrix");
    }
  }

  const std::size_t size = matrix.rows;
  std::vector<double> dense = matrix.ToDense();
  std::vector<double> b = rhs;
  double matrix_scale = 0.0;
  for (double value : dense) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "matrix contains a non-finite value");
    }
    matrix_scale = std::max(matrix_scale, std::abs(value));
  }
  for (double value : b) {
    if (!std::isfinite(value)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kSolve, "right-hand side contains a non-finite value");
    }
  }

  const double pivot_floor =
      std::numeric_limits<double>::epsilon() * std::max(1.0, matrix_scale) *
      static_cast<double>(std::max<std::size_t>(size, 1)) * 16.0;
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
    if (pivot_magnitude <= pivot_floor) {
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

} // namespace ohmnivore
