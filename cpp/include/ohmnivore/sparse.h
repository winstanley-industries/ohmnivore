#ifndef OHMNIVORE_SPARSE_H_
#define OHMNIVORE_SPARSE_H_

#include <complex>
#include <cstddef>
#include <vector>

namespace ohmnivore {

template <typename T> struct CsrMatrixBase {
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::vector<T> values;
  std::vector<std::size_t> column_indices;
  std::vector<std::size_t> row_offsets;

  [[nodiscard]] std::vector<T> ToDense() const {
    std::vector<T> dense(rows * columns, T{});
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t index = row_offsets[row]; index < row_offsets[row + 1];
           ++index) {
        dense[row * columns + column_indices[index]] = values[index];
      }
    }
    return dense;
  }
};

using CsrMatrix = CsrMatrixBase<double>;
using ComplexCsrMatrix = CsrMatrixBase<std::complex<double>>;

} // namespace ohmnivore

#endif // OHMNIVORE_SPARSE_H_
