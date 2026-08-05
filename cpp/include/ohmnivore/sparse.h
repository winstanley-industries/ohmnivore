#ifndef OHMNIVORE_SPARSE_H_
#define OHMNIVORE_SPARSE_H_

#include <cstddef>
#include <vector>

namespace ohmnivore {

struct CsrMatrix {
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::vector<double> values;
  std::vector<std::size_t> column_indices;
  std::vector<std::size_t> row_offsets;

  [[nodiscard]] std::vector<double> ToDense() const {
    std::vector<double> dense(rows * columns, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t index = row_offsets[row]; index < row_offsets[row + 1];
           ++index) {
        dense[row * columns + column_indices[index]] = values[index];
      }
    }
    return dense;
  }
};

} // namespace ohmnivore

#endif // OHMNIVORE_SPARSE_H_
