#ifndef OHMNIVORE_SOLVER_H_
#define OHMNIVORE_SOLVER_H_

#include <complex>
#include <vector>

#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// CPU correctness paths. These dense partial-pivoting implementations are
// intentionally temporary and will be replaced by the selected hermetic
// sparse-direct FP64 oracle in a later phase.
[[nodiscard]] Result<std::vector<double>>
SolveCpuReference(const CsrMatrix &matrix, const std::vector<double> &rhs);

[[nodiscard]] Result<std::vector<std::complex<double>>>
SolveCpuComplexReference(const ComplexCsrMatrix &matrix,
                         const std::vector<std::complex<double>> &rhs);

} // namespace ohmnivore

#endif // OHMNIVORE_SOLVER_H_
