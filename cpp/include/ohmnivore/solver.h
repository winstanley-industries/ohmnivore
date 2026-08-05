#ifndef OHMNIVORE_SOLVER_H_
#define OHMNIVORE_SOLVER_H_

#include <vector>

#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// Phase 1 correctness path. This dense partial-pivoting implementation is
// intentionally bounded to the vertical slice and will be replaced by the
// selected hermetic sparse-direct FP64 oracle in a later phase.
[[nodiscard]] Result<std::vector<double>>
SolveCpuReference(const CsrMatrix &matrix, const std::vector<double> &rhs);

} // namespace ohmnivore

#endif // OHMNIVORE_SOLVER_H_
