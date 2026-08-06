#ifndef OHMNIVORE_CPP_SRC_NONLINEAR_INTERNAL_H_
#define OHMNIVORE_CPP_SRC_NONLINEAR_INTERNAL_H_

#include <cstddef>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/status.h"

namespace ohmnivore::internal {

// A false node row must already have been replaced by an independently
// selected reactive-state constraint. Canonical public nonlinear entry points
// always stamp both non-ground diode terminal equations.
[[nodiscard]] Result<NonlinearPointResult> RunNonlinearPointForProjection(
    const MnaSystem &system, const std::vector<double> &initial_guess,
    const std::vector<bool> &active_node_equations,
    SparseRealFactorization *factorization, std::size_t maximum_iterations);

} // namespace ohmnivore::internal

#endif // OHMNIVORE_CPP_SRC_NONLINEAR_INTERNAL_H_
