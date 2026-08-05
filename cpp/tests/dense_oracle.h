#ifndef OHMNIVORE_TESTS_DENSE_ORACLE_H_
#define OHMNIVORE_TESTS_DENSE_ORACLE_H_

#include <complex>
#include <vector>

#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

// Independent exact-small oracle. This API is provided only by the Bazel
// testonly dense_oracle target and is never a production dependency.
[[nodiscard]] Result<std::vector<double>>
SolveDenseOracleReal(const CsrMatrix &matrix, const std::vector<double> &rhs);

[[nodiscard]] Result<std::vector<std::complex<double>>>
SolveDenseOracleComplex(const ComplexCsrMatrix &matrix,
                        const std::vector<std::complex<double>> &rhs);

} // namespace ohmnivore

#endif // OHMNIVORE_TESTS_DENSE_ORACLE_H_
