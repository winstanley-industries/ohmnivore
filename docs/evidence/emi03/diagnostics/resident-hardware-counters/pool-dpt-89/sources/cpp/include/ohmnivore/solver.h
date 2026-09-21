#ifndef OHMNIVORE_SOLVER_H_
#define OHMNIVORE_SOLVER_H_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

namespace internal {
class PreparedNewtonWorkspace;
}

inline constexpr double kSparseBackwardErrorTolerance = 1e-10;
inline constexpr double kSparseComponentwiseBackwardErrorTolerance = 1e-5;

// KLU consumes compressed sparse column (CSC) storage with signed 32-bit
// indexes. The conversion retains every canonical CSR entry, including
// explicit numerical zeros. csr_value_indices maps each CSC value back to its
// source CSR value and makes later numeric refactorizations deterministic.
struct SolverCscPattern {
  std::size_t size = 0;
  std::vector<std::int32_t> column_offsets;
  std::vector<std::int32_t> row_indices;
  std::vector<std::size_t> csr_value_indices;
};

[[nodiscard]] Result<SolverCscPattern>
ConvertCsrToSolverCsc(const CsrMatrix &matrix);
[[nodiscard]] Result<SolverCscPattern>
ConvertCsrToSolverCsc(const ComplexCsrMatrix &matrix);

struct SparseSolverStatistics {
  std::size_t symbolic_analyses = 0;
  std::size_t numeric_factorizations = 0;
  std::size_t numeric_refactorizations = 0;
  std::size_t numeric_refactorization_fallbacks = 0;
  std::size_t numeric_reuses = 0;
  std::size_t solves = 0;
  std::size_t iterative_refinement_solves = 0;
};

// Returns a typed validation failure unless every result is finite, the
// row-equilibrated normwise backward error is at most
// kSparseBackwardErrorTolerance, and the maximum rowwise componentwise guard
// is at most kSparseComponentwiseBackwardErrorTolerance. The second check
// scales each row by its own |b_i| plus sum_j |a_ij|*|x_j|, so unrelated
// large-unit variables cannot hide a grossly bad small-unit equation.
[[nodiscard]] Result<double>
ValidateSparseSolution(const CsrMatrix &matrix, const std::vector<double> &rhs,
                       const std::vector<double> &solution);
[[nodiscard]] Result<double>
ValidateSparseSolution(const ComplexCsrMatrix &matrix,
                       const std::vector<std::complex<double>> &rhs,
                       const std::vector<std::complex<double>> &solution);

class SparseRealFactorization {
public:
  [[nodiscard]] static Result<std::unique_ptr<SparseRealFactorization>>
  Analyze(const CsrMatrix &matrix);

  ~SparseRealFactorization();
  SparseRealFactorization(SparseRealFactorization &&) noexcept;
  SparseRealFactorization &operator=(SparseRealFactorization &&) noexcept;
  SparseRealFactorization(const SparseRealFactorization &) = delete;
  SparseRealFactorization &operator=(const SparseRealFactorization &) = delete;

  [[nodiscard]] Result<std::vector<double>>
  FactorAndSolve(const CsrMatrix &matrix, const std::vector<double> &rhs);
  // Explicit EMI-02 option: correct a nonzero original residual, then retry
  // failed backward-error guards, with at most four FP64 KLU correction solves.
  [[nodiscard]] Result<std::vector<double>>
  FactorAndSolveRefined(const CsrMatrix &matrix, const std::vector<double> &rhs,
                        std::size_t maximum_refinements = 4);
  [[nodiscard]] const SparseSolverStatistics &statistics() const;

private:
  friend class internal::PreparedNewtonWorkspace;
  // Only an immediately completed prepared full assembly may establish this
  // finite-input precondition. Structure and all result guards still run.
  [[nodiscard]] Result<std::vector<double>>
  FactorAndSolveAdmitted(const CsrMatrix &matrix,
                         const std::vector<double> &rhs,
                         std::size_t maximum_refinements);
  class Impl;
  explicit SparseRealFactorization(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

class SparseComplexFactorization {
public:
  [[nodiscard]] static Result<std::unique_ptr<SparseComplexFactorization>>
  Analyze(const ComplexCsrMatrix &matrix);

  ~SparseComplexFactorization();
  SparseComplexFactorization(SparseComplexFactorization &&) noexcept;
  SparseComplexFactorization &operator=(SparseComplexFactorization &&) noexcept;
  SparseComplexFactorization(const SparseComplexFactorization &) = delete;
  SparseComplexFactorization &
  operator=(const SparseComplexFactorization &) = delete;

  [[nodiscard]] Result<std::vector<std::complex<double>>>
  FactorAndSolve(const ComplexCsrMatrix &matrix,
                 const std::vector<std::complex<double>> &rhs);
  [[nodiscard]] const SparseSolverStatistics &statistics() const;

private:
  class Impl;
  explicit SparseComplexFactorization(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

[[nodiscard]] Result<std::vector<double>>
SolveSparseReal(const CsrMatrix &matrix, const std::vector<double> &rhs);
[[nodiscard]] Result<std::vector<std::complex<double>>>
SolveSparseComplex(const ComplexCsrMatrix &matrix,
                   const std::vector<std::complex<double>> &rhs);

} // namespace ohmnivore

#endif // OHMNIVORE_SOLVER_H_
