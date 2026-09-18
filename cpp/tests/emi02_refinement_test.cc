#include "cpp/tests/google_test.h"

#include <cmath>
#include <limits>
#include <vector>

#include "ohmnivore/solver.h"

namespace ohmnivore {
namespace {

// Independently generated mixed-unit matrix, frozen before the test assertions.
// The last equation constrains two near-zero unknowns. Large unrelated rows
// cannot hide a relative violation of that equation. No device/model text or
// production compiler/stamping helper supplies this fixture.
CsrMatrix MixedScaleMatrix() {
  return {.rows = 6,
          .columns = 6,
          .values = {4000,   4e-7,  8e-5,  -1e10, -6e7, -50,  -5e-12, 6e-11,
                     -3e-12, .008,  -7e12, 1e7,   4e5,  -3,   6e8,    7e-7,
                     .2,     4e-9,  -6000, -2e5,  .008, 9e-8, 5e-10,  -4e-5,
                     .0002,  4e-11, -7e-7, 7e-8,  6000, -4e9, 1,      1},
          .column_indices = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
                             4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 2, 3},
          .row_offsets = {0, 6, 12, 18, 24, 30, 32}};
}

std::vector<double> MixedScaleRhs() {
  return {-119995900.0000004, -14000020000000,    400003.39999999199,
          194000.00008000099, 8000012000.0002003, 0};
}

TEST(Emi02Refinement, RecoversRejectedSystemWithoutWeakeningOriginalGuards) {
  const auto matrix = MixedScaleMatrix();
  const auto rhs = MixedScaleRhs();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto &factorization = *analyzed.value();
  auto ordinary = factorization.FactorAndSolve(matrix, rhs);
  ASSERT_FALSE(ordinary.ok());
  EXPECT_EQ(ordinary.error().code, ErrorCode::kSolutionValidation);
  EXPECT_EQ(factorization.statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(factorization.statistics().solves, 0U);
  auto disabled = factorization.FactorAndSolveRefined(matrix, rhs, 0);
  ASSERT_FALSE(disabled.ok());
  EXPECT_EQ(disabled.error().code, ErrorCode::kSolutionValidation);
  EXPECT_EQ(factorization.statistics().iterative_refinement_solves, 0U);

  auto refined = factorization.FactorAndSolveRefined(matrix, rhs);
  ASSERT_TRUE(refined.ok()) << refined.error().message;
  EXPECT_GT(factorization.statistics().iterative_refinement_solves, 0U);
  EXPECT_LE(factorization.statistics().iterative_refinement_solves, 4U);
  EXPECT_EQ(factorization.statistics().numeric_factorizations, 1U);
  EXPECT_EQ(factorization.statistics().solves, 1U);
  ASSERT_TRUE(ValidateSparseSolution(matrix, rhs, refined.value()).ok());

  // Independent equation residuals against the original fixture. The small
  // final equation is included with its own scale, not the large RHS norm.
  for (std::size_t row = 0; row < 6; ++row) {
    long double residual = -static_cast<long double>(rhs[row]);
    long double scale = std::abs(static_cast<long double>(rhs[row]));
    for (std::size_t entry = matrix.row_offsets[row];
         entry < matrix.row_offsets[row + 1]; ++entry) {
      const long double term = static_cast<long double>(matrix.values[entry]) *
                               refined.value()[matrix.column_indices[entry]];
      residual += term;
      scale += std::abs(term);
    }
    EXPECT_LE(std::abs(residual), 1e-5L * scale) << "original equation " << row;
  }
  // The opt-in solve has not altered ordinary triangular-solve behavior.
  const auto previous_corrections =
      factorization.statistics().iterative_refinement_solves;
  auto unchanged = factorization.FactorAndSolve(matrix, rhs);
  ASSERT_FALSE(unchanged.ok());
  EXPECT_EQ(unchanged.error().code, ErrorCode::kSolutionValidation);
  EXPECT_EQ(factorization.statistics().iterative_refinement_solves,
            previous_corrections);
}

TEST(Emi02Refinement, ExactlyZeroResidualUsesNoCorrectionSolve) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {2, 4},
                         .column_indices = {0, 1},
                         .row_offsets = {0, 1, 2}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  for (std::size_t budget : {0U, 1U, 4U}) {
    auto result =
        analyzed.value()->FactorAndSolveRefined(matrix, {6, -8}, budget);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value(), (std::vector<double>{3, -2}));
    EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  }
  EXPECT_EQ(analyzed.value()->statistics().solves, 3U);
}

TEST(Emi02Refinement, ImprovesWeakModeEvenWhenBackwardErrorAlreadyPasses) {
  // Independent two-equation oracle: adding the rows gives x+y=256,
  // subtracting them gives (2e8+1)*(x-y)=0. Thus x=y=128 exactly.
  // Large differential coupling makes a weak common-mode error almost
  // invisible to both backward-error guards, despite their unchanged limits.
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1e8 + 1, -1e8, -1e8, 1e8 + 1},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  const std::vector<double> rhs{128, 128};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto ordinary = analyzed.value()->FactorAndSolve(matrix, rhs);
  ASSERT_TRUE(ordinary.ok()) << ordinary.error().message;
  EXPECT_TRUE(ValidateSparseSolution(matrix, rhs, ordinary.value()).ok());
  const double ordinary_error = std::abs(ordinary.value()[0] - 128.0);
  EXPECT_GT(ordinary_error, 1e-7);
  auto disabled = analyzed.value()->FactorAndSolveRefined(matrix, rhs, 0);
  ASSERT_TRUE(disabled.ok());
  EXPECT_EQ(disabled.value(), ordinary.value());
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  auto refined = analyzed.value()->FactorAndSolveRefined(matrix, rhs, 1);
  ASSERT_TRUE(refined.ok()) << refined.error().message;
  for (const double value : refined.value())
    EXPECT_NEAR(value, 128.0, 5e-9);
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 1U);
  EXPECT_TRUE(ValidateSparseSolution(matrix, rhs, refined.value()).ok());
}

TEST(Emi02Refinement,
     ResourceStructureNonfiniteAndSingularityErrorsDoNotRefine) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1, 2, 2, 4},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto excessive = analyzed.value()->FactorAndSolveRefined(matrix, {1, 2}, 5);
  ASSERT_FALSE(excessive.ok());
  EXPECT_EQ(excessive.error().code, ErrorCode::kUnsupportedSize);
  auto dimension = analyzed.value()->FactorAndSolveRefined(matrix, {});
  ASSERT_FALSE(dimension.ok());
  EXPECT_EQ(dimension.error().code, ErrorCode::kInvalidStructure);
  auto nonfinite = analyzed.value()->FactorAndSolveRefined(
      matrix, {std::numeric_limits<double>::infinity(), 0});
  ASSERT_FALSE(nonfinite.ok());
  EXPECT_EQ(nonfinite.error().code, ErrorCode::kNonFinite);
  auto singular = analyzed.value()->FactorAndSolveRefined(matrix, {1, 2});
  ASSERT_FALSE(singular.ok());
  EXPECT_EQ(singular.error().code, ErrorCode::kSingular);
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
}

} // namespace
} // namespace ohmnivore
