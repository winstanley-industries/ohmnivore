#include "cpp/tests/google_test.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "ohmnivore/solver.h"

namespace ohmnivore {
namespace {

CsrMatrix TriangularMatrix() {
  // Full canonical structure deliberately retains the numerical zeros.
  return {.rows = 3,
          .columns = 3,
          .values = {4, 1, 0, 0, 8, 2, 0, 0, 16},
          .column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2},
          .row_offsets = {0, 3, 6, 9}};
}

void ExpectBits(const std::vector<double> &actual,
                const std::vector<double> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[i]),
              std::bit_cast<std::uint64_t>(expected[i]))
        << "unknown " << i;
  }
}

void ExpectFailure(SparseRealFactorization *factorization,
                   const CsrMatrix &matrix, const std::vector<double> &rhs,
                   ErrorCode code, const std::string &message) {
  auto ordinary = factorization->FactorAndSolve(matrix, rhs);
  ASSERT_FALSE(ordinary.ok());
  EXPECT_EQ(ordinary.error().code, code);
  EXPECT_EQ(ordinary.error().message, message);
  auto refined = factorization->FactorAndSolveRefined(matrix, rhs);
  ASSERT_FALSE(refined.ok());
  EXPECT_EQ(refined.error().code, code);
  EXPECT_EQ(refined.error().message, message);
}

TEST(Emi02SolverReuse, PreservesExactSolutionsNumericCallsAndInputOwnership) {
  auto matrix = TriangularMatrix();
  const auto original = matrix;
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto &factorization = *analyzed.value();
  auto first = factorization.FactorAndSolve(matrix, {4, -16, 128});
  ASSERT_TRUE(first.ok()) << first.error().message;
  // The independent triangular equations give z=8, y=-4, x=2 exactly.
  ExpectBits(first.value(), {2, -4, 8});
  for (double &value : matrix.values)
    value *= 2;
  auto second = factorization.FactorAndSolveRefined(matrix, {8, -32, 256});
  ASSERT_TRUE(second.ok()) << second.error().message;
  ExpectBits(second.value(), {2, -4, 8});
  auto reused = factorization.FactorAndSolve(matrix, {4, -16, 128});
  ASSERT_TRUE(reused.ok()) << reused.error().message;
  ExpectBits(reused.value(), {1, -2, 4});
  auto zero = factorization.FactorAndSolve(matrix, {0, 0, 0});
  ASSERT_TRUE(zero.ok()) << zero.error().message;
  EXPECT_EQ(zero.value(), (std::vector<double>{0, 0, 0}));
  const auto statistics = factorization.statistics();
  EXPECT_EQ(statistics.symbolic_analyses, 1U);
  EXPECT_EQ(statistics.numeric_factorizations, 1U);
  EXPECT_EQ(statistics.numeric_refactorizations, 1U);
  EXPECT_EQ(statistics.numeric_refactorization_fallbacks, 0U);
  EXPECT_EQ(statistics.numeric_reuses, 2U);
  EXPECT_EQ(statistics.iterative_refinement_solves, 0U);
  EXPECT_EQ(statistics.solves, 4U);
  EXPECT_EQ(matrix.row_offsets, original.row_offsets);
  EXPECT_EQ(matrix.column_indices, original.column_indices);
  for (std::size_t i = 0; i < matrix.values.size(); ++i)
    EXPECT_EQ(matrix.values[i], 2 * original.values[i]);
}

TEST(Emi02SolverReuse, PreservesSimultaneousInvalidInputPrecedence) {
  const auto matrix = TriangularMatrix();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto *factorization = analyzed.value().get();
  const double infinity = std::numeric_limits<double>::infinity();

  auto invalid = matrix;
  invalid.values[0] = infinity;
  ExpectFailure(factorization, invalid, {}, ErrorCode::kNonFinite,
                "matrix contains a non-finite value");

  invalid.values.pop_back();
  ExpectFailure(factorization, invalid, {}, ErrorCode::kInvalidStructure,
                "invalid CSR structure");

  invalid = matrix;
  invalid.values[0] = infinity;
  invalid.row_offsets[1] = 10;
  ExpectFailure(factorization, invalid, {}, ErrorCode::kInvalidStructure,
                "invalid CSR row offsets");

  invalid = matrix;
  invalid.values[0] = infinity;
  invalid.column_indices[3] = 3;
  // The original row-ordered converter sees the first-row nonfinite value
  // before the later row's invalid column, even though the pattern differs.
  ExpectFailure(factorization, invalid, {}, ErrorCode::kNonFinite,
                "matrix contains a non-finite value");

  invalid = matrix;
  invalid.rows =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  ExpectFailure(
      factorization, invalid, {}, ErrorCode::kUnsupportedSize,
      "sparse solver requires dimensions and nonzero count representable "
      "as signed 32-bit indexes");

  ExpectFailure(factorization, matrix, {infinity}, ErrorCode::kInvalidStructure,
                "matrix and right-hand-side dimensions disagree");
  ExpectFailure(factorization, matrix, {infinity, 0, 0}, ErrorCode::kNonFinite,
                "right-hand side contains a non-finite value");
  EXPECT_EQ(factorization->statistics().numeric_factorizations, 0U);
  EXPECT_EQ(factorization->statistics().solves, 0U);
  EXPECT_EQ(factorization->statistics().iterative_refinement_solves, 0U);
}

TEST(Emi02SolverReuse, CachedStructureIsOwnedAndRejectsCanonicalReplacement) {
  CsrMatrix original{.rows = 2,
                     .columns = 2,
                     .values = {2, 4},
                     .column_indices = {0, 1},
                     .row_offsets = {0, 1, 2}};
  const auto preserved = original;
  auto analyzed = SparseRealFactorization::Analyze(original);
  ASSERT_TRUE(analyzed.ok());
  original.column_indices = {1, 0};
  ExpectFailure(analyzed.value().get(), original, {0, 0},
                ErrorCode::kInvalidStructure,
                "numeric refactorization requires the analyzed CSR pattern");
  original.values[0] = std::numeric_limits<double>::quiet_NaN();
  ExpectFailure(analyzed.value().get(), original, {0, 0}, ErrorCode::kNonFinite,
                "matrix contains a non-finite value");
  original.row_offsets = {0, 2, 1};
  ExpectFailure(
      analyzed.value().get(), original, {0, 0}, ErrorCode::kInvalidStructure,
      "CSR row offsets must start at zero and end at the value count");
  auto recovered = analyzed.value()->FactorAndSolve(preserved, {6, -8});
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  ExpectBits(recovered.value(), {3, -2});
}

TEST(Emi02SolverReuse, ZeroRightHandSideStillChecksChangedJacobianRank) {
  auto matrix = TriangularMatrix();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto first = analyzed.value()->FactorAndSolve(matrix, {0, 0, 0});
  ASSERT_TRUE(first.ok());
  matrix.values.assign(matrix.values.size(), 0.0);
  auto singular = analyzed.value()->FactorAndSolve(matrix, {0, 0, 0});
  ASSERT_FALSE(singular.ok());
  EXPECT_EQ(singular.error().code, ErrorCode::kSingular);
  EXPECT_EQ(analyzed.value()->statistics().solves, 1U);
  EXPECT_GT(analyzed.value()->statistics().numeric_refactorization_fallbacks,
            0U);
  auto recovered =
      analyzed.value()->FactorAndSolve(TriangularMatrix(), {4, -16, 128});
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  ExpectBits(recovered.value(), {2, -4, 8});
}

TEST(Emi02SolverReuse, PublicValidationStillIndependentlyChecksEveryInput) {
  auto matrix = TriangularMatrix();
  matrix.row_offsets[1] = 10;
  auto invalid = ValidateSparseSolution(matrix, {}, {});
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidStructure);
  EXPECT_EQ(invalid.error().message, "invalid CSR row offsets");
  matrix = TriangularMatrix();
  matrix.values[0] = std::numeric_limits<double>::infinity();
  auto nonfinite = ValidateSparseSolution(matrix, {}, {});
  ASSERT_FALSE(nonfinite.ok());
  EXPECT_EQ(nonfinite.error().code, ErrorCode::kNonFinite);
  matrix = TriangularMatrix();
  auto wrong = ValidateSparseSolution(matrix, {4, -16, 128}, {2, -4, 7});
  ASSERT_FALSE(wrong.ok());
  EXPECT_EQ(wrong.error().code, ErrorCode::kSolutionValidation);
  EXPECT_NE(wrong.error().message.find("componentwise_error="),
            std::string::npos);
}

TEST(Emi02SolverReuse,
     RepeatedRefinementOverwritesItsPrivateCorrectionScratch) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1e8 + 1, -1e8, -1e8, 1e8 + 1},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  for (double value : {128.0, -64.0, 32.0}) {
    auto solved =
        analyzed.value()->FactorAndSolveRefined(matrix, {value, value});
    ASSERT_TRUE(solved.ok()) << solved.error().message;
    // Adding/subtracting the two independently stated equations gives x=y=b.
    for (double actual : solved.value())
      EXPECT_NEAR(actual, value, 5e-9);
    EXPECT_TRUE(
        ValidateSparseSolution(matrix, {value, value}, solved.value()).ok());
  }
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 3U);
  EXPECT_EQ(analyzed.value()->statistics().numeric_factorizations, 1U);
  EXPECT_EQ(analyzed.value()->statistics().numeric_reuses, 2U);
}

TEST(Emi02SolverReuse, CachedResidualMatchesIndependentOldCorrectionBits) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1e8 + 1, -1e8, -1e8, 1e8 + 1},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  auto baseline = SparseRealFactorization::Analyze(matrix);
  auto optimized = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(baseline.ok());
  ASSERT_TRUE(optimized.ok());
  std::size_t corrections = 0;
  for (double value : {128.0, -64.0, 1e100, -1e-100, 32.0}) {
    SCOPED_TRACE(value);
    const std::vector<double> rhs{value, value};
    auto ordinary = baseline.value()->FactorAndSolve(matrix, rhs);
    ASSERT_TRUE(ordinary.ok()) << ordinary.error().message;
    auto expected = ordinary.value();
    std::vector<double> residual(2, 0.0);
    bool nonzero = false;
    // Independently retain the old second traversal, including signed
    // subtraction and the conversion boundary before the correction solve.
    for (std::size_t row = 0; row < 2; ++row) {
      long double product = 0.0L;
      for (std::size_t entry = matrix.row_offsets[row];
           entry < matrix.row_offsets[row + 1]; ++entry) {
        product += static_cast<long double>(matrix.values[entry]) *
                   static_cast<long double>(
                       ordinary.value()[matrix.column_indices[entry]]);
      }
      residual[row] =
          static_cast<double>(static_cast<long double>(rhs[row]) - product);
      ASSERT_TRUE(std::isfinite(residual[row]));
      nonzero |= residual[row] != 0.0;
    }
    if (nonzero) {
      auto correction = baseline.value()->FactorAndSolve(matrix, residual);
      ASSERT_TRUE(correction.ok()) << correction.error().message;
      for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] += correction.value()[i];
      ++corrections;
    }
    auto solved = optimized.value()->FactorAndSolveRefined(matrix, rhs, 1);
    ASSERT_TRUE(solved.ok()) << solved.error().message;
    ExpectBits(solved.value(), expected);
  }
  EXPECT_GT(corrections, 1U); // Exercise reuse after its first allocation.
  EXPECT_EQ(optimized.value()->statistics().iterative_refinement_solves,
            corrections);
  EXPECT_EQ(optimized.value()->statistics().solves, 5U);
}

TEST(Emi02SolverReuse, ZeroAndNonfiniteValidationKeepsEveryGuard) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {2, -4},
                         .column_indices = {0, 1},
                         .row_offsets = {0, 1, 2}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto positive_zero = analyzed.value()->FactorAndSolve(matrix, {0.0, 0.0});
  ASSERT_TRUE(positive_zero.ok()) << positive_zero.error().message;
  ExpectBits(positive_zero.value(), {0.0, -0.0});
  auto negative_zero =
      analyzed.value()->FactorAndSolveRefined(matrix, {-0.0, -0.0});
  ASSERT_TRUE(negative_zero.ok()) << negative_zero.error().message;
  ExpectBits(negative_zero.value(), {-0.0, 0.0});
  auto repeated = analyzed.value()->FactorAndSolveRefined(matrix, {0.0, 0.0});
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  ExpectBits(repeated.value(), positive_zero.value());
  EXPECT_EQ(analyzed.value()->statistics().solves, 3U);
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto &rhs :
       {std::vector<double>{nan, 0}, std::vector<double>{0, nan}}) {
    auto result = analyzed.value()->FactorAndSolve(matrix, rhs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::kNonFinite);
    EXPECT_EQ(result.error().message,
              "right-hand side contains a non-finite value");
  }
  auto nonfinite_solution = ValidateSparseSolution(matrix, {0, 0}, {0, nan});
  ASSERT_FALSE(nonfinite_solution.ok());
  EXPECT_EQ(nonfinite_solution.error().code, ErrorCode::kNonFinite);
  EXPECT_EQ(nonfinite_solution.error().message,
            "sparse solve produced a non-finite value");
  auto both = ValidateSparseSolution(matrix, {nan, 0}, {0, nan});
  ASSERT_FALSE(both.ok());
  EXPECT_EQ(both.error().message,
            "solution validation right-hand side is non-finite");
  EXPECT_EQ(analyzed.value()->statistics().solves, 3U);
}

TEST(Emi02SolverReuse, ExactExtremeFiniteRowsAndZerosRetainOriginalGuards) {
  const double maximum = std::numeric_limits<double>::max();
  const double minimum = std::numeric_limits<double>::denorm_min();
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {maximum, -minimum},
                         .column_indices = {0, 1},
                         .row_offsets = {0, 1, 2}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto solved =
      analyzed.value()->FactorAndSolveRefined(matrix, {maximum, -minimum});
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ExpectBits(solved.value(), {1, 1});
  EXPECT_TRUE(
      ValidateSparseSolution(matrix, {maximum, -minimum}, solved.value()).ok());
  auto zero = analyzed.value()->FactorAndSolveRefined(matrix, {0.0, 0.0});
  ASSERT_TRUE(zero.ok()) << zero.error().message;
  ExpectBits(zero.value(), {0.0, -0.0});
  EXPECT_TRUE(ValidateSparseSolution(matrix, {0.0, 0.0}, zero.value()).ok());
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(analyzed.value()->statistics().solves, 2U);
}

TEST(Emi02SolverReuse,
     RowMetadataMatchesUncachedValidationAcrossChangedMatrixAndRhs) {
  CsrMatrix matrix{.rows = 2,
                   .columns = 2,
                   .values = {1e8 + 1, -1e8, -1e8, 1e8 + 1},
                   .column_indices = {0, 1, 0, 1},
                   .row_offsets = {0, 2, 4}};
  auto baseline = SparseRealFactorization::Analyze(matrix);
  auto cached = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(baseline.ok());
  ASSERT_TRUE(cached.ok());
  struct Case {
    double first_scale;
    double second_scale;
    double rhs_value;
  };
  std::size_t corrections = 0;
  for (const Case input :
       {Case{1, 1, 128}, Case{1e-80, 1e80, -64}, Case{1e80, 1e-80, 32},
        Case{1, 1, 1e100}, Case{1e-100, 1e-100, 128}, Case{1e100, 1e100, 128},
        Case{1, 1, -1e-100}, Case{1, 1, 0}, Case{1, 1, -32}}) {
    SCOPED_TRACE(input.first_scale);
    SCOPED_TRACE(input.second_scale);
    SCOPED_TRACE(input.rhs_value);
    matrix.values = {(1e8 + 1) * input.first_scale, -1e8 * input.first_scale,
                     -1e8 * input.second_scale, (1e8 + 1) * input.second_scale};
    const std::vector<double> rhs{input.rhs_value * input.first_scale,
                                  input.rhs_value * input.second_scale};
    auto ordinary = baseline.value()->FactorAndSolve(matrix, rhs);
    ASSERT_TRUE(ordinary.ok()) << ordinary.error().message;
    auto expected = ordinary.value();
    std::vector<double> correction(2);
    bool nonzero = false;
    // Independent original residual traversal and correction. The public
    // validator below retains all original row-scale/norm computations and
    // shares no metadata with either factorization's private validation.
    for (std::size_t row = 0; row < 2; ++row) {
      long double product = 0.0L;
      for (std::size_t index = matrix.row_offsets[row];
           index < matrix.row_offsets[row + 1]; ++index) {
        product +=
            static_cast<long double>(matrix.values[index]) *
            static_cast<long double>(expected[matrix.column_indices[index]]);
      }
      correction[row] =
          static_cast<double>(static_cast<long double>(rhs[row]) - product);
      nonzero |= correction[row] != 0.0;
    }
    if (nonzero) {
      auto delta = baseline.value()->FactorAndSolve(matrix, correction);
      ASSERT_TRUE(delta.ok()) << delta.error().message;
      for (std::size_t index = 0; index < expected.size(); ++index)
        expected[index] += delta.value()[index];
      ++corrections;
    }
    const auto original_validation =
        ValidateSparseSolution(matrix, rhs, expected);
    auto optimized = cached.value()->FactorAndSolveRefined(matrix, rhs, 1);
    ASSERT_EQ(optimized.ok(), original_validation.ok());
    if (original_validation.ok()) {
      ExpectBits(optimized.value(), expected);
      const auto independent_validation =
          ValidateSparseSolution(matrix, rhs, optimized.value());
      ASSERT_TRUE(independent_validation.ok());
      EXPECT_EQ(std::bit_cast<std::uint64_t>(independent_validation.value()),
                std::bit_cast<std::uint64_t>(original_validation.value()));
    } else {
      EXPECT_EQ(optimized.error().code, original_validation.error().code);
      EXPECT_EQ(optimized.error().message, original_validation.error().message);
    }
  }
  EXPECT_GT(corrections, 5U);
  EXPECT_EQ(cached.value()->statistics().iterative_refinement_solves,
            corrections);

  // A failed call cannot make previously cached row metadata valid for the
  // next RHS or numeric matrix. Rank rejection still precedes any zero
  // shortcut.
  auto invalid = matrix;
  invalid.values[0] = std::numeric_limits<double>::infinity();
  ExpectFailure(cached.value().get(), invalid, {0, 0}, ErrorCode::kNonFinite,
                "matrix contains a non-finite value");
  invalid = matrix;
  invalid.values.assign(4, 0.0);
  auto singular = cached.value()->FactorAndSolveRefined(invalid, {0, 0});
  ASSERT_FALSE(singular.ok());
  EXPECT_EQ(singular.error().code, ErrorCode::kSingular);
  auto recovered = cached.value()->FactorAndSolveRefined(matrix, {64, 64});
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  EXPECT_TRUE(ValidateSparseSolution(matrix, {64, 64}, recovered.value()).ok());
  for (const double value : recovered.value())
    EXPECT_NEAR(value, 64.0, 5e-9);
}

TEST(Emi02SolverReuse, ProductMagnitudeIdentityCoversFiniteFp64Extremes) {
  // Independently exercise the arithmetic identity used only in the private
  // validator. Multiplication of two finite FP64 values cannot overflow or
  // underflow the pinned extended-precision format. Sign reflection commutes
  // with round-to-nearest, including signed zero and subnormal FP64 operands.
  ASSERT_EQ(std::fegetround(), FE_TONEAREST);
  ASSERT_GT(std::numeric_limits<long double>::max_exponent,
            2 * std::numeric_limits<double>::max_exponent);
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double normal = std::numeric_limits<double>::min();
  const double maximum = std::numeric_limits<double>::max();
  std::vector<double> values{0.0,
                             -0.0,
                             minimum,
                             -minimum,
                             normal,
                             -normal,
                             maximum,
                             -maximum,
                             1.0,
                             -1.0,
                             0.1,
                             -0.1,
                             std::nextafter(1.0, 2.0),
                             std::nextafter(-1.0, -2.0)};
  std::uint64_t bits = 0x9e3779b97f4a7c15ULL;
  for (std::size_t index = 0; index < 64; ++index) {
    bits = bits * 6364136223846793005ULL + 1442695040888963407ULL;
    const double value = std::bit_cast<double>(bits);
    if (std::isfinite(value))
      values.push_back(value);
  }
  for (const long double coefficient : values) {
    for (const long double variable : values) {
      const long double original = std::abs(coefficient) * std::abs(variable);
      const long double reused = std::abs(coefficient * variable);
      EXPECT_EQ(reused, original);
      EXPECT_EQ(std::signbit(reused), std::signbit(original));
      EXPECT_TRUE(std::isfinite(reused));
    }
  }
}

TEST(Emi02SolverReuse, PrivateProductReuseMatchesPublicExtremeSignedRows) {
  auto matrix = TriangularMatrix();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  for (const int exponent : {-1022, -900, -100, 0, 100, 900, 1020}) {
    for (const double sign : {-1.0, 1.0}) {
      SCOPED_TRACE(exponent);
      SCOPED_TRACE(sign);
      const double scale = sign * std::ldexp(1.0, exponent);
      matrix.values = {4 * scale,   0.1 * scale, -0.0, 0.0,  -2 * scale,
                       0.3 * scale, 0.0,         -0.0, scale};
      // Independent triangular equations prescribe z=1, y=-1/2, x=1/4.
      // Round each forcing only after extended-precision evaluation; the
      // smallest row also exercises subnormal coefficients and corrections.
      const std::vector<double> rhs{
          static_cast<double>(static_cast<long double>(matrix.values[0]) / 4 -
                              static_cast<long double>(matrix.values[1]) / 2),
          static_cast<double>(-static_cast<long double>(matrix.values[4]) / 2 +
                              static_cast<long double>(matrix.values[5])),
          scale};
      auto solved = analyzed.value()->FactorAndSolveRefined(matrix, rhs);
      ASSERT_TRUE(solved.ok()) << solved.error().message;
      auto independent = ValidateSparseSolution(matrix, rhs, solved.value());
      ASSERT_TRUE(independent.ok()) << independent.error().message;
      EXPECT_NEAR(solved.value()[0], 0.25, 1e-14);
      EXPECT_NEAR(solved.value()[1], -0.5, 1e-14);
      EXPECT_DOUBLE_EQ(solved.value()[2], 1.0);
    }
  }
}

TEST(Emi02SolverReuse, DeferredCheckMatchesOriginalCheckedCorrectionSequence) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1e8 + 1, -1e8, -1e8, 1e8 + 1},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  for (const std::size_t budget : {0U, 1U, 4U}) {
    for (const double forcing :
         {128.0, -64.0, 1e100, -1e-280, 1e-250, 0.0, -0.0}) {
      SCOPED_TRACE(budget);
      SCOPED_TRACE(forcing);
      const std::vector<double> rhs{forcing, forcing};
      auto original = SparseRealFactorization::Analyze(matrix);
      auto optimized = SparseRealFactorization::Analyze(matrix);
      ASSERT_TRUE(original.ok());
      ASSERT_TRUE(optimized.ok());
      auto initial = original.value()->FactorAndSolve(matrix, rhs);
      ASSERT_TRUE(initial.ok()) << initial.error().message;
      auto expected = initial.TakeValue();
      // Independent old control flow: full public validation precedes the
      // first correction and follows every correction. Its uncached validator
      // still computes each denominator/norm using the original arithmetic.
      auto checked = ValidateSparseSolution(matrix, rhs, expected);
      std::size_t corrections = 0;
      for (std::size_t attempt = 0;
           attempt < budget && (attempt == 0 || !checked.ok()); ++attempt) {
        std::vector<double> residual(matrix.rows);
        bool nonzero = false;
        for (std::size_t row = 0; row < matrix.rows; ++row) {
          long double product = 0.0L;
          for (std::size_t entry = matrix.row_offsets[row];
               entry < matrix.row_offsets[row + 1]; ++entry) {
            product += static_cast<long double>(matrix.values[entry]) *
                       static_cast<long double>(
                           expected[matrix.column_indices[entry]]);
          }
          residual[row] =
              static_cast<double>(static_cast<long double>(rhs[row]) - product);
          ASSERT_TRUE(std::isfinite(residual[row]));
          nonzero |= residual[row] != 0.0;
        }
        if (!nonzero)
          break;
        auto correction = original.value()->FactorAndSolve(matrix, residual);
        ASSERT_TRUE(correction.ok()) << correction.error().message;
        for (std::size_t index = 0; index < expected.size(); ++index)
          expected[index] += correction.value()[index];
        ++corrections;
        checked = ValidateSparseSolution(matrix, rhs, expected);
      }
      auto actual =
          optimized.value()->FactorAndSolveRefined(matrix, rhs, budget);
      ASSERT_EQ(actual.ok(), checked.ok());
      if (checked.ok()) {
        ExpectBits(actual.value(), expected);
      } else {
        EXPECT_EQ(actual.error().code, checked.error().code);
        EXPECT_EQ(actual.error().message, checked.error().message);
      }
      EXPECT_EQ(optimized.value()->statistics().iterative_refinement_solves,
                corrections);
      EXPECT_EQ(optimized.value()->statistics().numeric_factorizations, 1U);
      EXPECT_EQ(optimized.value()->statistics().numeric_refactorizations, 0U);
    }
  }
}

TEST(Emi02SolverReuse, RoundedZeroCorrectionAndExhaustedBudgetKeepFullGuards) {
  const double minimum = std::numeric_limits<double>::denorm_min();
  CsrMatrix matrix{.rows = 1,
                   .columns = 1,
                   .values = {3 * minimum},
                   .column_indices = {0},
                   .row_offsets = {0, 1}};
  auto original = SparseRealFactorization::Analyze(matrix);
  auto refined = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(original.ok());
  ASSERT_TRUE(refined.ok());
  auto ordinary = original.value()->FactorAndSolve(matrix, {minimum});
  ASSERT_TRUE(ordinary.ok()) << ordinary.error().message;
  const long double residual =
      static_cast<long double>(minimum) -
      static_cast<long double>(matrix.values[0]) * ordinary.value()[0];
  ASSERT_NE(residual, 0.0L);
  ASSERT_EQ(static_cast<double>(residual), 0.0);
  auto solved = refined.value()->FactorAndSolveRefined(matrix, {minimum});
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ExpectBits(solved.value(), ordinary.value());
  EXPECT_TRUE(ValidateSparseSolution(matrix, {minimum}, solved.value()).ok());
  EXPECT_EQ(refined.value()->statistics().iterative_refinement_solves, 0U);

  // The exact scalar solution underflows FP64 to zero. Its finite nonzero
  // residual must consume the complete correction budget, then fail the
  // unchanged componentwise guard even though normwise error is tiny.
  matrix.values[0] = std::numeric_limits<double>::max();
  auto failing = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(failing.ok());
  auto failure = failing.value()->FactorAndSolveRefined(matrix, {minimum}, 4);
  auto checked = ValidateSparseSolution(matrix, {minimum}, {0.0});
  ASSERT_FALSE(failure.ok());
  ASSERT_FALSE(checked.ok());
  EXPECT_EQ(failure.error().code, checked.error().code);
  EXPECT_EQ(failure.error().message, checked.error().message);
  EXPECT_EQ(failing.value()->statistics().iterative_refinement_solves, 4U);
  EXPECT_EQ(failing.value()->statistics().solves, 0U);

  matrix.values[0] = minimum;
  for (const std::size_t budget : {0U, 4U}) {
    auto overflowing = SparseRealFactorization::Analyze(matrix);
    ASSERT_TRUE(overflowing.ok());
    auto invalid = overflowing.value()->FactorAndSolveRefined(
        matrix, {std::numeric_limits<double>::max()}, budget);
    ASSERT_FALSE(invalid.ok());
    EXPECT_EQ(invalid.error().code, ErrorCode::kNonFinite);
    EXPECT_EQ(invalid.error().message,
              "sparse solve produced a non-finite value");
    EXPECT_EQ(overflowing.value()->statistics().iterative_refinement_solves,
              0U);
  }
}

TEST(Emi02SolverReuse, RoundedZeroResidualCannotBypassOriginalValidation) {
  const double minimum = std::numeric_limits<double>::denorm_min();
  CsrMatrix matrix{.rows = 1,
                   .columns = 1,
                   .values = {1.0},
                   .column_indices = {0},
                   .row_offsets = {0, 1}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  auto seeded = analyzed.value()->FactorAndSolveRefined(matrix, {minimum});
  ASSERT_TRUE(seeded.ok());
  ExpectBits(seeded.value(), {minimum});

  matrix.values[0] = 1.5;
  // The exact solution is 2/3 of the smallest subnormal; KLU rounds it to
  // that subnormal. The extended-precision residual is half a subnormal and
  // rounds to signed zero, while the original backward error is exactly 0.2.
  const long double residual =
      static_cast<long double>(minimum) -
      static_cast<long double>(matrix.values[0]) * minimum;
  ASSERT_LT(residual, 0.0L);
  ASSERT_EQ(static_cast<double>(residual), 0.0);
  const auto expected = ValidateSparseSolution(matrix, {minimum}, {minimum});
  ASSERT_FALSE(expected.ok());
  ASSERT_EQ(expected.error().code, ErrorCode::kSolutionValidation);

  auto failed = analyzed.value()->FactorAndSolveRefined(matrix, {minimum}, 4);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, expected.error().code);
  EXPECT_EQ(failed.error().message, expected.error().message);
  // A failed refactored solve still retries with fresh pivoting. Neither
  // rounded-zero attempt may consume a correction or report a solved point.
  EXPECT_EQ(analyzed.value()->statistics().numeric_factorizations, 2U);
  EXPECT_EQ(analyzed.value()->statistics().numeric_refactorizations, 1U);
  EXPECT_EQ(analyzed.value()->statistics().numeric_refactorization_fallbacks,
            1U);
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(analyzed.value()->statistics().solves, 1U);
}

TEST(Emi02SolverReuse,
     PrivateZeroAndUnitRowShortcutsPreserveSignedSparseEquationsAndAdmission) {
  const CsrMatrix matrix{
      .rows = 4,
      .columns = 4,
      .values = {1, -1, 0.0, -0.0, -0.0, 1, 0.0, -0.0, 0.0, -0.0, -1, 0.0, -0.0,
                 0.0, 0.0, .5},
      .column_indices = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
      .row_offsets = {0, 4, 8, 12, 16}};
  auto checked = SparseRealFactorization::Analyze(matrix);
  auto optimized = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(checked.ok());
  ASSERT_TRUE(optimized.ok());
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double large = std::numeric_limits<double>::max() / 4;
  const std::vector<std::vector<double>> truths{
      {1, 1, -.5, .25},
      {minimum, minimum, -minimum, 2 * minimum},
      {large, large, -large, large},
      {0.0, -0.0, 0.0, -0.0},
      {1, 1, 0.0, .25}};
  for (const auto &truth : truths) {
    // Independent triangular equations: x-y=b0, y=b1, -z=b2, w/2=b3.
    // The first row has exact cancellation, not merely zero state. The last
    // case also has a genuine zero denominator in the third row. Every row
    // keeps its stored positive/negative zero coefficients in canonical CSR.
    const std::vector<double> rhs{truth[0] - truth[1], truth[1], -truth[2],
                                  truth[3] / 2};
    auto ordinary = checked.value()->FactorAndSolve(matrix, rhs);
    auto refined = optimized.value()->FactorAndSolveRefined(matrix, rhs, 4);
    ASSERT_TRUE(ordinary.ok()) << ordinary.error().message;
    ASSERT_TRUE(refined.ok()) << refined.error().message;
    ExpectBits(refined.value(), ordinary.value());
    for (std::size_t index = 0; index < truth.size(); ++index) {
      if (truth[index] != 0.0)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(refined.value()[index]),
                  std::bit_cast<std::uint64_t>(truth[index]));
    }
    const auto original_validation =
        ValidateSparseSolution(matrix, rhs, refined.value());
    ASSERT_TRUE(original_validation.ok());
    EXPECT_EQ(original_validation.value(), 0.0);
  }
  EXPECT_EQ(optimized.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(optimized.value()->statistics().numeric_factorizations, 1U);
  EXPECT_EQ(optimized.value()->statistics().numeric_reuses, truths.size() - 1);
  EXPECT_EQ(optimized.value()->statistics().solves, truths.size());
  EXPECT_EQ(matrix.values.size(), 16U);
  EXPECT_TRUE(std::signbit(matrix.values[3]));
  EXPECT_FALSE(std::signbit(matrix.values[2]));

  auto overflow_matrix = matrix;
  overflow_matrix.values.back() = minimum;
  auto overflow = SparseRealFactorization::Analyze(overflow_matrix);
  ASSERT_TRUE(overflow.ok());
  // The overflowing unknown also appears behind stored zero coefficients.
  // Skipping those terms is permitted only after the global finite-state
  // guard, which must still reject before residual validation/refinement.
  const auto rejected = overflow.value()->FactorAndSolveRefined(
      overflow_matrix, {0, 0, 0, std::numeric_limits<double>::max()}, 4);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kNonFinite);
  EXPECT_EQ(rejected.error().message,
            "sparse solve produced a non-finite value");
  EXPECT_EQ(overflow.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(overflow.value()->statistics().solves, 0U);
}

TEST(Emi02SolverReuse,
     MixedExactAndNonzeroResidualRowsMatchOriginalCorrectionBits) {
  const CsrMatrix matrix{
      .rows = 4,
      .columns = 4,
      .values = {1e8 + 1, -1e8, 0.0, -0.0, -1e8, 1e8 + 1, -0.0, 0.0, 0.0, -0.0,
                 1, 0.0, -0.0, 0.0, 0.0, -1},
      .column_indices = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
      .row_offsets = {0, 4, 8, 12, 16}};
  const std::vector<double> rhs{128, 128, 1, 0};
  auto original = SparseRealFactorization::Analyze(matrix);
  auto optimized = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(original.ok());
  ASSERT_TRUE(optimized.ok());
  auto initial = original.value()->FactorAndSolve(matrix, rhs);
  ASSERT_TRUE(initial.ok());
  auto expected = initial.value();
  std::vector<double> residual(4);
  // This independent oracle retains every signed-zero multiplication and the
  // old residual traversal. Weak coupled rows require a correction, while the
  // two unit rows have exact zero residuals in that same validation pass.
  for (std::size_t row = 0; row < 4; ++row) {
    long double product = 0;
    for (std::size_t entry = matrix.row_offsets[row];
         entry < matrix.row_offsets[row + 1]; ++entry)
      product +=
          static_cast<long double>(matrix.values[entry]) *
          static_cast<long double>(expected[matrix.column_indices[entry]]);
    residual[row] =
        static_cast<double>(static_cast<long double>(rhs[row]) - product);
  }
  EXPECT_NE(residual[0], 0.0);
  EXPECT_NE(residual[1], 0.0);
  EXPECT_EQ(residual[2], 0.0);
  EXPECT_EQ(residual[3], 0.0);
  auto correction = original.value()->FactorAndSolve(matrix, residual);
  ASSERT_TRUE(correction.ok());
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] += correction.value()[index];
  ASSERT_TRUE(ValidateSparseSolution(matrix, rhs, expected).ok());
  auto actual = optimized.value()->FactorAndSolveRefined(matrix, rhs, 1);
  ASSERT_TRUE(actual.ok()) << actual.error().message;
  ExpectBits(actual.value(), expected);
  EXPECT_EQ(optimized.value()->statistics().iterative_refinement_solves, 1U);
  EXPECT_EQ(optimized.value()->statistics().solves, 1U);
}

TEST(Emi02SolverReuse, ComponentwiseCertificationKeepsBothSubnormalThresholds) {
  ASSERT_EQ(std::fegetround(), FE_TONEAREST);
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double tolerance = kSparseComponentwiseBackwardErrorTolerance;
  std::vector<double> coefficients{1.0};
  // A scalar solution rounds to denorm_min throughout this range. Its true
  // relative backward error is (a-1)/(a+1); an independent unit row makes the
  // normwise guard harmless, so only the componentwise threshold decides.
  for (const double error :
       {kSparseBackwardErrorTolerance / 4, kSparseBackwardErrorTolerance / 2,
        .75 * kSparseBackwardErrorTolerance, kSparseBackwardErrorTolerance,
        2 * kSparseBackwardErrorTolerance, tolerance / 4, tolerance / 2,
        .75 * tolerance, tolerance, 2 * tolerance}) {
    const double center = 1 + 2 * error / (1 - error);
    coefficients.push_back(std::nextafter(center, 1.0));
    coefficients.push_back(center);
    coefficients.push_back(std::nextafter(center, 2.0));
  }
  coefficients.push_back(1.5);
  for (const std::size_t budget : {0U, 4U}) {
    CsrMatrix matrix{.rows = 2,
                     .columns = 2,
                     .values = {1, 1},
                     .column_indices = {0, 1},
                     .row_offsets = {0, 1, 2}};
    auto analyzed = SparseRealFactorization::Analyze(matrix);
    ASSERT_TRUE(analyzed.ok());
    bool certified_nonzero = false;
    bool uncertain_pass = false;
    bool below_half = false;
    bool above_half = false;
    bool below_gate = false;
    bool above_gate = false;
    for (const double coefficient : coefficients) {
      for (const double sign : {-1.0, 1.0}) {
        SCOPED_TRACE(budget);
        SCOPED_TRACE(coefficient);
        SCOPED_TRACE(sign);
        matrix.values[0] = coefficient;
        const std::vector<double> rhs{sign * minimum, 1};
        const long double relative_error =
            (static_cast<long double>(coefficient) - 1) /
            (static_cast<long double>(coefficient) + 1);
        certified_nonzero |= relative_error > 0 &&
                             relative_error < kSparseBackwardErrorTolerance / 2;
        uncertain_pass |= relative_error > kSparseBackwardErrorTolerance / 2 &&
                          relative_error < kSparseBackwardErrorTolerance;
        below_half |= relative_error > 0 && relative_error < tolerance / 2;
        above_half |=
            relative_error > tolerance / 2 && relative_error < tolerance;
        below_gate |=
            relative_error > .99 * tolerance && relative_error <= tolerance;
        above_gate |=
            relative_error > tolerance && relative_error < 1.01 * tolerance;
        const auto independent = ValidateSparseSolution(matrix, rhs, rhs);
        EXPECT_EQ(independent.ok(),
                  static_cast<double>(relative_error) <= tolerance);
        const auto actual =
            analyzed.value()->FactorAndSolveRefined(matrix, rhs, budget);
        ASSERT_EQ(actual.ok(), independent.ok());
        if (actual.ok()) {
          ExpectBits(actual.value(), rhs);
          if (coefficient > 1)
            EXPECT_GT(independent.value(), 0.0);
        } else {
          EXPECT_EQ(actual.error().code, independent.error().code);
          EXPECT_EQ(actual.error().message, independent.error().message);
        }
      }
    }
    EXPECT_TRUE(certified_nonzero);
    EXPECT_TRUE(uncertain_pass);
    EXPECT_TRUE(below_half);
    EXPECT_TRUE(above_half);
    EXPECT_TRUE(below_gate);
    EXPECT_TRUE(above_gate);
    // Even the rejected cases have a residual that rounds to FP64 zero.
    // Certification cannot turn those failures into successful solves or
    // consume correction budget to hide them.
    EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
    EXPECT_GT(analyzed.value()->statistics().numeric_refactorization_fallbacks,
              0U);
  }
}

TEST(Emi02SolverReuse, UncertainRowsKeepExactNormwiseFailureDiagnostics) {
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double coefficient = 1 + 2e-7;
  const CsrMatrix matrix{.rows = 1,
                         .columns = 1,
                         .values = {coefficient},
                         .column_indices = {0},
                         .row_offsets = {0, 1}};
  const double relative_error = (coefficient - 1) / (coefficient + 1);
  ASSERT_GT(relative_error, kSparseBackwardErrorTolerance);
  ASSERT_LT(relative_error, kSparseComponentwiseBackwardErrorTolerance / 2);
  for (const std::size_t budget : {0U, 4U}) {
    auto analyzed = SparseRealFactorization::Analyze(matrix);
    ASSERT_TRUE(analyzed.ok());
    for (const double sign : {-1.0, 1.0}) {
      const std::vector<double> rhs{sign * minimum};
      const auto independent = ValidateSparseSolution(matrix, rhs, rhs);
      ASSERT_FALSE(independent.ok());
      const auto actual =
          analyzed.value()->FactorAndSolveRefined(matrix, rhs, budget);
      ASSERT_FALSE(actual.ok());
      EXPECT_EQ(actual.error().code, independent.error().code);
      // This uncertain row must report both original nonzero error estimates;
      // passing the looser componentwise tolerance cannot hide normwise
      // failure.
      EXPECT_EQ(actual.error().message, independent.error().message);
      EXPECT_EQ(actual.error().code, ErrorCode::kSolutionValidation);
    }
    EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
    EXPECT_EQ(analyzed.value()->statistics().solves, 0U);
  }
}

TEST(Emi02SolverReuse, ComponentwiseCertificationPreservesZeroDenominators) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1, -0.0, 0.0, 2},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  for (const double zero : {-0.0, 0.0}) {
    // The first row has denominator zero in a nonzero overall system, so the
    // all-zero solve shortcut cannot hide the per-row zero-denominator case.
    const std::vector<double> rhs{zero, 2};
    const auto actual = analyzed.value()->FactorAndSolveRefined(matrix, rhs);
    ASSERT_TRUE(actual.ok()) << actual.error().message;
    EXPECT_EQ(actual.value()[0], 0.0);
    EXPECT_EQ(actual.value()[1], 1.0);
    const auto independent =
        ValidateSparseSolution(matrix, rhs, actual.value());
    ASSERT_TRUE(independent.ok());
    EXPECT_EQ(independent.value(), 0.0);
  }
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(analyzed.value()->statistics().solves, 2U);
}

TEST(Emi02SolverReuse, CertificationCoversExtremeProductExponentRange) {
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double maximum = std::numeric_limits<double>::max();
  CsrMatrix matrix{.rows = 2,
                   .columns = 2,
                   .values = {1, -1, 1},
                   .column_indices = {0, 1, 1},
                   .row_offsets = {0, 2, 3}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok());
  for (const double magnitude : {minimum, maximum}) {
    matrix.values[0] = magnitude;
    matrix.values[1] = -magnitude;
    for (const double sign : {-1.0, 1.0}) {
      // The independent equations are m*(x-y)=0 and y=sign*m. The
      // componentwise denominator therefore includes two m*m products,
      // extending far below/above the FP64 representable exponent range.
      const std::vector<double> rhs{0.0, sign * magnitude};
      const auto actual = analyzed.value()->FactorAndSolveRefined(matrix, rhs);
      ASSERT_TRUE(actual.ok()) << actual.error().message;
      ExpectBits(actual.value(), {sign * magnitude, sign * magnitude});
      const auto independent =
          ValidateSparseSolution(matrix, rhs, actual.value());
      ASSERT_TRUE(independent.ok()) << independent.error().message;
      EXPECT_EQ(independent.value(), 0.0);
    }
  }
  EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_EQ(analyzed.value()->statistics().solves, 4U);
}

TEST(Emi02SolverReuse, AllRowCertificateRetainsTheTighterNormwiseBoundary) {
  const double minimum = std::numeric_limits<double>::denorm_min();
  for (const std::size_t budget : {0U, 4U}) {
    CsrMatrix matrix{.rows = 1,
                     .columns = 1,
                     .values = {1},
                     .column_indices = {0},
                     .row_offsets = {0, 1}};
    auto analyzed = SparseRealFactorization::Analyze(matrix);
    ASSERT_TRUE(analyzed.ok());
    bool accepted_uncertain = false;
    bool rejected_uncertain = false;
    for (const double target :
         {kSparseBackwardErrorTolerance / 4, kSparseBackwardErrorTolerance / 2,
          kSparseBackwardErrorTolerance}) {
      const double center = 1 + 2 * target / (1 - target);
      for (const double coefficient :
           {std::nextafter(center, 1.0), center, std::nextafter(center, 2.0)}) {
        for (const double sign : {-1.0, 1.0}) {
          SCOPED_TRACE(coefficient);
          SCOPED_TRACE(sign);
          matrix.values[0] = coefficient;
          const std::vector<double> rhs{sign * minimum};
          const long double relative_error =
              (static_cast<long double>(coefficient) - 1) /
              (static_cast<long double>(coefficient) + 1);
          const auto independent = ValidateSparseSolution(matrix, rhs, rhs);
          const auto actual =
              analyzed.value()->FactorAndSolveRefined(matrix, rhs, budget);
          ASSERT_EQ(actual.ok(), independent.ok());
          if (actual.ok()) {
            ExpectBits(actual.value(), rhs);
            EXPECT_GT(independent.value(), 0.0);
            accepted_uncertain |=
                relative_error > kSparseBackwardErrorTolerance / 2;
          } else {
            EXPECT_EQ(actual.error().code, independent.error().code);
            EXPECT_EQ(actual.error().message, independent.error().message);
            rejected_uncertain = true;
          }
        }
      }
    }
    EXPECT_TRUE(accepted_uncertain);
    EXPECT_TRUE(rejected_uncertain);
    EXPECT_EQ(analyzed.value()->statistics().iterative_refinement_solves, 0U);
    EXPECT_GT(analyzed.value()->statistics().numeric_refactorization_fallbacks,
              0U);
  }
}

} // namespace
} // namespace ohmnivore
