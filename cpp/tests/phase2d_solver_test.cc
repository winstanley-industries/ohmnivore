#include "cpp/tests/google_test.h"

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "SuiteSparse_config.h"
#include "cpp/tests/dense_oracle.h"
#include "ohmnivore/compiler.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

void ExpectNear(const std::vector<double> &actual,
                const std::vector<double> &expected, double tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance);
  }
}

void ExpectNear(const std::vector<std::complex<double>> &actual,
                const std::vector<std::complex<double>> &expected,
                double tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index].real(), expected[index].real(), tolerance);
    EXPECT_NEAR(actual[index].imag(), expected[index].imag(), tolerance);
  }
}

[[nodiscard]] MnaSystem Compile(const std::string &netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.error().message;
    return {};
  }
  auto compiled = CompileMna(parsed.value());
  if (!compiled.ok()) {
    ADD_FAILURE() << compiled.error().message;
    return {};
  }
  return compiled.TakeValue();
}

void *RejectAllocation(std::size_t) { return nullptr; }
void *RejectCalloc(std::size_t, std::size_t) { return nullptr; }
void *RejectReallocation(void *, std::size_t) { return nullptr; }

TEST(Phase2DSolverTest, ConvertsCanonicalCsrToDeterministicCscExactly) {
  const CsrMatrix matrix{
      .rows = 3,
      .columns = 3,
      .values = {1.0, 0.0, 2.0, 3.0, 4.0},
      .column_indices = {0, 2, 0, 1, 1},
      .row_offsets = {0, 2, 4, 5},
  };
  auto converted = ConvertCsrToSolverCsc(matrix);
  ASSERT_TRUE(converted.ok()) << converted.error().message;
  EXPECT_EQ(converted.value().size, 3U);
  EXPECT_EQ(converted.value().column_offsets,
            (std::vector<std::int32_t>{0, 2, 4, 5}));
  EXPECT_EQ(converted.value().row_indices,
            (std::vector<std::int32_t>{0, 1, 1, 2, 0}));
  EXPECT_EQ(converted.value().csr_value_indices,
            (std::vector<std::size_t>{0, 2, 3, 4, 1}));

  ComplexCsrMatrix complex{
      .rows = matrix.rows,
      .columns = matrix.columns,
      .values = {{1.0, 2.0}, {0.0, 0.0}, {2.0, -1.0}, {3.0, 0.0}, {4.0, 5.0}},
      .column_indices = matrix.column_indices,
      .row_offsets = matrix.row_offsets,
  };
  auto repeated = ConvertCsrToSolverCsc(complex);
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  EXPECT_EQ(repeated.value().column_offsets, converted.value().column_offsets);
  EXPECT_EQ(repeated.value().row_indices, converted.value().row_indices);
  EXPECT_EQ(repeated.value().csr_value_indices,
            converted.value().csr_value_indices);
}

TEST(Phase2DSolverTest, SolvesEmptyScalarDiagonalAsymmetricAndPivotSystems) {
  const CsrMatrix empty{.rows = 0,
                        .columns = 0,
                        .values = {},
                        .column_indices = {},
                        .row_offsets = {0}};
  auto empty_solution = SolveSparseReal(empty, {});
  ASSERT_TRUE(empty_solution.ok()) << empty_solution.error().message;
  EXPECT_TRUE(empty_solution.value().empty());

  const CsrMatrix scalar{.rows = 1,
                         .columns = 1,
                         .values = {5.0},
                         .column_indices = {0},
                         .row_offsets = {0, 1}};
  auto scalar_solution = SolveSparseReal(scalar, {15.0});
  ASSERT_TRUE(scalar_solution.ok()) << scalar_solution.error().message;
  EXPECT_EQ(scalar_solution.value(), (std::vector<double>{3.0}));

  const CsrMatrix diagonal{
      .rows = 3,
      .columns = 3,
      .values = {2.0, -4.0, 0.5},
      .column_indices = {0, 1, 2},
      .row_offsets = {0, 1, 2, 3},
  };
  auto diagonal_solution = SolveSparseReal(diagonal, {4.0, 8.0, 1.0});
  ASSERT_TRUE(diagonal_solution.ok()) << diagonal_solution.error().message;
  EXPECT_EQ(diagonal_solution.value(), (std::vector<double>{2.0, -2.0, 2.0}));

  const CsrMatrix pivot{
      .rows = 3,
      .columns = 3,
      .values = {2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0},
      .column_indices = {1, 2, 0, 1, 2, 0, 1},
      .row_offsets = {0, 2, 5, 7},
  };
  auto sparse = SolveSparseReal(pivot, {5.0, 4.0, 4.0});
  auto dense = SolveDenseOracleReal(pivot, {5.0, 4.0, 4.0});
  ASSERT_TRUE(sparse.ok()) << sparse.error().message;
  ASSERT_TRUE(dense.ok()) << dense.error().message;
  ExpectNear(sparse.value(), dense.value(), 1e-14);
}

TEST(Phase2DSolverTest, MatchesIndependentDenseOracleForRealAndComplex) {
  const CsrMatrix real{
      .rows = 4,
      .columns = 4,
      .values = {4.0, -1.0, 1.0, 3.0, -1.0, 2.0, 5.0, 1.0, -2.0, 4.0},
      .column_indices = {0, 2, 0, 1, 3, 1, 2, 0, 2, 3},
      .row_offsets = {0, 2, 5, 7, 10},
  };
  const std::vector<double> real_rhs = {2.0, -1.0, 7.0, 3.0};
  auto sparse_real = SolveSparseReal(real, real_rhs);
  auto dense_real = SolveDenseOracleReal(real, real_rhs);
  ASSERT_TRUE(sparse_real.ok()) << sparse_real.error().message;
  ASSERT_TRUE(dense_real.ok()) << dense_real.error().message;
  ExpectNear(sparse_real.value(), dense_real.value(), 1e-13);

  const ComplexCsrMatrix complex{
      .rows = 3,
      .columns = 3,
      .values = {{0.0, 2.0},
                 {1.0, 0.0},
                 {1.0, -1.0},
                 {3.0, 0.0},
                 {2.0, 1.0},
                 {4.0, -2.0}},
      .column_indices = {1, 2, 0, 1, 1, 2},
      .row_offsets = {0, 2, 4, 6},
  };
  const std::vector<std::complex<double>> complex_rhs = {
      {1.0, 2.0}, {3.0, -1.0}, {4.0, 5.0}};
  auto sparse_complex = SolveSparseComplex(complex, complex_rhs);
  auto dense_complex = SolveDenseOracleComplex(complex, complex_rhs);
  ASSERT_TRUE(sparse_complex.ok()) << sparse_complex.error().message;
  ASSERT_TRUE(dense_complex.ok()) << dense_complex.error().message;
  ExpectNear(sparse_complex.value(), dense_complex.value(), 1e-13);
}

TEST(Phase2DSolverTest,
     ReusesSymbolicAnalysisAndNumericFactorsDeterministically) {
  CsrMatrix matrix{
      .rows = 3,
      .columns = 3,
      .values = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0},
      .column_indices = {0, 1, 0, 1, 2, 1, 2},
      .row_offsets = {0, 2, 5, 7},
  };
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  std::unique_ptr<SparseRealFactorization> factorization = analyzed.TakeValue();
  auto first = factorization->FactorAndSolve(matrix, {1.0, 2.0, 3.0});
  ASSERT_TRUE(first.ok()) << first.error().message;
  matrix.values = {5.0, -1.0, -1.0, 5.0, -1.0, -1.0, 5.0};
  auto refactored = factorization->FactorAndSolve(matrix, {1.0, 2.0, 3.0});
  ASSERT_TRUE(refactored.ok()) << refactored.error().message;
  auto reused = factorization->FactorAndSolve(matrix, {3.0, 2.0, 1.0});
  ASSERT_TRUE(reused.ok()) << reused.error().message;
  const SparseSolverStatistics &statistics = factorization->statistics();
  EXPECT_EQ(statistics.symbolic_analyses, 1U);
  EXPECT_EQ(statistics.numeric_factorizations, 1U);
  EXPECT_EQ(statistics.numeric_refactorizations, 1U);
  EXPECT_EQ(statistics.numeric_reuses, 1U);
  EXPECT_EQ(statistics.solves, 3U);

  auto repeated = factorization->FactorAndSolve(matrix, {3.0, 2.0, 1.0});
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  ASSERT_EQ(reused.value().size(), repeated.value().size());
  for (std::size_t index = 0; index < reused.value().size(); ++index) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(reused.value()[index]),
              std::bit_cast<std::uint64_t>(repeated.value()[index]));
  }
}

TEST(Phase2DSolverTest, ReusesOneComplexUnionPatternAcrossNumericChanges) {
  const CsrMatrix g{.rows = 2,
                    .columns = 2,
                    .values = {2.0, 1.0},
                    .column_indices = {0, 1},
                    .row_offsets = {0, 1, 2}};
  const CsrMatrix c{.rows = 2,
                    .columns = 2,
                    .values = {3.0, 4.0},
                    .column_indices = {1, 0},
                    .row_offsets = {0, 1, 2}};
  auto zero = FormAcMatrix(g, c, 0.0);
  auto first = FormAcMatrix(g, c, 1.0);
  auto second = FormAcMatrix(g, c, 10.0);
  ASSERT_TRUE(zero.ok());
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(zero.value().row_offsets, first.value().row_offsets);
  EXPECT_EQ(zero.value().column_indices, first.value().column_indices);
  EXPECT_EQ(first.value().row_offsets, second.value().row_offsets);
  EXPECT_EQ(first.value().column_indices, second.value().column_indices);
  ASSERT_EQ(zero.value().values.size(), 4U);
  EXPECT_EQ(zero.value().values[1], std::complex<double>(0.0, 0.0));

  auto analyzed = SparseComplexFactorization::Analyze(first.value());
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  auto factorization = analyzed.TakeValue();
  auto first_solution =
      factorization->FactorAndSolve(first.value(), {{1.0, 0.0}, {2.0, 0.0}});
  auto second_solution =
      factorization->FactorAndSolve(second.value(), {{1.0, 0.0}, {2.0, 0.0}});
  ASSERT_TRUE(first_solution.ok()) << first_solution.error().message;
  ASSERT_TRUE(second_solution.ok()) << second_solution.error().message;
  EXPECT_EQ(factorization->statistics().symbolic_analyses, 1U);
  EXPECT_EQ(factorization->statistics().numeric_refactorizations, 1U);
}

TEST(Phase2DSolverTest,
     ReusesSymbolicAnalysisButRepivotsChangedRealAndComplexValues) {
  CsrMatrix real{
      .rows = 2,
      .columns = 2,
      .values = {2.0, 1.0, 1.0, 2.0},
      .column_indices = {0, 1, 0, 1},
      .row_offsets = {0, 2, 4},
  };
  auto real_analysis = SparseRealFactorization::Analyze(real);
  ASSERT_TRUE(real_analysis.ok()) << real_analysis.error().message;
  auto real_factorization = real_analysis.TakeValue();
  auto real_initial = real_factorization->FactorAndSolve(real, {4.0, 5.0});
  ASSERT_TRUE(real_initial.ok()) << real_initial.error().message;
  real.values = {0.0, 1.0, 1.0, 0.0};
  auto real_repivoted = real_factorization->FactorAndSolve(real, {2.0, 1.0});
  ASSERT_TRUE(real_repivoted.ok()) << real_repivoted.error().message;
  EXPECT_EQ(real_repivoted.value(), (std::vector<double>{1.0, 2.0}));
  EXPECT_EQ(real_factorization->statistics().symbolic_analyses, 1U);
  EXPECT_EQ(real_factorization->statistics().numeric_factorizations, 2U);
  EXPECT_EQ(real_factorization->statistics().numeric_refactorization_fallbacks,
            1U);

  ComplexCsrMatrix complex{
      .rows = 2,
      .columns = 2,
      .values = {{2.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}},
      .column_indices = {0, 1, 0, 1},
      .row_offsets = {0, 2, 4},
  };
  auto complex_analysis = SparseComplexFactorization::Analyze(complex);
  ASSERT_TRUE(complex_analysis.ok()) << complex_analysis.error().message;
  auto complex_factorization = complex_analysis.TakeValue();
  auto complex_initial =
      complex_factorization->FactorAndSolve(complex, {{0.0, 2.5}, {-3.0, 2.0}});
  ASSERT_TRUE(complex_initial.ok()) << complex_initial.error().message;
  complex.values = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
  const std::vector<std::complex<double>> changed_rhs = {{-2.0, 0.5},
                                                         {1.0, 1.0}};
  auto complex_repivoted =
      complex_factorization->FactorAndSolve(complex, changed_rhs);
  ASSERT_TRUE(complex_repivoted.ok()) << complex_repivoted.error().message;
  EXPECT_EQ(complex_repivoted.value(),
            (std::vector<std::complex<double>>{{1.0, 1.0}, {-2.0, 0.5}}));
  EXPECT_EQ(complex_factorization->statistics().symbolic_analyses, 1U);
  EXPECT_EQ(complex_factorization->statistics().numeric_factorizations, 2U);
  EXPECT_EQ(
      complex_factorization->statistics().numeric_refactorization_fallbacks,
      1U);

  auto complex_repeated =
      complex_factorization->FactorAndSolve(complex, changed_rhs);
  ASSERT_TRUE(complex_repeated.ok()) << complex_repeated.error().message;
  ASSERT_EQ(complex_repivoted.value().size(), complex_repeated.value().size());
  for (std::size_t index = 0; index < complex_repeated.value().size();
       ++index) {
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(complex_repivoted.value()[index].real()),
        std::bit_cast<std::uint64_t>(complex_repeated.value()[index].real()));
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(complex_repivoted.value()[index].imag()),
        std::bit_cast<std::uint64_t>(complex_repeated.value()[index].imag()));
  }
}

TEST(Phase2DSolverTest, SolvesCircuitDerivedDcAcBackwardEulerAndTrapMatrices) {
  MnaSystem system = Compile(R"(V1 in 0 DC 5 AC 1
R1 in mid 10
L1 mid out 1m
C1 out 0 1u
.OP
.AC LIN 2 1k 2k
.TRAN 1u 2u UIC
)");
  auto sparse_dc = SolveSparseReal(system.g, system.b_dc);
  auto dense_dc = SolveDenseOracleReal(system.g, system.b_dc);
  ASSERT_TRUE(sparse_dc.ok()) << sparse_dc.error().message;
  ASSERT_TRUE(dense_dc.ok()) << dense_dc.error().message;
  ExpectNear(sparse_dc.value(), dense_dc.value(), 1e-12);

  auto ac = FormAcMatrix(system.g, system.c, 6283.185307179586);
  ASSERT_TRUE(ac.ok()) << ac.error().message;
  auto sparse_ac = SolveSparseComplex(ac.value(), system.b_ac);
  auto dense_ac = SolveDenseOracleComplex(ac.value(), system.b_ac);
  ASSERT_TRUE(sparse_ac.ok()) << sparse_ac.error().message;
  ASSERT_TRUE(dense_ac.ok()) << dense_ac.error().message;
  ExpectNear(sparse_ac.value(), dense_ac.value(), 1e-12);

  const std::vector<double> previous(system.g.rows, 0.0);
  auto transient_rhs = BuildTransientRhs(system, 1e-6);
  ASSERT_TRUE(transient_rhs.ok()) << transient_rhs.error().message;
  for (const double alpha : {1.0, 2.0}) {
    auto matrix = FormTransientCompanionMatrix(system.g, system.c, 1e-6, alpha);
    ASSERT_TRUE(matrix.ok()) << matrix.error().message;
    Result<std::vector<double>> rhs =
        alpha == 1.0 ? BuildBackwardEulerRhs(system.c, previous,
                                             transient_rhs.value(), 1e-6)
                     : BuildTrapezoidalRhs(system.g, system.c, previous,
                                           transient_rhs.value(),
                                           transient_rhs.value(), 1e-6);
    ASSERT_TRUE(rhs.ok()) << rhs.error().message;
    auto sparse = SolveSparseReal(matrix.value(), rhs.value());
    auto dense = SolveDenseOracleReal(matrix.value(), rhs.value());
    ASSERT_TRUE(sparse.ok()) << sparse.error().message;
    ASSERT_TRUE(dense.ok()) << dense.error().message;
    ExpectNear(sparse.value(), dense.value(), 1e-12);
  }
}

TEST(Phase2DSolverTest, AcceptsBadlyScaledNonsingularMixedUnitMnaSystem) {
  const CsrMatrix matrix{
      .rows = 4,
      .columns = 4,
      .values = {1e-12, 1.0, 1e12, -1.0, 1.0, 1e-9, -1.0, 1e9},
      .column_indices = {0, 2, 1, 3, 0, 2, 1, 3},
      .row_offsets = {0, 2, 4, 6, 8},
  };
  const std::vector<double> expected = {1.0, -2.0, 3.0, -4.0};
  std::vector<double> rhs(matrix.rows, 0.0);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      rhs[row] += matrix.values[index] * expected[matrix.column_indices[index]];
    }
  }
  auto solved = SolveSparseReal(matrix, rhs);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ExpectNear(solved.value(), expected, 1e-10);
}

TEST(Phase2DSolverTest, ReturnsTypedFailuresAndNeverDenseFallbacks) {
  const CsrMatrix structurally_singular{
      .rows = 2,
      .columns = 2,
      .values = {1.0},
      .column_indices = {0},
      .row_offsets = {0, 1, 1},
  };
  auto structural = SolveSparseReal(structurally_singular, {1.0, 0.0});
  ASSERT_FALSE(structural.ok());
  EXPECT_EQ(structural.error().code, ErrorCode::kSingular);

  const CsrMatrix rank_deficient{
      .rows = 2,
      .columns = 2,
      .values = {1.0, 2.0, 2.0, 4.0},
      .column_indices = {0, 1, 0, 1},
      .row_offsets = {0, 2, 4},
  };
  auto numeric = SolveSparseReal(rank_deficient, {3.0, 6.0});
  ASSERT_FALSE(numeric.ok());
  EXPECT_EQ(numeric.error().code, ErrorCode::kSingular);

  const CsrMatrix duplicate{
      .rows = 1,
      .columns = 1,
      .values = {1.0, 2.0},
      .column_indices = {0, 0},
      .row_offsets = {0, 2},
  };
  auto malformed = SolveSparseReal(duplicate, {1.0});
  ASSERT_FALSE(malformed.ok());
  EXPECT_EQ(malformed.error().code, ErrorCode::kInvalidStructure);

  const std::size_t oversized =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1U;
  const CsrMatrix too_large{.rows = oversized,
                            .columns = oversized,
                            .values = {},
                            .column_indices = {},
                            .row_offsets = {}};
  auto unsupported = SolveSparseReal(too_large, {});
  ASSERT_FALSE(unsupported.ok());
  EXPECT_EQ(unsupported.error().code, ErrorCode::kUnsupportedSize);

  CsrMatrix non_finite{.rows = 1,
                       .columns = 1,
                       .values = {std::numeric_limits<double>::infinity()},
                       .column_indices = {0},
                       .row_offsets = {0, 1}};
  auto matrix_failure = SolveSparseReal(non_finite, {1.0});
  ASSERT_FALSE(matrix_failure.ok());
  EXPECT_EQ(matrix_failure.error().code, ErrorCode::kNonFinite);
  non_finite.values[0] = 1.0;
  auto rhs_failure =
      SolveSparseReal(non_finite, {std::numeric_limits<double>::quiet_NaN()});
  ASSERT_FALSE(rhs_failure.ok());
  EXPECT_EQ(rhs_failure.error().code, ErrorCode::kNonFinite);
}

TEST(Phase2DSolverTest, ReturnsTypedNumericFactorizationFailure) {
  const CsrMatrix identity{.rows = 2,
                           .columns = 2,
                           .values = {1.0, 1.0},
                           .column_indices = {0, 1},
                           .row_offsets = {0, 1, 2}};
  auto analyzed = SparseRealFactorization::Analyze(identity);
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  auto factorization = analyzed.TakeValue();

  const auto original_malloc = SuiteSparse_config_malloc_func_get();
  const auto original_calloc = SuiteSparse_config_calloc_func_get();
  const auto original_realloc = SuiteSparse_config_realloc_func_get();
  SuiteSparse_config_malloc_func_set(RejectAllocation);
  SuiteSparse_config_calloc_func_set(RejectCalloc);
  SuiteSparse_config_realloc_func_set(RejectReallocation);
  auto rejected = factorization->FactorAndSolve(identity, {1.0, 2.0});
  SuiteSparse_config_malloc_func_set(original_malloc);
  SuiteSparse_config_calloc_func_set(original_calloc);
  SuiteSparse_config_realloc_func_set(original_realloc);

  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kFactorization);
}

TEST(Phase2DSolverTest,
     PreservesTypedMalformedAndNonFiniteFailuresForUicAndNonUic) {
  const MnaSystem valid = Compile(R"(V1 in 0 DC 1
R1 in out 1k
C1 out 0 1u
.TRAN 1u 2u UIC
)");

  MnaSystem malformed = valid;
  malformed.g.row_offsets.pop_back();
  for (const bool use_initial_conditions : {false, true}) {
    auto rejected =
        BuildTransientInitialState(malformed, use_initial_conditions);
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidStructure);
  }

  MnaSystem non_finite = valid;
  non_finite.b_dc[0] = std::numeric_limits<double>::infinity();
  for (const bool use_initial_conditions : {false, true}) {
    auto rejected =
        BuildTransientInitialState(non_finite, use_initial_conditions);
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.error().code, ErrorCode::kNonFinite);
  }
}

TEST(Phase2DSolverTest, ValidatesFiniteResultsAndScaledBackwardError) {
  const CsrMatrix identity{.rows = 2,
                           .columns = 2,
                           .values = {1.0, 1.0},
                           .column_indices = {0, 1},
                           .row_offsets = {0, 1, 2}};
  auto valid = ValidateSparseSolution(identity, {1.0, -2.0}, {1.0, -2.0});
  ASSERT_TRUE(valid.ok()) << valid.error().message;
  EXPECT_EQ(valid.value(), 0.0);

  auto hostile = ValidateSparseSolution(identity, {1.0, -2.0}, {2.0, -2.0});
  ASSERT_FALSE(hostile.ok());
  EXPECT_EQ(hostile.error().code, ErrorCode::kSolutionValidation);
  EXPECT_NE(hostile.error().message.find("tolerance=1e-10"), std::string::npos);

  auto mixed_scale_hostile =
      ValidateSparseSolution(identity, {0.0, 1e12}, {1.0, 1e12});
  ASSERT_FALSE(mixed_scale_hostile.ok());
  EXPECT_EQ(mixed_scale_hostile.error().code, ErrorCode::kSolutionValidation);

  const ComplexCsrMatrix complex_identity{
      .rows = 2,
      .columns = 2,
      .values = {{1.0, 0.0}, {1.0, 0.0}},
      .column_indices = {0, 1},
      .row_offsets = {0, 1, 2},
  };
  auto complex_mixed_scale_hostile =
      ValidateSparseSolution(complex_identity, {{0.0, 0.0}, {1e12, -1e12}},
                             {{1.0, 1.0}, {1e12, -1e12}});
  ASSERT_FALSE(complex_mixed_scale_hostile.ok());
  EXPECT_EQ(complex_mixed_scale_hostile.error().code,
            ErrorCode::kSolutionValidation);

  auto non_finite = ValidateSparseSolution(
      identity, {1.0, -2.0}, {std::numeric_limits<double>::quiet_NaN(), -2.0});
  ASSERT_FALSE(non_finite.ok());
  EXPECT_EQ(non_finite.error().code, ErrorCode::kNonFinite);
}

TEST(Phase2DSolverTest, PreservesSparsePathAnalyticAndCsvContracts) {
  constexpr char netlist[] = R"(V"1 in,node 0 DC 5 AC 1
R1 in,node out 1k
C1 out 0 1u
.OP
.AC LIN 2 10 20
.TRAN 10u 20u UIC
.END
)";
  auto csv = SimulateToCsv(netlist);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  EXPECT_NE(csv.value().find("Variable,Value\n"), std::string::npos);
  EXPECT_NE(csv.value().find("Frequency,\"V(in,node)_mag\""),
            std::string::npos);
  EXPECT_NE(csv.value().find("time,\"V(in,node)\",V(out),\"I(V\"\"1)\""),
            std::string::npos);
  EXPECT_NE(csv.value().find("\n0,"), std::string::npos);
  auto transient = SimulateTransient(netlist);
  ASSERT_TRUE(transient.ok()) << transient.error().message;
  EXPECT_DOUBLE_EQ(transient.value().times_seconds.back(), 20e-6);
}

TEST(Phase2DSolverTest, RepeatsCompleteComplexAcOutputExactly) {
  constexpr char netlist[] = R"(V1 in 0 AC 1
R1 in mid 10
L1 mid out 1m
C1 out 0 1u
.AC LIN 5 10 10000
)";
  auto first = SimulateAc(netlist);
  auto second = SimulateAc(netlist);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  EXPECT_EQ(first.value().frequencies_hz, second.value().frequencies_hz);
  EXPECT_EQ(first.value().node_voltages, second.value().node_voltages);
  EXPECT_EQ(first.value().branch_currents, second.value().branch_currents);
}

} // namespace
} // namespace ohmnivore
