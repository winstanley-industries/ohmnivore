#include <algorithm>
#include <charconv>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/utsname.h>

#include "ohmnivore/solver.h"
#include "ohmnivore/sparse.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::size_t warmups = 3;
  std::size_t repetitions = 15;
};

[[nodiscard]] std::size_t ParseCount(std::string_view text,
                                     std::string_view name) {
  std::size_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value == 0 || value > 10'000) {
    throw std::runtime_error(std::string(name) +
                             " must be an integer in [1, 10000]");
  }
  return value;
}

[[nodiscard]] Options ParseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    constexpr std::string_view kWarmups = "--warmups=";
    constexpr std::string_view kRepetitions = "--repetitions=";
    if (argument.starts_with(kWarmups)) {
      options.warmups = ParseCount(argument.substr(kWarmups.size()), "warmups");
    } else if (argument.starts_with(kRepetitions)) {
      options.repetitions =
          ParseCount(argument.substr(kRepetitions.size()), "repetitions");
    } else {
      throw std::runtime_error("unknown benchmark argument: " +
                               std::string(argument));
    }
  }
  return options;
}

[[nodiscard]] std::string CpuModel() {
  std::ifstream input("/proc/cpuinfo");
  for (std::string line; std::getline(input, line);) {
    constexpr std::string_view kModel = "model name";
    if (line.starts_with(kModel)) {
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) {
        const std::size_t value = line.find_first_not_of(" \t", colon + 1);
        return value == std::string::npos ? "unknown" : line.substr(value);
      }
    }
  }
  return "unknown";
}

[[nodiscard]] std::string KernelIdentity() {
  utsname identity{};
  if (uname(&identity) != 0) {
    return "unknown";
  }
  return std::string(identity.sysname) + " " + identity.release + " " +
         identity.machine;
}

template <typename T>
[[nodiscard]] ohmnivore::CsrMatrixBase<T>
BuildCircuitLadder(std::size_t node_count, T series_admittance,
                   T shunt_admittance) {
  ohmnivore::CsrMatrixBase<T> matrix;
  matrix.rows = node_count + 1;
  matrix.columns = node_count + 1;
  matrix.row_offsets.reserve(matrix.rows + 1);
  matrix.row_offsets.push_back(0);
  for (std::size_t row = 0; row < node_count; ++row) {
    if (row > 0) {
      matrix.column_indices.push_back(row - 1);
      matrix.values.push_back(-series_admittance);
    }
    matrix.column_indices.push_back(row);
    T diagonal = shunt_admittance;
    if (row > 0) {
      diagonal += series_admittance;
    }
    if (row + 1 < node_count) {
      diagonal += series_admittance;
    } else {
      diagonal += series_admittance;
    }
    matrix.values.push_back(diagonal);
    if (row + 1 < node_count) {
      matrix.column_indices.push_back(row + 1);
      matrix.values.push_back(-series_admittance);
    }
    if (row == 0) {
      matrix.column_indices.push_back(node_count);
      matrix.values.push_back(T{1.0});
    }
    matrix.row_offsets.push_back(matrix.values.size());
  }
  matrix.column_indices.push_back(0);
  matrix.values.push_back(T{1.0});
  matrix.row_offsets.push_back(matrix.values.size());
  return matrix;
}

template <typename T>
[[nodiscard]] std::vector<T> KnownSolution(std::size_t size) {
  std::vector<T> solution(size);
  for (std::size_t index = 0; index < size; ++index) {
    if constexpr (std::is_same_v<T, double>) {
      solution[index] = 1.0 + static_cast<double>(index % 17) / 8.0;
    } else {
      solution[index] = T{1.0 + static_cast<double>(index % 17) / 8.0,
                          -0.5 + static_cast<double>(index % 11) / 16.0};
    }
  }
  return solution;
}

template <typename T>
[[nodiscard]] std::vector<T> Multiply(const ohmnivore::CsrMatrixBase<T> &matrix,
                                      const std::vector<T> &solution) {
  std::vector<T> rhs(matrix.rows, T{});
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      rhs[row] += matrix.values[index] * solution[matrix.column_indices[index]];
    }
  }
  return rhs;
}

[[nodiscard]] std::uint64_t ElapsedNanoseconds(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

void PrintSamples(const std::string &name, std::string_view scalar,
                  std::string_view mode, std::size_t size, std::size_t nonzeros,
                  const std::vector<std::uint64_t> &samples) {
  for (std::size_t index = 0; index < samples.size(); ++index) {
    std::cout << "sample," << name << ',' << scalar << ',' << mode << ','
              << size << ',' << nonzeros << ',' << index << ','
              << samples[index] << '\n';
  }
  std::vector<std::uint64_t> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const auto quantile = [&](std::size_t numerator, std::size_t denominator) {
    return sorted[(sorted.size() - 1) * numerator / denominator];
  };
  std::cout << "summary," << name << ',' << scalar << ',' << mode << ',' << size
            << ',' << nonzeros << ',' << sorted.front() << ',' << quantile(1, 4)
            << ',' << quantile(1, 2) << ',' << quantile(3, 4) << ','
            << sorted.back() << '\n';
}

void BenchmarkReal(const std::string &name, ohmnivore::CsrMatrix matrix,
                   const Options &options) {
  const std::vector<double> known = KnownSolution<double>(matrix.rows);
  const std::vector<double> rhs = Multiply(matrix, known);
  std::vector<std::uint64_t> cold;
  for (std::size_t iteration = 0;
       iteration < options.warmups + options.repetitions; ++iteration) {
    const Clock::time_point start = Clock::now();
    auto solved = ohmnivore::SolveSparseReal(matrix, rhs);
    const std::uint64_t elapsed = ElapsedNanoseconds(start);
    if (!solved.ok()) {
      throw std::runtime_error(name +
                               " cold solve failed: " + solved.error().message);
    }
    if (iteration >= options.warmups) {
      cold.push_back(elapsed);
    }
  }
  PrintSamples(name, "real", "analyze_factor_solve", matrix.rows,
               matrix.values.size(), cold);

  auto analyzed = ohmnivore::SparseRealFactorization::Analyze(matrix);
  if (!analyzed.ok()) {
    throw std::runtime_error(name +
                             " analysis failed: " + analyzed.error().message);
  }
  auto factorization = analyzed.TakeValue();
  auto initial = factorization->FactorAndSolve(matrix, rhs);
  if (!initial.ok()) {
    throw std::runtime_error(
        name + " initial factorization failed: " + initial.error().message);
  }
  std::vector<std::uint64_t> refactor;
  for (std::size_t iteration = 0;
       iteration < options.warmups + options.repetitions; ++iteration) {
    matrix.values[0] += 1e-12;
    const std::vector<double> changed_rhs = Multiply(matrix, known);
    const Clock::time_point start = Clock::now();
    auto solved = factorization->FactorAndSolve(matrix, changed_rhs);
    const std::uint64_t elapsed = ElapsedNanoseconds(start);
    if (!solved.ok()) {
      throw std::runtime_error(
          name + " refactor solve failed: " + solved.error().message);
    }
    if (iteration >= options.warmups) {
      refactor.push_back(elapsed);
    }
  }
  const ohmnivore::SparseSolverStatistics &statistics =
      factorization->statistics();
  if (statistics.symbolic_analyses != 1 ||
      statistics.numeric_factorizations != 1 ||
      statistics.numeric_refactorizations !=
          options.warmups + options.repetitions ||
      statistics.numeric_refactorization_fallbacks != 0) {
    throw std::runtime_error(name +
                             " did not exercise the expected real symbolic "
                             "reuse/numeric-refactor path");
  }
  PrintSamples(name, "real", "numeric_refactor_solve", matrix.rows,
               matrix.values.size(), refactor);
}

void BenchmarkComplex(const std::string &name,
                      ohmnivore::ComplexCsrMatrix matrix,
                      const Options &options) {
  const std::vector<std::complex<double>> known =
      KnownSolution<std::complex<double>>(matrix.rows);
  const std::vector<std::complex<double>> rhs = Multiply(matrix, known);
  std::vector<std::uint64_t> cold;
  for (std::size_t iteration = 0;
       iteration < options.warmups + options.repetitions; ++iteration) {
    const Clock::time_point start = Clock::now();
    auto solved = ohmnivore::SolveSparseComplex(matrix, rhs);
    const std::uint64_t elapsed = ElapsedNanoseconds(start);
    if (!solved.ok()) {
      throw std::runtime_error(name +
                               " cold solve failed: " + solved.error().message);
    }
    if (iteration >= options.warmups) {
      cold.push_back(elapsed);
    }
  }
  PrintSamples(name, "complex", "analyze_factor_solve", matrix.rows,
               matrix.values.size(), cold);

  auto analyzed = ohmnivore::SparseComplexFactorization::Analyze(matrix);
  if (!analyzed.ok()) {
    throw std::runtime_error(name +
                             " analysis failed: " + analyzed.error().message);
  }
  auto factorization = analyzed.TakeValue();
  auto initial = factorization->FactorAndSolve(matrix, rhs);
  if (!initial.ok()) {
    throw std::runtime_error(
        name + " initial factorization failed: " + initial.error().message);
  }
  std::vector<std::uint64_t> refactor;
  for (std::size_t iteration = 0;
       iteration < options.warmups + options.repetitions; ++iteration) {
    matrix.values[0] += std::complex<double>{1e-12, 1e-13};
    const std::vector<std::complex<double>> changed_rhs =
        Multiply(matrix, known);
    const Clock::time_point start = Clock::now();
    auto solved = factorization->FactorAndSolve(matrix, changed_rhs);
    const std::uint64_t elapsed = ElapsedNanoseconds(start);
    if (!solved.ok()) {
      throw std::runtime_error(
          name + " refactor solve failed: " + solved.error().message);
    }
    if (iteration >= options.warmups) {
      refactor.push_back(elapsed);
    }
  }
  const ohmnivore::SparseSolverStatistics &statistics =
      factorization->statistics();
  if (statistics.symbolic_analyses != 1 ||
      statistics.numeric_factorizations != 1 ||
      statistics.numeric_refactorizations !=
          options.warmups + options.repetitions ||
      statistics.numeric_refactorization_fallbacks != 0) {
    throw std::runtime_error(name +
                             " did not exercise the expected complex symbolic "
                             "reuse/numeric-refactor path");
  }
  PrintSamples(name, "complex", "numeric_refactor_solve", matrix.rows,
               matrix.values.size(), refactor);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    std::cout << "metadata,key,value\n"
              << "metadata,repository_bazel_pin,"
              << OHMNIVORE_REPOSITORY_BAZEL_PIN << '\n'
              << "metadata,compiler," << __VERSION__ << '\n'
              << "metadata,cplusplus," << __cplusplus << '\n'
              << "metadata,kernel," << KernelIdentity() << '\n'
              << "metadata,cpu," << CpuModel() << '\n'
              << "metadata,hardware_threads,"
              << std::thread::hardware_concurrency() << '\n'
              << "metadata,warmups," << options.warmups << '\n'
              << "metadata,repetitions," << options.repetitions << '\n'
              << "metadata,clock,steady_clock nanoseconds\n"
              << "sample,name,scalar,mode,size,nnz,repetition,nanoseconds\n"
              << "summary,name,scalar,mode,size,nnz,min_ns,p25_ns,median_ns,"
                 "p75_ns,max_ns\n";

    BenchmarkReal("dc_mna_ladder_128",
                  BuildCircuitLadder<double>(128, 1e-3, 1e-12), options);
    BenchmarkReal("transient_companion_ladder_512",
                  BuildCircuitLadder<double>(512, 1e-3, 1e-1), options);
    BenchmarkReal("transient_companion_ladder_2048",
                  BuildCircuitLadder<double>(2048, 1e-3, 1.0), options);
    BenchmarkComplex(
        "ac_mna_ladder_256",
        BuildCircuitLadder<std::complex<double>>(
            256, {1e-3, -6.283185307179586}, {1e-12, 0.006283185307179586}),
        options);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "solver selection benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
