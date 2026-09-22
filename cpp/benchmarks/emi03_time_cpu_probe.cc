#include "ohmnivore/solver.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace ohmnivore;
template <class T> T ProbeChecked(Result<T> result) {
  if (!result.ok())
    throw std::runtime_error(result.error().message);
  return result.TakeValue();
}
struct ProbeSample {
  std::string name;
  CsrMatrix a;
  std::vector<double> b, oracle;
};
std::vector<ProbeSample> ReadProbe(const char *path) {
  std::ifstream input(path);
  std::string magic;
  int n = 0, ports = 0, groups = 0, count = 0;
  input >> magic >> n >> ports >> groups >> count;
  if (!input || magic != "EMI03_SCHUR_REPLAY_1" || n != 185 || ports != 14 ||
      groups != 15 || count != 144)
    throw std::runtime_error("invalid frozen replay");
  int ignored;
  for (int i = 0; i < ports; ++i)
    input >> ignored;
  for (int g = 0; g < groups; ++g) {
    int size = 0;
    input >> size;
    if (size < 1 || size > 64)
      throw std::runtime_error("bad group");
    for (int i = 0; i < size; ++i)
      input >> ignored;
  }
  std::vector<ProbeSample> samples(count);
  for (auto &s : samples) {
    std::size_t nnz = 0;
    input >> s.name >> nnz;
    if (!input || nnz > static_cast<std::size_t>(n * n))
      throw std::runtime_error("bad matrix");
    s.a.rows = s.a.columns = n;
    s.a.row_offsets.resize(n + 1);
    s.a.column_indices.resize(nnz);
    s.a.values.resize(nnz);
    s.b.resize(n);
    s.oracle.resize(n);
    for (auto &x : s.a.row_offsets)
      input >> x;
    for (auto &x : s.a.column_indices)
      input >> x;
    for (auto &x : s.a.values)
      input >> x;
    for (auto &x : s.b)
      input >> x;
    for (auto &x : s.oracle)
      input >> x;
    if (!input)
      throw std::runtime_error("truncated matrix");
    ProbeChecked(ValidateSparseSolution(s.a, s.b, s.oracle));
  }
  return samples;
}
struct HostTime {
  ProbeSample sample;
  CsrMatrix mass;
  double step;
  std::vector<double> rhs, truth;
};
std::vector<HostTime> ReadTime(const char *replay, const char *mass_path,
                               std::vector<int> &active) {
  auto samples = ReadProbe(replay);
  std::ifstream in(mass_path);
  std::string magic;
  int count = 0, n = 0, rank = 0;
  in >> magic >> count >> n >> rank;
  if (!in || magic != "EMI03_TIME_MASS_1" || count != 9 || n != 185 ||
      rank < 1 || rank > 64)
    throw std::runtime_error("invalid time replay dimensions");
  active.resize(rank);
  for (int &x : active)
    in >> x;
  if (!std::is_sorted(active.begin(), active.end()) ||
      std::adjacent_find(active.begin(), active.end()) != active.end() ||
      active.front() < 0 || active.back() >= n)
    throw std::runtime_error("invalid time state selectors");
  std::vector<HostTime> cases(count);
  for (int job = 0; job < count; ++job) {
    auto &h = cases[job];
    h.sample = samples[job * 16 + 4];
    std::string name;
    int nnz = 0;
    in >> name >> h.step >> nnz;
    if (!in || name != h.sample.name || !std::isfinite(h.step) || h.step <= 0 ||
        nnz < 0 || nnz > n * n)
      throw std::runtime_error("invalid time mass record");
    auto &c = h.mass;
    c.rows = c.columns = n;
    c.row_offsets.resize(n + 1);
    c.column_indices.resize(nnz);
    c.values.resize(nnz);
    for (auto &x : c.row_offsets)
      in >> x;
    for (auto &x : c.column_indices)
      in >> x;
    for (auto &x : c.values)
      in >> x;
    if (!in)
      throw std::runtime_error("truncated time mass matrix");
    ProbeChecked(ValidateSparseSolution(c, std::vector<double>(n),
                                        std::vector<double>(n)));
    for (std::size_t j = 0; j < c.values.size(); ++j)
      if (c.values[j] != 0 &&
          !std::binary_search(active.begin(), active.end(),
                              static_cast<int>(c.column_indices[j])))
        throw std::runtime_error("mass coupling omitted by selectors");
    h.truth.resize(513 * n);
    h.rhs.resize(512 * n);
    for (int point = 0; point <= 512; ++point)
      for (int row = 0; row < n; ++row)
        h.truth[point * n + row] = .2 * std::sin(.031 * point + .047 * row);
    for (int point = 0; point < 512; ++point)
      for (int row = 0; row < n; ++row) {
        long double value = 0;
        for (auto j = h.sample.a.row_offsets[row];
             j < h.sample.a.row_offsets[row + 1]; ++j)
          value += static_cast<long double>(h.sample.a.values[j]) *
                   h.truth[(point + 1) * n + h.sample.a.column_indices[j]];
        for (auto j = c.row_offsets[row]; j < c.row_offsets[row + 1]; ++j)
          value -= static_cast<long double>(c.values[j] / h.step) *
                   h.truth[point * n + c.column_indices[j]];
        h.rhs[point * n + row] = static_cast<double>(value);
      }
  }
  std::string extra;
  if (in >> extra)
    throw std::runtime_error("time mass trailing data");
  return cases;
}

int Oracle(const char *replay, const char *mass) {
  std::vector<int> active;
  const auto cases = ReadTime(replay, mass, active);
  std::cout << std::setprecision(17) << "EMI03_TIME_ORACLE_1 " << cases.size()
            << " 185 512\n";
  double maximum = 0, difference = 0;
  for (const auto &h : cases) {
    const auto &a = h.sample.a;
    const int n = a.rows;
    auto factor = ProbeChecked(SparseRealFactorization::Analyze(a));
    std::vector<double> previous(h.truth.begin(), h.truth.begin() + n), rhs(n);
    std::cout << h.sample.name << " " << h.step << "\n";
    for (int point = 0; point < 512; ++point) {
      for (int row = 0; row < n; ++row) {
        long double value = h.rhs[point * n + row];
        for (auto j = h.mass.row_offsets[row]; j < h.mass.row_offsets[row + 1];
             ++j)
          value += static_cast<long double>(h.mass.values[j] / h.step) *
                   previous[h.mass.column_indices[j]];
        rhs[row] = static_cast<double>(value);
      }
      auto solution = ProbeChecked(factor->FactorAndSolveRefined(a, rhs));
      maximum = std::max(
          maximum, ProbeChecked(ValidateSparseSolution(a, rhs, solution)));
      for (int row = 0; row < n; ++row) {
        difference =
            std::max(difference,
                     std::abs(solution[row] - h.truth[(point + 1) * n + row]));
        if (row)
          std::cout << ' ';
        std::cout << solution[row];
      }
      std::cout << '\n';
      previous = std::move(solution);
    }
    const auto stats = factor->statistics();
    std::cerr << "{\"case\":\"" << h.sample.name
              << "\",\"solves\":" << stats.solves
              << ",\"factors\":" << stats.numeric_factorizations
              << ",\"reuses\":" << stats.numeric_reuses
              << ",\"refinements\":" << stats.iterative_refinement_solves
              << "}\n";
  }
  if (difference > 2e-10)
    throw std::runtime_error("CPU time manufactured differential failure");
  std::cerr << std::setprecision(17)
            << "{\"original_systems\":4608,\"maximum_componentwise\":"
            << maximum << ",\"manufactured_difference\":" << difference
            << "}\n";
  return 0;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      return 2;
    return Oracle(argv[1], argv[2]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
