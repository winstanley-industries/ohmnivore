// Independent CPU reference for bounded time-window decomposition research.
#include "cpp/src/expression_internal.h"
#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/transient.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace ohmnivore;
template <class T> T Checked(Result<T> r) {
  if (!r.ok())
    throw std::runtime_error(r.error().message);
  return r.TakeValue();
}
template <class T> void Array(const std::vector<T> &v) {
  std::cout << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << v[i];
  }
  std::cout << ']';
}
void Matrix(const CsrMatrix &m) {
  std::cout << "{\"rows\":";
  Array(m.row_offsets);
  std::cout << ",\"columns\":";
  Array(m.column_indices);
  std::cout << ",\"values\":";
  Array(m.values);
  std::cout << '}';
}
int Run(const char *deck, const char *start, int length) {
  std::ifstream in(deck);
  if (!in)
    throw std::runtime_error("missing deck");
  const std::string text{std::istreambuf_iterator<char>(in), {}};
  const auto circuit = Checked(ParseBehavioralNetlist(text));
  const auto system = Checked(CompileBehavioralMna(circuit));
  int n = 0;
  double time = 0, h = 0;
  std::ifstream initial_file(start);
  initial_file >> n >> time >> h;
  if (!initial_file || n != static_cast<int>(system.g.rows) ||
      !std::isfinite(time) || time < 0 || !std::isfinite(h) || h <= 0 ||
      length < 1 || length > 512)
    throw std::runtime_error("invalid initial window");
  std::vector<double> initial(n);
  for (double &v : initial) {
    initial_file >> v;
    if (!initial_file || !std::isfinite(v))
      throw std::runtime_error("invalid state");
  }
  std::string extra;
  if (initial_file >> extra)
    throw std::runtime_error("extra initial state");
  auto working = system;
  working.g = Checked(FormTransientCompanionMatrix(system.g, system.c, h, 1));
  Checked(RemapBehavioralDescriptors(&working));
  const auto linear = Checked(BuildNonlinearDcLinearization(working, initial));
  std::cout << std::setprecision(17) << "{\"n\":" << n
            << ",\"node_count\":" << system.node_names.size()
            << ",\"t0\":" << time << ",\"h\":" << h << ",\"length\":" << length
            << ",\"g\":";
  Matrix(system.g);
  std::cout << ",\"c\":";
  Matrix(system.c);
  std::cout << ",\"a\":";
  Matrix(linear.jacobian);
  std::cout << ",\"initial\":";
  Array(initial);
  std::cout << ",\"programs\":[";
  for (std::size_t i = 0; i < system.behavioral_descriptors.size(); ++i) {
    if (i)
      std::cout << ',';
    const auto &d = system.behavioral_descriptors[i];
    auto p = Checked(internal::ExportExpressionProgram(d.expression));
    std::cout << "{\"root\":" << p.root << ",\"rows\":[";
    for (std::size_t j = 0; j < d.rows.size(); ++j) {
      if (j)
        std::cout << ',';
      std::cout << '[' << d.rows[j].row << ',' << d.rows[j].coefficient << ']';
    }
    std::cout << "],\"nodes\":[";
    for (std::size_t j = 0; j < p.nodes.size(); ++j) {
      if (j)
        std::cout << ',';
      const auto &v = p.nodes[j];
      std::cout << '[' << static_cast<unsigned>(v.op) << ',' << v.first << ','
                << v.second << ',' << v.third << ',' << v.value << ','
                << v.constant << ']';
    }
    std::cout << "]}";
  }
  std::cout << "],\"reactive\":[";
  bool first = true;
  for (const auto &cap : system.capacitor_initial_constraints) {
    if (!first)
      std::cout << ',';
    first = false;
    std::cout << '['
              << (cap.positive_node_index
                      ? static_cast<int>(*cap.positive_node_index)
                      : -1)
              << ','
              << (cap.negative_node_index
                      ? static_cast<int>(*cap.negative_node_index)
                      : -1)
              << ",1e-7]";
  }
  for (const auto &ind : system.inductor_initial_constraints) {
    if (!first)
      std::cout << ',';
    first = false;
    std::cout << '[' << ind.branch_index << ",-1,1e-9]";
  }
  std::cout << "],\"sources\":[";
  std::vector<std::vector<double>> sources;
  for (int k = 0; k < length; ++k) {
    if (k)
      std::cout << ',';
    sources.push_back(Checked(BuildTransientRhs(system, time + (k + 1) * h)));
    Array(sources.back());
  }
  std::cout << "],\"oracle\":[";
  auto previous = initial;
  auto factor = Checked(SparseRealFactorization::Analyze(linear.jacobian));
  std::size_t iterations = 0;
  double residual = 0;
  for (int k = 0; k < length; ++k) {
    working.b_dc =
        Checked(BuildBackwardEulerRhs(system.c, previous, sources[k], h));
    auto solved =
        Checked(RunNonlinearPoint(working, previous, factor.get(), 100));
    iterations += solved.iteration_trace.size();
    residual = std::max(
        residual, Checked(ValidateNonlinearResidual(working, solved.solution)));
    if (k)
      std::cout << ',';
    Array(solved.solution);
    previous = std::move(solved.solution);
  }
  std::cout << "]}\n";
  std::cerr << "{\"steps\":" << length
            << ",\"newton_iterations\":" << iterations
            << ",\"max_nonlinear_residual\":" << residual << "}\n";
  return 0;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      return 2;
    return Run(argv[1], argv[2], std::stoi(argv[3]));
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
