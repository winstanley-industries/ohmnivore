// Read-only numerical decomposition probe; never linked into a simulator.
#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/transient.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
template <class T> T Checked(ohmnivore::Result<T> result) {
  if (!result.ok())
    throw std::runtime_error(result.error().message);
  return result.TakeValue();
}
template <class T> void Array(const std::vector<T> &values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}
} // namespace
int main(int argc, char **argv) {
  using namespace ohmnivore;
  if (argc < 2 || argc > 3 ||
      (argc == 3 && std::string(argv[2]) != "--structure"))
    return 2;
  try {
    std::ifstream input(argv[1]);
    if (!input)
      throw std::runtime_error("input unavailable");
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    const auto parsed = Checked(ParseBehavioralNetlist(text));
    const auto system = Checked(CompileBehavioralMna(parsed));
    if (argc == 3) {
      std::cout << std::setprecision(17) << "{\"n\":" << system.g.rows
                << ",\"c_rows\":";
      Array(system.c.row_offsets);
      std::cout << ",\"c_columns\":";
      Array(system.c.column_indices);
      std::cout << ",\"c_values\":";
      Array(system.c.values);
      std::cout << "}\n";
      return 0;
    }
    if (parsed.circuit.analyses.size() != 1)
      throw std::runtime_error("one analysis required");
    const auto *analysis =
        std::get_if<TranAnalysis>(&parsed.circuit.analyses[0]);
    if (!analysis)
      throw std::runtime_error("transient required");
    const std::vector<double> targets{0,    .9e-6, 1e-6,  1.03e-6,
                                      5e-6, 10e-6, 15e-6, 20 * 1e-6};
    std::size_t selected = 0;
    double previous = 0;
    TransientExecutionLimits limits;
    limits.minimum_step_divisor = 1e6;
    limits.maximum_accepted_steps = 1999999;
    limits.maximum_step_attempts = 4000000;
    limits.nonlinear_maximum_iterations =
        BehavioralNumericalPolicy::transient_maximum_iterations;
    limits.behavioral_error_estimator =
        BehavioralErrorEstimator::kDerivativeHistory;
    limits.retain_output_states = false;
    std::cout << std::setprecision(17);
    limits.accepted_state_observer = [&](double time,
                                         const std::vector<double> &state) {
      const double h =
          time > previous ? time - previous : analysis->time_step_seconds;
      previous = time;
      if (selected >= targets.size() || time < targets[selected])
        return Result<bool>::Ok(true);
      const auto index = selected++;
      for (double alpha : {1., 2.}) {
        auto working = system;
        working.g =
            Checked(FormTransientCompanionMatrix(system.g, system.c, h, alpha));
        Checked(RemapBehavioralDescriptors(&working));
        const auto linear =
            Checked(BuildNonlinearDcLinearization(working, state));
        const auto &a = linear.jacobian;
        std::vector<double> manufactured(a.rows), rhs(a.rows);
        for (std::size_t i = 0; i < a.rows; ++i)
          manufactured[i] = std::sin(static_cast<double>(i + 1));
        for (std::size_t row = 0; row < a.rows; ++row) {
          long double value = 0;
          for (auto k = a.row_offsets[row]; k < a.row_offsets[row + 1]; ++k)
            value += static_cast<long double>(a.values[k]) *
                     manufactured[a.column_indices[k]];
          rhs[row] = static_cast<double>(value);
        }
        auto factor = Checked(SparseRealFactorization::Analyze(a));
        const auto oracle = Checked(factor->FactorAndSolveRefined(a, rhs));
        Checked(ValidateSparseSolution(a, rhs, oracle));
        std::cout << "{\"sample\":" << index << ",\"time_s\":" << time
                  << ",\"step_s\":" << h << ",\"alpha\":" << alpha
                  << ",\"n\":" << a.rows
                  << ",\"node_count\":" << system.node_names.size()
                  << ",\"names\":[";
        auto names = system.node_names;
        names.insert(names.end(), system.branch_names.begin(),
                     system.branch_names.end());
        for (std::size_t i = 0; i < names.size(); ++i) {
          if (i)
            std::cout << ',';
          std::cout << std::quoted(names[i]);
        }
        std::cout << "],\"rows\":";
        Array(a.row_offsets);
        std::cout << ",\"columns\":";
        Array(a.column_indices);
        std::cout << ",\"values\":";
        Array(a.values);
        std::cout << ",\"rhs\":";
        Array(rhs);
        std::cout << ",\"oracle\":";
        Array(oracle);
        std::cout << ",\"state\":";
        Array(state);
        std::cout << "}\n";
      }
      return Result<bool>::Ok(true);
    };
    const auto result =
        Checked(RunTransientAnalysis(system, *analysis, limits));
    if (selected != targets.size())
      throw std::runtime_error("missing selected state");
    std::cerr << "complete: " << selected * 2 << " matrices from "
              << result.emitted_points << " accepted states\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
