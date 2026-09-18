#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "cpp/tests/transient_accuracy_cases.h"
#include "ohmnivore/behavioral.h"

int main(int argc, char **argv) {
  using namespace ohmnivore;
  if (argc > 2)
    return 2;
  int repetitions = 3;
  if (argc == 2) {
    const std::string_view argument(argv[1]);
    const auto parsed = std::from_chars(
        argument.data(), argument.data() + argument.size(), repetitions);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != argument.data() + argument.size())
      return 2;
  }
  if (repetitions < 1 || repetitions > 20)
    return 2;
  bool all_pass = true;
  std::cout << std::setprecision(17)
            << "case,policy,refinement,repetition,pass,points,attempts,rejects,"
               "solves,doubling,voltage_error,current_error,voltage_limit,"
               "current_limit,seconds\n";
  for (const auto &fixture : test::TransientAccuracyCases()) {
    for (const auto policy : {BehavioralErrorEstimator::kStepDoubling,
                              BehavioralErrorEstimator::kDerivativeHistory}) {
      for (int refinement = 0; refinement < 2; ++refinement) {
        for (int repetition = 0; repetition < repetitions; ++repetition) {
          const auto start = std::chrono::steady_clock::now();
          auto parsed = ParseBehavioralNetlist(fixture.deck);
          if (!parsed.ok()) {
            std::cerr << fixture.name << ": " << parsed.error().message << '\n';
            return 1;
          }
          auto compiled = CompileBehavioralMna(parsed.value());
          if (!compiled.ok()) {
            std::cerr << fixture.name << ": " << compiled.error().message
                      << '\n';
            return 1;
          }
          const auto &system = compiled.value();
          const auto node = std::find(system.node_names.begin(),
                                      system.node_names.end(), "out");
          if (node == system.node_names.end())
            return 1;
          const auto out =
              static_cast<std::size_t>(node - system.node_names.begin());
          TransientExecutionLimits limits;
          limits.behavioral_error_estimator = policy;
          limits.minimum_step_divisor = 1e6;
          limits.nonlinear_maximum_iterations = 100;
          auto analysis = fixture.analysis;
          analysis.time_step_seconds /= refinement == 0 ? 1 : 32;
          auto solved = RunTransientAnalysis(system, analysis, limits);
          if (!solved.ok()) {
            std::cerr << fixture.name << ": " << solved.error().message << '\n';
            return 1;
          }
          const auto &result = solved.value();
          double voltage_error = 0, current_error = 0;
          for (std::size_t i = 0; i < result.states.size(); ++i) {
            const auto exact = fixture.exact(result.times_seconds[i]);
            voltage_error =
                std::max(voltage_error, std::abs(result.states[i][out] -
                                                 fixture.bias - exact[0]));
            if (fixture.has_inductor) {
              const auto branch =
                  system.inductor_initial_constraints.at(0).branch_index;
              current_error = std::max(
                  current_error, std::abs(result.states[i][branch] - exact[1]));
            }
          }
          const bool pass = voltage_error <= fixture.voltage_limit &&
                            current_error <= fixture.current_limit;
          // Coarse runs deliberately expose accumulated global error; only
          // the independently bounded fine lane is an accuracy admission gate.
          if (refinement == 1)
            all_pass &= pass;
          const auto rejects =
              std::count_if(result.step_trace.begin(), result.step_trace.end(),
                            [](const auto &s) { return !s.accepted; });
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
          std::cout << fixture.name << ',' << static_cast<int>(policy) << ','
                    << refinement << ',' << repetition << ',' << pass << ','
                    << result.emitted_points << ',' << result.step_trace.size()
                    << ',' << rejects << ',' << result.solver_statistics.solves
                    << ',' << result.step_doubling_error_estimates << ','
                    << voltage_error << ',' << current_error << ','
                    << fixture.voltage_limit << ',' << fixture.current_limit
                    << ',' << seconds << '\n';
        }
      }
    }
  }
  return all_pass ? 0 : 1;
}
