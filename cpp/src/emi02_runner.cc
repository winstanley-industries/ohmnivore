#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/transient.h"

#ifdef OHMNIVORE_EMI03_RESIDENT
#include "cuda/emi03_resident.h"
#endif

#ifdef OHMNIVORE_EMI03_PROFILE
#include "cpp/benchmarks/emi03_profile.h"
#endif

namespace {
std::string Lower(std::string value) {
  for (char &c : value)
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  return value;
}
} // namespace

int RunEmi02Job(int argc, char **argv) {
  using namespace ohmnivore;
#ifdef OHMNIVORE_EMI03_PROFILE
  emi03_profile::counters = {};
#endif
  if (argc != 4) {
    std::cerr << "usage: emi02_runner input.spice output.raw metadata.json\n";
    return 2;
  }
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path output(argv[2]), metadata(argv[3]);
  const auto temporary = std::filesystem::path(output.string() + ".partial");
  const auto metadata_temporary =
      std::filesystem::path(metadata.string() + ".partial");
  bool raw_published = false;
  bool raw_created = false;
  bool metadata_created = false;
  const auto fail = [&](ErrorCode code, const std::string &message) {
    std::error_code ignored;
    if (raw_published)
      std::filesystem::remove(output, ignored);
    if (raw_created)
      std::filesystem::remove(temporary, ignored);
    if (metadata_created)
      std::filesystem::remove(metadata_temporary, ignored);
    std::cerr << ErrorCodeName(code) << ": " << message << '\n';
    return 1;
  };
  try {
    if (std::endian::native != std::endian::little)
      return fail(ErrorCode::kUnsupported,
                  "raw output requires little-endian host");
    std::set<std::filesystem::path> paths{
        std::filesystem::weakly_canonical(argv[1])};
    for (const auto &path : {output, metadata, temporary, metadata_temporary}) {
      if (!paths.insert(std::filesystem::weakly_canonical(path)).second) {
        return fail(ErrorCode::kIo, "input and output paths must be distinct");
      }
      if (std::filesystem::exists(std::filesystem::symlink_status(path))) {
        std::cerr << "io: output path already exists\n";
        return 1;
      }
    }
    if (std::filesystem::file_size(argv[1]) > 1024 * 1024)
      return fail(ErrorCode::kUnsupportedSize, "deck exceeds 1 MiB");
    std::ifstream input(argv[1]);
    if (!input)
      return fail(ErrorCode::kIo, "cannot read deck");
    const std::string text{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    auto parsed = ParseBehavioralNetlist(text);
    if (!parsed.ok())
      return fail(parsed.error().code, parsed.error().message);
    if (parsed.value().circuit.analyses.size() != 1)
      return fail(ErrorCode::kUnsupported, "exactly one analysis required");
    const Analysis &analysis = parsed.value().circuit.analyses.front();
    const auto *tran = std::get_if<TranAnalysis>(&analysis);
    if (tran == nullptr && !std::holds_alternative<DcAnalysis>(analysis))
      return fail(ErrorCode::kUnsupported, "only DC/transient are supported");
    if (tran != nullptr &&
        (tran->start_time_seconds != 0.0 || tran->use_initial_conditions))
      return fail(ErrorCode::kUnsupported,
                  "reference requires full no-UIC transient coverage");
    auto compiled = CompileBehavioralMna(parsed.value());
    if (!compiled.ok())
      return fail(compiled.error().code, compiled.error().message);
    const MnaSystem &system = compiled.value();
    std::vector<std::string> state_names;
    for (const auto &node : system.node_names)
      state_names.push_back("v(" + Lower(node) + ")");
    for (const auto &branch : system.branch_names)
      state_names.push_back("i(" + Lower(branch) + ")");
    auto names = parsed.value().saved_observables;
    if (names.empty())
      names = state_names;
    std::vector<std::size_t> selected;
    for (const auto &name : names) {
      const auto found =
          std::find(state_names.begin(), state_names.end(), name);
      if (found == state_names.end())
        return fail(ErrorCode::kCompile, "unknown saved observable: " + name);
      selected.push_back(static_cast<std::size_t>(found - state_names.begin()));
    }
    std::ofstream raw(temporary, std::ios::binary);
    if (!raw)
      return fail(ErrorCode::kIo, "cannot create raw output");
    raw_created = true;
    std::string title = Lower(text.substr(0, text.find('\n')));
    const auto title_start = title.find_first_not_of("* \t\r");
    title =
        title_start == std::string::npos ? "emi02" : title.substr(title_start);
    raw << "Title: " << title
        << "\nPlotname: " << (tran ? "Transient Analysis" : "Operating Point")
        << "\nFlags: real\nNo. Variables: " << names.size() + 1
        << "\nNo. Points: ";
    const auto count_position = raw.tellp();
    raw << "0000000000\nVariables:\n0\ttime\ttime\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
      raw << i + 1 << '\t' << names[i] << '\t'
          << (names[i].starts_with("i(") ? "current" : "voltage") << '\n';
    }
    raw << "Binary:\n";
    std::size_t points = 0;
    std::size_t bytes = static_cast<std::size_t>(raw.tellp());
    std::vector<double> output_record(selected.size() + 1);
    const auto emit = [&](double time,
                          const std::vector<double> &state) -> Result<bool> {
#ifdef OHMNIVORE_EMI03_PROFILE
      const emi03_profile::Scope profile(emi03_profile::Phase::kOutput);
#endif
      const std::size_t added = (selected.size() + 1) * sizeof(double);
      if (points >= 2'000'000 || bytes + added > 512 * 1024 * 1024)
        return Result<bool>::Fail(ErrorCode::kUnsupportedSize,
                                  "raw output budget exhausted");
      if (!std::isfinite(time) || state.size() != system.g.rows)
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "invalid observed state");
      output_record[0] = time;
      for (std::size_t column = 0; column < selected.size(); ++column) {
        const double value = state[selected[column]];
        if (!std::isfinite(value))
          return Result<bool>::Fail(ErrorCode::kNonFinite,
                                    "non-finite observed state");
        output_record[column + 1] = value;
      }
      raw.write(reinterpret_cast<const char *>(output_record.data()),
                static_cast<std::streamsize>(added));
      if (!raw)
        return Result<bool>::Fail(ErrorCode::kIo, "raw write failed");
      ++points;
      bytes += added;
      return Result<bool>::Ok(true);
    };
    std::size_t attempts = 0, rejected = 0;
    std::size_t nonlinear_rejections = 0;
    std::size_t history_estimates = 0, history_checks = 0,
                doubling_estimates = 0;
    std::size_t history_fallback_entries = 0, history_fallback_recoveries = 0;
    SparseSolverStatistics solver_statistics;
    if (tran != nullptr) {
      TransientExecutionLimits limits;
      limits.minimum_step_divisor = 1'000'000.0;
      if (!system.behavioral_descriptors.empty()) {
        limits.behavioral_error_estimator =
            BehavioralErrorEstimator::kDerivativeHistory;
      }
      limits.maximum_accepted_steps = 1'999'999;
      limits.maximum_step_attempts = 4'000'000;
      limits.nonlinear_maximum_iterations =
          system.behavioral_descriptors.empty()
              ? kDirectNewtonMaximumIterations
              : BehavioralNumericalPolicy::transient_maximum_iterations;
      limits.retain_output_states = false;
      limits.accepted_state_observer = emit;
#ifdef OHMNIVORE_EMI03_RESIDENT
      auto solved = RunEmi03ResidentTransient(system, *tran, limits);
      if (!solved.ok()) {
        raw.close();
        return fail(solved.error().code, solved.error().message);
      }
      const auto &r = solved.value();
      attempts = r.attempts;
      rejected = r.rejected;
      nonlinear_rejections = r.nonlinear_rejections;
      history_estimates = r.history_estimates;
      history_checks = r.history_checks;
      doubling_estimates = r.doubling_estimates;
      history_fallback_entries = r.history_fallback_entries;
      history_fallback_recoveries = r.history_fallback_recoveries;
      solver_statistics = r.solver_statistics;
      if (points != r.emitted_points) {
        raw.close();
        return fail(ErrorCode::kInvalidStructure,
                    "resident observer count mismatch");
      }
#else
      auto solved = RunTransientAnalysis(system, *tran, limits);
      if (!solved.ok()) {
        raw.close();
        return fail(solved.error().code, solved.error().message);
      }
      attempts = solved.value().step_trace.size();
      solver_statistics = solved.value().solver_statistics;
      history_estimates = solved.value().derivative_history_error_estimates;
      history_checks = solved.value().derivative_history_step_doubling_checks;
      doubling_estimates = solved.value().step_doubling_error_estimates;
      history_fallback_entries =
          solved.value().derivative_history_fallback_entries;
      history_fallback_recoveries =
          solved.value().derivative_history_fallback_recoveries;
      for (const auto &step : solved.value().step_trace)
        if (!step.accepted) {
          ++rejected;
          if (step.rejection_reason ==
              TransientStepRejectionReason::kNonlinearConvergence)
            ++nonlinear_rejections;
        }
      if (points != solved.value().emitted_points) {
        raw.close();
        return fail(ErrorCode::kInvalidStructure, "observer count mismatch");
      }
#endif
    } else {
      Result<std::vector<double>> state = [&]() {
        if (system.behavioral_descriptors.empty())
          return SolveSparseReal(system.g, system.b_dc);
        NonlinearDcOptions options;
        options.direct_maximum_iterations = 300;
        options.source_step_maximum_iterations = 300;
        options.gmin_step_maximum_iterations = 300;
        options.final_gmin_maximum_iterations = 300;
        auto solved = RunNonlinearDc(system, options);
        if (!solved.ok())
          return Result<std::vector<double>>::Fail(solved.error().code,
                                                   solved.error().message);
        return Result<std::vector<double>>::Ok(solved.TakeValue().solution);
      }();
      if (!state.ok()) {
        raw.close();
        return fail(state.error().code, state.error().message);
      }
      auto emitted = emit(0.0, state.value());
      if (!emitted.ok()) {
        raw.close();
        return fail(emitted.error().code, emitted.error().message);
      }
    }
    raw.seekp(count_position);
    raw << std::setw(10) << std::setfill('0') << points;
    raw.close();
    if (!raw || std::filesystem::file_size(temporary) != bytes)
      return fail(ErrorCode::kIo, "raw close or size check failed");
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::ofstream info(metadata_temporary);
    if (!info)
      return fail(ErrorCode::kIo, "cannot create metadata output");
    metadata_created = true;
    info << std::setprecision(17)
         << "{\"schema\":\"emi02-cpu-v2\",\"status\":\"complete\",\"points\":"
         << points << ",\"variables\":" << names.size() + 1
         << ",\"unknowns\":" << system.g.rows << ",\"raw_bytes\":" << bytes
         << ",\"attempts\":" << attempts << ",\"rejected_steps\":" << rejected
         << ",\"nonlinear_rejections\":" << nonlinear_rejections
         << ",\"behavioral_integration_method\":\""
         << (tran != nullptr && !system.behavioral_descriptors.empty()
                 ? "trapezoidal"
                 : "not-applicable")
         << "\""
         << ",\"behavioral_error_estimator\":\""
         << (tran != nullptr && !system.behavioral_descriptors.empty()
                 ? "derivative-history-audited-v1"
                 : "not-applicable")
         << "\""
         << ",\"derivative_history_error_estimates\":" << history_estimates
         << ",\"derivative_history_step_doubling_checks\":" << history_checks
         << ",\"step_doubling_error_estimates\":" << doubling_estimates
         << ",\"derivative_history_fallback_entries\":"
         << history_fallback_entries
         << ",\"derivative_history_fallback_recoveries\":"
         << history_fallback_recoveries
         << ",\"transient_solver_statistics\":{\"symbolic_analyses\":"
         << solver_statistics.symbolic_analyses
         << ",\"numeric_factorizations\":"
         << solver_statistics.numeric_factorizations
         << ",\"numeric_refactorizations\":"
         << solver_statistics.numeric_refactorizations
         << ",\"numeric_refactorization_fallbacks\":"
         << solver_statistics.numeric_refactorization_fallbacks
         << ",\"numeric_reuses\":" << solver_statistics.numeric_reuses
         << ",\"solves\":" << solver_statistics.solves
         << ",\"iterative_refinement_solves\":"
         << solver_statistics.iterative_refinement_solves << "}";
#ifdef OHMNIVORE_EMI03_PROFILE
    using emi03_profile::Phase;
    const double assembly_s = emi03_profile::Seconds(Phase::kAssembly);
    const double linear_s = emi03_profile::Seconds(Phase::kLinearSolve);
    const double output_s = emi03_profile::Seconds(Phase::kOutput);
    info << ",\"diagnostic_profile\":{\"schema\":\"emi03-cpu-profile-v2\""
         << ",\"assembly_evaluation_s\":" << assembly_s
         << ",\"linear_factor_solve_refinement_validation_s\":" << linear_s
         << ",\"accepted_state_output_s\":" << output_s
         << ",\"integration_setup_other_s\":"
         << elapsed - assembly_s - linear_s - output_s
         << ",\"assembly_calls\":" << emi03_profile::Calls(Phase::kAssembly)
         << ",\"linear_calls\":" << emi03_profile::Calls(Phase::kLinearSolve)
         << ",\"output_calls\":" << emi03_profile::Calls(Phase::kOutput)
         << ",\"nested_expression_profile\":{"
         << "\"interpretation\":\"nested in assembly or setup; not additive; "
            "includes diagnostic instrumentation effects\""
         << ",\"full_s\":" << emi03_profile::Seconds(Phase::kExpressionFull)
         << ",\"value_s\":" << emi03_profile::Seconds(Phase::kExpressionValue)
         << ",\"full_calls\":" << emi03_profile::Calls(Phase::kExpressionFull)
         << ",\"value_calls\":" << emi03_profile::Calls(Phase::kExpressionValue)
         << "}}";
#endif
    info << ",\"elapsed_seconds\":" << elapsed << "}\n";
    info.close();
    if (!info)
      return fail(ErrorCode::kIo, "metadata close failed");
    std::filesystem::rename(temporary, output);
    raw_published = true;
    std::filesystem::rename(metadata_temporary, metadata);
    return 0;
  } catch (const std::exception &error) {
    return fail(ErrorCode::kIo, error.what());
  }
}

#ifndef OHMNIVORE_EMI03_WORKER
int main(int argc, char **argv) { return RunEmi02Job(argc, argv); }
#endif
