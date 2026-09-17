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

namespace {
std::string Lower(std::string value) {
  for (char &c : value)
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  return value;
}
} // namespace

int main(int argc, char **argv) {
  using namespace ohmnivore;
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
    const auto emit = [&](double time,
                          const std::vector<double> &state) -> Result<bool> {
      const std::size_t added = (selected.size() + 1) * sizeof(double);
      if (points >= 2'000'000 || bytes + added > 512 * 1024 * 1024)
        return Result<bool>::Fail(ErrorCode::kUnsupportedSize,
                                  "raw output budget exhausted");
      if (!std::isfinite(time) || state.size() != system.g.rows)
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "invalid observed state");
      raw.write(reinterpret_cast<const char *>(&time), sizeof(time));
      for (const auto index : selected) {
        const double value = state[index];
        if (!std::isfinite(value))
          return Result<bool>::Fail(ErrorCode::kNonFinite,
                                    "non-finite observed state");
        raw.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
      if (!raw)
        return Result<bool>::Fail(ErrorCode::kIo, "raw write failed");
      ++points;
      bytes += added;
      return Result<bool>::Ok(true);
    };
    std::size_t attempts = 0, rejected = 0;
    if (tran != nullptr) {
      TransientExecutionLimits limits;
      limits.minimum_step_divisor = 1'000'000.0;
      limits.maximum_accepted_steps = 1'999'999;
      limits.maximum_step_attempts = 4'000'000;
      limits.nonlinear_maximum_iterations =
          system.behavioral_descriptors.empty()
              ? kDirectNewtonMaximumIterations
              : BehavioralNumericalPolicy::transient_maximum_iterations;
      limits.retain_output_states = false;
      limits.accepted_state_observer = emit;
      auto solved = RunTransientAnalysis(system, *tran, limits);
      if (!solved.ok()) {
        raw.close();
        return fail(solved.error().code, solved.error().message);
      }
      attempts = solved.value().step_trace.size();
      for (const auto &step : solved.value().step_trace)
        if (!step.accepted)
          ++rejected;
      if (points != solved.value().emitted_points) {
        raw.close();
        return fail(ErrorCode::kInvalidStructure, "observer count mismatch");
      }
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
         << "{\"schema\":\"emi02-cpu-v1\",\"status\":\"complete\",\"points\":"
         << points << ",\"variables\":" << names.size() + 1
         << ",\"unknowns\":" << system.g.rows << ",\"raw_bytes\":" << bytes
         << ",\"attempts\":" << attempts << ",\"rejected_steps\":" << rejected
         << ",\"elapsed_seconds\":" << elapsed << "}\n";
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
