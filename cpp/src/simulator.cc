#include "ohmnivore/simulator.h"

#include <charconv>
#include <cstddef>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/solver.h"

namespace ohmnivore {
namespace {

[[nodiscard]] Result<std::string> FormatDouble(double value) {
  char buffer[64];
  const auto formatted = std::to_chars(std::begin(buffer), std::end(buffer),
                                       value, std::chars_format::general);
  if (formatted.ec != std::errc{}) {
    return Result<std::string>::Fail(ErrorCode::kIo,
                                     "failed to format floating-point output");
  }
  return Result<std::string>::Ok(std::string(buffer, formatted.ptr));
}

} // namespace

Result<DcResult> SimulateDc(std::string_view netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    return Result<DcResult>::Fail(parsed.error().code, parsed.error().message);
  }
  Circuit circuit = parsed.TakeValue();
  bool has_dc = false;
  for (Analysis analysis : circuit.analyses) {
    has_dc = has_dc || analysis == Analysis::kDc;
  }
  if (!has_dc) {
    return Result<DcResult>::Fail(ErrorCode::kUnsupported,
                                  "phase 2A requires a .DC or .OP analysis");
  }

  auto compiled = CompileMna(circuit);
  if (!compiled.ok()) {
    return Result<DcResult>::Fail(compiled.error().code,
                                  compiled.error().message);
  }
  MnaSystem system = compiled.TakeValue();
  auto solved = SolveCpuReference(system.g, system.b_dc);
  if (!solved.ok()) {
    return Result<DcResult>::Fail(solved.error().code, solved.error().message);
  }
  std::vector<double> solution = solved.TakeValue();

  DcResult result;
  for (std::size_t index = 0; index < system.node_names.size(); ++index) {
    result.node_voltages.emplace_back(system.node_names[index],
                                      solution[index]);
  }
  const std::size_t branch_offset = system.node_names.size();
  for (std::size_t index = 0; index < system.branch_names.size(); ++index) {
    result.branch_currents.emplace_back(system.branch_names[index],
                                        solution[branch_offset + index]);
  }
  return Result<DcResult>::Ok(std::move(result));
}

Result<std::string> SimulateDcToCsv(std::string_view netlist) {
  auto simulated = SimulateDc(netlist);
  if (!simulated.ok()) {
    return Result<std::string>::Fail(simulated.error().code,
                                     simulated.error().message);
  }

  std::string csv = "Variable,Value\n";
  const auto append_row = [&](std::string variable,
                              double value) -> Result<std::string> {
    auto formatted = FormatDouble(value);
    if (!formatted.ok()) {
      return Result<std::string>::Fail(formatted.error().code,
                                       formatted.error().message);
    }
    return Result<std::string>::Ok(std::move(variable) + "," +
                                   formatted.TakeValue() + "\n");
  };

  for (const auto &[name, voltage] : simulated.value().node_voltages) {
    auto row = append_row("V(" + name + ")", voltage);
    if (!row.ok()) {
      return row;
    }
    csv += row.TakeValue();
  }
  for (const auto &[name, current] : simulated.value().branch_currents) {
    auto row = append_row("I(" + name + ")", current);
    if (!row.ok()) {
      return row;
    }
    csv += row.TakeValue();
  }
  return Result<std::string>::Ok(std::move(csv));
}

} // namespace ohmnivore
