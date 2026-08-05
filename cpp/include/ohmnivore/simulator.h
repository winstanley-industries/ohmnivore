#ifndef OHMNIVORE_SIMULATOR_H_
#define OHMNIVORE_SIMULATOR_H_

#include <complex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ohmnivore/ir.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

struct DcResult {
  std::vector<std::pair<std::string, double>> node_voltages;
  std::vector<std::pair<std::string, double>> branch_currents;
};

struct AcResult {
  std::vector<double> frequencies_hz;
  std::vector<std::pair<std::string, std::vector<std::complex<double>>>>
      node_voltages;
  std::vector<std::pair<std::string, std::vector<std::complex<double>>>>
      branch_currents;
};

// LIN emits exactly `points` frequencies including both endpoints. DEC/OCT
// treat `points` as points per decade/octave, emit the geometric grid below
// the stop frequency, and append the exact stop frequency once. All sweeps are
// strictly increasing and include both requested endpoints.
[[nodiscard]] Result<std::vector<double>>
GenerateAcFrequencies(const AcAnalysis &analysis);

[[nodiscard]] Result<DcResult> SimulateDc(std::string_view netlist);
[[nodiscard]] Result<std::string> SimulateDcToCsv(std::string_view netlist);
[[nodiscard]] Result<AcResult> SimulateAc(std::string_view netlist);
[[nodiscard]] Result<std::string> SimulateAcToCsv(std::string_view netlist);

// Executes requested analyses in netlist order and concatenates their
// legacy-compatible CSV tables, matching the CLI behavior.
[[nodiscard]] Result<std::string> SimulateToCsv(std::string_view netlist);

} // namespace ohmnivore

#endif // OHMNIVORE_SIMULATOR_H_
