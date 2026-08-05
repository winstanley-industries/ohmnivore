#ifndef OHMNIVORE_SIMULATOR_H_
#define OHMNIVORE_SIMULATOR_H_

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ohmnivore/status.h"

namespace ohmnivore {

struct DcResult {
  std::vector<std::pair<std::string, double>> node_voltages;
  std::vector<std::pair<std::string, double>> branch_currents;
};

[[nodiscard]] Result<DcResult> SimulateDc(std::string_view netlist);
[[nodiscard]] Result<std::string> SimulateDcToCsv(std::string_view netlist);

} // namespace ohmnivore

#endif // OHMNIVORE_SIMULATOR_H_
