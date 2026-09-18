#ifndef OHMNIVORE_BEHAVIORAL_H_
#define OHMNIVORE_BEHAVIORAL_H_

#include <string>
#include <string_view>
#include <vector>

#include "ohmnivore/compiler.h"

namespace ohmnivore {

struct BehavioralSource {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  std::string expression;
  bool is_voltage;
};

struct BehavioralCircuit {
  Circuit circuit;
  std::vector<BehavioralSource> sources;
  std::vector<std::string> saved_observables;
};

// Explicit EMI-02 surface; ordinary ParseNetlist remains unchanged.
[[nodiscard]] Result<BehavioralCircuit>
ParseBehavioralNetlist(std::string_view netlist);
[[nodiscard]] Result<MnaSystem>
CompileBehavioralMna(const BehavioralCircuit &circuit);
[[nodiscard]] Result<bool> RemapBehavioralDescriptors(MnaSystem *system);
[[nodiscard]] Result<bool>
ValidateBehavioralDescriptors(const MnaSystem &system);
[[nodiscard]] Result<bool> ValidateBehavioralTransient(const MnaSystem &system);

struct BehavioralNumericalPolicy {
  static constexpr double voltage_absolute_tolerance = 1e-7;
  static constexpr double current_absolute_tolerance = 1e-9;
  static constexpr double relative_tolerance = 1e-5;
  static constexpr double lte_relative_tolerance = 1e-4;
  static constexpr double newton_reactive_lte_fraction = 0.01;
  static constexpr std::size_t dc_maximum_iterations = 300;
  static constexpr std::size_t transient_maximum_iterations = 100;
};

} // namespace ohmnivore
#endif // OHMNIVORE_BEHAVIORAL_H_
