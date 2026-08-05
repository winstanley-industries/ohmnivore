#ifndef OHMNIVORE_IR_H_
#define OHMNIVORE_IR_H_

#include <string>
#include <variant>
#include <vector>

namespace ohmnivore {

struct Resistor {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  double resistance_ohms;
};

struct VoltageSource {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  double dc_volts;
};

using Component = std::variant<Resistor, VoltageSource>;

enum class Analysis {
  kDc,
};

struct Circuit {
  std::vector<Component> components;
  std::vector<Analysis> analyses;
};

} // namespace ohmnivore

#endif // OHMNIVORE_IR_H_
