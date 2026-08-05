#ifndef OHMNIVORE_IR_H_
#define OHMNIVORE_IR_H_

#include <cstddef>
#include <optional>
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

struct Capacitor {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  double capacitance_farads;
};

struct Inductor {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  double inductance_henries;
};

struct AcSourceSpecification {
  double magnitude;
  double phase_degrees;
};

struct VoltageSource {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  std::optional<double> dc_volts;
  std::optional<AcSourceSpecification> ac;
};

struct CurrentSource {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  std::optional<double> dc_amperes;
  std::optional<AcSourceSpecification> ac;
};

using Component =
    std::variant<Resistor, Capacitor, Inductor, VoltageSource, CurrentSource>;

struct DcAnalysis {};

enum class AcSweepType {
  kDec,
  kOct,
  kLin,
};

struct AcAnalysis {
  AcSweepType sweep_type;
  std::size_t points;
  double start_frequency_hz;
  double stop_frequency_hz;
};

using Analysis = std::variant<DcAnalysis, AcAnalysis>;

struct Circuit {
  std::vector<Component> components;
  std::vector<Analysis> analyses;
};

} // namespace ohmnivore

#endif // OHMNIVORE_IR_H_
