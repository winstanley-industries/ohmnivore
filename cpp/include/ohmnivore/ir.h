#ifndef OHMNIVORE_IR_H_
#define OHMNIVORE_IR_H_

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ohmnivore {

inline constexpr double kDiodeThermalVoltageVolts = 0.02585;
inline constexpr double kDiodeMaximumParameterMagnitude = 1e100;
inline constexpr double kDiodeMaximumEmissionVoltageVolts =
    kDiodeMaximumParameterMagnitude * kDiodeThermalVoltageVolts;

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

struct PulseWaveform {
  double initial_value;
  double pulsed_value;
  double delay_seconds = 0.0;
  double rise_time_seconds = 0.0;
  double fall_time_seconds = 0.0;
  double pulse_width_seconds = std::numeric_limits<double>::max();
  double period_seconds = std::numeric_limits<double>::max();
};

struct SinWaveform {
  double offset;
  double amplitude;
  double frequency_hz;
  double delay_seconds = 0.0;
  double damping_factor_per_second = 0.0;
};

struct PwlWaveform {
  std::vector<std::pair<double, double>> time_value_pairs;
};

struct ExpWaveform {
  double initial_value;
  double pulsed_value;
  double rise_delay_seconds = 0.0;
  double rise_time_constant_seconds = std::numeric_limits<double>::max();
  double fall_delay_seconds = std::numeric_limits<double>::max();
  double fall_time_constant_seconds = std::numeric_limits<double>::max();
};

using TransientWaveform =
    std::variant<PulseWaveform, SinWaveform, PwlWaveform, ExpWaveform>;

struct VoltageSource {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  std::optional<double> dc_volts;
  std::optional<AcSourceSpecification> ac;
  std::optional<TransientWaveform> transient = std::nullopt;
};

struct CurrentSource {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  std::optional<double> dc_amperes;
  std::optional<AcSourceSpecification> ac;
  std::optional<TransientWaveform> transient = std::nullopt;
};

struct Diode {
  std::string name;
  std::string positive_node;
  std::string negative_node;
  std::string model_name;
};

struct DiodeModel {
  std::string name;
  double saturation_current_amperes = 1e-14;
  double ideality_factor = 1.0;
};

struct Bjt {
  std::string name;
  std::string collector_node;
  std::string base_node;
  std::string emitter_node;
  std::string model_name;
};

struct BjtModel {
  std::string name;
  double saturation_current_amperes = 1e-16;
  double forward_current_gain = 100.0;
  double reverse_current_gain = 1.0;
  double forward_ideality_factor = 1.0;
  double reverse_ideality_factor = 1.0;
  bool is_npn = true;
};

using Component = std::variant<Resistor, Capacitor, Inductor, VoltageSource,
                               CurrentSource, Diode, Bjt>;

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

struct TranAnalysis {
  double time_step_seconds;
  double stop_time_seconds;
  double start_time_seconds;
  bool use_initial_conditions;
};

using Analysis = std::variant<DcAnalysis, AcAnalysis, TranAnalysis>;

struct Circuit {
  std::vector<Component> components;
  std::vector<Analysis> analyses;
  std::vector<DiodeModel> diode_models = {};
  std::vector<BjtModel> bjt_models = {};
};

} // namespace ohmnivore

#endif // OHMNIVORE_IR_H_
