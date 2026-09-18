#include "ohmnivore/compiler.h"

#include "ohmnivore/waveform.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ohmnivore {
namespace {

using Coordinate = std::pair<std::size_t, std::size_t>;

[[nodiscard]] bool IsGround(const std::string &node) {
  if (node == "0") {
    return true;
  }
  if (node.size() != 3) {
    return false;
  }
  return (node[0] == 'G' || node[0] == 'g') &&
         (node[1] == 'N' || node[1] == 'n') &&
         (node[2] == 'D' || node[2] == 'd');
}

void RegisterNode(const std::string &node,
                  std::unordered_map<std::string, std::size_t> *node_map,
                  std::vector<std::string> *node_names) {
  if (IsGround(node) || node_map->contains(node)) {
    return;
  }
  const std::size_t index = node_names->size();
  node_map->emplace(node, index);
  node_names->push_back(node);
}

[[nodiscard]] std::optional<std::size_t>
FindNode(const std::string &node,
         const std::unordered_map<std::string, std::size_t> &node_map) {
  if (IsGround(node)) {
    return std::nullopt;
  }
  const auto found = node_map.find(node);
  if (found == node_map.end()) {
    return std::nullopt;
  }
  return found->second;
}

void AddStamp(std::map<Coordinate, double> *entries, std::size_t row,
              std::size_t column, double value) {
  (*entries)[{row, column}] += value;
}

[[nodiscard]] CsrMatrix
BuildCsr(std::size_t size, const std::map<Coordinate, double> &entries,
         const std::set<Coordinate> &retained_zeros = {}) {
  std::map<Coordinate, double> union_entries = entries;
  for (const Coordinate &coordinate : retained_zeros) {
    union_entries.try_emplace(coordinate, 0.0);
  }
  CsrMatrix matrix;
  matrix.rows = size;
  matrix.columns = size;
  matrix.row_offsets.assign(size + 1, 0);

  for (const auto &[coordinate, value] : union_entries) {
    if (value == 0.0 && !retained_zeros.contains(coordinate)) {
      continue;
    }
    matrix.values.push_back(value);
    matrix.column_indices.push_back(coordinate.second);
    ++matrix.row_offsets[coordinate.first + 1];
  }
  for (std::size_t row = 0; row < size; ++row) {
    matrix.row_offsets[row + 1] += matrix.row_offsets[row];
  }
  return matrix;
}

[[nodiscard]] bool EqualCaseInsensitive(std::string_view first,
                                        std::string_view second) {
  if (first.size() != second.size()) {
    return false;
  }
  for (std::size_t index = 0; index < first.size(); ++index) {
    char left = first[index];
    char right = second[index];
    if (left >= 'a' && left <= 'z') {
      left = static_cast<char>(left - 'a' + 'A');
    }
    if (right >= 'a' && right <= 'z') {
      right = static_cast<char>(right - 'a' + 'A');
    }
    if (left != right) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool IsAsciiModelIdentifier(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_';
  });
}

struct ResolvedInductorCoupling {
  const Inductor *first;
  const Inductor *second;
  double mutual_henries;
};

[[nodiscard]] Result<std::vector<ResolvedInductorCoupling>>
ResolveInductorCouplings(const Circuit &circuit) {
  using CouplingsResult = Result<std::vector<ResolvedInductorCoupling>>;
  // These are separate passes: a bad later coefficient must take precedence
  // over an earlier identity or inductance error in direct IR.
  for (const InductorCoupling &coupling : circuit.inductor_couplings) {
    if (!std::isfinite(coupling.coefficient)) {
      return CouplingsResult::Fail(ErrorCode::kNonFinite,
                                   "coupling coefficient is non-finite");
    }
  }
  for (const InductorCoupling &coupling : circuit.inductor_couplings) {
    if (std::abs(coupling.coefficient) > 0.999) {
      return CouplingsResult::Fail(
          ErrorCode::kUnsupported,
          "coupling coefficient must be in [-0.999,0.999]");
    }
  }
  std::vector<ResolvedInductorCoupling> resolved;
  std::set<const Inductor *> used_windings;
  for (std::size_t index = 0; index < circuit.inductor_couplings.size();
       ++index) {
    const InductorCoupling &coupling = circuit.inductor_couplings[index];
    if (coupling.name.size() < 2 ||
        (coupling.name.front() != 'K' && coupling.name.front() != 'k') ||
        !IsAsciiModelIdentifier(coupling.name)) {
      return CouplingsResult::Fail(ErrorCode::kCompile,
                                   "invalid coupling identity '" +
                                       coupling.name + "'");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (EqualCaseInsensitive(coupling.name,
                               circuit.inductor_couplings[prior].name)) {
        return CouplingsResult::Fail(ErrorCode::kCompile,
                                     "duplicate coupling identity '" +
                                         coupling.name + "'");
      }
    }
    if (coupling.first_inductor.empty() || coupling.second_inductor.empty()) {
      return CouplingsResult::Fail(
          ErrorCode::kCompile, "coupling winding references must be nonempty");
    }
    std::array<const Inductor *, 2> windings = {nullptr, nullptr};
    const std::array<std::string_view, 2> references = {
        coupling.first_inductor, coupling.second_inductor};
    for (std::size_t winding = 0; winding < 2; ++winding) {
      std::size_t matches = 0;
      for (const Component &component : circuit.components) {
        const bool matches_name = std::visit(
            [&](const auto &typed) {
              return EqualCaseInsensitive(typed.name, references[winding]);
            },
            component);
        if (matches_name) {
          ++matches;
          windings[winding] = std::get_if<Inductor>(&component);
        }
      }
      if (matches != 1 || windings[winding] == nullptr) {
        return CouplingsResult::Fail(
            ErrorCode::kCompile,
            "coupling winding '" + std::string(references[winding]) +
                "' must identify exactly one linear inductor");
      }
    }
    if (windings[0] == windings[1] || used_windings.contains(windings[0]) ||
        used_windings.contains(windings[1])) {
      return CouplingsResult::Fail(
          ErrorCode::kCompile,
          "coupling windings must form disjoint distinct pairs");
    }
    used_windings.insert(windings[0]);
    used_windings.insert(windings[1]);
    resolved.push_back(
        {.first = windings[0], .second = windings[1], .mutual_henries = 0.0});
  }
  for (std::size_t index = 0; index < resolved.size(); ++index) {
    ResolvedInductorCoupling &pair = resolved[index];
    const double first = pair.first->inductance_henries;
    const double second = pair.second->inductance_henries;
    if (!std::isfinite(first) || !std::isfinite(second)) {
      return CouplingsResult::Fail(ErrorCode::kNonFinite,
                                   "coupled inductance is non-finite");
    }
    if (first <= 0.0 || second <= 0.0) {
      return CouplingsResult::Fail(ErrorCode::kCompile,
                                   "coupled inductance must be positive");
    }
    const double coefficient = circuit.inductor_couplings[index].coefficient;
    if (coefficient == 0.0) {
      continue;
    }
    // Multiply mantissas separately from exponents. Neither L1*L2 nor a
    // partial k*sqrt(L) product is necessarily representable even when M is.
    int first_exponent = 0;
    int second_exponent = 0;
    int coefficient_exponent = 0;
    const double first_mantissa = std::frexp(std::sqrt(first), &first_exponent);
    const double second_mantissa =
        std::frexp(std::sqrt(second), &second_exponent);
    const double coefficient_mantissa =
        std::frexp(coefficient, &coefficient_exponent);
    pair.mutual_henries =
        std::scalbn(coefficient_mantissa * first_mantissa * second_mantissa,
                    coefficient_exponent + first_exponent + second_exponent);
    if (!std::isfinite(pair.mutual_henries) || pair.mutual_henries == 0.0) {
      return CouplingsResult::Fail(
          ErrorCode::kNonFinite,
          "nonzero mutual inductance is not representable");
    }
    int mutual_exponent = 0;
    const double mutual_mantissa =
        std::frexp(std::abs(pair.mutual_henries), &mutual_exponent);
    const double represented_coupling =
        std::scalbn(mutual_mantissa / (first_mantissa * second_mantissa),
                    mutual_exponent - first_exponent - second_exponent);
    if (!std::isfinite(represented_coupling) || represented_coupling >= 1.0) {
      return CouplingsResult::Fail(
          ErrorCode::kNonFinite,
          "represented coupled inductance matrix is not positive definite");
    }
  }
  return CouplingsResult::Ok(std::move(resolved));
}

[[nodiscard]] std::optional<std::size_t>
FindValueIndex(const CsrMatrix &matrix, std::size_t row, std::size_t column) {
  for (std::size_t index = matrix.row_offsets[row];
       index < matrix.row_offsets[row + 1]; ++index) {
    if (matrix.column_indices[index] == column) {
      return index;
    }
    if (matrix.column_indices[index] > column) {
      break;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::complex<double>
AcToComplex(const AcSourceSpecification &specification) {
  const double reduced_phase_degrees =
      std::remainder(specification.phase_degrees, 360.0);
  const double phase_radians =
      reduced_phase_degrees * (std::numbers::pi / 180.0);
  return {specification.magnitude * std::cos(phase_radians),
          specification.magnitude * std::sin(phase_radians)};
}

[[nodiscard]] bool IsFinite(std::complex<double> value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

[[nodiscard]] std::optional<std::string>
ValidateTransientSource(const std::optional<TransientWaveform> &waveform) {
  if (!waveform.has_value()) {
    return std::nullopt;
  }
  const Result<double> evaluated = EvaluateTransientWaveform(*waveform, 0.0);
  if (!evaluated.ok()) {
    return evaluated.error().message;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string> ValidateCsr(const CsrMatrix &matrix) {
  if (matrix.row_offsets.size() != matrix.rows + 1 ||
      matrix.column_indices.size() != matrix.values.size()) {
    return "invalid CSR structure";
  }
  if (matrix.row_offsets.empty() || matrix.row_offsets.front() != 0 ||
      matrix.row_offsets.back() != matrix.values.size()) {
    return "invalid CSR row offsets";
  }
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
        matrix.row_offsets[row + 1] > matrix.values.size()) {
      return "invalid CSR row offsets";
    }
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      if (matrix.column_indices[index] >= matrix.columns) {
        return "CSR column index is outside the matrix";
      }
      if (index > matrix.row_offsets[row] &&
          matrix.column_indices[index - 1] >= matrix.column_indices[index]) {
        return "CSR columns must be strictly increasing within each row";
      }
      if (!std::isfinite(matrix.values[index])) {
        return "CSR matrix contains a non-finite value";
      }
    }
  }
  return std::nullopt;
}

} // namespace

Result<MnaSystem> CompileMna(const Circuit &circuit) {
  for (std::size_t index = 0; index < circuit.diode_models.size(); ++index) {
    const DiodeModel &model = circuit.diode_models[index];
    if (!IsAsciiModelIdentifier(model.name)) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "diode model name must contain only ASCII "
                                     "letters, digits, or underscore");
    }
    const double emission_voltage =
        model.ideality_factor * kDiodeThermalVoltageVolts;
    if (!std::isfinite(model.saturation_current_amperes) ||
        model.saturation_current_amperes <= 0.0 ||
        model.saturation_current_amperes > kDiodeMaximumParameterMagnitude ||
        !std::isfinite(model.ideality_factor) || model.ideality_factor <= 0.0 ||
        model.ideality_factor > kDiodeMaximumParameterMagnitude ||
        !std::isfinite(emission_voltage) || emission_voltage <= 0.0) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "diode model '" + model.name +
                                         "' has invalid IS or N data");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (EqualCaseInsensitive(model.name, circuit.diode_models[prior].name)) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile, "duplicate diode model name '" + model.name +
                                     "' under case-insensitive comparison");
      }
    }
  }

  for (std::size_t index = 0; index < circuit.bjt_models.size(); ++index) {
    const BjtModel &model = circuit.bjt_models[index];
    if (!IsAsciiModelIdentifier(model.name)) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "BJT model name must contain only ASCII "
                                     "letters, digits, or underscore");
    }
    const auto valid_parameter = [](double value) {
      return std::isfinite(value) && value > 0.0 &&
             value <= kDiodeMaximumParameterMagnitude;
    };
    const double forward_emission_voltage =
        model.forward_ideality_factor * kDiodeThermalVoltageVolts;
    const double reverse_emission_voltage =
        model.reverse_ideality_factor * kDiodeThermalVoltageVolts;
    if (!valid_parameter(model.saturation_current_amperes) ||
        !valid_parameter(model.forward_current_gain) ||
        !valid_parameter(model.reverse_current_gain) ||
        !valid_parameter(model.forward_ideality_factor) ||
        !valid_parameter(model.reverse_ideality_factor) ||
        !std::isfinite(forward_emission_voltage) ||
        forward_emission_voltage <= 0.0 ||
        forward_emission_voltage > kDiodeMaximumEmissionVoltageVolts ||
        !std::isfinite(reverse_emission_voltage) ||
        reverse_emission_voltage <= 0.0 ||
        reverse_emission_voltage > kDiodeMaximumEmissionVoltageVolts) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "BJT model '" + model.name +
                                         "' has invalid IS, BF, BR, NF, or "
                                         "NR data");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (EqualCaseInsensitive(model.name, circuit.bjt_models[prior].name)) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile, "duplicate model name '" + model.name +
                                     "' under case-insensitive comparison");
      }
    }
    for (const DiodeModel &diode_model : circuit.diode_models) {
      if (EqualCaseInsensitive(model.name, diode_model.name)) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile, "duplicate model name '" + model.name +
                                     "' under case-insensitive comparison");
      }
    }
  }

  auto couplings = ResolveInductorCouplings(circuit);
  if (!couplings.ok()) {
    return Result<MnaSystem>::Fail(couplings.error().code,
                                   couplings.error().message);
  }

  std::unordered_map<std::string, std::size_t> node_map;
  std::vector<std::string> node_names;
  for (const Component &component : circuit.components) {
    std::visit(
        [&](const auto &typed) {
          using Typed = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<Typed, Bjt>) {
            RegisterNode(typed.collector_node, &node_map, &node_names);
            RegisterNode(typed.base_node, &node_map, &node_names);
            RegisterNode(typed.emitter_node, &node_map, &node_names);
          } else {
            RegisterNode(typed.positive_node, &node_map, &node_names);
            RegisterNode(typed.negative_node, &node_map, &node_names);
          }
        },
        component);
  }

  std::unordered_map<std::string, std::size_t> branch_map;
  std::vector<std::string> branch_names;
  for (const Component &component : circuit.components) {
    const std::string *branch_name = nullptr;
    if (const auto *source = std::get_if<VoltageSource>(&component)) {
      branch_name = &source->name;
    } else if (const auto *inductor = std::get_if<Inductor>(&component)) {
      branch_name = &inductor->name;
    }
    if (branch_name != nullptr) {
      if (branch_map.contains(*branch_name)) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "duplicate branch-element name '" +
                                           *branch_name + "'");
      }
      branch_map.emplace(*branch_name, branch_names.size());
      branch_names.push_back(*branch_name);
    }
  }

  const std::size_t node_count = node_names.size();
  if (branch_names.size() >
      std::numeric_limits<std::size_t>::max() - node_count) {
    return Result<MnaSystem>::Fail(
        ErrorCode::kUnsupportedSize,
        "compiled MNA dimension is not representable");
  }
  const std::size_t size = node_count + branch_names.size();
  if (size == 0) {
    return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                   "circuit contains no solvable variables");
  }
  if (size >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Result<MnaSystem>::Fail(
        ErrorCode::kUnsupportedSize,
        "compiled MNA dimension exceeds the signed 32-bit KLU index bound");
  }

  std::map<Coordinate, double> g_entries;
  std::map<Coordinate, double> c_entries;
  std::set<Coordinate> coupling_coordinates;
  std::set<Coordinate> nonlinear_union_coordinates;
  std::vector<double> b_dc(size, 0.0);
  std::vector<std::complex<double>> b_ac(size, {0.0, 0.0});
  std::vector<TransientSourceStamp> transient_sources;
  std::vector<CapacitorInitialConstraint> capacitor_constraints;
  std::vector<InductorInitialConstraint> inductor_constraints;
  struct PendingDiode {
    std::string name;
    std::optional<std::size_t> anode;
    std::optional<std::size_t> cathode;
    double saturation_current_amperes;
    double emission_voltage_volts;
  };
  std::vector<PendingDiode> pending_diodes;
  struct PendingBjt {
    std::string name;
    std::optional<std::size_t> collector;
    std::optional<std::size_t> base;
    std::optional<std::size_t> emitter;
    double polarity;
    double saturation_current_amperes;
    double forward_current_gain;
    double reverse_current_gain;
    double forward_emission_voltage_volts;
    double reverse_emission_voltage_volts;
  };
  std::vector<PendingBjt> pending_bjts;
  for (std::size_t node = 0; node < node_count; ++node) {
    AddStamp(&g_entries, node, node, kGminSiemens);
  }

  for (const ResolvedInductorCoupling &coupling : couplings.value()) {
    const std::size_t first = node_count + branch_map.at(coupling.first->name);
    const std::size_t second =
        node_count + branch_map.at(coupling.second->name);
    AddStamp(&c_entries, first, second, -coupling.mutual_henries);
    AddStamp(&c_entries, second, first, -coupling.mutual_henries);
    coupling_coordinates.emplace(first, second);
    coupling_coordinates.emplace(second, first);
  }

  for (const Component &component : circuit.components) {
    if (const auto *resistor = std::get_if<Resistor>(&component)) {
      if (!std::isfinite(resistor->resistance_ohms) ||
          resistor->resistance_ohms <= 0.0) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "resistor '" + resistor->name +
                                           "' has invalid resistance");
      }
      const double conductance = 1.0 / resistor->resistance_ohms;
      const auto positive = FindNode(resistor->positive_node, node_map);
      const auto negative = FindNode(resistor->negative_node, node_map);
      if (positive == negative) {
        continue;
      }
      if (positive.has_value()) {
        AddStamp(&g_entries, *positive, *positive, conductance);
      }
      if (negative.has_value()) {
        AddStamp(&g_entries, *negative, *negative, conductance);
      }
      if (positive.has_value() && negative.has_value()) {
        AddStamp(&g_entries, *positive, *negative, -conductance);
        AddStamp(&g_entries, *negative, *positive, -conductance);
      }
      continue;
    }

    if (const auto *capacitor = std::get_if<Capacitor>(&component)) {
      if (!std::isfinite(capacitor->capacitance_farads) ||
          capacitor->capacitance_farads <= 0.0) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "capacitor '" + capacitor->name +
                                           "' has invalid capacitance");
      }
      const auto positive = FindNode(capacitor->positive_node, node_map);
      const auto negative = FindNode(capacitor->negative_node, node_map);
      if (positive == negative) {
        continue;
      }
      capacitor_constraints.push_back(CapacitorInitialConstraint{
          .name = capacitor->name,
          .positive_node_index = positive,
          .negative_node_index = negative,
          .capacitance_farads = capacitor->capacitance_farads,
      });
      if (positive.has_value()) {
        AddStamp(&c_entries, *positive, *positive,
                 capacitor->capacitance_farads);
      }
      if (negative.has_value()) {
        AddStamp(&c_entries, *negative, *negative,
                 capacitor->capacitance_farads);
      }
      if (positive.has_value() && negative.has_value()) {
        AddStamp(&c_entries, *positive, *negative,
                 -capacitor->capacitance_farads);
        AddStamp(&c_entries, *negative, *positive,
                 -capacitor->capacitance_farads);
      }
      continue;
    }

    if (const auto *current = std::get_if<CurrentSource>(&component)) {
      if (!current->dc_amperes.has_value() && !current->ac.has_value() &&
          !current->transient.has_value()) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile, "current source '" + current->name +
                                     "' has no DC, AC, or transient value");
      }
      if (current->dc_amperes.has_value() &&
          !std::isfinite(*current->dc_amperes)) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "current source '" + current->name +
                                           "' has invalid DC value");
      }
      if (current->ac.has_value() &&
          (!std::isfinite(current->ac->magnitude) ||
           current->ac->magnitude < 0.0 ||
           !std::isfinite(current->ac->phase_degrees) ||
           !IsFinite(AcToComplex(*current->ac)))) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "current source '" + current->name +
                                           "' has invalid AC value");
      }
      if (const auto error = ValidateTransientSource(current->transient);
          error.has_value()) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile,
            "current source '" + current->name +
                "' has invalid transient value: " + *error);
      }
      const auto positive = FindNode(current->positive_node, node_map);
      const auto negative = FindNode(current->negative_node, node_map);
      if (positive == negative) {
        continue;
      }
      if (current->dc_amperes.has_value()) {
        if (positive.has_value()) {
          b_dc[*positive] -= *current->dc_amperes;
        }
        if (negative.has_value()) {
          b_dc[*negative] += *current->dc_amperes;
        }
      }
      if (current->ac.has_value()) {
        const std::complex<double> ac_value = AcToComplex(*current->ac);
        if (positive.has_value()) {
          b_ac[*positive] -= ac_value;
        }
        if (negative.has_value()) {
          b_ac[*negative] += ac_value;
        }
      }
      if (current->transient.has_value()) {
        TransientSourceStamp source{
            .name = current->name,
            .dc_value = current->dc_amperes.value_or(0.0),
            .waveform = *current->transient,
            .rhs_stamps = {},
        };
        if (positive.has_value()) {
          source.rhs_stamps.push_back(
              TransientRhsStamp{.index = *positive, .coefficient = -1.0});
        }
        if (negative.has_value()) {
          source.rhs_stamps.push_back(
              TransientRhsStamp{.index = *negative, .coefficient = 1.0});
        }
        transient_sources.push_back(std::move(source));
      }
      continue;
    }

    if (const auto *diode = std::get_if<Diode>(&component)) {
      if (diode->name.empty() || diode->positive_node.empty() ||
          diode->negative_node.empty() || diode->model_name.empty()) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kInvalidStructure,
            "diode instances require non-empty name, terminals, and model "
            "reference");
      }
      if (!IsAsciiModelIdentifier(diode->model_name)) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile,
            "diode '" + diode->name +
                "' model reference must contain only ASCII letters, digits, "
                "or underscore");
      }
      const auto model = std::find_if(
          circuit.diode_models.begin(), circuit.diode_models.end(),
          [&](const DiodeModel &candidate) {
            return EqualCaseInsensitive(candidate.name, diode->model_name);
          });
      if (model == circuit.diode_models.end()) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "diode '" + diode->name +
                                           "' references missing model '" +
                                           diode->model_name + "'");
      }
      const auto anode = FindNode(diode->positive_node, node_map);
      const auto cathode = FindNode(diode->negative_node, node_map);
      if (anode == cathode) {
        return Result<MnaSystem>::Fail(ErrorCode::kInvalidStructure,
                                       "diode '" + diode->name +
                                           "' connects a node to itself");
      }
      if (anode.has_value()) {
        nonlinear_union_coordinates.emplace(*anode, *anode);
      }
      if (cathode.has_value()) {
        nonlinear_union_coordinates.emplace(*cathode, *cathode);
      }
      if (anode.has_value() && cathode.has_value()) {
        nonlinear_union_coordinates.emplace(*anode, *cathode);
        nonlinear_union_coordinates.emplace(*cathode, *anode);
      }
      pending_diodes.push_back(PendingDiode{
          .name = diode->name,
          .anode = anode,
          .cathode = cathode,
          .saturation_current_amperes = model->saturation_current_amperes,
          .emission_voltage_volts =
              model->ideality_factor * kDiodeThermalVoltageVolts,
      });
      continue;
    }

    if (const auto *bjt = std::get_if<Bjt>(&component)) {
      if (bjt->name.empty() || bjt->collector_node.empty() ||
          bjt->base_node.empty() || bjt->emitter_node.empty() ||
          bjt->model_name.empty()) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kInvalidStructure,
            "BJT instances require non-empty name, terminals, and model "
            "reference");
      }
      if (!IsAsciiModelIdentifier(bjt->model_name)) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile,
            "BJT '" + bjt->name +
                "' model reference must contain only ASCII letters, digits, "
                "or underscore");
      }
      const auto model = std::find_if(
          circuit.bjt_models.begin(), circuit.bjt_models.end(),
          [&](const BjtModel &candidate) {
            return EqualCaseInsensitive(candidate.name, bjt->model_name);
          });
      if (model == circuit.bjt_models.end()) {
        const bool wrong_type = std::any_of(
            circuit.diode_models.begin(), circuit.diode_models.end(),
            [&](const DiodeModel &item) {
              return EqualCaseInsensitive(item.name, bjt->model_name);
            });
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile, "BJT '" + bjt->name + "' references " +
                                     (wrong_type ? "non-BJT" : "missing") +
                                     " model '" + bjt->model_name + "'");
      }
      const std::array<std::optional<std::size_t>, 3> terminals = {
          FindNode(bjt->collector_node, node_map),
          FindNode(bjt->base_node, node_map),
          FindNode(bjt->emitter_node, node_map),
      };
      if (terminals[0] == terminals[1] && terminals[1] == terminals[2]) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kInvalidStructure,
            "BJT '" + bjt->name +
                "' connects all three terminals to the same electrical node");
      }
      for (const std::optional<std::size_t> row : terminals) {
        for (const std::optional<std::size_t> column : terminals) {
          if (row.has_value() && column.has_value()) {
            nonlinear_union_coordinates.emplace(*row, *column);
          }
        }
      }
      pending_bjts.push_back(PendingBjt{
          .name = bjt->name,
          .collector = terminals[0],
          .base = terminals[1],
          .emitter = terminals[2],
          .polarity = model->is_npn ? 1.0 : -1.0,
          .saturation_current_amperes = model->saturation_current_amperes,
          .forward_current_gain = model->forward_current_gain,
          .reverse_current_gain = model->reverse_current_gain,
          .forward_emission_voltage_volts =
              model->forward_ideality_factor * kDiodeThermalVoltageVolts,
          .reverse_emission_voltage_volts =
              model->reverse_ideality_factor * kDiodeThermalVoltageVolts,
      });
      continue;
    }

    const std::string *name = nullptr;
    const std::string *positive_node = nullptr;
    const std::string *negative_node = nullptr;
    if (const auto *inductor = std::get_if<Inductor>(&component)) {
      if (!std::isfinite(inductor->inductance_henries) ||
          inductor->inductance_henries <= 0.0) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "inductor '" + inductor->name +
                                           "' has invalid inductance");
      }
      name = &inductor->name;
      positive_node = &inductor->positive_node;
      negative_node = &inductor->negative_node;
    } else {
      const auto &source = std::get<VoltageSource>(component);
      if (!source.dc_volts.has_value() && !source.ac.has_value() &&
          !source.transient.has_value()) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile, "voltage source '" + source.name +
                                     "' has no DC, AC, or transient value");
      }
      if (source.dc_volts.has_value() && !std::isfinite(*source.dc_volts)) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "voltage source '" + source.name +
                                           "' has invalid DC value");
      }
      if (source.ac.has_value() &&
          (!std::isfinite(source.ac->magnitude) || source.ac->magnitude < 0.0 ||
           !std::isfinite(source.ac->phase_degrees) ||
           !IsFinite(AcToComplex(*source.ac)))) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "voltage source '" + source.name +
                                           "' has invalid AC value");
      }
      if (const auto error = ValidateTransientSource(source.transient);
          error.has_value()) {
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile,
            "voltage source '" + source.name +
                "' has invalid transient value: " + *error);
      }
      name = &source.name;
      positive_node = &source.positive_node;
      negative_node = &source.negative_node;
    }

    const auto positive = FindNode(*positive_node, node_map);
    const auto negative = FindNode(*negative_node, node_map);
    if (positive == negative) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "branch element '" + *name +
                                         "' connects a node to itself");
    }
    const std::size_t branch = node_count + branch_map.at(*name);
    if (positive.has_value()) {
      AddStamp(&g_entries, *positive, branch, 1.0);
      AddStamp(&g_entries, branch, *positive, 1.0);
    }
    if (negative.has_value()) {
      AddStamp(&g_entries, *negative, branch, -1.0);
      AddStamp(&g_entries, branch, *negative, -1.0);
    }

    if (const auto *inductor = std::get_if<Inductor>(&component)) {
      AddStamp(&c_entries, branch, branch, -inductor->inductance_henries);
      inductor_constraints.push_back(InductorInitialConstraint{
          .name = inductor->name,
          .branch_index = branch,
      });
    } else {
      const auto &source = std::get<VoltageSource>(component);
      if (source.dc_volts.has_value()) {
        b_dc[branch] = *source.dc_volts;
      }
      if (source.ac.has_value()) {
        b_ac[branch] = AcToComplex(*source.ac);
      }
      if (source.transient.has_value()) {
        transient_sources.push_back(TransientSourceStamp{
            .name = source.name,
            .dc_value = source.dc_volts.value_or(0.0),
            .waveform = *source.transient,
            .rhs_stamps = {TransientRhsStamp{.index = branch,
                                             .coefficient = 1.0}},
        });
      }
    }
  }

  for (const auto &entry : g_entries) {
    if (!std::isfinite(entry.second)) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kCompile,
          "conductance-matrix stamp accumulation produced a non-finite value");
    }
  }
  for (const auto &entry : c_entries) {
    if (!std::isfinite(entry.second)) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kCompile,
          "dynamic-matrix stamp accumulation produced a non-finite value");
    }
  }
  for (double value : b_dc) {
    if (!std::isfinite(value)) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kCompile,
          "DC right-hand-side accumulation produced a non-finite value");
    }
  }
  for (std::complex<double> value : b_ac) {
    if (!IsFinite(value)) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kCompile,
          "AC right-hand-side accumulation produced a non-finite value");
    }
  }

  CsrMatrix g = BuildCsr(size, g_entries, nonlinear_union_coordinates);
  if (g.values.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Result<MnaSystem>::Fail(
        ErrorCode::kUnsupportedSize,
        "compiled CSR nonzero count exceeds the signed 32-bit KLU "
        "index bound");
  }
  std::vector<DiodeDescriptor> diode_descriptors;
  diode_descriptors.reserve(pending_diodes.size());
  for (const PendingDiode &pending : pending_diodes) {
    const auto resolve =
        [&](std::optional<std::size_t> row,
            std::optional<std::size_t> column) -> std::optional<std::size_t> {
      if (!row.has_value() || !column.has_value()) {
        return std::nullopt;
      }
      return FindValueIndex(g, *row, *column);
    };
    DiodeDescriptor descriptor{
        .name = pending.name,
        .anode_node_index = pending.anode,
        .cathode_node_index = pending.cathode,
        .saturation_current_amperes = pending.saturation_current_amperes,
        .emission_voltage_volts = pending.emission_voltage_volts,
        .anode_anode_value_index = resolve(pending.anode, pending.anode),
        .anode_cathode_value_index = resolve(pending.anode, pending.cathode),
        .cathode_anode_value_index = resolve(pending.cathode, pending.anode),
        .cathode_cathode_value_index =
            resolve(pending.cathode, pending.cathode),
    };
    if ((pending.anode.has_value() &&
         !descriptor.anode_anode_value_index.has_value()) ||
        (pending.cathode.has_value() &&
         !descriptor.cathode_cathode_value_index.has_value()) ||
        (pending.anode.has_value() && pending.cathode.has_value() &&
         (!descriptor.anode_cathode_value_index.has_value() ||
          !descriptor.cathode_anode_value_index.has_value()))) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kInvalidStructure,
          "diode '" + pending.name +
              "' could not resolve its canonical CSR union coordinates");
    }
    diode_descriptors.push_back(std::move(descriptor));
  }

  std::vector<BjtDescriptor> bjt_descriptors;
  bjt_descriptors.reserve(pending_bjts.size());
  for (const PendingBjt &pending : pending_bjts) {
    const std::array<std::optional<std::size_t>, 3> terminals = {
        pending.collector, pending.base, pending.emitter};
    std::array<std::optional<std::size_t>, 9> positions;
    for (std::size_t row = 0; row < terminals.size(); ++row) {
      for (std::size_t column = 0; column < terminals.size(); ++column) {
        const std::size_t position = row * terminals.size() + column;
        if (!terminals[row].has_value() || !terminals[column].has_value()) {
          positions[position] = std::nullopt;
          continue;
        }
        positions[position] =
            FindValueIndex(g, *terminals[row], *terminals[column]);
        if (!positions[position].has_value()) {
          return Result<MnaSystem>::Fail(
              ErrorCode::kInvalidStructure,
              "BJT '" + pending.name +
                  "' could not resolve its canonical CSR union coordinates");
        }
      }
    }
    bjt_descriptors.push_back(BjtDescriptor{
        .name = pending.name,
        .collector_node_index = pending.collector,
        .base_node_index = pending.base,
        .emitter_node_index = pending.emitter,
        .polarity = pending.polarity,
        .saturation_current_amperes = pending.saturation_current_amperes,
        .forward_current_gain = pending.forward_current_gain,
        .reverse_current_gain = pending.reverse_current_gain,
        .forward_emission_voltage_volts =
            pending.forward_emission_voltage_volts,
        .reverse_emission_voltage_volts =
            pending.reverse_emission_voltage_volts,
        .jacobian_value_indices = positions,
    });
  }

  return Result<MnaSystem>::Ok(MnaSystem{
      .g = std::move(g),
      .c = BuildCsr(size, c_entries, coupling_coordinates),
      .b_dc = std::move(b_dc),
      .b_ac = std::move(b_ac),
      .node_names = std::move(node_names),
      .branch_names = std::move(branch_names),
      .transient_sources = std::move(transient_sources),
      .capacitor_initial_constraints = std::move(capacitor_constraints),
      .inductor_initial_constraints = std::move(inductor_constraints),
      .diode_descriptors = std::move(diode_descriptors),
      .bjt_descriptors = std::move(bjt_descriptors),
  });
}

Result<ComplexCsrMatrix> FormAcMatrix(const CsrMatrix &g, const CsrMatrix &c,
                                      double angular_frequency) {
  if (g.rows != g.columns || c.rows != c.columns || g.rows != c.rows ||
      g.columns != c.columns) {
    return Result<ComplexCsrMatrix>::Fail(
        ErrorCode::kCompile, "G and C must be square matrices of equal size");
  }
  if (!std::isfinite(angular_frequency) || angular_frequency < 0.0) {
    return Result<ComplexCsrMatrix>::Fail(
        ErrorCode::kCompile,
        "angular frequency must be finite and nonnegative");
  }
  if (const auto error = ValidateCsr(g); error.has_value()) {
    return Result<ComplexCsrMatrix>::Fail(ErrorCode::kCompile,
                                          "invalid G matrix: " + *error);
  }
  if (const auto error = ValidateCsr(c); error.has_value()) {
    return Result<ComplexCsrMatrix>::Fail(ErrorCode::kCompile,
                                          "invalid C matrix: " + *error);
  }

  ComplexCsrMatrix result;
  result.rows = g.rows;
  result.columns = g.columns;
  result.row_offsets.reserve(g.rows + 1);
  result.row_offsets.push_back(0);
  result.values.reserve(g.values.size() + c.values.size());
  result.column_indices.reserve(g.values.size() + c.values.size());

  for (std::size_t row = 0; row < g.rows; ++row) {
    std::size_t g_index = g.row_offsets[row];
    std::size_t c_index = c.row_offsets[row];
    const std::size_t g_end = g.row_offsets[row + 1];
    const std::size_t c_end = c.row_offsets[row + 1];
    while (g_index < g_end || c_index < c_end) {
      const bool take_g = c_index == c_end ||
                          (g_index < g_end && g.column_indices[g_index] <
                                                  c.column_indices[c_index]);
      const bool take_c = g_index == g_end ||
                          (c_index < c_end && c.column_indices[c_index] <
                                                  g.column_indices[g_index]);

      std::size_t column = 0;
      std::complex<double> value;
      if (take_g) {
        column = g.column_indices[g_index];
        value = {g.values[g_index], 0.0};
        ++g_index;
      } else if (take_c) {
        column = c.column_indices[c_index];
        value = {0.0, angular_frequency * c.values[c_index]};
        ++c_index;
      } else {
        column = g.column_indices[g_index];
        value = {g.values[g_index], angular_frequency * c.values[c_index]};
        ++g_index;
        ++c_index;
      }
      if (!IsFinite(value)) {
        return Result<ComplexCsrMatrix>::Fail(
            ErrorCode::kCompile,
            "AC matrix formation produced a non-finite value");
      }
      // Retain the deterministic G/C union pattern even when a particular
      // frequency produces an exact numerical zero. This lets the sparse
      // solver reuse one symbolic analysis throughout the sweep.
      result.column_indices.push_back(column);
      result.values.push_back(value);
    }
    result.row_offsets.push_back(result.values.size());
  }

  return Result<ComplexCsrMatrix>::Ok(std::move(result));
}

} // namespace ohmnivore
