#include "ohmnivore/nonlinear.h"

#include "cpp/src/nonlinear_internal.h"
#include "ohmnivore/behavioral.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ohmnivore {
namespace {

struct AssembledNewtonSystem {
  CsrMatrix jacobian;
  std::vector<double> residual;
  std::vector<double> row_scales;
  std::vector<double> behavioral_rhs = {};
};

[[nodiscard]] bool IsBounded(double value) {
  return std::isfinite(value) && std::abs(value) <= kNonlinearMaximumMagnitude;
}

[[nodiscard]] std::optional<std::size_t>
FindValueIndex(const CsrMatrix &matrix, std::size_t row, std::size_t column) {
  if (row >= matrix.rows || matrix.row_offsets.size() != matrix.rows + 1) {
    return std::nullopt;
  }
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

[[nodiscard]] Result<bool>
ValidateDescriptor(const MnaSystem &system, const DiodeDescriptor &descriptor) {
  const std::size_t node_count = system.node_names.size();
  if (descriptor.name.empty()) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "diode descriptor name must not be empty");
  }
  if ((!descriptor.anode_node_index.has_value() &&
       !descriptor.cathode_node_index.has_value()) ||
      descriptor.anode_node_index == descriptor.cathode_node_index) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "diode descriptor '" + descriptor.name +
                                  "' has invalid or identical terminals");
  }
  for (const std::optional<std::size_t> node :
       {descriptor.anode_node_index, descriptor.cathode_node_index}) {
    if (node.has_value() && *node >= node_count) {
      return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                "diode descriptor '" + descriptor.name +
                                    "' references a non-node solution index");
    }
  }
  if (!std::isfinite(descriptor.saturation_current_amperes) ||
      descriptor.saturation_current_amperes <= 0.0 ||
      descriptor.saturation_current_amperes > kNonlinearMaximumMagnitude ||
      !std::isfinite(descriptor.emission_voltage_volts) ||
      descriptor.emission_voltage_volts <= 0.0 ||
      descriptor.emission_voltage_volts > kDiodeMaximumEmissionVoltageVolts) {
    return Result<bool>::Fail(ErrorCode::kCompile,
                              "diode descriptor '" + descriptor.name +
                                  "' has invalid model parameters");
  }

  const auto require_position = [&](std::optional<std::size_t> row,
                                    std::optional<std::size_t> column,
                                    std::optional<std::size_t> actual,
                                    const char *label) -> Result<bool> {
    if (!row.has_value() || !column.has_value()) {
      if (actual.has_value()) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "ground-related diode descriptor position " +
                                      std::string(label) + " must be absent");
      }
      return Result<bool>::Ok(true);
    }
    const auto expected = FindValueIndex(system.g, *row, *column);
    if (!expected.has_value() || actual != expected) {
      return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                "diode descriptor '" + descriptor.name +
                                    "' has malformed " + label +
                                    " CSR value index");
    }
    return Result<bool>::Ok(true);
  };

  for (Result<bool> position :
       {require_position(descriptor.anode_node_index,
                         descriptor.anode_node_index,
                         descriptor.anode_anode_value_index, "aa"),
        require_position(descriptor.anode_node_index,
                         descriptor.cathode_node_index,
                         descriptor.anode_cathode_value_index, "ac"),
        require_position(descriptor.cathode_node_index,
                         descriptor.anode_node_index,
                         descriptor.cathode_anode_value_index, "ca"),
        require_position(descriptor.cathode_node_index,
                         descriptor.cathode_node_index,
                         descriptor.cathode_cathode_value_index, "cc")}) {
    if (!position.ok()) {
      return position;
    }
  }
  return Result<bool>::Ok(true);
}

[[nodiscard]] Result<bool>
ValidateBjtDescriptor(const MnaSystem &system,
                      const BjtDescriptor &descriptor) {
  const std::size_t node_count = system.node_names.size();
  if (descriptor.name.empty()) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "BJT descriptor name must not be empty");
  }
  const std::array<std::optional<std::size_t>, 3> terminals = {
      descriptor.collector_node_index, descriptor.base_node_index,
      descriptor.emitter_node_index};
  if (terminals[0] == terminals[1] && terminals[1] == terminals[2]) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "BJT descriptor '" + descriptor.name +
                                  "' has one electrical terminal");
  }
  for (const std::optional<std::size_t> node : terminals) {
    if (node.has_value() && *node >= node_count) {
      return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                "BJT descriptor '" + descriptor.name +
                                    "' references a non-node solution index");
    }
  }
  const auto valid_parameter = [](double value) {
    return std::isfinite(value) && value > 0.0 &&
           value <= kNonlinearMaximumMagnitude;
  };
  if ((descriptor.polarity != 1.0 && descriptor.polarity != -1.0) ||
      !valid_parameter(descriptor.saturation_current_amperes) ||
      !valid_parameter(descriptor.forward_current_gain) ||
      !valid_parameter(descriptor.reverse_current_gain) ||
      !valid_parameter(descriptor.forward_emission_voltage_volts) ||
      descriptor.forward_emission_voltage_volts >
          kDiodeMaximumEmissionVoltageVolts ||
      !valid_parameter(descriptor.reverse_emission_voltage_volts) ||
      descriptor.reverse_emission_voltage_volts >
          kDiodeMaximumEmissionVoltageVolts) {
    return Result<bool>::Fail(ErrorCode::kCompile,
                              "BJT descriptor '" + descriptor.name +
                                  "' has invalid model parameters");
  }

  for (std::size_t row = 0; row < terminals.size(); ++row) {
    for (std::size_t column = 0; column < terminals.size(); ++column) {
      const std::size_t position = row * terminals.size() + column;
      const std::optional<std::size_t> actual =
          descriptor.jacobian_value_indices[position];
      if (!terminals[row].has_value() || !terminals[column].has_value()) {
        if (actual.has_value()) {
          return Result<bool>::Fail(
              ErrorCode::kInvalidStructure,
              "ground-related BJT descriptor position must be absent");
        }
        continue;
      }
      const auto expected =
          FindValueIndex(system.g, *terminals[row], *terminals[column]);
      if (!expected.has_value() || actual != expected) {
        return Result<bool>::Fail(
            ErrorCode::kInvalidStructure,
            "BJT descriptor '" + descriptor.name +
                "' has malformed CSR value index at row-major position " +
                std::to_string(position));
      }
    }
  }
  return Result<bool>::Ok(true);
}

[[nodiscard]] Result<bool> ValidateNonlinearSystem(const MnaSystem &system) {
  const std::size_t size =
      system.node_names.size() + system.branch_names.size();
  if (system.diode_descriptors.empty() && system.bjt_descriptors.empty() &&
      system.behavioral_descriptors.empty()) {
    return Result<bool>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear solve requires at least one nonlinear descriptor");
  }
  if (system.g.rows != size || system.g.columns != size ||
      system.b_dc.size() != size) {
    return Result<bool>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear MNA matrix, names, and right-hand side disagree");
  }
  Result<SolverCscPattern> structure = ConvertCsrToSolverCsc(system.g);
  if (!structure.ok()) {
    return Result<bool>::Fail(structure.error().code,
                              structure.error().message);
  }
  for (double value : system.g.values) {
    if (!IsBounded(value)) {
      return Result<bool>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear base matrix contains a non-finite or over-bound value");
    }
  }
  for (double value : system.b_dc) {
    if (!IsBounded(value)) {
      return Result<bool>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear right-hand side contains a non-finite or over-bound "
          "value");
    }
  }
  for (const DiodeDescriptor &descriptor : system.diode_descriptors) {
    Result<bool> valid = ValidateDescriptor(system, descriptor);
    if (!valid.ok()) {
      return valid;
    }
  }
  for (const BjtDescriptor &descriptor : system.bjt_descriptors) {
    Result<bool> valid = ValidateBjtDescriptor(system, descriptor);
    if (!valid.ok()) {
      return valid;
    }
  }
  auto behavioral = ValidateBehavioralDescriptors(system);
  if (!behavioral.ok())
    return behavioral;
  if (!system.behavioral_descriptors.empty()) {
    if (!system.diode_descriptors.empty() || !system.bjt_descriptors.empty()) {
      return Result<bool>::Fail(
          ErrorCode::kUnsupported,
          "behavioral points exclude native semiconductor mixtures");
    }
    for (const auto &capacitor : system.capacitor_initial_constraints) {
      if ((capacitor.positive_node_index &&
           *capacitor.positive_node_index >= system.node_names.size()) ||
          (capacitor.negative_node_index &&
           *capacitor.negative_node_index >= system.node_names.size()) ||
          capacitor.positive_node_index == capacitor.negative_node_index) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "invalid Newton capacitor coordinate");
      }
    }
    for (const auto &inductor : system.inductor_initial_constraints) {
      if (inductor.branch_index < system.node_names.size() ||
          inductor.branch_index >= size) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "invalid Newton inductor coordinate");
      }
    }
  }
  return Result<bool>::Ok(true);
}

[[nodiscard]] double NodeVoltage(const std::vector<double> &solution,
                                 std::optional<std::size_t> node) {
  return node.has_value() ? solution[*node] : 0.0;
}

[[nodiscard]] Result<AssembledNewtonSystem>
Assemble(const MnaSystem &system, const std::vector<double> &solution,
         double source_scale, double extra_gmin_siemens,
         const std::vector<bool> *active_node_equations = nullptr) {
  if (solution.size() != system.g.columns) {
    return Result<AssembledNewtonSystem>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear solution dimensions disagree with the MNA system");
  }
  if (active_node_equations != nullptr &&
      active_node_equations->size() != system.node_names.size()) {
    return Result<AssembledNewtonSystem>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear projection row mask dimensions disagree with the MNA "
        "system");
  }
  if (!std::isfinite(source_scale) || source_scale < 0.0 ||
      source_scale > 1.0 || !std::isfinite(extra_gmin_siemens) ||
      extra_gmin_siemens < 0.0 ||
      extra_gmin_siemens > kNonlinearMaximumMagnitude) {
    return Result<AssembledNewtonSystem>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear continuation values are outside their finite bounds");
  }
  for (double value : solution) {
    if (!IsBounded(value)) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear solution contains a non-finite or over-bound value");
    }
  }

  AssembledNewtonSystem assembled{
      .jacobian = system.g,
      .residual = std::vector<double>(system.g.rows, 0.0),
      .row_scales = std::vector<double>(system.g.rows, 0.0),
  };
  std::vector<long double> behavioral_rhs(
      system.behavioral_descriptors.empty() ? 0 : system.g.rows, 0.0L);
  for (std::size_t row = 0; row < system.g.rows; ++row) {
    double product = 0.0;
    const double scaled_source = source_scale * system.b_dc[row];
    if (!behavioral_rhs.empty())
      behavioral_rhs[row] = scaled_source;
    if (!IsBounded(scaled_source)) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear right-hand side contains a non-finite or over-bound "
          "value");
    }
    double scale = std::abs(scaled_source);
    for (std::size_t index = system.g.row_offsets[row];
         index < system.g.row_offsets[row + 1]; ++index) {
      const double term =
          system.g.values[index] * solution[system.g.column_indices[index]];
      product += term;
      scale += std::abs(term);
      if (!IsBounded(term) || !IsBounded(product) || !IsBounded(scale)) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite,
            "nonlinear linear-row evaluation produced a non-finite or "
            "over-bound intermediate");
      }
    }
    if (row < system.node_names.size() && extra_gmin_siemens != 0.0) {
      const auto diagonal = FindValueIndex(assembled.jacobian, row, row);
      if (!diagonal.has_value()) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kInvalidStructure,
            "extra GMIN requires every node diagonal in the union pattern");
      }
      assembled.jacobian.values[*diagonal] += extra_gmin_siemens;
      const double term = extra_gmin_siemens * solution[row];
      product += term;
      scale += std::abs(term);
      if (!IsBounded(term) || !IsBounded(product) || !IsBounded(scale)) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite,
            "extra GMIN evaluation produced a non-finite or over-bound "
            "intermediate");
      }
    }
    assembled.residual[row] = product - scaled_source;
    assembled.row_scales[row] = scale;
    if (!IsBounded(assembled.residual[row]) ||
        !IsBounded(assembled.row_scales[row])) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear residual initialization produced a non-finite or "
          "over-bound intermediate");
    }
  }

  for (const DiodeDescriptor &descriptor : system.diode_descriptors) {
    const double junction_voltage =
        NodeVoltage(solution, descriptor.anode_node_index) -
        NodeVoltage(solution, descriptor.cathode_node_index);
    Result<DiodeEvaluation> evaluated =
        EvaluateDiode(junction_voltage, descriptor.saturation_current_amperes,
                      descriptor.emission_voltage_volts);
    if (!evaluated.ok()) {
      return Result<AssembledNewtonSystem>::Fail(
          evaluated.error().code,
          "diode '" + descriptor.name + "': " + evaluated.error().message);
    }
    const double current = evaluated.value().current_amperes;
    const double conductance = evaluated.value().conductance_siemens;
    const auto equation_is_active = [&](std::optional<std::size_t> node) {
      return node.has_value() && (active_node_equations == nullptr ||
                                  (*active_node_equations)[*node]);
    };
    if (equation_is_active(descriptor.anode_node_index)) {
      const std::size_t row = *descriptor.anode_node_index;
      assembled.residual[row] += current;
      assembled.row_scales[row] += std::abs(current);
      assembled.jacobian.values[*descriptor.anode_anode_value_index] +=
          conductance;
      if (descriptor.anode_cathode_value_index.has_value()) {
        assembled.jacobian.values[*descriptor.anode_cathode_value_index] -=
            conductance;
      }
    }
    if (equation_is_active(descriptor.cathode_node_index)) {
      const std::size_t row = *descriptor.cathode_node_index;
      assembled.residual[row] -= current;
      assembled.row_scales[row] += std::abs(current);
      assembled.jacobian.values[*descriptor.cathode_cathode_value_index] +=
          conductance;
      if (descriptor.cathode_anode_value_index.has_value()) {
        assembled.jacobian.values[*descriptor.cathode_anode_value_index] -=
            conductance;
      }
    }
    for (const std::optional<std::size_t> row :
         {equation_is_active(descriptor.anode_node_index)
              ? descriptor.anode_node_index
              : std::nullopt,
          equation_is_active(descriptor.cathode_node_index)
              ? descriptor.cathode_node_index
              : std::nullopt}) {
      if (row.has_value() && (!IsBounded(assembled.residual[*row]) ||
                              !IsBounded(assembled.row_scales[*row]))) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite,
            "diode residual accumulation produced a non-finite or "
            "over-bound intermediate");
      }
    }
  }

  for (const BjtDescriptor &descriptor : system.bjt_descriptors) {
    const double collector_voltage =
        NodeVoltage(solution, descriptor.collector_node_index);
    const double base_voltage =
        NodeVoltage(solution, descriptor.base_node_index);
    const double emitter_voltage =
        NodeVoltage(solution, descriptor.emitter_node_index);
    Result<BjtEvaluation> evaluated = EvaluateBjt(
        collector_voltage, base_voltage, emitter_voltage, descriptor.polarity,
        descriptor.saturation_current_amperes, descriptor.forward_current_gain,
        descriptor.reverse_current_gain,
        descriptor.forward_emission_voltage_volts,
        descriptor.reverse_emission_voltage_volts);
    if (!evaluated.ok()) {
      return Result<AssembledNewtonSystem>::Fail(
          evaluated.error().code,
          "BJT '" + descriptor.name + "': " + evaluated.error().message);
    }
    const double emitter_current =
        -(evaluated.value().collector_current_amperes +
          evaluated.value().base_current_amperes);
    if (!IsBounded(emitter_current)) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "BJT terminal-current conservation produced a non-finite or "
          "over-bound value");
    }
    const std::array<double, 3> currents = {
        evaluated.value().collector_current_amperes,
        evaluated.value().base_current_amperes, emitter_current};
    const double collector_vbe =
        evaluated.value().collector_vbe_derivative_siemens;
    const double collector_vbc =
        evaluated.value().collector_vbc_derivative_siemens;
    const double base_vbe = evaluated.value().base_vbe_derivative_siemens;
    const double base_vbc = evaluated.value().base_vbc_derivative_siemens;
    const std::array<double, 9> jacobian = {
        -collector_vbc,
        collector_vbe + collector_vbc,
        -collector_vbe,
        -base_vbc,
        base_vbe + base_vbc,
        -base_vbe,
        collector_vbc + base_vbc,
        -(collector_vbe + collector_vbc + base_vbe + base_vbc),
        collector_vbe + base_vbe,
    };
    const std::array<std::optional<std::size_t>, 3> terminals = {
        descriptor.collector_node_index, descriptor.base_node_index,
        descriptor.emitter_node_index};
    for (double value : jacobian) {
      if (!IsBounded(value)) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite,
            "BJT Jacobian expansion produced a non-finite or over-bound "
            "value");
      }
    }
    for (std::size_t row = 0; row < terminals.size(); ++row) {
      if (!terminals[row].has_value() ||
          (active_node_equations != nullptr &&
           !(*active_node_equations)[*terminals[row]])) {
        continue;
      }
      const std::size_t physical_row = *terminals[row];
      assembled.residual[physical_row] += currents[row];
      assembled.row_scales[physical_row] += std::abs(currents[row]);
      if (!IsBounded(assembled.residual[physical_row]) ||
          !IsBounded(assembled.row_scales[physical_row])) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite,
            "BJT residual accumulation produced a non-finite or over-bound "
            "value");
      }
      for (std::size_t column = 0; column < terminals.size(); ++column) {
        if (!terminals[column].has_value()) {
          continue;
        }
        const std::size_t position = row * terminals.size() + column;
        const std::size_t value_index =
            *descriptor.jacobian_value_indices[position];
        assembled.jacobian.values[value_index] += jacobian[position];
        if (!IsBounded(assembled.jacobian.values[value_index])) {
          return Result<AssembledNewtonSystem>::Fail(
              ErrorCode::kNonFinite,
              "BJT Jacobian accumulation produced a non-finite or over-bound "
              "value");
        }
      }
    }
  }

  for (const BehavioralDescriptor &descriptor : system.behavioral_descriptors) {
    auto evaluated = EvaluateExpression(descriptor.expression, solution);
    if (!evaluated.ok())
      return Result<AssembledNewtonSystem>::Fail(evaluated.error().code,
                                                 descriptor.name + ": " +
                                                     evaluated.error().message);
    const auto dependencies = descriptor.expression.dependencies();
    long double affine_rhs = -static_cast<long double>(evaluated.value().value);
    for (const auto &[column, derivative] : evaluated.value().derivatives) {
      const long double term =
          static_cast<long double>(derivative) * solution[column];
      affine_rhs += term;
      if (!std::isfinite(term) || std::abs(term) > kNonlinearMaximumMagnitude ||
          !std::isfinite(affine_rhs) ||
          std::abs(affine_rhs) > kNonlinearMaximumMagnitude) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite, "behavioral affine RHS overflow");
      }
    }
    for (const auto &row : descriptor.rows) {
      if (active_node_equations != nullptr &&
          row.row < system.node_names.size() &&
          !(*active_node_equations)[row.row])
        continue;
      behavioral_rhs[row.row] += row.coefficient * affine_rhs;
      if (!std::isfinite(behavioral_rhs[row.row]) ||
          std::abs(behavioral_rhs[row.row]) > kNonlinearMaximumMagnitude) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite, "behavioral RHS accumulation overflow");
      }
      const double value = row.coefficient * evaluated.value().value;
      assembled.residual[row.row] += value;
      assembled.row_scales[row.row] += std::abs(value);
      for (const auto &[column, derivative] : evaluated.value().derivatives) {
        const auto found =
            std::lower_bound(dependencies.begin(), dependencies.end(), column);
        if (found == dependencies.end() || *found != column) {
          return Result<AssembledNewtonSystem>::Fail(
              ErrorCode::kInvalidStructure,
              "expression returned unbound derivative");
        }
        const auto offset =
            static_cast<std::size_t>(found - dependencies.begin());
        assembled.jacobian.values[row.jacobian_value_indices[offset]] +=
            row.coefficient * derivative;
      }
    }
  }

  assembled.behavioral_rhs.assign(behavioral_rhs.begin(), behavioral_rhs.end());

  for (double value : assembled.jacobian.values) {
    if (!IsBounded(value)) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear Jacobian contains a non-finite or over-bound value");
    }
  }
  for (double value : assembled.residual) {
    if (!IsBounded(value)) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear residual contains a non-finite or over-bound value");
    }
  }
  for (double value : assembled.row_scales) {
    if (!IsBounded(value)) {
      return Result<AssembledNewtonSystem>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear residual scale contains a non-finite or over-bound "
          "value");
    }
  }
  return Result<AssembledNewtonSystem>::Ok(std::move(assembled));
}

[[nodiscard]] Result<double>
MaximumNormalizedResidual(const MnaSystem &system,
                          const AssembledNewtonSystem &assembled) {
  double maximum = 0.0;
  for (std::size_t row = 0; row < assembled.residual.size(); ++row) {
    const bool behavioral = !system.behavioral_descriptors.empty();
    const double current_absolute =
        behavioral ? BehavioralNumericalPolicy::current_absolute_tolerance
                   : kNewtonCurrentAbsoluteTolerance;
    const double voltage_absolute =
        behavioral ? BehavioralNumericalPolicy::voltage_absolute_tolerance
                   : kNewtonVoltageAbsoluteTolerance;
    const double relative = behavioral
                                ? BehavioralNumericalPolicy::relative_tolerance
                                : kNewtonRelativeTolerance;
    const double absolute =
        row < system.node_names.size() ? current_absolute : voltage_absolute;
    const double tolerance = absolute + relative * assembled.row_scales[row];
    const double normalized = std::abs(assembled.residual[row]) / tolerance;
    if (!IsBounded(tolerance) || tolerance <= 0.0 || !IsBounded(normalized)) {
      return Result<double>::Fail(
          ErrorCode::kNonFinite,
          "nonlinear residual normalization produced a non-finite or "
          "over-bound value");
    }
    maximum = std::max(maximum, normalized);
  }
  return Result<double>::Ok(maximum);
}

[[nodiscard]] Result<double>
LimitAllJunctions(const MnaSystem &system, const std::vector<double> &previous,
                  std::vector<double> *proposed) {
  for (const DiodeDescriptor &descriptor : system.diode_descriptors) {
    const double previous_voltage =
        NodeVoltage(previous, descriptor.anode_node_index) -
        NodeVoltage(previous, descriptor.cathode_node_index);
    const double proposed_voltage =
        NodeVoltage(*proposed, descriptor.anode_node_index) -
        NodeVoltage(*proposed, descriptor.cathode_node_index);
    Result<double> limited =
        LimitDiodeJunctionVoltage(proposed_voltage, previous_voltage,
                                  descriptor.saturation_current_amperes,
                                  descriptor.emission_voltage_volts);
    if (!limited.ok()) {
      return Result<double>::Fail(limited.error().code,
                                  "diode '" + descriptor.name +
                                      "': " + limited.error().message);
    }
    if (limited.value() != proposed_voltage) {
      const double junction_delta = proposed_voltage - previous_voltage;
      if (junction_delta == 0.0 || !IsBounded(junction_delta)) {
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "diode limiting encountered an invalid junction update");
      }
      const double scale =
          (limited.value() - previous_voltage) / junction_delta;
      if (!IsBounded(scale)) {
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "diode limiting produced a non-finite or over-bound scale");
      }
      for (const std::optional<std::size_t> node :
           {descriptor.anode_node_index, descriptor.cathode_node_index}) {
        if (node.has_value()) {
          const double node_delta = (*proposed)[*node] - previous[*node];
          const double scaled_delta = node_delta * scale;
          const double updated = previous[*node] + scaled_delta;
          if (!IsBounded(node_delta) || !IsBounded(scaled_delta) ||
              !IsBounded(updated)) {
            return Result<double>::Fail(
                ErrorCode::kNonFinite,
                "diode limiting produced a non-finite or over-bound node "
                "update");
          }
          (*proposed)[*node] = updated;
        }
      }
    }
  }
  for (const BjtDescriptor &descriptor : system.bjt_descriptors) {
    const auto limit_junction = [&](std::optional<std::size_t> first,
                                    std::optional<std::size_t> second,
                                    double emission_voltage) -> Result<double> {
      const double previous_voltage =
          descriptor.polarity *
          (NodeVoltage(previous, first) - NodeVoltage(previous, second));
      const double proposed_voltage =
          descriptor.polarity *
          (NodeVoltage(*proposed, first) - NodeVoltage(*proposed, second));
      Result<double> limited = LimitDiodeJunctionVoltage(
          proposed_voltage, previous_voltage,
          descriptor.saturation_current_amperes, emission_voltage);
      if (!limited.ok()) {
        return limited;
      }
      if (limited.value() == proposed_voltage) {
        return Result<double>::Ok(0.0);
      }
      const double junction_delta = proposed_voltage - previous_voltage;
      if (junction_delta == 0.0 || !IsBounded(junction_delta)) {
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "BJT limiting encountered an invalid junction update");
      }
      const double scale =
          (limited.value() - previous_voltage) / junction_delta;
      if (!IsBounded(scale)) {
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "BJT limiting produced a non-finite or over-bound scale");
      }
      for (const std::optional<std::size_t> node : {first, second}) {
        if (!node.has_value()) {
          continue;
        }
        const double node_delta = (*proposed)[*node] - previous[*node];
        const double scaled_delta = node_delta * scale;
        const double updated = previous[*node] + scaled_delta;
        if (!IsBounded(node_delta) || !IsBounded(scaled_delta) ||
            !IsBounded(updated)) {
          return Result<double>::Fail(
              ErrorCode::kNonFinite,
              "BJT limiting produced a non-finite or over-bound node update");
        }
        (*proposed)[*node] = updated;
      }
      return Result<double>::Ok(0.0);
    };
    Result<double> limited_forward = limit_junction(
        descriptor.base_node_index, descriptor.emitter_node_index,
        descriptor.forward_emission_voltage_volts);
    if (!limited_forward.ok()) {
      return Result<double>::Fail(
          limited_forward.error().code,
          "BJT '" + descriptor.name +
              "' forward junction: " + limited_forward.error().message);
    }
    Result<double> limited_reverse = limit_junction(
        descriptor.base_node_index, descriptor.collector_node_index,
        descriptor.reverse_emission_voltage_volts);
    if (!limited_reverse.ok()) {
      return Result<double>::Fail(
          limited_reverse.error().code,
          "BJT '" + descriptor.name +
              "' reverse junction: " + limited_reverse.error().message);
    }
  }
  for (double value : *proposed) {
    if (!IsBounded(value)) {
      return Result<double>::Fail(
          ErrorCode::kNonFinite,
          "limited Newton iterate contains a non-finite or over-bound value");
    }
  }
  return Result<double>::Ok(0.0);
}

[[nodiscard]] Result<double>
MaximumNormalizedUpdate(const MnaSystem &system,
                        const std::vector<double> &previous,
                        const std::vector<double> &current) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < current.size(); ++index) {
    const bool behavioral = !system.behavioral_descriptors.empty();
    const double current_absolute =
        behavioral ? BehavioralNumericalPolicy::current_absolute_tolerance
                   : kNewtonCurrentAbsoluteTolerance;
    const double voltage_absolute =
        behavioral ? BehavioralNumericalPolicy::voltage_absolute_tolerance
                   : kNewtonVoltageAbsoluteTolerance;
    const double relative = behavioral
                                ? BehavioralNumericalPolicy::relative_tolerance
                                : kNewtonRelativeTolerance;
    const double absolute =
        index < system.node_names.size() ? voltage_absolute : current_absolute;
    const double tolerance =
        absolute + relative * std::max(std::abs(previous[index]),
                                       std::abs(current[index]));
    const double difference = current[index] - previous[index];
    const double normalized = std::abs(difference) / tolerance;
    if (!IsBounded(difference) || !IsBounded(tolerance) || tolerance <= 0.0 ||
        !IsBounded(normalized)) {
      return Result<double>::Fail(
          ErrorCode::kNonFinite,
          "Newton update normalization produced a non-finite or over-bound "
          "value");
    }
    maximum = std::max(maximum, normalized);
  }
  if (!system.behavioral_descriptors.empty()) {
    const auto compare = [&](double before, double after,
                             double absolute) -> Result<bool> {
      const double difference = after - before;
      if (!IsBounded(before) || !IsBounded(after) || !IsBounded(difference)) {
        return Result<bool>::Fail(ErrorCode::kNonFinite,
                                  "over-bound reactive Newton coordinate");
      }
      const double tolerance =
          BehavioralNumericalPolicy::newton_reactive_lte_fraction *
          (absolute + BehavioralNumericalPolicy::lte_relative_tolerance *
                          std::max(std::abs(before), std::abs(after)));
      const double normalized = std::abs(difference) / tolerance;
      if (!IsBounded(normalized) || !IsBounded(tolerance) || tolerance <= 0.0) {
        return Result<bool>::Fail(ErrorCode::kNonFinite,
                                  "non-finite reactive Newton update");
      }
      maximum = std::max(maximum, normalized);
      return Result<bool>::Ok(true);
    };
    for (const auto &capacitor : system.capacitor_initial_constraints) {
      const auto voltage = [&](const std::vector<double> &state) {
        return NodeVoltage(state, capacitor.positive_node_index) -
               NodeVoltage(state, capacitor.negative_node_index);
      };
      auto valid =
          compare(voltage(previous), voltage(current),
                  BehavioralNumericalPolicy::voltage_absolute_tolerance);
      if (!valid.ok())
        return Result<double>::Fail(valid.error().code, valid.error().message);
    }
    for (const auto &inductor : system.inductor_initial_constraints) {
      auto valid = compare(
          previous[inductor.branch_index], current[inductor.branch_index],
          BehavioralNumericalPolicy::current_absolute_tolerance);
      if (!valid.ok())
        return Result<double>::Fail(valid.error().code, valid.error().message);
    }
  }
  return Result<double>::Ok(maximum);
}

[[nodiscard]] Result<bool>
ValidateAcceptedJacobian(const AssembledNewtonSystem &assembled,
                         SparseRealFactorization *factorization) {
  std::vector<double> zero_right_hand_side(assembled.jacobian.rows, 0.0);
  Result<std::vector<double>> validated =
      factorization->FactorAndSolve(assembled.jacobian, zero_right_hand_side);
  if (!validated.ok()) {
    return Result<bool>::Fail(validated.error().code,
                              validated.error().message);
  }
  return Result<bool>::Ok(true);
}

struct AttemptResult {
  std::vector<double> solution;
  std::size_t iterations;
};

[[nodiscard]] Result<AttemptResult> RunNewtonAttempt(
    const MnaSystem &system, NonlinearStrategy strategy,
    double continuation_value, double source_scale, double extra_gmin_siemens,
    const std::vector<double> &initial_guess, std::size_t maximum_iterations,
    const std::vector<bool> *active_node_equations,
    SparseRealFactorization *factorization,
    std::vector<NonlinearIterationRecord> *iteration_trace) {
  std::vector<double> solution = initial_guess;
  Result<AssembledNewtonSystem> initial =
      Assemble(system, solution, source_scale, extra_gmin_siemens,
               active_node_equations);
  if (!initial.ok()) {
    return Result<AttemptResult>::Fail(initial.error().code,
                                       initial.error().message);
  }
  Result<double> initial_residual =
      MaximumNormalizedResidual(system, initial.value());
  if (!initial_residual.ok()) {
    return Result<AttemptResult>::Fail(initial_residual.error().code,
                                       initial_residual.error().message);
  }
  double residual = initial_residual.value();
  // A rounded FP64 companion residual can be zero at a wrong common-mode
  // state. Behavioral points must solve the stable affine RHS at least once.
  if (residual <= 1.0 && system.behavioral_descriptors.empty()) {
    Result<bool> valid_jacobian =
        ValidateAcceptedJacobian(initial.value(), factorization);
    if (!valid_jacobian.ok()) {
      return Result<AttemptResult>::Fail(valid_jacobian.error().code,
                                         valid_jacobian.error().message);
    }
    iteration_trace->push_back(NonlinearIterationRecord{
        .strategy = strategy,
        .continuation_value = continuation_value,
        .iteration = 0,
        .maximum_normalized_update = 0.0,
        .maximum_normalized_residual = residual,
        .accepted = true,
    });
    return Result<AttemptResult>::Ok(
        AttemptResult{.solution = std::move(solution), .iterations = 0});
  }

  // Assemblies are pure functions of this attempt's immutable system and
  // exact state. Behavioral trials can reuse their already checked assembly;
  // the caller still independently recomputes the final original residual.
  std::optional<AssembledNewtonSystem> current_assembly;
  if (!system.behavioral_descriptors.empty())
    current_assembly = initial.TakeValue();
  for (std::size_t iteration = 1; iteration <= maximum_iterations;
       ++iteration) {
    Result<AssembledNewtonSystem> assembled =
        current_assembly.has_value()
            ? Result<AssembledNewtonSystem>::Ok(std::move(*current_assembly))
            : Assemble(system, solution, source_scale, extra_gmin_siemens,
                       active_node_equations);
    current_assembly.reset();
    if (!assembled.ok()) {
      return Result<AttemptResult>::Fail(assembled.error().code,
                                         assembled.error().message);
    }
    std::vector<double> right_hand_side = assembled.value().residual;
    for (double &value : right_hand_side) {
      value = -value;
    }
    if (!system.behavioral_descriptors.empty()) {
      right_hand_side = assembled.value().behavioral_rhs;
    }
    Result<std::vector<double>> delta =
        system.behavioral_descriptors.empty()
            ? factorization->FactorAndSolve(assembled.value().jacobian,
                                            right_hand_side)
            : factorization->FactorAndSolveRefined(assembled.value().jacobian,
                                                   right_hand_side);
    if (!delta.ok()) {
      return Result<AttemptResult>::Fail(delta.error().code,
                                         delta.error().message);
    }
    std::vector<double> delta_values = delta.TakeValue();
    if (!system.behavioral_descriptors.empty()) {
      for (std::size_t i = 0; i < delta_values.size(); ++i)
        delta_values[i] -= solution[i];
    }
    std::vector<double> proposed = solution;
    for (std::size_t index = 0; index < proposed.size(); ++index) {
      const double update_value = delta_values[index];
      if (!IsBounded(update_value)) {
        return Result<AttemptResult>::Fail(
            ErrorCode::kNonFinite,
            "Newton delta contains a non-finite or over-bound value");
      }
      const double updated = proposed[index] + update_value;
      if (!IsBounded(updated) && system.behavioral_descriptors.empty()) {
        return Result<AttemptResult>::Fail(
            ErrorCode::kNonFinite,
            "Newton update produced a non-finite or over-bound value");
      }
      proposed[index] = updated;
    }
    Result<double> limited = LimitAllJunctions(system, solution, &proposed);
    if (!limited.ok()) {
      return Result<AttemptResult>::Fail(limited.error().code,
                                         limited.error().message);
    }
    std::optional<AssembledNewtonSystem> accepted_trial;
    if (!system.behavioral_descriptors.empty()) {
      bool decreased = false;
      const auto merit = [&](const AssembledNewtonSystem &value) {
        double maximum = 0.0;
        for (std::size_t row = 0; row < value.residual.size(); ++row) {
          const double absolute =
              row < system.node_names.size()
                  ? BehavioralNumericalPolicy::current_absolute_tolerance
                  : BehavioralNumericalPolicy::voltage_absolute_tolerance;
          const double tolerance =
              absolute + BehavioralNumericalPolicy::relative_tolerance *
                             assembled.value().row_scales[row];
          maximum =
              std::max(maximum, std::abs(value.residual[row]) / tolerance);
        }
        return maximum;
      };
      const double previous_merit = merit(assembled.value());
      double scale = 1.0;
      for (std::size_t backtrack = 0; backtrack <= 16; ++backtrack) {
        for (std::size_t j = 0; j < proposed.size(); ++j)
          proposed[j] = solution[j] + scale * delta_values[j];
        auto trial = Assemble(system, proposed, source_scale,
                              extra_gmin_siemens, active_node_equations);
        if (trial.ok()) {
          auto trial_norm = MaximumNormalizedResidual(system, trial.value());
          if (!trial_norm.ok()) {
            if (trial_norm.error().code != ErrorCode::kNonFinite) {
              return Result<AttemptResult>::Fail(trial_norm.error().code,
                                                 trial_norm.error().message);
            }
          } else if (strategy == NonlinearStrategy::kDirect ||
                     trial_norm.value() <= 1.0 ||
                     merit(trial.value()) < previous_merit) {
            accepted_trial = trial.TakeValue();
            decreased = true;
            break;
          }
        } else if (trial.error().code != ErrorCode::kNonFinite) {
          return Result<AttemptResult>::Fail(trial.error().code,
                                             trial.error().message);
        }
        scale *= 0.5;
      }
      if (!decreased)
        return Result<AttemptResult>::Fail(
            ErrorCode::kNonConvergence,
            "behavioral Newton backtracking exhausted");
    }
    Result<double> normalized_update =
        MaximumNormalizedUpdate(system, solution, proposed);
    if (!normalized_update.ok()) {
      return Result<AttemptResult>::Fail(normalized_update.error().code,
                                         normalized_update.error().message);
    }
    const double update = normalized_update.value();
    Result<AssembledNewtonSystem> checked =
        accepted_trial.has_value()
            ? Result<AssembledNewtonSystem>::Ok(std::move(*accepted_trial))
            : Assemble(system, proposed, source_scale, extra_gmin_siemens,
                       active_node_equations);
    if (!checked.ok()) {
      return Result<AttemptResult>::Fail(checked.error().code,
                                         checked.error().message);
    }
    Result<double> normalized_residual =
        MaximumNormalizedResidual(system, checked.value());
    if (!normalized_residual.ok()) {
      return Result<AttemptResult>::Fail(normalized_residual.error().code,
                                         normalized_residual.error().message);
    }
    residual = normalized_residual.value();
    const bool accepted = update <= 1.0 && residual <= 1.0;
    if (accepted) {
      Result<bool> valid_jacobian =
          ValidateAcceptedJacobian(checked.value(), factorization);
      if (!valid_jacobian.ok()) {
        return Result<AttemptResult>::Fail(valid_jacobian.error().code,
                                           valid_jacobian.error().message);
      }
    }
    iteration_trace->push_back(NonlinearIterationRecord{
        .strategy = strategy,
        .continuation_value = continuation_value,
        .iteration = iteration,
        .maximum_normalized_update = update,
        .maximum_normalized_residual = residual,
        .accepted = accepted,
    });
    solution = std::move(proposed);
    if (accepted) {
      return Result<AttemptResult>::Ok(AttemptResult{
          .solution = std::move(solution), .iterations = iteration});
    }
    if (!system.behavioral_descriptors.empty())
      current_assembly = checked.TakeValue();
  }
  return Result<AttemptResult>::Fail(
      ErrorCode::kNonConvergence,
      "Newton iteration exhausted its fixed maximum without update and "
      "residual convergence");
}

[[nodiscard]] Result<bool>
AcceptOriginalSystem(const MnaSystem &system,
                     const std::vector<double> &solution,
                     const std::vector<bool> *active_node_equations) {
  Result<AssembledNewtonSystem> assembled =
      Assemble(system, solution, 1.0, 0.0, active_node_equations);
  if (!assembled.ok()) {
    return Result<bool>::Fail(assembled.error().code,
                              assembled.error().message);
  }
  Result<double> normalized =
      MaximumNormalizedResidual(system, assembled.value());
  if (!normalized.ok()) {
    return Result<bool>::Fail(normalized.error().code,
                              normalized.error().message);
  }
  if (normalized.value() > 1.0) {
    return Result<bool>::Fail(
        ErrorCode::kSolutionValidation,
        "nonlinear solution failed original-system residual validation: "
        "maximum normalized residual=" +
            std::to_string(normalized.value()));
  }
  return Result<bool>::Ok(true);
}

} // namespace

Result<DiodeEvaluation> EvaluateDiode(double junction_voltage_volts,
                                      double saturation_current_amperes,
                                      double emission_voltage_volts) {
  if (!IsBounded(junction_voltage_volts) ||
      !IsBounded(saturation_current_amperes) ||
      saturation_current_amperes <= 0.0 || !IsBounded(emission_voltage_volts) ||
      emission_voltage_volts <= 0.0 ||
      emission_voltage_volts > kDiodeMaximumEmissionVoltageVolts) {
    return Result<DiodeEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "diode evaluation inputs are non-finite, over-bound, or nonpositive");
  }
  const double raw_exponent = junction_voltage_volts / emission_voltage_volts;
  if (!IsBounded(raw_exponent)) {
    return Result<DiodeEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "diode exponential argument is non-finite or over-bound");
  }
  const double exponent =
      std::clamp(raw_exponent, kDiodeMinimumExponent, kDiodeMaximumExponent);
  const double exponential = std::exp(exponent);
  const double current = saturation_current_amperes * std::expm1(exponent);
  const double conductance =
      saturation_current_amperes * exponential / emission_voltage_volts;
  if (!IsBounded(exponential) || !IsBounded(current) ||
      !IsBounded(conductance) || conductance <= 0.0) {
    return Result<DiodeEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "diode exponential evaluation produced a non-finite, underflowed, "
        "or over-bound result");
  }
  return Result<DiodeEvaluation>::Ok(DiodeEvaluation{
      .current_amperes = current,
      .conductance_siemens = conductance,
      .exponent = exponent,
  });
}

Result<BjtEvaluation>
EvaluateBjt(double collector_voltage_volts, double base_voltage_volts,
            double emitter_voltage_volts, double polarity,
            double saturation_current_amperes, double forward_current_gain,
            double reverse_current_gain, double forward_emission_voltage_volts,
            double reverse_emission_voltage_volts) {
  if (!IsBounded(collector_voltage_volts) || !IsBounded(base_voltage_volts) ||
      !IsBounded(emitter_voltage_volts) ||
      (polarity != 1.0 && polarity != -1.0) ||
      !IsBounded(saturation_current_amperes) ||
      saturation_current_amperes <= 0.0 || !IsBounded(forward_current_gain) ||
      forward_current_gain <= 0.0 || !IsBounded(reverse_current_gain) ||
      reverse_current_gain <= 0.0 ||
      !IsBounded(forward_emission_voltage_volts) ||
      forward_emission_voltage_volts <= 0.0 ||
      forward_emission_voltage_volts > kDiodeMaximumEmissionVoltageVolts ||
      !IsBounded(reverse_emission_voltage_volts) ||
      reverse_emission_voltage_volts <= 0.0 ||
      reverse_emission_voltage_volts > kDiodeMaximumEmissionVoltageVolts) {
    return Result<BjtEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "BJT evaluation inputs are non-finite, over-bound, or outside the "
        "model domain");
  }
  const double base_emitter_difference =
      base_voltage_volts - emitter_voltage_volts;
  const double base_collector_difference =
      base_voltage_volts - collector_voltage_volts;
  const double forward_voltage = polarity * base_emitter_difference;
  const double reverse_voltage = polarity * base_collector_difference;
  if (!IsBounded(base_emitter_difference) ||
      !IsBounded(base_collector_difference) || !IsBounded(forward_voltage) ||
      !IsBounded(reverse_voltage)) {
    return Result<BjtEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "BJT junction-voltage formation produced a non-finite or over-bound "
        "value");
  }
  Result<DiodeEvaluation> forward =
      EvaluateDiode(forward_voltage, saturation_current_amperes,
                    forward_emission_voltage_volts);
  if (!forward.ok()) {
    return Result<BjtEvaluation>::Fail(
        forward.error().code, "forward junction: " + forward.error().message);
  }
  Result<DiodeEvaluation> reverse =
      EvaluateDiode(reverse_voltage, saturation_current_amperes,
                    reverse_emission_voltage_volts);
  if (!reverse.ok()) {
    return Result<BjtEvaluation>::Fail(
        reverse.error().code, "reverse junction: " + reverse.error().message);
  }

  const double forward_denominator = forward_current_gain + 1.0;
  const double reverse_denominator = reverse_current_gain + 1.0;
  const double forward_alpha = forward_current_gain / forward_denominator;
  const double inverse_forward_denominator = 1.0 / forward_denominator;
  const double inverse_reverse_denominator = 1.0 / reverse_denominator;
  if (!IsBounded(forward_denominator) || forward_denominator <= 0.0 ||
      !IsBounded(reverse_denominator) || reverse_denominator <= 0.0 ||
      !IsBounded(forward_alpha) || forward_alpha <= 0.0 ||
      !IsBounded(inverse_forward_denominator) ||
      inverse_forward_denominator <= 0.0 ||
      !IsBounded(inverse_reverse_denominator) ||
      inverse_reverse_denominator <= 0.0) {
    return Result<BjtEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "BJT gain normalization underflowed or produced an invalid value");
  }

  const double collector_forward =
      forward_alpha * forward.value().current_amperes;
  const double collector_reverse =
      inverse_reverse_denominator * reverse.value().current_amperes;
  const double base_forward =
      inverse_forward_denominator * forward.value().current_amperes;
  const double base_reverse =
      inverse_reverse_denominator * reverse.value().current_amperes;
  const double collector_current =
      polarity * (collector_forward - collector_reverse);
  const double base_current = polarity * (base_forward + base_reverse);
  const double collector_vbe =
      forward_alpha * forward.value().conductance_siemens;
  const double collector_vbc =
      -inverse_reverse_denominator * reverse.value().conductance_siemens;
  const double base_vbe =
      inverse_forward_denominator * forward.value().conductance_siemens;
  const double base_vbc =
      inverse_reverse_denominator * reverse.value().conductance_siemens;
  for (double value : {collector_forward, collector_reverse, base_forward,
                       base_reverse, collector_current, base_current,
                       collector_vbe, collector_vbc, base_vbe, base_vbc}) {
    if (!IsBounded(value)) {
      return Result<BjtEvaluation>::Fail(
          ErrorCode::kNonFinite,
          "BJT current or derivative evaluation produced a non-finite or "
          "over-bound value");
    }
  }
  if (collector_vbe <= 0.0 || collector_vbc >= 0.0 || base_vbe <= 0.0 ||
      base_vbc <= 0.0) {
    return Result<BjtEvaluation>::Fail(
        ErrorCode::kNonFinite,
        "BJT junction derivative underflowed or has an invalid sign");
  }
  return Result<BjtEvaluation>::Ok(BjtEvaluation{
      .collector_current_amperes = collector_current,
      .base_current_amperes = base_current,
      .collector_vbe_derivative_siemens = collector_vbe,
      .collector_vbc_derivative_siemens = collector_vbc,
      .base_vbe_derivative_siemens = base_vbe,
      .base_vbc_derivative_siemens = base_vbc,
      .forward_exponent = forward.value().exponent,
      .reverse_exponent = reverse.value().exponent,
  });
}

Result<double> LimitDiodeJunctionVoltage(double proposed_voltage_volts,
                                         double previous_voltage_volts,
                                         double saturation_current_amperes,
                                         double emission_voltage_volts) {
  if (!IsBounded(proposed_voltage_volts) ||
      !IsBounded(previous_voltage_volts) ||
      !IsBounded(saturation_current_amperes) ||
      saturation_current_amperes <= 0.0 || !IsBounded(emission_voltage_volts) ||
      emission_voltage_volts <= 0.0 ||
      emission_voltage_volts > kDiodeMaximumEmissionVoltageVolts) {
    return Result<double>::Fail(
        ErrorCode::kNonFinite,
        "diode limiting inputs are non-finite, over-bound, or nonpositive");
  }
  const double critical_voltage =
      emission_voltage_volts *
      (std::log(emission_voltage_volts) - 0.5 * std::log(2.0) -
       std::log(saturation_current_amperes));
  if (!IsBounded(critical_voltage)) {
    return Result<double>::Fail(
        ErrorCode::kNonFinite,
        "diode critical-voltage calculation is non-finite or over-bound");
  }

  double limited = proposed_voltage_volts;
  const double voltage_change = proposed_voltage_volts - previous_voltage_volts;
  const double doubled_emission_voltage = 2.0 * emission_voltage_volts;
  if (!IsBounded(voltage_change) || !IsBounded(doubled_emission_voltage)) {
    return Result<double>::Fail(
        ErrorCode::kNonFinite,
        "diode limiting difference produced a non-finite or over-bound "
        "value");
  }
  if (proposed_voltage_volts > critical_voltage &&
      std::abs(voltage_change) > doubled_emission_voltage) {
    if (previous_voltage_volts > 0.0) {
      const double argument = 1.0 + voltage_change / emission_voltage_volts;
      if (!IsBounded(argument)) {
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "diode limiting logarithm argument is non-finite or over-bound");
      }
      limited = argument > 0.0 ? previous_voltage_volts +
                                     emission_voltage_volts * std::log(argument)
                               : critical_voltage;
    } else {
      const double ratio = proposed_voltage_volts / emission_voltage_volts;
      if (!IsBounded(ratio)) {
        return Result<double>::Fail(
            ErrorCode::kNonFinite,
            "diode limiting voltage ratio is non-finite or over-bound");
      }
      limited = ratio > 0.0 ? emission_voltage_volts * std::log(ratio)
                            : critical_voltage;
    }
  }
  if (!IsBounded(limited)) {
    return Result<double>::Fail(
        ErrorCode::kNonFinite,
        "diode voltage limiting produced a non-finite or over-bound value");
  }
  return Result<double>::Ok(limited);
}

Result<double> ValidateNonlinearResidual(const MnaSystem &system,
                                         const std::vector<double> &solution,
                                         double source_scale,
                                         double extra_gmin_siemens) {
  try {
    Result<bool> valid = ValidateNonlinearSystem(system);
    if (!valid.ok()) {
      return Result<double>::Fail(valid.error().code, valid.error().message);
    }
    Result<AssembledNewtonSystem> assembled =
        Assemble(system, solution, source_scale, extra_gmin_siemens);
    if (!assembled.ok()) {
      return Result<double>::Fail(assembled.error().code,
                                  assembled.error().message);
    }
    Result<double> normalized =
        MaximumNormalizedResidual(system, assembled.value());
    if (!normalized.ok()) {
      return Result<double>::Fail(normalized.error().code,
                                  normalized.error().message);
    }
    const double maximum = normalized.value();
    if (maximum > 1.0) {
      return Result<double>::Fail(
          ErrorCode::kSolutionValidation,
          "nonlinear solution failed original-system residual validation: "
          "maximum normalized residual=" +
              std::to_string(maximum));
    }
    return Result<double>::Ok(maximum);
  } catch (const std::bad_alloc &) {
    return Result<double>::Fail(
        ErrorCode::kFactorization,
        "nonlinear residual validation allocation failed");
  }
}

Result<NonlinearLinearization>
BuildNonlinearDcLinearization(const MnaSystem &system,
                              const std::vector<double> &solution,
                              double source_scale, double extra_gmin_siemens) {
  try {
    Result<bool> valid = ValidateNonlinearSystem(system);
    if (!valid.ok()) {
      return Result<NonlinearLinearization>::Fail(valid.error().code,
                                                  valid.error().message);
    }
    Result<AssembledNewtonSystem> assembled =
        Assemble(system, solution, source_scale, extra_gmin_siemens);
    if (!assembled.ok()) {
      return Result<NonlinearLinearization>::Fail(assembled.error().code,
                                                  assembled.error().message);
    }
    AssembledNewtonSystem linearization = assembled.TakeValue();
    return Result<NonlinearLinearization>::Ok(NonlinearLinearization{
        .jacobian = std::move(linearization.jacobian),
        .residual = std::move(linearization.residual),
    });
  } catch (const std::bad_alloc &) {
    return Result<NonlinearLinearization>::Fail(
        ErrorCode::kFactorization, "nonlinear linearization allocation failed");
  }
}

Result<std::vector<double>>
BuildDiodeResidualContribution(const MnaSystem &system,
                               const std::vector<double> &solution) {
  try {
    Result<bool> valid = ValidateNonlinearSystem(system);
    if (!valid.ok()) {
      return Result<std::vector<double>>::Fail(valid.error().code,
                                               valid.error().message);
    }
    if (solution.size() != system.g.columns) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kInvalidStructure,
          "diode residual solution dimensions disagree with the MNA system");
    }
    for (double value : solution) {
      if (!IsBounded(value)) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kNonFinite,
            "diode residual solution contains a non-finite or over-bound "
            "value");
      }
    }

    std::vector<double> residual(system.g.rows, 0.0);
    for (const DiodeDescriptor &descriptor : system.diode_descriptors) {
      const double junction_voltage =
          NodeVoltage(solution, descriptor.anode_node_index) -
          NodeVoltage(solution, descriptor.cathode_node_index);
      Result<DiodeEvaluation> evaluated =
          EvaluateDiode(junction_voltage, descriptor.saturation_current_amperes,
                        descriptor.emission_voltage_volts);
      if (!evaluated.ok()) {
        return Result<std::vector<double>>::Fail(
            evaluated.error().code,
            "diode '" + descriptor.name + "': " + evaluated.error().message);
      }
      if (descriptor.anode_node_index.has_value()) {
        residual[*descriptor.anode_node_index] +=
            evaluated.value().current_amperes;
        if (!IsBounded(residual[*descriptor.anode_node_index])) {
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite,
              "diode residual accumulation produced a non-finite or "
              "over-bound value");
        }
      }
      if (descriptor.cathode_node_index.has_value()) {
        residual[*descriptor.cathode_node_index] -=
            evaluated.value().current_amperes;
        if (!IsBounded(residual[*descriptor.cathode_node_index])) {
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite,
              "diode residual accumulation produced a non-finite or "
              "over-bound value");
        }
      }
    }
    for (const auto &descriptor : system.behavioral_descriptors) {
      auto evaluated = EvaluateExpression(descriptor.expression, solution);
      if (!evaluated.ok())
        return Result<std::vector<double>>::Fail(evaluated.error().code,
                                                 evaluated.error().message);
      for (const auto &row : descriptor.rows) {
        residual[row.row] += row.coefficient * evaluated.value().value;
        if (!IsBounded(residual[row.row]))
          return Result<std::vector<double>>::Fail(
              ErrorCode::kNonFinite, "behavioral history residual overflow");
      }
    }
    return Result<std::vector<double>>::Ok(std::move(residual));
  } catch (const std::bad_alloc &) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kFactorization, "diode residual allocation failed");
  }
}

namespace {

[[nodiscard]] Result<NonlinearPointResult> RunNonlinearPointImpl(
    const MnaSystem &system, const std::vector<double> &initial_guess,
    const std::vector<bool> *active_node_equations,
    SparseRealFactorization *factorization, std::size_t maximum_iterations) {
  if (factorization == nullptr) {
    return Result<NonlinearPointResult>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear point solve requires an analyzed KLU factorization");
  }
  if (maximum_iterations >
      (system.behavioral_descriptors.empty()
           ? kDirectNewtonMaximumIterations
           : BehavioralNumericalPolicy::dc_maximum_iterations)) {
    return Result<NonlinearPointResult>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear point iteration limit exceeds the fixed nonlinear bound");
  }
  if (active_node_equations != nullptr &&
      active_node_equations->size() != system.node_names.size()) {
    return Result<NonlinearPointResult>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear projection row mask dimensions disagree with the MNA "
        "system");
  }
  Result<bool> valid = ValidateNonlinearSystem(system);
  if (!valid.ok()) {
    return Result<NonlinearPointResult>::Fail(valid.error().code,
                                              valid.error().message);
  }
  std::vector<NonlinearIterationRecord> trace;
  Result<AttemptResult> solved = RunNewtonAttempt(
      system, NonlinearStrategy::kDirect, 1.0, 1.0, 0.0, initial_guess,
      maximum_iterations, active_node_equations, factorization, &trace);
  if (!solved.ok()) {
    return Result<NonlinearPointResult>::Fail(solved.error().code,
                                              solved.error().message);
  }
  Result<bool> accepted = AcceptOriginalSystem(system, solved.value().solution,
                                               active_node_equations);
  if (!accepted.ok()) {
    return Result<NonlinearPointResult>::Fail(accepted.error().code,
                                              accepted.error().message);
  }
  return Result<NonlinearPointResult>::Ok(NonlinearPointResult{
      .solution = solved.TakeValue().solution,
      .iteration_trace = std::move(trace),
  });
}

} // namespace

Result<NonlinearPointResult> RunNonlinearPoint(
    const MnaSystem &system, const std::vector<double> &initial_guess,
    SparseRealFactorization *factorization, std::size_t maximum_iterations) {
  try {
    return RunNonlinearPointImpl(system, initial_guess, nullptr, factorization,
                                 maximum_iterations);
  } catch (const std::bad_alloc &) {
    return Result<NonlinearPointResult>::Fail(
        ErrorCode::kFactorization, "nonlinear point allocation failed");
  }
}

Result<NonlinearPointResult> internal::RunNonlinearPointForProjection(
    const MnaSystem &system, const std::vector<double> &initial_guess,
    const std::vector<bool> &active_node_equations,
    SparseRealFactorization *factorization, std::size_t maximum_iterations) {
  try {
    return RunNonlinearPointImpl(system, initial_guess, &active_node_equations,
                                 factorization, maximum_iterations);
  } catch (const std::bad_alloc &) {
    return Result<NonlinearPointResult>::Fail(
        ErrorCode::kFactorization,
        "nonlinear projection point allocation failed");
  }
}

Result<NonlinearDcResult> RunNonlinearDc(const MnaSystem &system,
                                         const NonlinearDcOptions &options) {
  try {
    const std::size_t direct_bound =
        system.behavioral_descriptors.empty()
            ? kDirectNewtonMaximumIterations
            : BehavioralNumericalPolicy::dc_maximum_iterations;
    const std::size_t continuation_bound =
        system.behavioral_descriptors.empty()
            ? kContinuationNewtonMaximumIterations
            : BehavioralNumericalPolicy::dc_maximum_iterations;
    if (options.direct_maximum_iterations > direct_bound ||
        options.source_step_maximum_iterations > continuation_bound ||
        options.gmin_step_maximum_iterations > continuation_bound ||
        options.final_gmin_maximum_iterations > direct_bound) {
      return Result<NonlinearDcResult>::Fail(
          ErrorCode::kInvalidStructure,
          "nonlinear iteration options exceed the fixed Phase 3A bounds");
    }
    Result<bool> valid = ValidateNonlinearSystem(system);
    if (!valid.ok()) {
      return Result<NonlinearDcResult>::Fail(valid.error().code,
                                             valid.error().message);
    }
    Result<std::unique_ptr<SparseRealFactorization>> analyzed =
        SparseRealFactorization::Analyze(system.g);
    if (!analyzed.ok()) {
      return Result<NonlinearDcResult>::Fail(analyzed.error().code,
                                             analyzed.error().message);
    }
    std::unique_ptr<SparseRealFactorization> factorization =
        analyzed.TakeValue();
    NonlinearDcResult result;
    const std::vector<double> zero(system.g.rows, 0.0);

    const auto record_attempt = [&](NonlinearStrategy strategy, double value,
                                    std::size_t trace_start,
                                    const Result<AttemptResult> &attempt) {
      std::size_t iterations = attempt.ok() ? attempt.value().iterations : 0;
      for (std::size_t index = trace_start;
           index < result.iteration_trace.size(); ++index) {
        iterations =
            std::max(iterations, result.iteration_trace[index].iteration);
      }
      result.attempt_trace.push_back(NonlinearAttemptRecord{
          .strategy = strategy,
          .continuation_value = value,
          .iterations = iterations,
          .converged = attempt.ok(),
      });
    };
    const std::size_t direct_trace_start = result.iteration_trace.size();
    Result<AttemptResult> direct =
        RunNewtonAttempt(system, NonlinearStrategy::kDirect, 1.0, 1.0, 0.0,
                         zero, options.direct_maximum_iterations, nullptr,
                         factorization.get(), &result.iteration_trace);
    record_attempt(NonlinearStrategy::kDirect, 1.0, direct_trace_start, direct);
    if (direct.ok()) {
      Result<bool> accepted =
          AcceptOriginalSystem(system, direct.value().solution, nullptr);
      if (!accepted.ok()) {
        return Result<NonlinearDcResult>::Fail(accepted.error().code,
                                               accepted.error().message);
      }
      result.solution = direct.TakeValue().solution;
      result.solver_statistics = factorization->statistics();
      return Result<NonlinearDcResult>::Ok(std::move(result));
    }
    if (direct.error().code != ErrorCode::kNonConvergence) {
      return Result<NonlinearDcResult>::Fail(direct.error().code,
                                             direct.error().message);
    }

    std::vector<double> source_seed = zero;
    bool source_failed = false;
    for (std::size_t step = 0; step <= 10; ++step) {
      const double scale = static_cast<double>(step) / 10.0;
      const std::size_t trace_start = result.iteration_trace.size();
      Result<AttemptResult> attempt = RunNewtonAttempt(
          system, NonlinearStrategy::kSourceStepping, scale, scale, 0.0,
          source_seed, options.source_step_maximum_iterations, nullptr,
          factorization.get(), &result.iteration_trace);
      record_attempt(NonlinearStrategy::kSourceStepping, scale, trace_start,
                     attempt);
      if (!attempt.ok()) {
        if (attempt.error().code != ErrorCode::kNonConvergence) {
          return Result<NonlinearDcResult>::Fail(attempt.error().code,
                                                 attempt.error().message);
        }
        source_failed = true;
        break;
      }
      source_seed = attempt.TakeValue().solution;
    }
    if (!source_failed) {
      Result<bool> accepted =
          AcceptOriginalSystem(system, source_seed, nullptr);
      if (!accepted.ok()) {
        return Result<NonlinearDcResult>::Fail(accepted.error().code,
                                               accepted.error().message);
      }
      result.solution = std::move(source_seed);
      result.solver_statistics = factorization->statistics();
      return Result<NonlinearDcResult>::Ok(std::move(result));
    }

    constexpr std::array<double, 11> kExtraGminSchedule = {
        1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 0.0};
    std::vector<double> gmin_seed = source_seed;
    for (double extra_gmin : kExtraGminSchedule) {
      const std::size_t trace_start = result.iteration_trace.size();
      const std::size_t maximum_iterations =
          extra_gmin == 0.0 ? options.final_gmin_maximum_iterations
                            : options.gmin_step_maximum_iterations;
      Result<AttemptResult> attempt = RunNewtonAttempt(
          system, NonlinearStrategy::kGminStepping, extra_gmin, 1.0, extra_gmin,
          gmin_seed, maximum_iterations, nullptr, factorization.get(),
          &result.iteration_trace);
      record_attempt(NonlinearStrategy::kGminStepping, extra_gmin, trace_start,
                     attempt);
      if (!attempt.ok()) {
        if (attempt.error().code != ErrorCode::kNonConvergence) {
          return Result<NonlinearDcResult>::Fail(attempt.error().code,
                                                 attempt.error().message);
        }
        return Result<NonlinearDcResult>::Fail(
            ErrorCode::kNonConvergence,
            "direct Newton, fixed source stepping, and fixed GMIN stepping "
            "all failed within their bounded iteration schedules");
      }
      gmin_seed = attempt.TakeValue().solution;
    }
    Result<bool> accepted = AcceptOriginalSystem(system, gmin_seed, nullptr);
    if (!accepted.ok()) {
      return Result<NonlinearDcResult>::Fail(accepted.error().code,
                                             accepted.error().message);
    }
    result.solution = std::move(gmin_seed);
    result.solver_statistics = factorization->statistics();
    return Result<NonlinearDcResult>::Ok(std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<NonlinearDcResult>::Fail(ErrorCode::kFactorization,
                                           "nonlinear DC allocation failed");
  }
}

} // namespace ohmnivore
