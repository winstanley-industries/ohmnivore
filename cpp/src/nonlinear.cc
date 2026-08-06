#include "ohmnivore/nonlinear.h"

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

[[nodiscard]] Result<bool> ValidateNonlinearSystem(const MnaSystem &system) {
  const std::size_t size =
      system.node_names.size() + system.branch_names.size();
  if (system.diode_descriptors.empty()) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "nonlinear DC requires at least one diode");
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
  return Result<bool>::Ok(true);
}

[[nodiscard]] double NodeVoltage(const std::vector<double> &solution,
                                 std::optional<std::size_t> node) {
  return node.has_value() ? solution[*node] : 0.0;
}

[[nodiscard]] Result<AssembledNewtonSystem>
Assemble(const MnaSystem &system, const std::vector<double> &solution,
         double source_scale, double extra_gmin_siemens) {
  if (solution.size() != system.g.columns) {
    return Result<AssembledNewtonSystem>::Fail(
        ErrorCode::kInvalidStructure,
        "nonlinear solution dimensions disagree with the MNA system");
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
  for (std::size_t row = 0; row < system.g.rows; ++row) {
    double product = 0.0;
    const double scaled_source = source_scale * system.b_dc[row];
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
    if (descriptor.anode_node_index.has_value()) {
      const std::size_t row = *descriptor.anode_node_index;
      assembled.residual[row] += current;
      assembled.row_scales[row] += std::abs(current);
      assembled.jacobian.values[*descriptor.anode_anode_value_index] +=
          conductance;
    }
    if (descriptor.cathode_node_index.has_value()) {
      const std::size_t row = *descriptor.cathode_node_index;
      assembled.residual[row] -= current;
      assembled.row_scales[row] += std::abs(current);
      assembled.jacobian.values[*descriptor.cathode_cathode_value_index] +=
          conductance;
    }
    if (descriptor.anode_cathode_value_index.has_value()) {
      assembled.jacobian.values[*descriptor.anode_cathode_value_index] -=
          conductance;
      assembled.jacobian.values[*descriptor.cathode_anode_value_index] -=
          conductance;
    }
    for (const std::optional<std::size_t> row :
         {descriptor.anode_node_index, descriptor.cathode_node_index}) {
      if (row.has_value() && (!IsBounded(assembled.residual[*row]) ||
                              !IsBounded(assembled.row_scales[*row]))) {
        return Result<AssembledNewtonSystem>::Fail(
            ErrorCode::kNonFinite,
            "diode residual accumulation produced a non-finite or "
            "over-bound intermediate");
      }
    }
  }

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
    const double absolute = row < system.node_names.size()
                                ? kNewtonCurrentAbsoluteTolerance
                                : kNewtonVoltageAbsoluteTolerance;
    const double tolerance =
        absolute + kNewtonRelativeTolerance * assembled.row_scales[row];
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

[[nodiscard]] Result<double> LimitAllDiodes(const MnaSystem &system,
                                            const std::vector<double> &previous,
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
    const double absolute = index < system.node_names.size()
                                ? kNewtonVoltageAbsoluteTolerance
                                : kNewtonCurrentAbsoluteTolerance;
    const double tolerance = absolute + kNewtonRelativeTolerance *
                                            std::max(std::abs(previous[index]),
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
    SparseRealFactorization *factorization,
    std::vector<NonlinearIterationRecord> *iteration_trace) {
  std::vector<double> solution = initial_guess;
  Result<AssembledNewtonSystem> initial =
      Assemble(system, solution, source_scale, extra_gmin_siemens);
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
  if (residual <= 1.0) {
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

  for (std::size_t iteration = 1; iteration <= maximum_iterations;
       ++iteration) {
    Result<AssembledNewtonSystem> assembled =
        Assemble(system, solution, source_scale, extra_gmin_siemens);
    if (!assembled.ok()) {
      return Result<AttemptResult>::Fail(assembled.error().code,
                                         assembled.error().message);
    }
    std::vector<double> right_hand_side = assembled.value().residual;
    for (double &value : right_hand_side) {
      value = -value;
    }
    Result<std::vector<double>> delta = factorization->FactorAndSolve(
        assembled.value().jacobian, right_hand_side);
    if (!delta.ok()) {
      return Result<AttemptResult>::Fail(delta.error().code,
                                         delta.error().message);
    }
    std::vector<double> proposed = solution;
    for (std::size_t index = 0; index < proposed.size(); ++index) {
      const double update_value = delta.value()[index];
      if (!IsBounded(update_value)) {
        return Result<AttemptResult>::Fail(
            ErrorCode::kNonFinite,
            "Newton delta contains a non-finite or over-bound value");
      }
      const double updated = proposed[index] + update_value;
      if (!IsBounded(updated)) {
        return Result<AttemptResult>::Fail(
            ErrorCode::kNonFinite,
            "Newton update produced a non-finite or over-bound value");
      }
      proposed[index] = updated;
    }
    Result<double> limited = LimitAllDiodes(system, solution, &proposed);
    if (!limited.ok()) {
      return Result<AttemptResult>::Fail(limited.error().code,
                                         limited.error().message);
    }
    Result<double> normalized_update =
        MaximumNormalizedUpdate(system, solution, proposed);
    if (!normalized_update.ok()) {
      return Result<AttemptResult>::Fail(normalized_update.error().code,
                                         normalized_update.error().message);
    }
    const double update = normalized_update.value();
    Result<AssembledNewtonSystem> checked =
        Assemble(system, proposed, source_scale, extra_gmin_siemens);
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
  }
  return Result<AttemptResult>::Fail(
      ErrorCode::kNonConvergence,
      "Newton iteration exhausted its fixed maximum without update and "
      "residual convergence");
}

[[nodiscard]] Result<bool>
AcceptOriginalSystem(const MnaSystem &system,
                     const std::vector<double> &solution) {
  Result<double> validation =
      ValidateNonlinearResidual(system, solution, 1.0, 0.0);
  if (!validation.ok()) {
    return Result<bool>::Fail(validation.error().code,
                              validation.error().message);
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

Result<NonlinearDcResult> RunNonlinearDc(const MnaSystem &system,
                                         const NonlinearDcOptions &options) {
  try {
    if (options.direct_maximum_iterations > kDirectNewtonMaximumIterations ||
        options.source_step_maximum_iterations >
            kContinuationNewtonMaximumIterations ||
        options.gmin_step_maximum_iterations >
            kContinuationNewtonMaximumIterations ||
        options.final_gmin_maximum_iterations >
            kDirectNewtonMaximumIterations) {
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
                         zero, options.direct_maximum_iterations,
                         factorization.get(), &result.iteration_trace);
    record_attempt(NonlinearStrategy::kDirect, 1.0, direct_trace_start, direct);
    if (direct.ok()) {
      Result<bool> accepted =
          AcceptOriginalSystem(system, direct.value().solution);
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
          source_seed, options.source_step_maximum_iterations,
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
      Result<bool> accepted = AcceptOriginalSystem(system, source_seed);
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
      Result<AttemptResult> attempt =
          RunNewtonAttempt(system, NonlinearStrategy::kGminStepping, extra_gmin,
                           1.0, extra_gmin, gmin_seed, maximum_iterations,
                           factorization.get(), &result.iteration_trace);
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
    Result<bool> accepted = AcceptOriginalSystem(system, gmin_seed);
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
