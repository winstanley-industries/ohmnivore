#include "ohmnivore/compiler.h"

#include "ohmnivore/waveform.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <map>
#include <numbers>
#include <optional>
#include <string>
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

[[nodiscard]] CsrMatrix BuildCsr(std::size_t size,
                                 const std::map<Coordinate, double> &entries) {
  CsrMatrix matrix;
  matrix.rows = size;
  matrix.columns = size;
  matrix.row_offsets.assign(size + 1, 0);

  for (const auto &[coordinate, value] : entries) {
    if (value == 0.0) {
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
  std::unordered_map<std::string, std::size_t> node_map;
  std::vector<std::string> node_names;
  for (const Component &component : circuit.components) {
    std::visit(
        [&](const auto &typed) {
          RegisterNode(typed.positive_node, &node_map, &node_names);
          RegisterNode(typed.negative_node, &node_map, &node_names);
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
  const std::size_t size = node_count + branch_names.size();
  if (size == 0) {
    return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                   "circuit contains no solvable variables");
  }

  std::map<Coordinate, double> g_entries;
  std::map<Coordinate, double> c_entries;
  std::vector<double> b_dc(size, 0.0);
  std::vector<std::complex<double>> b_ac(size, {0.0, 0.0});
  std::vector<TransientSourceStamp> transient_sources;
  std::vector<CapacitorInitialConstraint> capacitor_constraints;
  std::vector<InductorInitialConstraint> inductor_constraints;
  for (std::size_t node = 0; node < node_count; ++node) {
    AddStamp(&g_entries, node, node, kGminSiemens);
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

  return Result<MnaSystem>::Ok(MnaSystem{
      .g = BuildCsr(size, g_entries),
      .c = BuildCsr(size, c_entries),
      .b_dc = std::move(b_dc),
      .b_ac = std::move(b_ac),
      .node_names = std::move(node_names),
      .branch_names = std::move(branch_names),
      .transient_sources = std::move(transient_sources),
      .capacitor_initial_constraints = std::move(capacitor_constraints),
      .inductor_initial_constraints = std::move(inductor_constraints),
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
      if (value != std::complex<double>{0.0, 0.0}) {
        result.column_indices.push_back(column);
        result.values.push_back(value);
      }
    }
    result.row_offsets.push_back(result.values.size());
  }

  return Result<ComplexCsrMatrix>::Ok(std::move(result));
}

} // namespace ohmnivore
