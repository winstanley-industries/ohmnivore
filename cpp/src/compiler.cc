#include "ohmnivore/compiler.h"

#include <cmath>
#include <cstddef>
#include <map>
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
      if (!std::isfinite(current->dc_amperes)) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "current source '" + current->name +
                                           "' has invalid DC value");
      }
      const auto positive = FindNode(current->positive_node, node_map);
      const auto negative = FindNode(current->negative_node, node_map);
      if (positive == negative) {
        continue;
      }
      if (positive.has_value()) {
        b_dc[*positive] -= current->dc_amperes;
      }
      if (negative.has_value()) {
        b_dc[*negative] += current->dc_amperes;
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
      if (!std::isfinite(source.dc_volts)) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "voltage source '" + source.name +
                                           "' has invalid DC value");
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
    } else {
      b_dc[branch] = std::get<VoltageSource>(component).dc_volts;
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

  return Result<MnaSystem>::Ok(MnaSystem{
      .g = BuildCsr(size, g_entries),
      .c = BuildCsr(size, c_entries),
      .b_dc = std::move(b_dc),
      .node_names = std::move(node_names),
      .branch_names = std::move(branch_names),
  });
}

} // namespace ohmnivore
