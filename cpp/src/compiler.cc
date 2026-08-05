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
    if (const auto *source = std::get_if<VoltageSource>(&component)) {
      if (branch_map.contains(source->name)) {
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "duplicate voltage-source name '" +
                                           source->name + "'");
      }
      branch_map.emplace(source->name, branch_names.size());
      branch_names.push_back(source->name);
    }
  }

  const std::size_t node_count = node_names.size();
  const std::size_t size = node_count + branch_names.size();
  if (size == 0) {
    return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                   "circuit contains no solvable variables");
  }

  std::map<Coordinate, double> entries;
  std::vector<double> b_dc(size, 0.0);
  for (std::size_t node = 0; node < node_count; ++node) {
    AddStamp(&entries, node, node, kGminSiemens);
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
      if (positive.has_value()) {
        AddStamp(&entries, *positive, *positive, conductance);
      }
      if (negative.has_value()) {
        AddStamp(&entries, *negative, *negative, conductance);
      }
      if (positive.has_value() && negative.has_value()) {
        AddStamp(&entries, *positive, *negative, -conductance);
        AddStamp(&entries, *negative, *positive, -conductance);
      }
      continue;
    }

    const auto &source = std::get<VoltageSource>(component);
    const auto positive = FindNode(source.positive_node, node_map);
    const auto negative = FindNode(source.negative_node, node_map);
    if (!positive.has_value() && !negative.has_value()) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "voltage source '" + source.name +
                                         "' connects ground to ground");
    }
    const std::size_t branch = node_count + branch_map.at(source.name);
    if (positive.has_value()) {
      AddStamp(&entries, *positive, branch, 1.0);
      AddStamp(&entries, branch, *positive, 1.0);
    }
    if (negative.has_value()) {
      AddStamp(&entries, *negative, branch, -1.0);
      AddStamp(&entries, branch, *negative, -1.0);
    }
    b_dc[branch] = source.dc_volts;
  }

  return Result<MnaSystem>::Ok(MnaSystem{
      .g = BuildCsr(size, entries),
      .b_dc = std::move(b_dc),
      .node_names = std::move(node_names),
      .branch_names = std::move(branch_names),
  });
}

} // namespace ohmnivore
