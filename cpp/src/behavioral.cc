#include "ohmnivore/behavioral.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <variant>

#include "ohmnivore/parser.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/waveform.h"

namespace ohmnivore {
namespace {
std::string Lower(std::string value) {
  for (char &c : value) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  return value;
}
std::string Trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}
std::optional<std::size_t> Position(const CsrMatrix &matrix, std::size_t row,
                                    std::size_t column) {
  if (row >= matrix.rows || matrix.row_offsets.size() != matrix.rows + 1)
    return {};
  for (std::size_t i = matrix.row_offsets[row]; i < matrix.row_offsets[row + 1];
       ++i) {
    if (i >= matrix.column_indices.size())
      return {};
    if (matrix.column_indices[i] == column)
      return i;
  }
  return {};
}
} // namespace

Result<BehavioralCircuit> ParseBehavioralNetlist(std::string_view netlist) {
  if (netlist.size() > 1024 * 1024) {
    return Result<BehavioralCircuit>::Fail(ErrorCode::kUnsupportedSize,
                                           "behavioral deck exceeds 1 MiB");
  }
  BehavioralCircuit result;
  std::istringstream input{std::string(netlist)};
  std::ostringstream native;
  std::string line;
  bool ended = false;
  std::set<std::string> saves;
  while (std::getline(input, line)) {
    const std::string stripped = Trim(line);
    if (stripped.empty() || stripped[0] == '*') {
      native << line << '\n';
      continue;
    }
    if (ended)
      return Result<BehavioralCircuit>::Fail(ErrorCode::kParse,
                                             "content after .end");
    std::istringstream fields(stripped);
    std::string name;
    fields >> name;
    const std::string key = Lower(name);
    if (key == ".save") {
      std::string observable;
      bool any = false;
      while (fields >> observable) {
        observable = Lower(observable);
        if (!saves.insert(observable).second) {
          return Result<BehavioralCircuit>::Fail(ErrorCode::kParse,
                                                 "duplicate .save observable");
        }
        result.saved_observables.push_back(observable);
        any = true;
      }
      if (!any)
        return Result<BehavioralCircuit>::Fail(ErrorCode::kParse,
                                               "empty .save");
      continue;
    }
    if (key == ".end")
      ended = true;
    if (key[0] != 'e' && key[0] != 'g' && key[0] != 'b') {
      native << line << '\n';
      continue;
    }
    std::string positive, negative, expression;
    if (!(fields >> positive >> negative)) {
      return Result<BehavioralCircuit>::Fail(ErrorCode::kParse,
                                             "incomplete behavioral source");
    }
    std::getline(fields, expression);
    expression = Trim(expression);
    const std::string prefix = key[0] == 'b' ? "i=" : "value=";
    if (!Lower(expression).starts_with(prefix)) {
      return Result<BehavioralCircuit>::Fail(
          ErrorCode::kUnsupported,
          "behavioral source requires VALUE={...} or I={...}");
    }
    expression = Trim(expression.substr(prefix.size()));
    if (expression.size() < 2 || expression.front() != '{' ||
        expression.back() != '}') {
      return Result<BehavioralCircuit>::Fail(
          ErrorCode::kParse, "behavioral expression requires braces");
    }
    result.sources.push_back(
        {name, positive, negative, expression, key[0] == 'e'});
    if (result.sources.size() > 512) {
      return Result<BehavioralCircuit>::Fail(ErrorCode::kUnsupportedSize,
                                             "too many behavioral sources");
    }
  }
  auto parsed = ParseNetlist(native.str());
  if (!parsed.ok())
    return Result<BehavioralCircuit>::Fail(parsed.error().code,
                                           parsed.error().message);
  result.circuit = parsed.TakeValue();
  return Result<BehavioralCircuit>::Ok(std::move(result));
}

Result<bool> RemapBehavioralDescriptors(MnaSystem *system) {
  if (system == nullptr)
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "null behavioral system");
  for (auto &descriptor : system->behavioral_descriptors) {
    for (auto &row : descriptor.rows) {
      row.jacobian_value_indices.clear();
      for (const auto column : descriptor.expression.dependencies()) {
        const auto position = Position(system->g, row.row, column);
        if (!position)
          return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                    "missing behavioral Jacobian coordinate");
        row.jacobian_value_indices.push_back(*position);
      }
    }
  }
  return Result<bool>::Ok(true);
}

Result<bool> ValidateBehavioralDescriptors(const MnaSystem &system) {
  std::set<std::string> names;
  std::size_t total_nodes = 0;
  for (const auto &descriptor : system.behavioral_descriptors) {
    const auto dependencies = descriptor.expression.dependencies();
    if (descriptor.name.empty() ||
        !names.insert(Lower(descriptor.name)).second ||
        descriptor.expression.node_count() == 0 || descriptor.rows.empty() ||
        descriptor.rows.size() > 2) {
      return Result<bool>::Fail(
          ErrorCode::kInvalidStructure,
          "invalid behavioral descriptor identity or expression");
    }
    total_nodes += descriptor.expression.node_count();
    if (total_nodes > 16384)
      return Result<bool>::Fail(ErrorCode::kUnsupportedSize,
                                "behavioral expression budget exceeded");
    if ((descriptor.positive_node_index &&
         *descriptor.positive_node_index >= system.node_names.size()) ||
        (descriptor.negative_node_index &&
         *descriptor.negative_node_index >= system.node_names.size()) ||
        descriptor.positive_node_index == descriptor.negative_node_index) {
      return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                "invalid behavioral terminal metadata");
    }
    std::vector<std::pair<std::size_t, double>> expected_rows;
    if (descriptor.is_voltage) {
      if (!descriptor.branch_index ||
          *descriptor.branch_index < system.node_names.size() ||
          *descriptor.branch_index >= system.g.rows ||
          *descriptor.branch_index - system.node_names.size() >=
              system.branch_names.size() ||
          Lower(system.branch_names[*descriptor.branch_index -
                                    system.node_names.size()]) !=
              Lower(descriptor.name)) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "invalid behavioral voltage branch metadata");
      }
      expected_rows.emplace_back(*descriptor.branch_index, -1.0);
    } else {
      if (descriptor.branch_index)
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "current source cannot own a voltage branch");
      if (descriptor.positive_node_index)
        expected_rows.emplace_back(*descriptor.positive_node_index, 1.0);
      if (descriptor.negative_node_index)
        expected_rows.emplace_back(*descriptor.negative_node_index, -1.0);
    }
    if (descriptor.rows.size() != expected_rows.size()) {
      return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                "behavioral terminal row count mismatch");
    }
    for (std::size_t i = 0; i < descriptor.rows.size(); ++i) {
      if (descriptor.rows[i].row != expected_rows[i].first ||
          descriptor.rows[i].coefficient != expected_rows[i].second) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "behavioral terminal row or sign mismatch");
      }
    }
    std::set<std::size_t> rows;
    for (const auto &row : descriptor.rows) {
      if (row.row >= system.g.rows || !rows.insert(row.row).second ||
          (row.coefficient != 1.0 && row.coefficient != -1.0) ||
          row.jacobian_value_indices.size() != dependencies.size()) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "invalid behavioral residual row");
      }
      for (std::size_t j = 0; j < dependencies.size(); ++j) {
        const auto position = Position(system.g, row.row, dependencies[j]);
        if (!position || *position != row.jacobian_value_indices[j]) {
          return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                    "invalid behavioral Jacobian coordinate");
        }
      }
    }
  }
  return Result<bool>::Ok(true);
}

Result<MnaSystem> CompileBehavioralMna(const BehavioralCircuit &input) {
  Circuit native = input.circuit;
  auto sources = input.sources;
  // The experimental SPICE graph binds nodes case-insensitively, while keeping
  // the first spelling for stable output names. Ordinary CompileMna is
  // unchanged.
  std::map<std::string, std::string> node_spellings{{"0", "0"}, {"gnd", "0"}};
  const auto canonical_node = [&](const std::string &node) {
    return node_spellings.try_emplace(Lower(node), node).first->second;
  };
  for (auto &component : native.components) {
    std::visit(
        [&](auto &value) {
          if constexpr (requires { value.positive_node; }) {
            value.positive_node = canonical_node(value.positive_node);
            value.negative_node = canonical_node(value.negative_node);
          }
        },
        component);
  }
  for (auto &source : sources) {
    source.positive_node = canonical_node(source.positive_node);
    source.negative_node = canonical_node(source.negative_node);
  }
  if (!native.diode_models.empty() || !native.bjt_models.empty()) {
    return Result<MnaSystem>::Fail(
        ErrorCode::kUnsupported,
        "behavioral graph excludes native semiconductor models");
  }
  std::set<std::string> names;
  std::set<std::string> voltage_sensors;
  for (const auto &component : native.components) {
    if (std::holds_alternative<Diode>(component) ||
        std::holds_alternative<Bjt>(component)) {
      return Result<MnaSystem>::Fail(
          ErrorCode::kUnsupported,
          "behavioral graph excludes native semiconductors");
    }
    const std::string name = std::visit(
        [](const auto &value) { return Lower(value.name); }, component);
    if (std::holds_alternative<VoltageSource>(component) &&
        name.starts_with('v'))
      voltage_sensors.insert(name);
    if (!names.insert(name).second)
      return Result<MnaSystem>::Fail(
          ErrorCode::kCompile, "ambiguous component name in behavioral graph");
  }
  for (const auto &source : sources) {
    if (source.name.empty() || source.positive_node.empty() ||
        source.negative_node.empty() ||
        !names.insert(Lower(source.name)).second) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "invalid behavioral source identity");
    }
    if (source.is_voltage) {
      native.components.emplace_back(
          VoltageSource{source.name, source.positive_node, source.negative_node,
                        0.0, std::nullopt});
    } else {
      native.components.emplace_back(
          CurrentSource{source.name, source.positive_node, source.negative_node,
                        0.0, std::nullopt});
    }
  }
  auto compiled = CompileMna(native);
  if (!compiled.ok())
    return compiled;
  MnaSystem system = compiled.TakeValue();
  if (system.g.rows > 512)
    return Result<MnaSystem>::Fail(ErrorCode::kUnsupportedSize,
                                   "behavioral graph exceeds 512 unknowns");
  ExpressionBindings bindings;
  bindings.state_size = system.g.rows;
  bindings.node_indices["0"] = std::nullopt;
  bindings.node_indices["gnd"] = std::nullopt;
  for (std::size_t i = 0; i < system.node_names.size(); ++i) {
    if (!bindings.node_indices.emplace(Lower(system.node_names[i]), i).second) {
      return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                     "ambiguous behavioral node name");
    }
  }
  for (std::size_t i = 0; i < system.branch_names.size(); ++i) {
    const std::string name = Lower(system.branch_names[i]);
    // Only explicitly named voltage-sensing sources are expression inputs.
    if (voltage_sensors.contains(name))
      bindings.current_indices[name] = system.node_names.size() + i;
  }
  std::map<std::pair<std::size_t, std::size_t>, double> entries;
  for (std::size_t row = 0; row < system.g.rows; ++row) {
    for (std::size_t i = system.g.row_offsets[row];
         i < system.g.row_offsets[row + 1]; ++i) {
      entries[{row, system.g.column_indices[i]}] = system.g.values[i];
    }
  }
  for (const auto &source : sources) {
    auto expression = CompileExpression(source.expression, bindings);
    if (!expression.ok())
      return Result<MnaSystem>::Fail(expression.error().code,
                                     source.name + ": " +
                                         expression.error().message);
    BehavioralDescriptor descriptor{
        .name = source.name,
        .expression = expression.TakeValue(),
        .rows = {},
        .is_voltage = source.is_voltage,
        .positive_node_index =
            bindings.node_indices.at(Lower(source.positive_node)),
        .negative_node_index =
            bindings.node_indices.at(Lower(source.negative_node)),
        .branch_index = std::nullopt};
    if (source.is_voltage) {
      const auto found = std::find(system.branch_names.begin(),
                                   system.branch_names.end(), source.name);
      if (found == system.branch_names.end())
        return Result<MnaSystem>::Fail(ErrorCode::kCompile,
                                       "missing behavioral source branch");
      descriptor.branch_index =
          system.node_names.size() +
          static_cast<std::size_t>(found - system.branch_names.begin());
      descriptor.rows.push_back({*descriptor.branch_index, -1.0, {}});
    } else {
      const auto positive =
          bindings.node_indices.at(Lower(source.positive_node));
      const auto negative =
          bindings.node_indices.at(Lower(source.negative_node));
      if (positive == negative)
        return Result<MnaSystem>::Fail(
            ErrorCode::kCompile,
            "behavioral current source has identical terminals");
      if (positive)
        descriptor.rows.push_back({*positive, 1.0, {}});
      if (negative)
        descriptor.rows.push_back({*negative, -1.0, {}});
    }
    for (const auto &row : descriptor.rows) {
      for (const auto column : descriptor.expression.dependencies())
        entries.try_emplace({row.row, column}, 0.0);
    }
    system.behavioral_descriptors.push_back(std::move(descriptor));
  }
  system.g.values.clear();
  system.g.column_indices.clear();
  system.g.row_offsets.assign(system.g.rows + 1, 0);
  for (const auto &[coordinate, value] : entries) {
    system.g.column_indices.push_back(coordinate.second);
    system.g.values.push_back(value);
    ++system.g.row_offsets[coordinate.first + 1];
  }
  for (std::size_t row = 0; row < system.g.rows; ++row)
    system.g.row_offsets[row + 1] += system.g.row_offsets[row];
  auto remapped = RemapBehavioralDescriptors(&system);
  if (!remapped.ok())
    return Result<MnaSystem>::Fail(remapped.error().code,
                                   remapped.error().message);
  auto valid = ValidateBehavioralDescriptors(system);
  if (!valid.ok())
    return Result<MnaSystem>::Fail(valid.error().code, valid.error().message);
  return Result<MnaSystem>::Ok(std::move(system));
}

Result<bool> ValidateBehavioralTransient(const MnaSystem &system) {
  if (system.c.rows != system.g.rows || system.c.columns != system.g.columns ||
      system.node_names.size() + system.branch_names.size() != system.g.rows) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "behavioral transient dimensions disagree");
  }
  auto dynamic = ConvertCsrToSolverCsc(system.c);
  if (!dynamic.ok())
    return Result<bool>::Fail(dynamic.error().code, dynamic.error().message);
  const std::size_t node_count = system.node_names.size();
  std::set<std::string> reactive_names;
  std::map<std::pair<std::size_t, std::size_t>, double> capacitance;
  for (const auto &capacitor : system.capacitor_initial_constraints) {
    if (!std::isfinite(capacitor.capacitance_farads)) {
      return Result<bool>::Fail(ErrorCode::kNonFinite,
                                "non-finite behavioral capacitance metadata");
    }
    if (capacitor.name.empty() ||
        !reactive_names.insert(Lower(capacitor.name)).second ||
        capacitor.capacitance_farads <= 0.0 ||
        capacitor.positive_node_index == capacitor.negative_node_index ||
        (capacitor.positive_node_index &&
         *capacitor.positive_node_index >= node_count) ||
        (capacitor.negative_node_index &&
         *capacitor.negative_node_index >= node_count)) {
      return Result<bool>::Fail(
          ErrorCode::kInvalidStructure,
          "invalid behavioral capacitor state constraint");
    }
    const auto positive = capacitor.positive_node_index;
    const auto negative = capacitor.negative_node_index;
    // Match the compiler's declaration-order FP64 accumulation, including
    // parallel capacitors with different scales; no tolerance can hide
    // omission.
    if (positive)
      capacitance[{*positive, *positive}] += capacitor.capacitance_farads;
    if (negative)
      capacitance[{*negative, *negative}] += capacitor.capacitance_farads;
    if (positive && negative) {
      capacitance[{*positive, *negative}] -= capacitor.capacitance_farads;
      capacitance[{*negative, *positive}] -= capacitor.capacitance_farads;
    }
  }
  std::set<std::size_t> inductive_branches;
  for (const auto &inductor : system.inductor_initial_constraints) {
    if (inductor.name.empty() ||
        !reactive_names.insert(Lower(inductor.name)).second ||
        !inductive_branches.insert(inductor.branch_index).second ||
        inductor.branch_index < node_count ||
        inductor.branch_index >= system.g.rows ||
        Lower(system.branch_names[inductor.branch_index - node_count]) !=
            Lower(inductor.name)) {
      return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                "invalid behavioral inductor state constraint");
    }
    const auto diagonal =
        Position(system.c, inductor.branch_index, inductor.branch_index);
    if (!diagonal || system.c.values[*diagonal] >= 0.0) {
      return Result<bool>::Fail(
          ErrorCode::kInvalidStructure,
          "behavioral inductor lacks negative C diagonal");
    }
  }
  for (std::size_t row = 0; row < system.c.rows; ++row) {
    for (std::size_t index = system.c.row_offsets[row];
         index < system.c.row_offsets[row + 1]; ++index) {
      const std::size_t column = system.c.column_indices[index];
      if (row < node_count && column < node_count) {
        const auto found = capacitance.find({row, column});
        if (found == capacitance.end() ||
            found->second != system.c.values[index]) {
          return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                    "capacitor metadata disagrees with C");
        }
        capacitance.erase(found);
      } else if (row < node_count || column < node_count ||
                 !inductive_branches.contains(row) ||
                 !inductive_branches.contains(column)) {
        return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                                  "C coordinate lacks reactive state metadata");
      }
    }
  }
  if (!capacitance.empty()) {
    return Result<bool>::Fail(ErrorCode::kInvalidStructure,
                              "capacitor metadata has absent C coordinates");
  }
  for (const auto &source : system.transient_sources) {
    if (const auto *pulse = std::get_if<PulseWaveform>(&source.waveform)) {
      if (pulse->rise_time_seconds <= 0.0 || pulse->fall_time_seconds <= 0.0) {
        return Result<bool>::Fail(
            ErrorCode::kUnsupported,
            "behavioral transient requires finite source edges");
      }
    } else if (!std::holds_alternative<PwlWaveform>(source.waveform)) {
      return Result<bool>::Fail(
          ErrorCode::kUnsupported,
          "behavioral transient supports only PWL/PULSE sources");
    }
    auto initial = EvaluateTransientWaveform(source.waveform, 0.0);
    if (!initial.ok())
      return Result<bool>::Fail(initial.error().code, initial.error().message);
    if (initial.value() != source.dc_value) {
      return Result<bool>::Fail(
          ErrorCode::kUnsupported,
          "behavioral transient requires explicit DC equal to waveform(0)");
    }
  }
  return ValidateBehavioralDescriptors(system);
}
} // namespace ohmnivore
