#include "ohmnivore/expression.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <system_error>

namespace ohmnivore {
namespace {

constexpr std::size_t kMaxNodes = 512;
constexpr std::size_t kMaxDepth = 64;
constexpr std::size_t kMaxTotalNodes = 16384;
constexpr double kDivisionOffset = 1e-32;
constexpr double kPsExpSlope = 1202604.284;
constexpr double kMagnitudeLimit = 1e100;

bool IsBounded(double value) {
  return std::isfinite(value) && std::abs(value) <= kMagnitudeLimit;
}

enum class Op {
  kConstant,
  kState,
  kNegate,
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kPower,
  kExp,
  kLess,
  kGreater,
  kIf
};

struct Node {
  Op op = Op::kConstant;
  std::size_t first = 0;
  std::size_t second = 0;
  std::size_t third = 0;
  std::size_t depth = 1;
  double value = 0.0;
  bool constant = true;
};

std::string Upper(std::string_view text) {
  std::string result(text);
  for (char &c : result) {
    if (c >= 'a' && c <= 'z')
      c = static_cast<char>(c - 'a' + 'A');
  }
  return result;
}

bool IsAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool IsDigit(char c) { return c >= '0' && c <= '9'; }

bool IsName(char c) { return IsAlpha(c) || IsDigit(c) || c == '.' || c == ':'; }

bool ValidParameterName(std::string_view name) {
  return !name.empty() && IsAlpha(name.front()) &&
         std::all_of(name.begin(), name.end(), IsName);
}

} // namespace

struct ExpressionProgram {
  std::vector<Node> nodes;
  std::vector<std::size_t> dependencies;
  std::size_t root = 0;
  std::size_t state_size = 0;
  ExpressionDialect dialect = ExpressionDialect::kBehavioral;
};

namespace {

using ParameterResolver = std::function<Result<double>(const std::string &)>;

class Parser {
public:
  Parser(std::string_view text, const ExpressionBindings &bindings,
         ExpressionDialect dialect, ParameterResolver resolver)
      : text_(text), bindings_(bindings), resolver_(std::move(resolver)) {
    program_.state_size = bindings.state_size;
    program_.dialect = dialect;
  }

  Result<ExpressionProgram> Parse() {
    if (text_.size() > 65536 || bindings_.state_size > 512) {
      return Result<ExpressionProgram>::Fail(ErrorCode::kUnsupportedSize,
                                             "expression resource limit");
    }
    program_.root = Comparison(1);
    Space();
    if (!error_ && position_ != text_.size()) {
      Fail(ErrorCode::kUnsupported, "unsupported trailing expression token");
    }
    if (error_) {
      return Result<ExpressionProgram>::Fail(error_->code, error_->message);
    }
    for (const auto &node : program_.nodes) {
      if (node.op == Op::kState)
        program_.dependencies.push_back(node.first);
    }
    auto &deps = program_.dependencies;
    std::sort(deps.begin(), deps.end());
    deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
    return Result<ExpressionProgram>::Ok(std::move(program_));
  }

private:
  void Fail(ErrorCode code, const std::string &message) {
    if (!error_)
      error_ = Error{code, message};
  }

  void Space() {
    while (position_ < text_.size() &&
           (text_[position_] == ' ' || text_[position_] == '\t' ||
            text_[position_] == '\r' || text_[position_] == '\n'))
      ++position_;
  }

  bool Take(char c) {
    Space();
    if (position_ < text_.size() && text_[position_] == c) {
      ++position_;
      return true;
    }
    return false;
  }

  bool Guard(std::size_t depth) {
    if (error_)
      return false;
    if (depth > kMaxDepth) {
      Fail(ErrorCode::kUnsupportedSize, "expression nesting limit");
      return false;
    }
    return true;
  }

  std::size_t Add(Node node) {
    if (error_)
      return 0;
    if (program_.nodes.size() >= kMaxNodes || node.depth > kMaxDepth) {
      Fail(ErrorCode::kUnsupportedSize, "expression node or tree depth limit");
      return 0;
    }
    program_.nodes.push_back(node);
    return program_.nodes.size() - 1;
  }

  std::size_t UnaryNode(Op op, std::size_t first) {
    if (error_)
      return 0;
    return Add(Node{.op = op,
                    .first = first,
                    .depth = program_.nodes[first].depth + 1,
                    .constant = program_.nodes[first].constant});
  }

  std::size_t BinaryNode(Op op, std::size_t first, std::size_t second) {
    if (error_)
      return 0;
    return Add(Node{.op = op,
                    .first = first,
                    .second = second,
                    .depth = 1 + std::max(program_.nodes[first].depth,
                                          program_.nodes[second].depth),
                    .constant = program_.nodes[first].constant &&
                                program_.nodes[second].constant});
  }

  std::size_t Comparison(std::size_t depth) {
    if (!Guard(depth))
      return 0;
    auto first = Sum(depth);
    while (!error_) {
      Op op;
      if (Take('<'))
        op = Op::kLess;
      else if (Take('>'))
        op = Op::kGreater;
      else
        break;
      auto second = Sum(depth);
      first = BinaryNode(op, first, second);
    }
    return first;
  }

  std::size_t Sum(std::size_t depth) {
    auto first = Product(depth);
    while (!error_) {
      Op op;
      if (Take('+'))
        op = Op::kAdd;
      else if (Take('-'))
        op = Op::kSubtract;
      else
        break;
      auto second = Product(depth);
      first = BinaryNode(op, first, second);
    }
    return first;
  }

  std::size_t Product(std::size_t depth) {
    auto first = Unary(depth);
    while (!error_) {
      Space();
      if (text_.substr(position_, 2) == "**")
        break;
      Op op;
      if (Take('*'))
        op = Op::kMultiply;
      else if (Take('/'))
        op = Op::kDivide;
      else
        break;
      auto second = Unary(depth);
      first = BinaryNode(op, first, second);
    }
    return first;
  }

  std::size_t Unary(std::size_t depth) {
    if (!Guard(depth))
      return 0;
    if (Take('+'))
      return Unary(depth + 1);
    if (Take('-'))
      return UnaryNode(Op::kNegate, Unary(depth + 1));
    auto first = Atom(depth);
    Space();
    if (!error_ && text_.substr(position_, 2) == "**") {
      position_ += 2;
      auto second = Unary(depth + 1);
      return BinaryNode(Op::kPower, first, second);
    }
    return first;
  }

  std::string Name() {
    Space();
    const auto begin = position_;
    while (position_ < text_.size() && IsName(text_[position_]))
      ++position_;
    if (begin == position_)
      Fail(ErrorCode::kParse, "expected expression name");
    return Upper(text_.substr(begin, position_ - begin));
  }

  std::size_t State(const std::string &name, bool current) {
    if (program_.dialect == ExpressionDialect::kParameter) {
      Fail(ErrorCode::kUnsupported, "parameter expressions cannot sense state");
      return 0;
    }
    if (!current && (name == "0" || name == "GND"))
      return Add(Node{});
    std::optional<std::size_t> index;
    if (current) {
      const auto found = bindings_.current_indices.find(name);
      if (found == bindings_.current_indices.end()) {
        Fail(ErrorCode::kUnsupported, "unknown sensed voltage source: " + name);
        return 0;
      }
      index = found->second;
    } else {
      const auto found = bindings_.node_indices.find(name);
      if (found == bindings_.node_indices.end()) {
        Fail(ErrorCode::kUnsupported, "unknown voltage node: " + name);
        return 0;
      }
      index = found->second;
    }
    if (!index)
      return Add(Node{});
    return Add(Node{.op = Op::kState, .first = *index, .constant = false});
  }

  std::size_t Number() {
    const char *begin = text_.data() + position_;
    const char *end = text_.data() + text_.size();
    double value = 0.0;
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr == begin) {
      Fail(result.ec == std::errc::result_out_of_range ? ErrorCode::kNonFinite
                                                       : ErrorCode::kParse,
           "invalid expression number");
      return 0;
    }
    position_ = static_cast<std::size_t>(result.ptr - text_.data());
    const auto start = position_;
    while (position_ < text_.size() && IsAlpha(text_[position_]))
      ++position_;
    const auto suffix = Upper(text_.substr(start, position_ - start));
    const std::map<std::string, double> multipliers = {
        {"", 1.0},   {"T", 1e12}, {"G", 1e9},  {"MEG", 1e6}, {"K", 1e3},
        {"M", 1e-3}, {"U", 1e-6}, {"N", 1e-9}, {"P", 1e-12}, {"F", 1e-15}};
    const auto scale = multipliers.find(suffix);
    if (scale == multipliers.end()) {
      Fail(ErrorCode::kUnsupported, "unknown expression number suffix");
      return 0;
    }
    value *= scale->second;
    if (!IsBounded(value)) {
      Fail(ErrorCode::kNonFinite, "nonfinite expression number");
      return 0;
    }
    return Add(Node{.value = value});
  }

  std::size_t Atom(std::size_t depth) {
    if (!Guard(depth))
      return 0;
    Space();
    if (position_ == text_.size()) {
      Fail(ErrorCode::kParse, "missing expression operand");
      return 0;
    }
    const char first = text_[position_];
    if (first == '(' || first == '{') {
      ++position_;
      const auto node = Comparison(depth + 1);
      if (!Take(first == '(' ? ')' : '}'))
        Fail(ErrorCode::kParse, "unclosed expression");
      return node;
    }
    if (IsDigit(first) || first == '.')
      return Number();
    if (!IsAlpha(first)) {
      Fail(ErrorCode::kParse, "expected expression operand");
      return 0;
    }
    const auto name = Name();
    if (!Take('(')) {
      auto value = resolver_(name);
      if (!value.ok()) {
        Fail(value.error().code, value.error().message);
        return 0;
      }
      return Add(Node{.value = value.value()});
    }
    if (name == "V" || name == "I") {
      auto node = State(Name(), name == "I");
      if (name == "V" && Take(',')) {
        auto second = State(Name(), false);
        node = BinaryNode(Op::kSubtract, node, second);
      }
      if (!Take(')'))
        Fail(ErrorCode::kParse, "invalid state reference");
      return node;
    }
    if ((name != "EXP" && name != "IF") ||
        (name == "IF" && program_.dialect == ExpressionDialect::kParameter)) {
      Fail(ErrorCode::kUnsupported, "unsupported expression function: " + name);
      return 0;
    }
    const auto arg1 = Comparison(depth + 1);
    if (name == "EXP") {
      if (!Take(')'))
        Fail(ErrorCode::kParse, "exp requires one argument");
      return UnaryNode(Op::kExp, arg1);
    }
    if (!Take(','))
      Fail(ErrorCode::kParse, "if requires three arguments");
    const auto arg2 = Comparison(depth + 1);
    if (!Take(','))
      Fail(ErrorCode::kParse, "if requires three arguments");
    const auto arg3 = Comparison(depth + 1);
    if (!Take(')'))
      Fail(ErrorCode::kParse, "if requires three arguments");
    if (error_)
      return 0;
    return Add(Node{.op = Op::kIf,
                    .first = arg1,
                    .second = arg2,
                    .third = arg3,
                    .depth = 1 + std::max({program_.nodes[arg1].depth,
                                           program_.nodes[arg2].depth,
                                           program_.nodes[arg3].depth}),
                    .constant = program_.nodes[arg1].constant &&
                                program_.nodes[arg2].constant &&
                                program_.nodes[arg3].constant});
  }

  std::string_view text_;
  const ExpressionBindings &bindings_;
  ParameterResolver resolver_;
  ExpressionProgram program_;
  std::size_t position_ = 0;
  std::optional<Error> error_;
};

Result<ExpressionBindings> Normalize(const ExpressionBindings &source) {
  if (source.state_size > 512 || source.parameters.size() > 512 ||
      source.node_indices.size() > 514 || source.current_indices.size() > 512) {
    return Result<ExpressionBindings>::Fail(ErrorCode::kUnsupportedSize,
                                            "expression binding limit");
  }
  ExpressionBindings result;
  result.state_size = source.state_size;
  for (const auto &[name, value] : source.parameters) {
    if (!ValidParameterName(name) ||
        !result.parameters.emplace(Upper(name), value).second) {
      return Result<ExpressionBindings>::Fail(
          ErrorCode::kCompile, "invalid or duplicate parameter binding");
    }
    if (!IsBounded(value)) {
      return Result<ExpressionBindings>::Fail(ErrorCode::kNonFinite,
                                              "nonfinite parameter binding");
    }
  }
  for (const auto &[name, index] : source.node_indices) {
    const auto canonical = Upper(name);
    if (name.empty() || !std::all_of(name.begin(), name.end(), IsName) ||
        (index && *index >= source.state_size) ||
        ((canonical == "0" || canonical == "GND") && index) ||
        !result.node_indices.emplace(canonical, index).second) {
      return Result<ExpressionBindings>::Fail(
          ErrorCode::kCompile, "invalid or duplicate voltage binding");
    }
  }
  for (const auto &[name, index] : source.current_indices) {
    if (name.empty() || !std::all_of(name.begin(), name.end(), IsName) ||
        index >= source.state_size ||
        !result.current_indices.emplace(Upper(name), index).second) {
      return Result<ExpressionBindings>::Fail(
          ErrorCode::kCompile, "invalid or duplicate current binding");
    }
  }
  return Result<ExpressionBindings>::Ok(std::move(result));
}

double Denominator(double value) {
  return value + (value >= 0.0 ? kDivisionOffset : -kDivisionOffset);
}

class Evaluator {
public:
  Evaluator(const ExpressionProgram &program, std::span<const double> state)
      : program_(program), state_(state) {
    std::fill_n(values_.begin(), program.nodes.size(), 0.0);
    std::fill_n(adjoints_.begin(), program.nodes.size(), 0.0);
  }

  Result<ExpressionEvaluation> Run() {
    if (state_.size() != program_.state_size) {
      return Result<ExpressionEvaluation>::Fail(
          ErrorCode::kInvalidStructure, "expression state size mismatch");
    }
    for (const auto index : program_.dependencies) {
      if (!IsBounded(state_[index]))
        return Failure();
    }
    ExpressionEvaluation result;
    result.value = Value(program_.root);
    if (error_)
      return Failure();
    adjoints_[program_.root] = 1.0;
    std::array<double, kMaxNodes> gradient;
    std::fill_n(gradient.begin(), program_.dependencies.size(), 0.0);
    for (std::size_t i = program_.nodes.size(); i-- > 0;) {
      const auto &node = program_.nodes[i];
      const double adjoint = adjoints_[i];
      if (adjoint == 0.0 || node.constant)
        continue;
      if (node.op == Op::kState) {
        const auto found =
            std::lower_bound(program_.dependencies.begin(),
                             program_.dependencies.end(), node.first);
        const auto offset =
            static_cast<std::size_t>(found - program_.dependencies.begin());
        gradient[offset] += adjoint;
        if (!IsBounded(gradient[offset]))
          return Failure();
        continue;
      }
      const double a = values_[node.first];
      const double b = values_[node.second];
      auto add = [&](std::size_t child, double factor) {
        if (!program_.nodes[child].constant) {
          adjoints_[child] += adjoint * factor;
          if (!IsBounded(factor) || !IsBounded(adjoint * factor) ||
              !IsBounded(adjoints_[child]))
            error_ = true;
        }
      };
      switch (node.op) {
      case Op::kConstant:
        break;
      case Op::kState:
        break;
      case Op::kNegate:
        add(node.first, -1.0);
        break;
      case Op::kAdd:
        add(node.first, 1.0);
        add(node.second, 1.0);
        break;
      case Op::kSubtract:
        add(node.first, 1.0);
        add(node.second, -1.0);
        break;
      case Op::kMultiply:
        add(node.first, b);
        add(node.second, a);
        break;
      case Op::kDivide: {
        const double denominator =
            program_.dialect == ExpressionDialect::kBehavioral ? Denominator(b)
                                                               : b;
        add(node.first, 1.0 / denominator);
        add(node.second, -(values_[i] / denominator));
        break;
      }
      case Op::kPower: {
        if (!program_.nodes[node.first].constant) {
          double slope;
          if (b == 0.0 || (a == 0.0 && b > 1.0))
            slope = 0.0;
          else if (a == 0.0 && b == 1.0)
            slope = 1.0;
          else
            slope = b * std::pow(std::abs(a), b - 1.0) * (a < 0.0 ? -1.0 : 1.0);
          add(node.first, slope);
        }
        if (!program_.nodes[node.second].constant) {
          // At zero, a positive exponent makes the value identically zero as
          // a function of exponent. Avoid the undefined product 0*log(0).
          add(node.second,
              a == 0.0 && b > 0.0 ? 0.0 : values_[i] * std::log(std::abs(a)));
        }
        break;
      }
      case Op::kExp:
        add(node.first,
            program_.dialect == ExpressionDialect::kBehavioral && a > 14.0
                ? kPsExpSlope
                : values_[i]);
        break;
      case Op::kLess:
      case Op::kGreater:
        break;
      case Op::kIf:
        add(a != 0.0 ? node.second : node.third, 1.0);
        break;
      }
    }
    if (error_)
      return Failure();
    result.derivatives.reserve(program_.dependencies.size());
    for (std::size_t i = 0; i < program_.dependencies.size(); ++i) {
      if (!IsBounded(gradient[i])) {
        return Result<ExpressionEvaluation>::Fail(
            ErrorCode::kNonFinite, "nonfinite expression derivative");
      }
      if (gradient[i] != 0.0)
        result.derivatives.emplace_back(program_.dependencies[i], gradient[i]);
    }
    return Result<ExpressionEvaluation>::Ok(std::move(result));
  }

private:
  Result<ExpressionEvaluation> Failure() const {
    return Result<ExpressionEvaluation>::Fail(
        ErrorCode::kNonFinite, "nonfinite expression value or domain");
  }

  double Value(std::size_t index) {
    const auto &node = program_.nodes[index];
    double value = 0.0;
    if (node.op == Op::kConstant)
      value = node.value;
    else if (node.op == Op::kState)
      value = state_[node.first];
    else {
      const double a = Value(node.first);
      if (error_)
        return 0.0;
      if (node.op == Op::kIf)
        value = Value(a != 0.0 ? node.second : node.third);
      else if (node.op == Op::kNegate)
        value = -a;
      else if (node.op == Op::kExp) {
        value = program_.dialect == ExpressionDialect::kBehavioral && a > 14.0
                    ? kPsExpSlope * (a - 13.0)
                    : std::exp(a);
      } else {
        const double b = Value(node.second);
        if (error_)
          return 0.0;
        switch (node.op) {
        case Op::kAdd:
          value = a + b;
          break;
        case Op::kSubtract:
          value = a - b;
          break;
        case Op::kMultiply:
          value = a * b;
          break;
        case Op::kDivide:
          value = a / (program_.dialect == ExpressionDialect::kBehavioral
                           ? Denominator(b)
                           : b);
          break;
        case Op::kPower:
          value = std::pow(std::abs(a), b);
          break;
        case Op::kLess:
          value = a < b ? 1.0 : 0.0;
          break;
        case Op::kGreater:
          value = a > b ? 1.0 : 0.0;
          break;
        default:
          break;
        }
      }
    }
    if (!IsBounded(value))
      error_ = true;
    values_[index] = value;
    return value;
  }

  const ExpressionProgram &program_;
  std::span<const double> state_;
  std::array<double, kMaxNodes> values_;
  std::array<double, kMaxNodes> adjoints_;
  bool error_ = false;
};

} // namespace

std::size_t CompiledExpression::node_count() const {
  return program_ ? program_->nodes.size() : 0;
}

std::span<const std::size_t> CompiledExpression::dependencies() const {
  return program_ ? std::span<const std::size_t>(program_->dependencies)
                  : std::span<const std::size_t>();
}

Result<CompiledExpression> CompileExpression(std::string_view text,
                                             const ExpressionBindings &bindings,
                                             ExpressionDialect dialect) {
  auto normalized = Normalize(bindings);
  if (!normalized.ok()) {
    return Result<CompiledExpression>::Fail(normalized.error().code,
                                            normalized.error().message);
  }
  auto resolver = [&](const std::string &name) {
    const auto found = normalized.value().parameters.find(name);
    if (found == normalized.value().parameters.end()) {
      return Result<double>::Fail(ErrorCode::kUnsupported,
                                  "unknown expression parameter: " + name);
    }
    return Result<double>::Ok(found->second);
  };
  auto program = Parser(text, normalized.value(), dialect, resolver).Parse();
  if (!program.ok())
    return Result<CompiledExpression>::Fail(program.error().code,
                                            program.error().message);
  CompiledExpression result;
  result.program_ =
      std::make_shared<const ExpressionProgram>(program.TakeValue());
  return Result<CompiledExpression>::Ok(std::move(result));
}

Result<ExpressionEvaluation>
EvaluateExpression(const CompiledExpression &expression,
                   std::span<const double> state) {
  if (!expression.program_) {
    return Result<ExpressionEvaluation>::Fail(ErrorCode::kInvalidStructure,
                                              "uncompiled expression");
  }
  return Evaluator(*expression.program_, state).Run();
}

Result<ParameterValues>
ResolveParameters(const std::vector<ParameterDefinition> &definitions,
                  const ParameterValues &caller,
                  const std::vector<ParameterDefinition> &overrides) {
  if (definitions.size() > 512 || overrides.size() > 512) {
    return Result<ParameterValues>::Fail(ErrorCode::kUnsupportedSize,
                                         "parameter count limit");
  }
  ExpressionBindings source;
  source.parameters = caller;
  auto normalized = Normalize(source);
  if (!normalized.ok())
    return Result<ParameterValues>::Fail(normalized.error().code,
                                         normalized.error().message);
  std::map<std::string, std::string> pending;
  for (const auto &definition : definitions) {
    if (!ValidParameterName(definition.name) ||
        !pending.emplace(Upper(definition.name), definition.expression)
             .second) {
      return Result<ParameterValues>::Fail(
          ErrorCode::kCompile, "invalid or duplicate parameter definition");
    }
  }
  ParameterValues resolved;
  std::size_t total_nodes = 0;
  for (const auto &override : overrides) {
    const auto name = Upper(override.name);
    if (!pending.contains(name))
      return Result<ParameterValues>::Fail(ErrorCode::kUnsupported,
                                           "unknown parameter override");
    if (resolved.contains(name))
      return Result<ParameterValues>::Fail(ErrorCode::kCompile,
                                           "duplicate parameter override");
    auto expression = CompileExpression(override.expression, normalized.value(),
                                        ExpressionDialect::kParameter);
    if (!expression.ok())
      return Result<ParameterValues>::Fail(expression.error().code,
                                           expression.error().message);
    total_nodes += expression.value().node_count();
    if (total_nodes > kMaxTotalNodes)
      return Result<ParameterValues>::Fail(
          ErrorCode::kUnsupportedSize, "parameter expression total node limit");
    auto value = EvaluateExpression(expression.value(), {});
    if (!value.ok())
      return Result<ParameterValues>::Fail(value.error().code,
                                           value.error().message);
    resolved.emplace(name, value.value().value);
  }
  std::set<std::string> active;
  std::function<Result<double>(const std::string &)> resolve;
  resolve = [&](const std::string &name) -> Result<double> {
    if (resolved.contains(name))
      return Result<double>::Ok(resolved.at(name));
    const auto local = pending.find(name);
    if (local == pending.end()) {
      const auto outer = normalized.value().parameters.find(name);
      if (outer == normalized.value().parameters.end())
        return Result<double>::Fail(ErrorCode::kUnsupported,
                                    "unknown parameter dependency: " + name);
      return Result<double>::Ok(outer->second);
    }
    if (active.contains(name))
      return Result<double>::Fail(ErrorCode::kUnsupported,
                                  "parameter dependency cycle: " + name);
    if (active.size() >= kMaxDepth)
      return Result<double>::Fail(ErrorCode::kUnsupportedSize,
                                  "parameter dependency depth limit");
    active.insert(name);
    auto program = Parser(local->second, normalized.value(),
                          ExpressionDialect::kParameter, resolve)
                       .Parse();
    active.erase(name);
    if (!program.ok())
      return Result<double>::Fail(program.error().code,
                                  program.error().message);
    total_nodes += program.value().nodes.size();
    if (total_nodes > kMaxTotalNodes)
      return Result<double>::Fail(ErrorCode::kUnsupportedSize,
                                  "parameter expression total node limit");
    auto value = Evaluator(program.value(), {}).Run();
    if (!value.ok())
      return Result<double>::Fail(value.error().code, value.error().message);
    resolved.emplace(name, value.value().value);
    return Result<double>::Ok(value.value().value);
  };
  for (const auto &[name, expression] : pending) {
    static_cast<void>(expression);
    auto value = resolve(name);
    if (!value.ok())
      return Result<ParameterValues>::Fail(value.error().code,
                                           value.error().message);
  }
  ParameterValues result = normalized.value().parameters;
  for (const auto &[name, value] : resolved)
    result[name] = value;
  return Result<ParameterValues>::Ok(std::move(result));
}

} // namespace ohmnivore
