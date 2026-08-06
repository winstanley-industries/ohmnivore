#ifndef OHMNIVORE_STATUS_H_
#define OHMNIVORE_STATUS_H_

#include <string>
#include <utility>
#include <variant>

namespace ohmnivore {

enum class ErrorCode {
  kParse,
  kCompile,
  kSolve,
  kInvalidStructure,
  kUnsupportedSize,
  kSingular,
  kFactorization,
  kNonFinite,
  kSolutionValidation,
  kNonConvergence,
  kIo,
  kUnsupported,
};

struct Error {
  ErrorCode code;
  std::string message;
};

template <typename T> class Result {
public:
  [[nodiscard]] static Result Ok(T value) { return Result(std::move(value)); }

  [[nodiscard]] static Result Fail(ErrorCode code, std::string message) {
    return Result(Error{.code = code, .message = std::move(message)});
  }

  [[nodiscard]] bool ok() const { return std::holds_alternative<T>(storage_); }
  [[nodiscard]] const T &value() const { return std::get<T>(storage_); }
  [[nodiscard]] T TakeValue() { return std::move(std::get<T>(storage_)); }
  [[nodiscard]] const Error &error() const { return std::get<Error>(storage_); }

private:
  explicit Result(T value) : storage_(std::move(value)) {}
  explicit Result(Error error) : storage_(std::move(error)) {}

  std::variant<T, Error> storage_;
};

[[nodiscard]] inline const char *ErrorCodeName(ErrorCode code) {
  switch (code) {
  case ErrorCode::kParse:
    return "parse";
  case ErrorCode::kCompile:
    return "compile";
  case ErrorCode::kSolve:
    return "solve";
  case ErrorCode::kInvalidStructure:
    return "invalid-structure";
  case ErrorCode::kUnsupportedSize:
    return "unsupported-size";
  case ErrorCode::kSingular:
    return "singular";
  case ErrorCode::kFactorization:
    return "factorization";
  case ErrorCode::kNonFinite:
    return "non-finite";
  case ErrorCode::kSolutionValidation:
    return "solution-validation";
  case ErrorCode::kNonConvergence:
    return "non-convergence";
  case ErrorCode::kIo:
    return "io";
  case ErrorCode::kUnsupported:
    return "unsupported";
  }
  return "unknown";
}

} // namespace ohmnivore

#endif // OHMNIVORE_STATUS_H_
