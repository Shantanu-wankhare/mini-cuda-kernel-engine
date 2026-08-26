// =============================================================================
//  mcke/core/status.hpp
//
//  WHAT: The error-handling vocabulary of the runtime: `Status`, `StatusOr<T>`,
//        and the propagation macros.
//
//  DESIGN DECISION — exceptions vs. status codes.
//  We use BOTH, on purpose, split by error class:
//
//   1. *Expected* runtime failures  -> `Status` / `StatusOr<T>` return values.
//      Out-of-memory is the canonical example. A pool allocator that cannot
//      satisfy a 2 GiB request has not encountered a bug; the caller may want
//      to evict, defragment, or fall back to cudaMalloc. Encoding that in the
//      return type forces every caller to think about it.
//
//   2. *Programmer errors / invariant violations* -> throw, or abort.
//      Passing a rank-9 shape to a rank-4 tensor is a bug in our code, not a
//      condition to recover from. Same for "cudaLaunchKernel returned
//      cudaErrorInvalidConfiguration".
//
//  Why not exceptions everywhere? Because the allocator sits on the hot path of
//  graph execution and we want its failure mode to be branch-predictable and
//  visible in the signature. Why not status codes everywhere? Because
//  `MCKE_RETURN_IF_ERROR` on a constructor is impossible, and we would end up
//  with two-phase init everywhere.
//
//  Why not std::expected? It is C++23; we target C++20 (Apple clang and the
//  nvcc host compiler on Explorer both handle C++20 cleanly). StatusOr is a
//  15-line stand-in with the same shape, so migrating later is mechanical.
// =============================================================================
#pragma once

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mcke {

enum class StatusCode : std::uint8_t {
  kOk = 0,
  kOutOfMemory,        // allocator could not satisfy the request
  kInvalidArgument,    // caller passed something structurally wrong
  kFailedPrecondition, // object is in the wrong state (e.g. graph not planned)
  kUnimplemented,      // reached a phase we have not built yet
  kInternal,           // our invariant broke
  kCudaError,          // a CUDA API/driver call failed; message carries details
};

[[nodiscard]] constexpr std::string_view to_string(StatusCode c) noexcept {
  switch (c) {
    case StatusCode::kOk:                 return "Ok";
    case StatusCode::kOutOfMemory:        return "OutOfMemory";
    case StatusCode::kInvalidArgument:    return "InvalidArgument";
    case StatusCode::kFailedPrecondition: return "FailedPrecondition";
    case StatusCode::kUnimplemented:      return "Unimplemented";
    case StatusCode::kInternal:           return "Internal";
    case StatusCode::kCudaError:          return "CudaError";
  }
  return "Unknown";
}

// `Status` is cheap when OK: the common path allocates nothing (empty
// std::string does not heap-allocate for the empty case in any mainstream
// implementation, thanks to SSO). Only the failure path pays for a message.
class [[nodiscard]] Status {
 public:
  Status() = default;                                   // OK
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

  [[nodiscard]] std::string to_string() const {
    if (ok()) return "Ok";
    return std::string(mcke::to_string(code_)) + ": " + message_;
  }

  // Escape hatch for call sites that genuinely cannot recover (main(), tests,
  // destructors). Explicit name so it is greppable in review.
  void throw_if_error() const {
    if (!ok()) throw std::runtime_error(to_string());
  }

 private:
  StatusCode  code_ = StatusCode::kOk;
  std::string message_;
};

[[nodiscard]] inline Status OkStatus() { return Status{}; }
[[nodiscard]] inline Status OutOfMemoryError(std::string m) {
  return {StatusCode::kOutOfMemory, std::move(m)};
}
[[nodiscard]] inline Status InvalidArgumentError(std::string m) {
  return {StatusCode::kInvalidArgument, std::move(m)};
}
[[nodiscard]] inline Status FailedPreconditionError(std::string m) {
  return {StatusCode::kFailedPrecondition, std::move(m)};
}
[[nodiscard]] inline Status UnimplementedError(std::string m) {
  return {StatusCode::kUnimplemented, std::move(m)};
}
[[nodiscard]] inline Status InternalError(std::string m) {
  return {StatusCode::kInternal, std::move(m)};
}
[[nodiscard]] inline Status CudaError(std::string m) {
  return {StatusCode::kCudaError, std::move(m)};
}

// -----------------------------------------------------------------------------
// StatusOr<T>: either a value or a Status. Modelled on Abseil's; deliberately
// minimal. We store the value in std::optional rather than a union so we do not
// have to hand-write the lifetime management — this type is used on the host
// control path only, never inside a kernel, so the extra bool is free.
// -----------------------------------------------------------------------------
template <typename T>
class [[nodiscard]] StatusOr {
 public:
  StatusOr(T value) : value_(std::move(value)) {}                 // NOLINT: implicit is the point
  StatusOr(Status status) : status_(std::move(status)) {           // NOLINT
    assert(!status_.ok() && "StatusOr must not be constructed from OkStatus");
  }

  [[nodiscard]] bool ok() const noexcept { return value_.has_value(); }
  [[nodiscard]] const Status& status() const noexcept { return status_; }

  [[nodiscard]] T&       value()       { return value_.value(); }
  [[nodiscard]] const T& value() const { return value_.value(); }
  [[nodiscard]] T&       operator*()       noexcept { return *value_; }
  [[nodiscard]] const T& operator*() const noexcept { return *value_; }
  T* operator->() noexcept { return &*value_; }

  // Consume the value, or throw. For main()/tests.
  [[nodiscard]] T value_or_throw() && {
    status_.throw_if_error();
    return std::move(*value_);
  }

 private:
  Status           status_{};
  std::optional<T> value_{};
};

}  // namespace mcke

// -----------------------------------------------------------------------------
// Propagation macros.
//
// Why macros and not a monadic `and_then`? Because early-return is what makes
// status-code code readable, and C++20 has no `?` operator. These two macros
// are the entire ergonomics story for Status; without them people start
// ignoring return values.
//
// The `do { } while (0)` wrapper makes the macro a single statement so it works
// correctly as the body of an unbraced `if`.
// -----------------------------------------------------------------------------
#define MCKE_RETURN_IF_ERROR(expr)                    \
  do {                                                \
    ::mcke::Status _mcke_status = (expr);             \
    if (!_mcke_status.ok()) return _mcke_status;      \
  } while (0)

// Assign-or-return. Note the unique temporary name: nesting two of these in one
// scope would otherwise shadow. (A real codebase would append __LINE__ via a
// token-paste helper; we keep it simple and just avoid nesting.)
#define MCKE_ASSIGN_OR_RETURN(lhs, expr)              \
  auto _mcke_or = (expr);                             \
  if (!_mcke_or.ok()) return _mcke_or.status();       \
  lhs = std::move(*_mcke_or)
