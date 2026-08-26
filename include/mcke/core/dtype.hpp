// =============================================================================
//  mcke/core/dtype.hpp
//
//  WHAT: The runtime's type tag (`DType`), its size/name metadata, and the
//        compile-time mapping between C++ types and tags.
//
//  DESIGN DECISION — runtime tag vs. `Tensor<T>` template.
//  A templated `Tensor<float>` gives compile-time dispatch and zero overhead,
//  but a *graph* has to store heterogeneous nodes in one container, which then
//  needs type erasure anyway (std::variant or a virtual base). Every real
//  runtime (PyTorch, XLA, TVM) therefore carries a runtime dtype tag on the
//  tensor and dispatches once, at kernel-launch granularity. The dispatch cost
//  is a switch executed once per *kernel launch* (microseconds apart), not once
//  per element — so it is free in practice.
//
//  We keep the compile-time side too (`DTypeOf<T>`) so kernel launchers can be
//  templated and still check that the tensor they were handed matches.
//
//  Phase note: only F32 is exercised in Phases 0-4. F16/BF16 land in Phase 6
//  (tensor cores) — but the tag values are reserved now so we never renumber a
//  serialized enum.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "mcke/core/config.hpp"

namespace mcke {

enum class DType : std::uint8_t {
  kF32 = 0,   // IEEE binary32. The only dtype in Phases 0-4.
  kF16,       // IEEE binary16  — 10-bit mantissa. Tensor-core input type.
  kBF16,      // bfloat16       — 7-bit mantissa, f32 exponent range. Safer for
              //                  training-style accumulation; needs sm_80+.
  kI32,
  kU8,
  kCount
};

[[nodiscard]] MCKE_HOST_DEVICE constexpr std::size_t dtype_size(DType t) noexcept {
  switch (t) {
    case DType::kF32:  return 4;
    case DType::kF16:  return 2;
    case DType::kBF16: return 2;
    case DType::kI32:  return 4;
    case DType::kU8:   return 1;
    default:           return 0;
  }
}

[[nodiscard]] constexpr std::string_view dtype_name(DType t) noexcept {
  switch (t) {
    case DType::kF32:  return "f32";
    case DType::kF16:  return "f16";
    case DType::kBF16: return "bf16";
    case DType::kI32:  return "i32";
    case DType::kU8:   return "u8";
    default:           return "invalid";
  }
}

// -----------------------------------------------------------------------------
// Compile-time C++ type  ->  DType tag.
// Specialised per supported type; an unsupported type fails to compile with a
// readable message instead of silently picking a default.
// -----------------------------------------------------------------------------
// NOTE (subtle, and worth internalising): the primary template is left *empty*
// rather than holding a `static_assert(sizeof(T)==0, ...)`. A static_assert here
// would be a hard error the moment anything names `DTypeOf<T>` — including the
// `requires`-expression in the `DeviceScalar` concept below — which would make
// the concept uncheckable instead of simply false. Empty primary => "no member
// named value" => SFINAE-friendly => concepts work.
template <typename T>
struct DTypeOf {};
template <> struct DTypeOf<float>        { static constexpr DType value = DType::kF32; };
template <> struct DTypeOf<std::int32_t> { static constexpr DType value = DType::kI32; };
template <> struct DTypeOf<std::uint8_t> { static constexpr DType value = DType::kU8;  };
// f16/bf16 specialisations are added in Phase 6 alongside <cuda_fp16.h>.

template <typename T>
inline constexpr DType kDTypeOf = DTypeOf<T>::value;

// -----------------------------------------------------------------------------
// C++20 concept: what may appear as a kernel element type.
// Using a concept (rather than a static_assert inside the function) means the
// error message points at the *call site* and the overload simply drops out of
// the candidate set — which matters once we have f32 and f16 GEMM overloads.
// -----------------------------------------------------------------------------
template <typename T>
concept DeviceScalar = requires { DTypeOf<T>::value; };

}  // namespace mcke
