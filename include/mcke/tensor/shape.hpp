// =============================================================================
//  mcke/tensor/shape.hpp
//
//  WHAT: Fixed-capacity shape/stride vectors and the index math shared by host
//        planning code and device kernels.
//
//  DESIGN DECISION — inline fixed-capacity array, not std::vector.
//  A Shape must be usable inside a kernel (to compute an index from a linear
//  thread id) and must be copyable into a kernel argument. std::vector cannot
//  be either: it heap-allocates and its data lives in host memory. So Shape is
//  a POD with a compile-time max rank. Rank 4 covers everything in this project
//  (batch, head, row, col); we set 5 for headroom. `sizeof(Shape)` = 48 bytes,
//  well inside the 4 KB kernel-parameter limit even if we pass several.
//
//  DESIGN DECISION — row-major contiguous only, in Phases 0-4.
//  Supporting arbitrary strides means every kernel needs a general index
//  computation (a multiply-add per dimension, per element) instead of a single
//  linear offset. That destroys the tight inner loops we are building in Phase 3
//  and it hides coalescing behaviour. So: `Shape` carries strides (so the design
//  is ready), kernels assert `is_contiguous()`, and a `transpose` op materialises
//  data instead of producing a strided view. We revisit in Phase 6, where a
//  strided GEMM epilogue is genuinely useful.
// =============================================================================
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "mcke/core/config.hpp"
#include "mcke/core/dtype.hpp"

namespace mcke {

inline constexpr int kMaxRank = 5;

using dim_t = std::int64_t;   // signed: makes reverse loops and -1 sentinels safe

class Shape {
 public:
  Shape() = default;

  // Host-only constructor (initializer_list is not device-constructible).
  Shape(std::initializer_list<dim_t> dims) {
    rank_ = static_cast<int>(dims.size());
    int i = 0;
    for (dim_t d : dims) dims_[i++] = d;
    compute_contiguous_strides();
  }

  MCKE_HOST_DEVICE int rank() const noexcept { return rank_; }

  MCKE_HOST_DEVICE dim_t dim(int i) const noexcept { return dims_[i]; }
  MCKE_HOST_DEVICE dim_t stride(int i) const noexcept { return strides_[i]; }

  MCKE_HOST_DEVICE dim_t numel() const noexcept {
    dim_t n = 1;
    for (int i = 0; i < rank_; ++i) n *= dims_[i];
    return n;
  }

  MCKE_HOST_DEVICE std::size_t bytes(DType t) const noexcept {
    return static_cast<std::size_t>(numel()) * dtype_size(t);
  }

  // True iff strides are exactly the row-major contiguous ones. Every kernel
  // in Phases 0-4 asserts this before launch.
  MCKE_HOST_DEVICE bool is_contiguous() const noexcept {
    dim_t expect = 1;
    for (int i = rank_ - 1; i >= 0; --i) {
      if (strides_[i] != expect) return false;
      expect *= dims_[i];
    }
    return true;
  }

  // Linear offset (in *elements*) of a multi-index. Used on the device for the
  // general case; contiguous kernels skip it entirely and index linearly, which
  // is the point of asserting contiguity.
  MCKE_HOST_DEVICE dim_t offset_of(const dim_t* idx) const noexcept {
    dim_t off = 0;
    for (int i = 0; i < rank_; ++i) off += idx[i] * strides_[i];
    return off;
  }

  // Collapse to 2D as (rows = product of leading dims, cols = last dim).
  // This is the shape every "row-wise" kernel actually wants: softmax over the
  // last axis of a [B, H, S, S] tensor is just a row-wise softmax on a
  // [B*H*S, S] matrix, so one kernel covers all ranks. Recognising this early
  // is what stops the kernel count from exploding.
  MCKE_HOST_DEVICE dim_t rows() const noexcept {
    dim_t r = 1;
    for (int i = 0; i < rank_ - 1; ++i) r *= dims_[i];
    return r;
  }
  MCKE_HOST_DEVICE dim_t cols() const noexcept { return rank_ > 0 ? dims_[rank_ - 1] : 0; }

  [[nodiscard]] bool operator==(const Shape& o) const noexcept {
    if (rank_ != o.rank_) return false;
    for (int i = 0; i < rank_; ++i)
      if (dims_[i] != o.dims_[i] || strides_[i] != o.strides_[i]) return false;
    return true;
  }

  [[nodiscard]] std::string to_string() const {
    std::string s = "[";
    for (int i = 0; i < rank_; ++i) { s += std::to_string(dims_[i]); if (i + 1 < rank_) s += ","; }
    return s + "]";
  }

 private:
  MCKE_HOST_DEVICE void compute_contiguous_strides() noexcept {
    dim_t acc = 1;
    for (int i = rank_ - 1; i >= 0; --i) { strides_[i] = acc; acc *= dims_[i]; }
  }

  // Raw C arrays, deliberately, NOT std::array.
  //
  // std::array would be nicer C++ (value semantics, .at()), but its
  // operator[] is a *host* constexpr member function. Calling it from
  // __device__ code only works via --expt-relaxed-constexpr, and even then nvcc
  // emits warnings and the behaviour depends on the host standard library.
  // Since these members are read inside kernels, raw arrays keep the type
  // unambiguously device-safe: aggregate, trivially copyable, no member
  // functions involved. This is a recurring rule in CUDA C++ — a type that
  // crosses into device code should be a plain aggregate.
  dim_t dims_[kMaxRank]{};
  dim_t strides_[kMaxRank]{};
  int rank_ = 0;
};

}  // namespace mcke
