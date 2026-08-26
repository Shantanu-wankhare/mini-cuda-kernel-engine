// =============================================================================
//  mcke/tensor/tensor.hpp
//
//  WHAT: `Storage` (owns device bytes) and `Tensor` (a typed, shaped *view* into
//        storage). Also `TensorRef`, the POD we actually pass to kernels.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — split ownership (Storage) from interpretation (Tensor).
//  This is the single most important structural choice in any tensor library,
//  and every mature one (PyTorch, NumPy, JAX) makes it the same way:
//
//     Storage : void* + bytes + allocator + last-use stream.  Refcounted.
//     Tensor  : shared_ptr<Storage> + byte offset + Shape + DType.
//
//  Why: reshape, slice, and "two graph nodes writing disjoint halves of one
//  buffer" all become free operations on the *view*, with no copies and no
//  ambiguity about who frees the memory. If Tensor owned raw memory directly,
//  every view would need either a copy or a raw non-owning pointer (dangling
//  bugs).
//
//  Why shared_ptr and not a hand-rolled intrusive refcount? Because refcount
//  traffic here is *host-side* and happens once per tensor per graph build —
//  nanoseconds, thousands of times per second at most, versus microsecond-scale
//  kernel launches. Hand-rolling would buy nothing and cost correctness. (In a
//  library where tensors are created per-element, the answer flips; know why
//  yours differs.)
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — Storage records `last_use_stream`.
//  This is what makes the stream-ordered allocator safe *by construction*: when
//  a Storage's refcount hits zero, its destructor returns the block to the
//  allocator tagged with the stream that last touched it, which is precisely
//  what the allocator needs to decide when reuse is safe. If we made the caller
//  remember, we would eventually get it wrong — and the bug would be a silent
//  wrong number, not a crash. Design the API so the dangerous thing is
//  impossible, not merely documented.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — `TensorRef` for kernel arguments.
//  A Tensor holds a shared_ptr; you cannot pass that to a kernel. So kernels
//  take `TensorRef`: a trivially-copyable {ptr, Shape} aggregate produced by
//  Tensor::ref(). Making this a distinct type (rather than passing raw pointers)
//  keeps shape information available inside the kernel and makes the
//  host/device boundary explicit in every signature.
// =============================================================================
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "mcke/core/dtype.hpp"
#include "mcke/core/status.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/runtime/stream.hpp"
#include "mcke/tensor/shape.hpp"

namespace mcke {

// -----------------------------------------------------------------------------
// Storage: owning handle on a device allocation.
// -----------------------------------------------------------------------------
class Storage {
 public:
  // Factory rather than a public constructor: allocation can fail, and we want
  // that in the return type.
  [[nodiscard]] static StatusOr<std::shared_ptr<Storage>> create(
      DeviceAllocator& alloc, std::size_t bytes, rt::StreamHandle stream);

  Storage(const Storage&)            = delete;
  Storage& operator=(const Storage&) = delete;
  ~Storage();

  [[nodiscard]] void*       data()       noexcept { return alloc_.ptr; }
  [[nodiscard]] const void* data() const noexcept { return alloc_.ptr; }
  [[nodiscard]] std::size_t bytes() const noexcept { return alloc_.bytes; }

  // Called by ops after enqueuing work that reads or writes this storage.
  // Keeping it a single mutable field (rather than a set of streams) is a
  // conscious simplification: a tensor consumed by two streams in parallel gets
  // the *later* stream recorded, which is unsafe. Phase 4 upgrades this to a
  // small vector of (stream, event) pairs when the scheduler can actually
  // produce that situation. Noted here so the limitation is explicit rather
  // than a lurking bug.
  void note_use(rt::StreamHandle s) noexcept { last_use_stream_ = s; }
  [[nodiscard]] rt::StreamHandle last_use_stream() const noexcept { return last_use_stream_; }

 private:
  Storage(DeviceAllocator& a, Allocation alloc, rt::StreamHandle s)
      : allocator_(a), alloc_(alloc), last_use_stream_(s) {}

  DeviceAllocator& allocator_;
  Allocation       alloc_{};
  rt::StreamHandle last_use_stream_{};
};

// -----------------------------------------------------------------------------
// TensorRef: the POD that crosses into device code.
// -----------------------------------------------------------------------------
template <typename T>
struct TensorRef {
  T*    data = nullptr;
  Shape shape{};

  MCKE_HOST_DEVICE dim_t numel() const noexcept { return shape.numel(); }
  MCKE_HOST_DEVICE dim_t rows()  const noexcept { return shape.rows(); }
  MCKE_HOST_DEVICE dim_t cols()  const noexcept { return shape.cols(); }
};

// -----------------------------------------------------------------------------
// Tensor: shaped, typed view.
// -----------------------------------------------------------------------------
class Tensor {
 public:
  Tensor() = default;

  // Allocate a fresh contiguous tensor.
  [[nodiscard]] static StatusOr<Tensor> empty(DeviceAllocator& alloc, Shape shape,
                                              DType dtype, rt::StreamHandle stream);

  // Zero-copy view with a different shape. Fails if numel differs.
  [[nodiscard]] StatusOr<Tensor> reshape(Shape new_shape) const;

  // Sub-view starting at `elem_offset` with `shape`. This is how the Phase-4
  // memory planner hands two graph nodes disjoint slices of one buffer.
  [[nodiscard]] StatusOr<Tensor> slice(dim_t elem_offset, Shape shape) const;

  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] DType dtype() const noexcept { return dtype_; }
  [[nodiscard]] dim_t numel() const noexcept { return shape_.numel(); }
  [[nodiscard]] std::size_t nbytes() const noexcept { return shape_.bytes(dtype_); }
  [[nodiscard]] bool defined() const noexcept { return storage_ != nullptr; }

  [[nodiscard]] void* data_ptr() const noexcept {
    return storage_ ? static_cast<char*>(storage_->data()) + byte_offset_ : nullptr;
  }

  template <typename T>
  [[nodiscard]] T* data_as() const {
    // The one place we check the runtime tag against the compile-time type.
    // Checking here rather than in every kernel launcher means a type confusion
    // becomes an immediate, localised failure instead of garbage numbers.
    if (dtype_ != kDTypeOf<T>) return nullptr;
    return static_cast<T*>(data_ptr());
  }

  template <typename T>
  [[nodiscard]] TensorRef<T> ref() const {
    return TensorRef<T>{data_as<T>(), shape_};
  }

  void note_use(rt::StreamHandle s) const { if (storage_) storage_->note_use(s); }
  [[nodiscard]] const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }

  [[nodiscard]] std::string to_string() const;

  // --- Host <-> device transfer. Async, so `stream` must be synchronised (or
  //     an event waited on) before the host buffer is read/reused. We take a
  //     span-like (ptr, count) rather than a container to keep the runtime free
  //     of container policy.
  [[nodiscard]] Status copy_from_host(const void* src, std::size_t bytes, rt::StreamHandle stream);
  [[nodiscard]] Status copy_to_host(void* dst, std::size_t bytes, rt::StreamHandle stream) const;

 private:
  Tensor(std::shared_ptr<Storage> s, std::size_t byte_offset, Shape shape, DType dt)
      : storage_(std::move(s)), byte_offset_(byte_offset), shape_(shape), dtype_(dt) {}

  std::shared_ptr<Storage> storage_;
  std::size_t              byte_offset_ = 0;
  Shape                    shape_{};
  DType                    dtype_ = DType::kF32;
};

}  // namespace mcke
