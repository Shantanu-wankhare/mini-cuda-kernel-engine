// =============================================================================
//  mcke/memory/allocator.hpp
//
//  WHAT: The abstract device-allocator interface every memory strategy
//        implements, the `Allocation` handle, and the statistics block that
//        makes our benchmark claims verifiable.
//
//  ---------------------------------------------------------------------------
//  WHY WE ARE BUILDING THIS AT ALL (the number that justifies the project)
//
//  cudaMalloc / cudaFree are *not* malloc. They:
//    - take a driver lock,
//    - may synchronise the device implicitly (cudaFree is a synchronising call
//      on older drivers, and always a heavyweight one),
//    - manipulate GPU page tables / VA mappings.
//  Cost is typically 10-100 us, sometimes worse, versus ~100 ns for a pool hit.
//  A graph with 50 intermediate tensors executed 1000 times would spend
//  50 * 1000 * 2 * ~30 us = 3 seconds in the allocator. Worse, because cudaFree
//  can synchronise, it destroys the async overlap we build in Phase 4. So the
//  allocator is not an optimisation detail — it is a *correctness requirement*
//  for an async runtime.
//
//  Our strategy is the standard one: reserve a few large slabs from cudaMalloc
//  once, then sub-allocate from them on the host with pure arithmetic.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — abstract base class with virtual calls.
//  A CRTP/templated allocator would avoid the vtable. We deliberately take the
//  virtual call, because:
//    * cost is ~2 ns of indirect branch against a 100 ns fast path and a
//      microsecond-scale kernel launch. Unmeasurable.
//    * it lets `--allocator=buddy|freelist|raw` be a *runtime* flag, so the
//      Phase 2 benchmark runs all three strategies in one process, on one
//      allocation trace, with one warm cache. That comparability is worth far
//      more than 2 ns.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — stream-ordered semantics (the subtle, important one).
//
//  Naive pooling is *wrong* on a GPU. Consider:
//      t = alloc(); kernelA<<<..., streamA>>>(t); free(t);
//      u = alloc();                       // pool returns the same memory!
//      kernelB<<<..., streamB>>>(u);      // may run CONCURRENTLY with kernelA
//  The host `free` happened long before kernelA actually executed — freeing is
//  a host-side bookkeeping operation, the GPU work is still queued. If the pool
//  hands that block to a different stream, kernelA and kernelB race on the same
//  bytes and you get silent, nondeterministic wrong numbers. This is the single
//  hardest bug class in GPU memory management, and it is why every allocate /
//  deallocate call in this interface takes a stream.
//
//  The rules we implement (same model as CUDA's own cudaMallocAsync pools):
//    1. A block freed on stream S may be immediately reused by a later
//       allocation *on the same stream S* — the stream is in-order, so any
//       kernel using the block has been enqueued before the reusing kernel.
//       No synchronisation needed. This is the fast path and the common case.
//    2. Reuse from a *different* stream requires proof that the prior work
//       finished. Two options:
//         (a) record an Event on the freeing stream and make the new consumer
//             cudaStreamWaitEvent on it — no host stall (preferred), or
//         (b) poll `Event::query()` and only reuse blocks whose event has
//             completed — no stall either, but the block stays parked.
//       We implement (b) in Phase 2 (simple, always safe) and add (a) in
//       Phase 4 when the scheduler already owns per-node events.
//
//  `deallocate` therefore records which stream last touched the block; the
//  allocator keeps it on a per-stream "pending" list until rule 1 or 2 allows
//  reuse.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "mcke/core/config.hpp"
#include "mcke/core/status.hpp"
#include "mcke/runtime/stream.hpp"

namespace mcke {

// -----------------------------------------------------------------------------
// Allocation: what the allocator hands back.
//
// It is a *value* (16-24 bytes, trivially copyable), not an owning RAII type.
// Ownership lives in `Storage` (see tensor.hpp) one layer up. Reason: the
// allocator must stay usable by non-RAII code paths (the graph's memory planner
// hands out slices of one big block), and mixing ownership into the low-level
// handle makes that awkward.
//
// `bytes` (what we actually reserved) is kept alongside `requested_bytes` (what
// the caller asked for) specifically so we can compute internal fragmentation —
// the headline weakness of a buddy allocator. If you don't record it you cannot
// report it.
// -----------------------------------------------------------------------------
struct Allocation {
  void*       ptr             = nullptr;
  std::size_t bytes           = 0;   // usable size of the block handed out
  std::size_t requested_bytes = 0;   // what the caller asked for
  std::uint32_t slab_id       = 0;   // which reserved slab this came from
  std::uint32_t block_id      = 0;   // implementation-defined (buddy: heap index)

  [[nodiscard]] bool valid() const noexcept { return ptr != nullptr; }
  [[nodiscard]] std::size_t internal_waste() const noexcept {
    return bytes - requested_bytes;
  }
};

// -----------------------------------------------------------------------------
// AllocatorStats: the evidence for every performance claim we will make.
//
// `raw_malloc_calls` is the one to watch. The entire thesis of this component
// is "this number stops growing after warm-up". A benchmark that reports only
// wall-clock leaves a reader unable to tell a real pool from a lucky cache;
// reporting raw_malloc_calls == 4 after 100k allocations proves it.
// -----------------------------------------------------------------------------
struct AllocatorStats {
  std::uint64_t alloc_calls        = 0;   // allocate() invocations
  std::uint64_t free_calls         = 0;
  std::uint64_t raw_malloc_calls   = 0;   // actual cudaMalloc calls made
  std::uint64_t raw_free_calls     = 0;
  std::uint64_t oom_events         = 0;   // requests we could not satisfy
  std::uint64_t deferred_reuses    = 0;   // blocks parked awaiting an event (rule 2)

  std::size_t bytes_reserved       = 0;   // total cudaMalloc'd (our footprint)
  std::size_t bytes_in_use         = 0;   // handed out right now
  std::size_t bytes_requested      = 0;   // sum of requested_bytes in use
  std::size_t peak_bytes_in_use    = 0;
  std::size_t largest_free_block   = 0;   // external-fragmentation indicator

  // Internal fragmentation: bytes we reserved but the caller cannot use.
  [[nodiscard]] std::size_t internal_waste() const noexcept {
    return bytes_in_use >= bytes_requested ? bytes_in_use - bytes_requested : 0;
  }
  // Fraction of our reserved footprint that is actually doing useful work.
  [[nodiscard]] double utilisation() const noexcept {
    return bytes_reserved ? static_cast<double>(bytes_requested) / static_cast<double>(bytes_reserved) : 0.0;
  }
  [[nodiscard]] std::string to_string() const;
};

// -----------------------------------------------------------------------------
// The interface.
// -----------------------------------------------------------------------------
class DeviceAllocator {
 public:
  virtual ~DeviceAllocator() = default;

  // Allocate at least `bytes`, aligned to at least kDeviceAlignment.
  // `stream` is the stream the caller intends to use the memory on — see the
  // stream-ordered discussion above. Pass rt::Stream::default_stream() if you
  // are in synchronous code.
  [[nodiscard]] virtual StatusOr<Allocation> allocate(std::size_t bytes,
                                                     rt::StreamHandle stream) = 0;

  // Return a block. `stream` must be the stream on which the last work using
  // this block was enqueued. Passing the wrong stream is a correctness bug the
  // allocator cannot detect — which is why Tensor/Storage records it for you.
  virtual Status deallocate(const Allocation& a, rt::StreamHandle stream) = 0;

  // Release *unused* reserved slabs back to the driver (calls cudaFree).
  // Needed when another process/library on the same GPU needs memory. Expensive
  // and synchronising; never call it inside a timed region.
  virtual Status trim() { return OkStatus(); }

  [[nodiscard]] virtual AllocatorStats stats() const = 0;
  [[nodiscard]] virtual std::string_view name() const = 0;

  // Debug aid: walk internal structures and verify invariants (no overlapping
  // live blocks, free-list sizes match the bitmap, coalescing is maximal).
  // O(n) — tests and asserts only. Having this from day one is the difference
  // between "the allocator is broken somewhere" and "node 4712 has a live
  // parent".
  [[nodiscard]] virtual Status validate() const { return OkStatus(); }
};

// A pass-through allocator: every allocate() is a cudaMalloc. This is the
// *baseline* we benchmark against, and it is genuinely useful as a fallback for
// allocations larger than a slab.
class RawDeviceAllocator final : public DeviceAllocator {
 public:
  StatusOr<Allocation> allocate(std::size_t bytes, rt::StreamHandle stream) override;
  Status deallocate(const Allocation& a, rt::StreamHandle stream) override;
  [[nodiscard]] AllocatorStats stats() const override { return stats_; }
  [[nodiscard]] std::string_view name() const override { return "raw(cudaMalloc)"; }

 private:
  AllocatorStats stats_{};
};

}  // namespace mcke
