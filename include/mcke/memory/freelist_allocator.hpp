// =============================================================================
//  mcke/memory/freelist_allocator.hpp
//
//  WHAT: A segregated size-class caching pool — the *other* classic design, and
//        the one PyTorch's CUDACachingAllocator is closest to. Phase 2b.
//
//  WHY BUILD BOTH: because the interesting engineering claim is not "I wrote an
//  allocator", it is "here is the workload where each design wins, with
//  numbers". The two designs fail in opposite directions, and one trace makes
//  both failures visible.
//
//  ---------------------------------------------------------------------------
//  THE SIZE-CLASS LADDER
//
//  Two regimes, because no single rule is good across five orders of magnitude:
//
//    bytes <= small_large_split      : LINEAR, multiples of small_class_granularity
//        class i  ->  block (i+1)*G          (G = 512 B by default)
//        n_small = small_large_split / G     (2048 classes by default)
//
//    bytes >  small_large_split      : POWERS OF TWO
//        class n_small+k  ->  block 2^(log2(S)+1+k)
//
//  The linear regime is the whole point: it is *finer* than a power of two, so a
//  768-element f32 tensor (3072 B) gets exactly 3072 B instead of buddy's 4096.
//  That is where this design beats buddy.
//
//  ---------------------------------------------------------------------------
//  A PREDICTION THIS LADDER MAKES, WORTH KNOWING BEFORE THE BENCHMARK RUNS
//
//  Above small_large_split the ladder IS a power-of-two ladder — identical
//  rounding to buddy. So for any workload whose bytes are dominated by
//  multi-MiB tensors (which is every transformer: weight matrices and attention
//  scores are 2-16 MiB while biases and layernorm scales are kilobytes), this
//  allocator and buddy must round the same way and tie on internal
//  fragmentation. The advantage only appears in the size range where the ladder
//  is genuinely finer, i.e. below 1 MiB.
//
//  That contradicts the naive expectation ("size classes beat buddy on
//  non-power-of-two shapes"), and it is a real lesson about ladders rather than a
//  bug: a size-class scheme is only as good as its granularity *in the range
//  where your bytes actually are*. Phase 2c measures it per phase so the tie and
//  the win are both visible.
//
//  ---------------------------------------------------------------------------
//  THE DEFINING TRADEOFF: no coalescing, ever.
//
//  deallocate() pushes the pointer back onto its class list and stops. Two
//  adjacent free blocks are never merged. That is what makes free O(1) — no
//  cascade, no buddy lookup, no tree walk — and it is also what makes this design
//  fragment externally: you can hold 90 MiB free across a thousand 512 B blocks
//  and still fail a 1 MiB request. Buddy cannot get into that state; this design
//  cannot get out of it.
//
//  So the honest summary is that buddy and freelist trade the SAME mechanism in
//  opposite directions:
//      coalescing  =>  slower free, larger contiguous free extents
//      no coalescing => faster free, fragmenting free extents
//  Phase 2c reports both halves of that sentence as numbers.
//
//  `split_large_blocks` is the usual mitigation, and it is a tunable rather than
//  a default because it *creates* small blocks that can never merge back — it
//  trades a fragmentation problem now for a worse one later.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "mcke/memory/allocator.hpp"
#include "mcke/memory/reuse_policy.hpp"

namespace mcke {

struct FreeListConfig {
  std::size_t slab_bytes              = std::size_t{256} << 20;
  // Cap on total pool footprint (0 = unlimited). Present for parity with
  // BuddyConfig::max_total_bytes: the Phase 2c comparison is only fair if both
  // allocators are held to the same footprint budget, and without a cap this one
  // would simply grow instead of ever reporting the external fragmentation we are
  // trying to observe.
  std::size_t max_total_bytes         = 0;
  std::size_t small_class_granularity = 512;                 // below the split point
  std::size_t small_large_split       = std::size_t{1} << 20; // 1 MiB
  std::size_t large_alloc_threshold   = std::size_t{64} << 20;
  // If true, a request may be served from a larger cached block by splitting it
  // (and the remainder re-cached). Reduces fragmentation, costs O(1) extra
  // bookkeeping, but *creates* small blocks that can never merge back — the
  // classic reason this heuristic is a tunable and not a default.
  bool        split_large_blocks      = false;

  ReusePolicy reuse_policy            = ReusePolicy::kCoarseStreamPoll;
};

// Not `final`, for the same reason BuddyAllocator is not: the stream-completion
// probes are protected virtuals so a test subclass can drive the deferral path on
// a machine with no GPU. See buddy_allocator.hpp for why that costs nothing here.
class FreeListAllocator : public DeviceAllocator {
 public:
  explicit FreeListAllocator(FreeListConfig cfg = {});
  ~FreeListAllocator() override;

  [[nodiscard]] Status reserve(std::size_t bytes);

  StatusOr<Allocation> allocate(std::size_t bytes, rt::StreamHandle stream) override;
  Status               deallocate(const Allocation& a, rt::StreamHandle stream) override;
  Status               trim() override;
  [[nodiscard]] AllocatorStats  stats() const override;
  [[nodiscard]] std::string_view name() const override { return "freelist"; }
  [[nodiscard]] Status validate() const override;

  // Maps a byte count onto its size-class index, and back to the block size that
  // class hands out.
  //
  // These are the single most test-worthy functions in the class: an off-by-one
  // here silently hands out blocks that are TOO SMALL, which is memory corruption
  // rather than mere waste. The round-trip invariant
  //     class_block_bytes(size_class_of(n)) >= n
  // is asserted exhaustively over every byte count in the small regime.
  [[nodiscard]] std::size_t size_class_of(std::size_t bytes) const;
  [[nodiscard]] std::size_t class_block_bytes(std::size_t class_idx) const;
  [[nodiscard]] std::size_t num_classes() const { return classes_.size(); }

  [[nodiscard]] std::string dump_free_map() const;

 protected:
  // Test seams, identical in shape to BuddyAllocator's. See reuse_policy.hpp.
  [[nodiscard]] virtual bool stream_completed(rt::StreamHandle stream) const {
    return rt::stream_query(stream);
  }
  [[nodiscard]] virtual bool event_completed(std::uint32_t event_slot) const;

 private:
  struct Slab {
    void*       base  = nullptr;   // nullptr => dead slot (see trim())
    std::size_t bytes = 0;
    // Bump pointer: bytes [0, bump) have been carved into blocks at least once.
    // Everything from bump to the end is virgin. Carving is how a class list gets
    // its first block; after that, blocks only ever cycle through the lists.
    std::size_t bump  = 0;
  };

  struct CachedBlock {
    void*       ptr   = nullptr;
    std::size_t bytes = 0;         // == class_block_bytes of the list it sits in
  };

  // A block whose caller has released it but whose last GPU work may still be
  // running. While parked it is in NEITHER live_ nor any class list, so it cannot
  // be handed out — the same safety property as buddy keeping a parked node kUsed.
  struct PendingFree {
    void*            ptr        = nullptr;
    std::size_t      bytes      = 0;
    rt::StreamHandle stream     = {};
    std::uint32_t    event_slot = kNoEventSlot;
  };

  [[nodiscard]] StatusOr<std::uint32_t> add_slab(std::size_t bytes);
  [[nodiscard]] StatusOr<Allocation>    allocate_bypass(std::size_t bytes);
  [[nodiscard]] Status                  deallocate_bypass(const Allocation& a);
  // Try to serve `cls` by splitting a larger cached block. Only used when
  // cfg_.split_large_blocks is set.
  [[nodiscard]] void*                   try_split_larger(std::size_t cls);
  void                                  cache_block(void* ptr, std::size_t bytes);
  void                                  reclaim_completed(rt::StreamHandle stream);
  void                                  drain_pending_blocking();
  [[nodiscard]] StatusOr<std::uint32_t> acquire_event_slot();
  void                                  release_event_slot(std::uint32_t slot);
  [[nodiscard]] std::size_t             compute_largest_free_block() const;
  [[nodiscard]] Status                  validate_config() const;

  FreeListConfig                          cfg_;
  std::vector<Slab>                       slabs_;
  std::vector<std::vector<CachedBlock>>   classes_;   // indexed by size class
  // ptr -> block bytes, for every LIVE allocation.
  //
  // Partly redundant: Allocation already carries `bytes`, so deallocate could
  // route the block home without this map. It is kept because it also buys a real
  // double-free / bogus-pointer check, which buddy gets for free from its
  // per-node state and this design otherwise has no way to perform. So the map is
  // paying for validation, not for routing — and Phase 2c measures what that
  // validation costs (one hash insert per allocate, one find+erase per free,
  // plus rehash spikes in the p99 tail).
  std::unordered_map<void*, std::size_t>  live_;
  std::vector<PendingFree>                pending_;
  std::vector<rt::Event>                  event_pool_;
  std::vector<std::uint32_t>              free_event_slots_;
  std::unordered_map<void*, std::size_t>  bypassed_;
  std::size_t                             n_small_ = 0;   // count of linear classes
  unsigned                                split_log2_ = 0;// log2(small_large_split)
  AllocatorStats                          stats_{};

  friend struct FreeListTestAccess;
};

}  // namespace mcke
