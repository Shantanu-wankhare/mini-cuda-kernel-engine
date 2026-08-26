// =============================================================================
//  mcke/memory/freelist_allocator.hpp
//
//  WHAT: A segregated size-class caching pool — the *other* classic design, and
//        the one PyTorch's CUDACachingAllocator is closest to. Phase 2b.
//
//  WHY BUILD BOTH: because the interesting engineering claim is not "I wrote an
//  allocator", it is "here is the workload where each design wins, with
//  numbers". Concretely we expect:
//
//    * DL-ish shapes (768, 3072, 50257 elements — not powers of two) →
//      buddy rounds up to the next power of two and wastes up to 2x. The
//      size-class pool with a finer ladder (e.g. 128-byte granularity below
//      1 MiB, then power-of-two above) wastes far less. Prediction:
//      buddy utilisation ~55-70%, freelist ~85-95%.
//    * Long, churny sequences of mixed sizes → the freelist pool cannot merge
//      neighbours, so `largest_free_block` collapses and it eventually OOMs
//      while still holding plenty of free bytes. Buddy holds up.
//   We will run exactly that A/B in Phase 2c and put the real numbers in
//   RESULTS.md. If the prediction is wrong, that goes in RESULTS.md too — a
//   wrong prediction you measured is worth more than a right one you assumed.
//
//  ---------------------------------------------------------------------------
//  DESIGN
//    size class ladder:  round_up(bytes) →  class index
//       < 1 MiB   : multiples of 512 B         (2048 classes)
//       >= 1 MiB  : powers of two              (~14 classes)
//    Each class owns a vector<void*> of cached blocks. allocate():
//       pop from class list                                       -> O(1)
//       else carve from the current bump-pointer region in a slab -> O(1)
//       else new slab / bypass to cudaMalloc
//    deallocate(): push back onto its class list. No coalescing, ever.
//
//  The "no coalescing" is the entire tradeoff. It is what makes it O(1) and
//  what makes it fragment. `split_large_blocks` (below) is the usual mitigation
//  and is worth an experiment of its own.
// =============================================================================
#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "mcke/memory/allocator.hpp"

namespace mcke {

struct FreeListConfig {
  std::size_t slab_bytes            = std::size_t{256} << 20;
  std::size_t small_class_granularity = 512;                 // below the split point
  std::size_t small_large_split      = std::size_t{1} << 20; // 1 MiB
  std::size_t large_alloc_threshold  = std::size_t{64} << 20;
  // If true, a request may be served from a larger cached block by splitting it
  // (and the remainder re-cached). Reduces fragmentation, costs O(1) extra
  // bookkeeping, but *creates* small blocks that can never merge back — the
  // classic reason this heuristic is a tunable and not a default.
  bool        split_large_blocks     = false;
};

class FreeListAllocator final : public DeviceAllocator {
 public:
  explicit FreeListAllocator(FreeListConfig cfg = {});
  ~FreeListAllocator() override;

  [[nodiscard]] Status reserve(std::size_t bytes);

  StatusOr<Allocation> allocate(std::size_t bytes, rt::StreamHandle stream) override;
  Status               deallocate(const Allocation& a, rt::StreamHandle stream) override;
  Status               trim() override;
  [[nodiscard]] AllocatorStats  stats() const override { return stats_; }
  [[nodiscard]] std::string_view name() const override { return "freelist"; }
  [[nodiscard]] Status validate() const override;

  // Maps a byte count onto its size-class index. Exposed (and constexpr-ish)
  // because it is the single most test-worthy function in the class: an
  // off-by-one here silently hands out blocks that are too small.
  [[nodiscard]] std::size_t size_class_of(std::size_t bytes) const;
  [[nodiscard]] std::size_t class_block_bytes(std::size_t class_idx) const;

 private:
  struct Slab { void* base = nullptr; std::size_t bytes = 0; std::size_t bump = 0; };
  struct CachedBlock { void* ptr; std::size_t bytes; rt::StreamHandle last_stream; };

  FreeListConfig                          cfg_;
  std::vector<Slab>                       slabs_;
  std::vector<std::vector<CachedBlock>>   classes_;   // indexed by size class
  std::unordered_map<void*, std::size_t>  live_;      // ptr -> bytes, for deallocate
  AllocatorStats                          stats_{};
};

}  // namespace mcke
