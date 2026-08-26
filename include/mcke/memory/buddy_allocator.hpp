// =============================================================================
//  mcke/memory/buddy_allocator.hpp
//
//  WHAT: A binary-buddy suballocator over one or more large cudaMalloc'd slabs,
//        with stream-ordered reuse. Implementation lands in Phase 2a
//        (src/memory/buddy_allocator.cpp); this header is the contract.
//
//  ---------------------------------------------------------------------------
//  STRUCTURE
//
//    BuddyAllocator
//      └── vector<Slab>                     // each slab = one cudaMalloc
//            ├── void*  base                // 2^K bytes, 256B-aligned
//            ├── vector<NodeState> nodes    // 2^(L+1)-1 entries, heap order
//            └── array<free-list, L+1>      // intrusive index lists per level
//
//  NodeState is 1 byte, an enum: {kFree, kSplit, kUsed}. Why not a 1-bit
//  bitmap? Because we need three states, and because the classic 2-bit
//  "longest free block below me" encoding is clever but much harder to debug.
//  1 byte/node for a 256 MiB slab with 256 B min blocks = 2^20 nodes = 1 MiB of
//  *host* metadata. That is free. Choosing the debuggable encoding over the
//  compact one is the right call for a learning project — and we can measure
//  whether it matters before optimising it.
//
//  The per-level free lists hold node indices, so allocate() is:
//     level = level_for_size(bytes)
//     if free_list[level] non-empty -> pop, mark kUsed, return   (O(1))
//     else find the nearest ancestor level with a free block, then split
//          downward, pushing the buddy of each split onto its level's list
//          (O(log n), and log n <= 22 in practice)
//
//  and deallocate() is:
//     mark kFree; while buddy is kFree: remove buddy from its list,
//                 mark parent kFree, ascend      (O(log n) amortised)
//
//  ---------------------------------------------------------------------------
//  WHY MULTIPLE SLABS instead of one giant one
//   - You cannot always get one contiguous 20 GiB block, especially on a shared
//     HPC node or a Colab instance where another process holds memory.
//   - Growing by doubling (256 MiB, 512 MiB, ...) keeps `raw_malloc_calls`
//     logarithmic in total footprint, which is the metric we care about.
//   - Requests larger than the largest slab bypass the pool entirely and go to
//     RawDeviceAllocator. Trying to serve a 4 GiB weight tensor out of a buddy
//     tree just rounds it up to 8 GiB — pure waste. Big, long-lived, one-off
//     allocations are exactly the case where cudaMalloc's cost is amortised
//     anyway.
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mcke/memory/allocator.hpp"
#include "mcke/memory/buddy_math.hpp"

namespace mcke {

struct BuddyConfig {
  // Size of the first slab. 256 MiB is a deliberate default: large enough that
  // a Phase-3 GEMM benchmark (a few 4096^2 f32 matrices = 64 MiB each) fits
  // without growth, small enough to not fail on a shared 16 GB Colab T4.
  std::size_t initial_slab_bytes = std::size_t{256} << 20;
  std::size_t max_total_bytes    = std::size_t{0};   // 0 = unlimited (bounded by VRAM)
  std::size_t min_block_bytes    = kMinBlockBytes;   // 256 B
  double      growth_factor      = 2.0;

  // Requests above this go straight to cudaMalloc rather than into a slab.
  // Rationale in the header comment above.
  std::size_t large_alloc_threshold = std::size_t{64} << 20;
};

class BuddyAllocator final : public DeviceAllocator {
 public:
  explicit BuddyAllocator(BuddyConfig cfg = {});
  ~BuddyAllocator() override;

  // Reserve the first slab eagerly. Kept out of the constructor because it can
  // fail (OOM) and constructors cannot return Status — the standard two-phase
  // init pattern, used here rather than exceptions because "the GPU is too full
  // for a 256 MiB slab" is a condition a caller may reasonably retry smaller.
  [[nodiscard]] Status reserve(std::size_t bytes);

  StatusOr<Allocation> allocate(std::size_t bytes, rt::StreamHandle stream) override;
  Status               deallocate(const Allocation& a, rt::StreamHandle stream) override;
  Status               trim() override;
  [[nodiscard]] AllocatorStats  stats() const override { return stats_; }
  [[nodiscard]] std::string_view name() const override { return "buddy"; }
  [[nodiscard]] Status validate() const override;

  // Human-readable dump of the tree, for the Phase 2 fragmentation write-up.
  [[nodiscard]] std::string dump_free_map() const;

 private:
  enum class NodeState : std::uint8_t { kFree = 0, kSplit, kUsed };

  struct Slab {
    void*                  base = nullptr;
    std::size_t            bytes = 0;
    unsigned               arena_log2 = 0;
    unsigned               min_log2 = 0;
    unsigned               max_level = 0;
    std::vector<NodeState> nodes;                     // heap order, see buddy_math.hpp
    std::vector<std::vector<std::size_t>> free_lists; // free_lists[level] = node indices
  };

  // A block that has been logically freed but whose last GPU work may still be
  // running. See the stream-ordered discussion in allocator.hpp.
  struct PendingFree {
    std::uint32_t     slab_id;
    std::size_t       node;
    rt::StreamHandle  stream;
  };

  [[nodiscard]] StatusOr<std::uint32_t> add_slab(std::size_t bytes);
  [[nodiscard]] StatusOr<std::size_t>   alloc_node(Slab& s, unsigned level);
  void                                  free_node(Slab& s, std::size_t node);

  // Drain pending frees whose stream has completed (rule 2b). Called at the top
  // of allocate() — a cheap opportunistic reclaim that avoids any host stall.
  void reclaim_completed(rt::StreamHandle stream);

  BuddyConfig               cfg_;
  std::vector<Slab>         slabs_;
  std::vector<PendingFree>  pending_;
  // Large allocations that bypassed the pool; we must remember which pointers
  // those were so deallocate() routes them back to cudaFree.
  std::unordered_map<void*, std::size_t> bypassed_;
  AllocatorStats            stats_{};
};

}  // namespace mcke
