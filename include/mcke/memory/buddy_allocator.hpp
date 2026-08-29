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
//            ├── vector<uint32>    pos      // node -> index within its free list
//            ├── vector<vector<size_t>>     // free_lists[level] = node indices
//            └── uint32 nonempty_mask       // bit l = free_lists[l] non-empty
//
//  NodeState is 1 byte, an enum: {kDetached, kFree, kSplit, kUsed}. Why not a
//  1-bit bitmap? Because we need four states, and because the classic 2-bit
//  "longest free block below me" encoding is clever but much harder to debug.
//  Choosing the debuggable encoding over the compact one is the right call for a
//  learning project — and we can measure whether it matters before optimising.
//
//  METADATA COST, computed honestly (an earlier version of this comment was 2x
//  low because it counted only the leaf level instead of the whole tree):
//  a 256 MiB slab with 256 B min blocks has K=28, M=8, L=20, so
//      node_count = 2^(L+1) - 1 = 2,097,151 nodes
//      nodes[]      1 B/node  =  2.00 MiB
//      pos[]        4 B/node  =  8.00 MiB
//      free_lists   worst case=  8.00 MiB   (bound derived below)
//                              ----------
//                              18.00 MiB host RAM = 7.0% of the slab
//  All of it host-side; the GPU never sees allocator metadata. 7% is acceptable
//  at 256 MiB, but note the scaling is *linear in slab size*: a 4 GiB slab at the
//  same 256 B granularity would need 288 MiB of host RAM. The fix there is to
//  raise min_block_bytes (4 KiB brings it back to 18 MiB), and add_slab enforces
//  a node-count cap whose error message says exactly that. "1 byte per node is
//  free" is true at one scale, not in general.
//
//  The free-list worst case is derived, not guessed: if node g is kFree then
//  buddy_of(g) is NOT kFree (they would have coalesced), so at most 2^(l-1) nodes
//  are free at level l, summing to <= 2^L free nodes overall. That same fact is
//  the maximal-coalescing invariant validate() checks — one derivation, two uses.
//
//  The per-level free lists hold node indices, so allocate() is:
//     level = level_for_size(bytes)
//     if free_list[level] non-empty -> pop, mark kUsed, return   (O(1))
//     else find the nearest ancestor level with a free block, then split
//          downward, pushing the buddy of each split onto its level's list
//          (O(L), and L <= 24 in practice)
//
//  and deallocate() is:
//     mark kFree; while buddy is kFree: remove buddy from its list,
//                 mark parent kFree, ascend      (O(L))
//
//  ---------------------------------------------------------------------------
//  A CLAIM THAT NEEDS QUALIFYING — read this before believing "O(1) coalescing"
//
//  Two *different* operations get conflated when people say a buddy allocator
//  coalesces in constant time:
//
//    1. IDENTIFYING the buddy: buddy_of(g) = ((g-1) ^ 1) + 1. Genuinely O(1),
//       no search, no lookup. This is the real selling point.
//    2. REMOVING that buddy from its free list. NOT inherently O(1) — the buddy
//       sits at an arbitrary position in the middle of the list.
//
//  With a plain vector and a linear scan, (2) is O(list length), and the lists
//  get long: level 20 of a 256 MiB slab can hold 524,288 entries, and that state
//  is genuinely reachable (allocate every leaf, then free every other one). So a
//  single deallocate could scan half a million entries, and the "O(log n) free"
//  claim would be false.
//
//  That is why Slab carries `pos` — pos[g] is g's index within its own free
//  list, so removal is swap-with-back + pop_back, O(1). It costs 4 B/node
//  (8 MiB for a 256 MiB slab) and it is what makes the complexity claim true.
//
//  Swap-and-pop reorders the list. That is legal *here* because any free block at
//  a given level is interchangeable — order carries no information. In an
//  address-ordered first-fit allocator the same trick would be a bug. Knowing why
//  it is safe in this design and not in that one is the actual lesson.
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
#include "mcke/memory/reuse_policy.hpp"

namespace mcke {

// The three cross-stream reuse policies, the rule-1 foundation they share, and
// the single decision function both pooling allocators route through, all live in
// mcke/memory/reuse_policy.hpp — physically shared so that the Phase 2c
// buddy-vs-freelist comparison cannot be contaminated by the two pools
// disagreeing about what "safe to reuse" means.

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

  // Default: coarse polling, matching what allocator.hpp documents as the Phase 2
  // mechanism. Phase 2c benchmarks all three and RESULTS.md records the winner.
  ReusePolicy reuse_policy = ReusePolicy::kCoarseStreamPoll;
};

// NOTE: deliberately NOT `final`, unlike the other allocators.
//
// A unit test needs to drive the "this block's stream has not finished yet"
// branch, and in a host-only build that branch is unreachable — rt::stream_query
// is unconditionally true with no GPU. The seam is a protected virtual
// (`stream_completed`) that a test-only subclass overrides, which is why the
// class must be derivable.
//
// The cost of dropping `final` is that the compiler can no longer devirtualise a
// call made through a BuddyAllocator*. In this codebase that costs nothing
// measurable, because every production call site goes through DeviceAllocator&
// on purpose — runtime polymorphism is the whole mechanism that lets one
// benchmark process A/B three allocators against one identical trace. Nothing
// was being devirtualised to begin with.
class BuddyAllocator : public DeviceAllocator {
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
  // Release entirely-unused slabs back to the driver.
  //
  // CRITICAL: this must never erase from slabs_. Erasing shifts the indices of
  // every later slab, which silently invalidates the slab_id stored in every
  // outstanding Allocation the caller still holds — the next deallocate() would
  // then operate on the wrong slab. Instead a released slab becomes a dead slot
  // (base == nullptr) and add_slab reuses it, so slab ids are stable for the
  // lifetime of the allocator.
  Status               trim() override;
  Status               settle_pending() override;
  // NOT inline: largest_free_block is computed on demand rather than maintained
  // incrementally, so this needs a body. See compute_largest_free_block().
  [[nodiscard]] AllocatorStats  stats() const override;
  [[nodiscard]] std::string_view name() const override { return "buddy"; }
  [[nodiscard]] Status validate() const override;

  // Human-readable dump of the tree, for the Phase 2 fragmentation write-up.
  [[nodiscard]] std::string dump_free_map() const;

 protected:
  // THE TEST SEAM. "Has all work enqueued on `stream` completed?"
  //
  // Production answer is rt::stream_query. A test-only subclass overrides this to
  // simulate in-flight GPU work, which is the only way to reach the
  // defer-and-park branch on a machine with no GPU (where stream_query is
  // unconditionally true and every free would otherwise be immediate).
  //
  // Called at most once per parked block per allocate() — never on the fast path,
  // because a single-stream workload keeps pending_ empty entirely.
  [[nodiscard]] virtual bool stream_completed(rt::StreamHandle stream) const {
    return rt::stream_query(stream);
  }

  // The same seam for kPerFreeEvent. Takes a POOL SLOT rather than an
  // rt::EventHandle deliberately: in a host-only build every event handle is
  // nullptr, so a handle-based seam could not distinguish one parked block from
  // another and the test could only flip completion globally. Slots are distinct
  // integers even with no GPU, so a test can hold block A in flight while
  // releasing block B — which is the property that actually needs covering.
  [[nodiscard]] virtual bool event_completed(std::uint32_t event_slot) const;

 private:
  // kDetached must be 0 so that a zero-initialised nodes vector means "not part
  // of the tree", NOT "free". With kFree = 0 (the original declaration) a fresh
  // 2-million-node tree would read as entirely free — a bug generator.
  //
  // kDetached also carries real diagnostic value: after two children merge into
  // their parent, the children are unreachable and their state is semantically
  // don't-care. Setting them to kDetached makes that canonical, so validate() can
  // assert the strong invariant "every unreachable node is kDetached" instead of
  // silently skipping them.
  enum class NodeState : std::uint8_t { kDetached = 0, kFree, kSplit, kUsed };

  static constexpr std::uint32_t kNotInList = 0xFFFFFFFFu;

  // Cap on nodes per slab, so host metadata cannot silently explode (see the
  // scaling table in the banner). 2^26 nodes is ~64 MiB of state on its own; at a
  // 256 B min block that caps a slab at 8 GiB. Also keeps every node index inside
  // uint32_t, which is what makes Allocation::block_id safe.
  static constexpr std::size_t kMaxNodesPerSlab = std::size_t{1} << 26;

  struct Slab {
    void*                  base = nullptr;   // nullptr => dead slot, see trim()
    std::size_t            bytes = 0;
    unsigned               arena_log2 = 0;
    unsigned               min_log2 = 0;
    unsigned               max_level = 0;
    std::vector<NodeState> nodes;                     // heap order, see buddy_math.hpp
    std::vector<std::uint32_t> pos;                   // node -> index in its free list
    std::vector<std::vector<std::size_t>> free_lists; // free_lists[level] = node indices
    // Bit l set iff free_lists[l] is non-empty. Turns "find the deepest level
    // with a free block" into one countl_zero and largest_free_block into one
    // countr_zero. L <= 31 so a uint32 is exactly wide enough.
    //
    // This duplicates state derivable from free_lists, which is normally a smell.
    // It is safe here only because validate() cross-checks every bit against the
    // corresponding list — the duplication is tested, not assumed.
    std::uint32_t          nonempty_mask = 0;
  };

  // A block that has been logically freed but whose last GPU work may still be
  // running. See the stream-ordered discussion in allocator.hpp.
  //
  // The node stays kUsed while parked. That is the entire safety mechanism: a
  // parked block must not coalesce with a free buddy, or the merged block could
  // be handed to a different stream while the original consumer is still reading
  // it — precisely the race this design exists to prevent.
  struct PendingFree {
    std::uint32_t     slab_id = 0;
    std::size_t       node    = 0;
    rt::StreamHandle  stream  = {};
    // Index into event_pool_, or kNoEventSlot for the policies that do not use one.
    // An INDEX rather than an rt::Event by value: rt::Event is move-only, and a
    // move-only member would forbid the cheap swap-and-compact that keeps
    // reclaim_completed a single O(|pending|) pass.
    std::uint32_t     event_slot = kNoEventSlot;
  };

  [[nodiscard]] StatusOr<std::uint32_t> add_slab(std::size_t bytes,
                                                std::size_t min_usable_bytes);
  [[nodiscard]] StatusOr<std::size_t>   alloc_node(Slab& s, unsigned level);
  void                                  free_node(Slab& s, std::size_t node);

  // The ONLY three functions permitted to touch free_lists / pos / nonempty_mask.
  // Keeping the four-way invariant between those three fields and `nodes` inside
  // three short functions means there is exactly one place to review it and
  // exactly one place for a bug to hide.
  void                    push_free(Slab& s, unsigned level, std::size_t node);
  [[nodiscard]] std::size_t pop_free(Slab& s, unsigned level);
  void                    remove_free(Slab& s, unsigned level, std::size_t node);

  // Drain pending frees that are now safe to reuse. Called at the top of
  // allocate() — a cheap opportunistic reclaim that avoids any host stall.
  void reclaim_completed(rt::StreamHandle stream);
  // Last resort before reporting OOM: wait for parked blocks instead of lying
  // about being out of memory while holding reclaimable capacity.
  void drain_pending_blocking();

  // Populates an Allocation for a just-claimed node and updates the in-use
  // watermarks. Factored out because allocate() reaches it from three different
  // places (existing slab / freshly grown slab / after a blocking drain) and the
  // stats bookkeeping must be identical in all three.
  [[nodiscard]] Allocation make_allocation(std::uint32_t slab_id, const Slab& s,
                                           std::size_t node, std::size_t requested);

  [[nodiscard]] StatusOr<Allocation> allocate_bypass(std::size_t bytes);
  [[nodiscard]] Status               deallocate_bypass(const Allocation& a);
  [[nodiscard]] std::size_t          next_slab_bytes() const;
  [[nodiscard]] std::size_t          compute_largest_free_block() const;
  [[nodiscard]] std::string          describe_node(const Slab& s, std::size_t node) const;
  [[nodiscard]] Status               validate_slab(const Slab& s, std::uint32_t id) const;

  // Event slots are pooled, never created per free: cudaEventCreate is far more
  // expensive than cudaEventRecord, and creating one on the hot path would make
  // kPerFreeEvent lose the benchmark for the wrong reason.
  [[nodiscard]] StatusOr<std::uint32_t> acquire_event_slot();
  void                                  release_event_slot(std::uint32_t slot);

  BuddyConfig               cfg_;
  std::vector<Slab>         slabs_;
  std::vector<PendingFree>  pending_;
  std::vector<rt::Event>    event_pool_;         // only populated for kPerFreeEvent
  std::vector<std::uint32_t> free_event_slots_;  // LIFO of available pool indices
  // Large allocations that bypassed the pool; we must remember which pointers
  // those were so deallocate() routes them back to cudaFree.
  std::unordered_map<void*, std::size_t> bypassed_;
  AllocatorStats            stats_{};

  // Tests need to assert free-list *shape* — the exact structural signature left
  // by one split-down, and that a full free coalesces back to only the root being
  // free. No public API can expose that, and it is precisely the state most worth
  // asserting. One friend declaration beats either making internals public or
  // #ifdef-ing the tested binary away from the shipped one.
  friend struct BuddyTestAccess;
};

}  // namespace mcke
