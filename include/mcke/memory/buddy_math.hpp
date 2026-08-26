// =============================================================================
//  mcke/memory/buddy_math.hpp
//
//  WHAT: The pure index arithmetic of a binary buddy allocator, as `constexpr`
//        free functions with no state and no I/O.
//
//  WHY this file exists separately from buddy_allocator.hpp:
//   1. It is the part that is easy to get wrong (off-by-one on levels, XOR
//      trick on buddies) and easy to unit-test exhaustively. Isolating it means
//      we can test it on the MacBook with no GPU and no allocator at all.
//   2. It is `constexpr`, so the tests run at compile time as well as runtime —
//      a static_assert that fails is a build error, not a red test.
//
//  ---------------------------------------------------------------------------
//  THE DATA STRUCTURE, IN WORDS
//
//  An arena of 2^K bytes is treated as a complete binary tree.
//    level 0            : one node,  2^K bytes        (the whole arena)
//    level 1            : two nodes, 2^(K-1) bytes each
//    ...
//    level L = K - M    : 2^L nodes, 2^M bytes each    (M = log2 of min block)
//
//  Nodes are numbered in *heap order* (breadth-first, root = 0), the same
//  numbering a binary heap in an array uses:
//        g(level, i) = 2^level - 1 + i
//  which gives the classic identities
//        parent(g)      = (g - 1) / 2
//        left_child(g)  = 2g + 1
//        buddy(g)       = ((g - 1) XOR 1) + 1
//
//  WHY heap order rather than a per-level array of free-lists with pointers:
//  heap order lets the entire allocator state be a flat bitset/byte array of
//  size 2^(L+1) - 1 with *no pointers at all*. For a 1 GiB arena with 256 B
//  min blocks that is L = 22, so ~8.4M nodes = 8.4 MB of host-side metadata if
//  1 byte/node, or ~1 MB as 2 bits/node. Crucially none of it lives in device
//  memory: the GPU never sees allocator metadata, which is exactly what we
//  want since device memory is the scarce resource.
//
//  WHY the XOR trick for buddies: two nodes are buddies iff they share a parent.
//  In heap order, converting to 1-based (g+1) makes siblings differ in exactly
//  the low bit, so flipping that bit swaps them. Shifting back to 0-based gives
//  ((g-1) ^ 1) + 1. This is O(1) with no division and no lookup — the reason
//  buddy allocators can coalesce in constant time, which is their whole selling
//  point over a general free-list.
//
//  ---------------------------------------------------------------------------
//  TRADEOFF vs. the alternatives (read this before the code):
//
//  * Buddy allocator
//      + O(log n) alloc, O(log n) free with *guaranteed* coalescing.
//      + Bounded external fragmentation; blocks always merge back into the
//        largest possible free block, so long-running graphs don't degrade.
//      - Up to 2x internal fragmentation: a 33 KiB request takes a 64 KiB
//        block. For DL tensor shapes (often not powers of two: 768, 3072,
//        50257) this is real and measurable — we will measure it.
//  * Segregated free-list / size-class pool (what PyTorch's caching allocator
//    is, roughly)
//      + Near-zero internal fragmentation if the size classes match the
//        workload; O(1) alloc/free.
//      - Coalescing is hard/absent, so external fragmentation grows: you can
//        end up with 2 GiB free but no contiguous 512 MiB. Needs a
//        defrag/release-cached-blocks escape hatch.
//  * Bump / arena allocator
//      + Trivially fast (pointer increment), perfect for a *single graph
//        execution* where every intermediate dies at the end.
//      - Cannot free individually at all.
//
//  Our plan: implement buddy first (Phase 2a) because it teaches the most and
//  handles the general case; implement the size-class pool (Phase 2b); then
//  benchmark both against raw cudaMalloc and against each other on the same
//  allocation trace. That comparison *is* the deliverable, not just the code.
// =============================================================================
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include "mcke/core/config.hpp"

namespace mcke::buddy {

// These are host-side (the allocator runs entirely on the CPU), so plain
// constexpr — not MCKE_HOST_DEVICE. std::bit_width & friends are constexpr but
// not __device__-callable without relaxed-constexpr, and there is no reason to
// call them from a kernel.

[[nodiscard]] constexpr bool is_power_of_two(std::size_t x) noexcept {
  return x != 0 && (x & (x - 1)) == 0;
}

// ceil(log2(x)) for x >= 1. std::bit_width(x) == floor(log2(x)) + 1.
[[nodiscard]] constexpr unsigned ceil_log2(std::size_t x) noexcept {
  if (x <= 1) return 0;
  return static_cast<unsigned>(std::bit_width(x - 1));
}

[[nodiscard]] constexpr unsigned floor_log2(std::size_t x) noexcept {
  return x == 0 ? 0 : static_cast<unsigned>(std::bit_width(x) - 1);
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t x, std::size_t a) noexcept {
  // `a` must be a power of two; the mask form avoids a division. Note the
  // overflow characteristic: x close to SIZE_MAX wraps. Callers validate sizes
  // against arena capacity first, so this is unreachable in practice.
  return (x + a - 1) & ~(a - 1);
}

// ---------------------------------------------------------------------------
// Tree geometry
// ---------------------------------------------------------------------------

// Number of levels *below* the root, i.e. the deepest level index.
// arena = 2^arena_log2 bytes, min block = 2^min_log2 bytes.
[[nodiscard]] constexpr unsigned max_level(unsigned arena_log2, unsigned min_log2) noexcept {
  return arena_log2 - min_log2;
}

// Total node count for a tree with levels [0, max_level]: 2^(L+1) - 1.
[[nodiscard]] constexpr std::size_t node_count(unsigned max_lvl) noexcept {
  return (std::size_t{1} << (max_lvl + 1)) - 1;
}

[[nodiscard]] constexpr std::size_t block_bytes_at_level(unsigned arena_log2,
                                                        unsigned level) noexcept {
  return std::size_t{1} << (arena_log2 - level);
}

[[nodiscard]] constexpr std::size_t nodes_at_level(unsigned level) noexcept {
  return std::size_t{1} << level;
}

// Deepest (i.e. smallest-block) level whose block still fits `bytes`.
// Returns max_level+1 as a sentinel meaning "request too large for this arena".
//
// Worked example, arena_log2 = 20 (1 MiB), min_log2 = 8 (256 B):
//   bytes = 1        -> rounded to 256   -> level 12 (=20-8)
//   bytes = 300      -> rounded to 512   -> level 11
//   bytes = 1 MiB    -> level 0
//   bytes = 1 MiB+1  -> sentinel 13
[[nodiscard]] constexpr unsigned level_for_size(std::size_t bytes, unsigned arena_log2,
                                                unsigned min_log2) noexcept {
  const unsigned need_log2 = ceil_log2(bytes < (std::size_t{1} << min_log2)
                                           ? (std::size_t{1} << min_log2)
                                           : bytes);
  if (need_log2 > arena_log2) return max_level(arena_log2, min_log2) + 1;  // too big
  return arena_log2 - need_log2;
}

// ---------------------------------------------------------------------------
// Heap-order node indexing
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::size_t heap_index(unsigned level, std::size_t idx_in_level) noexcept {
  return (std::size_t{1} << level) - 1 + idx_in_level;
}

[[nodiscard]] constexpr unsigned level_of(std::size_t heap_idx) noexcept {
  // level = floor(log2(heap_idx + 1))
  return static_cast<unsigned>(std::bit_width(heap_idx + 1) - 1);
}

[[nodiscard]] constexpr std::size_t index_in_level(std::size_t heap_idx) noexcept {
  const unsigned l = level_of(heap_idx);
  return heap_idx - ((std::size_t{1} << l) - 1);
}

[[nodiscard]] constexpr bool is_root(std::size_t heap_idx) noexcept { return heap_idx == 0; }

[[nodiscard]] constexpr std::size_t parent_of(std::size_t heap_idx) noexcept {
  return (heap_idx - 1) / 2;      // undefined for the root; callers check is_root
}

[[nodiscard]] constexpr std::size_t left_child_of(std::size_t heap_idx) noexcept {
  return 2 * heap_idx + 1;
}
[[nodiscard]] constexpr std::size_t right_child_of(std::size_t heap_idx) noexcept {
  return 2 * heap_idx + 2;
}

// The O(1) coalescing primitive. See header comment for the derivation.
[[nodiscard]] constexpr std::size_t buddy_of(std::size_t heap_idx) noexcept {
  return ((heap_idx - 1) ^ std::size_t{1}) + 1;  // undefined for the root
}

// Byte offset of a node's block from the base of the arena.
[[nodiscard]] constexpr std::size_t offset_of(std::size_t heap_idx, unsigned arena_log2) noexcept {
  const unsigned l = level_of(heap_idx);
  return index_in_level(heap_idx) * block_bytes_at_level(arena_log2, l);
}

// Inverse mapping: which node owns this offset at this level?
[[nodiscard]] constexpr std::size_t node_at(std::size_t offset, unsigned level,
                                            unsigned arena_log2) noexcept {
  return heap_index(level, offset / block_bytes_at_level(arena_log2, level));
}

// ---------------------------------------------------------------------------
// Compile-time self-checks. These cost nothing at runtime and catch a whole
// class of refactoring mistake at build time — the cheapest test there is.
// ---------------------------------------------------------------------------
static_assert(ceil_log2(1) == 0);
static_assert(ceil_log2(2) == 1);
static_assert(ceil_log2(3) == 2);
static_assert(ceil_log2(256) == 8);
static_assert(ceil_log2(257) == 9);
static_assert(heap_index(0, 0) == 0);
static_assert(heap_index(1, 0) == 1 && heap_index(1, 1) == 2);
static_assert(heap_index(2, 3) == 6);
static_assert(level_of(0) == 0 && level_of(1) == 1 && level_of(2) == 1 && level_of(6) == 2);
static_assert(buddy_of(1) == 2 && buddy_of(2) == 1);
static_assert(buddy_of(5) == 6 && buddy_of(6) == 5);
static_assert(parent_of(buddy_of(5)) == parent_of(5));
static_assert(offset_of(2, 20) == (std::size_t{1} << 19));   // right half of a 1 MiB arena
static_assert(level_for_size(1, 20, 8) == 12);
static_assert(level_for_size(300, 20, 8) == 11);
static_assert(level_for_size(std::size_t{1} << 20, 20, 8) == 0);
static_assert(level_for_size((std::size_t{1} << 20) + 1, 20, 8) == 13);  // sentinel

}  // namespace mcke::buddy
