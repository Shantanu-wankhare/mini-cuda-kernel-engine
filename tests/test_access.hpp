// =============================================================================
//  tests/test_access.hpp
//
//  WHAT: Friend-access shims that let tests assert on allocator INTERNALS.
//
//  WHY THIS IS A SHARED HEADER rather than a copy in each test file: both
//  tests/test_host_core.cpp (host-only) and tests/test_stream_safety.cu (CUDA)
//  need `pending_count()` to make the same claim -- that a cross-stream allocate
//  REFUSED to reclaim a parked block. Friendship in C++ is granted by name, so
//  each file could redeclare its own copy and it would compile; but then the two
//  copies could drift, and the host test and the GPU test would silently be
//  asserting on different things. One definition, both includers.
//
//  WHY THESE EXIST AT ALL: the most valuable assertions in Phase 2 are about
//  structure no public API exposes -- the exact free-list shape left by one
//  split-down, whether a parked block is still marked kUsed, whether
//  reclaim_completed walked the pending list and declined. Exposing that
//  publicly would be worse (it would become API anyone could depend on), and
//  #ifdef-ing the tested binary away from the shipped one would be worse still.
//  A friend struct is the narrowest tool that works.
//
//  WHY .hpp: pure host C++ with no device code. It is included from a .cu, which
//  is fine -- nvcc compiles the host portion of a .cu with the host compiler.
// =============================================================================
#pragma once

#include <cstddef>

#include "mcke/memory/buddy_allocator.hpp"
#include "mcke/memory/freelist_allocator.hpp"

namespace mcke {
struct BuddyTestAccess {
  using NodeState = BuddyAllocator::NodeState;

  static std::size_t slab_count(const BuddyAllocator& a) { return a.slabs_.size(); }
  static const void* base(const BuddyAllocator& a, std::size_t id) { return a.slabs_[id].base; }
  static std::size_t slab_bytes(const BuddyAllocator& a, std::size_t id) {
    return a.slabs_[id].bytes;
  }
  static unsigned max_level(const BuddyAllocator& a, std::size_t id) {
    return a.slabs_[id].max_level;
  }
  static std::size_t free_list_size(const BuddyAllocator& a, std::size_t id, unsigned level) {
    return a.slabs_[id].free_lists[level].size();
  }
  static NodeState node_state(const BuddyAllocator& a, std::size_t id, std::size_t node) {
    return a.slabs_[id].nodes[node];
  }
  static std::size_t pending_count(const BuddyAllocator& a) { return a.pending_.size(); }

  // --- deliberate corruption, for the negative validate() tests ---
  static void set_node_state(BuddyAllocator& a, std::size_t id, std::size_t node,
                             NodeState st) {
    a.slabs_[id].nodes[node] = st;
  }
  static void clear_mask_bit(BuddyAllocator& a, std::size_t id, unsigned level) {
    a.slabs_[id].nonempty_mask &= ~(1u << level);
  }
};

struct FreeListTestAccess {
  static std::size_t pending_count(const FreeListAllocator& a) { return a.pending_.size(); }
  static std::size_t live_count(const FreeListAllocator& a) { return a.live_.size(); }
  static std::size_t cached_count(const FreeListAllocator& a, std::size_t cls) {
    return a.classes_[cls].size();
  }
  static std::size_t n_small(const FreeListAllocator& a) { return a.n_small_; }
};
}  // namespace mcke
