// =============================================================================
//  src/memory/buddy_allocator.cpp
//
//  WHY .cpp: no device code. Every line here is host-side integer arithmetic and
//  vector bookkeeping; the only device interaction is raw_device_malloc /
//  raw_device_free, which are ordinary C calls into libcudart. This is what lets
//  the entire allocator — including its OOM paths, its coalescing logic, and
//  (via the stream_completed seam) its stream-ordered deferral — be compiled and
//  exhaustively unit-tested on a MacBook with no GPU at all.
//
//  ---------------------------------------------------------------------------
//  ORIENTATION: the one thing that will confuse you in this file
//
//  Level 0 is the ROOT and the LARGEST block (the whole arena). Higher level
//  index = SMALLER block. So:
//      "search toward the root"  = decreasing level index = larger blocks
//      "split downward"          = increasing level index = smaller blocks
//  Every loop below is annotated with which direction it goes, because getting
//  this backwards produces code that looks right and allocates the wrong size.
//
//  See include/mcke/memory/buddy_math.hpp for the index arithmetic (all constexpr
//  and already exhaustively tested) and buddy_allocator.hpp for the data-structure
//  rationale, the honest metadata cost, and why `pos` exists.
// =============================================================================
#include "mcke/memory/buddy_allocator.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdio>
#include <sstream>

#include "mcke/core/device.hpp"

namespace mcke {

namespace {

// Format a byte count for human-readable diagnostics.
std::string human_bytes(std::size_t b) {
  std::ostringstream os;
  if (b >= (std::size_t{1} << 30))      os << (b >> 30) << " GiB";
  else if (b >= (std::size_t{1} << 20)) os << (b >> 20) << " MiB";
  else if (b >= (std::size_t{1} << 10)) os << (b >> 10) << " KiB";
  else                                  os << b << " B";
  return os.str();
}

}  // namespace

// -----------------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------------

BuddyAllocator::BuddyAllocator(BuddyConfig cfg) : cfg_(cfg) {}

BuddyAllocator::~BuddyAllocator() {
  // A destructor cannot return Status, so failures are swallowed — but a leak is
  // worth complaining about, because it means a *caller* forgot a deallocate.
  // That is exactly the bug the Phase 4 graph executor will introduce.
  //
  // A warning rather than an assert on purpose: tearing down an allocator with
  // live blocks is legitimate (process exit, an aborted graph, a test that only
  // cares about one code path), so this must not be fatal. It is a diagnostic,
  // not an invariant.
  if (stats_.bytes_in_use != 0) {
    std::fprintf(stderr,
                 "[mcke] warning: BuddyAllocator destroyed with %zu bytes still "
                 "live across %llu allocate / %llu free calls — a caller leaked\n",
                 stats_.bytes_in_use,
                 static_cast<unsigned long long>(stats_.alloc_calls),
                 static_cast<unsigned long long>(stats_.free_calls));
  }

  for (Slab& s : slabs_) {
    if (s.base != nullptr) {
      // cudaFree synchronises, so releasing a slab with GPU work still in flight
      // is safe here even though it would not be safe to *reuse* it.
      (void)raw_device_free(s.base);
      s.base = nullptr;
    }
  }
  for (const auto& [ptr, bytes] : bypassed_) {
    (void)bytes;
    (void)raw_device_free(ptr);
  }
}

// -----------------------------------------------------------------------------
// Free-list primitives — the ONLY code allowed to touch free_lists/pos/mask.
//
// The invariant these three maintain, and that validate() checks:
//    nodes[g] == kFree                      <=>  pos[g] != kNotInList
//                                           and  free_lists[level_of(g)][pos[g]] == g
//    ((nonempty_mask >> l) & 1)             <=>  !free_lists[l].empty()
// -----------------------------------------------------------------------------

void BuddyAllocator::push_free(Slab& s, unsigned level, std::size_t node) {
  s.pos[node] = static_cast<std::uint32_t>(s.free_lists[level].size());
  s.free_lists[level].push_back(node);
  s.nonempty_mask |= (1u << level);
}

std::size_t BuddyAllocator::pop_free(Slab& s, unsigned level) {
  // Pop from the BACK: O(1), and no other free block at this level is affected.
  const std::size_t node = s.free_lists[level].back();
  s.free_lists[level].pop_back();
  s.pos[node] = kNotInList;
  if (s.free_lists[level].empty()) s.nonempty_mask &= ~(1u << level);
  return node;
}

void BuddyAllocator::remove_free(Slab& s, unsigned level, std::size_t node) {
  // Swap-with-back then pop. This is the O(1) removal that makes the "O(log n)
  // deallocate" claim true — a linear scan here would be O(list length), and
  // level L of a big slab can hold hundreds of thousands of entries.
  //
  // Reordering the list is legal because any free block at a given level is
  // interchangeable. In an address-ordered allocator this would be a bug.
  const std::uint32_t i = s.pos[node];
  assert(i != kNotInList && "remove_free on a node that is not in a free list");
  const std::size_t last = s.free_lists[level].back();
  s.free_lists[level][i] = last;
  s.pos[last] = i;                       // no-op when node *was* the back element
  s.free_lists[level].pop_back();
  s.pos[node] = kNotInList;
  if (s.free_lists[level].empty()) s.nonempty_mask &= ~(1u << level);
}

// -----------------------------------------------------------------------------
// Slab creation
// -----------------------------------------------------------------------------

std::size_t BuddyAllocator::next_slab_bytes() const {
  std::size_t largest = 0;
  for (const Slab& s : slabs_)
    if (s.base != nullptr) largest = std::max(largest, s.bytes);
  if (largest == 0) return cfg_.initial_slab_bytes;
  const double grown = static_cast<double>(largest) * cfg_.growth_factor;
  return static_cast<std::size_t>(grown);
}

StatusOr<std::uint32_t> BuddyAllocator::add_slab(std::size_t want_bytes,
                                                std::size_t min_usable_bytes) {
  // Config geometry is validated HERE — the one place it can actually matter —
  // rather than in the constructor, so a bad config surfaces as a Status the
  // caller can read instead of a constructor that cannot report anything.
  if (!buddy::is_power_of_two(cfg_.min_block_bytes))
    return InvalidArgumentError("buddy: min_block_bytes must be a power of two, got " +
                                std::to_string(cfg_.min_block_bytes));
  if (cfg_.min_block_bytes < kDeviceAlignment)
    // REJECTED, not clamped. Clamping would hide a config bug whose only symptom
    // is silently misaligned tensors and uncoalesced loads three phases later.
    return InvalidArgumentError("buddy: min_block_bytes (" +
                                std::to_string(cfg_.min_block_bytes) +
                                ") must be >= kDeviceAlignment (" +
                                std::to_string(kDeviceAlignment) + ")");
  if (cfg_.growth_factor <= 1.0)
    return InvalidArgumentError("buddy: growth_factor must be > 1.0, got " +
                                std::to_string(cfg_.growth_factor));

  const unsigned M = buddy::ceil_log2(cfg_.min_block_bytes);

  // Round the slab size UP to a power of two.
  //
  // Alternatives were round-down and reject-non-power-of-two. Round-up wins
  // because it keeps slab.bytes == 2^arena_log2 EXACTLY, so the tree tiles the
  // whole allocation with no dead tail: offset_of(node) always lands inside
  // [base, base+bytes), and bytes_reserved means one unambiguous thing. Carving
  // the largest power of two inside a non-power-of-two malloc would leave bytes
  // we own but can never hand out, and then "reserved" and "arena capacity"
  // become two different numbers that both want to be called the footprint.
  //
  // The risk round-up introduces — asking the driver for up to 2x what was
  // requested — is exactly what the halving retry below absorbs. That is safe
  // because a failed cudaMalloc does NOT poison the context (see
  // src/core/device.cpp), which is what makes "try large, fall back" viable.
  const unsigned k_want = std::max(buddy::ceil_log2(std::max(want_bytes, cfg_.min_block_bytes)), M);
  const unsigned k_floor = std::max(buddy::ceil_log2(std::max(min_usable_bytes, cfg_.min_block_bytes)), M);
  if (k_floor > k_want)
    return InvalidArgumentError("buddy: add_slab min_usable exceeds want");

  std::size_t reserved_slabs = 0;
  for (const Slab& s : slabs_)
    if (s.base != nullptr) reserved_slabs += s.bytes;

  std::string why_refused;
  for (unsigned k = k_want; k + 1 > k_floor; --k) {   // descending, k >= k_floor
    const std::size_t slab_bytes = std::size_t{1} << k;
    const unsigned    max_lvl    = buddy::max_level(k, M);

    if (cfg_.max_total_bytes != 0 && reserved_slabs + slab_bytes > cfg_.max_total_bytes) {
      why_refused = "max_total_bytes cap (" + human_bytes(cfg_.max_total_bytes) + ")";
      continue;
    }
    // L <= 31 keeps every heap index inside uint32_t, which is what makes
    // Allocation::block_id safe. Host metadata blows up long before this bites,
    // but the assert-by-construction is free.
    if (max_lvl > 31) {
      why_refused = "arena too deep (L > 31)";
      continue;
    }
    if (buddy::node_count(max_lvl) > kMaxNodesPerSlab) {
      why_refused = "node count " + std::to_string(buddy::node_count(max_lvl)) +
                    " exceeds cap; raise min_block_bytes";
      continue;
    }

    auto p = raw_device_malloc(slab_bytes);
    if (!p.ok()) {
      if (p.status().code() != StatusCode::kOutOfMemory)
        return p.status();          // a real driver error: do not retry, propagate
      why_refused = "driver OOM at " + human_bytes(slab_bytes);
      continue;                     // halve and retry
    }
    ++stats_.raw_malloc_calls;

    // Reuse a dead slot if trim() left one, so slab ids stay stable forever.
    std::uint32_t id = static_cast<std::uint32_t>(slabs_.size());
    for (std::size_t i = 0; i < slabs_.size(); ++i) {
      if (slabs_[i].base == nullptr) { id = static_cast<std::uint32_t>(i); break; }
    }
    if (id == slabs_.size()) slabs_.emplace_back();
    assert(slabs_.size() < kBypassSlabId && "slab count collides with the bypass sentinel");

    Slab& s      = slabs_[id];
    s.base       = *p;
    s.bytes      = slab_bytes;      // == 2^k exactly. No tail.
    s.arena_log2 = k;
    s.min_log2   = M;
    s.max_level  = max_lvl;
    const std::size_t n = buddy::node_count(max_lvl);
    s.nodes.assign(n, NodeState::kDetached);
    s.pos.assign(n, kNotInList);
    s.free_lists.assign(max_lvl + 1, {});
    s.nonempty_mask = 0;
    s.nodes[0] = NodeState::kFree;   // the root is the one free block
    push_free(s, 0, 0);

    stats_.bytes_reserved += slab_bytes;
    return id;
  }

  return OutOfMemoryError("buddy: could not reserve a slab for " +
                          human_bytes(want_bytes) + " (min usable " +
                          human_bytes(min_usable_bytes) + "): " + why_refused);
}

Status BuddyAllocator::reserve(std::size_t bytes) {
  if (bytes == 0) return InvalidArgumentError("buddy: reserve(0)");
  // Idempotent-ish: if we already hold a slab big enough, do nothing rather than
  // doubling our footprint on a redundant call.
  for (const Slab& s : slabs_)
    if (s.base != nullptr && s.bytes >= bytes) return OkStatus();
  auto id = add_slab(bytes, bytes);
  if (!id.ok()) return id.status();
  return OkStatus();
}

// -----------------------------------------------------------------------------
// alloc_node — split downward
// -----------------------------------------------------------------------------

StatusOr<std::size_t> BuddyAllocator::alloc_node(Slab& s, unsigned level) {
  // (a) Find the source level: the DEEPEST level <= `level` that has a free
  //     block. Deepest = largest index = smallest block, because taking a
  //     shallower (bigger) block than necessary would waste the difference.
  //
  //     Mask off every level deeper than `level`, then the highest remaining set
  //     bit is our source. countl_zero gives that in one instruction; the
  //     alternative is a loop over up to 32 levels.
  const std::uint32_t allowed = (level >= 31) ? ~0u : ((1u << (level + 1)) - 1);
  const std::uint32_t mask    = s.nonempty_mask & allowed;
  if (mask == 0)
    return OutOfMemoryError("buddy: no free block at or above level " +
                            std::to_string(level) + " (" +
                            human_bytes(buddy::block_bytes_at_level(s.arena_log2, level)) + ")");
  unsigned src = 31u - static_cast<unsigned>(std::countl_zero(mask));

  // (b) Take it.
  std::size_t g = pop_free(s, src);
  assert(s.nodes[g] == NodeState::kFree);

  // (c) Split downward toward `level`, always descending LEFT and freeing the
  //     RIGHT buddy. Exactly one push per level descended, and no
  //     push-then-immediately-pop churn.
  //
  //     Always-left has no correctness content — either child would do. It is
  //     fixed for determinism: tests can then assert exact block_ids and byte
  //     offsets rather than only asserting up to isomorphism. It also keeps churn
  //     at low addresses, leaving the high end coalesced.
  while (src < level) {
    s.nodes[g] = NodeState::kSplit;
    const std::size_t l = buddy::left_child_of(g);
    const std::size_t r = buddy::right_child_of(g);
    s.nodes[r] = NodeState::kFree;
    push_free(s, src + 1, r);            // the right buddy becomes available
    s.nodes[l] = NodeState::kFree;       // transient: taken or split next iteration
    g = l;
    ++src;
  }

  s.nodes[g] = NodeState::kUsed;
  return g;
}

// -----------------------------------------------------------------------------
// free_node — coalesce upward
// -----------------------------------------------------------------------------

void BuddyAllocator::free_node(Slab& s, std::size_t node) {
  std::size_t g = node;
  s.nodes[g] = NodeState::kFree;

  // Ascend toward the root while our buddy is also free.
  //
  // Note the asymmetry that keeps this clean: the SURVIVING node is never in a
  // free list during the loop (there is exactly one push, at the very end), while
  // the BUDDY is and must be removed. Writing it as push-then-pop-then-push would
  // be both slower and much harder to verify.
  while (!buddy::is_root(g)) {
    const std::size_t b = buddy::buddy_of(g);
    if (s.nodes[b] != NodeState::kFree) break;      // buddy is kUsed or kSplit

    remove_free(s, buddy::level_of(b), b);          // O(1), thanks to pos
    // Merged children are unreachable from here on (their parent becomes kFree,
    // not kSplit). Setting them kDetached rather than leaving stale values is
    // what lets validate() assert "every unreachable node is kDetached".
    s.nodes[b] = NodeState::kDetached;
    s.nodes[g] = NodeState::kDetached;
    g = buddy::parent_of(g);
    s.nodes[g] = NodeState::kFree;                  // was kSplit
  }

  push_free(s, buddy::level_of(g), g);
}

// -----------------------------------------------------------------------------
// Stream-ordered reclaim
// -----------------------------------------------------------------------------

bool BuddyAllocator::event_completed(std::uint32_t slot) const {
  if (slot == kNoEventSlot || slot >= event_pool_.size()) return true;
  return rt::event_query(event_pool_[slot].native());
}

StatusOr<std::uint32_t> BuddyAllocator::acquire_event_slot() {
  if (!free_event_slots_.empty()) {
    const std::uint32_t slot = free_event_slots_.back();
    free_event_slots_.pop_back();
    return slot;
  }
  // Grow the pool. Events are created here and never destroyed until the
  // allocator dies, so steady state costs zero cudaEventCreate calls.
  //
  // Purpose::kDependency, NOT kTiming: a timing event forces the GPU to record a
  // timestamp, which adds a pipeline hiccup and on some architectures prevents
  // the event being resolved entirely on-device. We only ever ask "is it done?",
  // never "how long did it take?", so timing support is pure cost.
  auto ev = rt::Event::create(rt::Event::Purpose::kDependency);
  if (!ev.ok()) return ev.status();
  event_pool_.push_back(std::move(*ev));
  return static_cast<std::uint32_t>(event_pool_.size() - 1);
}

void BuddyAllocator::release_event_slot(std::uint32_t slot) {
  if (slot != kNoEventSlot) free_event_slots_.push_back(slot);
}

void BuddyAllocator::reclaim_completed(rt::StreamHandle stream) {
  std::size_t w = 0;
  for (std::size_t r = 0; r < pending_.size(); ++r) {
    const PendingFree p = pending_[r];
    // The decision itself lives in reuse_policy.hpp, shared with
    // FreeListAllocator, so the Phase 2c comparison cannot be contaminated by the
    // two pools disagreeing about what "safe to reuse" means. Only the ACTION
    // below (return the node to the tree and coalesce) is buddy-specific.
    const bool reusable = mcke::pending_reusable(
        cfg_.reuse_policy, p.stream, p.event_slot, stream,
        [this](rt::StreamHandle h) { return stream_completed(h); },
        [this](std::uint32_t slot) { return event_completed(slot); });
    if (reusable) {
      release_event_slot(p.event_slot);
      free_node(slabs_[p.slab_id], p.node);
    } else {
      pending_[w++] = p;                 // compact in place; order carries no info
    }
  }
  pending_.resize(w);
}

void BuddyAllocator::drain_pending_blocking() {
  // Last resort. We would rather stall the host than report OOM while holding
  // memory we could reclaim — a spurious OOM is a lie, a stall is merely slow,
  // and stats_.blocking_drains makes the stall visible in the benchmark.
  for (const PendingFree& p : pending_) {
    // Wait on the event when we have one: it is strictly more precise than
    // draining the whole stream, so the stall is as short as it can be.
    if (p.event_slot != kNoEventSlot && p.event_slot < event_pool_.size()) {
      rt::event_synchronize(event_pool_[p.event_slot].native());
    } else {
#if MCKE_WITH_CUDA
      (void)cudaStreamSynchronize(p.stream);
#endif
    }
    release_event_slot(p.event_slot);
    free_node(slabs_[p.slab_id], p.node);
  }
  pending_.clear();
}

// -----------------------------------------------------------------------------
// Bypass path (requests too large to be worth a buddy tree)
// -----------------------------------------------------------------------------

StatusOr<Allocation> BuddyAllocator::allocate_bypass(std::size_t bytes) {
  auto p = raw_device_malloc(bytes);
  if (!p.ok()) {
    if (p.status().code() == StatusCode::kOutOfMemory) ++stats_.oom_events;
    return p.status();
  }
  ++stats_.raw_malloc_calls;
  bypassed_[*p] = bytes;

  stats_.bytes_reserved  += bytes;
  stats_.bytes_in_use    += bytes;
  stats_.bytes_requested += bytes;
  stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);

  Allocation a;
  a.ptr = *p;
  // bytes == requested_bytes deliberately, matching RawDeviceAllocator exactly so
  // that a Phase 2c internal_waste comparison stays apples-to-apples. The host
  // backend does round up to 256 B internally and the driver rounds by an unknown
  // amount, but neither is observable, and reporting a rounding for one allocator
  // and not the other would make buddy look worse on a metric where the two
  // behave identically. A deliberate small inaccuracy, stated rather than hidden.
  a.bytes           = bytes;
  a.requested_bytes = bytes;
  a.slab_id         = kBypassSlabId;
  a.block_id        = 0;
  return a;
}

Status BuddyAllocator::deallocate_bypass(const Allocation& a) {
  auto it = bypassed_.find(a.ptr);
  if (it == bypassed_.end())
    return InvalidArgumentError("buddy: deallocate of an unknown bypassed pointer");
  stats_.bytes_reserved  -= it->second;
  stats_.bytes_in_use    -= it->second;
  stats_.bytes_requested -= a.requested_bytes;
  bypassed_.erase(it);
  ++stats_.raw_free_calls;
  return raw_device_free(a.ptr);
}

// -----------------------------------------------------------------------------
// allocate
// -----------------------------------------------------------------------------

StatusOr<Allocation> BuddyAllocator::allocate(std::size_t bytes, rt::StreamHandle stream) {
  ++stats_.alloc_calls;      // count attempts, including failures: it is the
                             // denominator for the raw_malloc_calls claim
  if (bytes == 0) return InvalidArgumentError("buddy: zero-byte request");

  // Reclaim before considering a new cudaMalloc: parked blocks are memory we
  // already own. The empty() guard matters — a single-stream workload never parks
  // anything, so the whole mechanism costs one load and one branch on the fast
  // path. Pay for cross-stream machinery only when cross-stream reuse is in play.
  if (!pending_.empty()) reclaim_completed(stream);

  if (bytes > cfg_.large_alloc_threshold) return allocate_bypass(bytes);

  // Try every live slab.
  for (std::size_t id = 0; id < slabs_.size(); ++id) {
    Slab& s = slabs_[id];
    if (s.base == nullptr) continue;                       // dead slot from trim()
    const unsigned lvl = buddy::level_for_size(bytes, s.arena_log2, s.min_log2);
    if (lvl > s.max_level) continue;    // SENTINEL: too big for THIS slab, not fatal
    auto node = alloc_node(s, lvl);
    if (!node.ok()) continue;           // this slab is full; try the next
    return make_allocation(static_cast<std::uint32_t>(id), s, *node, bytes);
  }

  // Nothing fit. Grow before failing — growth is preferable to a stall, and a
  // stall is preferable to a false OOM.
  {
    auto id = add_slab(std::max(next_slab_bytes(), bytes), bytes);
    if (id.ok()) {
      Slab& s = slabs_[*id];
      const unsigned lvl = buddy::level_for_size(bytes, s.arena_log2, s.min_log2);
      auto node = alloc_node(s, lvl);   // fits by construction
      if (!node.ok())
        return InternalError("buddy: a fresh slab could not satisfy the request it "
                             "was sized for (" + human_bytes(bytes) + ")");
      return make_allocation(*id, s, *node, bytes);
    }
  }

  // Growth refused (max_total_bytes, or the driver is genuinely full). If blocks
  // are parked we are holding reclaimable capacity, so stall rather than lie.
  if (!pending_.empty()) {
    ++stats_.blocking_drains;
    drain_pending_blocking();
    for (std::size_t id = 0; id < slabs_.size(); ++id) {
      Slab& s = slabs_[id];
      if (s.base == nullptr) continue;
      const unsigned lvl = buddy::level_for_size(bytes, s.arena_log2, s.min_log2);
      if (lvl > s.max_level) continue;
      auto node = alloc_node(s, lvl);
      if (node.ok()) return make_allocation(static_cast<std::uint32_t>(id), s, *node, bytes);
    }
  }

  ++stats_.oom_events;
  return OutOfMemoryError(
      "buddy: out of memory for " + human_bytes(bytes) +
      " (reserved " + human_bytes(stats_.bytes_reserved) +
      ", in use " + human_bytes(stats_.bytes_in_use) +
      ", largest free block " + human_bytes(compute_largest_free_block()) + ")");
}

Allocation BuddyAllocator::make_allocation(std::uint32_t slab_id, const Slab& s,
                                           std::size_t node, std::size_t requested) {
  assert(node <= 0xFFFFFFFFull && "heap index does not fit Allocation::block_id");

  Allocation a;
  a.ptr = static_cast<char*>(s.base) + buddy::offset_of(node, s.arena_log2);
  // Alignment proof: base is 256 B aligned (cudaMalloc guarantees it; the host
  // backend mimics it), and offset_of is a multiple of the block size at this
  // level, which is >= min_block_bytes >= kDeviceAlignment. So every pointer we
  // return is 256 B aligned. That is why min_block_bytes < kDeviceAlignment is
  // rejected rather than clamped in add_slab.
  a.bytes           = buddy::block_bytes_at_level(s.arena_log2, buddy::level_of(node));
  a.requested_bytes = requested;        // ORIGINAL, never rounded
  a.slab_id         = slab_id;
  a.block_id        = static_cast<std::uint32_t>(node);
  // No `level` field: it is level_of(block_id). Do not add one.

  // bytes_in_use tracks the ROUNDED block (what we cannot give to anyone else),
  // bytes_requested the original ask. That is what makes
  // AllocatorStats::internal_waste() exactly the buddy rounding waste.
  // bytes_reserved is NOT touched here — only add_slab/bypass/trim move it, which
  // is what makes "raw_malloc_calls stops growing" structurally true.
  stats_.bytes_in_use    += a.bytes;
  stats_.bytes_requested += a.requested_bytes;
  stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
  return a;
}

// -----------------------------------------------------------------------------
// deallocate
// -----------------------------------------------------------------------------

Status BuddyAllocator::deallocate(const Allocation& a, rt::StreamHandle stream) {
  ++stats_.free_calls;
  if (a.ptr == nullptr) return OkStatus();          // free(nullptr) convention

  if (a.slab_id == kBypassSlabId) return deallocate_bypass(a);

  if (a.slab_id >= slabs_.size())
    return InvalidArgumentError("buddy: deallocate with slab_id " +
                                std::to_string(a.slab_id) + " out of range");
  Slab& s = slabs_[a.slab_id];
  if (s.base == nullptr)
    return InvalidArgumentError("buddy: deallocate into trimmed slab " +
                                std::to_string(a.slab_id));
  const std::size_t node = a.block_id;
  if (node >= s.nodes.size())
    return InvalidArgumentError("buddy: deallocate with block_id " +
                                std::to_string(node) + " out of range");
  // Returning a Status rather than asserting: a Phase 4 graph-executor bug WILL
  // hit this, and a FailedPrecondition naming the node is worth far more than UB.
  if (s.nodes[node] != NodeState::kUsed)
    return FailedPreconditionError("buddy: deallocate of a block that is not live "
                                   "(double free, or a bogus block_id): " +
                                   describe_node(s, node));

  stats_.bytes_in_use    -= a.bytes;
  stats_.bytes_requested -= a.requested_bytes;

  // Whether we can return the block to the tree right now, or must park it.
  //
  // Parking keeps the node kUsed on purpose — see PendingFree in the header: a
  // parked block must not coalesce, or the merged block could be handed to a
  // different stream while the original consumer is still reading it.
  switch (cfg_.reuse_policy) {
    case ReusePolicy::kSameStreamOnly: {
      // Never probes, by definition of the policy. Every free parks; the very
      // next allocate on this same stream reclaims it via rule 1, so in a
      // single-stream workload the block round-trips through pending_ and costs
      // one push_back plus one compaction pass. That overhead versus
      // kCoarseStreamPoll's probe is exactly what Phase 2c measures.
      ++stats_.deferred_reuses;
      pending_.push_back(PendingFree{a.slab_id, node, stream, kNoEventSlot});
      break;
    }
    case ReusePolicy::kCoarseStreamPoll: {
      if (stream_completed(stream)) {
        // Nothing outstanding on that stream at all, so ANY stream may reuse this
        // block immediately. Never enters the pending list.
        free_node(s, node);
      } else {
        ++stats_.deferred_reuses;
        pending_.push_back(PendingFree{a.slab_id, node, stream, kNoEventSlot});
      }
      break;
    }
    case ReusePolicy::kPerFreeEvent: {
      // Record unconditionally rather than skipping the event when the stream
      // already looks idle. An eager skip would be a legitimate production
      // optimisation, but it would turn this into a hybrid of the other two
      // policies and make the measured "cost of per-free events" meaningless —
      // on a quiet stream we would simply never pay it. Measure the policy as
      // stated; note the optimisation exists.
      auto slot = acquire_event_slot();
      if (!slot.ok()) {
        // Cannot obtain an event: fall back to the coarse rule rather than
        // leaking the block. Safety is never traded away, only precision.
        if (stream_completed(stream)) { free_node(s, node); break; }
        ++stats_.deferred_reuses;
        pending_.push_back(PendingFree{a.slab_id, node, stream, kNoEventSlot});
        break;
      }
      if (!rt::event_record(event_pool_[*slot].native(), stream)) {
        // The record failed, so this event will never complete and a block parked
        // on it would be unreclaimable forever -- a leak dressed up as safety.
        // Fall back to the coarse rule: less precise, still sound. Safety is
        // never traded away, only precision. (Same shape as the
        // acquire_event_slot failure path directly above.)
        release_event_slot(*slot);
        if (stream_completed(stream)) { free_node(s, node); break; }
        ++stats_.deferred_reuses;
        pending_.push_back(PendingFree{a.slab_id, node, stream, kNoEventSlot});
        break;
      }
      ++stats_.deferred_reuses;
      pending_.push_back(PendingFree{a.slab_id, node, stream, *slot});
      break;
    }
  }
  return OkStatus();
}

// -----------------------------------------------------------------------------
// trim
// -----------------------------------------------------------------------------

Status BuddyAllocator::trim() {
  // A parked block keeps its slab busy, so drain first or we would refuse to
  // release slabs that are in fact idle.
  if (!pending_.empty()) {
    ++stats_.blocking_drains;
    drain_pending_blocking();
  }

  Status first_error;
  for (Slab& s : slabs_) {
    if (s.base == nullptr) continue;
    // Maximal coalescing makes this test exact and O(1): the root is kFree if and
    // only if the entire slab is unused. No scan required.
    if (s.nodes[0] != NodeState::kFree) continue;

    Status st = raw_device_free(s.base);
    if (!st.ok() && first_error.ok()) first_error = st;
    ++stats_.raw_free_calls;
    stats_.bytes_reserved -= s.bytes;

    // DEAD SLOT, not erased — see the trim() comment in the header. Erasing would
    // shift indices and invalidate the slab_id in every outstanding Allocation.
    s.base  = nullptr;
    s.bytes = 0;
    // Release the metadata too; that is 18 MiB of host RAM for a 256 MiB slab.
    s.nodes.clear();
    s.nodes.shrink_to_fit();
    s.pos.clear();
    s.pos.shrink_to_fit();
    s.free_lists.clear();
    s.free_lists.shrink_to_fit();
    s.nonempty_mask = 0;
  }
  return first_error;
}

// -----------------------------------------------------------------------------
// stats / largest_free_block
// -----------------------------------------------------------------------------

std::size_t BuddyAllocator::compute_largest_free_block() const {
  // In a buddy allocator with maximal coalescing, "largest contiguous free
  // extent" and "largest allocatable block" are the SAME number, because two free
  // buddies always merge. The shallowest non-empty free level therefore *is* the
  // largest free block. In a non-coalescing size-class pool those two quantities
  // diverge wildly — and that divergence is the headline of the Phase 2c
  // comparison, which is the whole reason this stat exists.
  //
  // Two caveats worth knowing when reading the number:
  //   - a block parked in pending_ is still kUsed, so this UNDERSTATES recoverable
  //     capacity while blocks are parked;
  //   - it says nothing about "we could grow a new slab". It is a fragmentation
  //     indicator for the current footprint, not a capacity forecast.
  std::size_t best = 0;
  for (const Slab& s : slabs_) {
    if (s.base == nullptr || s.nonempty_mask == 0) continue;
    const unsigned l = static_cast<unsigned>(std::countr_zero(s.nonempty_mask));
    best = std::max(best, buddy::block_bytes_at_level(s.arena_log2, l));
  }
  return best;
}

AllocatorStats BuddyAllocator::stats() const {
  // Computed on demand rather than maintained incrementally. An incrementally
  // tracked maximum is the classic "goes stale on the path you forgot" bug, and
  // the free path both raises and lowers it — so the decreasing case would need a
  // full recompute anyway. With nonempty_mask this is a handful of instructions.
  //
  // Consequence to be aware of: stats() is no longer free. Do not call it inside
  // a timed benchmark loop.
  AllocatorStats s = stats_;
  s.largest_free_block = compute_largest_free_block();
  return s;
}

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------

std::string BuddyAllocator::describe_node(const Slab& s, std::size_t node) const {
  const char* state = "?";
  switch (s.nodes[node]) {
    case NodeState::kDetached: state = "kDetached"; break;
    case NodeState::kFree:     state = "kFree";     break;
    case NodeState::kSplit:    state = "kSplit";    break;
    case NodeState::kUsed:     state = "kUsed";     break;
  }
  const unsigned l = buddy::level_of(node);
  std::ostringstream os;
  os << "node " << node << " (level " << l
     << ", idx " << buddy::index_in_level(node)
     << ", off " << buddy::offset_of(node, s.arena_log2)
     << ", " << human_bytes(buddy::block_bytes_at_level(s.arena_log2, l))
     << ", state=" << state << ")";
  return os.str();
}

std::string BuddyAllocator::dump_free_map() const {
  std::ostringstream os;
  for (std::size_t id = 0; id < slabs_.size(); ++id) {
    const Slab& s = slabs_[id];
    if (s.base == nullptr) { os << "slab " << id << ": <trimmed>\n"; continue; }
    os << "slab " << id << ": " << human_bytes(s.bytes) << " K=" << s.arena_log2
       << " M=" << s.min_log2 << " L=" << s.max_level
       << " nodes=" << s.nodes.size() << '\n';
    for (unsigned l = 0; l <= s.max_level; ++l) {
      if (s.free_lists[l].empty()) continue;
      // Sorted, because swap-and-pop permutes the list and a dump you cannot
      // diff across runs is a dump you cannot use.
      std::vector<std::size_t> sorted = s.free_lists[l];
      std::sort(sorted.begin(), sorted.end());
      os << "  level " << l << " (" << human_bytes(buddy::block_bytes_at_level(s.arena_log2, l))
         << "): " << sorted.size() << " free   [";
      for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 8); ++i)
        os << sorted[i] << (i + 1 < std::min<std::size_t>(sorted.size(), 8) ? " " : "");
      if (sorted.size() > 8) os << " ...";
      os << "]\n";
    }
  }
  const AllocatorStats st = stats();
  os << "  largest_free_block=" << human_bytes(st.largest_free_block)
     << " in_use=" << human_bytes(st.bytes_in_use)
     << " requested=" << human_bytes(st.bytes_requested)
     << " internal_waste=" << human_bytes(st.internal_waste())
     << " pending=" << pending_.size() << '\n';
  return os.str();
}

// -----------------------------------------------------------------------------
// validate
// -----------------------------------------------------------------------------

Status BuddyAllocator::validate_slab(const Slab& s, std::uint32_t id) const {
  const std::string tag = "buddy: slab " + std::to_string(id) + " ";

  std::vector<char> visited(s.nodes.size(), 0);
  std::vector<std::size_t> free_count(s.max_level + 1, 0);
  std::size_t free_bytes = 0, used_bytes = 0;
  // (offset, bytes) of every live block, for the address-space disjointness check.
  std::vector<std::pair<std::size_t, std::size_t>> live;

  // Iterative walk from the root. Explicit stack, not recursion: a 2-million-node
  // tree would overflow the call stack.
  std::vector<std::size_t> stack{0};
  while (!stack.empty()) {
    const std::size_t g = stack.back();
    stack.pop_back();
    if (g >= s.nodes.size())
      return InternalError(tag + "walk reached out-of-range node " + std::to_string(g));
    visited[g] = 1;
    const unsigned l = buddy::level_of(g);

    switch (s.nodes[g]) {
      case NodeState::kFree: {
        if (s.pos[g] == kNotInList)
          return InternalError(tag + describe_node(s, g) + ": kFree but not in any free list");
        if (s.pos[g] >= s.free_lists[l].size() || s.free_lists[l][s.pos[g]] != g)
          return InternalError(tag + describe_node(s, g) + ": pos does not point back at it");
        // MAXIMAL COALESCING, from the kFree side.
        if (!buddy::is_root(g) && s.nodes[buddy::buddy_of(g)] == NodeState::kFree)
          return InternalError(tag + describe_node(s, g) + ": buddy " +
                               std::to_string(buddy::buddy_of(g)) +
                               " is also kFree -> coalescing not maximal");
        ++free_count[l];
        free_bytes += buddy::block_bytes_at_level(s.arena_log2, l);
        break;
      }
      case NodeState::kUsed: {
        if (s.pos[g] != kNotInList)
          return InternalError(tag + describe_node(s, g) + ": kUsed but present in a free list");
        used_bytes += buddy::block_bytes_at_level(s.arena_log2, l);
        live.emplace_back(buddy::offset_of(g, s.arena_log2),
                          buddy::block_bytes_at_level(s.arena_log2, l));
        break;
      }
      case NodeState::kSplit: {
        if (l >= s.max_level)
          return InternalError(tag + describe_node(s, g) + ": leaf level cannot be split");
        if (s.pos[g] != kNotInList)
          return InternalError(tag + describe_node(s, g) + ": kSplit but present in a free list");
        const std::size_t lc = buddy::left_child_of(g), rc = buddy::right_child_of(g);
        // MAXIMAL COALESCING, from the kSplit side. Catches the same bug class
        // from the opposite direction: a coalesce loop that exits one step early
        // trips one of these two checks.
        if (s.nodes[lc] == NodeState::kFree && s.nodes[rc] == NodeState::kFree)
          return InternalError(tag + describe_node(s, g) +
                               ": both children kFree -> should have coalesced");
        if (s.nodes[lc] == NodeState::kDetached && s.nodes[rc] == NodeState::kDetached)
          return InternalError(tag + describe_node(s, g) + ": kSplit with no live children");
        stack.push_back(lc);
        stack.push_back(rc);
        break;
      }
      case NodeState::kDetached:
        return InternalError(tag + describe_node(s, g) + ": reachable node is kDetached");
    }
  }

  // Unreachable sweep: anything the walk did not visit must be detached and out
  // of every list.
  for (std::size_t g = 0; g < s.nodes.size(); ++g) {
    if (visited[g]) continue;
    if (s.nodes[g] != NodeState::kDetached)
      return InternalError(tag + describe_node(s, g) + ": unreachable but not kDetached");
    if (s.pos[g] != kNotInList)
      return InternalError(tag + describe_node(s, g) + ": unreachable but in a free list");
  }

  // Free lists, checked in BOTH directions. One direction alone misses half the
  // bugs: forward catches stale entries, backward catches missing ones.
  for (unsigned l = 0; l <= s.max_level; ++l) {
    for (std::size_t i = 0; i < s.free_lists[l].size(); ++i) {
      const std::size_t g = s.free_lists[l][i];
      if (g >= s.nodes.size())
        return InternalError(tag + "free_lists[" + std::to_string(l) +
                             "] holds out-of-range node " + std::to_string(g));
      if (buddy::level_of(g) != l)
        return InternalError(tag + describe_node(s, g) + " is in free_lists[" +
                             std::to_string(l) + "] but belongs to level " +
                             std::to_string(buddy::level_of(g)));
      if (s.nodes[g] != NodeState::kFree)
        return InternalError(tag + describe_node(s, g) + " is in a free list but not kFree");
      if (s.pos[g] != i)
        return InternalError(tag + describe_node(s, g) + ": pos=" + std::to_string(s.pos[g]) +
                             " but sits at index " + std::to_string(i));
    }
    if (free_count[l] != s.free_lists[l].size())
      return InternalError(tag + "level " + std::to_string(l) + ": walk found " +
                           std::to_string(free_count[l]) + " free nodes but the list holds " +
                           std::to_string(s.free_lists[l].size()));
    // The derived mask must agree with the lists — this is what makes keeping a
    // redundant nonempty_mask safe rather than a liability.
    const bool bit = ((s.nonempty_mask >> l) & 1u) != 0;
    if (bit != !s.free_lists[l].empty())
      return InternalError(tag + "nonempty_mask bit " + std::to_string(l) +
                           " disagrees with free_lists[" + std::to_string(l) + "]");
  }

  // Live blocks must not overlap in the address space.
  //
  // The walk already proves this structurally (every kUsed node is a leaf of the
  // walk, and a kSplit node's children exactly tile it). Doing the address check
  // anyway is NOT redundant in value: the structural check proves it from the
  // tree, this one from what we actually handed out. If they ever disagree, the
  // bug is in offset_of/heap_index rather than in the tree — a useful signal.
  std::sort(live.begin(), live.end());
  for (std::size_t i = 0; i + 1 < live.size(); ++i) {
    if (live[i].first + live[i].second > live[i + 1].first)
      return InternalError(tag + "live blocks overlap: [" + std::to_string(live[i].first) +
                           ", +" + std::to_string(live[i].second) + ") vs [" +
                           std::to_string(live[i + 1].first) + ", +" +
                           std::to_string(live[i + 1].second) + ")");
  }

  // Bytes account exactly — this is where "no dead tail" pays off.
  if (free_bytes + used_bytes != s.bytes)
    return InternalError(tag + "bytes do not account: free " + std::to_string(free_bytes) +
                         " + used " + std::to_string(used_bytes) + " != slab " +
                         std::to_string(s.bytes));
  return OkStatus();
}

Status BuddyAllocator::validate() const {
  // Return on the FIRST failure: the first violation is almost always the cause
  // and everything after it is a consequence.
  std::size_t used_total = 0, reserved_slabs = 0;
  for (std::size_t id = 0; id < slabs_.size(); ++id) {
    const Slab& s = slabs_[id];
    if (s.base == nullptr) continue;
    MCKE_RETURN_IF_ERROR(validate_slab(s, static_cast<std::uint32_t>(id)));
    reserved_slabs += s.bytes;
    for (std::size_t g = 0; g < s.nodes.size(); ++g)
      if (s.nodes[g] == NodeState::kUsed)
        used_total += buddy::block_bytes_at_level(s.arena_log2, buddy::level_of(g));
  }

  // A parked block is still kUsed in the tree but has already been subtracted
  // from bytes_in_use (the caller genuinely released it). Rather than add a
  // bytes_pending field, state the exact invariant and check it:
  //     sum(kUsed blocks) == bytes_in_use + sum(parked blocks)
  std::size_t pending_bytes = 0;
  for (const PendingFree& p : pending_) {
    if (p.slab_id >= slabs_.size() || slabs_[p.slab_id].base == nullptr)
      return InternalError("buddy: pending free references a dead slab");
    const Slab& s = slabs_[p.slab_id];
    if (s.nodes[p.node] != NodeState::kUsed)
      return InternalError("buddy: parked " + describe_node(s, p.node) + " is not kUsed");
    pending_bytes += buddy::block_bytes_at_level(s.arena_log2, buddy::level_of(p.node));
  }

  std::size_t bypass_bytes = 0;
  for (const auto& [ptr, bytes] : bypassed_) {
    bypass_bytes += bytes;
    // Catches the nightmare case where a pooled pointer ends up in bypassed_ and
    // would be handed to cudaFree.
    for (const Slab& s : slabs_) {
      if (s.base == nullptr) continue;
      const auto p = static_cast<const char*>(ptr);
      const auto b = static_cast<const char*>(s.base);
      if (p >= b && p < b + s.bytes)
        return InternalError("buddy: a bypassed pointer lies inside a live slab");
    }
  }

  if (used_total != stats_.bytes_in_use - bypass_bytes + pending_bytes)
    return InternalError("buddy: kUsed bytes (" + std::to_string(used_total) +
                         ") != bytes_in_use - bypass + pending (" +
                         std::to_string(stats_.bytes_in_use - bypass_bytes + pending_bytes) + ")");
  if (stats_.bytes_reserved != reserved_slabs + bypass_bytes)
    return InternalError("buddy: bytes_reserved (" + std::to_string(stats_.bytes_reserved) +
                         ") != slabs + bypassed (" +
                         std::to_string(reserved_slabs + bypass_bytes) + ")");
  if (stats_.bytes_in_use < stats_.bytes_requested)
    return InternalError("buddy: bytes_in_use < bytes_requested");
  return OkStatus();
}

}  // namespace mcke
