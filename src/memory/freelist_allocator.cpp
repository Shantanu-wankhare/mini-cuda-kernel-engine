// =============================================================================
//  src/memory/freelist_allocator.cpp
//
//  WHY .cpp: no device code, same as the buddy allocator — pure host bookkeeping
//  over raw_device_malloc. Fully unit-testable on a MacBook.
//
//  ---------------------------------------------------------------------------
//  ORIENTATION
//
//  Three places a block can live, and it is in EXACTLY one of them at any time:
//
//     live_          the caller has it
//     classes_[c]    cached and immediately reusable
//     pending_       released by the caller but its GPU work may be in flight
//
//  That three-way partition is the whole correctness argument, and validate()
//  checks it directly (no pointer in two places, none in none of them). It is
//  also the analogue of buddy's kUsed/kFree/parked node states — same idea,
//  expressed as container membership instead of a state byte, because this design
//  has no tree to hang a state on.
//
//  There is deliberately NO coalescing anywhere in this file. If you find
//  yourself wanting to merge two adjacent free blocks, that is the buddy
//  allocator; the entire point of this one is what happens when you do not.
// =============================================================================
#include "mcke/memory/freelist_allocator.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <sstream>

#include "mcke/core/device.hpp"
#include "mcke/memory/buddy_math.hpp"   // ceil_log2 / is_power_of_two only

namespace mcke {

namespace {
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
// Construction
// -----------------------------------------------------------------------------

FreeListAllocator::FreeListAllocator(FreeListConfig cfg) : cfg_(cfg) {
  // Derive the ladder geometry once. If the config is invalid these stay zero and
  // every call fails cleanly through validate_config().
  if (!validate_config().ok()) return;
  split_log2_ = buddy::ceil_log2(cfg_.small_large_split);
  n_small_    = cfg_.small_large_split / cfg_.small_class_granularity;
  // Size the class table to cover everything the pool will ever serve. Anything
  // above large_alloc_threshold bypasses the pool, so no larger class can exist.
  const std::size_t max_cls = size_class_of(cfg_.large_alloc_threshold);
  classes_.assign(max_cls + 1, {});
}

FreeListAllocator::~FreeListAllocator() {
  if (stats_.bytes_in_use != 0) {
    std::fprintf(stderr,
                 "[mcke] warning: FreeListAllocator destroyed with %zu bytes still "
                 "live across %llu allocate / %llu free calls — a caller leaked\n",
                 stats_.bytes_in_use,
                 static_cast<unsigned long long>(stats_.alloc_calls),
                 static_cast<unsigned long long>(stats_.free_calls));
  }
  for (Slab& s : slabs_) {
    if (s.base != nullptr) { (void)raw_device_free(s.base); s.base = nullptr; }
  }
  for (const auto& [ptr, bytes] : bypassed_) { (void)bytes; (void)raw_device_free(ptr); }
}

Status FreeListAllocator::validate_config() const {
  const std::size_t G = cfg_.small_class_granularity;
  const std::size_t S = cfg_.small_large_split;
  if (!buddy::is_power_of_two(G))
    return InvalidArgumentError("freelist: small_class_granularity must be a power "
                                "of two, got " + std::to_string(G));
  // Rejected, not clamped, for the same reason as buddy's min_block_bytes: every
  // block we hand out is a multiple of G, so G < 256 would silently produce
  // misaligned pointers whose only symptom is uncoalesced loads much later.
  if (G < kDeviceAlignment)
    return InvalidArgumentError("freelist: small_class_granularity (" +
                                std::to_string(G) + ") must be >= kDeviceAlignment (" +
                                std::to_string(kDeviceAlignment) + ")");
  if (!buddy::is_power_of_two(S))
    return InvalidArgumentError("freelist: small_large_split must be a power of two, got " +
                                std::to_string(S));
  if (S < G)
    return InvalidArgumentError("freelist: small_large_split must be >= granularity");
  if (cfg_.large_alloc_threshold < S)
    return InvalidArgumentError("freelist: large_alloc_threshold must be >= "
                                "small_large_split");
  if (cfg_.slab_bytes < G)
    return InvalidArgumentError("freelist: slab_bytes too small");
  return OkStatus();
}

// -----------------------------------------------------------------------------
// The size-class ladder
// -----------------------------------------------------------------------------

std::size_t FreeListAllocator::size_class_of(std::size_t bytes) const {
  const std::size_t G = cfg_.small_class_granularity;
  const std::size_t S = cfg_.small_large_split;
  if (bytes == 0) return 0;
  if (bytes <= S) {
    // LINEAR regime: ceil(bytes / G) - 1. This is the half that can be finer than
    // a power of two, and therefore the half where this design beats buddy.
    return (bytes + G - 1) / G - 1;
  }
  // POWER-OF-TWO regime. Identical rounding to buddy above S — see the header for
  // why that matters more than it sounds.
  const unsigned k = buddy::ceil_log2(bytes);        // bytes <= 2^k
  return n_small_ + (k - (split_log2_ + 1));
}

std::size_t FreeListAllocator::class_block_bytes(std::size_t cls) const {
  if (cls < n_small_) return (cls + 1) * cfg_.small_class_granularity;
  return std::size_t{1} << (split_log2_ + 1 + (cls - n_small_));
}

// -----------------------------------------------------------------------------
// Slabs
// -----------------------------------------------------------------------------

StatusOr<std::uint32_t> FreeListAllocator::add_slab(std::size_t bytes) {
  MCKE_RETURN_IF_ERROR(validate_config());

  std::size_t reserved = 0;
  for (const Slab& s : slabs_)
    if (s.base != nullptr) reserved += s.bytes;
  if (cfg_.max_total_bytes != 0 && reserved + bytes > cfg_.max_total_bytes)
    return OutOfMemoryError("freelist: slab of " + human_bytes(bytes) +
                            " would exceed max_total_bytes (" +
                            human_bytes(cfg_.max_total_bytes) + ")");

  auto p = raw_device_malloc(bytes);
  if (!p.ok()) return p.status();
  ++stats_.raw_malloc_calls;

  // Reuse a dead slot so slab ids stay stable across trim(), for the same reason
  // buddy does: an outstanding Allocation carries a slab_id.
  std::uint32_t id = static_cast<std::uint32_t>(slabs_.size());
  for (std::size_t i = 0; i < slabs_.size(); ++i)
    if (slabs_[i].base == nullptr) { id = static_cast<std::uint32_t>(i); break; }
  if (id == slabs_.size()) slabs_.emplace_back();

  slabs_[id].base  = *p;
  slabs_[id].bytes = bytes;
  slabs_[id].bump  = 0;
  stats_.bytes_reserved += bytes;
  return id;
}

Status FreeListAllocator::reserve(std::size_t bytes) {
  if (bytes == 0) return InvalidArgumentError("freelist: reserve(0)");
  for (const Slab& s : slabs_)
    if (s.base != nullptr && s.bytes >= bytes) return OkStatus();
  auto id = add_slab(std::max(bytes, cfg_.slab_bytes));
  if (!id.ok()) return id.status();
  return OkStatus();
}

// -----------------------------------------------------------------------------
// Event pool + stream-ordered reclaim (mirrors BuddyAllocator exactly, on purpose)
// -----------------------------------------------------------------------------

bool FreeListAllocator::event_completed(std::uint32_t slot) const {
  if (slot == kNoEventSlot || slot >= event_pool_.size()) return true;
  return rt::event_query(event_pool_[slot].native());
}

StatusOr<std::uint32_t> FreeListAllocator::acquire_event_slot() {
  if (!free_event_slots_.empty()) {
    const std::uint32_t slot = free_event_slots_.back();
    free_event_slots_.pop_back();
    return slot;
  }
  auto ev = rt::Event::create(rt::Event::Purpose::kDependency);
  if (!ev.ok()) return ev.status();
  event_pool_.push_back(std::move(*ev));
  return static_cast<std::uint32_t>(event_pool_.size() - 1);
}

void FreeListAllocator::release_event_slot(std::uint32_t slot) {
  if (slot != kNoEventSlot) free_event_slots_.push_back(slot);
}

void FreeListAllocator::cache_block(void* ptr, std::size_t bytes) {
  const std::size_t cls = size_class_of(bytes);
  classes_[cls].push_back(CachedBlock{ptr, bytes});
}

void FreeListAllocator::reclaim_completed(rt::StreamHandle stream) {
  std::size_t w = 0;
  for (std::size_t r = 0; r < pending_.size(); ++r) {
    const PendingFree p = pending_[r];
    // Same shared decision function buddy uses, so the two pools cannot disagree
    // about what "safe to reuse" means. Only the action differs: buddy returns a
    // node to its tree and coalesces; we push a pointer onto a class list and
    // deliberately do not.
    const bool reusable = mcke::pending_reusable(
        cfg_.reuse_policy, p.stream, p.event_slot, stream,
        [this](rt::StreamHandle h) { return stream_completed(h); },
        [this](std::uint32_t slot) { return event_completed(slot); });
    if (reusable) {
      release_event_slot(p.event_slot);
      cache_block(p.ptr, p.bytes);
    } else {
      pending_[w++] = p;
    }
  }
  pending_.resize(w);
}

void FreeListAllocator::drain_pending_blocking() {
  for (const PendingFree& p : pending_) {
    // Braces OUTSIDE the #if, matching buddy_allocator.cpp. Putting the `else`
    // keyword inside the guard "works" but is a trap: the token disappears in a
    // host build, so a statement added later silently reassociates in one build
    // and fails to compile in the other.
    if (p.event_slot != kNoEventSlot && p.event_slot < event_pool_.size()) {
      rt::event_synchronize(event_pool_[p.event_slot].native());
    } else {
#if MCKE_WITH_CUDA
      (void)cudaStreamSynchronize(p.stream);
#endif
    }
    release_event_slot(p.event_slot);
    cache_block(p.ptr, p.bytes);
  }
  pending_.clear();
}

// -----------------------------------------------------------------------------
// Bypass
// -----------------------------------------------------------------------------

StatusOr<Allocation> FreeListAllocator::allocate_bypass(std::size_t bytes) {
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
  a.bytes = bytes;                 // zero reported waste, matching buddy's bypass
  a.requested_bytes = bytes;       // and RawDeviceAllocator, for comparability
  a.slab_id = kBypassSlabId;
  return a;
}

Status FreeListAllocator::deallocate_bypass(const Allocation& a) {
  auto it = bypassed_.find(a.ptr);
  if (it == bypassed_.end())
    return InvalidArgumentError("freelist: deallocate of an unknown bypassed pointer");
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

void* FreeListAllocator::try_split_larger(std::size_t cls) {
  // Serve `cls` from a larger cached block, re-caching the remainder.
  //
  // Only accept a split whose remainder is ITSELF exactly a valid class block
  // size — otherwise the leftover bytes would be unaddressable by any class and
  // would leak silently out of our accounting. Conveniently the ladder makes this
  // common: in the linear regime (j-i)*G is always a valid class, and in the
  // power-of-two regime 2^(k+1) - 2^k = 2^k is too.
  //
  // Bounded search (8 classes up) so this stays O(1)-ish rather than scanning the
  // whole ladder on every miss.
  const std::size_t need = class_block_bytes(cls);
  const std::size_t limit = std::min(classes_.size(), cls + 9);
  for (std::size_t c = cls + 1; c < limit; ++c) {
    if (classes_[c].empty()) continue;
    const std::size_t whole = class_block_bytes(c);
    const std::size_t rest  = whole - need;
    if (rest == 0) continue;
    const std::size_t rest_cls = size_class_of(rest);
    if (class_block_bytes(rest_cls) != rest) continue;   // would leak bytes: skip

    const CachedBlock blk = classes_[c].back();
    classes_[c].pop_back();
    void* head = blk.ptr;
    void* tail = static_cast<char*>(blk.ptr) + need;
    // The remainder can never be merged back into `head` — that is precisely the
    // cost this heuristic pays, and why it is off by default.
    classes_[rest_cls].push_back(CachedBlock{tail, rest});
    return head;
  }
  return nullptr;
}

StatusOr<Allocation> FreeListAllocator::allocate(std::size_t bytes, rt::StreamHandle stream) {
  ++stats_.alloc_calls;
  if (bytes == 0) return InvalidArgumentError("freelist: zero-byte request");
  MCKE_RETURN_IF_ERROR(validate_config());

  if (!pending_.empty()) reclaim_completed(stream);

  if (bytes > cfg_.large_alloc_threshold) return allocate_bypass(bytes);

  const std::size_t cls = size_class_of(bytes);
  const std::size_t bs  = class_block_bytes(cls);

  void* ptr = nullptr;

  // (1) Class hit — the O(1) fast path this design exists for.
  if (!classes_[cls].empty()) {
    ptr = classes_[cls].back().ptr;
    classes_[cls].pop_back();
  }
  // (2) Optional: split a larger cached block.
  if (ptr == nullptr && cfg_.split_large_blocks) ptr = try_split_larger(cls);

  // (3) Carve virgin space off a slab's bump pointer.
  if (ptr == nullptr) {
    for (Slab& s : slabs_) {
      if (s.base == nullptr) continue;
      if (s.bump + bs > s.bytes) continue;
      ptr = static_cast<char*>(s.base) + s.bump;
      s.bump += bs;
      break;
    }
  }
  // (4) Grow.
  if (ptr == nullptr) {
    auto id = add_slab(std::max(cfg_.slab_bytes, bs));
    if (id.ok()) {
      Slab& s = slabs_[*id];
      ptr = static_cast<char*>(s.base) + s.bump;
      s.bump += bs;
    }
  }
  // (5) Last resort before lying about OOM: reclaim parked blocks with a stall.
  if (ptr == nullptr && !pending_.empty()) {
    ++stats_.blocking_drains;
    drain_pending_blocking();
    if (!classes_[cls].empty()) {
      ptr = classes_[cls].back().ptr;
      classes_[cls].pop_back();
    }
  }

  if (ptr == nullptr) {
    ++stats_.oom_events;
    // The diagnostic that makes external fragmentation legible: free bytes can be
    // far larger than the request while the largest CONTIGUOUS free block is
    // smaller than it. That inequality IS the failure mode of this design, so the
    // message states all three numbers rather than just "out of memory".
    const std::size_t freeb = stats_.bytes_reserved - stats_.bytes_in_use;
    return OutOfMemoryError(
        "freelist: out of memory for " + human_bytes(bytes) + " (class block " +
        human_bytes(bs) + ") while holding " + human_bytes(freeb) +
        " free; largest contiguous free block is " +
        human_bytes(compute_largest_free_block()) + " -- external fragmentation");
  }

  live_[ptr] = bs;
  stats_.bytes_in_use    += bs;
  stats_.bytes_requested += bytes;
  stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);

  Allocation a;
  a.ptr = ptr;
  a.bytes = bs;                // rounded-up class block
  a.requested_bytes = bytes;   // original ask, so internal_waste() is meaningful
  // slab_id is informational here (the pool does not need it to route a free, and
  // blocks may be split across the bump boundary), so record which slab contains
  // the pointer for validate()'s benefit.
  a.slab_id = 0;
  for (std::size_t i = 0; i < slabs_.size(); ++i) {
    const Slab& s = slabs_[i];
    if (s.base == nullptr) continue;
    const auto p = static_cast<const char*>(ptr);
    const auto b = static_cast<const char*>(s.base);
    if (p >= b && p < b + s.bytes) { a.slab_id = static_cast<std::uint32_t>(i); break; }
  }
  return a;
}

// -----------------------------------------------------------------------------
// deallocate
// -----------------------------------------------------------------------------

Status FreeListAllocator::deallocate(const Allocation& a, rt::StreamHandle stream) {
  ++stats_.free_calls;
  if (a.ptr == nullptr) return OkStatus();
  if (a.slab_id == kBypassSlabId) return deallocate_bypass(a);

  // The double-free / bogus-pointer check that live_ is really paying for.
  auto it = live_.find(a.ptr);
  if (it == live_.end())
    return FailedPreconditionError("freelist: deallocate of a pointer that is not "
                                   "live (double free, or never allocated here)");
  const std::size_t bs = it->second;
  live_.erase(it);

  stats_.bytes_in_use    -= bs;
  stats_.bytes_requested -= a.requested_bytes;

  switch (cfg_.reuse_policy) {
    case ReusePolicy::kSameStreamOnly:
      ++stats_.deferred_reuses;
      pending_.push_back(PendingFree{a.ptr, bs, stream, kNoEventSlot});
      break;
    case ReusePolicy::kCoarseStreamPoll:
      if (stream_completed(stream)) cache_block(a.ptr, bs);
      else {
        ++stats_.deferred_reuses;
        pending_.push_back(PendingFree{a.ptr, bs, stream, kNoEventSlot});
      }
      break;
    case ReusePolicy::kPerFreeEvent: {
      auto slot = acquire_event_slot();
      if (!slot.ok()) {
        if (stream_completed(stream)) { cache_block(a.ptr, bs); break; }
        ++stats_.deferred_reuses;
        pending_.push_back(PendingFree{a.ptr, bs, stream, kNoEventSlot});
        break;
      }
      if (!rt::event_record(event_pool_[*slot].native(), stream)) {
        // See buddy_allocator.cpp: a block parked on an event that never
        // completes is a permanent leak. Degrade to the coarse rule instead.
        release_event_slot(*slot);
        if (stream_completed(stream)) { cache_block(a.ptr, bs); break; }
        ++stats_.deferred_reuses;
        pending_.push_back(PendingFree{a.ptr, bs, stream, kNoEventSlot});
        break;
      }
      ++stats_.deferred_reuses;
      pending_.push_back(PendingFree{a.ptr, bs, stream, *slot});
      break;
    }
  }
  return OkStatus();
}

// -----------------------------------------------------------------------------
// trim / stats / diagnostics
// -----------------------------------------------------------------------------

Status FreeListAllocator::settle_pending() {
  if (!pending_.empty()) {
    ++stats_.blocking_drains;
    drain_pending_blocking();
  }
  return OkStatus();
}

Status FreeListAllocator::trim() {
  if (!pending_.empty()) { ++stats_.blocking_drains; drain_pending_blocking(); }

  // A slab can only be released if NOTHING inside it is live and nothing cached
  // points into it. Unlike buddy — where "is the root free?" answers this in O(1)
  // because coalescing guarantees a fully-free slab has a free root — this design
  // has to scan, precisely BECAUSE it never coalesces. The cost of the missing
  // merge shows up here too, not only in fragmentation.
  Status first_error;
  for (std::size_t i = 0; i < slabs_.size(); ++i) {
    Slab& s = slabs_[i];
    if (s.base == nullptr) continue;
    const auto* b = static_cast<const char*>(s.base);
    auto inside = [&](const void* p) {
      const auto* c = static_cast<const char*>(p);
      return c >= b && c < b + s.bytes;
    };
    bool busy = false;
    for (const auto& [ptr, bytes] : live_) { (void)bytes; if (inside(ptr)) { busy = true; break; } }
    if (busy) continue;

    // Drop every cached block that lives in this slab.
    std::size_t cached_here = 0;
    for (auto& lst : classes_) {
      std::size_t w = 0;
      for (std::size_t r = 0; r < lst.size(); ++r) {
        if (inside(lst[r].ptr)) { cached_here += lst[r].bytes; continue; }
        lst[w++] = lst[r];
      }
      lst.resize(w);
    }
    (void)cached_here;

    Status st = raw_device_free(s.base);
    if (!st.ok() && first_error.ok()) first_error = st;
    ++stats_.raw_free_calls;
    stats_.bytes_reserved -= s.bytes;
    s.base = nullptr; s.bytes = 0; s.bump = 0;   // dead slot, never erased
  }
  return first_error;
}

std::size_t FreeListAllocator::compute_largest_free_block() const {
  // DEFINITION (must match buddy's, or the RESULTS.md column is not comparable):
  // "the largest single contiguous region this allocator could hand out right now
  // without going to the driver."
  //
  // For this design that is the max of (a) the largest cached block and (b) the
  // largest un-carved bump tail. Including the tail matters: a fresh slab has no
  // cached blocks at all, and reporting 0 there would be plainly wrong.
  //
  // In buddy, this quantity and "largest allocatable block" are the same number
  // because coalescing guarantees it. Here they are the same only by construction
  // of this function — and the gap between this number and total free bytes is
  // exactly the external fragmentation the Phase 2c comparison is looking for.
  //
  // SUBTLETY, found by the largest_free_block PROBE in bench/alloc_bench.cpp
  // (allocate exactly this many bytes and confirm it succeeds with no growth):
  // a virgin bump-pointer TAIL's raw byte count is NOT itself an allocatable
  // size. Any real request for `tail` bytes goes through size_class_of(), which
  // rounds UP to a class boundary — and that rounded amount can legitimately
  // exceed the tail, in which case the tail cannot serve the request at all
  // without growing a slab. Reporting the raw tail here would claim capacity
  // that allocate() cannot actually deliver, which is precisely the kind of
  // self-reported-counter lie the probe exists to catch. So for the tail we
  // report the LARGEST CLASS BLOCK SIZE that still fits inside it, not the raw
  // byte count — the two coincide only when the tail happens to land exactly on
  // a class boundary.
  std::size_t best = 0;
  for (const auto& lst : classes_)
    for (const auto& blk : lst) best = std::max(best, blk.bytes);
  for (const Slab& s : slabs_) {
    if (s.base == nullptr) continue;
    const std::size_t remaining = s.bytes - s.bump;
    if (remaining == 0) continue;
    for (std::size_t c = classes_.size(); c-- > 0;) {
      const std::size_t cb = class_block_bytes(c);
      if (cb <= remaining) { best = std::max(best, cb); break; }
    }
  }
  return best;
}

AllocatorStats FreeListAllocator::stats() const {
  AllocatorStats s = stats_;
  s.largest_free_block = compute_largest_free_block();
  return s;
}

std::string FreeListAllocator::dump_free_map() const {
  std::ostringstream os;
  os << "freelist: " << n_small_ << " linear classes of "
     << human_bytes(cfg_.small_class_granularity) << " up to "
     << human_bytes(cfg_.small_large_split) << ", then powers of two ("
     << classes_.size() << " classes total)\n";
  for (std::size_t i = 0; i < slabs_.size(); ++i) {
    const Slab& s = slabs_[i];
    if (s.base == nullptr) { os << "  slab " << i << ": <trimmed>\n"; continue; }
    os << "  slab " << i << ": " << human_bytes(s.bytes) << ", carved "
       << human_bytes(s.bump) << ", virgin tail " << human_bytes(s.bytes - s.bump) << '\n';
  }
  std::size_t cached_blocks = 0, cached_bytes = 0;
  for (std::size_t c = 0; c < classes_.size(); ++c) {
    if (classes_[c].empty()) continue;
    cached_blocks += classes_[c].size();
    cached_bytes  += classes_[c].size() * class_block_bytes(c);
    os << "  class " << c << " (" << human_bytes(class_block_bytes(c)) << "): "
       << classes_[c].size() << " cached\n";
  }
  const AllocatorStats st = stats();
  os << "  live=" << live_.size() << " cached_blocks=" << cached_blocks
     << " cached_bytes=" << human_bytes(cached_bytes)
     << " pending=" << pending_.size()
     << " largest_free_block=" << human_bytes(st.largest_free_block)
     << " internal_waste=" << human_bytes(st.internal_waste()) << '\n';
  return os.str();
}

// -----------------------------------------------------------------------------
// validate
// -----------------------------------------------------------------------------

Status FreeListAllocator::validate() const {
  MCKE_RETURN_IF_ERROR(validate_config());

  auto in_a_slab = [&](const void* p) {
    for (const Slab& s : slabs_) {
      if (s.base == nullptr) continue;
      const auto* c = static_cast<const char*>(p);
      const auto* b = static_cast<const char*>(s.base);
      if (c >= b && c < b + s.bump) return true;   // must be within CARVED space
    }
    return false;
  };

  // THE central invariant: live_ / classes_ / pending_ partition every carved
  // block, with no pointer in two of them and none in zero of them.
  std::unordered_map<const void*, int> seen;   // ptr -> bitmask of where we saw it
  std::size_t cached_bytes = 0;

  for (const auto& [ptr, bs] : live_) {
    if (!in_a_slab(ptr))
      return InternalError("freelist: live pointer is not inside any carved slab region");
    if (reinterpret_cast<std::uintptr_t>(ptr) % kDeviceAlignment != 0)
      return InternalError("freelist: live pointer is not 256 B aligned");
    if (class_block_bytes(size_class_of(bs)) != bs)
      return InternalError("freelist: live block size " + std::to_string(bs) +
                           " is not a class block size");
    seen[ptr] |= 1;
  }

  for (std::size_t c = 0; c < classes_.size(); ++c) {
    const std::size_t expect = class_block_bytes(c);
    for (const CachedBlock& blk : classes_[c]) {
      if (blk.bytes != expect)
        return InternalError("freelist: class " + std::to_string(c) + " holds a " +
                             std::to_string(blk.bytes) + " B block but its class size is " +
                             std::to_string(expect));
      if (!in_a_slab(blk.ptr))
        return InternalError("freelist: cached pointer is not inside any carved slab region");
      if ((seen[blk.ptr] & 1) != 0)
        return InternalError("freelist: pointer is BOTH live and cached -- it could be "
                             "handed out twice");
      if ((seen[blk.ptr] & 2) != 0)
        return InternalError("freelist: pointer is cached twice -- it could be handed "
                             "out twice");
      seen[blk.ptr] |= 2;
      cached_bytes += blk.bytes;
    }
  }

  for (const PendingFree& p : pending_) {
    if (!in_a_slab(p.ptr))
      return InternalError("freelist: parked pointer is not inside any carved slab region");
    if (seen[p.ptr] != 0)
      return InternalError("freelist: parked pointer is also live or cached -- a parked "
                           "block must be unreachable until its work completes");
    seen[p.ptr] |= 4;
  }

  std::size_t bypass_bytes = 0;
  for (const auto& [ptr, bytes] : bypassed_) {
    bypass_bytes += bytes;
    if (in_a_slab(ptr))
      return InternalError("freelist: a bypassed pointer lies inside a slab -- it would "
                           "be handed to cudaFree");
  }

  // Byte accounting.
  std::size_t reserved_slabs = 0, carved = 0;
  for (const Slab& s : slabs_) {
    if (s.base == nullptr) continue;
    reserved_slabs += s.bytes;
    carved += s.bump;
    if (s.bump > s.bytes) return InternalError("freelist: bump exceeds slab size");
  }
  if (stats_.bytes_reserved != reserved_slabs + bypass_bytes)
    return InternalError("freelist: bytes_reserved (" + std::to_string(stats_.bytes_reserved) +
                         ") != slabs + bypassed (" +
                         std::to_string(reserved_slabs + bypass_bytes) + ")");

  std::size_t live_bytes = 0, pending_bytes = 0;
  for (const auto& [ptr, bs] : live_) { (void)ptr; live_bytes += bs; }
  for (const PendingFree& p : pending_) pending_bytes += p.bytes;
  if (live_bytes + bypass_bytes != stats_.bytes_in_use)
    return InternalError("freelist: live+bypass bytes (" +
                         std::to_string(live_bytes + bypass_bytes) + ") != bytes_in_use (" +
                         std::to_string(stats_.bytes_in_use) + ")");
  // Every carved byte is accounted for in exactly one of the three places.
  if (live_bytes + cached_bytes + pending_bytes != carved)
    return InternalError("freelist: carved bytes (" + std::to_string(carved) +
                         ") != live + cached + pending (" +
                         std::to_string(live_bytes + cached_bytes + pending_bytes) + ")");
  if (stats_.bytes_in_use < stats_.bytes_requested)
    return InternalError("freelist: bytes_in_use < bytes_requested");
  return OkStatus();
}

}  // namespace mcke
