// =============================================================================
//  mcke/memory/reuse_policy.hpp
//
//  WHAT: `ReusePolicy` — how a pooling allocator proves that a freed block is
//        safe to hand to a DIFFERENT stream than the one that last used it —
//        plus the single shared decision function every pool must route through.
//
//  ---------------------------------------------------------------------------
//  WHY THIS IS ITS OWN HEADER, extracted out of buddy_allocator.hpp
//
//  Phase 2c benchmarks BuddyAllocator against FreeListAllocator on one identical
//  allocation trace. If each allocator carried its own copy of the "is this
//  parked block reusable yet?" logic, then any subtle divergence between the two
//  copies — an inverted rule-1 test, a policy case handled differently — would
//  show up in the results as an allocator-design difference. It would be
//  indistinguishable from the thing we are actually trying to measure.
//
//  So the *semantics* live here, in one function both pools call, and only the
//  storage and the "what to do once it IS reusable" action stay local (buddy
//  returns the node to its tree and coalesces; the free-list pool pushes the
//  pointer onto a size-class list). Shared where they must agree, separate where
//  they legitimately differ.
//
//  This is a general lesson about benchmark construction: anything two subjects
//  must have in common should be physically shared, not carefully duplicated.
// =============================================================================
#pragma once

#include <cstdint>

#include "mcke/runtime/stream.hpp"

namespace mcke {

// Sentinel for "this parked block has no event associated with it".
inline constexpr std::uint32_t kNoEventSlot = 0xFFFFFFFFu;

// -----------------------------------------------------------------------------
// The shared foundation, which NO policy has to pay for, is RULE 1:
//
//   A block freed on stream S is immediately reusable by a later allocation on
//   S, because a stream is in-order — the consumer kernel was enqueued before
//   whatever the caller is about to enqueue. No synchronisation, no probe.
//
// Note carefully that rule 1 cannot be evaluated inside deallocate(): the
// last-use stream is deallocate's argument, but the *reusing* stream is
// allocate's. Two different calls. That is why this decision lives on the
// reclaim path and takes both streams.
//
// The policies differ only in how they handle the cross-stream case.
// -----------------------------------------------------------------------------
enum class ReusePolicy : std::uint8_t {
  // Rule 1 and nothing else. Never probes for completion, so a free costs one
  // push_back and zero driver calls. The consequence is that capacity becomes
  // *stream-affine*: a block freed on stream A is reusable only by A (or by an
  // explicit drain / trim). In a multi-stream workload where one stream frees and
  // another allocates, this parks memory indefinitely and grows slabs it did not
  // need. Cheapest per free, worst capacity behaviour.
  kSameStreamOnly,

  // Rule 1, plus: reclaim when the freeing stream reports fully idle.
  //
  // SOUND — a stream reporting idle certainly implies that block's consumer
  // finished — but conservative, and the conservatism is a LIVENESS risk rather
  // than a safety one. A long kernel enqueued into S *after* the free keeps the
  // block parked even though its own consumer is done. Under continuous
  // submission (exactly what a graph executor produces) a stream may never report
  // idle, so blocks park indefinitely and the pool grows. Callers must bound that
  // with a counted blocking drain, so the failure mode is a measured stall rather
  // than silent unbounded growth.
  kCoarseStreamPoll,

  // Rule 1, plus: record an event at free time and reclaim when *that block's*
  // event completes. Precise, immune to the liveness problem above, and the only
  // policy that can release one block while a sibling freed on the same stream is
  // still in flight. Pays a cudaEventRecord (~1 us of CPU) on every free, which
  // is the same order as the pool hit it is trying to make fast — so whether the
  // trade is worth it is an empirical question, not an obvious one.
  kPerFreeEvent,
};

[[nodiscard]] constexpr const char* to_string(ReusePolicy p) noexcept {
  switch (p) {
    case ReusePolicy::kSameStreamOnly:   return "same_stream_only";
    case ReusePolicy::kCoarseStreamPoll: return "coarse_stream_poll";
    case ReusePolicy::kPerFreeEvent:     return "per_free_event";
  }
  return "unknown";
}

// -----------------------------------------------------------------------------
// THE shared decision. Every pooling allocator must route through this.
//
//   entry_stream : the stream the block was last used on (recorded at free time)
//   event_slot   : the block's event slot, or kNoEventSlot
//   for_stream   : the stream the pending allocation will run on
//   stream_probe : bool(rt::StreamHandle) — "is this whole stream idle?"
//   event_probe  : bool(std::uint32_t)    — "has this block's event completed?"
//
// Templated on the two probes rather than taking std::function so the call
// inlines and each allocator can route its probe through its own test seam
// (a protected virtual) without this header knowing anything about that.
// -----------------------------------------------------------------------------
template <typename StreamProbe, typename EventProbe>
[[nodiscard]] bool pending_reusable(ReusePolicy policy,
                                    rt::StreamHandle entry_stream,
                                    std::uint32_t event_slot,
                                    rt::StreamHandle for_stream,
                                    StreamProbe&& stream_probe,
                                    EventProbe&& event_probe) {
  // Rule 1, checked first because it is free under every policy.
  if (entry_stream == for_stream) return true;

  switch (policy) {
    case ReusePolicy::kSameStreamOnly:
      // Deliberately no fallback: cross-stream reuse waits for an explicit drain.
      return false;
    case ReusePolicy::kCoarseStreamPoll:
      return stream_probe(entry_stream);
    case ReusePolicy::kPerFreeEvent:
      return event_probe(event_slot);
  }
  return false;
}

}  // namespace mcke
