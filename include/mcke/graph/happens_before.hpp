// =============================================================================
//  mcke/graph/happens_before.hpp
//
//  WHAT: The happens-before relation of a scheduled plan, as vector clocks.
//        Phase 4. Pure host arithmetic over integers -- no CUDA, no allocator,
//        no Graph, no ExecutionPlan.
//
//  ---------------------------------------------------------------------------
//  WHY THIS EXISTS AT ALL: TOPOLOGICAL ORDER IS THE WRONG RELATION
//
//  A memory planner wants to know whether two tensors are ever alive at the same
//  time. The obvious answer -- compare their [def, last_use] intervals in the
//  topological order -- is WRONG the moment more than one stream is in play, and
//  wrong in the worst way: silently, only under load, and only on some policies.
//
//  The precise statement:
//
//      Topological order is ONE ARBITRARY LINEAR EXTENSION of the dependency
//      partial order. Happens-before is the dependency partial order PLUS
//      STREAM-SERIALISATION EDGES. Both are supersets of the dependencies and
//      subsets of nothing in common -- they extend the same partial order in
//      DIFFERENT directions. So interval non-overlap in a topological order
//      asserts an ordering the executor never established.
//
//  The gap `topo \ deps` is Kahn's arbitrary tie-breaking. The gap `hb \ deps` is
//  the stream assignment. There is no reason for them to agree.
//
//  Minimal graph where this races (verified by hand before any code was written):
//
//      X (input)
//      N0: a = f(X)          N2: b = g(X)
//      N1: c = h(a)          N3: d = k(b)
//              N4: out = m(c, d)
//
//  FIFO-Kahn order N0,N2,N1,N3,N4. Live ranges by position: a = [0,2],
//  d = [3,4] -- DISJOINT, so a linear-scan planner gives `d` the buffer that
//  held `a`. But under kChainGreedy, N0/N1 land on stream 0 and N2/N3 on stream
//  1, with events only at N4: nothing orders the two streams before N4, so N3
//  writes the buffer while N0/N1 still use it. Under kLevelParallel the
//  inter-level barrier does order N0 before N3 -- yet N1 and N3 are in the SAME
//  LEVEL on different streams, so N1 reads while N3 writes. The barrier does not
//  save you. Only kSequential is safe, and it is safe by accident of totality.
//
//  ---------------------------------------------------------------------------
//  THE REUSE CONDITION
//
//  Buffer B holding tensor T1 may be reused for T2 iff
//
//      for every u in ({producer(T1)} union consumers(T1)) :  u  <  producer(T2)
//
//  Two things to notice. First, the left-hand side is a SET, not a node: under a
//  parallel schedule T1's consumers may be mutually unordered, so there is no
//  single "last use", and any formulation that computes a scalar last_use and
//  compares it has already lost. Second, only producer(T2) needs to be
//  dominated -- SSA gives one producer, and every consumer of T2 happens-after
//  it by construction -- so the test is |accesses(T1)| checks, not a cross
//  product.
//
//  Under one stream the relation is total and this degenerates EXACTLY to the
//  naive interval test. So there is one planner, not two code paths, and that
//  equivalence is asserted in a host test.
//
//  ---------------------------------------------------------------------------
//  WHY IT IS A SEPARATE, SHARED HEADER
//
//  Both the memory planner and the host-side race checker need this relation. If
//  each derived it independently, a shared misconception would make the checker
//  VACUOUS -- it would confirm the planner's own belief rather than test it. So
//  the relation is computed in exactly one place and both callers ask it
//  questions. This is the same argument mcke/memory/reuse_policy.hpp already
//  makes for Phase 2's three cross-stream reuse policies, applied a second time.
//
//  It deliberately does NOT depend on ExecutionPlan or ScheduledNode -- it takes
//  plain integer arrays. That keeps it unit-testable from literals with no
//  executor, no graph and no device, and it is what lets executor.hpp include
//  this header rather than the reverse.
//
//  ---------------------------------------------------------------------------
//  THE ALGORITHM (Fidge-Mattern vector clocks)
//
//  clock[n][s] = the highest issue index on stream s that node n is guaranteed
//  to happen after. Walking nodes in the plan's issue order:
//
//      c = componentwise max of ( clock[same-stream predecessor],
//                                 clock[recorder of each event n waits on] )
//      c[stream(n)] = issue(n)
//
//  Then  u < v   <=>   clock[v][stream(u)] >= issue(u).
//
//  O(N*S) time and memory with S = 4..8 and N in the hundreds: microseconds.
//  This is the same primitive a happens-before data-race detector uses, which is
//  why it pays for itself twice -- see tests/ for the checker built on it.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mcke/core/status.hpp"

namespace mcke {

// Sentinel for "this node records no event" / "this event has no recorder".
inline constexpr std::uint32_t kNoEventId = 0xFFFFFFFFu;

class HappensBefore {
 public:
  // Inputs are per-node parallel arrays, deliberately primitive:
  //   stream_of[n]  which stream node n was placed on
  //   issue_of[n]   n's position WITHIN that stream (0,1,2,... per stream)
  //   record_of[n]  the event id n records after launching, or kNoEventId
  //   waits_of[n]   event ids n waits on before launching
  //   order         global issue order; must be a linear extension of the
  //                 dependency edges, i.e. the topological order the planner used
  [[nodiscard]] static StatusOr<HappensBefore> build(
      std::size_t num_streams,
      const std::vector<std::uint16_t>& stream_of,
      const std::vector<std::uint32_t>& issue_of,
      const std::vector<std::uint32_t>& record_of,
      const std::vector<std::vector<std::uint32_t>>& waits_of,
      const std::vector<std::uint32_t>& order);

  // Strict happens-before. False when u == v: a node does not precede itself,
  // and letting it would make every tensor trivially reusable by its own
  // producer.
  [[nodiscard]] bool precedes(std::uint32_t u, std::uint32_t v) const {
    if (u == v) return false;
    return clock_[v * num_streams_ + stream_of_[u]] >= static_cast<std::int64_t>(issue_of_[u]);
  }

  // Neither orders the other: they may execute simultaneously. This is the
  // predicate the race checker asks about every pair of conflicting accesses.
  [[nodiscard]] bool concurrent(std::uint32_t u, std::uint32_t v) const {
    return u != v && !precedes(u, v) && !precedes(v, u);
  }

  [[nodiscard]] std::size_t num_nodes() const noexcept { return issue_of_.size(); }
  [[nodiscard]] std::size_t num_streams() const noexcept { return num_streams_; }

  // For diagnostics: the reason a reuse was refused, or a race was reported.
  [[nodiscard]] std::string describe_pair(std::uint32_t u, std::uint32_t v) const;

 private:
  HappensBefore() = default;

  std::size_t                num_streams_ = 0;
  std::vector<std::uint16_t> stream_of_;
  std::vector<std::uint32_t> issue_of_;
  // Flat num_nodes x num_streams matrix. -1 means "no knowledge of that stream".
  std::vector<std::int64_t>  clock_;
};

}  // namespace mcke
