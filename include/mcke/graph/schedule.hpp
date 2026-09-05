// =============================================================================
//  mcke/graph/schedule.hpp
//
//  WHAT: Stream assignment and event insertion for the three schedule policies.
//        Pure host logic over a Graph, a policy, a stream count and a cost
//        model. No CUDA, no allocator, no Tensor. Phase 4.
//
//  ---------------------------------------------------------------------------
//  WHY A FREE FUNCTION AND NOT JUST GraphExecutor::assign_streams
//
//  Same reason happens_before.hpp is separate: everything decided here is
//  decidable without a device, so it should be assertable without one. Which
//  stream each node lands on, how many events get recorded, how many waits
//  survive deduplication -- these are exact integers, and they are the headline
//  numbers of RESULTS.md sec 4. Producing them from a free function means they
//  can be checked on a laptop, deterministically, with no timing noise, instead
//  of being inferred afterwards from a profiler trace.
//
//  It also produces EXACTLY the arrays HappensBefore::build consumes, so the
//  planner and the race checker are handed the same schedule description rather
//  than two reconstructions of it.
//
//  ---------------------------------------------------------------------------
//  DEAD NODES AND THE INDEX SPACE
//
//  This operates on the LIVE subgraph only -- a node that cannot reach any graph
//  output must not be scheduled, or every benchmark number is inflated by work
//  nobody asked for. That makes NodeId the wrong index: dead nodes would leave
//  holes, and HappensBefore requires per-stream issue indices to be DENSE (a gap
//  silently weakens the relation, so it rejects them).
//
//  So everything below is indexed by SCHEDULED INDEX 0..S-1, and `nodes[i]` maps
//  back to the NodeId. Mixing the two index spaces is the obvious way to get a
//  subtly wrong schedule, so they are named differently throughout.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mcke/core/status.hpp"
#include "mcke/graph/graph.hpp"
#include "mcke/graph/happens_before.hpp"
#include "mcke/profiling/profiler.hpp"

namespace mcke {

enum class SchedulePolicy : std::uint8_t { kSequential, kLevelParallel, kChainGreedy };

// Everything the executor and the checkers need to know about a schedule.
// Indexed by SCHEDULED INDEX (see the banner), not NodeId.
struct StreamAssignment {
  std::vector<NodeId>        nodes;       // scheduled index -> NodeId
  std::vector<std::uint16_t> stream_of;
  std::vector<std::uint32_t> issue_of;    // position within that node's stream
  std::vector<std::uint32_t> record_of;   // event id recorded after this node, or
                                          // kNoEventId when no successor needs it
  std::vector<std::vector<std::uint32_t>> waits_of;   // event ids, deduplicated
  std::vector<std::uint32_t> order;       // scheduled indices, in issue order

  std::size_t num_streams_used = 0;   // may be < requested; report both or the
                                      // RESULTS row claims concurrency it lacks
  std::size_t num_streams_requested = 0;
  std::size_t num_events   = 0;       // == records, one per recording node
  std::size_t waits_raw    = 0;       // before per-stream deduplication
  std::size_t waits_dedup  = 0;       // what actually gets issued

  // Estimated cost landing on each stream, from cost_model.hpp. Balancing by
  // node COUNT is wrong on every graph this project benchmarks.
  std::vector<double> stream_load_ms;

  [[nodiscard]] std::string describe() const;

  // The arrays above are exactly HappensBefore::build's parameters.
  [[nodiscard]] StatusOr<HappensBefore> happens_before() const;
};

// The three policies.
//
//   kSequential    one stream, topological order. 0 events. The correctness
//                  baseline every other policy is compared against, bit for bit.
//
//   kLevelParallel round-robin each level's nodes over K streams, with a TRUE
//                  barrier between consecutive levels. The barrier is
//                  implemented as a barrier and not weakened to per-edge waits:
//                  its cost IS the point of the policy, and softening it would
//                  turn it into chain-greedy-without-the-heuristic and destroy
//                  the comparison.
//
//   kChainGreedy   keep a node on its single predecessor's stream when that
//                  predecessor has not already donated it, else least-loaded
//                  stream plus events. See the corrected rule in executor.hpp:
//                  omitting the donation clause collapses the diamond to one
//                  stream and publishes a scheduler bug as a 1.00x result.
[[nodiscard]] StatusOr<StreamAssignment> plan_streams(const Graph& g,
                                                      SchedulePolicy policy,
                                                      int num_streams,
                                                      const Roofline& rl);

// Independent scheduler check: for every dependency edge (p -> n) in the LIVE
// subgraph, either p and n share a stream with p issued first, or n waits
// (transitively) on an event recorded at or after p.
//
// Deliberately separate from the memory-level race checker: this validates
// assign_streams ALONE, so a race-checker failure is unambiguously a memory
// planner failure rather than "something in the schedule is wrong".
[[nodiscard]] Status verify_plan_ordering(const Graph& g, const StreamAssignment& sa);

}  // namespace mcke
