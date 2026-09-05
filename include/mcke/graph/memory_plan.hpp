// =============================================================================
//  mcke/graph/memory_plan.hpp
//
//  WHAT: Static device-memory planning -- liveness, interference, and byte
//        offsets into ONE arena. Pure host logic. Phase 4.
//
//  ---------------------------------------------------------------------------
//  WHY A PLAN-TIME ARENA RATHER THAN RUNTIME allocate()/deallocate()
//
//  Every intermediate tensor becomes a Tensor::slice() view into one allocation
//  made once, at plan time. run_async() performs ZERO allocator calls. Four
//  reasons, in order of how much they matter:
//
//  1. IT REMOVES A BROKEN DEPENDENCY RATHER THAN PATCHING IT. Storage::note_use()
//     records a single stream, and under a parallel schedule it records an
//     ARBITRARY one -- it is called by the host at ENQUEUE time, and the
//     last-issued consumer routinely finishes first. Feeding that to Phase 2's
//     reuse policies defeats kCoarseStreamPoll AND kPerFreeEvent equally:
//     precision in the reclaim policy buys nothing when its input is wrong. A
//     static arena never asks the question.
//  2. IT MAKES RACE-FREEDOM DECIDABLE. With no dynamic allocation and no
//     data-dependent control flow, the byte assignment and the happens-before
//     relation are both fully known before anything runs -- so "is there a
//     conflicting unordered access pair?" is answerable by exhaustive pairwise
//     check, on a laptop, in milliseconds. That is not decidable for a planner
//     whose behaviour depends on pool state and free timing.
//  3. PEAK BECOMES DETERMINISTIC -- a function of the plan, not of allocator
//     fragmentation. That is what makes the number comparable across policies.
//  4. Zero per-iteration allocator traffic on the replay path.
//
//  Phase 2's pooling allocator is not obsoleted by this; it is EXPLAINED by it.
//  Its job becomes serving the arena, the per-stream workspaces, and graph I/O --
//  which is precisely why production runtimes plan memory, and why a graph
//  runtime drives raw_malloc_calls to a small constant.
//
//  ---------------------------------------------------------------------------
//  THIS IS REGISTER ALLOCATION
//
//  Liveness -> interference graph -> packing, with "spilling" replaced by
//  "grow the arena". Same algorithm, different units. The one thing that does
//  NOT carry over is the notion of a live INTERVAL: a register allocator works
//  on a linear instruction order, where non-overlapping intervals really are
//  disjoint in time. Here the executor runs several streams at once, so
//  interference must be asked of the happens-before relation, not of interval
//  overlap. Getting that wrong is the central Phase 4 trap; see
//  graph/happens_before.hpp.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mcke/core/status.hpp"
#include "mcke/graph/executor.hpp"
#include "mcke/graph/graph.hpp"
#include "mcke/graph/happens_before.hpp"
#include "mcke/graph/schedule.hpp"

namespace mcke {

inline constexpr std::size_t kNoOffset = static_cast<std::size_t>(-1);

struct MemoryPlan {
  MemoryPolicy policy = MemoryPolicy::kReuseHappensBefore;

  // Indexed by TensorId. kNoOffset for a tensor with no device buffer.
  std::vector<std::size_t> offset_of;
  std::vector<std::size_t> bytes_of;    // PADDED to kDeviceAlignment

  std::size_t arena_bytes = 0;   // what the packing achieved
  std::size_t naive_bytes = 0;   // one buffer per tensor, no reuse

  // Where the bytes go, so a RESULTS row is interpretable. Graph inputs and
  // outputs are in the arena but are NEVER reuse candidates, so a graph
  // dominated by them shows a poor ratio for a good reason.
  std::size_t input_bytes = 0, intermediate_bytes = 0, output_bytes = 0;

  // One workspace arena per stream, sized by the largest consumer on it.
  // Same-stream in-order issue makes sharing it safe with no analysis at all.
  std::vector<std::size_t> workspace_bytes;

  std::size_t buffers_used = 0;   // distinct offsets handed out

  [[nodiscard]] std::string describe() const;
};

// Plan the arena.
//
// TAKES THE HappensBefore BY VALUE-LIKE CONST REF ON PURPOSE, mirroring
// GraphExecutor::assign_memory's signature: the relation is derived from the
// STREAM ASSIGNMENT, so streams must be assigned first. Requiring it as an
// argument makes the wrong order fail to compile rather than silently plan
// against a default-initialised schedule where every node sits on stream 0 --
// which is exactly kReuseTopoNaive's bug, reintroduced by accident.
[[nodiscard]] StatusOr<MemoryPlan> plan_memory(const Graph& g,
                                               const StreamAssignment& sa,
                                               const HappensBefore& hb,
                                               MemoryPolicy policy);

// Do any two tensors sharing bytes actually fail to be ordered?
//
// The exhaustive pairwise check that the static-arena choice makes possible: for
// every pair of tensors whose assigned byte ranges overlap, assert that one's
// accesses all happen-before the other's producer. Returns a Status naming the
// offending pair and the unordered nodes.
//
// This is the memory-level race checker. It is deliberately separate from
// verify_plan_ordering (which validates the SCHEDULE alone), so a failure here
// is unambiguously a planner failure rather than "something upstream is wrong".
[[nodiscard]] Status verify_no_buffer_races(const Graph& g, const StreamAssignment& sa,
                                            const HappensBefore& hb,
                                            const MemoryPlan& mp);

}  // namespace mcke
