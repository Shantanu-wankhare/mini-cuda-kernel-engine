// =============================================================================
//  mcke/graph/cost_model.hpp
//
//  WHAT: A roofline lower bound on how long a node should take. Pure host
//        arithmetic over an OpCost and a Roofline -- no CUDA, no device, no
//        graph. Phase 4.
//
//  ---------------------------------------------------------------------------
//  IT HAS TWO CALLERS, AND THAT IS THE POINT
//
//  1. kChainGreedy's load balancer. "Assign the least-loaded stream" needs a
//     notion of load, and NODE COUNT IS THE WRONG ONE. Every benchmark graph in
//     Phase 4 mixes a ~10 ms GEMM with a ~0.2 ms softmax; balancing those by
//     count puts 50x more work on one stream than the other while reporting a
//     perfectly balanced schedule.
//  2. The whole-graph efficiency denominator in RESULTS.md sec 4.
//
//  Sharing one implementation between them is deliberate: it means the
//  scheduler's decisions can be predicted offline, by hand, from the same
//  numbers the results table reports. If the balancer used one cost notion and
//  the table another, a badly balanced schedule would still look efficient.
//
//  ---------------------------------------------------------------------------
//  WHY THERE IS NO GRAPH-LEVEL "% OF PEAK"
//
//  A graph that mixes a compute-bound GEMM (AI ~ 683) with a memory-bound
//  softmax (AI ~ 2.5) has NO single roofline -- the two nodes are limited by
//  different hardware. Dividing whole-graph FLOPs by peak TFLOP/s would report a
//  transformer block at a few percent of peak and invite "optimising" the
//  softmax, which is already at 100% of the roof that actually binds it.
//
//  So the defensible whole-graph number is not a percentage but a RATIO:
//
//      sum over nodes of max(flops/peak_flops, bytes/peak_bw)   <- predicted
//      ------------------------------------------------------
//                     measured wall time
//
//  Each node is judged against the roof that binds IT, and the sum is a genuine
//  lower bound for a sequential schedule. A ratio near 1.0 means the graph runs
//  as fast as its parts allow; below 1.0 means overlap beat the sequential bound
//  (which is exactly what a parallel policy is trying to do, and is why this is a
//  ratio rather than an efficiency).
// =============================================================================
#pragma once

#include <algorithm>
#include <cstdint>

#include "mcke/graph/op.hpp"
#include "mcke/profiling/profiler.hpp"

namespace mcke {

// The roofline lower bound for one op, in milliseconds: it cannot go faster than
// its FLOPs allow, and it cannot go faster than its compulsory traffic allows,
// so it takes at least the larger of the two.
//
// Returns 0 for a zero-cost op. Guards both denominators because a Roofline with
// a zero field is exactly the silent-garbage case bench_common.hpp's
// make_roofline() aborts on -- here we would rather return an obviously useless
// 0 than an infinity that propagates into a schedule.
[[nodiscard]] inline double plan_cost_ms(const OpCost& c, const Roofline& rl) {
  const double t_compute =
      rl.peak_tflops > 0.0 ? static_cast<double>(c.flops) / (rl.peak_tflops * 1e12) : 0.0;
  const double t_memory =
      rl.peak_gb_s > 0.0 ? static_cast<double>(c.bytes) / (rl.peak_gb_s * 1e9) : 0.0;
  return std::max(t_compute, t_memory) * 1e3;
}

// True when the memory roof binds this op -- i.e. its arithmetic intensity is
// below the ridge point. Same question Roofline::memory_bound() answers, asked
// about an OpCost rather than a measured KernelRecord, so it can be asked at
// PLAN time before anything has run.
[[nodiscard]] inline bool op_is_memory_bound(const OpCost& c, const Roofline& rl) {
  return rl.memory_bound(c.arithmetic_intensity());
}

}  // namespace mcke
