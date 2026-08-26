// =============================================================================
//  mcke/graph/executor.hpp
//
//  WHAT: `ExecutionPlan` (the compiled, immutable schedule) and `GraphExecutor`
//        (the thing that replays it). Phase 4.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — separate plan() from run().
//
//  Planning does: topological sort, stream assignment, event insertion, memory
//  planning (liveness + buffer assignment), workspace sizing. All of it is host
//  work that depends only on the graph *structure*, not on the data. So we do it
//  once and replay the plan N times.
//
//  This is not just tidiness — it is the entire performance argument for a
//  graph runtime versus eager execution. In eager mode every iteration pays
//  shape inference, allocation, and launch bookkeeping on the CPU. For small
//  kernels (< 20 us) that CPU work becomes the bottleneck: the GPU finishes
//  before the CPU can enqueue the next op, and you see low SM utilisation with
//  a busy CPU. Precomputing the plan reduces per-iteration host work to "issue
//  K launches with pre-baked arguments".
//
//  The logical end point of this idea is CUDA Graphs (cudaGraphLaunch), which
//  captures the launches themselves into a driver-side structure and cuts
//  per-launch CPU cost from ~5 us to ~0.5 us. Our ExecutionPlan is deliberately
//  shaped so that `capture_cuda_graph()` is a natural Phase 7 addition rather
//  than a rewrite.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — how streams get assigned.
//
//  Three policies, all implemented, all benchmarked (this comparison is the
//  Phase 4 deliverable):
//
//   kSequential   : one stream, topological order. The correctness baseline.
//                   Every other policy must produce bit-identical outputs.
//   kLevelParallel: round-robin the nodes of each topological level over K
//                   streams; barrier between levels. Simple, and captures most
//                   of the available overlap on wide graphs. Weakness: the
//                   inter-level barrier serialises a long node against short
//                   ones.
//   kChainGreedy  : walk the topo order; keep a node on its predecessor's
//                   stream when it has exactly one predecessor (so no event is
//                   needed at all — same-stream ordering is free), otherwise
//                   assign the least-loaded stream and insert events for each
//                   cross-stream predecessor. This minimises event count, which
//                   matters: each cudaStreamWaitEvent + cudaEventRecord pair
//                   costs ~1-2 us of CPU time and creates a real GPU-side
//                   dependency the scheduler must honour.
//
//  Expected result to verify, not assume: on a diamond graph
//  (A -> {B,C} -> D) with B,C each ~1 ms, kSequential ~2 ms and the parallel
//  policies ~1 ms *if and only if* B and C individually leave SMs idle. If each
//  already saturates the GPU, overlap buys nothing — and being able to explain
//  that non-result is more valuable than the speedup.
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mcke/core/status.hpp"
#include "mcke/graph/graph.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/stream.hpp"
#include "mcke/tensor/tensor.hpp"

namespace mcke {

enum class SchedulePolicy : std::uint8_t { kSequential, kLevelParallel, kChainGreedy };

// How the planner assigns device memory to intermediate tensors.
enum class MemoryPolicy : std::uint8_t {
  kAllocPerTensor,   // one allocator call per tensor; simple, high peak usage
  kReuseByLiveness   // linear-scan reuse of dead buffers; lower peak
};

struct ExecutorOptions {
  SchedulePolicy schedule     = SchedulePolicy::kChainGreedy;
  MemoryPolicy   memory       = MemoryPolicy::kReuseByLiveness;
  int            num_streams  = 4;    // > graph width is pointless; measure it
  bool           profile      = false;// insert per-node timing events + NVTX ranges
  bool           validate_numerics = false;  // compare against kSequential run
};

// One entry per node, precomputed. Deliberately a flat POD array: replaying the
// plan should be a tight loop over contiguous memory with no pointer chasing,
// because on small graphs this loop *is* the critical path.
struct ScheduledNode {
  NodeId               node   = kInvalidNode;
  std::uint16_t        stream_idx = 0;
  // Events this node must wait on before launching (cross-stream predecessors).
  std::vector<std::uint16_t> wait_events;
  // Event to record after launching, iff some successor is on another stream.
  // -1 = no successor needs it, so we skip the record entirely. Not recording
  // unnecessary events is a real optimisation on graphs with hundreds of nodes.
  int                  record_event = -1;
};

class ExecutionPlan {
 public:
  [[nodiscard]] const std::vector<ScheduledNode>& nodes() const noexcept { return sched_; }
  [[nodiscard]] std::size_t num_streams() const noexcept { return streams_.size(); }
  [[nodiscard]] std::size_t peak_memory_bytes() const noexcept { return peak_bytes_; }
  // How much the liveness-based planner saved versus naive per-tensor allocation.
  [[nodiscard]] std::size_t naive_memory_bytes() const noexcept { return naive_bytes_; }
  [[nodiscard]] std::string describe() const;   // for PROJECT_LOG.md entries

 private:
  friend class GraphExecutor;
  std::vector<ScheduledNode> sched_;
  std::vector<rt::Stream>    streams_;
  std::vector<rt::Event>     events_;
  // Bound tensors, indexed by TensorId. Intermediates may share Storage.
  std::vector<Tensor>        bound_;
  std::size_t                peak_bytes_  = 0;
  std::size_t                naive_bytes_ = 0;
};

class GraphExecutor {
 public:
  GraphExecutor(Graph graph, DeviceAllocator& alloc, DeviceInfo device,
                ExecutorOptions opts = {});
  ~GraphExecutor();

  // Compile the graph: sort, schedule, assign memory, create streams/events.
  // Idempotent; safe to call again after changing options.
  [[nodiscard]] Status plan();

  // Bind a host buffer to a graph input (async H2D on the input's stream).
  [[nodiscard]] Status set_input(TensorId t, const void* host_data, std::size_t bytes);

  // Enqueue the whole graph. Returns immediately — nothing has necessarily run.
  // This is the API that makes the runtime asynchronous, and the reason
  // `run_async` + explicit `synchronize` is the right shape: it lets the caller
  // enqueue iteration N+1 while N is still executing.
  [[nodiscard]] Status run_async();

  // Wait for every stream in the plan. Exactly one of the two places in the
  // runtime where a host-side barrier is allowed.
  [[nodiscard]] Status synchronize();

  [[nodiscard]] Status run_sync() {
    MCKE_RETURN_IF_ERROR(run_async());
    return synchronize();
  }

  [[nodiscard]] StatusOr<Tensor> output(TensorId t) const;
  [[nodiscard]] const ExecutionPlan& plan_ref() const { return plan_; }
  [[nodiscard]] const Profiler& profiler() const { return profiler_; }

 private:
  [[nodiscard]] Status assign_streams(const std::vector<NodeId>& order);
  [[nodiscard]] Status assign_memory(const std::vector<NodeId>& order);

  Graph            graph_;
  DeviceAllocator& alloc_;
  DeviceInfo       device_;
  ExecutorOptions  opts_;
  ExecutionPlan    plan_;
  Profiler         profiler_;
  bool             planned_ = false;
};

}  // namespace mcke
