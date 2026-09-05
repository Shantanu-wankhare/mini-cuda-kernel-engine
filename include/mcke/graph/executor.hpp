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
//                   stream when it has exactly one predecessor AND THAT
//                   PREDECESSOR HAS NOT ALREADY DONATED ITS STREAM (i.e. this
//                   node is the predecessor's first successor in topo order),
//                   so no event is needed at all — same-stream ordering is
//                   free. Otherwise assign the least-loaded stream and insert
//                   events for each cross-stream predecessor.
//
//                   THE SECOND CLAUSE IS NOT A REFINEMENT, IT IS THE FIX. An
//                   earlier version of this comment said only "when it has
//                   exactly one predecessor" — and on the diamond
//                   A -> {B,C} -> D, *both* B and C have exactly one
//                   predecessor. The literal rule therefore puts both on A's
//                   stream, producing a single-stream schedule with zero
//                   events and a RESULTS.md row reading "diamond /
//                   chain_greedy / 1.00x" that looks like an honest negative
//                   result and is really a scheduler bug. The exact
//                   stream_idx array is asserted in a host test for this
//                   reason: diamond/chain_greedy must be [0,0,1,.], not
//                   [0,0,0,0].
//
//                   "Least-loaded" is by ESTIMATED COST (see
//                   graph/cost_model.hpp), not by node count. Balancing a 10 ms
//                   GEMM against a 0.2 ms softmax by count is badly wrong on
//                   every graph this project benchmarks.
//
//                   On event count: each cudaStreamWaitEvent + cudaEventRecord
//                   pair costs ~1-2 us of CPU time and creates a real GPU-side
//                   dependency the scheduler must honour. The header used to
//                   claim chain-greedy "minimises" event count; the defensible
//                   statement is that its count scales with the number of
//                   CROSS-STREAM EDGES, while level-parallel's scales with
//                   LEVEL BOUNDARIES x STREAMS USED. On a chain that is 0 vs
//                   15; on the diamond they tie at 2. It is a greedy heuristic
//                   that happens to be optimal on the graphs we benchmark, not
//                   a proven minimum.
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
#include "mcke/graph/happens_before.hpp"
#include "mcke/graph/memory_plan.hpp"
#include "mcke/graph/schedule.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/stream.hpp"
#include "mcke/tensor/tensor.hpp"

namespace mcke {

// SchedulePolicy moved to mcke/graph/schedule.hpp (included below), next to
// plan_streams() and StreamAssignment -- it is the schedule's vocabulary, and
// keeping it here would have made schedule.hpp depend on the executor it is
// meant to be testable without.

// MemoryPolicy moved to mcke/graph/memory_plan.hpp (included below), next to
// plan_memory() -- same reasoning as SchedulePolicy: it is the planner's
// vocabulary, and leaving it here made memory_plan.hpp include the executor
// it is meant to be testable without, which is a circular include.

struct ExecutorOptions {
  SchedulePolicy schedule     = SchedulePolicy::kChainGreedy;
  MemoryPolicy   memory       = MemoryPolicy::kReuseHappensBefore;
  int            num_streams  = 4;    // > graph width is pointless; measure it
  bool           profile      = false;// insert per-node timing events + NVTX ranges
  int            profile_slots = 32;  // probe-ring depth; must be >= timed iterations
  bool           validate_numerics = false;  // compare against kSequential run
  // Fill every planned buffer with a distinctive pattern before each run.
  //
  // The highest-value debug affordance in the phase for its size. An op that
  // reads memory it did not write produces a DIFFERENT wrong answer per pattern,
  // which separates "the scheduler races" from "an op reads uninitialised
  // memory" -- different bugs with different fixes, and otherwise very hard to
  // tell apart from a bit-identity failure alone.
  bool           poison_buffers = false;
  std::uint32_t  poison_pattern = 0x7F7F7F7Fu;   // a finite float, deliberately:
                                                 // a NaN can be swallowed by a
                                                 // max() and mask the very read
                                                 // this is meant to expose
};

// Per-node timing, filled by collect_timings() after synchronize().
struct NodeTiming {
  NodeId        node = kInvalidNode;
  std::string   name;
  int           stream_idx = 0;
  double        median_ms  = 0.0;
  double        min_ms     = 0.0;
  std::uint64_t flops = 0, bytes = 0;
  int           samples = 0;
};

inline constexpr std::uint32_t kNoEvent = 0xFFFFFFFFu;

// One entry per node, precomputed. A flat POD: replaying the plan should be a
// tight loop over contiguous memory with no pointer chasing, because on small
// graphs this loop *is* the critical path.
//
// THE CLAIM ABOVE USED TO BE FALSE. This struct held a std::vector<uint16_t> for
// its wait list, so it was not trivially copyable (verified: sizeof 40,
// is_trivially_copyable false) and every node replay dereferenced a heap
// allocation -- exactly the pointer chasing the comment disclaimed. The waits now
// live in one plan-level pool and each node carries an (offset, count) into it,
// which makes the struct genuinely trivially copyable and the claim genuinely
// true. It is also a real, measurable difference on a launch-bound graph, which
// is one of the things this phase measures.
struct ScheduledNode {
  NodeId        node        = kInvalidNode;
  std::uint16_t stream_idx  = 0;
  std::uint16_t issue_index = 0;   // position within this node's own stream;
                                   // one half of the happens-before relation
  // Slice of ExecutionPlan::wait_pool() -- the events this node must wait on
  // before launching (its cross-stream predecessors).
  std::uint32_t wait_offset = 0;
  std::uint32_t wait_count  = 0;
  // Event to record after launching, iff some successor is on another stream.
  // kNoEvent = no successor needs it, so we skip the record entirely. Deciding
  // this requires a SECOND pass, after every stream is assigned: whether p must
  // record depends on p's successors' streams, which are unknown while p itself
  // is being placed. A one-pass implementation over-records, which is easy to
  // miss because it is merely wasteful rather than wrong.
  std::uint32_t record_event = kNoEvent;
};

class ExecutionPlan {
 public:
  [[nodiscard]] const std::vector<ScheduledNode>& nodes() const noexcept { return sched_; }
  [[nodiscard]] std::size_t num_streams() const noexcept { return streams_.size(); }
  [[nodiscard]] std::size_t peak_memory_bytes() const noexcept { return peak_bytes_; }
  // How much the liveness-based planner saved versus naive per-tensor allocation.
  //
  // CAREFUL: peak_bytes_ is a function of (graph, SCHEDULE policy, memory
  // policy, NUM_STREAMS) -- not of (graph, memory policy) as this pair of
  // accessors suggests. Concurrency destroys reuse opportunities, so peak is
  // predicted to rise monotonically with num_streams under kReuseHappensBefore.
  // describe() must therefore print all four coordinates, or a RESULTS.md row
  // is uninterpretable six months later.
  [[nodiscard]] std::size_t naive_memory_bytes() const noexcept { return naive_bytes_; }
  // Backing store for every ScheduledNode's wait slice.
  [[nodiscard]] const std::vector<std::uint32_t>& wait_pool() const noexcept {
    return wait_pool_;
  }
  [[nodiscard]] std::string describe() const;   // for PROJECT_LOG.md entries
  [[nodiscard]] const StreamAssignment& stream_plan() const noexcept { return streams_plan_; }
  [[nodiscard]] const MemoryPlan& memory_plan() const noexcept { return memory_plan_; }
  // Intra-graph events only. The per-iteration fork/join adds
  // num_streams + 2*(num_streams-1) more; they are reported separately because
  // folding them together would misattribute a fixed overhead to the policy.
  [[nodiscard]] std::size_t intra_events() const noexcept { return streams_plan_.num_events; }
  [[nodiscard]] std::size_t intra_waits() const noexcept { return streams_plan_.waits_dedup; }
  [[nodiscard]] std::size_t forkjoin_events() const noexcept {
    const std::size_t k = streams_plan_.num_streams_used;
    return k <= 1 ? 0 : k + 2 * (k - 1);
  }

 private:
  friend class GraphExecutor;
  std::vector<ScheduledNode>  sched_;
  std::vector<std::uint32_t>  wait_pool_;   // flattened wait lists; see ScheduledNode
  std::vector<rt::Stream>     streams_;
  std::vector<rt::Event>      events_;
  // CROSS-ITERATION FORK/JOIN. run_async() advertises that the caller may
  // enqueue iteration N+1 while N is still running -- and with buffer reuse and
  // K>1 that is a race the INTRA-iteration events do not cover: iteration N+1's
  // node on stream 1 can overwrite a reused buffer that iteration N's node on
  // stream 0 is still reading. So every iteration opens with a fork (stream 0
  // records, all others wait) and closes with a join (all others record, stream 0
  // waits). Cost is K records + 2(K-1) waits per iteration and it must be
  // reported SEPARATELY from the intra-graph event count, or the published
  // events/iteration figure is simply wrong.
  // A 0-or-1 vector rather than a plain member: rt::Event has no default
  // constructor (it owns a cudaEvent_t, and a default-constructed one would be
  // a null handle masquerading as an event), so a bare member would make
  // ExecutionPlan itself non-default-constructible.
  std::vector<rt::Event>      fork_event_;
  std::vector<rt::Event>      join_events_;
  // Per-node timing probes, opts.profile only. A RING: re-recording an event
  // overwrites its timestamp, so a single pair per node would yield only the
  // last iteration, and reading between iterations would need a host sync per
  // iteration -- destroying the async model this phase exists to measure.
  std::vector<rt::Event>      probes_;
  int                         probe_slots_ = 0;
  int                         iteration_   = 0;
  // The sub-plans, kept so a bench can print exactly what ran.
  StreamAssignment            streams_plan_;
  MemoryPlan                  memory_plan_;
  std::shared_ptr<Storage>    arena_;
  std::vector<std::shared_ptr<Storage>> workspaces_;
  // Per-node input/output Tensor vectors, built ONCE here rather than per
  // launch. Op::launch takes const std::vector<Tensor>&, so constructing them
  // in the replay loop would cost two heap allocations and N shared_ptr
  // refcount bumps per node per iteration -- on the exact loop the plan/run
  // split exists to make cheap.
  std::vector<std::vector<Tensor>> node_inputs_;
  std::vector<std::vector<Tensor>> node_outputs_;
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

  // Read the probe ring. Call after synchronize(); opts.profile only.
  [[nodiscard]] Status collect_timings();
  [[nodiscard]] const std::vector<NodeTiming>& node_timings() const noexcept {
    return node_timings_;
  }
  [[nodiscard]] const ExecutionPlan& plan_ref() const { return plan_; }
  [[nodiscard]] const Profiler& profiler() const { return profiler_; }

 private:
  // ORDER MATTERS AND IS ENFORCED BY THE TYPE SYSTEM.
  //
  // assign_memory needs stream(n) and issue_index(n) to build happens-before;
  // assign_streams needs only topology plus Op::cost. So streams must be
  // assigned first. With assign_memory's original signature -- taking just the
  // order -- calling them the wrong way round still compiled and silently read a
  // default-initialised schedule in which every node sits on stream 0. That is
  // exactly the kReuseTopoNaive bug, reintroduced by accident and with no
  // diagnostic. Requiring the HappensBefore argument makes the wrong call order
  // fail to compile instead.
  [[nodiscard]] Status assign_streams(const std::vector<NodeId>& order);
  [[nodiscard]] Status assign_memory(const std::vector<NodeId>& order,
                                     const HappensBefore& hb);

  Graph            graph_;
  DeviceAllocator& alloc_;
  DeviceInfo       device_;
  ExecutorOptions  opts_;
  ExecutionPlan    plan_;
  Profiler         profiler_;
  std::vector<NodeTiming> node_timings_;
  bool             planned_ = false;
};

}  // namespace mcke
