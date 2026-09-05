// =============================================================================
//  src/graph/executor.cpp -- compile a Graph into an ExecutionPlan, then replay
//  it. Phase 4.
//
//  Host-side CUDA runtime calls only (stream/event management, memcpy). No
//  kernels, so no nvcc: the boundary CLAUDE.md section 5 defines.
//
//  ---------------------------------------------------------------------------
//  THE ONE RULE THIS FILE MUST NOT BREAK
//
//  run_async() may not synchronise. cudaStreamSynchronize is allowed in exactly
//  two places in the whole runtime -- GraphExecutor::synchronize() and benchmark
//  timing -- and a single stray one here converts the entire runtime back to
//  synchronous execution while still looking asynchronous. That is why the
//  planner allocates ONCE (allocator calls can synchronise: Phase 2 measured a
//  cudaFree at up to 720 us with a device-wide sync) and why Op::launch is
//  handed a plain workspace pointer rather than an allocator.
// =============================================================================
#include "mcke/graph/executor.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "mcke/graph/cost_model.hpp"

#if MCKE_WITH_CUDA
#include "mcke/runtime/cuda_check.hpp"
#endif

namespace mcke {

GraphExecutor::GraphExecutor(Graph graph, DeviceAllocator& alloc, DeviceInfo device,
                             ExecutorOptions opts)
    : graph_(std::move(graph)), alloc_(alloc), device_(std::move(device)), opts_(opts) {}

GraphExecutor::~GraphExecutor() = default;

Status GraphExecutor::plan() {
  // Idempotent, and it finalizes for you: forgetting finalize() should not be a
  // failure mode when plan() can simply do it.
  MCKE_RETURN_IF_ERROR(graph_.finalize());

  Roofline rl;
  rl.peak_gb_s   = device_.peak_dram_gb_s() > 0 ? device_.peak_dram_gb_s() : 1.0;
  rl.peak_tflops = 1.0;   // only used to BALANCE load, never reported as a
                          // denominator; the bench passes measured peaks when it
                          // wants a real roofline number.

  // ORDER IS LOAD-BEARING: streams first, then happens-before, then memory.
  // assign_memory's signature requires the HappensBefore precisely so this
  // cannot be got wrong silently.
  MCKE_ASSIGN_OR_RETURN(StreamAssignment sa,
                        plan_streams(graph_, opts_.schedule, opts_.num_streams, rl));
  MCKE_RETURN_IF_ERROR(verify_plan_ordering(graph_, sa));
  MCKE_ASSIGN_OR_RETURN(HappensBefore hb, sa.happens_before());
  MCKE_ASSIGN_OR_RETURN(MemoryPlan mp, plan_memory(graph_, sa, hb, opts_.memory));

  // Belt and braces, cheap, and it runs in every build: the plan we are about to
  // execute is proven race-free before a single byte is allocated. This is only
  // possible because the arena is static -- a runtime-allocating planner's
  // behaviour depends on pool state and could not be checked here at all.
  MCKE_RETURN_IF_ERROR(verify_no_buffer_races(graph_, sa, hb, mp));

  plan_ = ExecutionPlan{};
  plan_.streams_plan_ = sa;
  plan_.memory_plan_  = mp;

  const std::size_t K = sa.num_streams_used;
  plan_.streams_.reserve(K);
  for (std::size_t s = 0; s < K; ++s) {
    MCKE_ASSIGN_OR_RETURN(rt::Stream st, rt::Stream::create());
    plan_.streams_.push_back(std::move(st));
  }
  plan_.events_.reserve(sa.num_events);
  for (std::size_t e = 0; e < sa.num_events; ++e) {
    // kDependency, not kTiming: a timing event carries a timestamp, costs a
    // pipeline hiccup, and on some architectures cannot be resolved on-device.
    // Dependency events are the ones on the critical path of every iteration.
    MCKE_ASSIGN_OR_RETURN(rt::Event ev, rt::Event::create(rt::Event::Purpose::kDependency));
    plan_.events_.push_back(std::move(ev));
  }
  if (K > 1) {
    MCKE_ASSIGN_OR_RETURN(rt::Event fk, rt::Event::create(rt::Event::Purpose::kDependency));
    plan_.fork_event_.push_back(std::move(fk));
    plan_.join_events_.reserve(K - 1);
    for (std::size_t s = 1; s < K; ++s) {
      MCKE_ASSIGN_OR_RETURN(rt::Event ev, rt::Event::create(rt::Event::Purpose::kDependency));
      plan_.join_events_.push_back(std::move(ev));
    }
  }

  // --- One arena, allocated once. Every intermediate is a view into it.
  const rt::StreamHandle s0 = plan_.streams_.empty() ? rt::StreamHandle{}
                                                     : plan_.streams_[0].native();
  if (mp.arena_bytes > 0) {
    MCKE_ASSIGN_OR_RETURN(plan_.arena_, Storage::create(alloc_, mp.arena_bytes, s0));
  }
  for (std::size_t s = 0; s < mp.workspace_bytes.size(); ++s) {
    if (mp.workspace_bytes[s] == 0) { plan_.workspaces_.emplace_back(); continue; }
    MCKE_ASSIGN_OR_RETURN(std::shared_ptr<Storage> w,
                          Storage::create(alloc_, mp.workspace_bytes[s], s0));
    plan_.workspaces_.push_back(std::move(w));
  }

  plan_.bound_.assign(graph_.num_tensors(), Tensor{});
  for (std::size_t t = 0; t < graph_.num_tensors(); ++t) {
    if (mp.offset_of[t] == kNoOffset) continue;
    const TensorDesc& d = graph_.tensor(static_cast<TensorId>(t));
    MCKE_ASSIGN_OR_RETURN(plan_.bound_[t],
                          Tensor::from_storage(plan_.arena_, mp.offset_of[t], d.shape, d.dtype));
  }

  // --- Flatten the schedule into the replay array, and PRECOMPUTE the per-node
  //     input/output vectors. Op::launch takes const std::vector<Tensor>&, so
  //     building them per launch would cost two heap allocations and N shared_ptr
  //     refcount bumps per node per iteration -- on the exact loop the plan/run
  //     split exists to make cheap.
  const std::size_t S = sa.nodes.size();
  plan_.sched_.resize(S);
  plan_.node_inputs_.resize(S);
  plan_.node_outputs_.resize(S);
  plan_.wait_pool_.clear();
  for (std::size_t i = 0; i < S; ++i) {
    ScheduledNode& sn = plan_.sched_[i];
    sn.node         = sa.nodes[i];
    sn.stream_idx   = sa.stream_of[i];
    sn.issue_index  = static_cast<std::uint16_t>(sa.issue_of[i]);
    sn.wait_offset  = static_cast<std::uint32_t>(plan_.wait_pool_.size());
    sn.wait_count   = static_cast<std::uint32_t>(sa.waits_of[i].size());
    for (std::uint32_t e : sa.waits_of[i]) plan_.wait_pool_.push_back(e);
    sn.record_event = sa.record_of[i];

    const Node& nd = graph_.node(sn.node);
    for (TensorId t : nd.inputs)  plan_.node_inputs_[i].push_back(plan_.bound_[t]);
    for (TensorId t : nd.outputs) plan_.node_outputs_[i].push_back(plan_.bound_[t]);
  }

  plan_.peak_bytes_  = mp.arena_bytes;
  plan_.naive_bytes_ = mp.naive_bytes;

  // --- Probe ring for per-node timing.
  if (opts_.profile) {
    plan_.probe_slots_ = std::max(1, opts_.profile_slots);
    const std::size_t n_probe = S * static_cast<std::size_t>(plan_.probe_slots_) * 2;
    plan_.probes_.reserve(n_probe);
    for (std::size_t k = 0; k < n_probe; ++k) {
      MCKE_ASSIGN_OR_RETURN(rt::Event ev, rt::Event::create(rt::Event::Purpose::kTiming));
      plan_.probes_.push_back(std::move(ev));
    }
  }
  plan_.iteration_ = 0;
  planned_ = true;
  return OkStatus();
}

Status GraphExecutor::set_input(TensorId t, const void* host_data, std::size_t bytes) {
  if (!planned_) return FailedPreconditionError("set_input: call plan() first");
  if (t >= plan_.bound_.size()) return InvalidArgumentError("set_input: tensor out of range");
  if (!graph_.tensor(t).is_graph_input)
    return InvalidArgumentError("set_input: tensor " + std::to_string(t) +
                                " is not a declared graph input");
  if (!plan_.bound_[t].defined())
    return FailedPreconditionError("set_input: tensor " + std::to_string(t) + " is unbound");
  // Always on stream 0. A graph input has no producer and therefore no stream of
  // its own, and under a parallel schedule it is read from several streams at
  // once -- so the copy needs a defined place plus an ordering edge to every
  // consumer. run_async()'s fork provides that edge: it records on stream 0
  // AFTER these copies and every other stream waits on it.
  const rt::StreamHandle s0 = plan_.streams_.empty() ? rt::StreamHandle{}
                                                     : plan_.streams_[0].native();
  return plan_.bound_[t].copy_from_host(host_data, bytes, s0);
}

Status GraphExecutor::run_async() {
  if (!planned_) return FailedPreconditionError("run_async: call plan() first");
  const std::size_t K = plan_.streams_.size();
  if (K == 0) return FailedPreconditionError("run_async: no streams");

  if (opts_.poison_buffers && plan_.arena_) {
    // A DIFFERENT pattern per iteration, not a fixed one. A read of
    // uninitialised memory then produces a different wrong answer each time,
    // which the run-to-run bit-identity check catches even where a NaN would be
    // swallowed (fmaxf drops NaN on some paths; 0 * NaN does not).
    const std::uint32_t pat = opts_.poison_pattern +
                              static_cast<std::uint32_t>(plan_.iteration_) * 0x9E3779B9u;
    std::vector<std::uint32_t> fill(plan_.arena_->bytes() / 4, pat);
#if MCKE_WITH_CUDA
    MCKE_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(plan_.arena_->data(), fill.data(),
                                              fill.size() * 4, cudaMemcpyHostToDevice,
                                              plan_.streams_[0].native()));
#else
    std::memcpy(plan_.arena_->data(), fill.data(), fill.size() * 4);
#endif
  }

  // --- FORK. Without it, iteration N+1's node on stream 1 can overwrite a
  //     reused buffer that iteration N's node on stream 0 is still reading. The
  //     intra-graph events do not cover that: they order nodes WITHIN one
  //     iteration.
  if (K > 1) {
    MCKE_RETURN_IF_ERROR(plan_.fork_event_[0].record(plan_.streams_[0]));
    for (std::size_t s = 1; s < K; ++s)
      MCKE_RETURN_IF_ERROR(plan_.streams_[s].wait(plan_.fork_event_[0]));
  }

  const int slot = plan_.probe_slots_ ? (plan_.iteration_ % plan_.probe_slots_) : 0;

  for (std::size_t i = 0; i < plan_.sched_.size(); ++i) {
    const ScheduledNode& sn = plan_.sched_[i];
    rt::Stream& st = plan_.streams_[sn.stream_idx];

    for (std::uint32_t w = 0; w < sn.wait_count; ++w)
      MCKE_RETURN_IF_ERROR(st.wait(plan_.events_[plan_.wait_pool_[sn.wait_offset + w]]));

    OpContext ctx;
    ctx.stream = st.native();
    ctx.device = &device_;
    if (sn.stream_idx < plan_.workspaces_.size() && plan_.workspaces_[sn.stream_idx]) {
      ctx.workspace       = plan_.workspaces_[sn.stream_idx]->data();
      ctx.workspace_bytes = plan_.workspaces_[sn.stream_idx]->bytes();
    }

    const std::size_t p = (i * static_cast<std::size_t>(plan_.probe_slots_) +
                           static_cast<std::size_t>(slot)) * 2;
    if (opts_.profile) MCKE_RETURN_IF_ERROR(plan_.probes_[p].record(st));

    {
      NvtxRange range(graph_.node(sn.node).name.c_str());
      MCKE_RETURN_IF_ERROR(graph_.node_op(sn.node)->launch(
          ctx, plan_.node_inputs_[i], plan_.node_outputs_[i]));
    }

    if (opts_.profile) MCKE_RETURN_IF_ERROR(plan_.probes_[p + 1].record(st));
    if (sn.record_event != kNoEvent)
      MCKE_RETURN_IF_ERROR(plan_.events_[sn.record_event].record(st));
  }

  // --- JOIN, the other half of the cross-iteration guard.
  if (K > 1) {
    for (std::size_t s = 1; s < K; ++s) {
      MCKE_RETURN_IF_ERROR(plan_.join_events_[s - 1].record(plan_.streams_[s]));
      MCKE_RETURN_IF_ERROR(plan_.streams_[0].wait(plan_.join_events_[s - 1]));
    }
  }
  ++plan_.iteration_;
  return OkStatus();
}

Status GraphExecutor::synchronize() {
  if (!planned_) return FailedPreconditionError("synchronize: call plan() first");
  // One of exactly two places in the runtime where a host-side barrier is legal.
  for (rt::Stream& s : plan_.streams_) MCKE_RETURN_IF_ERROR(s.synchronize());
  return OkStatus();
}

StatusOr<Tensor> GraphExecutor::output(TensorId t) const {
  if (!planned_) return FailedPreconditionError("output: call plan() first");
  if (t >= plan_.bound_.size()) return InvalidArgumentError("output: tensor out of range");
  if (!graph_.tensor(t).is_graph_output)
    return InvalidArgumentError("output: tensor " + std::to_string(t) +
                                " is not marked as a graph output");
  return plan_.bound_[t];
}

Status GraphExecutor::collect_timings() {
  node_timings_.clear();
  if (!opts_.profile) return OkStatus();
  const int slots = std::min(plan_.probe_slots_, plan_.iteration_);
  if (slots <= 0) return OkStatus();

  for (std::size_t i = 0; i < plan_.sched_.size(); ++i) {
    NodeTiming nt;
    nt.node       = plan_.sched_[i].node;
    nt.name       = graph_.node(nt.node).name;
    nt.stream_idx = plan_.sched_[i].stream_idx;
    const Node& nd = graph_.node(nt.node);
    if (nd.op) {
      std::vector<Shape> in;
      for (TensorId t : nd.inputs) in.push_back(graph_.tensor(t).shape);
      const OpCost c = nd.op->cost(in);
      nt.flops = c.flops;
      nt.bytes = c.bytes;
    }
    std::vector<double> ms;
    for (int s = 0; s < slots; ++s) {
      const std::size_t p = (i * static_cast<std::size_t>(plan_.probe_slots_) +
                             static_cast<std::size_t>(s)) * 2;
      auto e = rt::Event::elapsed_ms(plan_.probes_[p], plan_.probes_[p + 1]);
      if (e.ok()) ms.push_back(*e);
    }
    if (!ms.empty()) {
      std::sort(ms.begin(), ms.end());
      nt.median_ms = ms[ms.size() / 2];
      nt.min_ms    = ms.front();
      nt.samples   = static_cast<int>(ms.size());
    }
    node_timings_.push_back(std::move(nt));
  }
  return OkStatus();
}

std::string ExecutionPlan::describe() const {
  std::ostringstream os;
  os << streams_plan_.describe();
  os << memory_plan_.describe();
  os << "  events/iter: intra " << intra_events() << " records + " << intra_waits()
     << " waits; fork/join " << forkjoin_events() << " (reported separately -- a fixed\n"
     << "    per-iteration cost of the async contract, not attributable to the policy)\n";
  return os.str();
}

}  // namespace mcke
