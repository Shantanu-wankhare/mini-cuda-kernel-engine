// =============================================================================
//  src/graph/schedule.cpp -- the three schedule policies, event insertion,
//  record elision, and wait deduplication. Phase 4. Zero CUDA.
// =============================================================================
#include "mcke/graph/schedule.hpp"

#include <algorithm>
#include <sstream>

#include "mcke/graph/cost_model.hpp"

namespace mcke {
namespace {

// Both parallel policies reduce to the same two facts per node: "some other node
// must record an event after it" and "this node must wait for those nodes". So
// the policies produce this intermediate form in terms of NODES, and one shared
// pass afterwards turns it into dense event ids, elides unnecessary records, and
// deduplicates waits. Doing that once rather than per policy is what keeps the
// event counts comparable between them.
struct RawEdges {
  std::vector<std::vector<std::uint32_t>> wait_on;   // scheduled idx -> scheduled idxs
};

}  // namespace

StatusOr<StreamAssignment> plan_streams(const Graph& g, SchedulePolicy policy,
                                        int num_streams, const Roofline& rl) {
  if (num_streams < 1)
    return InvalidArgumentError("plan_streams: num_streams must be >= 1, got " +
                                std::to_string(num_streams));
  if (!g.finalized())
    return FailedPreconditionError("plan_streams: call Graph::finalize() first");

  MCKE_ASSIGN_OR_RETURN(std::vector<NodeId> topo, g.topological_order());

  // --- Live subgraph only. A dead node scheduled is work nobody asked for,
  //     inflating every benchmark number; and its hole in the issue indices
  //     would make HappensBefore reject the schedule outright.
  StreamAssignment sa;
  std::vector<std::uint32_t> sched_of(g.num_nodes(), kNoEventId);   // NodeId -> sched idx
  for (NodeId n : topo) {
    if (g.node(n).is_dead) continue;
    sched_of[n] = static_cast<std::uint32_t>(sa.nodes.size());
    sa.nodes.push_back(n);
  }
  const std::size_t S = sa.nodes.size();
  if (S == 0) return InvalidArgumentError("plan_streams: every node is dead");

  sa.num_streams_requested = static_cast<std::size_t>(num_streams);
  sa.stream_of.assign(S, 0);
  sa.issue_of.assign(S, 0);
  sa.record_of.assign(S, kNoEventId);
  sa.waits_of.assign(S, {});
  sa.order.resize(S);
  for (std::size_t i = 0; i < S; ++i) sa.order[i] = static_cast<std::uint32_t>(i);

  RawEdges raw;
  raw.wait_on.assign(S, {});

  // Cost of each scheduled node, from the shared roofline model. Node COUNT
  // would be the wrong load metric: every benchmark graph here mixes a ~10 ms
  // GEMM with a ~0.2 ms softmax, and counting nodes calls them equal.
  std::vector<double> cost_ms(S, 0.0);
  for (std::size_t i = 0; i < S; ++i) {
    const Node& nd = g.node(sa.nodes[i]);
    if (!nd.op) continue;
    std::vector<Shape> in;
    in.reserve(nd.inputs.size());
    for (TensorId t : nd.inputs) in.push_back(g.tensor(t).shape);
    cost_ms[i] = plan_cost_ms(nd.op->cost(in), rl);
  }

  const int K = std::min<int>(num_streams, static_cast<int>(S));

  switch (policy) {
    // -------------------------------------------------------------------------
    case SchedulePolicy::kSequential: {
      // One stream, topological order, zero events. Everything else is measured
      // against this, bit for bit.
      for (std::size_t i = 0; i < S; ++i) sa.stream_of[i] = 0;
      break;
    }

    // -------------------------------------------------------------------------
    case SchedulePolicy::kLevelParallel: {
      MCKE_ASSIGN_OR_RETURN(std::vector<std::vector<NodeId>> levels, g.levels());

      // Round-robin each level's LIVE nodes over K streams.
      std::vector<std::vector<std::uint32_t>> live_levels;
      for (const auto& lvl : levels) {
        std::vector<std::uint32_t> live;
        for (NodeId n : lvl)
          if (sched_of[n] != kNoEventId) live.push_back(sched_of[n]);
        if (!live.empty()) live_levels.push_back(std::move(live));
      }
      for (auto& lvl : live_levels)
        for (std::size_t j = 0; j < lvl.size(); ++j)
          sa.stream_of[lvl[j]] = static_cast<std::uint16_t>(j % static_cast<std::size_t>(K));

      // A TRUE barrier between consecutive levels, implemented as one.
      //
      // Weakening it to per-edge waits would quietly turn this policy into
      // chain-greedy-minus-the-heuristic and destroy the comparison the phase
      // exists to make. Its weakness -- that one long node in a level stalls
      // every short node in the next -- is the POINT, not a defect to paper over.
      for (std::size_t d = 0; d + 1 < live_levels.size(); ++d) {
        // "First" and "last" mean FIRST AND LAST BY SCHEDULED INDEX, which is
        // issue order on the stream -- NOT by position within the level's list.
        //
        // Those differ, and getting it wrong is a silent race. Graph::levels()
        // buckets by depth in NodeId order, and NodeId order is not topological
        // order, so a level's list can hold the same stream's nodes in an order
        // that has nothing to do with when they are issued. Attaching the
        // barrier wait to the wrong one leaves every earlier node on that stream
        // unordered against the previous level -- caught by the ordering
        // property test on 3 of 3000 random schedules, all kLevelParallel.
        //
        // This is precisely the index-space confusion this file's banner warns
        // about; the warning was written first and the mistake made anyway,
        // which is a decent argument for the property test existing.
        std::vector<std::uint32_t> last_on(static_cast<std::size_t>(K), kNoEventId);
        for (std::uint32_t idx : live_levels[d]) {
          std::uint32_t& cur = last_on[sa.stream_of[idx]];
          if (cur == kNoEventId || idx > cur) cur = idx;      // max scheduled index
        }
        std::vector<std::uint32_t> first_on(static_cast<std::size_t>(K), kNoEventId);
        for (std::uint32_t idx : live_levels[d + 1]) {
          std::uint32_t& cur = first_on[sa.stream_of[idx]];
          if (cur == kNoEventId || idx < cur) cur = idx;      // min scheduled index
        }

        for (int t = 0; t < K; ++t) {
          if (first_on[t] == kNoEventId) continue;
          for (int s = 0; s < K; ++s) {
            if (s == t || last_on[s] == kNoEventId) continue;
            raw.wait_on[first_on[t]].push_back(last_on[s]);
          }
        }
      }
      break;
    }

    // -------------------------------------------------------------------------
    case SchedulePolicy::kChainGreedy: {
      std::vector<double> load(static_cast<std::size_t>(K), 0.0);
      std::vector<bool>   donated(S, false);

      for (std::size_t i = 0; i < S; ++i) {
        // Live predecessors, in scheduled-index space.
        std::vector<std::uint32_t> preds;
        for (NodeId p : g.node(sa.nodes[i]).preds)
          if (sched_of[p] != kNoEventId) preds.push_back(sched_of[p]);

        // THE RULE, with the clause that makes it correct. "Exactly one
        // predecessor" alone is not enough: on the diamond A -> {B,C} -> D, BOTH
        // B and C have exactly one predecessor, so without the donation check
        // both inherit A's stream, the schedule collapses to one stream with
        // zero events, and the result publishes as an honest-looking 1.00x.
        if (preds.size() == 1 && !donated[preds[0]]) {
          sa.stream_of[i] = sa.stream_of[preds[0]];
          donated[preds[0]] = true;          // the stream is spent
        } else {
          const auto it = std::min_element(load.begin(), load.end());
          sa.stream_of[i] = static_cast<std::uint16_t>(it - load.begin());
        }
        load[sa.stream_of[i]] += cost_ms[i];

        for (std::uint32_t p : preds) raw.wait_on[i].push_back(p);
      }
      sa.stream_load_ms = load;
      break;
    }
  }

  // --- Dense per-stream issue indices, in topological order.
  {
    std::vector<std::uint32_t> next(static_cast<std::size_t>(K), 0);
    std::vector<bool> used(static_cast<std::size_t>(K), false);
    for (std::size_t i = 0; i < S; ++i) {
      const std::uint16_t s = sa.stream_of[i];
      sa.issue_of[i] = next[s]++;
      used[s] = true;
    }
    sa.num_streams_used = static_cast<std::size_t>(std::count(used.begin(), used.end(), true));
    // HappensBefore requires DENSE indices per stream, and an unused stream in
    // the middle would leave a hole. Compact the stream ids so `num_streams_used`
    // is also the id space, which is what the executor allocates rt::Streams for.
    if (sa.num_streams_used != static_cast<std::size_t>(K)) {
      std::vector<int> remap(static_cast<std::size_t>(K), -1);
      int next_id = 0;
      for (int s = 0; s < K; ++s) if (used[s]) remap[s] = next_id++;
      for (std::size_t i = 0; i < S; ++i)
        sa.stream_of[i] = static_cast<std::uint16_t>(remap[sa.stream_of[i]]);
    }
  }

  // --- THE SECOND PASS: which nodes actually need to record.
  //
  // This cannot be folded into the assignment loop above. Whether p must record
  // depends on p's SUCCESSORS' streams, which are unknown while p is being
  // placed. A one-pass implementation records unconditionally -- not wrong, but
  // it pays a cudaEventRecord per node on graphs where almost none are needed,
  // and it is easy to miss precisely because it is merely wasteful.
  std::vector<bool> needs_record(S, false);
  for (std::size_t i = 0; i < S; ++i)
    for (std::uint32_t p : raw.wait_on[i])
      if (sa.stream_of[p] != sa.stream_of[i]) needs_record[p] = true;

  std::vector<std::uint32_t> event_of(S, kNoEventId);
  std::uint32_t next_event = 0;
  for (std::size_t i = 0; i < S; ++i)
    if (needs_record[i]) { event_of[i] = next_event++; sa.record_of[i] = event_of[i]; }
  sa.num_events = next_event;

  // --- Waits, with per-stream deduplication.
  //
  // cudaStreamWaitEvent orders everything SUBSEQUENTLY enqueued on that stream
  // after the event, not just the next launch. So once stream t has waited on
  // e_p, no later node on t needs to wait on it again. Processing in issue order
  // makes the "already waited" set exact rather than approximate.
  std::vector<std::vector<std::uint32_t>> waited(sa.num_streams_used);
  for (std::size_t i = 0; i < S; ++i) {
    const std::uint16_t t = sa.stream_of[i];
    for (std::uint32_t p : raw.wait_on[i]) {
      if (sa.stream_of[p] == t) continue;      // same stream: ordering is free
      ++sa.waits_raw;
      const std::uint32_t e = event_of[p];
      auto& seen = waited[t];
      if (std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
      seen.push_back(e);
      sa.waits_of[i].push_back(e);
      ++sa.waits_dedup;
    }
  }

  if (sa.stream_load_ms.empty()) {
    sa.stream_load_ms.assign(sa.num_streams_used, 0.0);
    for (std::size_t i = 0; i < S; ++i) sa.stream_load_ms[sa.stream_of[i]] += cost_ms[i];
  } else {
    sa.stream_load_ms.resize(sa.num_streams_used);
  }
  return sa;
}

StatusOr<HappensBefore> StreamAssignment::happens_before() const {
  return HappensBefore::build(num_streams_used, stream_of, issue_of, record_of,
                              waits_of, order);
}

std::string StreamAssignment::describe() const {
  std::ostringstream os;
  os << "streams " << num_streams_used << "/" << num_streams_requested
     << "  events " << num_events << "  waits " << waits_dedup << "/" << waits_raw
     << " (dedup/raw)\n";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    os << "  [" << i << "] node " << nodes[i] << "  stream " << stream_of[i]
       << " issue " << issue_of[i];
    if (!waits_of[i].empty()) {
      os << "  waits {";
      for (std::size_t k = 0; k < waits_of[i].size(); ++k)
        os << (k ? "," : "") << waits_of[i][k];
      os << "}";
    }
    if (record_of[i] != kNoEventId) os << "  records " << record_of[i];
    os << "\n";
  }
  for (std::size_t s = 0; s < stream_load_ms.size(); ++s)
    os << "  stream " << s << " estimated load " << stream_load_ms[s] << " ms\n";
  return os.str();
}

Status verify_plan_ordering(const Graph& g, const StreamAssignment& sa) {
  MCKE_ASSIGN_OR_RETURN(HappensBefore hb, sa.happens_before());

  std::vector<std::uint32_t> sched_of(g.num_nodes(), kNoEventId);
  for (std::size_t i = 0; i < sa.nodes.size(); ++i) sched_of[sa.nodes[i]] = static_cast<std::uint32_t>(i);

  for (std::size_t i = 0; i < sa.nodes.size(); ++i) {
    for (NodeId p : g.node(sa.nodes[i]).preds) {
      const std::uint32_t pi = sched_of[p];
      if (pi == kNoEventId)
        return InternalError("verify_plan_ordering: live node " +
                             std::to_string(sa.nodes[i]) + " depends on unscheduled node " +
                             std::to_string(p));
      // The whole contract of a scheduler: every data dependency must be
      // realised as either same-stream issue order or an event edge. If this
      // fails, the schedule is wrong and no memory-planner result built on it
      // means anything -- which is why this check is separate from the race
      // checker.
      if (!hb.precedes(pi, static_cast<std::uint32_t>(i)))
        return InternalError("verify_plan_ordering: dependency node " +
                             std::to_string(p) + " -> node " + std::to_string(sa.nodes[i]) +
                             " is NOT ordered by the schedule. " +
                             hb.describe_pair(pi, static_cast<std::uint32_t>(i)));
    }
  }
  return OkStatus();
}

}  // namespace mcke
