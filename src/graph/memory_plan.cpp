// =============================================================================
//  src/graph/memory_plan.cpp -- liveness, interference, arena packing. Phase 4.
//  Zero CUDA: every number this file produces is computed on a laptop.
// =============================================================================
#include "mcke/graph/memory_plan.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace mcke {
namespace {

std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

// Every node that touches a tensor: its producer plus all of its consumers, in
// SCHEDULED-INDEX space.
//
// A SET, not a "last use". Under a parallel schedule a tensor's consumers may be
// mutually unordered, so there is no single last access -- any formulation that
// computes a scalar and compares it has already lost. This is the shape the
// reuse condition actually needs.
struct Accesses {
  std::vector<std::uint32_t> nodes;   // scheduled indices
  std::uint32_t producer = kNoEventId;
};

const char* policy_name(MemoryPolicy p) {
  switch (p) {
    case MemoryPolicy::kAllocPerTensor:     return "alloc_per_tensor";
    case MemoryPolicy::kReuseTopoNaive:     return "reuse_topo_naive(UNSAFE)";
    case MemoryPolicy::kReuseSameStream:    return "reuse_same_stream";
    case MemoryPolicy::kReuseHappensBefore: return "reuse_happens_before";
  }
  return "?";
}

}  // namespace

StatusOr<MemoryPlan> plan_memory(const Graph& g, const StreamAssignment& sa,
                                 const HappensBefore& hb, MemoryPolicy policy) {
  MemoryPlan mp;
  mp.policy = policy;
  const std::size_t T = g.num_tensors();
  mp.offset_of.assign(T, kNoOffset);
  mp.bytes_of.assign(T, 0);

  // NodeId -> scheduled index, and topo position -> for the naive policy only.
  std::vector<std::uint32_t> sched_of(g.num_nodes(), kNoEventId);
  std::vector<int> topo_pos(g.num_nodes(), -1);
  for (std::size_t i = 0; i < sa.nodes.size(); ++i) {
    sched_of[sa.nodes[i]] = static_cast<std::uint32_t>(i);
    topo_pos[sa.nodes[i]] = static_cast<int>(i);
  }

  // --- Which tensors need a device buffer, and who touches them.
  std::vector<Accesses> acc(T);
  std::vector<TensorId> placed;
  for (std::size_t t = 0; t < T; ++t) {
    const TensorDesc& d = g.tensor(static_cast<TensorId>(t));
    // A tensor produced only by a DEAD node needs no buffer: nothing will run.
    if (d.producer != kInvalidNode && sched_of[d.producer] == kNoEventId) continue;
    const std::size_t bytes = round_up(d.shape.bytes(d.dtype), kDeviceAlignment);
    if (bytes == 0) continue;
    mp.bytes_of[t] = bytes;
    placed.push_back(static_cast<TensorId>(t));

    if (d.producer != kInvalidNode) {
      acc[t].producer = sched_of[d.producer];
      acc[t].nodes.push_back(acc[t].producer);
    }
    for (NodeId c : d.consumers)
      if (sched_of[c] != kNoEventId) acc[t].nodes.push_back(sched_of[c]);

    if (d.is_graph_input)       mp.input_bytes += bytes;
    else if (d.is_graph_output) mp.output_bytes += bytes;
    else                        mp.intermediate_bytes += bytes;
    mp.naive_bytes += bytes;
  }

  // --- May the buffer holding `a` be reused for `b`?
  //
  //     THE CONDITION, per graph/happens_before.hpp: every access to `a` must
  //     happen-before b's PRODUCER. Only the producer of b needs to be
  //     dominated -- SSA gives it exactly one, and every consumer of b
  //     happens-after it by construction -- so this is |accesses(a)| checks, not
  //     a cross product.
  auto can_share = [&](TensorId a, TensorId b) -> bool {
    const TensorDesc& da = g.tensor(a);
    const TensorDesc& db = g.tensor(b);
    // Graph inputs and outputs never die: an input is filled by an async H2D the
    // planner does not track and may be re-read every iteration; an output must
    // outlive execution. Neither is ever a reuse candidate, in EITHER direction.
    if (!da.reusable() || !db.reusable()) return false;
    if (acc[b].producer == kNoEventId) return false;

    switch (policy) {
      case MemoryPolicy::kAllocPerTensor:
        return false;   // the baseline: one buffer per tensor, never shared

      case MemoryPolicy::kReuseTopoNaive: {
        // DELIBERATELY UNSAFE, and shipped so the trap is demonstrated rather
        // than merely avoided. Compares live INTERVALS in the topological order,
        // which is the obvious implementation and is wrong under any parallel
        // schedule -- a topological order is one arbitrary linear extension of
        // the dependency partial order, and the executor's happens-before is a
        // DIFFERENT extension of it.
        int a_last = -1;
        for (std::uint32_t n : acc[a].nodes) a_last = std::max(a_last, static_cast<int>(n));
        return a_last >= 0 && a_last < static_cast<int>(acc[b].producer);
      }

      case MemoryPolicy::kReuseSameStream: {
        // Sound but conservative: same-stream issue order IS a happens-before
        // edge, so this accepts a subset of what the correct policy accepts. It
        // declines every reuse that a cross-stream event had already made safe,
        // which is exactly the gap the next arm measures.
        const std::uint16_t s = sa.stream_of[acc[b].producer];
        for (std::uint32_t n : acc[a].nodes) {
          if (sa.stream_of[n] != s) return false;
          if (!hb.precedes(n, acc[b].producer)) return false;
        }
        return !acc[a].nodes.empty();
      }

      case MemoryPolicy::kReuseHappensBefore: {
        for (std::uint32_t n : acc[a].nodes)
          if (!hb.precedes(n, acc[b].producer)) return false;
        return !acc[a].nodes.empty();
      }
    }
    return false;
  };

  // --- Packing: greedy by size descending, first-fit offset.
  //
  //     Largest first because a big tensor placed late has to skip past every
  //     small one already sitting in the low offsets, which fragments the arena.
  //     This is the TFLite/TVM heuristic. The alternative -- first-fit in
  //     liveness order -- is easier to explain and measurably worse; it is
  //     mentioned rather than implemented because both are host-side and the
  //     comparison is free if it is ever wanted.
  std::vector<TensorId> by_size = placed;
  std::sort(by_size.begin(), by_size.end(), [&](TensorId x, TensorId y) {
    if (mp.bytes_of[x] != mp.bytes_of[y]) return mp.bytes_of[x] > mp.bytes_of[y];
    return x < y;   // deterministic tie-break: a nondeterministic arena layout
                    // would make peak_bytes irreproducible between runs
  });

  std::vector<TensorId> done;
  for (TensorId t : by_size) {
    // Every already-placed tensor that CANNOT share with t blocks its interval.
    std::vector<std::pair<std::size_t, std::size_t>> blocked;
    for (TensorId u : done)
      if (!can_share(u, t) && !can_share(t, u))
        blocked.emplace_back(mp.offset_of[u], mp.offset_of[u] + mp.bytes_of[u]);
    std::sort(blocked.begin(), blocked.end());

    // Lowest offset that clears every blocked interval. Offsets stay
    // kDeviceAlignment-aligned throughout: an unaligned slice would break the
    // coalescing assumption every Phase 3 kernel was tuned under, silently, with
    // no error -- the Phase 3d GEMM numbers would just quietly degrade.
    std::size_t off = 0;
    for (const auto& b : blocked) {
      if (off + mp.bytes_of[t] <= b.first) break;   // it fits in the gap
      off = std::max(off, round_up(b.second, kDeviceAlignment));
    }
    mp.offset_of[t] = off;
    mp.arena_bytes  = std::max(mp.arena_bytes, off + mp.bytes_of[t]);
    done.push_back(t);
  }

  std::vector<std::size_t> distinct;
  for (TensorId t : placed) distinct.push_back(mp.offset_of[t]);
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  mp.buffers_used = distinct.size();

  // --- Workspace: one arena per stream, sized by its largest consumer.
  //     Safe with zero analysis because same-stream issue is in order, and
  //     deterministic because it does not depend on which nodes happen to
  //     overlap.
  mp.workspace_bytes.assign(sa.num_streams_used, 0);
  for (std::size_t i = 0; i < sa.nodes.size(); ++i) {
    const Node& nd = g.node(sa.nodes[i]);
    if (!nd.op) continue;
    std::vector<Shape> in;
    in.reserve(nd.inputs.size());
    for (TensorId t : nd.inputs) in.push_back(g.tensor(t).shape);
    const std::size_t w = nd.op->workspace_bytes(in);
    std::size_t& slot = mp.workspace_bytes[sa.stream_of[i]];
    slot = std::max(slot, round_up(w, kDeviceAlignment));
  }
  return mp;
}

Status verify_no_buffer_races(const Graph& g, const StreamAssignment& sa,
                              const HappensBefore& hb, const MemoryPlan& mp) {
  std::vector<std::uint32_t> sched_of(g.num_nodes(), kNoEventId);
  for (std::size_t i = 0; i < sa.nodes.size(); ++i) sched_of[sa.nodes[i]] = static_cast<std::uint32_t>(i);

  auto accesses = [&](TensorId t) {
    std::vector<std::uint32_t> out;
    const TensorDesc& d = g.tensor(t);
    if (d.producer != kInvalidNode && sched_of[d.producer] != kNoEventId)
      out.push_back(sched_of[d.producer]);
    for (NodeId c : d.consumers)
      if (sched_of[c] != kNoEventId) out.push_back(sched_of[c]);
    return out;
  };

  const std::size_t T = g.num_tensors();
  for (std::size_t a = 0; a < T; ++a) {
    if (mp.offset_of[a] == kNoOffset) continue;
    for (std::size_t b = a + 1; b < T; ++b) {
      if (mp.offset_of[b] == kNoOffset) continue;
      // Do their byte ranges overlap at all?
      const std::size_t a0 = mp.offset_of[a], a1 = a0 + mp.bytes_of[a];
      const std::size_t b0 = mp.offset_of[b], b1 = b0 + mp.bytes_of[b];
      if (a1 <= b0 || b1 <= a0) continue;

      // They share bytes, so one tensor's accesses must ALL precede the other's
      // producer. Try both directions before reporting.
      const auto aa = accesses(static_cast<TensorId>(a));
      const auto ab = accesses(static_cast<TensorId>(b));
      const TensorDesc& da = g.tensor(static_cast<TensorId>(a));
      const TensorDesc& db = g.tensor(static_cast<TensorId>(b));

      auto all_precede = [&](const std::vector<std::uint32_t>& from, const TensorDesc& to,
                             std::uint32_t* bad_u, std::uint32_t* bad_v) {
        if (to.producer == kInvalidNode) return false;
        const std::uint32_t p = sched_of[to.producer];
        if (p == kNoEventId) return false;
        for (std::uint32_t u : from)
          if (!hb.precedes(u, p)) { *bad_u = u; *bad_v = p; return false; }
        return !from.empty();
      };
      std::uint32_t bu = 0, bv = 0;
      if (all_precede(aa, db, &bu, &bv)) continue;
      std::uint32_t bu2 = 0, bv2 = 0;
      if (all_precede(ab, da, &bu2, &bv2)) continue;

      std::ostringstream os;
      os << "verify_no_buffer_races: tensors " << a << " ('" << da.name << "') and " << b
         << " ('" << db.name << "') share bytes [" << std::max(a0, b0) << ","
         << std::min(a1, b1) << ") but neither's accesses are ordered before the "
         << "other's producer. " << hb.describe_pair(bu, bv);
      return InternalError(os.str());
    }
  }
  return OkStatus();
}

std::string MemoryPlan::describe() const {
  std::ostringstream os;
  os << "memory " << policy_name(policy) << ": arena " << arena_bytes << " B ("
     << (arena_bytes / 1024.0 / 1024.0) << " MiB) vs naive " << naive_bytes << " B ("
     << (naive_bytes / 1024.0 / 1024.0) << " MiB)";
  if (arena_bytes) os << "  ratio " << (double(naive_bytes) / double(arena_bytes)) << "x";
  os << "\n  buffers " << buffers_used << "  breakdown: inputs " << input_bytes
     << " B, intermediates " << intermediate_bytes << " B, outputs " << output_bytes << " B\n";
  for (std::size_t s = 0; s < workspace_bytes.size(); ++s)
    os << "  stream " << s << " workspace " << workspace_bytes[s] << " B\n";
  return os.str();
}

}  // namespace mcke
