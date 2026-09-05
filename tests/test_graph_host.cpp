// =============================================================================
//  tests/test_graph_host.cpp
//
//  WHAT: Host-only tests for the Phase 4 graph layer. No CUDA, no GPU, no
//        device memory -- so these run on the MacBook and in ctest everywhere.
//
//  WHY A SEPARATE TARGET from test_host_core.cpp: that file is Phases 0-3
//  (allocators, shapes, kernel references) and is already ~2,300 lines. The
//  graph layer is a new subsystem with its own fixtures, so it gets its own
//  translation unit rather than growing that one without bound.
//
//  ---------------------------------------------------------------------------
//  WHAT THIS FILE IS FOR, above all else
//
//  Phase 4's central correctness claim is that a memory planner MUST NOT use
//  topological-order liveness, because a topological order is one arbitrary
//  linear extension of the dependency partial order while the executor's
//  happens-before is that partial order plus stream-serialisation edges. They
//  extend the same partial order differently.
//
//  That claim was derived by hand. This file MEASURES it: it builds the same
//  5-node graph under all three schedule policies and asks the shared
//  happens-before relation whether the reuse a naive planner would perform is
//  actually safe. The answer is "yes" under kSequential and "no" under BOTH
//  parallel policies -- including kLevelParallel, whose inter-level barrier is
//  intuitively supposed to prevent exactly this and does not.
//
//  Deriving a race on paper and asserting it in a test are different things.
//  This is the second.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "mcke/graph/graph.hpp"
#include "mcke/graph/happens_before.hpp"

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      ++g_failures;                                                            \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                          \
  } while (0)

#define CHECK_EQ(a, b)                                                         \
  do {                                                                         \
    ++g_checks;                                                                \
    auto _a = (a);                                                             \
    auto _b = (b);                                                             \
    if (!(_a == _b)) {                                                         \
      ++g_failures;                                                            \
      std::printf("  FAIL %s:%d  %s == %s  (%lld vs %lld)\n", __FILE__,        \
                  __LINE__, #a, #b, (long long)_a, (long long)_b);             \
    }                                                                          \
  } while (0)

using mcke::HappensBefore;
using mcke::kNoEventId;

// -----------------------------------------------------------------------------
// THE FIXTURE: the minimal graph that exposes the trap.
//
//     X (graph input)
//     N0: a = f(X)          N2: b = g(X)
//     N1: c = h(a)          N3: d = k(b)
//             N4: out = m(c, d)
//
// Two chains of unequal reach joined at the end. FIFO-Kahn visits
// N0, N2, N1, N3, N4, so by TOPOLOGICAL POSITION:
//     a = [0, 2]   (defined by N0, last read by N1)
//     d = [3, 4]   (defined by N3, read by N4)
// Those intervals are disjoint, so a linear-scan planner gives `d` the buffer
// that held `a`. Whether that is safe depends entirely on the schedule.
//
// The unequal-branch shape is not contrived: it is the transformer block with a
// residual connection.
// -----------------------------------------------------------------------------
constexpr std::uint32_t N0 = 0, N1 = 1, N2 = 2, N3 = 3, N4 = 4;

// The global issue order (the topological order the planner used) is the same
// for every policy; only the stream/event assignment differs.
const std::vector<std::uint32_t> kOrder = {N0, N2, N1, N3, N4};

struct Sched {
  std::size_t                                   streams;
  std::vector<std::uint16_t>                    stream_of;
  std::vector<std::uint32_t>                    issue_of;
  std::vector<std::uint32_t>                    record_of;
  std::vector<std::vector<std::uint32_t>>       waits_of;
};

// kSequential: one stream, topological order. The correctness baseline.
Sched sequential() {
  return Sched{
      1,
      {0, 0, 0, 0, 0},
      // issue index within the stream, following kOrder: N0,N2,N1,N3,N4
      {0, 2, 1, 3, 4},
      {kNoEventId, kNoEventId, kNoEventId, kNoEventId, kNoEventId},
      {{}, {}, {}, {}, {}}};
}

// kChainGreedy: N0,N1 -> stream 0 (N1 inherits, being N0's first successor);
// N2,N3 -> stream 1; N4 joins on stream 0 and waits on N3's event.
// NOTE there is no event of any kind between the two streams before N4.
Sched chain_greedy() {
  Sched s{2,
          {0, 0, 1, 1, 0},
          {0, 1, 0, 1, 2},
          {kNoEventId, kNoEventId, kNoEventId, /*N3 records*/ 0, kNoEventId},
          {{}, {}, {}, {}, {}}};
  s.waits_of[N4] = {0};
  return s;
}

// kLevelParallel: levels {N0,N2}, {N1,N3}, {N4}, round-robined over 2 streams,
// with a TRUE barrier between consecutive levels.
Sched level_parallel() {
  Sched s{2,
          {0, 0, 1, 1, 0},
          {0, 1, 0, 1, 2},
          // N0 records ev0, N2 records ev1 (the L0->L1 barrier);
          // N3 records ev2 (the L1->L2 barrier).
          {0, kNoEventId, 1, 2, kNoEventId},
          {{}, {}, {}, {}, {}}};
  s.waits_of[N1] = {1};   // s0 waits on s1's level-0 work
  s.waits_of[N3] = {0};   // s1 waits on s0's level-0 work
  s.waits_of[N4] = {2};   // s0 waits on s1's level-1 work
  return s;
}

HappensBefore build(const Sched& s) {
  auto hb = HappensBefore::build(s.streams, s.stream_of, s.issue_of, s.record_of,
                                 s.waits_of, kOrder);
  hb.status().throw_if_error();
  return std::move(*hb);
}

// The reuse condition from happens_before.hpp: the buffer holding T1 may be
// reused for T2 iff EVERY access to T1 happens-before T2's producer.
bool reuse_is_safe(const HappensBefore& hb,
                   const std::vector<std::uint32_t>& t1_accesses,
                   std::uint32_t t2_producer) {
  for (std::uint32_t u : t1_accesses)
    if (!hb.precedes(u, t2_producer)) return false;
  return true;
}

void test_hb_sequential() {
  std::printf("test_hb_sequential\n");
  const HappensBefore hb = build(sequential());

  // One stream in issue order: the relation is TOTAL, so it must agree exactly
  // with the topological order and nothing may be concurrent with anything.
  const std::vector<std::uint32_t> by_issue = {N0, N2, N1, N3, N4};
  for (std::size_t i = 0; i < by_issue.size(); ++i)
    for (std::size_t j = 0; j < by_issue.size(); ++j) {
      const bool expect = i < j;
      CHECK_EQ(hb.precedes(by_issue[i], by_issue[j]), expect);
      if (i != j) CHECK(!hb.concurrent(by_issue[i], by_issue[j]));
    }
  // A node never precedes itself -- otherwise a tensor would be reusable by its
  // own producer.
  for (std::uint32_t n = 0; n < 5; ++n) {
    CHECK(!hb.precedes(n, n));
    CHECK(!hb.concurrent(n, n));
  }
}

void test_hb_parallel_structure() {
  std::printf("test_hb_parallel_structure\n");
  {
    const HappensBefore hb = build(chain_greedy());
    // Within each chain, same-stream ordering is free and must hold.
    CHECK(hb.precedes(N0, N1));
    CHECK(hb.precedes(N2, N3));
    // The join is ordered by the event.
    CHECK(hb.precedes(N3, N4));
    CHECK(hb.precedes(N2, N4));   // transitively, through N3's clock
    CHECK(hb.precedes(N1, N4));   // same stream as N4
    // The two chains are mutually unordered -- no events exist between them.
    CHECK(hb.concurrent(N0, N2));
    CHECK(hb.concurrent(N0, N3));
    CHECK(hb.concurrent(N1, N2));
    CHECK(hb.concurrent(N1, N3));
  }
  {
    const HappensBefore hb = build(level_parallel());
    // The inter-level barrier DOES order level 0 against level 1, in both
    // directions across the streams. This is the part people expect.
    CHECK(hb.precedes(N0, N1));
    CHECK(hb.precedes(N0, N3));   // <-- the barrier's cross-stream edge
    CHECK(hb.precedes(N2, N1));   // <-- and the other way
    CHECK(hb.precedes(N2, N3));
    CHECK(hb.precedes(N3, N4));
    // But WITHIN a level, nodes on different streams remain unordered. This is
    // the gap the barrier does not close, and the reason the race below fires
    // even under a policy that looks synchronised.
    CHECK(hb.concurrent(N1, N3));
    CHECK(hb.concurrent(N0, N2));
  }
}

// THE HEADLINE TEST.
void test_topo_liveness_is_unsound() {
  std::printf("test_topo_liveness_is_unsound\n");

  // What a naive linear-scan planner concludes from topological positions
  // alone. These are the intervals; they are genuinely disjoint.
  const int a_def = 0, a_last = 2;      // tensor a: def by N0, last read by N1
  const int d_def = 3, d_last = 4;      // tensor d: def by N3, read by N4
  CHECK(a_last < d_def);                // "disjoint, therefore reuse is safe"
  (void)a_def; (void)d_last;

  // What happens-before says. Tensor `a` is accessed by its producer N0 and its
  // consumer N1; `d` is produced by N3.
  const std::vector<std::uint32_t> a_accesses = {N0, N1};

  const HappensBefore seq = build(sequential());
  const HappensBefore cg  = build(chain_greedy());
  const HappensBefore lp  = build(level_parallel());

  CHECK(reuse_is_safe(seq, a_accesses, N3));    // safe: relation is total
  CHECK(!reuse_is_safe(cg,  a_accesses, N3));   // RACE
  CHECK(!reuse_is_safe(lp,  a_accesses, N3));   // RACE, despite the barrier

  std::printf("  topo positions say a=[%d,%d] d=[%d,%d] are disjoint -> reuse\n",
              a_def, a_last, d_def, d_last);
  std::printf("  happens-before says: sequential=SAFE  chain_greedy=RACE  "
              "level_parallel=RACE\n");
  std::printf("  chain_greedy   %s\n", cg.describe_pair(N1, N3).c_str());
  std::printf("  level_parallel %s\n", lp.describe_pair(N1, N3).c_str());
  std::printf("  level_parallel %s\n", lp.describe_pair(N0, N3).c_str());
  std::printf("  ^ note the barrier DID order N0 before N3; what it missed is\n"
              "    N1 vs N3, which are in the same level on different streams.\n");
}

// Under one stream, happens-before is total, so the sound planner and the naive
// one must agree on EVERY pair. That equivalence is why there is one planner
// with one code path rather than a fast-unsafe path and a slow-safe one.
void test_single_stream_degenerates_to_topo() {
  std::printf("test_single_stream_degenerates_to_topo\n");
  const Sched s = sequential();
  const HappensBefore hb = build(s);
  // topo position of each node, from kOrder
  std::vector<int> pos(5, -1);
  for (std::size_t i = 0; i < kOrder.size(); ++i) pos[kOrder[i]] = static_cast<int>(i);
  for (std::uint32_t u = 0; u < 5; ++u)
    for (std::uint32_t v = 0; v < 5; ++v)
      CHECK_EQ(hb.precedes(u, v), pos[u] < pos[v]);
}

void test_hb_rejects_malformed_input() {
  std::printf("test_hb_rejects_malformed_input\n");
  // Zero streams.
  CHECK(!HappensBefore::build(0, {0}, {0}, {kNoEventId}, {{}}, {0}).ok());
  // Mismatched array lengths.
  CHECK(!HappensBefore::build(1, {0, 0}, {0}, {kNoEventId}, {{}}, {0}).ok());
  // A node on a stream that does not exist.
  CHECK(!HappensBefore::build(1, {5}, {0}, {kNoEventId}, {{}}, {0}).ok());
  // Two nodes sharing an issue index on one stream -- the planner's job to
  // avoid, and a silent weakening of the relation if it does not.
  CHECK(!HappensBefore::build(1, {0, 0}, {0, 0}, {kNoEventId, kNoEventId}, {{}, {}},
                              {0, 1}).ok());
  // A gap in the issue indices, same reasoning.
  CHECK(!HappensBefore::build(1, {0, 0}, {0, 2}, {kNoEventId, kNoEventId}, {{}, {}},
                              {0, 1}).ok());
  // Waiting on an event nobody records.
  CHECK(!HappensBefore::build(1, {0}, {0}, {kNoEventId}, {{7}}, {0}).ok());
  // Two nodes recording the SAME event: the second record moves the timestamp
  // under a wait already issued against the first.
  CHECK(!HappensBefore::build(1, {0, 0}, {0, 1}, {0, 0}, {{}, {}}, {0, 1}).ok());
  // And the well-formed control, so the rejections above mean something.
  CHECK(HappensBefore::build(1, {0}, {0}, {kNoEventId}, {{}}, {0}).ok());
}


// =============================================================================
//  Graph: construction, sorting, depths, liveness
// =============================================================================

using mcke::DType;
using mcke::Graph;
using mcke::NodeId;
using mcke::OpCost;
using mcke::Shape;
using mcke::Status;
using mcke::TensorId;

// A stand-in Op, so the graph layer can be tested WITHOUT the four real ops.
//
// That decoupling is the point, not a shortcut: graph tests should fail when the
// graph is wrong, not when GemmOp::infer_shapes is wrong. It also means stage 4a
// is fully testable before stage 4b exists.
class FakeOp final : public mcke::Op {
 public:
  explicit FakeOp(Shape out, int n_out = 1) : out_(out), n_out_(n_out) {}
  [[nodiscard]] std::string_view type_name() const override { return "Fake"; }
  [[nodiscard]] Status launch(const mcke::OpContext&, const std::vector<mcke::Tensor>&,
                              const std::vector<mcke::Tensor>&) override {
    return mcke::UnimplementedError("FakeOp::launch is never called in host tests");
  }
  [[nodiscard]] mcke::StatusOr<std::vector<Shape>> infer_shapes(
      const std::vector<Shape>& in) const override {
    if (fail_) return mcke::InvalidArgumentError("FakeOp: deliberate infer_shapes failure");
    Shape s = in.empty() ? out_ : in[0];
    return std::vector<Shape>(static_cast<std::size_t>(n_out_), s);
  }
  [[nodiscard]] OpCost cost(const std::vector<Shape>& in) const override {
    const Shape s = in.empty() ? out_ : in[0];
    OpCost c;
    c.flops = static_cast<std::uint64_t>(s.numel());
    c.bytes = static_cast<std::uint64_t>(s.numel()) * 2u * sizeof(float);
    return c;
  }
  void set_fail(bool f) { fail_ = f; }

 private:
  Shape out_;
  int   n_out_ = 1;
  bool  fail_  = false;
};

mcke::OpPtr fake(Shape s = Shape{4, 4}, int n_out = 1) {
  return std::make_unique<FakeOp>(s, n_out);
}

// Position of each node in an order, for the "every edge points forward" checks.
std::vector<int> positions(const Graph& g, const std::vector<NodeId>& order) {
  std::vector<int> pos(g.num_nodes(), -1);
  for (std::size_t i = 0; i < order.size(); ++i) pos[order[i]] = static_cast<int>(i);
  return pos;
}

// Transitive reachability, used to check that a level really is an antichain and
// to exhibit an antichain larger than the max level width.
std::vector<std::vector<bool>> reachability(const Graph& g) {
  const std::size_t n = g.num_nodes();
  std::vector<std::vector<bool>> r(n, std::vector<bool>(n, false));
  auto ord = g.topological_order();
  ord.status().throw_if_error();
  for (auto it = ord->rbegin(); it != ord->rend(); ++it) {
    const NodeId u = *it;
    for (NodeId v : g.node(u).succs) {
      r[u][v] = true;
      for (std::size_t k = 0; k < n; ++k) if (r[v][k]) r[u][k] = true;
    }
  }
  return r;
}

void test_graph_diamond() {
  std::printf("test_graph_diamond\n");
  Graph g;
  auto x = g.add_input(Shape{8, 8}, DType::kF32, "x");
  x.status().throw_if_error();
  auto a = g.add_node(fake(), {*x}, "A");   a.status().throw_if_error();
  auto b = g.add_node(fake(), {(*a)[0]}, "B"); b.status().throw_if_error();
  auto c = g.add_node(fake(), {(*a)[0]}, "C"); c.status().throw_if_error();
  auto d = g.add_node(fake(), {(*b)[0], (*c)[0]}, "D"); d.status().throw_if_error();
  CHECK(g.mark_output((*d)[0]).ok());
  CHECK(g.finalize().ok());

  CHECK_EQ(g.num_nodes(), 4u);
  // preds/succs derived from tensor def/use, never declared.
  CHECK_EQ(g.node(3).preds.size(), 2u);
  CHECK_EQ(g.node(0).succs.size(), 2u);

  auto lv = g.levels();  lv.status().throw_if_error();
  CHECK_EQ(lv->size(), 3u);          // A | B,C | D
  CHECK_EQ((*lv)[0].size(), 1u);
  CHECK_EQ((*lv)[1].size(), 2u);
  CHECK_EQ((*lv)[2].size(), 1u);
  auto w = g.max_level_width(); w.status().throw_if_error();
  CHECK_EQ(*w, 2u);

  // Equal-length branches => everything is on the critical path.
  for (std::size_t n = 0; n < g.num_nodes(); ++n) CHECK_EQ(g.node(n).slack, 0);
  CHECK(g.dead_nodes().empty());
  CHECK_EQ(g.total_cost().flops, 4u * 64u);   // 4 nodes x 8x8
}

void test_graph_chain_and_liveness() {
  std::printf("test_graph_chain_and_liveness\n");
  Graph g;
  auto x = g.add_input(Shape{4, 4}, DType::kF32, "x"); x.status().throw_if_error();
  TensorId cur = *x;
  for (int i = 0; i < 16; ++i) {
    auto r = g.add_node(fake(), {cur}, "n" + std::to_string(i));
    r.status().throw_if_error();
    cur = (*r)[0];
  }
  CHECK(g.mark_output(cur).ok());
  CHECK(g.finalize().ok());

  auto lv = g.levels(); lv.status().throw_if_error();
  CHECK_EQ(lv->size(), 16u);                 // width 1, depth 16
  auto w = g.max_level_width(); w.status().throw_if_error();
  CHECK_EQ(*w, 1u);

  auto order = g.topological_order(); order.status().throw_if_error();
  const auto lr = g.compute_live_ranges(*order);

  // The graph INPUT never dies -- it is filled by an async H2D the planner does
  // not track, and would be re-read every iteration if it were weights.
  CHECK_EQ(lr[*x].last_use_pos, static_cast<int>(order->size()));
  // The graph OUTPUT never dies either.
  CHECK_EQ(lr[cur].last_use_pos, static_cast<int>(order->size()));
  // Every intermediate is a ping-pong pair: defined at i, last used at i+1.
  int intermediates = 0;
  for (std::size_t t = 0; t < g.num_tensors(); ++t) {
    if (!g.tensor(t).reusable()) continue;
    ++intermediates;
    CHECK_EQ(lr[t].last_use_pos, lr[t].def_pos + 1);
  }
  CHECK_EQ(intermediates, 15);   // 16 outputs, minus the one marked as the output
}

void test_graph_depth_vs_antichain() {
  std::printf("test_graph_depth_vs_antichain\n");
  // The counterexample that refutes the header's old claim that max level width
  // bounds useful concurrency:
  //     a -> b -> c ,  a -> x ,  y isolated
  // depths a=0 y=0 b=1 x=1 c=2  =>  levels {a,y} {b,x} {c}, max width 2.
  // But {b, x, y} is an antichain of THREE: once `a` completes, all three can run.
  Graph g;
  auto in = g.add_input(Shape{2, 2}, DType::kF32, "in"); in.status().throw_if_error();
  auto a = g.add_node(fake(), {*in}, "a"); a.status().throw_if_error();
  auto b = g.add_node(fake(), {(*a)[0]}, "b"); b.status().throw_if_error();
  auto c = g.add_node(fake(), {(*b)[0]}, "c"); c.status().throw_if_error();
  auto x = g.add_node(fake(), {(*a)[0]}, "x"); x.status().throw_if_error();
  auto y = g.add_node(fake(), {*in}, "y"); y.status().throw_if_error();
  CHECK(g.mark_output((*c)[0]).ok());
  CHECK(g.mark_output((*x)[0]).ok());
  CHECK(g.mark_output((*y)[0]).ok());
  CHECK(g.finalize().ok());

  const NodeId Na = 0, Nb = 1, Nc = 2, Nx = 3, Ny = 4;
  CHECK_EQ(g.node(Na).depth, 0);
  CHECK_EQ(g.node(Nb).depth, 1);
  CHECK_EQ(g.node(Nc).depth, 2);
  CHECK_EQ(g.node(Nx).depth, 1);
  CHECK_EQ(g.node(Ny).depth, 0);

  auto w = g.max_level_width(); w.status().throw_if_error();
  CHECK_EQ(*w, 2u);

  // {b, x, y} is mutually unreachable => a genuine antichain of size 3 > width 2.
  const auto r = reachability(g);
  const NodeId anti[3] = {Nb, Nx, Ny};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (i != j) CHECK(!r[anti[i]][anti[j]]);
  std::printf("  max_level_width=2 but {b,x,y} is an antichain of 3 -- the old\n"
              "  \"width bounds useful streams\" claim was false (Dilworth).\n");

  // Slack: the critical path is a->b->c, so those have zero slack; x and y have
  // room to move. This is what explains a level-parallel underperformance.
  CHECK_EQ(g.node(Na).slack, 0);
  CHECK_EQ(g.node(Nb).slack, 0);
  CHECK_EQ(g.node(Nc).slack, 0);
  CHECK_EQ(g.node(Nx).slack, 1);
  CHECK_EQ(g.node(Ny).slack, 2);
}

void test_graph_dead_nodes() {
  std::printf("test_graph_dead_nodes\n");
  Graph g;
  auto in = g.add_input(Shape{2, 2}, DType::kF32, "in"); in.status().throw_if_error();
  auto live = g.add_node(fake(), {*in}, "live"); live.status().throw_if_error();
  auto dead = g.add_node(fake(), {*in}, "dead"); dead.status().throw_if_error();
  auto also = g.add_node(fake(), {(*dead)[0]}, "also_dead"); also.status().throw_if_error();
  CHECK(g.mark_output((*live)[0]).ok());
  CHECK(g.finalize().ok());

  const auto d = g.dead_nodes();
  CHECK_EQ(d.size(), 2u);            // dead + also_dead, transitively
  CHECK(!g.node(0).is_dead);
  CHECK(g.node(1).is_dead);
  CHECK(g.node(2).is_dead);
  // Dead work must not inflate the cost denominator.
  CHECK_EQ(g.total_cost().flops, 4u);   // only `live`, 2x2
}

void test_graph_finalize_errors() {
  std::printf("test_graph_finalize_errors\n");
  {  // no nodes
    Graph g;
    CHECK(!g.finalize().ok());
  }
  {  // no outputs marked
    Graph g;
    auto in = g.add_input(Shape{2, 2}, DType::kF32, "in"); in.status().throw_if_error();
    auto n = g.add_node(fake(), {*in}, "n"); n.status().throw_if_error();
    CHECK(!g.finalize().ok());
  }
  {  // input id out of range, caught at add_node with a naming error
    Graph g;
    auto bad = g.add_node(fake(), {42}, "n");
    CHECK(!bad.ok());
    CHECK(bad.status().message().find("tensor id 42") != std::string::npos);
  }
  {  // mark_output on a bad id -- only reportable because it returns Status now
    Graph g;
    CHECK(!g.mark_output(7).ok());
  }
  {  // an op that refuses its shapes fails the BUILD, not the launch
    Graph g;
    auto in = g.add_input(Shape{2, 2}, DType::kF32, "in"); in.status().throw_if_error();
    auto op = std::make_unique<FakeOp>(Shape{2, 2});
    op->set_fail(true);
    CHECK(!g.add_node(std::move(op), {*in}, "n").ok());
  }
  {  // shape validation: non-positive extent
    Graph g;
    CHECK(!g.add_input(Shape{2, 0}, DType::kF32, "z").ok());
  }
  {  // queries before finalize() are a precondition failure, not UB
    Graph g;
    auto in = g.add_input(Shape{2, 2}, DType::kF32, "in"); in.status().throw_if_error();
    auto n = g.add_node(fake(), {*in}, "n"); n.status().throw_if_error();
    CHECK(g.mark_output((*n)[0]).ok());
    CHECK(!g.topological_order().ok());
    CHECK(!g.levels().ok());
    CHECK(g.finalize().ok());
    CHECK(g.topological_order().ok());
    CHECK(g.finalize().ok());          // idempotent
  }
}

// Property tests over random DAGs. Hand-written cases check what you thought of;
// these check what you did not.
void test_graph_random_dags() {
  std::printf("test_graph_random_dags\n");
  std::uint64_t rng = 0x9E3779B97F4A7C15ull;
  auto next = [&]() { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                      return static_cast<std::uint32_t>(rng >> 33); };

  int graphs = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int n = 1 + static_cast<int>(next() % 12);
    Graph g;
    auto in = g.add_input(Shape{2, 2}, DType::kF32, "in");
    in.status().throw_if_error();
    std::vector<TensorId> avail{*in};
    std::vector<TensorId> outs;
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
      // Inputs drawn only from already-created tensors, so the DAG is acyclic
      // by construction -- which is exactly why cycles are unrepresentable.
      std::vector<TensorId> ins;
      const int want = 1 + static_cast<int>(next() % 2);
      for (int k = 0; k < want; ++k) ins.push_back(avail[next() % avail.size()]);
      std::sort(ins.begin(), ins.end());
      ins.erase(std::unique(ins.begin(), ins.end()), ins.end());
      auto r = g.add_node(fake(), ins, "n" + std::to_string(i));
      if (!r.ok()) { ok = false; break; }
      avail.push_back((*r)[0]);
      outs.push_back((*r)[0]);
    }
    if (!ok) continue;
    // Mark a random subset as outputs, always at least the last node's.
    CHECK(g.mark_output(outs.back()).ok());
    for (TensorId t : outs) if (next() % 4 == 0) CHECK(g.mark_output(t).ok());
    if (!g.finalize().ok()) { CHECK(false); continue; }
    ++graphs;

    auto order = g.topological_order(); order.status().throw_if_error();
    CHECK_EQ(order->size(), g.num_nodes());
    const auto pos = positions(g, *order);

    for (std::size_t u = 0; u < g.num_nodes(); ++u) {
      // (1) every dependency edge points forward in the order
      for (NodeId v : g.node(u).succs) CHECK(pos[u] < pos[v]);
      // (2) the depth recurrence holds exactly
      int expect = 0;
      for (NodeId p : g.node(u).preds) expect = std::max(expect, g.node(p).depth + 1);
      CHECK_EQ(g.node(u).depth, expect);
      // (3) slack is non-negative; zero means on the critical path
      CHECK(g.node(u).slack >= 0);
      CHECK_EQ(g.node(u).alap_depth - g.node(u).depth, g.node(u).slack);
    }

    // (4) each level is an antichain -- the property kLevelParallel relies on
    //     when it round-robins a level's nodes over separate streams.
    auto lv = g.levels(); lv.status().throw_if_error();
    const auto reach = reachability(g);
    for (const auto& level : *lv)
      for (std::size_t i = 0; i < level.size(); ++i)
        for (std::size_t j = i + 1; j < level.size(); ++j) {
          CHECK(!reach[level[i]][level[j]]);
          CHECK(!reach[level[j]][level[i]]);
        }

    // (5) liveness sanity: nothing is used before it is defined, and inputs and
    //     outputs live to the end.
    const auto lr = g.compute_live_ranges(*order);
    for (std::size_t t = 0; t < g.num_tensors(); ++t) {
      if (g.tensor(t).reusable()) CHECK(lr[t].last_use_pos >= lr[t].def_pos);
      if (g.tensor(t).is_graph_input || g.tensor(t).is_graph_output)
        CHECK_EQ(lr[t].last_use_pos, static_cast<int>(order->size()));
    }

    // (6) the sort is deterministic -- kLevelParallel's event count depends on
    //     the node order WITHIN a level, so a nondeterministic sort would make
    //     the published number nondeterministic too.
    auto again = g.topological_order(); again.status().throw_if_error();
    CHECK(*order == *again);
  }
  std::printf("  %d random DAGs checked (order, depth, slack, antichain, liveness,\n"
              "  determinism)\n", graphs);
}

}  // namespace

int main() {
  std::printf("=== MCKE graph host tests (MCKE_WITH_CUDA=%d) ===\n", MCKE_WITH_CUDA);
  test_hb_sequential();
  test_hb_parallel_structure();
  test_topo_liveness_is_unsound();
  test_single_stream_degenerates_to_topo();
  test_hb_rejects_malformed_input();
  test_graph_diamond();
  test_graph_chain_and_liveness();
  test_graph_depth_vs_antichain();
  test_graph_dead_nodes();
  test_graph_finalize_errors();
  test_graph_random_dags();
  std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
