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
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "mcke/graph/cost_model.hpp"
#include "mcke/graph/graph.hpp"
#include "mcke/graph/happens_before.hpp"
#include "mcke/graph/executor.hpp"
#include "mcke/graph/memory_plan.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/graph/schedule.hpp"

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
  // A no-op launch that VALIDATES ITS ARGUMENTS. It used to return
  // Unimplemented on the theory that host tests never launch anything -- true
  // until GraphExecutor::run_async() existed, at which point it turned every
  // executor test into a failure that said nothing about the executor.
  //
  // Checking the tensors here is worth more than returning OkStatus blindly: it
  // asserts that the planner bound every input and output to real, distinct
  // storage before the replay loop ever ran.
  [[nodiscard]] Status launch(const mcke::OpContext& ctx,
                              const std::vector<mcke::Tensor>& inputs,
                              const std::vector<mcke::Tensor>& outputs) override {
    for (const auto& t : inputs)
      if (!t.defined()) return mcke::InternalError("FakeOp::launch: undefined input");
    for (const auto& t : outputs)
      if (!t.defined()) return mcke::InternalError("FakeOp::launch: undefined output");
    if (outputs.empty()) return mcke::InternalError("FakeOp::launch: no outputs");
    // An output must never alias an input: the graph is SSA, so a node writing
    // over a tensor it is also reading would mean the planner reused a buffer
    // whose live range had not ended.
    for (const auto& i : inputs)
      for (const auto& o : outputs)
        if (i.data_ptr() && i.data_ptr() == o.data_ptr())
          return mcke::InternalError("FakeOp::launch: output aliases an input");
    (void)ctx;
#if !MCKE_WITH_CUDA
    // A REAL HOST COMPUTATION, and it is what makes the numerics gate testable
    // without a device. In a host-only build raw_device_malloc returns host
    // memory, so data_ptr() is dereferenceable and a graph can actually be
    // executed end to end on the MacBook.
    //
    // It must (a) fully write every output -- an op that leaves bytes untouched
    // would legitimately fail a bit-identity gate under buffer reuse, because
    // the leftovers differ per layout, and that failure would be about the OP,
    // not the scheduler -- and (b) be deterministic, order-independent, and a
    // function of its inputs only.
    ++calls_;
    for (const auto& o : outputs) {
      float* dst = o.data_as<float>();
      if (!dst) return mcke::InternalError("FakeOp::launch: output is not f32");
      const mcke::dim_t n = o.numel();
      for (mcke::dim_t i = 0; i < n; ++i) {
        float acc = 1.0f;
        for (const auto& in : inputs) {
          const float* src = in.data_as<float>();
          if (src && i < in.numel()) acc += src[i];
        }
        dst[i] = acc * 0.5f + (nondet_ ? static_cast<float>(calls_) : 0.0f);
      }
    }
#endif
    return mcke::OkStatus();
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
  // Makes launch() return a different answer on every call -- the simplest
  // stand-in for a race, and the only way to prove the gate is not vacuous.
  void set_nondeterministic(bool v) { nondet_ = v; }

 private:
  Shape out_;
  int   n_out_ = 1;
  bool  fail_  = false;
  mutable bool nondet_ = false;
  mutable int  calls_  = 0;
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


// =============================================================================
//  The four Op subclasses: shapes, costs, and the enum bridges
//
//  The cost assertions use the EXACT byte counts already published in
//  RESULTS.md sec 3a-3d, which were produced by the Phase 3 benches. That makes
//  this a cross-validation rather than a restatement: if the graph layer and the
//  bench layer ever disagree about what an op costs, one of them is wrong and
//  this test says so. Deriving the expected values from the same formula the
//  code uses would assert nothing.
// =============================================================================

void test_op_costs_match_published_results() {
  std::printf("test_op_costs_match_published_results\n");
  const Shape big{8192, 4096};   // the pinned sec 3a/3b/3c shape

  // --- sec 3a, fused bias+activation: (2N + cols) * 4
  {
    mcke::BiasActParams p; p.act = mcke::BiasActParams::Act::kGeluTanh;
    mcke::BiasActOp op(p);
    const auto c = op.cost({big, Shape{4096}});
    CHECK_EQ(c.bytes, 268451840ull);            // RESULTS.md sec 3a, verbatim
    CHECK_EQ(c.flops, 33554432ull * 10ull);     // gelu_tanh = 10 flops/element
  }
  // --- sec 3b, row reduce: (N + rows) * 4
  {
    mcke::ReduceParams p; p.kind = mcke::ReduceParams::Kind::kSum;
    mcke::ReduceOp op(p);
    const auto c = op.cost({big});
    CHECK_EQ(c.bytes, 134250496ull);            // RESULTS.md sec 3b, verbatim
    CHECK_EQ(c.flops, 8192ull * 4095ull);       // rows * (cols - 1) additions
  }
  // --- kMax reports ZERO flops, deliberately: fmaxf is a comparison, not a
  //     floating-point operation in the FMA sense peak_tflops measures. Counting
  //     them would inflate the AI of an op that is hopelessly memory-bound.
  {
    mcke::ReduceParams p; p.kind = mcke::ReduceParams::Kind::kMax;
    mcke::ReduceOp op(p);
    const auto c = op.cost({big});
    CHECK_EQ(c.flops, 0ull);
    CHECK_EQ(c.bytes, 134250496ull);            // same traffic as kSum
  }
  // --- sec 3c, softmax: 2N * 4 (COMPULSORY, not the 3-4 passes actually made)
  {
    mcke::SoftmaxOp op(mcke::SoftmaxParams{});
    const auto c = op.cost({big});
    CHECK_EQ(c.bytes, 268435456ull);
    CHECK_EQ(c.flops, 33554432ull * 5ull);
  }
  // --- sec 3d, GEMM 4096-cubed
  {
    mcke::GemmOp op(mcke::GemmParams{});
    const Shape a{4096, 4096}, b{4096, 4096};
    const auto c = op.cost({a, b});
    CHECK_EQ(c.flops, 137438953472ull);         // 2*M*N*K
    CHECK_EQ(c.bytes, 201326592ull);            // (MK+KN+MN)*4
    // AI ~ 683, twenty times past the T4 ridge point of ~34.5.
    CHECK(c.arithmetic_intensity() > 682.0 && c.arithmetic_intensity() < 683.0);
  }
}

void test_op_infer_shapes() {
  std::printf("test_op_infer_shapes\n");
  {  // GEMM: [M,K] x [K,N] -> [M,N]
    mcke::GemmOp op(mcke::GemmParams{});
    auto r = op.infer_shapes({Shape{7, 5}, Shape{5, 3}});
    r.status().throw_if_error();
    CHECK_EQ(r->size(), 1u);
    CHECK((*r)[0] == Shape({7, 3}));
    // inner dimensions must agree, and the error must SAY the shapes
    auto bad = op.infer_shapes({Shape{7, 5}, Shape{4, 3}});
    CHECK(!bad.ok());
    CHECK(bad.status().message().find("inner dimensions disagree") != std::string::npos);
    CHECK(!op.infer_shapes({Shape{7, 5}}).ok());              // wrong arity
    CHECK(!op.infer_shapes({Shape{7}, Shape{5, 3}}).ok());    // wrong rank
  }
  {  // GEMM: beta != 0 refused at BUILD time, not at launch
    mcke::GemmParams p; p.beta = 1.0f;
    mcke::GemmOp op(p);
    auto r = op.infer_shapes({Shape{4, 4}, Shape{4, 4}});
    CHECK(!r.ok());
    CHECK(r.status().message().find("beta != 0") != std::string::npos);
  }
  {  // GEMM: THE TILE TRAP. kTiledSmem with a defaulted tile is the natural
     // thing to write and is an unsupported pairing; it must fail here rather
     // than inside launch_gemm_f32 on a GPU.
    mcke::GemmParams p; p.variant = mcke::kernels::GemmVariant::kTiledSmem;
    mcke::GemmOp op(p);                       // tile defaults to (128,128,8,8,8)
    CHECK(!op.infer_shapes({Shape{64, 64}, Shape{64, 64}}).ok());

    mcke::GemmParams ok = p;
    ok.tile = mcke::kernels::GemmTile{32, 32, 32, 1, 1};
    mcke::GemmOp op2(ok);
    CHECK(op2.infer_shapes({Shape{64, 64}, Shape{64, 64}}).ok());
  }
  {  // BiasAct: bias broadcasts over the last axis
    mcke::BiasActOp op(mcke::BiasActParams{});
    auto r = op.infer_shapes({Shape{8, 16}, Shape{16}});
    r.status().throw_if_error();
    CHECK((*r)[0] == Shape({8, 16}));
    CHECK(!op.infer_shapes({Shape{8, 16}, Shape{8}}).ok());        // wrong length
    CHECK(!op.infer_shapes({Shape{8, 16}, Shape{4, 4}}).ok());     // wrong rank
  }
  {  // BiasAct: the vector-width precondition is on COLS, not on numel.
     // rows=4, cols=3 has numel 12 (divisible by 4) while every odd row start is
     // misaligned -- the exact mistake kernels.hpp records having made once.
    mcke::BiasActParams p; p.vector_width = 4;
    mcke::BiasActOp op(p);
    CHECK(!op.infer_shapes({Shape{4, 3}, Shape{3}}).ok());
    CHECK(op.infer_shapes({Shape{3, 4}, Shape{4}}).ok());
    mcke::BiasActParams bad = p; bad.vector_width = 3;
    CHECK(!mcke::BiasActOp(bad).infer_shapes({Shape{4, 4}, Shape{4}}).ok());
  }
  {  // Reduce drops the last axis; rank 1 collapses to {1}, never to rank 0
    mcke::ReduceOp op(mcke::ReduceParams{});
    auto r2 = op.infer_shapes({Shape{8, 16}});   r2.status().throw_if_error();
    CHECK((*r2)[0] == Shape({8}));
    auto r3 = op.infer_shapes({Shape{4, 8, 16}}); r3.status().throw_if_error();
    CHECK((*r3)[0] == Shape({4, 8}));            // needs the runtime-rank ctor
    auto r1 = op.infer_shapes({Shape{16}});      r1.status().throw_if_error();
    CHECK((*r1)[0] == Shape({1}));
    mcke::ReduceParams ax; ax.axis = 0;
    CHECK(!mcke::ReduceOp(ax).infer_shapes({Shape{8, 16}}).ok());   // interior axis
  }
  {  // Softmax is shape-preserving, and refuses a flag it does not honour
    mcke::SoftmaxOp op(mcke::SoftmaxParams{});
    auto r = op.infer_shapes({Shape{8, 16}}); r.status().throw_if_error();
    CHECK((*r)[0] == Shape({8, 16}));
    mcke::SoftmaxParams p; p.numerically_stable = false;
    CHECK(!mcke::SoftmaxOp(p).infer_shapes({Shape{8, 16}}).ok());
  }
}

void test_cost_model_roofline() {
  std::printf("test_cost_model_roofline\n");
  mcke::Roofline rl;
  rl.peak_gb_s   = 235.4;    // measured Colab T4, RESULTS.md sec 0
  rl.peak_tflops = 8.130;

  // A 4096-cubed GEMM is COMPUTE bound: 16.9 ms of arithmetic against 0.86 ms of
  // compulsory traffic, so the compute roof is what binds it.
  mcke::GemmOp gemm(mcke::GemmParams{});
  const auto gc = gemm.cost({Shape{4096, 4096}, Shape{4096, 4096}});
  const double g_ms = mcke::plan_cost_ms(gc, rl);
  CHECK(g_ms > 16.0 && g_ms < 17.5);
  CHECK(!mcke::op_is_memory_bound(gc, rl));

  // Softmax at the same pinned shape is MEMORY bound, and the roofline bound
  // (~1.14 ms) lands close to what sec 3c actually measured (1.1-2.2 ms) --
  // which is the point of having a cost model at all.
  mcke::SoftmaxOp sm(mcke::SoftmaxParams{});
  const auto sc = sm.cost({Shape{8192, 4096}});
  const double s_ms = mcke::plan_cost_ms(sc, rl);
  CHECK(s_ms > 1.0 && s_ms < 1.3);
  CHECK(mcke::op_is_memory_bound(sc, rl));

  // THE POINT OF THE COST MODEL, in one assertion: balancing streams by NODE
  // COUNT would call these two nodes equal, when one is ~15x the work of the
  // other. Every Phase 4 benchmark graph mixes exactly these two scales.
  CHECK(g_ms / s_ms > 10.0);

  // Degenerate rooflines return 0 rather than infinity -- a zero denominator
  // must not propagate a garbage cost into a schedule.
  mcke::Roofline zero;
  CHECK_EQ(mcke::plan_cost_ms(gc, zero), 0.0);
}


// =============================================================================
//  Stream assignment: the three policies, event counts, and plan ordering
//
//  Every integer asserted here is a headline number of RESULTS.md sec 4, and
//  every one of them is produced on a laptop -- deterministically, with no
//  timing noise, before any GPU trip.
// =============================================================================

using mcke::SchedulePolicy;
using mcke::StreamAssignment;

mcke::Roofline t4_roofline() {
  mcke::Roofline rl;
  rl.peak_gb_s   = 235.4;
  rl.peak_tflops = 8.130;
  return rl;
}

// --- The pinned benchmark graphs, built once and shared by every test below.
Graph build_diamond() {
  Graph g;
  auto x = g.add_input(Shape{64, 64}, DType::kF32, "x"); x.status().throw_if_error();
  auto a = g.add_node(fake(), {*x}, "A");            a.status().throw_if_error();
  auto b = g.add_node(fake(), {(*a)[0]}, "B");       b.status().throw_if_error();
  auto c = g.add_node(fake(), {(*a)[0]}, "C");       c.status().throw_if_error();
  auto d = g.add_node(fake(), {(*b)[0], (*c)[0]}, "D"); d.status().throw_if_error();
  g.mark_output((*d)[0]).throw_if_error();
  g.finalize().throw_if_error();
  return g;
}

Graph build_chain(int n) {
  Graph g;
  auto x = g.add_input(Shape{64, 64}, DType::kF32, "x"); x.status().throw_if_error();
  TensorId cur = *x;
  for (int i = 0; i < n; ++i) {
    auto r = g.add_node(fake(), {cur}, "n" + std::to_string(i));
    r.status().throw_if_error();
    cur = (*r)[0];
  }
  g.mark_output(cur).throw_if_error();
  g.finalize().throw_if_error();
  return g;
}

// One input -> 4 independent chains of 4 -> 4 outputs. No join node, so no new
// op is needed. This is the ONLY pinned graph with width > 2, which makes it the
// only one on which the two parallel policies can actually differ.
Graph build_fanout(int branches, int depth) {
  Graph g;
  auto x = g.add_input(Shape{64, 64}, DType::kF32, "x"); x.status().throw_if_error();
  for (int b = 0; b < branches; ++b) {
    TensorId cur = *x;
    for (int d = 0; d < depth; ++d) {
      auto r = g.add_node(fake(), {cur}, "b" + std::to_string(b) + "n" + std::to_string(d));
      r.status().throw_if_error();
      cur = (*r)[0];
    }
    g.mark_output(cur).throw_if_error();
  }
  g.finalize().throw_if_error();
  return g;
}

StreamAssignment plan(const Graph& g, SchedulePolicy p, int k) {
  auto sa = mcke::plan_streams(g, p, k, t4_roofline());
  sa.status().throw_if_error();
  return std::move(*sa);
}

void test_schedule_diamond_streams() {
  std::printf("test_schedule_diamond_streams\n");
  const Graph g = build_diamond();   // nodes A=0 B=1 C=2 D=3

  // THE DEFECT-3 ASSERTION, and the reason this test exists.
  //
  // The rule as originally documented was "keep a node on its predecessor's
  // stream when it has exactly one predecessor" -- and on the diamond BOTH B and
  // C have exactly one predecessor. That rule puts both on A's stream, giving a
  // one-stream schedule with zero events and a RESULTS.md row reading
  // "diamond / chain_greedy / 1.00x" that looks like an honest negative result
  // and is a scheduler bug. The donation clause is what prevents it.
  const StreamAssignment cg = plan(g, SchedulePolicy::kChainGreedy, 2);
  CHECK_EQ(cg.stream_of[0], 0);              // A
  CHECK_EQ(cg.stream_of[1], 0);              // B inherits A's stream (free ordering)
  CHECK_EQ(cg.stream_of[2], 1);              // C CANNOT: A already donated
  CHECK_EQ(cg.num_streams_used, 2u);         // <- would be 1 with the broken rule
  CHECK(cg.num_events > 0);                  // <- would be 0 with the broken rule

  const StreamAssignment sq = plan(g, SchedulePolicy::kSequential, 4);
  for (std::size_t i = 0; i < 4; ++i) CHECK_EQ(sq.stream_of[i], 0);
  CHECK_EQ(sq.num_streams_used, 1u);
  // kSequential requests 4 and uses 1. Reporting only the request would claim
  // concurrency the run never had.
  CHECK_EQ(sq.num_streams_requested, 4u);
}

void test_schedule_event_counts() {
  std::printf("test_schedule_event_counts\n");
  struct Row { const char* graph; SchedulePolicy p; const char* name; int k;
               std::size_t rec; std::size_t waits; };

  const Graph dia = build_diamond();
  const Graph ch  = build_chain(16);
  const Graph fan = build_fanout(4, 4);

  auto check = [&](const Graph& g, const char* gname, SchedulePolicy p, const char* pname,
                   int k, std::size_t rec, std::size_t waits) {
    const StreamAssignment sa = plan(g, p, k);
    CHECK_EQ(sa.num_events, rec);
    CHECK_EQ(sa.waits_dedup, waits);
    std::printf("  %-12s %-14s K=%d -> %2zu records, %2zu waits (raw %zu), %zu streams\n",
                gname, pname, k, sa.num_events, sa.waits_dedup, sa.waits_raw,
                sa.num_streams_used);
    // A recorded event must be waited on by somebody, or we paid for nothing.
    std::size_t total_waits = 0;
    for (const auto& w : sa.waits_of) total_waits += w.size();
    CHECK_EQ(total_waits, sa.waits_dedup);
  };

  check(dia, "diamond", SchedulePolicy::kSequential,    "sequential",  4, 0, 0);
  check(dia, "diamond", SchedulePolicy::kLevelParallel, "level_par",   2, 2, 2);
  check(dia, "diamond", SchedulePolicy::kChainGreedy,   "chain_greedy",2, 2, 2);

  check(ch,  "chain16", SchedulePolicy::kSequential,    "sequential",  4, 0, 0);
  check(ch,  "chain16", SchedulePolicy::kLevelParallel, "level_par",   4, 0, 0);
  check(ch,  "chain16", SchedulePolicy::kChainGreedy,   "chain_greedy",4, 0, 0);

  check(fan, "fanout4x4", SchedulePolicy::kSequential,    "sequential",  4, 0, 0);
  check(fan, "fanout4x4", SchedulePolicy::kLevelParallel, "level_par",   4, 12, 36);
  check(fan, "fanout4x4", SchedulePolicy::kChainGreedy,   "chain_greedy",4, 0, 0);

  std::printf("  ^ fanout is the ONLY pinned graph where the two parallel policies\n"
              "    differ: level_parallel manufactures 12 records + 36 waits on a\n"
              "    graph with ZERO cross-stream data edges, because its barrier is\n"
              "    between LEVELS, not between dependencies. chain_greedy pays none.\n");
}

void test_schedule_chain16_is_not_15_events() {
  std::printf("test_schedule_chain16_is_not_15_events\n");
  // A CORRECTION TO THE PLAN, recorded rather than quietly absorbed.
  //
  // The design review predicted chain16/kLevelParallel would cost 15 records and
  // 15 waits. That assumed the round-robin index runs GLOBALLY across levels, so
  // node at depth d lands on stream d % K and every level boundary becomes a
  // cross-stream barrier.
  //
  // We reset the round-robin PER LEVEL instead -- "round-robin the nodes of each
  // level" reads that way, and it is the better behaviour: a width-1 graph has
  // exactly one node per level, index 0 every time, so the whole chain stays on
  // stream 0 and costs nothing. Manufacturing 15 barriers on a chain with no
  // parallelism to exploit would be a policy defect, not a policy cost.
  const Graph ch = build_chain(16);
  const StreamAssignment lp = plan(ch, SchedulePolicy::kLevelParallel, 4);
  CHECK_EQ(lp.num_streams_used, 1u);
  CHECK_EQ(lp.num_events, 0u);
  CHECK_EQ(lp.waits_dedup, 0u);
  // The consequence: on a chain the two parallel policies are INDISTINGUISHABLE.
  // That is why the fanout graph had to be added -- without it, every pinned
  // graph has width <= 2 and the K=4 columns of RESULTS.md sec 4 are decorative.
  const StreamAssignment cg = plan(ch, SchedulePolicy::kChainGreedy, 4);
  CHECK_EQ(cg.num_events, lp.num_events);
  CHECK_EQ(cg.num_streams_used, lp.num_streams_used);
}

void test_schedule_wait_dedup() {
  std::printf("test_schedule_wait_dedup\n");
  // A node with several successors on ONE other stream needs one record and --
  // after dedup -- one wait, not one per successor. cudaStreamWaitEvent orders
  // everything SUBSEQUENTLY enqueued on that stream, not just the next launch.
  Graph g;
  auto x = g.add_input(Shape{64, 64}, DType::kF32, "x"); x.status().throw_if_error();
  auto src = g.add_node(fake(), {*x}, "src"); src.status().throw_if_error();
  // Three consumers, plus a long chain to occupy stream 0 so the consumers get
  // pushed onto another stream.
  TensorId cur = (*src)[0];
  for (int i = 0; i < 3; ++i) {
    auto r = g.add_node(fake(Shape{512, 512}), {cur}, "heavy" + std::to_string(i));
    r.status().throw_if_error();
    cur = (*r)[0];
  }
  g.mark_output(cur).throw_if_error();
  std::vector<TensorId> cons;
  for (int i = 0; i < 3; ++i) {
    auto r = g.add_node(fake(), {(*src)[0]}, "c" + std::to_string(i));
    r.status().throw_if_error();
    g.mark_output((*r)[0]).throw_if_error();
    cons.push_back((*r)[0]);
  }
  g.finalize().throw_if_error();

  const StreamAssignment cg = plan(g, SchedulePolicy::kChainGreedy, 2);
  CHECK(cg.waits_dedup <= cg.waits_raw);
  // Whatever the placement, no stream may wait on the same event twice.
  for (std::size_t s = 0; s < cg.num_streams_used; ++s) {
    std::vector<std::uint32_t> seen;
    for (std::size_t i = 0; i < cg.nodes.size(); ++i) {
      if (cg.stream_of[i] != s) continue;
      for (std::uint32_t e : cg.waits_of[i]) {
        CHECK(std::find(seen.begin(), seen.end(), e) == seen.end());
        seen.push_back(e);
      }
    }
  }
}

void test_schedule_record_elision() {
  std::printf("test_schedule_record_elision\n");
  // record_of must be kNoEventId whenever every successor shares the stream.
  // Not recording an unnecessary event is real: a cudaEventRecord is ~0.3-0.5 us
  // of host time, and on a launch-bound graph that is the critical path.
  for (int k : {1, 2, 4}) {
    for (auto p : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                   SchedulePolicy::kChainGreedy}) {
      const Graph g = build_fanout(4, 4);
      const StreamAssignment sa = plan(g, p, k);
      for (std::size_t i = 0; i < sa.nodes.size(); ++i) {
        bool any_cross = false;
        for (std::size_t j = 0; j < sa.nodes.size(); ++j)
          for (std::uint32_t e : sa.waits_of[j])
            if (e == sa.record_of[i] && sa.record_of[i] != mcke::kNoEventId)
              any_cross = any_cross || (sa.stream_of[j] != sa.stream_of[i]);
        // An event that is recorded must be waited on from another stream.
        if (sa.record_of[i] != mcke::kNoEventId) CHECK(any_cross);
      }
    }
  }
}

// The single highest-value property in the phase per unit of effort: the
// scheduler's whole contract, checked over thousands of graphs with no GPU.
void test_schedule_ordering_property() {
  std::printf("test_schedule_ordering_property\n");
  std::uint64_t rng = 0xD1B54A32D192ED03ull;
  auto next = [&]() { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                      return static_cast<std::uint32_t>(rng >> 33); };

  int planned = 0;
  for (int trial = 0; trial < 250; ++trial) {
    const int n = 1 + static_cast<int>(next() % 14);
    Graph g;
    auto in = g.add_input(Shape{16, 16}, DType::kF32, "in"); in.status().throw_if_error();
    std::vector<TensorId> avail{*in}, outs;
    for (int i = 0; i < n; ++i) {
      std::vector<TensorId> ins;
      const int want = 1 + static_cast<int>(next() % 3);
      for (int k = 0; k < want; ++k) ins.push_back(avail[next() % avail.size()]);
      std::sort(ins.begin(), ins.end());
      ins.erase(std::unique(ins.begin(), ins.end()), ins.end());
      auto r = g.add_node(fake(Shape{16, 16}), ins, "n" + std::to_string(i));
      r.status().throw_if_error();
      avail.push_back((*r)[0]);
      outs.push_back((*r)[0]);
    }
    g.mark_output(outs.back()).throw_if_error();
    for (TensorId t : outs) if (next() % 3 == 0) g.mark_output(t).throw_if_error();
    g.finalize().throw_if_error();

    for (auto p : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                   SchedulePolicy::kChainGreedy}) {
      for (int k : {1, 2, 4, 8}) {
        const StreamAssignment sa = plan(g, p, k);
        // (1) EVERY data dependency is realised as same-stream order or an event.
        const Status ord = mcke::verify_plan_ordering(g, sa);
        CHECK(ord.ok());
        if (!ord.ok()) {
          const char* pn = (p == SchedulePolicy::kSequential) ? "sequential"
                         : (p == SchedulePolicy::kLevelParallel) ? "level_par"
                                                                 : "chain_greedy";
          std::printf("    [policy=%s K=%d trial=%d]\n    %s\n    %s",
                      pn, k, trial, ord.message().c_str(), sa.describe().c_str());
        }
        // (2) no dead node is ever scheduled
        for (NodeId nd : sa.nodes) CHECK(!g.node(nd).is_dead);
        // (3) streams actually used are a dense 0..used-1, which HappensBefore
        //     requires and which the stream-compaction pass exists to guarantee
        CHECK(sa.num_streams_used >= 1u);
        CHECK(sa.num_streams_used <= static_cast<std::size_t>(k));
        for (std::size_t i = 0; i < sa.nodes.size(); ++i)
          CHECK(sa.stream_of[i] < sa.num_streams_used);
        // (4) kSequential is always exactly one stream and zero events
        if (p == SchedulePolicy::kSequential) {
          CHECK_EQ(sa.num_streams_used, 1u);
          CHECK_EQ(sa.num_events, 0u);
        }
        // (5) K=1 forces one stream for EVERY policy, hence zero events
        if (k == 1) {
          CHECK_EQ(sa.num_streams_used, 1u);
          CHECK_EQ(sa.num_events, 0u);
        }
        ++planned;
      }
    }
  }
  std::printf("  %d schedules verified (3 policies x 4 stream counts x 250 DAGs)\n", planned);
}


// =============================================================================
//  The memory planner: interference, packing, and the unsafe arm
// =============================================================================

using mcke::MemoryPlan;
using mcke::MemoryPolicy;

MemoryPlan plan_mem(const Graph& g, const StreamAssignment& sa, MemoryPolicy pol) {
  auto hb = sa.happens_before();
  hb.status().throw_if_error();
  auto mp = mcke::plan_memory(g, sa, *hb, pol);
  mp.status().throw_if_error();
  return std::move(*mp);
}

// The 5-node graph from happens_before.hpp's banner, as a real Graph:
//   X -> {N0 -> N1, N2 -> N3} -> N4
// Two chains of unequal reach joined at the end. By TOPOLOGICAL POSITION,
// tensor `a` (def by N0, last read by N1) and tensor `d` (def by N3) have
// disjoint live ranges, so a naive planner shares their buffer. Whether that is
// safe depends entirely on the schedule.
Graph build_two_chain_join() {
  Graph g;
  auto x  = g.add_input(Shape{64, 64}, DType::kF32, "X"); x.status().throw_if_error();
  auto n0 = g.add_node(fake(), {*x}, "N0_a");        n0.status().throw_if_error();
  auto n2 = g.add_node(fake(), {*x}, "N2_b");        n2.status().throw_if_error();
  auto n1 = g.add_node(fake(), {(*n0)[0]}, "N1_c");  n1.status().throw_if_error();
  auto n3 = g.add_node(fake(), {(*n2)[0]}, "N3_d");  n3.status().throw_if_error();
  auto n4 = g.add_node(fake(), {(*n1)[0], (*n3)[0]}, "N4_out"); n4.status().throw_if_error();
  g.mark_output((*n4)[0]).throw_if_error();
  g.finalize().throw_if_error();
  return g;
}

void test_memory_chain16_exact_bytes() {
  std::printf("test_memory_chain16_exact_bytes\n");
  // chain x16 at 4096x4096 f32: every tensor is exactly 64 MiB, so the whole
  // table is hand-computable and asserted as exact integers.
  Graph g;
  auto x = g.add_input(Shape{4096, 4096}, DType::kF32, "x"); x.status().throw_if_error();
  TensorId cur = *x;
  for (int i = 0; i < 16; ++i) {
    auto r = g.add_node(fake(Shape{4096, 4096}), {cur}, "n" + std::to_string(i));
    r.status().throw_if_error();
    cur = (*r)[0];
  }
  g.mark_output(cur).throw_if_error();
  g.finalize().throw_if_error();

  const std::size_t MiB64 = 64ull * 1024 * 1024;
  const StreamAssignment sa = plan(g, SchedulePolicy::kSequential, 1);

  const MemoryPlan naive = plan_mem(g, sa, MemoryPolicy::kAllocPerTensor);
  CHECK_EQ(naive.naive_bytes, 17ull * MiB64);   // 1 input + 16 produced
  CHECK_EQ(naive.arena_bytes, 17ull * MiB64);   // no reuse: arena == naive
  CHECK_EQ(naive.buffers_used, 17u);

  const MemoryPlan reuse = plan_mem(g, sa, MemoryPolicy::kReuseHappensBefore);
  // FOUR buffers, and each one is there for a stated reason:
  //   the graph INPUT  -- never dies (async H2D the planner does not track)
  //   the graph OUTPUT -- must outlive execution
  //   two ping-pong buffers for the 15 intermediates, because consecutive
  //   intermediates overlap (t_i is live [i, i+1], t_{i+1} is [i+1, i+2]) while
  //   t_i and t_{i+2} do not.
  CHECK_EQ(reuse.buffers_used, 4u);
  CHECK_EQ(reuse.arena_bytes, 4ull * MiB64);
  CHECK_EQ(reuse.naive_bytes, 17ull * MiB64);
  std::printf("  chain16 @4096^2: naive %zu MiB -> arena %zu MiB (%.2fx, %zu buffers)\n",
              reuse.naive_bytes / (1024 * 1024), reuse.arena_bytes / (1024 * 1024),
              double(reuse.naive_bytes) / double(reuse.arena_bytes), reuse.buffers_used);

  // Every offset must be device-aligned, or a Tensor::slice() view breaks the
  // coalescing assumption every Phase 3 kernel was tuned under -- silently,
  // with no error, just quietly worse GEMM numbers.
  for (std::size_t t = 0; t < g.num_tensors(); ++t)
    if (reuse.offset_of[t] != mcke::kNoOffset)
      CHECK_EQ(reuse.offset_of[t] % mcke::kDeviceAlignment, 0u);
}

// THE HEADLINE DEMONSTRATION: the unsafe arm is caught on a laptop.
void test_memory_unsafe_arm_is_flagged() {
  std::printf("test_memory_unsafe_arm_is_flagged\n");
  const Graph g = build_two_chain_join();

  auto check = [&](SchedulePolicy sp, const char* spn, MemoryPolicy mp, const char* mpn) {
    const StreamAssignment sa = plan(g, sp, 2);
    auto hb = sa.happens_before(); hb.status().throw_if_error();
    const MemoryPlan m = plan_mem(g, sa, mp);
    const Status race = mcke::verify_no_buffer_races(g, sa, *hb, m);
    std::printf("  %-13s x %-24s -> %s\n", spn, mpn, race.ok() ? "clean" : "RACE DETECTED");
    if (!race.ok()) std::printf("      %s\n", race.message().c_str());
    return race.ok();
  };

  // The naive planner is SAFE under one stream -- happens-before is total there,
  // so it degenerates to the interval test and agrees with it exactly.
  CHECK(check(SchedulePolicy::kSequential, "sequential", MemoryPolicy::kReuseTopoNaive,
              "reuse_topo_naive"));
  // ...and UNSAFE under both parallel policies, including kLevelParallel, whose
  // inter-level barrier is intuitively supposed to prevent exactly this.
  CHECK(!check(SchedulePolicy::kChainGreedy, "chain_greedy", MemoryPolicy::kReuseTopoNaive,
               "reuse_topo_naive"));
  CHECK(!check(SchedulePolicy::kLevelParallel, "level_par", MemoryPolicy::kReuseTopoNaive,
               "reuse_topo_naive"));

  // The correct policy is clean everywhere, which is the whole point.
  for (auto sp : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                  SchedulePolicy::kChainGreedy})
    CHECK(check(sp, "any", MemoryPolicy::kReuseHappensBefore, "reuse_happens_before"));
  // And so is the conservative one.
  for (auto sp : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                  SchedulePolicy::kChainGreedy})
    CHECK(check(sp, "any", MemoryPolicy::kReuseSameStream, "reuse_same_stream"));

  std::printf("  ^ found with no GPU, deterministically. The runtime numerics gate\n"
              "    would catch this too -- but only probabilistically, only under\n"
              "    load, and only on hardware.\n");
}

void test_memory_single_stream_degeneracy() {
  std::printf("test_memory_single_stream_degeneracy\n");
  // Under one stream happens-before is TOTAL, so the sound planner and the naive
  // one must produce byte-identical layouts. That equivalence is why there is one
  // planner rather than a fast-unsafe path and a slow-safe one.
  for (int trial = 0; trial < 40; ++trial) {
    const Graph g = (trial % 2) ? build_diamond() : build_fanout(3, 3);
    const StreamAssignment sa = plan(g, SchedulePolicy::kSequential, 1);
    const MemoryPlan a = plan_mem(g, sa, MemoryPolicy::kReuseTopoNaive);
    const MemoryPlan b = plan_mem(g, sa, MemoryPolicy::kReuseHappensBefore);
    CHECK_EQ(a.arena_bytes, b.arena_bytes);
    CHECK_EQ(a.buffers_used, b.buffers_used);
    for (std::size_t t = 0; t < g.num_tensors(); ++t)
      CHECK_EQ(a.offset_of[t], b.offset_of[t]);
  }
}

void test_memory_peak_rises_with_streams() {
  std::printf("test_memory_peak_rises_with_streams\n");
  // THE PREDICTION, now measurable with no GPU: parallelism and memory reuse are
  // in direct tension. More concurrent streams means fewer pairs of tensors are
  // ordered, which means fewer reuse opportunities, which means a bigger arena.
  const Graph g = build_fanout(4, 4);
  std::size_t prev = 0;
  std::printf("  fanout4x4, chain_greedy + reuse_happens_before:\n");
  for (int k : {1, 2, 4}) {
    const StreamAssignment sa = plan(g, SchedulePolicy::kChainGreedy, k);
    const MemoryPlan m = plan_mem(g, sa, MemoryPolicy::kReuseHappensBefore);
    std::printf("    K=%d -> %zu streams, arena %zu B, %zu buffers\n",
                k, sa.num_streams_used, m.arena_bytes, m.buffers_used);
    CHECK(m.arena_bytes >= prev);          // monotone non-decreasing
    CHECK(m.arena_bytes <= m.naive_bytes); // reuse never costs more than none
    prev = m.arena_bytes;
  }
  std::printf("  ^ peak memory is a function of (graph, SCHEDULE, memory policy,\n"
              "    NUM_STREAMS) -- not of (graph, memory policy). A RESULTS row that\n"
              "    omits the last two coordinates is uninterpretable.\n");
}

// The fuzz loop: the sound policies must NEVER race, and the unsafe one must
// sometimes race -- a demonstration that never fires is not a demonstration.
void test_memory_race_fuzz() {
  std::printf("test_memory_race_fuzz\n");
  std::uint64_t rng = 0x2545F4914F6CDD1Dull;
  auto next = [&]() { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                      return static_cast<std::uint32_t>(rng >> 33); };

  int checked = 0, naive_races = 0;
  for (int trial = 0; trial < 150; ++trial) {
    const int n = 2 + static_cast<int>(next() % 10);
    Graph g;
    auto in = g.add_input(Shape{32, 32}, DType::kF32, "in"); in.status().throw_if_error();
    std::vector<TensorId> avail{*in}, outs;
    for (int i = 0; i < n; ++i) {
      std::vector<TensorId> ins;
      const int want = 1 + static_cast<int>(next() % 2);
      for (int k = 0; k < want; ++k) ins.push_back(avail[next() % avail.size()]);
      std::sort(ins.begin(), ins.end());
      ins.erase(std::unique(ins.begin(), ins.end()), ins.end());
      auto r = g.add_node(fake(Shape{32, 32}), ins, "n" + std::to_string(i));
      r.status().throw_if_error();
      avail.push_back((*r)[0]);
      outs.push_back((*r)[0]);
    }
    g.mark_output(outs.back()).throw_if_error();
    g.finalize().throw_if_error();

    for (auto sp : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                    SchedulePolicy::kChainGreedy}) {
      for (int k : {1, 2, 4}) {
        const StreamAssignment sa = plan(g, sp, k);
        auto hb = sa.happens_before(); hb.status().throw_if_error();
        for (auto mp : {MemoryPolicy::kAllocPerTensor, MemoryPolicy::kReuseSameStream,
                        MemoryPolicy::kReuseHappensBefore}) {
          const MemoryPlan m = plan_mem(g, sa, mp);
          const Status race = mcke::verify_no_buffer_races(g, sa, *hb, m);
          CHECK(race.ok());
          if (!race.ok()) std::printf("    %s\n", race.message().c_str());
          // Alignment is not optional: an unaligned slice silently degrades
          // every Phase 3 kernel's coalescing.
          for (std::size_t t = 0; t < g.num_tensors(); ++t)
            if (m.offset_of[t] != mcke::kNoOffset)
              CHECK_EQ(m.offset_of[t] % mcke::kDeviceAlignment, 0u);
          ++checked;
        }
        const MemoryPlan bad = plan_mem(g, sa, MemoryPolicy::kReuseTopoNaive);
        if (!mcke::verify_no_buffer_races(g, sa, *hb, bad).ok()) ++naive_races;
      }
    }
  }
  std::printf("  %d sound plans verified race-free; the naive arm raced in %d cases\n",
              checked, naive_races);
  // A demonstration that never fires is not a demonstration.
  CHECK(naive_races > 0);
}


// =============================================================================
//  GraphExecutor, end to end, on a machine with no GPU
//
//  This works because a host-only build's raw_device_malloc returns HOST memory
//  and every rt::StreamHandle is null -- so plan() exercises its entire real
//  code path (scheduling, happens-before, arena packing, stream/event creation,
//  tensor binding) and run_async() walks the real replay loop.
//
//  Its honest limit, stated the way tests/test_stream_safety.cu states its own:
//  with MCKE_WITH_CUDA=0 all "streams" are the same null stream, so there is no
//  concurrency and this can NOT exercise a race. Race-freedom is proven by
//  verify_no_buffer_races on the plan; this proves the plan is BUILDABLE and the
//  replay loop is well-formed.
// =============================================================================

mcke::DeviceInfo host_device() {
  mcke::DeviceInfo d;
  d.name = "host (synthetic)";
  d.sm_count = 1;
  d.max_threads_per_sm = 1024;
  d.max_threads_per_block = 1024;
  d.memory_bus_width_bits = 256;
  d.memory_clock_khz = 1000000;
  return d;
}

void test_executor_end_to_end() {
  std::printf("test_executor_end_to_end\n");
  for (auto sp : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                  SchedulePolicy::kChainGreedy}) {
    for (auto mp : {MemoryPolicy::kAllocPerTensor, MemoryPolicy::kReuseSameStream,
                    MemoryPolicy::kReuseHappensBefore}) {
      Graph g = build_diamond();
      const TensorId in_id = 0;   // the graph input is always tensor 0 here
      TensorId out_id = 0;
      for (std::size_t t = 0; t < g.num_tensors(); ++t)
        if (g.tensor(t).is_graph_output) out_id = static_cast<TensorId>(t);

      mcke::RawDeviceAllocator alloc;
      mcke::ExecutorOptions opts;
      opts.schedule = sp;
      opts.memory   = mp;
      opts.num_streams = 2;
      mcke::GraphExecutor ex(std::move(g), alloc, host_device(), opts);

      // plan() runs verify_plan_ordering AND verify_no_buffer_races internally,
      // so a successful plan is already a proof that the schedule realises every
      // dependency and that no two overlapping buffers are unordered.
      const Status planned = ex.plan();
      CHECK(planned.ok());
      if (!planned.ok()) { std::printf("    %s\n", planned.message().c_str()); continue; }

      const mcke::ExecutionPlan& pl = ex.plan_ref();
      CHECK(pl.peak_memory_bytes() > 0);
      CHECK(pl.peak_memory_bytes() <= pl.naive_memory_bytes());
      // kAllocPerTensor is the no-reuse baseline BY DEFINITION, so its arena must
      // equal the naive figure exactly. If it ever does not, naive_bytes is being
      // computed differently from the thing it is the baseline for.
      if (mp == MemoryPolicy::kAllocPerTensor)
        CHECK_EQ(pl.peak_memory_bytes(), pl.naive_memory_bytes());

      // Fork/join is a FIXED per-iteration cost of the async contract, not
      // something attributable to the schedule policy, so it is reported
      // separately: K records + 2(K-1) waits.
      const std::size_t k = pl.num_streams();
      CHECK_EQ(pl.forkjoin_events(), k <= 1 ? 0u : k + 2 * (k - 1));
      if (sp == SchedulePolicy::kSequential) {
        CHECK_EQ(pl.num_streams(), 1u);
        CHECK_EQ(pl.intra_events(), 0u);
        CHECK_EQ(pl.forkjoin_events(), 0u);   // one stream needs no fork at all
      }

      std::vector<float> host(64 * 64, 1.5f);
      CHECK(ex.set_input(in_id, host.data(), host.size() * sizeof(float)).ok());
      // Three back-to-back run_async calls with no intervening sync: the exact
      // pattern the header advertises, and the one the fork/join exists to make
      // safe under buffer reuse.
      for (int i = 0; i < 3; ++i) CHECK(ex.run_async().ok());
      CHECK(ex.synchronize().ok());

      auto out = ex.output(out_id);
      CHECK(out.ok());
      if (out.ok()) CHECK(out->defined());
      // Asking for a non-output must fail rather than hand back an intermediate
      // whose buffer is about to be reused by the next iteration.
      CHECK(!ex.output(in_id).ok());
    }
  }
}

void test_executor_preconditions() {
  std::printf("test_executor_preconditions\n");
  Graph g = build_diamond();
  mcke::RawDeviceAllocator alloc;
  mcke::GraphExecutor ex(std::move(g), alloc, host_device(), mcke::ExecutorOptions{});
  // Every entry point must refuse before plan(), rather than dereference an
  // empty schedule.
  float dummy = 0.0f;
  CHECK(!ex.set_input(0, &dummy, sizeof(float)).ok());
  CHECK(!ex.run_async().ok());
  CHECK(!ex.synchronize().ok());
  CHECK(!ex.output(0).ok());
  CHECK(ex.plan().ok());
  CHECK(ex.plan().ok());               // idempotent
  CHECK(ex.run_async().ok());
  CHECK(ex.synchronize().ok());
  // set_input on a tensor that is not a declared graph input.
  CHECK(!ex.set_input(1, &dummy, sizeof(float)).ok());
}

void test_executor_poison_and_arena_alignment() {
  std::printf("test_executor_poison_and_arena_alignment\n");
  Graph g = build_chain(6);
  mcke::RawDeviceAllocator alloc;
  mcke::ExecutorOptions opts;
  opts.poison_buffers = true;          // fills the arena with a per-iteration pattern
  opts.num_streams = 1;
  mcke::GraphExecutor ex(std::move(g), alloc, host_device(), opts);
  CHECK(ex.plan().ok());
  for (int i = 0; i < 3; ++i) CHECK(ex.run_async().ok());
  CHECK(ex.synchronize().ok());

  // Every bound tensor must sit at a device-aligned address inside the arena.
  const mcke::MemoryPlan& mp = ex.plan_ref().memory_plan();
  for (std::size_t t = 0; t < mp.offset_of.size(); ++t)
    if (mp.offset_of[t] != mcke::kNoOffset)
      CHECK_EQ(mp.offset_of[t] % mcke::kDeviceAlignment, 0u);
}


// =============================================================================
//  The numerics gate
// =============================================================================

void test_bit_mismatch_helper() {
  std::printf("test_bit_mismatch_helper\n");
  // The two cases that quietly ruin a bit-identity gate compared as float.
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  CHECK_EQ(mcke::first_bit_mismatch(a, b, sizeof(a)), sizeof(a));   // identical

  // NaN vs the same NaN: a float compare says "different" (NaN != NaN) and would
  // report a FALSE FAILURE. Bitwise says identical, which is the truth.
  const std::uint32_t qnan = 0x7FC00000u;
  std::memcpy(&a[1], &qnan, 4);
  std::memcpy(&b[1], &qnan, 4);
  CHECK_EQ(mcke::first_bit_mismatch(a, b, sizeof(a)), sizeof(a));

  // -0.0 vs +0.0: a float compare says "equal" and would report a FALSE PASS.
  // They differ in the sign bit, and under a race that difference is real signal.
  a[2] = 0.0f;
  b[2] = -0.0f;
  CHECK(a[2] == b[2]);                                    // ...as floats, equal
  CHECK_EQ(mcke::first_bit_mismatch(a, b, sizeof(a)), 8u); // ...bitwise, word 2
}

void test_numerics_gate_on_host() {
  std::printf("test_numerics_gate_on_host\n");
#if !MCKE_WITH_CUDA
  // The full {3 schedules} x {3 memory policies} matrix, executed for real on
  // the laptop because FakeOp has a host compute path. This is the gate's own
  // logic under test -- the plumbing, the re-planning, the input replay, the
  // bitwise compare -- so that a Colab session debugs KERNELS, not the harness.
  Graph g = build_fanout(3, 3);
  const TensorId in_id = 0;

  mcke::RawDeviceAllocator alloc;
  mcke::ExecutorOptions opts;
  opts.validate_numerics = true;      // also switches on input caching
  opts.num_streams = 3;
  mcke::GraphExecutor ex(std::move(g), alloc, host_device(), opts);
  CHECK(ex.plan().ok());

  std::vector<float> host(64 * 64);
  for (std::size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(i % 97) * 0.25f;
  CHECK(ex.set_input(in_id, host.data(), host.size() * sizeof(float)).ok());

  auto res = ex.validate_numerics(/*repeats=*/4);
  CHECK(res.ok());
  if (res.ok()) {
    CHECK(res->passed);
    CHECK_EQ(res->configs_compared, 9);      // 3 schedules x 3 memory policies
    CHECK(res->elements_compared > 0);
    std::printf("  gate: %s, %d configs x %d repeats, %zu elements compared\n",
                res->passed ? "PASS" : "FAIL", res->configs_compared, res->repeats,
                res->elements_compared);
    if (!res->passed) std::printf("    %s\n", res->detail.c_str());
  }
#else
  std::printf("  (skipped: needs the host compute backend, MCKE_WITH_CUDA=0)\n");
#endif
}

void test_numerics_gate_catches_nondeterminism() {
  std::printf("test_numerics_gate_catches_nondeterminism\n");
#if !MCKE_WITH_CUDA
  // A GATE THAT NEVER FIRES IS NOT A GATE. One op is made to return a different
  // answer on every call -- the cheapest possible stand-in for a race -- and the
  // gate must report it rather than pass.
  Graph g;
  auto x = g.add_input(Shape{32, 32}, DType::kF32, "x"); x.status().throw_if_error();
  auto op = std::make_unique<FakeOp>(Shape{32, 32});
  op->set_nondeterministic(true);
  auto n = g.add_node(std::move(op), {*x}, "flaky"); n.status().throw_if_error();
  g.mark_output((*n)[0]).throw_if_error();
  g.finalize().throw_if_error();

  mcke::RawDeviceAllocator alloc;
  mcke::ExecutorOptions opts;
  opts.validate_numerics = true;
  mcke::GraphExecutor ex(std::move(g), alloc, host_device(), opts);
  CHECK(ex.plan().ok());
  std::vector<float> host(32 * 32, 2.0f);
  CHECK(ex.set_input(0, host.data(), host.size() * sizeof(float)).ok());

  auto res = ex.validate_numerics(/*repeats=*/3);
  CHECK(res.ok());
  if (res.ok()) {
    CHECK(!res->passed);
    std::printf("  gate correctly FAILED: %s\n", res->detail.c_str());
    // It must be caught as run-to-run variance WITHIN a configuration, which is
    // how a real race presents -- not as a disagreement between policies.
    CHECK(res->detail.find("run-to-run") != std::string::npos ||
          res->detail.find("NONDETERMINISTIC") != std::string::npos);
  }
#else
  std::printf("  (skipped: needs the host compute backend)\n");
#endif
}

void test_numerics_gate_preconditions() {
  std::printf("test_numerics_gate_preconditions\n");
  Graph g = build_diamond();
  mcke::RawDeviceAllocator alloc;
  mcke::ExecutorOptions opts;                 // validate_numerics defaults to false
  mcke::GraphExecutor ex(std::move(g), alloc, host_device(), opts);
  CHECK(ex.plan().ok());
  // Refuses without the option, because set_input only caches host bytes in that
  // mode and the gate must re-feed identical inputs after re-planning.
  CHECK(!ex.validate_numerics(2).ok());
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
  test_op_costs_match_published_results();
  test_op_infer_shapes();
  test_cost_model_roofline();
  test_schedule_diamond_streams();
  test_schedule_event_counts();
  test_schedule_chain16_is_not_15_events();
  test_schedule_wait_dedup();
  test_schedule_record_elision();
  test_schedule_ordering_property();
  test_memory_chain16_exact_bytes();
  test_memory_unsafe_arm_is_flagged();
  test_memory_single_stream_degeneracy();
  test_memory_peak_rises_with_streams();
  test_memory_race_fuzz();
  test_executor_end_to_end();
  test_executor_preconditions();
  test_executor_poison_and_arena_alignment();
  test_bit_mismatch_helper();
  test_numerics_gate_on_host();
  test_numerics_gate_catches_nondeterminism();
  test_numerics_gate_preconditions();
  std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
