// =============================================================================
//  mcke/graph/graph.hpp
//
//  WHAT: The computation DAG: a tensor table, a node table, and the topology
//        queries the executor needs (topological order, levels, liveness).
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — DAG vs. flat op list.
//
//  A flat list ("run op 0, then 1, then 2") is simpler and is what a naive
//  runtime does. We build a DAG because three things we want are *impossible*
//  with a flat list:
//
//   1. Concurrency. A list encodes a total order, so the runtime cannot know
//      that ops 3 and 4 are independent and may overlap. On a modern GPU a
//      single small kernel often leaves SMs idle; overlapping two independent
//      kernels on two streams is frequently a 1.3-1.8x win on real graphs. You
//      cannot recover that information from a list — you must never have thrown
//      it away.
//   2. Memory reuse. Knowing that tensor t is dead after node 5 lets the
//      planner give its bytes to tensor u created at node 7. Peak memory drops
//      substantially (2-4x on deep chains). Liveness requires the dependency
//      graph.
//   3. Fusion. "Is this BiasAdd's only consumer a GELU?" is a graph query.
//
//  Cost of the DAG: a topological sort (O(V+E), microseconds) and the need to
//  keep the graph acyclic and well-formed. Cheap for what it buys.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — edges are *derived* from tensor def/use, not stored.
//
//  We do not ask the user to declare "node 5 depends on node 3". Instead each
//  tensor records its single producer node, and each node lists its input
//  tensor ids. The dependency edge 3 -> 5 is then implied. This is SSA form
//  (single static assignment) applied to tensors.
//
//  Why: a manually-declared edge list can disagree with the actual data flow —
//  and when it does you get a race that appears only under load. Deriving edges
//  makes that class of bug unrepresentable. It also gives us def/use chains for
//  free, which is what liveness analysis and fusion need. The constraint it
//  imposes: **one producer per tensor** (no in-place mutation). In-place ops are
//  modelled as "reads x, produces a new tensor aliasing x's storage", which the
//  planner handles explicitly — that is a feature, since silent aliasing is the
//  other big source of GPU races.
// =============================================================================
#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mcke/core/status.hpp"
#include "mcke/graph/op.hpp"
#include "mcke/tensor/shape.hpp"

namespace mcke {

// A tensor as the graph knows it *before* memory is assigned: shape + dtype +
// who produces it. The actual Tensor (with storage) is bound at plan/run time.
struct TensorDesc {
  Shape       shape{};
  DType       dtype    = DType::kF32;
  NodeId      producer = kInvalidNode;   // kInvalidNode => graph input / constant
  std::vector<NodeId> consumers;         // filled in by finalize()
  std::string name;                      // debugging + Nsight range labels
  bool        is_graph_input  = false;
  bool        is_graph_output = false;   // outputs must outlive execution: never reused

  // THE REUSE-CANDIDATE RULE, in one place, because the two ends are asymmetric
  // and only one of them was previously written down:
  //   * graph OUTPUTS never die -- they must outlive execution.
  //   * graph INPUTS never die either, and this was previously unstated. They
  //     are filled by set_input()'s async H2D, whose completion the planner does
  //     not track, and if they are weights they are re-read on every iteration.
  //     Treating an input as reusable is a race on iteration 1.
  // Anything else -- produced by a node, not an output -- is fair game.
  [[nodiscard]] bool reusable() const noexcept {
    return producer != kInvalidNode && !is_graph_input && !is_graph_output;
  }
};

struct Node {
  OpPtr                 op;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::string           name;

  // --- Filled in by Graph::finalize() / the planner ---
  std::vector<NodeId> preds;   // derived: producers of my inputs
  std::vector<NodeId> succs;   // derived: consumers of my outputs

  // depth = longest path from any source = the Kahn wave index (identical
  // recurrence, identical base case) = the ASAP schedule.
  int depth = 0;
  // ALAP: longest path to any sink, counted back from the graph's critical-path
  // length. slack = alap_depth - depth is the freedom a smarter scheduler has.
  //
  // Costs one extra backward pass and is not needed to EXECUTE anything -- it is
  // here because ASAP levelisation is a known-mediocre schedule (it front-loads
  // work into early waves, which maximises peak memory), and slack is what turns
  // "kLevelParallel underperformed kChainGreedy" from a shrug into an answer.
  int alap_depth = 0;
  int slack      = 0;
  bool is_dead   = false;   // cannot reach any graph output; see dead_nodes()
};

class Graph {
 public:
  Graph() = default;
  Graph(const Graph&) = delete;              // owns unique_ptr<Op>; moving is fine
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;

  // --- Construction -------------------------------------------------------

  // Declare an externally-provided tensor (weights, activations from the host).
  // Returns StatusOr, not a bare TensorId: a caller can hand this a shape with a
  // non-positive dim or a numel that overflows, and an API with nowhere to put
  // an error message cannot report one. (Free to change -- nothing called this
  // before Phase 4.)
  //
  // HONEST BOUNDARY on what it can catch: rank > kMaxRank is NOT detectable
  // here, because Shape's constructor already clamped it before this function
  // saw it. That case is an assert in debug builds plus
  // Shape::rank_is_representable() for callers building shapes from untrusted
  // dims. Everything detectable from a constructed Shape -- rank 0, dims <= 0,
  // numel overflow, non-contiguous strides -- is checked here.
  [[nodiscard]] StatusOr<TensorId> add_input(Shape shape, DType dtype, std::string name);

  // Add an op. Output TensorDescs are created automatically from
  // op->infer_shapes(), so shapes can never disagree with the kernel's
  // expectations — the op is the single source of truth.
  [[nodiscard]] StatusOr<std::vector<TensorId>> add_node(OpPtr op,
                                                        std::vector<TensorId> inputs,
                                                        std::string name);

  // Returns Status, not void: it can be handed an out-of-range TensorId, and a
  // user-facing API with nowhere to put an error message cannot report one.
  [[nodiscard]] Status mark_output(TensorId t);

  // Derive preds/succs/consumers/depth (ASAP and ALAP), find dead nodes, and
  // validate. Must be called before planning; plan() calls it if you forgot.
  // Returns a useful error naming the node/tensor rather than a bool — the graph
  // is user-facing, so its errors should be readable.
  //
  // WHAT IT CAN AND CANNOT VERIFY -- worth stating precisely, because the answer
  // undersells the design less than the original comment did. That comment
  // claimed finalize() "verifies acyclicity, single-producer, and that every
  // input is defined". Only the third is a real check:
  //
  //   * SINGLE-PRODUCER IS UNREPRESENTABLE. add_node() is the only way to create
  //     a non-input tensor, and it creates fresh TensorDescs with `producer` set
  //     exactly once. There is no alias()/mutate() API, so no tensor can acquire
  //     a second producer.
  //   * CYCLES ARE UNREPRESENTABLE, for the same reason: add_node() can only
  //     reference already-created TensorIds and creates its outputs afterwards,
  //     so every edge points strictly backward in creation order. Kahn will
  //     always succeed.
  //
  // We still run both checks, classified as kInternal rather than
  // kInvalidArgument, because they guard a future alias/mutation API and because
  // an assertion that cannot fire is cheap. But the honest framing is that
  // SSA-by-construction eliminated these two bug classes outright -- which is a
  // stronger claim than the file banner makes, not a weaker one.
  [[nodiscard]] Status finalize();

  [[nodiscard]] bool finalized() const noexcept { return finalized_; }

  // --- Queries ------------------------------------------------------------

  [[nodiscard]] std::size_t num_nodes() const noexcept { return nodes_.size(); }
  [[nodiscard]] std::size_t num_tensors() const noexcept { return tensors_.size(); }
  // Asserted rather than unchecked: an out-of-range id here is UB in a class
  // this header calls user-facing. The Status-returning construction API is
  // where a bad id should be caught; these are the fast path after that.
  [[nodiscard]] const Node& node(NodeId n) const {
    assert(n < nodes_.size() && "Graph::node: NodeId out of range");
    return nodes_[n];
  }
  [[nodiscard]] const TensorDesc& tensor(TensorId t) const {
    assert(t < tensors_.size() && "Graph::tensor: TensorId out of range");
    return tensors_[t];
  }

  // Kahn's algorithm, WITH A FIFO FRONTIER. Returns a valid execution order, or
  // kInvalidArgument if the graph has a cycle.
  //
  // WHY Kahn and not DFS post-order: Kahn processes nodes in *waves* of
  // in-degree zero, which is exactly the information the scheduler wants —
  // every node in one wave is mutually independent and can go on a different
  // stream. A DFS post-order gives a correct topological order but destroys
  // that grouping, so you would have to recompute it. Choose the algorithm
  // whose intermediate state is the thing you need.
  //
  // "FIFO" is part of the algorithm's specification here, not an implementation
  // detail. Depth-monotonicity of the pop order -- the property that makes
  // levels() readable off the frontier -- holds for a FIFO queue and is DESTROYED
  // by a LIFO stack. Concretely, on `a->b->c->d` plus `e->f`: a stack pops `e`
  // (depth 0), pushes `f`, then pops `f` (depth 1) BEFORE `a` (depth 0). Still a
  // valid topological order, no longer grouped by depth. So a plausible "use a
  // stack, better cache locality" refactor would silently break levels(), which
  // is why levels() below is implemented level-synchronously rather than reading
  // the property back off this queue.
  [[nodiscard]] StatusOr<std::vector<NodeId>> topological_order() const;

  // The same sort, grouped: levels()[d] = all nodes at depth d, computed
  // LEVEL-SYNCHRONOUSLY (drain the whole frontier, then advance) so the grouping
  // is structural rather than an emergent property of the queue discipline.
  //
  // This is the ASAP schedule: depth(n) = longest path from any source, which
  // for this recurrence is identical to the Kahn wave index.
  //
  // ON "WIDTH" AS A BOUND -- an earlier version of this comment said "width =
  // max level size is the upper bound on useful concurrent streams". That is
  // FALSE. The true bound on any schedule's concurrency is the maximum ANTICHAIN
  // (Dilworth), which can strictly exceed max level width. Counterexample:
  //     a -> b -> c ,  a -> x ,  y isolated
  // depths a=0 y=0 b=1 x=1 c=2, so levels are {a,y}, {b,x}, {c} and max level
  // width is 2 -- but {b, x, y} is an antichain of 3, and once `a` completes all
  // three genuinely can run at once. kChainGreedy with 3 streams exploits that;
  // kLevelParallel cannot, because it pins `y` into wave 0.
  //
  // So: max level width bounds kLevelParallel. Max antichain bounds ANY schedule.
  // We report the former (exact, cheap) and label it honestly rather than
  // claiming a bound we are not computing.
  [[nodiscard]] StatusOr<std::vector<std::vector<NodeId>>> levels() const;

  // Max level width: the exact concurrency bound for kLevelParallel only.
  [[nodiscard]] StatusOr<std::size_t> max_level_width() const;

  // Nodes whose outputs cannot transitively reach any graph output. NOT an
  // error -- but executing them inflates every benchmark number with work
  // nobody asked for, so the executor drops them by default. Requires a
  // backward reachability pass, which is the one place in this file a backward
  // pass is genuinely necessary.
  [[nodiscard]] std::vector<NodeId> dead_nodes() const;

  // --- Liveness, as POSITIONS IN `order` (not NodeIds), indexed by TensorId.
  //
  //     Computed in a single FORWARD pass. An earlier version of this comment
  //     said "a single backward pass", justified by classical dataflow liveness
  //     -- but that needs a backward FIXPOINT only because "live" in a general
  //     CFG means "reaches a use along some path", and loops force iteration to
  //     convergence. None of that applies here: this is a DAG, in SSA form,
  //     already linearised. A live range is just [def position, max use
  //     position], and going forward the final write to last_use_pos IS the max.
  //     The old justification made a trivial computation look subtle, which
  //     invites more trust in the result than it has earned.
  //
  //     A tensor produced but never consumed would leave last_use_pos == -1
  //     while def_pos == i. That interval LOOKS EMPTY, so a planner would hand
  //     those bytes to another tensor -- while the producing kernel is still
  //     writing them. Hence last_use_pos = max(def_pos, last consumer). Better
  //     still, dead_nodes() removes the case before the planner sees it.
  //
  //     !!! THESE POSITIONS ARE NOT A HAPPENS-BEFORE RELATION. !!!
  //     Non-overlap of two [def_pos, last_use_pos] intervals does NOT imply the
  //     two tensors are never simultaneously live in wall-clock time. A
  //     topological order is one arbitrary linear extension of the dependency
  //     partial order; the executor's happens-before is that partial order plus
  //     STREAM-SERIALISATION edges. Both extend the dependencies, but they
  //     extend them DIFFERENTLY, so interval non-overlap asserts an ordering the
  //     executor never established. Using these intervals directly as a buffer
  //     reuse test is a silent data race that appears only under a parallel
  //     schedule. Use mcke/graph/happens_before.hpp for reuse decisions.
  struct LiveRange { int def_pos = -1; int last_use_pos = -1; };
  [[nodiscard]] std::vector<LiveRange> compute_live_ranges(
      const std::vector<NodeId>& order) const;

  // Sum of ideal cost over all nodes — the denominator for whole-graph
  // efficiency numbers in RESULTS.md.
  [[nodiscard]] OpCost total_cost() const;

  // Graphviz output. Cheap to write, and being able to *see* the DAG (with
  // stream assignment coloured in) is how you catch a scheduling bug in 10
  // seconds instead of an hour.
  [[nodiscard]] std::string to_dot() const;

 private:
  // Kahn without the finalized_ precondition, so finalize() itself can use it
  // to detect cycles before declaring the graph finalized.
  [[nodiscard]] StatusOr<std::vector<NodeId>> kahn_order() const;

  // Uniform "node 7 'gemm2' (Gemm)" / "tensor 5 'tmp' [8192,4096]" strings, so
  // every finalize() error names its subject the same way.
  [[nodiscard]] std::string describe(NodeId n) const;
  [[nodiscard]] std::string describe_tensor(TensorId t) const;

  std::vector<Node>       nodes_;
  std::vector<TensorDesc> tensors_;
  bool                    finalized_ = false;
};

}  // namespace mcke
