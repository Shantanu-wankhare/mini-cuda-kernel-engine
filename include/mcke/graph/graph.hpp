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
};

struct Node {
  OpPtr                 op;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::string           name;

  // --- Filled in by Graph::finalize() / the planner ---
  std::vector<NodeId> preds;   // derived: producers of my inputs
  std::vector<NodeId> succs;   // derived: consumers of my outputs
  int                 depth = 0;   // longest path from any source; = "level"
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
  TensorId add_input(Shape shape, DType dtype, std::string name);

  // Add an op. Output TensorDescs are created automatically from
  // op->infer_shapes(), so shapes can never disagree with the kernel's
  // expectations — the op is the single source of truth.
  [[nodiscard]] StatusOr<std::vector<TensorId>> add_node(OpPtr op,
                                                        std::vector<TensorId> inputs,
                                                        std::string name);

  void mark_output(TensorId t);

  // Derive preds/succs/consumers/depth; verify acyclicity, single-producer, and
  // that every input is defined. Must be called before planning. Returns a
  // useful error (which node, which tensor) rather than a bool — the graph is
  // user-facing, so its errors should be readable.
  [[nodiscard]] Status finalize();

  // --- Queries ------------------------------------------------------------

  [[nodiscard]] std::size_t num_nodes() const noexcept { return nodes_.size(); }
  [[nodiscard]] std::size_t num_tensors() const noexcept { return tensors_.size(); }
  [[nodiscard]] const Node&       node(NodeId n) const { return nodes_[n]; }
  [[nodiscard]] const TensorDesc& tensor(TensorId t) const { return tensors_[t]; }

  // Kahn's algorithm. Returns a valid execution order, or kInvalidArgument if
  // the graph has a cycle.
  //
  // WHY Kahn and not DFS post-order: Kahn processes nodes in *waves* of
  // in-degree zero, which is exactly the information the scheduler wants —
  // every node in one wave is mutually independent and can go on a different
  // stream. A DFS post-order gives a correct topological order but destroys
  // that grouping, so you would have to recompute it. Choose the algorithm
  // whose intermediate state is the thing you need.
  [[nodiscard]] StatusOr<std::vector<NodeId>> topological_order() const;

  // The same sort, grouped: levels()[d] = all nodes at depth d.
  // This is the parallelism structure of the graph. width = max level size is
  // the upper bound on useful concurrent streams.
  [[nodiscard]] StatusOr<std::vector<std::vector<NodeId>>> levels() const;

  // --- Liveness: [first_use, last_use] node index (in the given topo order)
  //     for every tensor. This is the input to the memory planner. Computing it
  //     is a single backward pass over the order: a tensor dies after the last
  //     node that reads it. Graph outputs never die.
  struct LiveRange { int def = -1; int last_use = -1; };
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
  std::vector<Node>       nodes_;
  std::vector<TensorDesc> tensors_;
  bool                    finalized_ = false;
};

}  // namespace mcke
