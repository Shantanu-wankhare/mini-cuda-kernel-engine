// =============================================================================
//  src/graph/graph.cpp
//
//  WHAT: Graph construction and analysis -- finalize(), Kahn's sort, levels,
//        ASAP/ALAP depths, dead-node detection, liveness, to_dot(). Phase 4.
//
//  WHY .cpp AND NOT HEADER-INLINE: all of it is host code with real error prose,
//  and none of it is on a per-iteration hot path (it runs once, at plan time).
//  Keeping it out of the header also keeps <queue>/<sstream> out of every TU
//  that merely wants to describe a graph.
//
//  Zero CUDA. Deliberately: every function here is unit-testable on a laptop,
//  which is the whole reason docs/ROADMAP.md puts Phase 4's graph logic on [Mac].
// =============================================================================
#include "mcke/graph/graph.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <sstream>

namespace mcke {
namespace {

// Shape validation, shared by add_input and add_node's inferred outputs.
//
// Deliberately strict about contiguity: every Phase 0-3 kernel indexes with
// row-major arithmetic and none of them consult Shape::stride(). A
// non-contiguous tensor would be read as though it were contiguous -- wrong
// numbers, no error. Better to refuse it at build time than to compute garbage.
Status validate_shape(const Shape& s, const std::string& what) {
  if (s.rank() < 1)
    return InvalidArgumentError(what + ": rank is 0; a scalar must be rank 1 with dim 1");
  dim_t acc = 1;
  for (int i = 0; i < s.rank(); ++i) {
    const dim_t d = s.dim(i);
    if (d <= 0)
      return InvalidArgumentError(what + ": dim " + std::to_string(i) + " is " +
                                  std::to_string(d) + "; every extent must be > 0");
    // Overflow guard before the multiply, not after -- signed overflow is UB, so
    // "multiply and check for negative" is not a check at all.
    if (d > std::numeric_limits<dim_t>::max() / acc)
      return InvalidArgumentError(what + ": numel overflows int64 at dim " +
                                  std::to_string(i));
    acc *= d;
  }
  if (!s.is_contiguous())
    return InvalidArgumentError(what + ": strides are not row-major contiguous; "
                                "Phase 0-4 kernels assume contiguity and do not "
                                "consult stride()");
  return OkStatus();
}

}  // namespace

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
StatusOr<TensorId> Graph::add_input(Shape shape, DType dtype, std::string name) {
  MCKE_RETURN_IF_ERROR(validate_shape(shape, "add_input('" + name + "')"));
  if (finalized_)
    return FailedPreconditionError("add_input: graph is already finalized");
  TensorDesc d;
  d.shape          = shape;
  d.dtype          = dtype;
  d.producer       = kInvalidNode;
  d.name           = std::move(name);
  d.is_graph_input = true;
  tensors_.push_back(std::move(d));
  return static_cast<TensorId>(tensors_.size() - 1);
}

StatusOr<std::vector<TensorId>> Graph::add_node(OpPtr op, std::vector<TensorId> inputs,
                                                std::string name) {
  if (finalized_)
    return FailedPreconditionError("add_node: graph is already finalized");
  if (!op) return InvalidArgumentError("add_node('" + name + "'): null op");

  const std::string where = "add_node('" + name + "', " + std::string(op->type_name()) + ")";

  // Inputs must already exist. This is also WHY cycles are unrepresentable: a
  // node can only name tensors that exist, and its own outputs are created
  // afterwards, so every edge points strictly backward in creation order.
  std::vector<Shape> in_shapes;
  in_shapes.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] >= tensors_.size())
      return InvalidArgumentError(where + ": input slot " + std::to_string(i) +
                                  " references tensor id " + std::to_string(inputs[i]) +
                                  ", but the graph has only " +
                                  std::to_string(tensors_.size()) + " tensors");
    in_shapes.push_back(tensors_[inputs[i]].shape);
  }

  // The op is the single source of truth for output shapes, so a shape can never
  // disagree with what the kernel expects. Errors surface HERE, at build time,
  // naming the node -- not at launch time on a metered GPU session.
  MCKE_ASSIGN_OR_RETURN(std::vector<Shape> out_shapes, op->infer_shapes(in_shapes));
  if (out_shapes.empty())
    return InvalidArgumentError(where + ": infer_shapes returned no outputs");

  const NodeId nid = static_cast<NodeId>(nodes_.size());
  std::vector<TensorId> out_ids;
  out_ids.reserve(out_shapes.size());

  for (std::size_t k = 0; k < out_shapes.size(); ++k) {
    MCKE_RETURN_IF_ERROR(validate_shape(
        out_shapes[k], where + " output " + std::to_string(k)));
    TensorDesc d;
    d.shape = out_shapes[k];
    // DTYPE RULE, written down because Op::infer_shapes has no dtype channel and
    // therefore cannot express one: outputs inherit input 0's dtype, or f32 for a
    // source node. True for every op in Phases 0-4 (all f32). The header's claim
    // that "the op is the single source of truth" holds for SHAPES only; when a
    // mixed-precision op arrives, infer_shapes needs a sibling infer_dtypes and
    // this line becomes wrong rather than merely narrow.
    d.dtype    = inputs.empty() ? DType::kF32 : tensors_[inputs[0]].dtype;
    d.producer = nid;   // set exactly once, ever: this is what makes
                        // multiple-producers unrepresentable
    d.name     = name + (out_shapes.size() > 1 ? ":" + std::to_string(k) : "");
    tensors_.push_back(std::move(d));
    out_ids.push_back(static_cast<TensorId>(tensors_.size() - 1));
  }

  Node n;
  n.op      = std::move(op);
  n.inputs  = std::move(inputs);
  n.outputs = out_ids;
  n.name    = std::move(name);
  nodes_.push_back(std::move(n));
  return out_ids;
}

Status Graph::mark_output(TensorId t) {
  if (finalized_) return FailedPreconditionError("mark_output: graph is already finalized");
  if (t >= tensors_.size())
    return InvalidArgumentError("mark_output: tensor id " + std::to_string(t) +
                                " is out of range (" + std::to_string(tensors_.size()) +
                                " tensors)");
  tensors_[t].is_graph_output = true;
  return OkStatus();
}

// -----------------------------------------------------------------------------
// finalize()
// -----------------------------------------------------------------------------
Status Graph::finalize() {
  if (finalized_) return OkStatus();   // idempotent: plan() may call this for you

  if (nodes_.empty())
    return FailedPreconditionError("finalize: graph has no nodes");

  bool any_output = false;
  for (const TensorDesc& t : tensors_) any_output = any_output || t.is_graph_output;
  if (!any_output)
    return InvalidArgumentError("finalize: graph has no outputs; call mark_output()");

  // --- Every input must be defined: produced by a node, or declared an input.
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    for (std::size_t i = 0; i < nodes_[n].inputs.size(); ++i) {
      const TensorId t = nodes_[n].inputs[i];
      const TensorDesc& d = tensors_[t];
      if (d.producer == kInvalidNode && !d.is_graph_input)
        return InvalidArgumentError(
            "finalize: " + describe(static_cast<NodeId>(n)) + " input slot " +
            std::to_string(i) + " is " + describe_tensor(t) +
            ", which has no producer and is not a declared graph input");
    }

  // --- An output must be reachable: produced, or itself an input (identity graph).
  for (std::size_t t = 0; t < tensors_.size(); ++t) {
    const TensorDesc& d = tensors_[t];
    if (d.is_graph_output && d.producer == kInvalidNode && !d.is_graph_input)
      return InvalidArgumentError("finalize: " + describe_tensor(static_cast<TensorId>(t)) +
                                  " is marked as a graph output but has no producer");
  }

  // --- Derive consumers, then preds/succs from them. Edges are DERIVED from
  //     tensor def/use, never declared -- see the file banner in graph.hpp. A
  //     declared edge list can disagree with the dataflow; a derived one cannot.
  for (TensorDesc& d : tensors_) d.consumers.clear();
  for (std::size_t n = 0; n < nodes_.size(); ++n) {
    nodes_[n].preds.clear();
    nodes_[n].succs.clear();
  }
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    for (TensorId t : nodes_[n].inputs)
      tensors_[t].consumers.push_back(static_cast<NodeId>(n));

  for (std::size_t n = 0; n < nodes_.size(); ++n) {
    for (TensorId t : nodes_[n].inputs) {
      const NodeId p = tensors_[t].producer;
      if (p != kInvalidNode) nodes_[n].preds.push_back(p);
    }
    // A node reading two outputs of the same predecessor would otherwise appear
    // twice, which would corrupt Kahn's in-degree counting.
    std::sort(nodes_[n].preds.begin(), nodes_[n].preds.end());
    nodes_[n].preds.erase(std::unique(nodes_[n].preds.begin(), nodes_[n].preds.end()),
                          nodes_[n].preds.end());
  }
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    for (NodeId p : nodes_[n].preds) nodes_[p].succs.push_back(static_cast<NodeId>(n));

  // --- Acyclicity + ASAP depth, level-synchronously (see below).
  MCKE_ASSIGN_OR_RETURN(std::vector<NodeId> order, kahn_order());
  if (order.size() != nodes_.size()) {
    // UNREACHABLE with the current API, and classified kInternal to say so:
    // add_node() can only reference existing tensors and creates its outputs
    // afterwards, so no cycle can be constructed. The check guards a future
    // alias/mutation API rather than any present possibility.
    std::ostringstream os;
    os << "finalize: graph has a cycle; nodes with unresolved in-degree: [";
    std::vector<bool> placed(nodes_.size(), false);
    for (NodeId n : order) placed[n] = true;
    bool first = true;
    for (std::size_t n = 0; n < nodes_.size(); ++n)
      if (!placed[n]) { os << (first ? "" : ", ") << describe(static_cast<NodeId>(n)); first = false; }
    os << "]";
    return InternalError(os.str());
  }

  // ASAP: depth(n) = 1 + max over preds. Identical recurrence to the Kahn wave
  // index, which is why the two coincide -- given a FIFO frontier.
  for (Node& n : nodes_) n.depth = 0;
  for (NodeId n : order)
    for (NodeId p : nodes_[n].preds)
      nodes_[n].depth = std::max(nodes_[n].depth, nodes_[p].depth + 1);

  int critical = 0;
  for (const Node& n : nodes_) critical = std::max(critical, n.depth);

  // ALAP: height(n) = 1 + max over succs, walked in reverse topological order.
  // This is the one place in this file a BACKWARD pass is genuinely required --
  // unlike liveness, where the header used to claim one was needed.
  std::vector<int> height(nodes_.size(), 0);
  for (auto it = order.rbegin(); it != order.rend(); ++it)
    for (NodeId s : nodes_[*it].succs)
      height[*it] = std::max(height[*it], height[s] + 1);

  for (std::size_t n = 0; n < nodes_.size(); ++n) {
    nodes_[n].alap_depth = critical - height[n];
    nodes_[n].slack      = nodes_[n].alap_depth - nodes_[n].depth;
  }

  // --- Dead nodes: backward reachability from the graph outputs. Not an error,
  //     but executing them inflates every benchmark number with work nobody
  //     asked for, so the executor drops them by default.
  std::vector<bool> alive(nodes_.size(), false);
  {
    std::vector<NodeId> stack;
    for (std::size_t t = 0; t < tensors_.size(); ++t)
      if (tensors_[t].is_graph_output && tensors_[t].producer != kInvalidNode) {
        const NodeId p = tensors_[t].producer;
        if (!alive[p]) { alive[p] = true; stack.push_back(p); }
      }
    while (!stack.empty()) {
      const NodeId n = stack.back();
      stack.pop_back();
      for (NodeId p : nodes_[n].preds)
        if (!alive[p]) { alive[p] = true; stack.push_back(p); }
    }
  }
  for (std::size_t n = 0; n < nodes_.size(); ++n) nodes_[n].is_dead = !alive[n];

  finalized_ = true;
  return OkStatus();
}

// -----------------------------------------------------------------------------
// Sorting
// -----------------------------------------------------------------------------
StatusOr<std::vector<NodeId>> Graph::kahn_order() const {
  // FIFO, and that is part of the specification rather than an implementation
  // detail -- see the comment on topological_order() in graph.hpp. Sources are
  // seeded in NodeId order and successors are visited in NodeId order, so the
  // result is DETERMINISTIC. That matters more than it looks: kLevelParallel
  // round-robins the nodes of a level over streams, so a nondeterministic order
  // within a level would make the published event count nondeterministic too.
  std::vector<int> indeg(nodes_.size(), 0);
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    indeg[n] = static_cast<int>(nodes_[n].preds.size());

  std::queue<NodeId> q;
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    if (indeg[n] == 0) q.push(static_cast<NodeId>(n));

  std::vector<NodeId> order;
  order.reserve(nodes_.size());
  while (!q.empty()) {
    const NodeId n = q.front();
    q.pop();
    order.push_back(n);
    for (NodeId s : nodes_[n].succs)
      if (--indeg[s] == 0) q.push(s);
  }
  return order;   // shorter than nodes_ => cycle; finalize() reports it
}

StatusOr<std::vector<NodeId>> Graph::topological_order() const {
  if (!finalized_)
    return FailedPreconditionError("topological_order: call finalize() first");
  return kahn_order();
}

StatusOr<std::vector<std::vector<NodeId>>> Graph::levels() const {
  if (!finalized_) return FailedPreconditionError("levels: call finalize() first");
  // Bucketed straight off Node::depth rather than read back off the Kahn
  // frontier. That makes the grouping STRUCTURAL: it stays correct even if
  // someone later swaps the queue for a stack (which would still yield a valid
  // topological order but would destroy depth-monotonicity of the pop order).
  int critical = 0;
  for (const Node& n : nodes_) critical = std::max(critical, n.depth);
  std::vector<std::vector<NodeId>> out(static_cast<std::size_t>(critical) + 1);
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    out[static_cast<std::size_t>(nodes_[n].depth)].push_back(static_cast<NodeId>(n));
  return out;
}

StatusOr<std::size_t> Graph::max_level_width() const {
  MCKE_ASSIGN_OR_RETURN(std::vector<std::vector<NodeId>> lv, levels());
  std::size_t w = 0;
  for (const auto& l : lv) w = std::max(w, l.size());
  return w;
}

std::vector<NodeId> Graph::dead_nodes() const {
  std::vector<NodeId> out;
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    if (nodes_[n].is_dead) out.push_back(static_cast<NodeId>(n));
  return out;
}

// -----------------------------------------------------------------------------
// Liveness
// -----------------------------------------------------------------------------
std::vector<Graph::LiveRange> Graph::compute_live_ranges(
    const std::vector<NodeId>& order) const {
  std::vector<LiveRange> lr(tensors_.size());

  // Position of each node within `order`, so uses can be recorded by position
  // in one forward sweep.
  std::vector<int> pos(nodes_.size(), -1);
  for (std::size_t i = 0; i < order.size(); ++i)
    if (order[i] < pos.size()) pos[order[i]] = static_cast<int>(i);

  // ONE FORWARD PASS. Going forward, the final write to last_use_pos IS the
  // maximum, so no backward pass and no fixpoint is needed -- this is a
  // linearised SSA DAG, not a general control-flow graph.
  for (std::size_t i = 0; i < order.size(); ++i) {
    const Node& n = nodes_[order[i]];
    const int p = static_cast<int>(i);
    for (TensorId t : n.outputs) lr[t].def_pos = p;
    for (TensorId t : n.inputs)  lr[t].last_use_pos = p;
  }

  for (std::size_t t = 0; t < tensors_.size(); ++t) {
    const TensorDesc& d = tensors_[t];
    if (d.is_graph_input) {
      // Graph inputs never die: filled by set_input()'s async H2D whose
      // completion the planner does not track, and re-read every iteration if
      // they are weights. Marked live for the whole schedule so no planner can
      // hand their bytes away.
      lr[t].def_pos      = -1;
      lr[t].last_use_pos = static_cast<int>(order.size());
    } else if (d.is_graph_output) {
      // Outputs must outlive execution.
      lr[t].last_use_pos = static_cast<int>(order.size());
    } else {
      // A tensor produced but never consumed would leave last_use_pos == -1
      // while def_pos == i. That interval LOOKS EMPTY, so a planner would hand
      // those bytes to another tensor while the producing kernel is still
      // writing them. Clamping to def_pos closes it; dead_nodes() usually
      // removes the case first.
      lr[t].last_use_pos = std::max(lr[t].last_use_pos, lr[t].def_pos);
    }
  }
  return lr;
}

OpCost Graph::total_cost() const {
  OpCost total;
  for (const Node& n : nodes_) {
    if (n.is_dead || !n.op) continue;
    std::vector<Shape> in;
    in.reserve(n.inputs.size());
    for (TensorId t : n.inputs) in.push_back(tensors_[t].shape);
    const OpCost c = n.op->cost(in);
    total.flops += c.flops;
    total.bytes += c.bytes;
  }
  return total;
}

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------
std::string Graph::describe(NodeId n) const {
  std::ostringstream os;
  os << "node " << n;
  if (n < nodes_.size()) {
    os << " '" << nodes_[n].name << "'";
    if (nodes_[n].op) os << " (" << nodes_[n].op->type_name() << ")";
  }
  return os.str();
}

std::string Graph::describe_tensor(TensorId t) const {
  std::ostringstream os;
  os << "tensor " << t;
  if (t < tensors_.size())
    os << " '" << tensors_[t].name << "' " << tensors_[t].shape.to_string();
  return os.str();
}

std::string Graph::to_dot() const {
  // Cheap to write, and being able to SEE the DAG is how a scheduling bug takes
  // ten seconds to spot instead of an hour. Slack is rendered because a
  // zero-slack node is on the critical path, which is the first thing to look at
  // when a parallel policy fails to beat the sequential one.
  std::ostringstream os;
  os << "digraph mcke_graph {\n  rankdir=TB;\n  node [shape=record, fontname=\"monospace\"];\n";
  for (std::size_t n = 0; n < nodes_.size(); ++n) {
    const Node& nd = nodes_[n];
    os << "  n" << n << " [label=\"{" << n << ": " << nd.name;
    if (nd.op) os << "\\n" << nd.op->type_name();
    os << "|depth " << nd.depth << "\\nslack " << nd.slack << "}\"";
    if (nd.is_dead)      os << ", style=filled, fillcolor=\"#dddddd\"";
    else if (nd.slack == 0) os << ", color=\"#cc0000\", penwidth=2";   // critical path
    os << "];\n";
  }
  for (std::size_t n = 0; n < nodes_.size(); ++n)
    for (TensorId t : nodes_[n].inputs) {
      const NodeId p = tensors_[t].producer;
      if (p == kInvalidNode) continue;
      os << "  n" << p << " -> n" << n << " [label=\"" << tensors_[t].name << "\"];\n";
    }
  os << "}\n";
  return os.str();
}

}  // namespace mcke
