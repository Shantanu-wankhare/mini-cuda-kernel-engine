// =============================================================================
//  mcke/graph/op.hpp
//
//  WHAT: The `Op` interface — one unit of GPU work — plus the parameter structs
//        for the four ops this project implements, and the cost model each op
//        must report.
//
//  ---------------------------------------------------------------------------
//  DESIGN DECISION — polymorphic Op vs. a tagged union / enum + switch.
//
//  Options considered:
//   (a) enum OpKind + a big switch in the executor. Fast, no allocation, all
//       dispatch in one place. But adding an op touches the executor, the
//       cost model, and the validator — three places, and the switch grows
//       into the thing everyone is afraid to edit.
//   (b) std::variant<GemmOp, SoftmaxOp, ...>. Type-safe, no heap, but every
//       visitor must handle every alternative, and the variant's size is the
//       max over all ops. Adding an op recompiles everything that visits.
//   (c) Virtual `Op` base with unique_ptr ownership.  <-- chosen
//
//  Chosen (c) because the dispatch happens once per *kernel launch*. A launch
//  costs 3-10 us of driver time; a virtual call costs ~2 ns. Spending 0.00003%
//  of the launch cost to make each op a self-contained file — its own launcher,
//  its own FLOP count, its own shape inference, its own validation — is
//  obviously right. The rule generalises: **pay for polymorphism at the
//  granularity where the payload dwarfs the dispatch.**
//
//  ---------------------------------------------------------------------------
//  WHY EVERY OP MUST REPORT flops() AND bytes().
//  These two numbers are not telemetry decoration; they are the axes of the
//  roofline model:
//        arithmetic intensity  AI = flops / bytes            [FLOP/byte]
//        achieved TFLOP/s      = flops / seconds
//        achieved GB/s         = bytes / seconds
//  With them, "my kernel takes 4.2 ms" becomes "my kernel achieves 63% of peak
//  bandwidth at AI = 0.5, so it is memory-bound and there is 37% left" — which
//  is an engineering statement. Without them you cannot tell a good kernel from
//  a bad one. Requiring them in the interface means we can never *forget* to
//  compute them, and the executor can print a roofline for a whole graph.
//
//  `bytes()` must be the *ideal* traffic (compulsory misses: read each input
//  once, write each output once), not the actual traffic. The ratio between
//  ideal and what Nsight reports as dram__bytes is your cache-efficiency
//  measurement — if you put actual traffic in the model you lose that signal.
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "mcke/core/device.hpp"   // DeviceInfo, for launch-time tile/occupancy choices
#include "mcke/core/status.hpp"
#include "mcke/kernels/kernels.hpp"   // GemmVariant/GemmTile: aliased, not re-declared
#include "mcke/runtime/stream.hpp"
#include "mcke/tensor/tensor.hpp"

namespace mcke {

// Handles into the graph's tensor table. Indices, not pointers: the graph owns
// a vector of tensors and vectors reallocate. Also makes the graph trivially
// serialisable and printable.
using TensorId = std::uint32_t;
using NodeId   = std::uint32_t;
inline constexpr TensorId kInvalidTensor = 0xFFFFFFFFu;
inline constexpr NodeId   kInvalidNode   = 0xFFFFFFFFu;

// Everything an op needs at launch time. Passed by const& so ops cannot mutate
// scheduler state.
struct OpContext {
  rt::StreamHandle  stream = {};        // stream this op was scheduled onto
  const DeviceInfo* device = nullptr;   // for occupancy/tile decisions at launch
  // Scratch for ops that need workspace (e.g. a two-pass reduction's
  // partial-sums buffer). A PLAIN POINTER AND SIZE, not a DeviceAllocator*.
  //
  // An earlier version handed ops an allocator here, which invited
  // allocate() on the per-iteration hot path. Phase 2 measured raw cudaMalloc at
  // up to 720 us with a cudaFree that SYNCHRONISES THE DEVICE -- so a single
  // such call inside run_async() silently converts this whole runtime back to
  // synchronous execution, which is precisely the invariant Op::launch's own
  // contract below declares load-bearing. The planner sizes workspace once from
  // Op::workspace_bytes() and carves one arena per stream; same-stream in-order
  // issue then makes sharing it safe with no analysis at all.
  void*             workspace       = nullptr;
  std::size_t       workspace_bytes = 0;
};

// Cost model, reported per op instance (it depends on the shapes).
struct OpCost {
  std::uint64_t flops = 0;   // useful floating-point operations
  std::uint64_t bytes = 0;   // ideal DRAM traffic: inputs read once + outputs written once
  [[nodiscard]] double arithmetic_intensity() const {
    return bytes ? static_cast<double>(flops) / static_cast<double>(bytes) : 0.0;
  }
};

class Op {
 public:
  virtual ~Op() = default;

  [[nodiscard]] virtual std::string_view type_name() const = 0;

  // Enqueue this op's kernel(s) on ctx.stream. MUST NOT synchronise. Any
  // implementation that calls cudaDeviceSynchronize or cudaStreamSynchronize
  // here silently converts the whole runtime back to synchronous execution —
  // this is the invariant Phase 4's overlap depends on.
  [[nodiscard]] virtual Status launch(const OpContext& ctx,
                                     const std::vector<Tensor>& inputs,
                                     const std::vector<Tensor>& outputs) = 0;

  // Compute output shapes/dtypes from input shapes. Run at graph-build time so
  // the memory planner knows every buffer size *before* any allocation happens.
  [[nodiscard]] virtual StatusOr<std::vector<Shape>> infer_shapes(
      const std::vector<Shape>& input_shapes) const = 0;

  [[nodiscard]] virtual OpCost cost(const std::vector<Shape>& input_shapes) const = 0;

  // Optional: extra device bytes needed as scratch for these shapes.
  [[nodiscard]] virtual std::size_t workspace_bytes(const std::vector<Shape>&) const { return 0; }
};

using OpPtr = std::unique_ptr<Op>;

// =============================================================================
//  Parameter structs + op declarations. Implementations live in
//  src/graph/ops_*.cpp and call into kernels/*.cu launchers.
// =============================================================================

// ---- 1. GEMM: C = alpha * op(A) @ op(B) + beta * C -------------------------
struct GemmParams {
  float alpha = 1.f;
  // beta != 0 is REJECTED in Phase 4, by GemmOp's constructor. Not laziness:
  // a beta != 0 GEMM reads C before writing it, so C is a third INPUT that the
  // graph does not model. Under any buffer-reuse policy C is recycled bytes
  // whose contents depend on the schedule, which (a) violates the SSA
  // single-producer rule the graph is built on and (b) guarantees a bit-identity
  // failure in the numerics gate that would look like a race and would not be
  // one. Modelling it properly means taking C as a declared input producing a
  // new output tensor; that is a later phase.
  float beta  = 0.f;

  // Which kernel variant to launch. Exposed as a parameter (not auto-chosen)
  // because Phase 3 is a *comparison*: the same graph must be runnable with
  // each variant so the speedups are apples-to-apples on one machine.
  //
  // ALIASED, NOT RE-DECLARED. This was its own five-enumerator enum through
  // Phase 3, which then grew kernels::GemmVariant to eight (kWarpTileNoDbuf,
  // kWarpTileVec4, kNaiveUncoalesced) and left three of them unreachable from
  // the graph layer with no compile error. The rule this establishes:
  // AN ENUM THAT EXISTS ONLY TO BE TRANSLATED 1:1 INTO ANOTHER ENUM SHOULD BE
  // THE OTHER ENUM. Aliasing keeps every existing spelling
  // (GemmParams::Variant::kNaive still compiles) and makes drift structurally
  // impossible rather than merely warned about -- see the note on Act below,
  // which warns about exactly this class of bug and was written before it
  // happened here.
  using Variant = kernels::GemmVariant;
  Variant variant = Variant::kTiledRegBlock;

  // launch_gemm_f32 requires a tile, and it CANNOT be defaulted safely: the
  // struct's own defaults are (128,128,8,8,8), while kTiledSmem is only
  // instantiated for (32,32,32,1,1) and hard-rejects anything else. So the
  // natural-looking GemmParams{.variant = kTiledSmem} is an InvalidArgumentError
  // discovered at launch time -- the worst possible place. GemmOp's constructor
  // validates the (variant, tile) pair at graph-build time instead.
  kernels::GemmTile tile{};

  // transpose_a / transpose_b DELETED. launch_gemm_f32 has no transpose
  // parameters, so these were unimplementable -- and worse than the enum drift
  // above, because a silently-ignored transpose flag computes A@B when the
  // caller asked for A^T@B and nothing anywhere reports it.
};

class GemmOp final : public Op {
 public:
  explicit GemmOp(GemmParams p) : p_(p) {}
  [[nodiscard]] std::string_view type_name() const override { return "Gemm"; }
  [[nodiscard]] Status launch(const OpContext&, const std::vector<Tensor>&,
                              const std::vector<Tensor>&) override;
  [[nodiscard]] StatusOr<std::vector<Shape>> infer_shapes(const std::vector<Shape>&) const override;

  // FLOPs for an MxNxK GEMM = 2*M*N*K (one multiply + one add per MAC).
  // Bytes (ideal) = (M*K + K*N + M*N) * sizeof(dtype).
  // Note AI grows with the matrix size: for square N, AI ~ N/6 * ... — which is
  // exactly why GEMM is the canonical compute-bound kernel and why tiling
  // (raising data reuse) is the whole game.
  [[nodiscard]] OpCost cost(const std::vector<Shape>&) const override;

 private:
  GemmParams p_;
};

// ---- 2. Fused BiasAdd + activation ---------------------------------------
// y = act(x + bias), bias broadcast over the last dimension.
//
// WHY FUSE: unfused, this is two kernels, each reading and writing the whole
// tensor: 4 * N * 4 bytes of traffic for ~2 FLOPs/element. AI ~ 0.12 — utterly
// memory-bound. Fused: 2 * N * 4 bytes (+ the tiny bias vector), i.e. a 2x
// reduction in traffic and therefore ~2x speedup on a bandwidth-bound op. This
// is the cheapest large win in the whole project and the reason kernel fusion
// exists as a discipline.
struct BiasActParams {
  // Spelled kGeluErf, matching kernels::Activation exactly. These two enums
  // describe the same four activations and BiasActOp::launch will translate
  // between them in Phase 4; letting the same activation carry two different
  // names across that boundary is how a translation quietly maps the wrong case.
  enum class Act { kNone, kRelu, kGeluErf, kGeluTanh } act = Act::kGeluErf;
  // GELU has an exact form (0.5x(1+erf(x/sqrt2))) and a tanh approximation.
  // erff() on device is ~20 instructions; the tanh form is ~10 but differs in
  // the 3rd decimal. We implement both and measure, because "which GELU" is a
  // question every real kernel author has to answer.
};

class BiasActOp final : public Op {
 public:
  explicit BiasActOp(BiasActParams p) : p_(p) {}
  [[nodiscard]] std::string_view type_name() const override { return "BiasAct"; }
  [[nodiscard]] Status launch(const OpContext&, const std::vector<Tensor>&,
                              const std::vector<Tensor>&) override;
  [[nodiscard]] StatusOr<std::vector<Shape>> infer_shapes(const std::vector<Shape>&) const override;
  [[nodiscard]] OpCost cost(const std::vector<Shape>&) const override;

 private:
  BiasActParams p_;
};

// ---- 3. Row-wise reduction ------------------------------------------------
struct ReduceParams {
  enum class Kind { kSum, kMax, kMean } kind = Kind::kSum;
  int  axis = -1;            // -1 = last axis
  enum class Variant { kSmemTree, kWarpShuffle, kTwoPass } variant = Variant::kWarpShuffle;
};

class ReduceOp final : public Op {
 public:
  explicit ReduceOp(ReduceParams p) : p_(p) {}
  [[nodiscard]] std::string_view type_name() const override { return "Reduce"; }
  [[nodiscard]] Status launch(const OpContext&, const std::vector<Tensor>&,
                              const std::vector<Tensor>&) override;
  [[nodiscard]] StatusOr<std::vector<Shape>> infer_shapes(const std::vector<Shape>&) const override;
  // A reduction does N-1 adds for N elements read: AI ~ 0.25 FLOP/byte for f32.
  // Hopelessly memory-bound — so the *only* meaningful metric for a reduction
  // kernel is achieved bandwidth as a fraction of peak. Reporting TFLOP/s for a
  // reduction is a red flag in an interview.
  [[nodiscard]] OpCost cost(const std::vector<Shape>&) const override;

 private:
  ReduceParams p_;
};

// ---- 4. Row-wise softmax -------------------------------------------------
struct SoftmaxParams {
  int axis = -1;
  // The numerically-stable formulation subtracts the row max before exp.
  // Without it, exp(x) overflows f32 for x > 88 and you get NaNs — the classic
  // softmax bug. Cost: one extra pass (or a fused max+sum pass).
  bool numerically_stable = true;
  // Online softmax (Milakov & Gimelshein) computes max and sum in ONE pass by
  // rescaling the running sum whenever a new max appears. This is the same
  // trick that makes FlashAttention possible. Implementing both and measuring
  // the pass-count reduction is a Phase 3 deliverable.
  enum class Variant { kThreePass, kOnlineOnePass } variant = Variant::kOnlineOnePass;
};

class SoftmaxOp final : public Op {
 public:
  explicit SoftmaxOp(SoftmaxParams p) : p_(p) {}
  [[nodiscard]] std::string_view type_name() const override { return "Softmax"; }
  [[nodiscard]] Status launch(const OpContext&, const std::vector<Tensor>&,
                              const std::vector<Tensor>&) override;
  [[nodiscard]] StatusOr<std::vector<Shape>> infer_shapes(const std::vector<Shape>&) const override;
  [[nodiscard]] OpCost cost(const std::vector<Shape>&) const override;

 private:
  SoftmaxParams p_;
};

}  // namespace mcke
