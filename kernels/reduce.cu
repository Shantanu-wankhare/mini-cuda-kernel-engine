// =============================================================================
//  kernels/reduce.cu
//
//  WHAT: Phase 3b. Row-wise reduction of a [rows, cols] f32 matrix into [rows],
//        in three variants that isolate one idea each.
//
//  WHY .cu: __global__ functions and <<<>>> launch sites.
//
//  ---------------------------------------------------------------------------
//  THIS KERNEL IS MEMORY-BOUND AND THAT DETERMINES WHAT TO REPORT
//
//  A reduction does n-1 combines for n elements read: AI = 0.25 FLOP/byte for
//  f32, against this GPU's ridge point of 34.2. That is 137x inside memory-bound
//  territory, so the ONLY meaningful metric is achieved bandwidth as a fraction
//  of peak. Reporting TFLOP/s for a reduction is a red flag.
//
//  ---------------------------------------------------------------------------
//  MAPPING: ONE BLOCK PER ROW, gridDim.x == rows EXACTLY
//
//  At 8192 x 4096 the alternative mappings barely differ on the things people
//  usually argue about:
//    * Parallelism is not the constraint. 8192 blocks over 40 SMs at 4 resident
//      blocks each is 51.2 waves; the tail is 0.2 of one wave out of 51, so load
//      imbalance is under 0.4%.
//    * Coalescing is perfect either way -- block-per-row has thread t read
//      x[row*cols + t + k*256], a warp covering 128 contiguous bytes; warp-per-row
//      is also 128 contiguous bytes. Both are the ideal 4 sectors per request.
//
//  Warp-per-row would actually be architecturally cleaner for kWarpShuffle: the
//  whole combine becomes 5 shuffles with NO __syncthreads and no shared memory at
//  all. It is used anyway because kSmemTree *needs* a block to have a tree to
//  build -- and if the two variants used different row-to-thread mappings, their
//  results rows would differ in TWO variables (combine strategy AND mapping) and
//  the attribution the whole table exists to provide would be gone.
//
//  gridDim.x == rows exactly, with no grid-stride over rows, for a related
//  reason: if one block handled several rows it would have to re-use its shared
//  staging array, which needs an EXTRA __syncthreads at the top of each iteration
//  to stop a fast thread clobbering smem before a slow one has read it. That
//  would push kWarpShuffle's barrier count from 1 to 2 -- and the barrier count
//  is a column in RESULTS.md. Launching exactly `rows` blocks keeps the measured
//  quantity equal to the analysed quantity.
//
//  The regime where this mapping choice DOES matter is few rows: 64 rows is 0.4
//  waves with 24 of 40 SMs idle, and that is exactly what kTwoPass is for.
// =============================================================================
#include "mcke/kernels/kernels.hpp"
#include "mcke/runtime/cuda_check.hpp"

#include "reduce_ops.cuh"

namespace mcke::kernels {
namespace {

constexpr int kThreads = 256;

// The number of resident blocks a T4 wants in order to be full: 40 SMs x 4
// blocks (1024 threads/SM / 256 threads per block). kTwoPass splits rows only
// until it reaches this.
constexpr std::int64_t kTargetBlocks = 160;
constexpr std::int64_t kMaxSplit     = 64;

// Deterministic, so a reader can reconstruct it from the shape alone and the
// variant string can name it (two_pass_256t_s8).
__host__ __device__ inline std::int64_t two_pass_split(std::int64_t rows) {
  if (rows <= 0) return 1;
  std::int64_t s = (kTargetBlocks + rows - 1) / rows;   // ceil(160 / rows)
  if (s < 1) s = 1;
  if (s > kMaxSplit) s = kMaxSplit;
  return s;
}

// -----------------------------------------------------------------------------
// Variant 1: the classic shared-memory tree.
//
// BARRIER COUNT: 1 + log2(blockDim) = 9 at 256 threads. Nine, not eight -- the
// load-into-smem barrier before the tree begins is a real barrier. (An earlier
// version of kernels.hpp and the RESULTS.md column both said 8.)
//
// BANK CONFLICTS: this version has NONE, and that is worth stating precisely
// because the standard teaching example is about the version that does.
// T4 shared memory is 32 banks x 4 B, bank = (byte_addr / 4) mod 32.
//   * SEQUENTIAL addressing (used here): for s >= 32, a warp's 32 consecutive
//     threads touch smem[t] (banks 0..31, one each) and smem[t+s] (a rotation of
//     0..31, one each). Conflict-free. For s < 32 only one warp is active and it
//     touches <= 32 distinct consecutive addresses. Conflict-free.
//   * INTERLEAVED addressing -- `if (tid % (2*s) == 0) smem[2*s*tid] = ...` --
//     has active threads hitting banks (2s*t) mod 32, a min(2s,32)-way conflict:
//     2-way at s=1 degrading to 32-way (fully serialised) at s >= 16.
//
// So "the standard fix" is switching from interleaved to sequential addressing,
// and it fixes TWO things at once, which is why it is the canonical example:
// it removes those bank conflicts AND it removes warp divergence, because
// `tid < s` makes whole warps inactive where `tid % (2*s) == 0` diverges inside
// every warp. Do not manufacture a conflict that is not there -- write the
// correct version and record what was avoided.
// -----------------------------------------------------------------------------
template <class Op>
__global__ void row_reduce_tree_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       std::int64_t cols, float scale) {
  __shared__ float smem[kThreads];
  const std::int64_t row_base = static_cast<std::int64_t>(blockIdx.x) * cols;

  // Phase 1: serial accumulate into a register. No barriers, no smem, and the
  // strided access keeps every warp's 32 loads contiguous.
  float acc = Op::identity();
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    acc = Op::combine(acc, x[row_base + c]);

  smem[threadIdx.x] = acc;
  __syncthreads();                                   // barrier 1 of 9

  // Phase 2: sequential-addressing tree (Harris reduction #3).
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(s))
      smem[threadIdx.x] = Op::combine(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();                                 // barriers 2..9
  }
  if (threadIdx.x == 0) out[blockIdx.x] = smem[0] * scale;
}

// -----------------------------------------------------------------------------
// Variant 2: warp shuffle. Exactly ONE __syncthreads.
//
// Three reasons this should win, and one of them is worth less than it sounds:
//   1. 9 barriers -> 1. A __syncthreads is not free even when threads arrive
//      together: it drains the pipeline and forces every warp in the block to a
//      common point, so the block runs at the speed of its slowest warp eight
//      extra times. On a memory-bound kernel where some warps hit L2 and some go
//      to DRAM, those rendezvous serialise variance that would otherwise average
//      out.
//   2. ~500x less shared-memory traffic. The tree writes 256 floats and makes 8
//      read-modify-write passes over them (~4096 smem accesses per block); this
//      does the intra-warp work entirely in the register file and touches smem 8
//      times total, once per warp.
//   3. Less smem -> higher occupancy: 1024 B/block vs 32 B/block. HONESTLY, at
//      this configuration that is worth NOTHING: 64 KiB/SM / 1024 B = 64 blocks
//      by smem, but the 1024-threads-per-SM cap gives 4 blocks either way, so the
//      binding limit is identical. Reason 3 is real in principle and zero in
//      practice here; it starts mattering with bigger blocks, or when a reduction
//      is fused into a kernel that already uses smem -- which is exactly the
//      softmax and GEMM case.
// -----------------------------------------------------------------------------
template <class Op>
__global__ void row_reduce_shfl_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       std::int64_t cols, float scale) {
  __shared__ float smem[kThreads / 32];              // 8 floats = 32 bytes
  const std::int64_t row_base = static_cast<std::int64_t>(blockIdx.x) * cols;

  float acc = Op::identity();
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    acc = Op::combine(acc, x[row_base + c]);

  acc = block_reduce<Op>(acc, smem);
  if (threadIdx.x == 0) out[blockIdx.x] = acc * scale;
}

// -----------------------------------------------------------------------------
// Variant 3: two-pass, for the too-few-rows regime.
//
// Pass 1: grid is (rows, split); block (r, s) reduces its slice of row r into
//         partial[r*split + s].
// Pass 2: grid is (rows); block r reduces the `split` partials of row r.
//
// The extra traffic is 2 * rows * split * 4 bytes -- 512 KiB on top of 128 MiB
// at the benchmark shape, +0.4%. The extra launch is ~5 us on a ~570 us kernel.
// Both are pure overhead when the machine is already full, which is why this is
// predicted 1-3% SLOWER at 8192 rows and 3-10x FASTER at 64 rows.
// -----------------------------------------------------------------------------
template <class Op>
__global__ void row_reduce_pass1_kernel(const float* __restrict__ x,
                                        float* __restrict__ partial,
                                        std::int64_t cols, std::int64_t split) {
  __shared__ float smem[kThreads / 32];
  const std::int64_t row   = blockIdx.x;
  const std::int64_t slice = blockIdx.y;
  const std::int64_t base  = row * cols;

  // Each of `split` blocks strides through the row with a stride of
  // split*blockDim, so the union of all blocks covers the row exactly once and
  // every individual warp access stays contiguous.
  float acc = Op::identity();
  for (std::int64_t c = slice * blockDim.x + threadIdx.x; c < cols;
       c += split * blockDim.x)
    acc = Op::combine(acc, x[base + c]);

  acc = block_reduce<Op>(acc, smem);
  if (threadIdx.x == 0) partial[row * split + slice] = acc;
}

template <class Op>
__global__ void row_reduce_pass2_kernel(const float* __restrict__ partial,
                                        float* __restrict__ out,
                                        std::int64_t split, float scale) {
  __shared__ float smem[kThreads / 32];
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * split;

  float acc = Op::identity();
  for (std::int64_t s = threadIdx.x; s < split; s += blockDim.x)
    acc = Op::combine(acc, partial[base + s]);

  acc = block_reduce<Op>(acc, smem);
  if (threadIdx.x == 0) out[blockIdx.x] = acc * scale;
}

}  // namespace

// -----------------------------------------------------------------------------
// Workspace query
// -----------------------------------------------------------------------------
std::size_t row_reduce_workspace_bytes(std::int64_t rows, std::int64_t cols,
                                       ReduceVariant variant) noexcept {
  (void)cols;
  if (variant != ReduceVariant::kTwoPass || rows <= 0) return 0;
  return static_cast<std::size_t>(rows * two_pass_split(rows)) * sizeof(float);
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------

Status launch_row_reduce_f32(const float* x, float* out,
                             std::int64_t rows, std::int64_t cols,
                             ReduceKind kind, ReduceVariant variant,
                             float* workspace, std::size_t workspace_bytes,
                             rt::StreamHandle stream) {
  if (rows == 0 || cols == 0) return OkStatus();      // an empty grid is a launch error
  if (rows < 0 || cols < 0)
    return InvalidArgumentError("launch_row_reduce_f32: negative extent");
  if (!x || !out) return InvalidArgumentError("launch_row_reduce_f32: null pointer");
  // gridDim.x is the row index and caps at 2^31-1. The int64_t in the signature
  // invites a value the launch geometry cannot express, so reject it explicitly.
  if (rows > 2147483647LL)
    return InvalidArgumentError("launch_row_reduce_f32: rows exceeds gridDim.x limit");

  // kMean is kSum with one multiply applied once per ROW, by the single thread
  // that stores the result -- not once per element. So three kinds cost two
  // kernel instantiations.
  const float scale = (kind == ReduceKind::kMean)
                          ? 1.0f / static_cast<float>(cols)
                          : 1.0f;
  const bool is_max = (kind == ReduceKind::kMax);

  if (variant == ReduceVariant::kTwoPass) {
    const std::size_t need = row_reduce_workspace_bytes(rows, cols, variant);
    if (workspace == nullptr || workspace_bytes < need)
      return InvalidArgumentError(
          "launch_row_reduce_f32: kTwoPass needs " + std::to_string(need) +
          " bytes of workspace (got " + std::to_string(workspace_bytes) +
          "); query row_reduce_workspace_bytes() first");
    const std::int64_t split = two_pass_split(rows);
    const dim3 g1(static_cast<unsigned>(rows), static_cast<unsigned>(split), 1u);
    if (is_max) {
      row_reduce_pass1_kernel<MaxOp><<<g1, kThreads, 0, stream>>>(x, workspace, cols, split);
      MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
      row_reduce_pass2_kernel<MaxOp><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
          workspace, out, split, scale);
    } else {
      row_reduce_pass1_kernel<SumOp><<<g1, kThreads, 0, stream>>>(x, workspace, cols, split);
      MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
      row_reduce_pass2_kernel<SumOp><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
          workspace, out, split, scale);
    }
    MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return OkStatus();
  }

  const unsigned grid = static_cast<unsigned>(rows);
  if (variant == ReduceVariant::kSmemTree) {
    if (is_max) row_reduce_tree_kernel<MaxOp><<<grid, kThreads, 0, stream>>>(x, out, cols, scale);
    else        row_reduce_tree_kernel<SumOp><<<grid, kThreads, 0, stream>>>(x, out, cols, scale);
  } else {
    if (is_max) row_reduce_shfl_kernel<MaxOp><<<grid, kThreads, 0, stream>>>(x, out, cols, scale);
    else        row_reduce_shfl_kernel<SumOp><<<grid, kThreads, 0, stream>>>(x, out, cols, scale);
  }
  MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

// The frozen 7-argument form, kept verbatim so nothing that already calls it
// breaks. It delegates, and refuses kTwoPass rather than silently allocating --
// see the rationale on row_reduce_workspace_bytes in kernels.hpp.
Status launch_row_reduce_f32(const float* x, float* out,
                             std::int64_t rows, std::int64_t cols,
                             ReduceKind kind, ReduceVariant variant,
                             rt::StreamHandle stream) {
  if (variant == ReduceVariant::kTwoPass)
    return UnimplementedError(
        "launch_row_reduce_f32: kTwoPass requires a workspace; call "
        "row_reduce_workspace_bytes() and use the 9-argument overload");
  return launch_row_reduce_f32(x, out, rows, cols, kind, variant,
                               nullptr, 0, stream);
}

}  // namespace mcke::kernels
