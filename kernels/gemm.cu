// =============================================================================
//  kernels/gemm.cu
//
//  WHAT: Phase 3d. C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N], row-major
//        f32, as a LADDER of variants that isolate one idea per step.
//
//  WHY .cu: __global__ functions and <<<>>> launch sites. Everything in this
//  file that is NOT device code -- the tile dispatch, the occupancy arithmetic --
//  was deliberately hoisted into mcke/kernels/gemm_tile.hpp so it can be
//  unit-tested on a laptop with no GPU.
//
//  THE EIGHT ROWS, in presentation order (bench/gemm_bench.cpp owns the order;
//  GemmVariant is append-only so its numeric order deliberately differs):
//
//    1 naive_uncoalesced   thread->output transpose            }  Stage 5:
//    2 naive               coalesced B reads and C writes      }  the floor
//    3 tiled_smem          shared-memory staging               }  and the
//    8 cublas              the ceiling                         }  ceiling
//
//    4 tiled_regblock      register blocking, float acc[8][8]  }  Stage 6:
//    5 warptile_nodbuf     lane->output permutation only       }  one template,
//    6 warptile_dbuf       double-buffered shared memory       }  four
//    7 warptile_vec4       float4 global->shared loads         }  instantiations
//
//  Rows 4-7 are ONE kernel template differing by ONE argument each -- see the
//  banner above gemm_regblock_kernel. Rows 1-3 are separate kernels because they
//  have genuinely different structures.
//
//  ---------------------------------------------------------------------------
//  WHY GEMM IS THE ONLY COMPUTE-BOUND KERNEL IN THIS PROJECT
//
//  flops = 2*M*N*K,  compulsory bytes = (M*K + K*N + M*N)*4.
//  At M=N=K=4096 that is 1.374e11 flops against 2.01e8 bytes, so the arithmetic
//  intensity is ~683 FLOP/byte -- twenty times past the T4's ridge point of
//  ~34.5. Every kernel in sec 3a-3c sat 12x-400x BELOW the ridge. This one is
//  the reason the measured 8.130 TFLOP/s FMA peak from Phase 1 exists.
//
//  Note carefully: AI is a property of the OPERATION, not of an implementation.
//  The naive kernel is often quoted at "AI = 0.25", which is its ACCESS-PATTERN
//  intensity (2K flops per 8K bytes touched) -- a different quantity. Feeding
//  that number to the roofline as the denominator makes a naive kernel running
//  at 2% of peak report 550% of peak. bench/gemm_bench.cpp uses the operation's
//  compulsory bytes for all eight rows for exactly this reason.
//
//  ---------------------------------------------------------------------------
//  BOUNDS HANDLING, ONCE, FOR EVERY TILED VARIANT
//
//  Shapes are not multiples of the tile. The cheap and correct answer is to
//  guard the two ENDS and leave the middle alone:
//    * global -> shared loads are predicated, writing 0.0f when out of range
//    * the final C store is predicated
//    * the inner FFMA loop is COMPLETELY UNGUARDED
//  That last point is the trick: a zero in the shared tile contributes exactly
//  zero to the accumulator, so padding with zeros makes the arithmetic correct
//  without a branch in the hot loop. A K that is not a multiple of BK is handled
//  by the same mechanism, at no extra cost.
//
//  ---------------------------------------------------------------------------
//  beta == 0 MUST NOT READ C
//
//  Not an optimisation -- a correctness requirement. C is frequently
//  uninitialised (or deliberately poisoned) on entry, and 0 * NaN = NaN, so
//  multiplying an unread garbage value by a zero beta does not give zero. cuBLAS
//  guarantees the non-reading semantic, so a variant that reads C anyway would
//  disagree with the ceiling row it is being compared against.
// =============================================================================
#include "mcke/kernels/kernels.hpp"
#include "mcke/runtime/cublas_check.hpp"
#include "mcke/runtime/cuda_check.hpp"

#include <mutex>
#include <utility>
#include <vector>

namespace mcke::kernels {
namespace {

// Every GEMM kernel in this file has the SAME signature, deliberately. That is
// what lets the dispatch table below be a plain array of function pointers with
// one entry per (variant, tile) pair, consumed by BOTH launch_gemm_f32 and
// gemm_kernel_attrs. Two parallel switch statements -- one to launch, one to
// query attributes -- would drift with no failing test: the regs/thread and
// occupancy columns of RESULTS.md would faithfully describe an instantiation
// that was never timed. Internally consistent, entirely wrong, and silent.
using GemmKernelPtr = void (*)(const float* __restrict__, const float* __restrict__,
                               float* __restrict__, std::int64_t, std::int64_t,
                               std::int64_t, float, float);

// Applies alpha/beta and stores. Shared by every variant so the beta==0 rule
// above is implemented exactly once.
__device__ __forceinline__ void store_c(float* __restrict__ c, std::int64_t idx,
                                        float acc, float alpha, float beta) {
  const float prev = (beta != 0.0f) ? c[idx] : 0.0f;   // guarded: see banner
  c[idx] = alpha * acc + beta * prev;
}

// -----------------------------------------------------------------------------
// Rows 1 and 2: naive, one thread per output element.
//
// These two kernels differ in EXACTLY ONE LINE -- which of threadIdx.x/.y
// indexes the row. Same grid, same block shape, same instruction count, same
// arithmetic. Only the thread -> element map inside the block changes.
//
// WHY THAT ONE LINE IS WORTH A ROW OF THE TABLE. A warp is 32 threads with
// consecutive threadIdx.x. In the COALESCED version those 32 lanes read 32
// consecutive floats of a B row and write 32 consecutive floats of a C row:
// 128 contiguous bytes = 4 sectors of 32 B, the hardware ideal. In the
// UNCOALESCED version the same 32 lanes hit 32 DIFFERENT ROWS, whose starts are
// K*4 = 16 KiB apart, so a single request becomes 32 separate 32-byte sectors --
// 8x the bytes moved for the same useful data.
//
// The uncoalesced variant is measured rather than asserted because it is what
// JUSTIFIES using the coalesced one as the ladder's baseline. An uncoalesced
// baseline would silently inflate every speedup above it by the same 5-10x.
//
// Honest caveat, recorded before the run: the transpose changes THREE things at
// once, and the measured slowdown is their net. A reads become 32-sector
// (worse), C writes become 32-sector (worse), but B reads become a broadcast --
// all 32 lanes want the same element -- which is BETTER. So the row is labelled
// "thread->output transpose", not "coalescing", and the attribution belongs to
// the sectors-per-request metric rather than to the wall clock.
// -----------------------------------------------------------------------------
constexpr int kNaiveTile = 16;   // 16x16 = 256 threads/block

__global__ __launch_bounds__(kNaiveTile* kNaiveTile)
void gemm_naive_kernel(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, std::int64_t m, std::int64_t n,
                       std::int64_t k, float alpha, float beta) {
  // threadIdx.x -> COLUMN. Consecutive lanes touch consecutive columns, so the
  // B load and the C store both coalesce into 4 sectors per request.
  const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * kNaiveTile + threadIdx.x;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * kNaiveTile + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (std::int64_t p = 0; p < k; ++p)
    acc += a[row * k + p] * b[p * n + col];   // A: broadcast, B: coalesced
  store_c(c, row * n + col, acc, alpha, beta);
}

__global__ __launch_bounds__(kNaiveTile* kNaiveTile)
void gemm_naive_uncoalesced_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ c, std::int64_t m, std::int64_t n,
                                   std::int64_t k, float alpha, float beta) {
  // THE ONE CHANGED LINE PAIR: threadIdx.x now indexes the ROW. The grid and the
  // block geometry are identical to the kernel above, so nothing else can be
  // credited with the difference.
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * kNaiveTile + threadIdx.x;
  const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * kNaiveTile + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (std::int64_t p = 0; p < k; ++p)
    acc += a[row * k + p] * b[p * n + col];   // A: 32 rows 16 KiB apart, B: broadcast
  store_c(c, row * n + col, acc, alpha, beta);
}

// -----------------------------------------------------------------------------
// Row 3: shared-memory tiling, 32x32x32, one output element per thread.
//
// THE IDEA. The naive kernel reads each element of A once per output COLUMN
// (N times) and each element of B once per output ROW (M times). Staging a
// BM x BK slab of A and a BK x BN slab of B in shared memory means each element
// is fetched from global once per TILE instead: A is read N/BN times, B is read
// M/BM times. At 4096 with BN=32 that is a 128x cut in modelled DRAM traffic.
//
// WHY NO PADDING HERE, AND WHY THAT IS NOT A GENERAL RULE. 32 banks of 4 B, and
// blockDim.x = 32, so a warp is exactly one `ty` row of the block:
//   store  As[ty][tx] / Bs[ty][tx] : address ty*32 + tx, bank = tx -> 32
//                                    distinct banks, conflict-free
//   read   As[ty][p]               : ty and p are both warp-uniform -> one
//                                    address, a BROADCAST, conflict-free
//   read   Bs[p][tx]               : address p*32 + tx, bank = tx -> 32 distinct
//                                    banks, conflict-free
// So this tile is conflict-free with no padding at all. But read the reason:
// it is conflict-free BECAUSE the shared row length is exactly 32 floats = the
// bank count. That is a property of BN=32, not a general truth -- Stage 6's
// 128-wide tile violates it immediately and needs a +1 pad on the A slab to
// avoid an 8-way store conflict. "Never pad" and "always pad" are both wrong;
// the row length modulo 32 is what decides.
//
// WHAT ACTUALLY LIMITS THIS KERNEL, PREDICTED IN ADVANCE. Not conflicts -- the
// shared-load to FFMA ratio. The inner loop does 2 shared loads per FMA, i.e.
// 8 B of shared traffic per 2 FLOP = 0.25 FLOP/B. The T4's shared bandwidth is
// 32 banks * 4 B * 1.59 GHz * 40 SM = 8.14 TB/s, giving a ceiling of
// ~2.03 TFLOP/s = 25% of the measured 8.130 FMA peak. Independently, the
// modelled DRAM traffic (A and B each re-read 4096/32 = 128x = 17.2 GB) at
// 235.4 GB/s gives ~73 ms = ~1.88 TFLOP/s = 23%. TWO independent limiters
// landing within two points of each other -- so the sec 3d prediction of 15-25%
// is right but does not identify WHICH roof was hit, and only
// `dram__bytes_read.sum` can settle it.
//
// OCCUPANCY. 1024 threads/block on a chip with a 1024-thread SM cap means at
// most ONE resident block per SM: 100% warp occupancy with ZERO cross-block
// overlap to hide the __syncthreads barriers. It also means the register budget
// is exactly 64/thread with no headroom (32 warps * ceil(R*32/256) <= 65536
// forces R <= 64), so R=65 does not run slowly -- it fails to launch with
// cudaErrorLaunchOutOfResources. __launch_bounds__ makes ptxas responsible for
// fitting, and turns a runtime failure on Colab into a spill count visible in
// -Xptxas -v on the Mac.
// -----------------------------------------------------------------------------
template <int BS>
__global__ __launch_bounds__(BS* BS, 1)
void gemm_tiled_smem_kernel(const float* __restrict__ a, const float* __restrict__ b,
                            float* __restrict__ c, std::int64_t m, std::int64_t n,
                            std::int64_t k, float alpha, float beta) {
  __shared__ float As[BS][BS];
  __shared__ float Bs[BS][BS];
  // Pins the host-side smem_bytes() -- which feeds the occupancy calculator and
  // the RESULTS.md column -- to what the device actually allocates. Checkable on
  // the Mac: the fake-CUDA harness maps __shared__ to static, which preserves
  // sizeof, so this assert is verified with no GPU present.
  static_assert(sizeof(As) + sizeof(Bs) ==
                    smem_bytes(GemmTile{BS, BS, BS, 1, 1}, /*double_buffered=*/false),
                "shared-memory footprint disagrees with gemm_tile.hpp::smem_bytes");

  const int tx = threadIdx.x, ty = threadIdx.y;
  const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * BS + tx;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * BS + ty;

  float acc = 0.0f;
  for (std::int64_t k0 = 0; k0 < k; k0 += BS) {
    // Predicated staging loads -- zero-fill out of range, so the inner loop
    // below needs no bounds check at all (see the banner).
    const std::int64_t ak = k0 + tx;
    const std::int64_t bk = k0 + ty;
    As[ty][tx] = (row < m && ak < k) ? a[row * k + ak] : 0.0f;
    Bs[ty][tx] = (bk < k && col < n) ? b[bk * n + col] : 0.0f;
    __syncthreads();                      // tile is published

    #pragma unroll
    for (int p = 0; p < BS; ++p)
      acc += As[ty][p] * Bs[p][tx];       // broadcast x stride-1: no conflicts

    __syncthreads();                      // all reads done before the next store
                                          // overwrites the tile -- WITH ONE
                                          // BUFFER THIS BARRIER IS MANDATORY.
  }

  if (row < m && col < n) store_c(c, row * n + col, acc, alpha, beta);
}

// =============================================================================
//  Rows 4-7: register blocking, warp tiling, double buffering, vectorized loads.
//
//  ONE KERNEL TEMPLATE, FOUR INSTANTIATIONS, and each step down the ladder flips
//  exactly ONE template parameter:
//
//      tiled_regblock    <..., kRowMajor, DBUF=false, VW=1>
//      warptile_nodbuf   <..., kWarp4x8,  DBUF=false, VW=1>   <- lane map only
//      warptile_dbuf     <..., kWarp4x8,  DBUF=true,  VW=1>   <- buffering only
//      warptile_vec4     <..., kWarp4x8,  DBUF=true,  VW=4>   <- load width only
//
//  The one-variable-per-row rule is therefore enforced by the type system rather
//  than promised in a comment: there is no way for two changes to sneak into one
//  row, because the rows differ by one template argument and share every line of
//  code.
//
//  ---------------------------------------------------------------------------
//  WHY REGISTER BLOCKING IS THE BIG STEP
//
//  tiled_smem does 2 shared loads per FMA. That is 8 B of shared traffic per
//  2 FLOP = 0.25 FLOP/B, and the T4's shared bandwidth caps that at ~2 TFLOP/s
//  no matter how good the rest of the kernel is.
//
//  Giving each thread a TM x TN micro-tile in registers changes the ratio
//  outright. Per k-step a thread loads TM + TN = 16 shared words and does
//  TM * TN = 64 FMAs: 16 words per 128 FLOP = 2 FLOP/B, an 8x improvement in
//  the metric that was binding. The accumulator never leaves registers across
//  the entire k-loop.
//
//  THIS ONLY WORKS IF acc IS COMPILE-TIME SIZED. `float acc[TM][TN]` with
//  constant TM/TN lives in registers. With runtime bounds it would be "local
//  memory", which is a polite name for DRAM, and the kernel would be slower than
//  the one it replaced while looking identical in source. `local_bytes` in
//  GemmKernelAttrs is printed by the bench precisely so this failure is visible
//  instead of merely suspected.
//
//  ---------------------------------------------------------------------------
//  THE SHARED-MEMORY BANK ARITHMETIC, DERIVED RATHER THAN COPIED
//
//  32 banks x 4 B, so bank(addr_in_floats) = addr % 32. A warp's access costs one
//  phase per distinct address per bank; identical addresses BROADCAST for free.
//
//  (1) A IS STORED TRANSPOSED, as As[BK][BM], so that a thread's TM-element A
//      fragment is CONTIGUOUS in shared memory (a column of the A tile becomes a
//      row). Without the transpose the fragment load would stride by BM.
//
//  (2) THE TRANSPOSED STORE IS WHERE THE 8-WAY CONFLICT LIVES. With the natural
//      unpadded layout, stride BM = 128, and the scalar loader's decomposition
//      (arow = tid/BK, acol = tid%BK):
//          bank = (acol*128 + arow) % 32 = arow % 32
//      Lanes 0-7 all have arow = 0 and acol = 0..7, so they hit EIGHT DISTINCT
//      ADDRESSES IN BANK 0 -- an 8-way conflict, paid K/BK = 512 times per block.
//
//  (3) THE FIX IS PAD = 4, NOT PAD = 1. Padding to stride BM + p makes
//          bank = (acol*(128 + p) + arow) % 32 = (acol*p + arow) % 32
//      and we need that injective over acol in 0..7, arow in 0..3 (the 32 lanes
//      of a warp). p = 1 gives acol + arow, which collides immediately
//      ((0,1) and (1,0) both land in bank 1) -- still a 4-way conflict. p = 4
//      gives 4*acol + arow, which walks 0,4,8,...,28 with a 0..3 offset and
//      therefore covers all 32 banks EXACTLY ONCE. Conflict-free.
//      Cost: BK*4 floats = 128 B per buffer. Occupancy unchanged.
//
//  (4) THE PAD ALSO MAKES ROW 7 A ONE-VARIABLE CHANGE, which is the real payoff.
//      The VW=4 loader must decompose differently (arow = tid/2, acol = 4*(tid%2))
//      because a float4 of A is 4 consecutive K values, not 4 consecutive rows.
//      Under stride 132 its store bank is (4*acol + arow) % 32 with arow in 0..15
//      and acol in {i, 4+i}, which also covers all 32 banks exactly once. So both
//      loaders are conflict-free, vectorization changes ONLY the load width, and
//      the row does not silently carry a second cause. Had we left the store
//      8-way conflicting, vec4 would have improved it to 2-way as a side effect
//      and the attribution would have been ruined.
//
//  (5) B NEEDS NO PADDING. Store bank = (brow*BN + bcol) % 32 = bcol % 32, and a
//      warp covers bcol = 0..31 -> 32 distinct banks. This is the same reason
//      tiled_smem needs no padding, and it is why "always pad" is as wrong as
//      "never pad": what matters is the leading dimension modulo 32.
//
//  (6) THE FRAGMENT READ IS THE CONFLICT WARP TILING ADDRESSES, and it cannot be
//      eliminated -- only reduced. Reading Bs[p][thread_col*TN + i], the bank is
//      (8*thread_col + i) % 32, which has PERIOD 4 in thread_col. At most four
//      distinct banks are reachable by any set of TN-aligned column groups, so no
//      lane permutation can make it conflict-free:
//          kRowMajor (2x16 lanes) : A 1-way (broadcast), B 4-way -> 5 phases
//          kWarp4x8  (4x8  lanes) : A 1-way (broadcast), B 2-way -> 3 phases
//      A 1.67x cut in shared-load cycles, not elimination. The Phase-0 header
//      claimed removal; it was wrong, and RESULTS.md sec 3d records the revision
//      (5-12% predicted, not "comparable to double buffering").
//
//  ---------------------------------------------------------------------------
//  DOUBLE BUFFERING: WHY ONE BARRIER REQUIRES TWO BUFFERS
//
//  The single-buffer loop needs TWO barriers per k-step:
//      publish tile k -> smem;  __syncthreads();  compute;  __syncthreads();
//  The second one is not optional. Without it a fast thread races ahead and
//  overwrites smem with tile k+1 while a slow thread in the same block is still
//  reading tile k -- a silent wrong answer that depends on scheduling.
//
//  Dropping to ONE barrier is only legal because the store targets a DISJOINT
//  buffer, so it cannot alias the reads it overlaps:
//
//      prefetch tile k+1 -> REGISTERS      (global loads issued here...)
//      compute on smem[cur]                (...~512 FFMAs of latency cover)
//      store registers -> smem[nxt]        (disjoint: cannot race the reads)
//      __syncthreads();                    (the ONE barrier)
//
//  DISCLOSED: this row therefore has TWO causes, not one -- barriers 2 -> 1, and
//  the global-load issue point moving a whole compute phase earlier so DRAM
//  latency overlaps arithmetic. The second is probably the larger half.
//  Separating them would need a third variant (two buffers, no register
//  prefetch); we did not build it, so the limitation is stated instead of hidden.
//
//  The k-loop is manually unrolled 2x so the buffer index is a COMPILE-TIME
//  constant. With a runtime `smem[cur]` ptxas emits register-offset shared
//  addressing and loses the immediate-offset form of LDS.
// =============================================================================

// Pad on the A slab's leading dimension. 4, derived in note (3) above -- not 1,
// which is the reflexive answer and still leaves a 4-way conflict.
constexpr int kGemmAPad = 4;

enum class LaneMap {
  kRowMajor,   // thread_row = tid / 16, thread_col = tid % 16   (the obvious map)
  kWarp4x8,    // 8 warps as a 4x2 grid, 32 lanes as 4x8         (the warp tile)
};

template <int BM, int BN, int BK, int TM, int TN, LaneMap LM, bool DBUF, int VW>
__global__ __launch_bounds__((BM / TM) * (BN / TN), 2)
void gemm_regblock_kernel(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ c, std::int64_t m, std::int64_t n,
                          std::int64_t k, float alpha, float beta) {
  constexpr int kThreadsPB = (BM / TM) * (BN / TN);   // 256
  constexpr int kAPerThread = BM * BK / kThreadsPB;   // 4
  constexpr int kBPerThread = BK * BN / kThreadsPB;   // 4
  constexpr int kNBuf = DBUF ? 2 : 1;
  constexpr int kALd  = BM + kGemmAPad;               // 132: the padded stride

  static_assert(kAPerThread == VW * (kAPerThread / VW), "A tile must divide by VW");
  static_assert(kBPerThread == VW * (kBPerThread / VW), "B tile must divide by VW");
  // The vectorized path issues exactly ONE float4 per thread per tile. If a
  // future tile made kAPerThread 8, that path would silently load half the tile
  // and leave the rest as stale registers -- so make it a compile error instead.
  static_assert(VW == 1 || kAPerThread == VW,
                "VW>1 assumes one vector load per thread per tile");
  static_assert(VW == 1 || kBPerThread == VW,
                "VW>1 assumes one vector load per thread per tile");

  // alignas(16) is load-bearing for VW=4, not decoration: a shared float array is
  // only guaranteed 4-byte aligned, and the B publish does a 128-bit store
  // through a float4*. Without this the address could be 4- or 8-aligned and the
  // store would fault (or silently split) on hardware, while a scalar-only build
  // would never notice.
  alignas(16) __shared__ float As[kNBuf][BK][kALd];  // TRANSPOSED + padded, notes (1),(3)
  alignas(16) __shared__ float Bs[kNBuf][BK][BN];    // natural order, unpadded, note (5)

  // Pins the host-side smem model (which feeds the occupancy calculator AND the
  // RESULTS.md column) to what the device actually allocates. Verified on a Mac:
  // the fake-CUDA harness maps __shared__ to static, which preserves sizeof.
  static_assert(sizeof(As) + sizeof(Bs) ==
                    smem_bytes(GemmTile{BM, BN, BK, TM, TN}, DBUF, kGemmAPad),
                "shared-memory footprint disagrees with gemm_tile.hpp::smem_bytes");

  const int tid = static_cast<int>(threadIdx.x);

  // --- Lane map. THE ONLY DIFFERENCE between tiled_regblock and
  //     warptile_nodbuf. Same tile, same thread count, same shared memory, same
  //     instruction mix -- a pure permutation of which output patch each lane
  //     owns. Its only observable effect is the bank pattern of the B fragment
  //     load, note (6).
  int thread_row, thread_col;
  if constexpr (LM == LaneMap::kRowMajor) {
    thread_row = tid / (BN / TN);
    thread_col = tid % (BN / TN);
  } else {
    // The 4x2 warp grid and 4x8 lane grid below are specific to a 16x16 thread
    // grid (8 warps). A different tile would need a different factorisation, and
    // getting it silently wrong would produce threads that own overlapping
    // output patches -- a race on C with no diagnostic. Fail to compile instead.
    static_assert(kThreadsPB == 256 && (BM / TM) == 16 && (BN / TN) == 16,
                  "kWarp4x8 assumes a 16x16 thread grid (8 warps as 4x2, lanes as 4x8)");
    const int warp = tid / 32, lane = tid % 32;
    thread_row = (warp / 2) * 4 + (lane / 8);   // 4x2 warps, 4x8 lanes
    thread_col = (warp % 2) * 8 + (lane % 8);
  }

  const std::int64_t block_row = static_cast<std::int64_t>(blockIdx.y) * BM;
  const std::int64_t block_col = static_cast<std::int64_t>(blockIdx.x) * BN;

  // --- Loader decomposition. VW=4 needs a different one because a float4 of A
  //     is four consecutive K values (one row), not four consecutive rows.
  //     Both are conflict-free under kGemmAPad = 4 -- note (4).
  int arow, acol, brow, bcol;
  if constexpr (VW == 1) {
    arow = tid / BK;            // 0..31, +32 per step
    acol = tid % BK;            // 0..7
    brow = tid / 32;            // 0..7
    bcol = tid % 32;            // 0..31, +32 per step
  } else {
    arow = tid / (BK / VW);            // 0..127, one float4 each
    acol = (tid % (BK / VW)) * VW;     // 0 or 4
    brow = tid / (BN / VW);            // 0..7
    bcol = (tid % (BN / VW)) * VW;     // 0,4,...,124
  }

  float acc[TM][TN];                    // COMPILE-TIME sized: must stay in registers
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  float ra[kAPerThread], rb[kBPerThread];   // the prefetch staging registers

  // --- Global -> registers. Predicated: out-of-range reads become 0.0f, which
  //     is what lets the inner FFMA loop run with no bounds checks at all (a
  //     zero in the tile contributes zero to the accumulator).
  auto prefetch = [&](std::int64_t k0) {
    if constexpr (VW == 1) {
      #pragma unroll
      for (int i = 0; i < kAPerThread; ++i) {
        const std::int64_t r = block_row + arow + i * (kThreadsPB / BK);
        const std::int64_t cc = k0 + acol;
        ra[i] = (r < m && cc < k) ? a[r * k + cc] : 0.0f;
      }
      #pragma unroll
      for (int j = 0; j < kBPerThread; ++j) {
        const std::int64_t r = k0 + brow;
        const std::int64_t cc = block_col + bcol + j * 32;
        rb[j] = (r < k && cc < n) ? b[r * n + cc] : 0.0f;
      }
    } else {
      // One LDG.128 each instead of four LDG.32 -- identical bytes, a quarter of
      // the load instructions. The launcher guarantees k % 4 == 0 and n % 4 == 0
      // before selecting this instantiation, so a float4 never straddles a row
      // end; the row-tail predicate below is on whole float4s.
      const std::int64_t ar = block_row + arow;
      const std::int64_t ac = k0 + acol;
      if (ar < m && ac + VW <= k) {
        const float4 v = *reinterpret_cast<const float4*>(&a[ar * k + ac]);
        ra[0] = v.x; ra[1] = v.y; ra[2] = v.z; ra[3] = v.w;
      } else {
        #pragma unroll
        for (int i = 0; i < VW; ++i)
          ra[i] = (ar < m && ac + i < k) ? a[ar * k + ac + i] : 0.0f;
      }
      const std::int64_t br = k0 + brow;
      const std::int64_t bc = block_col + bcol;
      if (br < k && bc + VW <= n) {
        const float4 v = *reinterpret_cast<const float4*>(&b[br * n + bc]);
        rb[0] = v.x; rb[1] = v.y; rb[2] = v.z; rb[3] = v.w;
      } else {
        #pragma unroll
        for (int j = 0; j < VW; ++j)
          rb[j] = (br < k && bc + j < n) ? b[br * n + bc + j] : 0.0f;
      }
    }
  };

  // --- Registers -> shared. A is TRANSPOSED here: As[col][row].
  auto publish = [&](int buf) {
    if constexpr (VW == 1) {
      #pragma unroll
      for (int i = 0; i < kAPerThread; ++i)
        As[buf][acol][arow + i * (kThreadsPB / BK)] = ra[i];
      #pragma unroll
      for (int j = 0; j < kBPerThread; ++j)
        Bs[buf][brow][bcol + j * 32] = rb[j];
    } else {
      // The float4 of A becomes FOUR scattered 32-bit shared stores, because the
      // transpose scatters consecutive-in-K values across BK rows of the slab.
      // The vectorized global load survives; the vectorized shared store does
      // not. Still conflict-free -- note (4).
      #pragma unroll
      for (int i = 0; i < VW; ++i) As[buf][acol + i][arow] = ra[i];
      // B keeps its natural order, so its store stays a single 128-bit write.
      *reinterpret_cast<float4*>(&Bs[buf][brow][bcol]) =
          make_float4(rb[0], rb[1], rb[2], rb[3]);
    }
  };

  // --- The arithmetic. TM + TN shared loads feed TM * TN FMAs: 16 words per
  //     128 FLOP, versus tiled_smem's 2 words per 2 FLOP. That ratio IS register
  //     blocking.
  auto compute = [&](int buf) {
    #pragma unroll
    for (int p = 0; p < BK; ++p) {
      float af[TM], bf[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = As[buf][p][thread_row * TM + i];
      #pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = Bs[buf][p][thread_col * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += af[i] * bf[j];
    }
  };

  const std::int64_t kt = (k + BK - 1) / BK;      // number of k-tiles

  if constexpr (!DBUF) {
    // Single buffer: TWO barriers per tile, and the second one is mandatory --
    // it stops the next tile's store from overtaking this tile's reads.
    for (std::int64_t t = 0; t < kt; ++t) {
      prefetch(t * BK);
      publish(0);
      __syncthreads();          // tile published
      compute(0);
      __syncthreads();          // all reads done before the next store
    }
  } else {
    // Two buffers: ONE barrier per tile. Unrolled 2x so the buffer index is a
    // literal, keeping immediate-offset LDS addressing.
    prefetch(0);
    publish(0);
    __syncthreads();
    // The parity and the tail come from gemm_tile.hpp::dbuf_schedule, which is
    // unit-tested exhaustively on the host. Deriving them inline here is how the
    // last BK columns of K get silently dropped at a shape nobody hand-checks.
    const DbufSchedule sch = dbuf_schedule(kt);
    for (std::int64_t it = 0; it < sch.pair_iters; ++it) {
      const std::int64_t t = it * 2;
      prefetch((t + 1) * BK);   // issued BEFORE compute: its latency hides here
      compute(0);               // buffer 0 holds tile t
      publish(1);               // disjoint from the reads above -- no race
      __syncthreads();          // the ONE barrier of this half-step
      prefetch((t + 2) * BK);   // may be past the end: predicated to zeros, and
      compute(1);               // buffer 1 holds tile t+1
      publish(0);               // leaves buffer 0 holding tile t+2
      __syncthreads();
    }
    // Odd tile count: buffer 0 holds the final tile, published by the last
    // iteration's second prefetch (or by the pre-loop publish when kt == 1).
    if (sch.has_tail) compute(0);
  }

  // --- Epilogue. Predicated store; beta == 0 must not read C.
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const std::int64_t r = block_row + thread_row * TM + i;
    if (r >= m) continue;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const std::int64_t cc = block_col + thread_col * TN + j;
      if (cc < n) store_c(c, r * n + cc, acc[i][j], alpha, beta);
    }
  }
}

// -----------------------------------------------------------------------------
// The dispatch table. ONE definition, consumed by launch_gemm_f32 (to launch)
// and gemm_kernel_attrs (to query what ptxas produced). See GemmKernelPtr above
// for why they must not be two separate switches.
// -----------------------------------------------------------------------------
struct GemmLaunchSpec {
  GemmKernelPtr kernel   = nullptr;
  dim3          block{1, 1, 1};
  int           threads  = 0;
  int           tile_m   = 0;   // output rows per block  -> grid.y
  int           tile_n   = 0;   // output cols per block  -> grid.x
  std::size_t   smem     = 0;   // STATIC shared memory; we use no dynamic smem
};

// `allow_vec4` is false when the runtime shape cannot support 128-bit loads.
// Defaulted true so gemm_kernel_attrs -- which has no shape -- reports the
// instantiation that the pinned benchmark shape actually uses.
[[nodiscard]] StatusOr<GemmLaunchSpec> gemm_launch_spec(GemmVariant variant,
                                                        const GemmTile& tile,
                                                        bool allow_vec4 = true) {
  GemmLaunchSpec s;
  switch (variant) {
    case GemmVariant::kNaive:
    case GemmVariant::kNaiveUncoalesced:
      // The naive variants own no tile -- one thread, one output element -- so
      // the caller's GemmTile is deliberately ignored rather than validated.
      s.kernel  = (variant == GemmVariant::kNaive) ? &gemm_naive_kernel
                                                   : &gemm_naive_uncoalesced_kernel;
      s.block   = dim3(kNaiveTile, kNaiveTile, 1);
      s.threads = kNaiveTile * kNaiveTile;
      s.tile_m  = kNaiveTile;
      s.tile_n  = kNaiveTile;
      s.smem    = 0;
      return s;

    case GemmVariant::kTiledSmem: {
      if (!tile_is_self_consistent(tile))
        return InvalidArgumentError("launch_gemm_f32: tile is not self-consistent");
      // Reject rather than fall back to the nearest compiled tile. Same rule
      // launch_bias_act_f32 follows for an illegal vector width: a benchmark
      // that cannot tell you what it actually measured is worse than one that
      // errors.
      if (select_tile_config(tile) != GemmTileConfig::k32x32x32_1x1)
        return InvalidArgumentError(
            "launch_gemm_f32: kTiledSmem is only instantiated for tile "
            "(bm,bn,bk,tm,tn) = (32,32,32,1,1); no fallback is applied");
      s.kernel  = &gemm_tiled_smem_kernel<32>;
      s.block   = dim3(32, 32, 1);
      s.threads = 32 * 32;
      s.tile_m  = 32;
      s.tile_n  = 32;
      s.smem    = smem_bytes(tile, /*double_buffered=*/false);
      return s;
    }

    case GemmVariant::kTiledRegBlock:
    case GemmVariant::kWarpTile:
    case GemmVariant::kWarpTileNoDbuf:
    case GemmVariant::kWarpTileVec4: {
      if (!tile_is_self_consistent(tile))
        return InvalidArgumentError("launch_gemm_f32: tile is not self-consistent");
      if (select_tile_config(tile) != GemmTileConfig::k128x128x8_8x8)
        return InvalidArgumentError(
            "launch_gemm_f32: the register-blocked variants are only instantiated "
            "for tile (bm,bn,bk,tm,tn) = (128,128,8,8,8); no fallback is applied");

      // Four instantiations of ONE template, each flipping exactly one argument
      // relative to the row above it. That is the one-variable-per-row rule,
      // enforced by the type system instead of by discipline.
      constexpr auto kRowMaj = LaneMap::kRowMajor;
      constexpr auto kWarp   = LaneMap::kWarp4x8;
      bool dbuf = false;
      switch (variant) {
        case GemmVariant::kTiledRegBlock:
          s.kernel = &gemm_regblock_kernel<128, 128, 8, 8, 8, kRowMaj, false, 1>;
          break;
        case GemmVariant::kWarpTileNoDbuf:
          s.kernel = &gemm_regblock_kernel<128, 128, 8, 8, 8, kWarp, false, 1>;
          break;
        case GemmVariant::kWarpTile:
          s.kernel = &gemm_regblock_kernel<128, 128, 8, 8, 8, kWarp, true, 1>;
          dbuf = true;
          break;
        default:   // kWarpTileVec4
          // Falls back to the scalar instantiation when the shape cannot support
          // 128-bit loads. That fallback is the SAME kernel as warptile_dbuf, so
          // at an unaligned shape this row measures warptile_dbuf -- which is
          // correct behaviour (a float4 load cannot be partially predicated) but
          // must not be mistaken for a vec4 measurement. The benchmark shape is
          // 4096-cubed and always takes the vectorized path; the 253-cubed
          // validation shape exists to exercise this branch.
          s.kernel = allow_vec4
              ? &gemm_regblock_kernel<128, 128, 8, 8, 8, kWarp, true, 4>
              : &gemm_regblock_kernel<128, 128, 8, 8, 8, kWarp, true, 1>;
          dbuf = true;
          break;
      }
      s.threads = threads_per_block(tile);          // 256
      s.block   = dim3(static_cast<unsigned>(s.threads), 1, 1);
      s.tile_m  = tile.bm;
      s.tile_n  = tile.bn;
      s.smem    = smem_bytes(tile, dbuf, kGemmAPad);
      return s;
    }

    case GemmVariant::kCublasRef:
      // Not a kernel of ours. Both callers special-case it before getting here.
      return InvalidArgumentError("gemm_launch_spec: kCublasRef has no MCKE kernel");
  }
  return InvalidArgumentError("gemm_launch_spec: unknown GemmVariant");
}

// -----------------------------------------------------------------------------
// Row 8: cuBLAS, the ceiling.
//
// Not cheating. Knowing the gap is the only way to know whether 65% of peak is
// good, and docs/ROADMAP.md makes "a written explanation of the remaining gap to
// cuBLAS" the Phase-3 exit criterion.
//
// THE ROW-MAJOR / COLUMN-MAJOR SWAP TRICK, spelled out because it is the single
// easiest thing in this file to get wrong and the hardest to notice:
//
//   A row-major matrix X[r][c] with leading dimension `c` occupies EXACTLY the
//   same bytes as a column-major X^T with leading dimension `c`. Storage order
//   and transposition are the same operation, so reinterpreting costs nothing.
//
//   We want row-major C = A @ B. Transposing: C^T = B^T @ A^T. Reading each of
//   our row-major buffers as its column-major transpose, that is a plain
//   no-transpose column-major GEMM with the OPERANDS SWAPPED:
//
//       (col-major, N x M)  =  (col-major, N x K)  @  (col-major, K x M)
//            C^T                     B^T                    A^T
//
//   cublasSgemm(h, N, N, m=N, n=M, k=K, alpha, B, lda=N, A, ldb=K, beta, C, ldc=N)
//
//   No CUBLAS_OP_T anywhere -- passing a transpose flag here is the classic way
//   to produce a plausible-looking but wrong baseline. Because a transposed
//   result still passes for a wide class of inputs when M == N, the cuBLAS row
//   MUST be validated at a non-square shape before it is trusted as an oracle.
// -----------------------------------------------------------------------------

// The handle is created once and DELIBERATELY LEAKED.
//
// Two reasons it is not a per-call create/destroy: cublasCreate allocates a
// workspace and costs O(100 us)-O(100 ms), which inside a timed loop would be
// measuring cuBLAS's constructor rather than its GEMM; and the frozen launcher
// signature has nowhere to pass a handle in from outside.
//
// It is leaked rather than wrapped in an RAII type on purpose: a static
// destructor calling cublasDestroy runs during program teardown, potentially
// AFTER the CUDA context has been destroyed, which is a classic crash-at-exit.
// The OS reclaims it either way. The leak is the safer of the two bugs.
[[nodiscard]] StatusOr<cublasHandle_t> cublas_handle_for(rt::StreamHandle stream) {
  // ONE HANDLE PER STREAM, keyed on the stream itself.
  //
  // Phase 3 had a single static handle and that was safe, because Phase 3 only
  // ever had one stream. PHASE 4 IS THE FIRST PHASE THAT CAN RUN TWO cuBLAS
  // GEMMs CONCURRENTLY, and a cuBLAS handle carries internal workspace: two
  // concurrent SGEMMs sharing one handle corrupt each other's results. NVIDIA's
  // rule is one handle per concurrent stream.
  //
  // Keyed on rt::StreamHandle rather than a stream index because
  // launch_gemm_f32's signature is frozen and receives a handle, not an index.
  // The mutex is cheap insurance: our executor enqueues from one host thread, so
  // it is uncontended, but a handle cache that silently required single-threaded
  // access would be a trap for whoever adds a worker pool.
  static std::mutex mu;
  static std::vector<std::pair<rt::StreamHandle, cublasHandle_t>> cache;
  {
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& kv : cache)
      if (kv.first == stream) return kv.second;
  }

  cublasHandle_t handle = nullptr;
  cublasStatus_t create_status = CUBLAS_STATUS_SUCCESS;
  {
    create_status = cublasCreate(&handle);
    if (create_status != CUBLAS_STATUS_SUCCESS)
      return rt::cublas_status(create_status, "cublasCreate", __FILE__, __LINE__);
    // PEDANTIC, not DEFAULT. Without this, cuBLAS is free to run SGEMM on TF32
    // tensor cores on sm_80+ (L4 / A100 / 5060 are all in CLAUDE.md's target
    // table) -- which would measure the ceiling row in DIFFERENT ARITHMETIC than
    // the seven hand-written f32 rows it is the ceiling for, and report a
    // "% of measured FMA peak" above 100% against a denominator from a pure-f32
    // microbenchmark. RESULTS.md rule 5 exists to prevent exactly this.
    create_status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (create_status != CUBLAS_STATUS_SUCCESS)
      return rt::cublas_status(create_status, "cublasSetMathMode", __FILE__, __LINE__);
    // alpha/beta are passed as pointers to host stack locals below, which is
    // only legal in HOST pointer mode. It is the default, but the default is not
    // guaranteed across versions, so state it.
    create_status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (create_status != CUBLAS_STATUS_SUCCESS)
      return rt::cublas_status(create_status, "cublasSetMathMode/SetPointerMode",
                               __FILE__, __LINE__);
  }
  {
    std::lock_guard<std::mutex> lk(mu);
    // Re-check: another thread may have created one for this stream while we
    // were outside the lock. Keep theirs and leak ours -- see below on why
    // cublasDestroy is never called here.
    for (const auto& kv : cache)
      if (kv.first == stream) return kv.second;
    cache.emplace_back(stream, handle);
  }
  return handle;
}

[[nodiscard]] Status launch_cublas(const float* a, const float* b, float* c,
                                   std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, float beta, rt::StreamHandle stream) {
  auto h = cublas_handle_for(stream);
  MCKE_RETURN_IF_ERROR(h.status());

  // MANDATORY, not hygiene. rt::Stream creates every stream with
  // cudaStreamNonBlocking, which by definition does NOT implicitly synchronise
  // with the legacy default stream. Without this call cuBLAS would run on the
  // default stream while Profiler::time_op's event pair brackets an empty one --
  // producing a cuBLAS row of ~0.01 ms and an absurd TFLOP/s that looks like a
  // triumph. Re-set on every call because the stream is a parameter.
  MCKE_CUBLAS_RETURN_IF_ERROR(cublasSetStream(*h, stream));

  const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
  // See the banner: operands swapped, no transpose flags, leading dimensions are
  // the row-major row lengths.
  MCKE_CUBLAS_RETURN_IF_ERROR(cublasSgemm(*h, CUBLAS_OP_N, CUBLAS_OP_N,
                                          /*m=*/ni, /*n=*/mi, /*k=*/ki,
                                          &alpha,
                                          /*A=*/b, /*lda=*/ni,
                                          /*B=*/a, /*ldb=*/ki,
                                          &beta,
                                          /*C=*/c, /*ldc=*/ni));
  return OkStatus();
}

}  // namespace

// -----------------------------------------------------------------------------
Status launch_gemm_f32(const float* a, const float* b, float* c,
                       std::int64_t m, std::int64_t n, std::int64_t k,
                       float alpha, float beta,
                       GemmVariant variant, GemmTile tile, rt::StreamHandle stream) {
  if (m == 0 || n == 0) return OkStatus();
  if (m < 0 || n < 0 || k < 0)
    return InvalidArgumentError("launch_gemm_f32: negative extent");
  if (!a || !b || !c) return InvalidArgumentError("launch_gemm_f32: null pointer");
  // __restrict__ on all three pointers is a promise that they do not alias.
  // An in-place call would break it silently rather than loudly -- the same
  // check launch_row_softmax_f32 makes for x == y.
  if (a == c || b == c)
    return InvalidArgumentError("launch_gemm_f32: output aliases an input, which "
                                "violates the __restrict__ contract");
  // cuBLAS takes int, and our signature takes int64_t. Narrowing silently is how
  // a 3-billion-element problem becomes a negative leading dimension.
  if (m > 2147483647LL || n > 2147483647LL || k > 2147483647LL)
    return InvalidArgumentError("launch_gemm_f32: extent exceeds INT32_MAX");

  if (variant == GemmVariant::kCublasRef)
    return launch_cublas(a, b, c, m, n, k, alpha, beta, stream);

  // Can this shape take 128-bit loads?
  //   A's thread address is a[row*k + k0 + 4j], so k % 4 == 0 makes every such
  //     address 16B-aligned relative to the base (k0 is a multiple of BK = 8).
  //   B's is b[(k0+r)*n + 4j], so n % 4 == 0.
  //   M IS UNCONSTRAINED -- it only bounds a row index, and nothing in the
  //     vectorization touches it. Requiring m % 4 would reject valid shapes.
  //   The base pointers must be 16B aligned; cudaMalloc gives 256B, but a
  //     sub-buffer offset by an odd number of floats would not be, so check.
  const bool allow_vec4 =
      (k % 4 == 0) && (n % 4 == 0) &&
      (reinterpret_cast<std::uintptr_t>(a) % 16 == 0) &&
      (reinterpret_cast<std::uintptr_t>(b) % 16 == 0);

  auto spec = gemm_launch_spec(variant, tile, allow_vec4);
  MCKE_RETURN_IF_ERROR(spec.status());

  // k == 0 is a legal degenerate GEMM: the product is empty, so C = beta*C. The
  // kernels handle it (the k-loop simply does not execute) so it is NOT an early
  // return -- returning early would skip the beta scaling and leave C stale.
  const std::int64_t grid_x = (n + spec->tile_n - 1) / spec->tile_n;
  const std::int64_t grid_y = (m + spec->tile_m - 1) / spec->tile_m;
  // gridDim.y and .z are capped at 65535 on every architecture we target, unlike
  // gridDim.x's 2^31-1. At a 16-row tile that caps M at ~1.05M rows.
  if (grid_y > 65535)
    return InvalidArgumentError("launch_gemm_f32: m too large for gridDim.y limit");

  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1);
  spec->kernel<<<grid, spec->block, /*dynamic smem=*/0, stream>>>(a, b, c, m, n, k,
                                                                  alpha, beta);
  MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

StatusOr<GemmKernelAttrs> gemm_kernel_attrs(GemmVariant variant, GemmTile tile) {
  if (variant == GemmVariant::kCublasRef)
    return UnimplementedError("gemm_kernel_attrs: cuBLAS's kernels are not ours to "
                              "introspect; the ceiling row has no regs/smem column");

  auto spec = gemm_launch_spec(variant, tile);
  MCKE_RETURN_IF_ERROR(spec.status());

  cudaFuncAttributes fa{};
  MCKE_CUDA_RETURN_IF_ERROR(cudaFuncGetAttributes(&fa, spec->kernel));

  GemmKernelAttrs out;
  out.regs_per_thread   = fa.numRegs;
  out.static_smem_bytes = fa.sharedSizeBytes;
  out.local_bytes       = fa.localSizeBytes;   // > 0 means register spilling
  out.threads_per_block = spec->threads;

  MCKE_CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &out.max_blocks_per_sm_api, spec->kernel, spec->threads, /*dynamic smem=*/0));
  return out;
}

}  // namespace mcke::kernels
