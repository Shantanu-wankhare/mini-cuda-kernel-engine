// =============================================================================
//  mcke/kernels/gemm_tile.hpp
//
//  WHAT: The GEMM tile descriptor, the runtime-tile -> compile-time-instantiation
//        dispatch decision, and the four-limiter occupancy calculator.
//
//  WHY .hpp AND NOT .cuh, and why a separate header at all: every function here
//  is PURE HOST C++ over plain integers and a `DeviceInfo` POD. Nothing in this
//  file includes a CUDA header, calls a CUDA API, or needs nvcc -- so all of it
//  can be unit-tested on the MacBook, with no GPU, against cases whose answers
//  are known by hand. `softmax_online.hpp` is the precedent and the argument is
//  identical: the piece of arithmetic that is easy to get subtly wrong gets
//  hoisted out of the .cu so a laptop can check it before a metered GPU session
//  ever sees it.
//
//  This matters more here than it looks. docs/ROADMAP.md's Phase 3d exit
//  criterion is not the TFLOP/s -- it is "compute theoretical occupancy by hand,
//  then compare against Nsight's measured achieved_occupancy. The
//  hand-calculation vs. measurement comparison is the learning." A hand
//  calculation that lives in a comment cannot be wrong in any way that a test
//  notices. As a function, it can.
//
//  ---------------------------------------------------------------------------
//  WHY `GemmTile` LIVES HERE RATHER THAN IN kernels.hpp
//
//  It was declared in kernels/kernels.hpp through Phase 0-3c. But kernels.hpp
//  includes runtime/stream.hpp, which includes runtime/cuda_check.hpp, which is
//  the CUDA boundary. A header that advertises itself as pure host logic while
//  transitively dragging in the CUDA boundary is only accidentally testable --
//  it works today because every CUDA include is guarded, not because the
//  layering is right. Moving the struct down here and having kernels.hpp include
//  this file makes the claim structurally true instead of coincidentally true.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#include "mcke/core/config.hpp"
#include "mcke/core/device.hpp"

namespace mcke::kernels {

// -----------------------------------------------------------------------------
// The tile descriptor
// -----------------------------------------------------------------------------
struct GemmTile {
  int bm = 128, bn = 128, bk = 8;   // block tile: the BM x BN output patch a
                                    // block owns, staged BK columns at a time
  int tm = 8,   tn = 8;             // per-thread (register) micro-tile
  // threads per block = (bm/tm) * (bn/tn); with the defaults, 16*16 = 256.
  // Shared memory per block = (bm*bk + bk*bn) * 4 B * (2 if double-buffered).
  // With the defaults: (128*8 + 8*128)*4 = 8 KiB, x2 = 16 KiB.
  //
  // An earlier version of this comment (in kernels.hpp, before the struct moved
  // here) then said "leaves room for 3 concurrent blocks per SM" and called
  // these "the entire occupancy calculation". Both claims were wrong:
  //   * 64 KiB / 16 KiB = 4 blocks, not 3.
  //   * Occupancy has FOUR limiters, not one -- see OccupancyLimiter below. The
  //     answer is the MINIMUM of all four, and on this kernel shared memory is
  //     NOT the binding one: `float acc[8][8]` alone is 64 registers, and the
  //     register file caps it at 2 blocks, half what shared memory allows.
  // Identifying the obvious constraint and finding it is not the binding one is
  // the actual lesson.
};

// Threads per block implied by the tile. One thread owns one TM x TN micro-tile,
// so the thread grid is exactly the block tile divided by the thread tile.
[[nodiscard]] constexpr int threads_per_block(const GemmTile& t) {
  if (t.tm <= 0 || t.tn <= 0) return 0;          // caller-error guard: no div by 0
  return (t.bm / t.tm) * (t.bn / t.tn);
}

// Shared-memory floats needed to stage one k-slice: a BM x BK slab of A plus a
// BK x BN slab of B. Doubled when double-buffered, because the point of double
// buffering is that the store of tile k+1 targets memory disjoint from the reads
// of tile k -- see kernels/gemm.cu.
//
// `a_row_pad` widens the A slab's leading dimension. It is a parameter rather
// than a constant because it is genuinely a per-kernel choice, and because this
// function feeds BOTH the occupancy calculator and a static_assert inside each
// kernel -- so if the padding were modelled here and not applied there (or vice
// versa) the RESULTS.md occupancy column would describe an allocation that does
// not exist. The static_assert is what keeps the two honest; see kGemmAPad in
// kernels/gemm.cu for why the register-blocked tile pads by exactly 4.
[[nodiscard]] constexpr int smem_floats(const GemmTile& t, bool double_buffered,
                                        int a_row_pad = 0) {
  return (t.bk * (t.bm + a_row_pad) + t.bk * t.bn) * (double_buffered ? 2 : 1);
}

[[nodiscard]] constexpr std::size_t smem_bytes(const GemmTile& t, bool double_buffered,
                                               int a_row_pad = 0) {
  return static_cast<std::size_t>(smem_floats(t, double_buffered, a_row_pad)) *
         sizeof(float);
}

// Is this tile internally coherent, independently of whether we compiled it?
//
// Every one of these is a real failure mode, not a formality:
//   * bm % tm / bn % tn : a leftover strip of the block tile that no thread owns
//   * threads % 32      : a partial warp per block -- silently wasted issue slots
//                         that skew every occupancy number downstream
//   * (bm*bk) % threads : the global->shared staging loop assumes each thread
//                         moves the same whole number of elements. If it does
//                         not divide, the loader needs a bounds check it does
//                         not have, and the tail of the tile is left uninitialised.
[[nodiscard]] constexpr bool tile_is_self_consistent(const GemmTile& t) {
  if (t.bm <= 0 || t.bn <= 0 || t.bk <= 0 || t.tm <= 0 || t.tn <= 0) return false;
  if (t.bm % t.tm != 0 || t.bn % t.tn != 0) return false;
  const int threads = threads_per_block(t);
  if (threads <= 0 || threads > 1024) return false;
  if (threads % kWarpSize != 0) return false;
  if ((t.bm * t.bk) % threads != 0) return false;
  if ((t.bk * t.bn) % threads != 0) return false;
  return true;
}

// -----------------------------------------------------------------------------
// The double-buffered k-loop schedule.
//
// WHY THIS IS A FUNCTION AND NOT JUST A LOOP CONDITION. The double-buffered GEMM
// unrolls its k-loop 2x so the shared-memory buffer index is a compile-time
// literal. That introduces a parity: even tiles are computed from buffer 0, odd
// tiles from buffer 1, and an odd tile count leaves one tile to compute from
// buffer 0 after the loop. Get the tail wrong and the kernel silently drops or
// double-counts the last BK columns of K.
//
// That bug is invisible where it would be caught. The validation shapes use
// K in {1, 3, 257}, and the benchmark uses K = 4096 -- but a 4096-cubed CPU
// reference is minutes, so the shape that produces the published number is the
// one nobody re-derives by hand. Hoisting the schedule here means the parity is
// checked EXHAUSTIVELY on the Mac, for every tile count, before nvcc sees it.
// Same argument as softmax_online.hpp: the arithmetic that is easy to get subtly
// wrong does not get to live only inside a .cu.
struct DbufSchedule {
  std::int64_t pair_iters = 0;   // iterations of the unrolled-by-2 loop
  bool         has_tail   = false;  // one final tile, computed from buffer 0
};

[[nodiscard]] constexpr DbufSchedule dbuf_schedule(std::int64_t k_tiles) {
  if (k_tiles <= 0) return {};
  return DbufSchedule{k_tiles / 2, (k_tiles % 2) != 0};
}

// -----------------------------------------------------------------------------
// Runtime tile -> compile-time instantiation
//
// WHY THIS INDIRECTION EXISTS: tile sizes must be compile-time constants. Shared
// array extents need them, `#pragma unroll` trip counts need them, and above all
// `float acc[TM][TN]` needs them -- a runtime-sized local array cannot live in
// registers, so it spills to local memory (which is DRAM), destroying the exact
// thing register blocking exists to buy. But `GemmTile` is a runtime struct in
// the frozen launcher signature. So the launcher must map a runtime tile onto a
// fixed, explicitly-instantiated set, and REJECT anything outside it.
//
// Rejecting rather than falling back to the nearest tile is the same rule
// launch_bias_act_f32 follows for an illegal vector width: a benchmark that
// cannot tell you what it actually measured is worse than one that errors.
// -----------------------------------------------------------------------------
enum class GemmTileConfig : std::uint8_t {
  k32x32x32_1x1,     // tiled_smem     -- 1024 threads, no register blocking
  k128x128x8_8x8,    // everything from tiled_regblock up
  kUnsupported,      // self-consistent, perhaps, but not compiled
};

[[nodiscard]] constexpr GemmTileConfig select_tile_config(const GemmTile& t) {
  if (t.bm == 32 && t.bn == 32 && t.bk == 32 && t.tm == 1 && t.tn == 1)
    return GemmTileConfig::k32x32x32_1x1;
  if (t.bm == 128 && t.bn == 128 && t.bk == 8 && t.tm == 8 && t.tn == 8)
    return GemmTileConfig::k128x128x8_8x8;
  return GemmTileConfig::kUnsupported;
}

// ---------------------------------------------------------------------------
// APPEND-ONLY ENUM. Every addition goes at the END, so no existing enumerator's
// std::uint8_t value shifts and nothing already recorded or serialised changes
// meaning. The consequence is that NUMERIC ORDER IS NOT PRESENTATION ORDER --
// the ladder reads naive_uncoalesced, naive, tiled_smem, tiled_regblock,
// warptile_nodbuf, warptile_dbuf, warptile_vec4, cublas, and the enum does not.
// bench/gemm_bench.cpp owns the presentation order; do not infer it from here.
//
// kWarpTileNoDbuf exists because kWarpTile bundles TWO independent changes -- a
// warp-level tile layer AND double-buffered shared memory -- which would give
// its row two attributable causes. kWarpTileVec4 exists for the same reason
// applied to vectorized loads. kNaiveUncoalesced exists so the coalesced naive
// baseline is a MEASURED choice rather than an asserted one.
//
// ---------------------------------------------------------------------------
// TWO CORRECTIONS TO THE PHASE-0 DESCRIPTION ABOVE, both found in design review
// before any of this ran, and both recorded rather than quietly edited:
//
// 1. The earlier text said warp tiling "removes a bank conflict on the
//    B-fragment load". It CANNOT remove it. With Bs[BK][BN] and TN=8 a lane's
//    bank is (c*8 + i) % 32, which has PERIOD 4 in the column group c -- at most
//    4 distinct banks are reachable by any set of 8-aligned column groups, no
//    matter how lanes are assigned. A pure lane remap can only go from a 2x16
//    lane layout (A 1-way, B 4-way = 5 phases) to 4x8 (A 1-way, B 2-way =
//    3 phases). That is a 1.67x cut in shared-load cycles, not elimination.
//    Padding does not help either: the period comes from the intra-row stride
//    TN=8, not from the row stride, so widening the row stride leaves it intact.
//
// 2. The earlier text said the two effects were "predicted to be comparable in
//    size". They are not. Warp tiling moves the shared-memory roofline from
//    ~6.5 to ~9-10.9 TFLOP/s, but the kernel runs at ~3.5-4, so that ceiling was
//    only marginally binding to begin with. Revised prediction: kWarpTileNoDbuf
//    buys 5-12%, materially less than double buffering. Recorded here BEFORE the
//    run, per RESULTS.md rule 6 -- the point is to be judged, not to be right.
// ---------------------------------------------------------------------------
enum class GemmVariant : std::uint8_t {
  kNaive, kTiledSmem, kTiledRegBlock, kWarpTile, kCublasRef,
  kWarpTileNoDbuf,      // warp tiling WITHOUT double buffering -- the isolator
  kWarpTileVec4,        // kWarpTile + float4 global->smem loads, nothing else
  kNaiveUncoalesced,    // kNaive with threadIdx.x mapped to the C ROW, not column
};


// Do the naive variants and cuBLAS own a tile? No -- one thread per output
// element, or a library call. They ignore the GemmTile entirely.
[[nodiscard]] constexpr bool gemm_variant_uses_tile(GemmVariant v) {
  return v != GemmVariant::kNaive && v != GemmVariant::kNaiveUncoalesced &&
         v != GemmVariant::kCublasRef;
}

// Which compile-time instantiation a variant requires.
[[nodiscard]] constexpr GemmTileConfig gemm_required_config(GemmVariant v) {
  return v == GemmVariant::kTiledSmem ? GemmTileConfig::k32x32x32_1x1
                                      : GemmTileConfig::k128x128x8_8x8;
}

// Is this (variant, tile) pair actually instantiated in kernels/gemm.cu?
//
// WHY A SEPARATE, PURE-HOST PREDICATE: the pairing is easy to get wrong and the
// failure is badly placed. GemmTile's own defaults are (128,128,8,8,8), but
// kTiledSmem is only instantiated for (32,32,32,1,1) -- so the entirely natural
// `GemmParams{.variant = kTiledSmem}` with a defaulted tile is an
// InvalidArgumentError discovered inside launch_gemm_f32: at run time, on a GPU,
// mid-benchmark. Exposing the check here lets the graph layer reject it at BUILD
// time, naming the node, on a laptop.
[[nodiscard]] constexpr bool gemm_tile_is_supported(GemmVariant v, const GemmTile& t) {
  if (!gemm_variant_uses_tile(v)) return true;
  return tile_is_self_consistent(t) && select_tile_config(t) == gemm_required_config(v);
}

// -----------------------------------------------------------------------------
// Occupancy: the minimum of four independent limiters
//
// "Occupancy" = resident warps / maximum resident warps, per SM. It is a
// LATENCY-HIDING budget, not a speed: more resident warps means more independent
// work for the scheduler to issue while some warp waits on memory. It is NOT the
// metric -- Phase 3d's own table is expected to show tiled_smem at 100%
// occupancy losing badly to tiled_regblock at 50%, because the latter does far
// more arithmetic per byte moved. Occupancy only matters when you are latency-
// bound, and register blocking deliberately trades occupancy for arithmetic
// intensity.
//
// Four caps, and the answer is the smallest:
//   1. registers      -- the register file is finite and allocated per warp
//   2. shared memory  -- per SM, and blocks hold theirs for their whole lifetime
//   3. threads per SM -- a hard architectural cap (1024 on sm_75)
//   4. blocks per SM  -- a separate hard cap (16 on sm_75, 32 on sm_70/sm_80)
// -----------------------------------------------------------------------------
enum class OccupancyLimiter : std::uint8_t {
  kRegisters,
  kSharedMemory,
  kThreadsPerSm,
  kBlocksPerSm,
  kInvalid,        // the configuration cannot launch at all
};

[[nodiscard]] constexpr const char* limiter_name(OccupancyLimiter l) {
  switch (l) {
    case OccupancyLimiter::kRegisters:    return "registers";
    case OccupancyLimiter::kSharedMemory: return "shared memory";
    case OccupancyLimiter::kThreadsPerSm: return "threads/SM cap";
    case OccupancyLimiter::kBlocksPerSm:  return "blocks/SM cap";
    case OccupancyLimiter::kInvalid:      return "WILL NOT LAUNCH";
  }
  return "?";
}

struct OccupancyEstimate {
  int  blocks_per_sm = 0;
  int  warps_per_sm  = 0;
  double occupancy   = 0.0;     // fraction in [0, 1]
  OccupancyLimiter limiter = OccupancyLimiter::kInvalid;

  // True when more than one cap achieves the minimum. Not a footnote: at 1024
  // threads/block on a 1024-thread SM, the threads cap and the register cap
  // BOTH give 1 block, and reporting either alone as "the" limiter invites the
  // wrong fix (cutting registers cannot help, because 2 blocks would need 2048
  // threads no matter what).
  bool tied = false;

  // What each individual cap allowed, so a caller can print the whole picture
  // rather than just the winner. Seeing "registers 2, smem 8, threads 4,
  // blocks 16" is the difference between knowing the answer and understanding it.
  int by_registers = 0, by_smem = 0, by_threads = 0, by_blocks = 0;
};

// Sentinel for "this cap imposes no limit at all" (zero registers, zero shared
// memory). Deliberately NOT max_blocks_per_sm: using that would make an
// unconstrained resource tie with the blocks cap and then win or lose the
// tie-break on an accident of ordering, reporting "shared memory" as the limiter
// of a kernel that uses no shared memory.
inline constexpr int kNoLimit = 1 << 24;

// Architectural allocation granularities. These are NOT the same thing as the
// per-SM totals, and ignoring them is the most common way a hand calculation
// disagrees with cudaOccupancyMaxActiveBlocksPerMultiprocessor -- at which point
// the temptation is to blame the hardware rather than the arithmetic, and the
// whole three-way comparison the roadmap asks for stops being informative.
//
// Registers are allocated PER WARP, rounded up to a 256-register unit, and the
// resulting warp count is then rounded DOWN to a multiple of 4. So the naive
// `regs_per_sm / (regs_per_thread * threads_per_block)` is only accidentally
// right. Worked example where the two disagree, on a T4:
//     R = 17, 512 threads/block (16 warps/block)
//     naive   : floor(65536 / (17*512))                    = 7 blocks
//     granular: regs/warp = ceil(17*32, 256) = 768
//               warps = floor(65536/768) = 85 -> round down to 84
//               blocks = floor(84 / 16)                    = 5 blocks
// (In that particular case the threads/SM cap would bind at 2 anyway, which is
// itself the point: on sm_75 a 256-thread block can never exceed 4 blocks/SM, so
// registers only ever bind once R > 64.)
inline constexpr int kRegAllocUnit        = 256;  // registers, per warp
inline constexpr int kWarpAllocGranularity = 4;   // warps
inline constexpr int kMaxRegsPerThread    = 255;

// Shared memory is also allocated in units. 256 B is the sm_7x/8x figure from
// NVIDIA's occupancy calculator, but it is the constant in this file I am least
// certain of -- so it is named and isolated rather than folded into an
// expression. FALSIFIABLE PREDICTION: if leg 1 (this function) and leg 2
// (cudaOccupancyMaxActiveBlocksPerMultiprocessor) ever disagree on a
// SHARED-MEMORY-BOUND configuration, this constant is the first suspect, and the
// disagreement is data rather than a nuisance.
inline constexpr std::size_t kSmemAllocUnit = 256;

namespace detail {
[[nodiscard]] constexpr int round_up_div(int a, int b) { return (a + b - 1) / b; }
[[nodiscard]] constexpr int round_up(int a, int unit) { return round_up_div(a, unit) * unit; }
[[nodiscard]] constexpr int round_down(int a, int unit) { return (a / unit) * unit; }
[[nodiscard]] constexpr std::size_t round_up_sz(std::size_t a, std::size_t unit) {
  return ((a + unit - 1) / unit) * unit;
}
[[nodiscard]] constexpr int min4(int a, int b, int c, int d) {
  const int ab = a < b ? a : b;
  const int cd = c < d ? c : d;
  return ab < cd ? ab : cd;
}
}  // namespace detail

// The hand calculation. `regs_per_thread` and `smem_bytes_per_block` come from
// cudaFuncGetAttributes on the SAME kernel that was timed -- not from source
// constants -- because a register count is a property of what ptxas emitted, and
// guessing it is how you end up confidently reporting an occupancy for a kernel
// that never had that register count.
[[nodiscard]] constexpr OccupancyEstimate occupancy_blocks_per_sm(
    const DeviceInfo& d, int threads_per_blk, int regs_per_thread,
    std::size_t smem_bytes_per_block) {
  OccupancyEstimate e;

  // Degenerate inputs return "will not launch" rather than dividing by zero. A
  // default-constructed DeviceInfo has every field at 0, and host-only builds
  // cannot produce a real one, so tests hand-construct -- which means this path
  // is reachable in normal use, not just in adversarial use.
  if (threads_per_blk <= 0 || d.sm_count <= 0 || d.max_threads_per_sm <= 0 ||
      d.regs_per_sm <= 0 || d.max_blocks_per_sm <= 0 || d.shared_mem_per_sm == 0)
    return e;

  // Hard launch failures -- these are `cudaErrorLaunchOutOfResources` at launch
  // time, not slow kernels. Reporting 0 is the honest answer.
  if (d.max_threads_per_block > 0 && threads_per_blk > d.max_threads_per_block) return e;
  if (regs_per_thread > kMaxRegsPerThread) return e;
  if (smem_bytes_per_block > d.shared_mem_per_sm) return e;
  if (d.regs_per_block > 0 && regs_per_thread * threads_per_blk > d.regs_per_block) return e;

  const int warps_per_blk = detail::round_up_div(threads_per_blk, kWarpSize);

  // 1. Registers, with per-warp allocation granularity (see above).
  if (regs_per_thread <= 0) {
    e.by_registers = kNoLimit;              // no register pressure => not binding
  } else {
    const int regs_per_warp = detail::round_up(regs_per_thread * kWarpSize, kRegAllocUnit);
    const int warps_by_regs = detail::round_down(d.regs_per_sm / regs_per_warp,
                                                 kWarpAllocGranularity);
    e.by_registers = warps_by_regs / warps_per_blk;
  }

  // 2. Shared memory.
  if (smem_bytes_per_block == 0) {
    e.by_smem = kNoLimit;
  } else {
    const std::size_t per_block = detail::round_up_sz(smem_bytes_per_block, kSmemAllocUnit);
    e.by_smem = static_cast<int>(d.shared_mem_per_sm / per_block);
  }

  // 3. The threads-per-SM cap.  4. The blocks-per-SM cap.
  e.by_threads = d.max_threads_per_sm / threads_per_blk;
  e.by_blocks  = d.max_blocks_per_sm;

  // The answer is the minimum. TIES ARE BROKEN TOWARD THE ARCHITECTURAL CAPS
  // (threads/SM, then blocks/SM) rather than the resource caps, because the
  // architectural ones are the ones you cannot buy your way out of. If registers
  // and the threads cap both allow 1 block, "registers" is a true but useless
  // answer -- it invites cutting register pressure, which cannot possibly help.
  // `tied` is set whenever the choice was not unique, so a caller never has to
  // take the single reported limiter as the whole story.
  const int best = detail::min4(e.by_registers, e.by_smem, e.by_threads, e.by_blocks);
  if (best <= 0) return e;                  // kInvalid, blocks_per_sm = 0

  OccupancyLimiter which = OccupancyLimiter::kRegisters;
  if      (e.by_threads   == best) which = OccupancyLimiter::kThreadsPerSm;
  else if (e.by_blocks    == best) which = OccupancyLimiter::kBlocksPerSm;
  else if (e.by_registers == best) which = OccupancyLimiter::kRegisters;
  else                             which = OccupancyLimiter::kSharedMemory;

  e.tied = ((e.by_registers == best) + (e.by_smem    == best) +
            (e.by_threads   == best) + (e.by_blocks  == best)) > 1;

  e.blocks_per_sm = best;
  e.limiter       = which;
  e.warps_per_sm  = best * warps_per_blk;
  e.occupancy     = static_cast<double>(best * threads_per_blk) /
                    static_cast<double>(d.max_threads_per_sm);
  return e;
}

}  // namespace mcke::kernels
