// =============================================================================
//  bench/fma_peak.cu
//
//  WHAT: Measures *sustained achievable* f32 FMA throughput with (almost) zero
//        memory traffic — the compute-side counterpart to stream_triad.cu.
//
//  WHY THIS EXISTS — it produces the "measured FMA peak" denominator for every
//  "% of peak" figure on compute-bound kernels (the GEMM variants in Phase 3).
//  See `theoretical_peak_f32_tflops()` in src/core/profiler.cpp: it deliberately
//  returns 0 rather than computing SMs * cores/SM * 2 * clock, because
//  cudaDeviceProp's clockRate is the BASE boost clock, not what the chip
//  actually sustains under a real FMA load (which depends on power/thermal
//  limits a spec sheet doesn't capture). Measuring it directly is both easier
//  to justify and more honest — it already reflects this GPU's real clock
//  under load, on this day, in this box.
//
//  DESIGN — keeping the GPU compute-bound, not latency-bound:
//  A single `fmaf` has a pipeline latency of a few cycles. A thread that issues
//  `x = fmaf(x, m, b)` in one dependent chain must WAIT for each result before
//  issuing the next — the pipeline sits mostly idle behind a chain of length 1.
//  This is Instruction-Level Parallelism (ILP), a DIFFERENT axis from the
//  warp-level parallelism that occupancy measures: occupancy tells you how many
//  warps can hide EACH OTHER's stalls by swapping in when one blocks; ILP tells
//  you how much independent work exists WITHIN one thread to hide its OWN
//  instruction latency, with no warp switch required at all. We give each
//  thread `kAccumulators` independent chains so the compiler can interleave
//  their FMAs and keep the pipeline full regardless of how many warps the
//  scheduler can actually keep resident.
//
//  WHY THE RECURRENCE IS A CONTRACTION (|m| < 1), NOT m > 1:
//  `acc = acc*m + b` with |m| < 1 converges to the fixed point `b/(1-m)` and
//  STAYS bounded there for any iteration count — no risk of overflow to inf, and
//  (just as important) no risk of decaying into a subnormal, since denormal FMA
//  results run measurably slower on some architectures and would silently
//  corrupt the very throughput number we're trying to measure. With m=0.999,
//  b=1.0, the fixed point is 1000 — comfortably inside normal float range for
//  every one of the 100,000 iterations.
//
//  WHY THE SEED COMES FROM A DEVICE ARRAY, NOT A LITERAL:
//  If every accumulator started from the same compile-time constant, a
//  sufficiently aggressive compiler is entitled to prove the loop's final value
//  is itself a compile-time constant and fold the whole thing away. Seeding
//  from `seed[tid]` — unknown to the compiler until runtime — keeps the
//  computation genuine. The final `out[tid] = sum` write is the other half of
//  this: without any observable use of the result, the compiler is free to
//  delete the entire kernel body as dead code.
// =============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mcke/core/device.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/cuda_check.hpp"
#include "mcke/runtime/stream.hpp"

using namespace mcke;

namespace {

constexpr int kAccumulators   = 8;       // independent FMA chains per thread (the ILP width)
constexpr int kItersPerLaunch = 100000;  // trip count; keeps one launch well above launch overhead

__global__ void fma_peak_kernel(const float* __restrict__ seed, float* __restrict__ out) {
  const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;

  float acc[kAccumulators];
  const float s = seed[tid];
  // kAccumulators is a small compile-time constant: fully unrolling this init
  // loop costs nothing and lets every acc[a] live in its own register instead
  // of becoming a local-memory (i.e. DRAM-backed) array — the failure mode
  // most likely to quietly wreck a "peak FLOPS" kernel.
  #pragma unroll
  for (int a = 0; a < kAccumulators; ++a) acc[a] = s + static_cast<float>(a);

  const float m = 0.999f;   // contraction factor, |m| < 1 — see header comment
  const float b = 1.0f;

  // Deliberately NOT unrolled: with 8 independent chains already exposing
  // plenty of ILP, unrolling this loop would mainly trade code size and
  // compile time for a marginal cut in loop-branch overhead that is already
  // tiny next to 8 FMAs per iteration. Simpler is the right call here.
  for (int i = 0; i < kItersPerLaunch; ++i) {
    #pragma unroll
    for (int a = 0; a < kAccumulators; ++a) {
      acc[a] = fmaf(acc[a], m, b);
    }
  }

  float sum = 0.f;
  #pragma unroll
  for (int a = 0; a < kAccumulators; ++a) sum += acc[a];
  out[tid] = sum;   // the observable side effect that keeps the whole kernel alive
}

// -----------------------------------------------------------------------------
// --warmup= / --iters=, added in Phase 5 stage 5f (DECISIONS.md Q12).
//
// The DEFAULTS STAY 3 + 10 -- below RESULTS.md rule 3's >= 5 warmup / >= 20
// timed -- and that is deliberate, not an oversight left in place. This program
// produces the DENOMINATOR for every compute-bound "% of peak" in the project
// (8.130 TFLOP/s on the T4, 15.601 on the V100). Changing its iteration count
// blind would silently move every one of those percentages at once. The flags
// exist so stage 5g can measure BOTH ways on the same machine and record whether
// the denominator actually moves before anything is changed. Unknown flags are
// fatal, matching gemm_bench/graph_bench: a typo'd flag that silently falls back
// to the defaults would make the A/B compare a run against itself.
// -----------------------------------------------------------------------------
struct Args { int warmup = 3; int iters = 10; };

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const char* s = argv[i];
    if      (std::strncmp(s, "--warmup=", 9) == 0) a.warmup = std::atoi(s + 9);
    else if (std::strncmp(s, "--iters=", 8) == 0)  a.iters  = std::atoi(s + 8);
    else {
      std::fprintf(stderr, "fma_peak: unknown argument '%s'\n"
                           "usage: mcke_fma_peak [--warmup=N] [--iters=N]  (defaults 3, 10)\n", s);
      std::exit(2);
    }
  }
  if (a.warmup < 0 || a.iters <= 0) {
    std::fprintf(stderr, "fma_peak: need --warmup >= 0 and --iters > 0\n");
    std::exit(2);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to benchmark\n");
    return 0;
  }
  auto dev = query_device(0);
  dev.status().throw_if_error();
  set_device(0).throw_if_error();

  auto stream = rt::Stream::create();
  stream.status().throw_if_error();

  // Grid size: enough blocks to fill every SM many times over, REGARDLESS of
  // what occupancy this particular kernel achieves. We don't know in advance
  // how many of these register-heavy blocks (8 live accumulators + loop state)
  // will be co-resident per SM, so rather than compute it, we launch far more
  // blocks than SMs and let the hardware scheduler queue the remainder — the
  // same "oversubscribe and let the scheduler sort it out" idea as the
  // grid-stride kernels, applied to blocks instead of elements.
  constexpr int kThreads = 256;
  const int blocks = dev->sm_count > 0 ? dev->sm_count * 32 : 2048;
  const std::size_t total_threads = static_cast<std::size_t>(blocks) * kThreads;

  RawDeviceAllocator alloc;
  auto d_seed = alloc.allocate(total_threads * sizeof(float), stream->native());
  auto d_out  = alloc.allocate(total_threads * sizeof(float), stream->native());
  d_seed.status().throw_if_error();
  d_out.status().throw_if_error();

  std::vector<float> h_seed(total_threads);
  for (std::size_t i = 0; i < total_threads; ++i) h_seed[i] = static_cast<float>(i % 13);
  MCKE_CUDA_CHECK(cudaMemcpyAsync(d_seed->ptr, h_seed.data(), total_threads * sizeof(float),
                                 cudaMemcpyHostToDevice, stream->native()));

  Profiler prof;
  const std::uint64_t total_fma =
      static_cast<std::uint64_t>(total_threads) * static_cast<std::uint64_t>(kAccumulators) *
      static_cast<std::uint64_t>(kItersPerLaunch);
  const std::uint64_t ideal_flops = total_fma * 2;  // FMA = 1 mul + 1 add
  // seed read + out write: negligible next to the FLOP count above -- the
  // resulting AI is enormous, which IS the point. This kernel is designed to
  // sit as far right on the roofline as physically possible, so its own
  // reported AI/efficiency numbers are not meaningful; only rec->tflops() is.
  const std::uint64_t ideal_bytes = 2ull * total_threads * sizeof(float);

  auto rec = prof.time_op(
      "fma_peak", "8acc_100000iter", *stream, ideal_flops, ideal_bytes,
      args.warmup, args.iters,
      [&](const rt::Stream& s) {
        fma_peak_kernel<<<blocks, kThreads, 0, s.native()>>>(
            static_cast<const float*>(d_seed->ptr), static_cast<float*>(d_out->ptr));
        MCKE_CUDA_RETURN_IF_ERROR(cudaGetLastError());
        return OkStatus();
      });
  rec.status().throw_if_error();
  stream->synchronize().throw_if_error();

  std::printf("\nblocks=%d threads/block=%d total_threads=%zu\n", blocks, kThreads, total_threads);
  // Echoed so a log records which iteration config produced its number --
  // without this line the 5g A/B's two logs would be indistinguishable.
  std::printf("timing        %d warmup + %d timed iterations%s\n", args.warmup, args.iters,
              (args.warmup < 5 || args.iters < 20) ? "  (below RESULTS.md rule 3 -- the historical default)" : "");
  std::printf(
      "\nmeasured f32 FMA peak: %.3f TFLOP/s\n\n"
      "*** Use this number as Roofline::peak_tflops for every compute-bound\n"
      "*** kernel benchmark from Phase 3 onward (docs/PROFILING.md sec 2).\n"
      "*** Do NOT substitute a spec-sheet FLOPS figure -- see the header comment\n"
      "*** in this file and theoretical_peak_f32_tflops() in src/core/profiler.cpp\n"
      "*** for why the spec number would misrepresent this specific chip's\n"
      "*** actual sustained clock.\n",
      rec->tflops());

  alloc.deallocate(*d_seed, stream->native()).throw_if_error();
  alloc.deallocate(*d_out, stream->native()).throw_if_error();
  return 0;
}
