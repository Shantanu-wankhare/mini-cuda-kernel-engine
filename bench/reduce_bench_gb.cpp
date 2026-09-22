// =============================================================================
//  bench/reduce_bench_gb.cpp
//
//  WHAT: Google Benchmark pilot for the row-reduce kernel (Phase 3b), using
//        bench/gb_adapter.hpp. NOT a replacement for bench/reduce_bench.cpp,
//        which stays the source of RESULTS.md section 3b -- see that file's
//        banner for the shapes and why the "starved" one exists.
//
//  WHY THIS KERNEL WAS PICKED AS A PILOT (over gemm/graph): it and
//  bias_act_bench_gb.cpp are the two simplest fixtures, and the "starved" shape
//  here (64 x 524288) is a genuinely SHORT kernel. Short kernels are exactly
//  where kPerIter's per-iteration host sync (see gb_adapter.hpp) costs the most
//  as a fraction of the measurement, which is what makes the stage 5g
//  kBurst-vs-kPerIter cross-check informative here rather than academic.
//
//  Deliberately NOT gemm_bench or graph_bench: their bespoke tables (the wave
//  sweep, fork/join event counts, the enqueue_us column) ARE the Phase 3d/4
//  deliverables, and Google Benchmark's row-per-benchmark output format cannot
//  express them.
//
//  WHY .cpp: no __global__, no <<<>>>; same boundary as reduce_bench.cpp.
// =============================================================================
#include <benchmark/benchmark.h>

#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gb_adapter.hpp"

#include "mcke/core/device.hpp"
#include "mcke/kernels/kernels.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/cuda_check.hpp"
#include "mcke/runtime/stream.hpp"

#include "bench_common.hpp"
#include "reference.hpp"

using namespace mcke;
namespace K = mcke::kernels;

namespace {

// Same two shapes as reduce_bench.cpp, same N, for the same reason: only the
// row/column split changes, so any GB-vs-time_op or kBurst-vs-kPerIter
// difference observed here isolates the TIMING METHOD, not a different kernel.
constexpr std::int64_t kRowsA = 8192, kColsA = 4096;      // saturated
constexpr std::int64_t kRowsB = 64,   kColsB = 524288;    // starved -- the short one
constexpr int kWarmup = 5;
constexpr int kInnerIters = 20;   // kBurst's inner Profiler::time_op iters

std::uint64_t ideal_bytes(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(rows * cols + rows) * sizeof(float);
}
std::uint64_t ideal_flops(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(rows * (cols - 1));
}

// -----------------------------------------------------------------------------
// Fixture. GB registers benchmarks at STATIC-INIT time (the BENCHMARK_CAPTURE
// macros below run before main()), but device setup (query, allocate, upload)
// can only happen inside main() -- there is no device to query at static-init
// time, and constructing it there would run before argv is even parsed. So
// this is a plain global pointer, set once in main() before
// RunSpecifiedBenchmarks(), and read (never written) by every registered
// benchmark function. Not a GB Fixture class: this project's other benches use
// plain functions + captured state, and matching that style keeps
// gb_adapter.hpp usable without also adopting GB's class-based API.
// -----------------------------------------------------------------------------
struct Fixture {
  rt::Stream stream;
  RawDeviceAllocator alloc;
  Allocation dx, dout, dws;
  Roofline rl;
};
std::unique_ptr<Fixture> g_fx;

// -----------------------------------------------------------------------------
// kBurst: one GB iteration == one full Profiler::time_op burst. This is what
// goes in RESULTS.md; see gb_adapter.hpp's banner for why its physics are
// identical to reduce_bench.cpp's own numbers.
// -----------------------------------------------------------------------------
void bm_reduce_burst(benchmark::State& state, std::int64_t rows, std::int64_t cols,
                    const char* shape_tag) {
  auto& fx = *g_fx;
  const std::string variant = std::string("warp_shuffle_256t") + shape_tag;
  KernelRecord last{};
  mcke::benchgb::run_burst(
      state, "row_reduce_sum", variant, fx.stream, fx.rl,
      ideal_flops(rows, cols), ideal_bytes(rows, cols), kWarmup, kInnerIters,
      &last, [&](const rt::Stream& s) {
        return K::launch_row_reduce_f32(
            static_cast<const float*>(fx.dx.ptr), static_cast<float*>(fx.dout.ptr),
            rows, cols, K::ReduceKind::kSum, K::ReduceVariant::kWarpShuffle,
            static_cast<float*>(fx.dws.ptr), fx.dws.bytes, s.native());
      });
}

// -----------------------------------------------------------------------------
// kPerIter: GB's native model, one host sync per launch. NEVER a RESULTS.md
// number -- this exists to feed the stage 5g cross-check, which measures what
// that sync costs as a function of kernel duration (this file's starved shape
// vs bias_act_bench_gb.cpp's L2-resident shape span roughly two orders of
// magnitude in duration).
// -----------------------------------------------------------------------------
void bm_reduce_periter(benchmark::State& state, std::int64_t rows, std::int64_t cols,
                      const char* shape_tag) {
  auto& fx = *g_fx;
  const std::string variant = std::string("warp_shuffle_256t") + shape_tag;
  mcke::benchgb::run_per_iteration(
      state, variant, fx.stream, fx.rl, ideal_flops(rows, cols),
      ideal_bytes(rows, cols), kWarmup, [&](const rt::Stream& s) {
        return K::launch_row_reduce_f32(
            static_cast<const float*>(fx.dx.ptr), static_cast<float*>(fx.dout.ptr),
            rows, cols, K::ReduceKind::kSum, K::ReduceVariant::kWarpShuffle,
            static_cast<float*>(fx.dws.ptr), fx.dws.bytes, s.native());
      });
}

MCKE_GB_GPU(bm_reduce_burst, saturated, kRowsA, kColsA, "")->Iterations(1)->Repetitions(10);
MCKE_GB_GPU(bm_reduce_burst, starved, kRowsB, kColsB, "_starved")->Iterations(1)->Repetitions(10);
MCKE_GB_GPU(bm_reduce_periter, saturated, kRowsA, kColsA, "")->Iterations(20);
MCKE_GB_GPU(bm_reduce_periter, starved, kRowsB, kColsB, "_starved")->Iterations(20);

}  // namespace

int main(int argc, char** argv) {
  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to benchmark\n");
    return 0;
  }

  // See gb_adapter.hpp's snapshot_cmdline comment: MUST happen before
  // benchmark::Initialize deletes the --benchmark_* flags from argv.
  const std::string cmdline = mcke::benchgb::snapshot_cmdline(argc, argv);
  benchmark::Initialize(&argc, argv);

  auto dev = query_device(0);
  dev.status().throw_if_error();
  set_device(0).throw_if_error();
  auto stream = rt::Stream::create();
  stream.status().throw_if_error();

  const Roofline rl = benchcfg::make_roofline(argc, argv);
  benchcfg::print_denominators(rl, *dev);

  RawDeviceAllocator alloc;
  const std::int64_t n = kRowsA * kColsA;   // same N as reduce_bench.cpp
  const std::size_t xbytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t obytes = static_cast<std::size_t>(kRowsA) * sizeof(float);

  auto grab = [&](std::size_t b) {
    auto r = alloc.allocate(b, stream->native());
    r.status().throw_if_error();
    return *r;
  };
  Allocation dx  = grab(xbytes);
  Allocation dout = grab(obytes);
  const std::size_t wsA = K::row_reduce_workspace_bytes(kRowsA, kColsA, K::ReduceVariant::kTwoPass);
  const std::size_t wsB = K::row_reduce_workspace_bytes(kRowsB, kColsB, K::ReduceVariant::kTwoPass);
  Allocation dws = grab(wsA > wsB ? wsA : wsB);

  std::vector<float> hx(static_cast<std::size_t>(n));
  testing::fill_random(hx.data(), hx.size(), 0x5EED12345ull, -1.0f, 1.0f);
  MCKE_CUDA_CHECK(cudaMemcpyAsync(dx.ptr, hx.data(), xbytes,
                                  cudaMemcpyHostToDevice, stream->native()));
  stream->synchronize().throw_if_error();

  // ---------------------------------------------------------------------------
  // Correctness, once, before any GB benchmark runs. GB has no notion of "verify
  // once then time" -- if this ran inside a registered benchmark it would
  // memcpy-to-host and compare on EVERY iteration, which is not what
  // reduce_bench.cpp does and would make the two files' timings incomparable.
  // ---------------------------------------------------------------------------
  {
    std::vector<float> hout(static_cast<std::size_t>(kRowsA));
    std::vector<float> href(static_cast<std::size_t>(kRowsA));
    const Status st = K::launch_row_reduce_f32(
        static_cast<const float*>(dx.ptr), static_cast<float*>(dout.ptr),
        kRowsA, kColsA, K::ReduceKind::kSum, K::ReduceVariant::kWarpShuffle,
        static_cast<float*>(dws.ptr), dws.bytes, stream->native());
    st.throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(hout.data(), dout.ptr, obytes, cudaMemcpyDeviceToHost));
    testing::reference_row_reduce(hx.data(), href.data(), kRowsA, kColsA, K::ReduceKind::kSum);
    const auto r = testing::compare(hout.data(), href.data(), hout.size(),
                                    testing::kTolReduce4096, testing::kAbsTolReduceSum4096);
    benchcfg::print_validation("warp_shuffle/sum (pre-GB)", r.ok(), r.max_rel_err,
                               testing::kTolReduce4096, "sqrt(4096)*f32 eps");
    if (!r.ok()) {
      std::fprintf(stderr, "*** VALIDATION FAILURE -- refusing to run GB benchmarks ***\n");
      return 1;
    }
  }

  g_fx = std::make_unique<Fixture>(Fixture{std::move(*stream), std::move(alloc), dx, dout, dws, rl});

  benchmark::AddCustomContext("mcke_cmdline", cmdline);
  benchmark::AddCustomContext("mcke_timing",
                              "kBurst rows: Profiler::time_op (RESULTS.md-comparable). "
                              "kPerIter rows: GB native, host sync per iteration -- "
                              "NOT comparable, see gb_adapter.hpp.");
  benchmark::AddCustomContext("mcke_peak_gb_s", std::to_string(rl.peak_gb_s));
  benchmark::AddCustomContext("mcke_peak_tflops", std::to_string(rl.peak_tflops));

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  g_fx->alloc.deallocate(g_fx->dx, g_fx->stream.native()).throw_if_error();
  g_fx->alloc.deallocate(g_fx->dout, g_fx->stream.native()).throw_if_error();
  g_fx->alloc.deallocate(g_fx->dws, g_fx->stream.native()).throw_if_error();
  g_fx.reset();
  return 0;
}
