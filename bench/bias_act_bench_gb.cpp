// =============================================================================
//  bench/bias_act_bench_gb.cpp
//
//  WHAT: Google Benchmark pilot for the fused bias+activation kernel (Phase
//        3a), using bench/gb_adapter.hpp. NOT a replacement for
//        bench/bias_act_bench.cpp, which stays the source of RESULTS.md
//        section 3a -- see that file's banner for the full sweep and the two
//        controlled experiments.
//
//  WHY THIS SHAPE, SPECIFICALLY THE L2-RESIDENT ONE (512x512): it is the
//  shortest kernel in the whole Phase-3 ladder -- the working set fits in the
//  T4's 4 MiB L2, so the launch itself is a large fraction of the measured
//  time. That makes it the single best case for the stage 5g cross-check: if
//  kPerIter's per-iteration host sync (see gb_adapter.hpp) is going to visibly
//  distort a number, it will distort THIS one hardest, in contrast to
//  reduce_bench_gb.cpp's starved shape, which is short but for a different
//  reason (few blocks, not a small working set).
//
//  Also registers the primary 8192x4096 shape at full occupancy, so the
//  cross-check spans roughly three orders of magnitude in kernel duration
//  between the two pilot files.
//
//  WHY .cpp: no __global__, no <<<>>>; same boundary as bias_act_bench.cpp.
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

constexpr int kWarmup = 5;
constexpr int kInnerIters = 20;

std::uint64_t fused_bytes(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(2 * rows * cols + cols) * sizeof(float);
}
// Only kGeluTanh is exercised here (10 FLOPs/element -- see
// bias_act_bench.cpp's act_flops for the derivation); the full activation
// sweep stays in the CSV-producing bench.
std::uint64_t gelu_tanh_flops(std::int64_t n) { return static_cast<std::uint64_t>(n) * 10; }

// See reduce_bench_gb.cpp's identical comment: GB registers benchmarks at
// static-init time, before main() can query a device, so this is a plain
// global set once in main().
struct Fixture {
  rt::Stream stream;
  RawDeviceAllocator alloc;
  Allocation x, bias, y;
  Roofline rl;
};
std::unique_ptr<Fixture> g_fx;

void bm_bias_act_burst(benchmark::State& state, std::int64_t rows, std::int64_t cols,
                      const char* shape_tag) {
  auto& fx = *g_fx;
  const std::string variant = std::string("fused_vw4") + shape_tag;
  KernelRecord last{};
  mcke::benchgb::run_burst(
      state, "bias_gelu_tanh", variant, fx.stream, fx.rl,
      gelu_tanh_flops(rows * cols), fused_bytes(rows, cols), kWarmup, kInnerIters,
      &last, [&](const rt::Stream& s) {
        return K::launch_bias_act_f32(
            static_cast<const float*>(fx.x.ptr), static_cast<const float*>(fx.bias.ptr),
            static_cast<float*>(fx.y.ptr), rows, cols, K::Activation::kGeluTanh, 4,
            s.native());
      });
}

// kPerIter: never a RESULTS.md number. See gb_adapter.hpp's banner and this
// file's own header comment on why the 512x512 shape is the sharpest test of
// what the per-iteration host sync costs.
void bm_bias_act_periter(benchmark::State& state, std::int64_t rows, std::int64_t cols,
                        const char* shape_tag) {
  auto& fx = *g_fx;
  const std::string variant = std::string("fused_vw4") + shape_tag;
  mcke::benchgb::run_per_iteration(
      state, variant, fx.stream, fx.rl, gelu_tanh_flops(rows * cols),
      fused_bytes(rows, cols), kWarmup, [&](const rt::Stream& s) {
        return K::launch_bias_act_f32(
            static_cast<const float*>(fx.x.ptr), static_cast<const float*>(fx.bias.ptr),
            static_cast<float*>(fx.y.ptr), rows, cols, K::Activation::kGeluTanh, 4,
            s.native());
      });
}

MCKE_GB_GPU(bm_bias_act_burst, full, 8192, 4096, "")->Iterations(1)->Repetitions(10);
MCKE_GB_GPU(bm_bias_act_burst, l2_resident, 512, 512, "_512x512")->Iterations(1)->Repetitions(10);
MCKE_GB_GPU(bm_bias_act_periter, full, 8192, 4096, "")->Iterations(20);
MCKE_GB_GPU(bm_bias_act_periter, l2_resident, 512, 512, "_512x512")->Iterations(20);

}  // namespace

int main(int argc, char** argv) {
  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to benchmark\n");
    return 0;
  }

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
  // Sized for the LARGER shape (8192x4096); the 512x512 runs reuse the same
  // buffers with a smaller rows/cols passed to the launcher, exactly as
  // bias_act_bench.cpp does not do (it allocates per-shape) -- the difference
  // is deliberate here: this file exists to exercise the adapter and the
  // cross-check, not to reproduce bias_act_bench.cpp's own allocation pattern.
  const std::int64_t n = 8192 * 4096;
  const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(float);

  auto grab = [&](std::size_t b) {
    auto r = alloc.allocate(b, stream->native());
    r.status().throw_if_error();
    return *r;
  };
  Allocation x = grab(nbytes);
  Allocation y = grab(nbytes);
  Allocation bias = grab(static_cast<std::size_t>(4096) * sizeof(float));

  std::vector<float> hx(static_cast<std::size_t>(n));
  std::vector<float> hbias(4096);
  testing::fill_random(hx.data(), hx.size(), 0xB1A5AC7ull, -3.0f, 3.0f);
  testing::fill_random(hbias.data(), hbias.size(), 0xB1A5B1A5ull, -1.0f, 1.0f);
  MCKE_CUDA_CHECK(cudaMemcpyAsync(x.ptr, hx.data(), nbytes, cudaMemcpyHostToDevice, stream->native()));
  MCKE_CUDA_CHECK(cudaMemcpyAsync(bias.ptr, hbias.data(), hbias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream->native()));
  stream->synchronize().throw_if_error();

  // Correctness once, at the LARGER shape, before any GB benchmark runs -- same
  // reasoning as reduce_bench_gb.cpp: GB has no "verify once, then time" notion,
  // and verifying inside a registered benchmark would make every iteration pay
  // a device->host copy, which is not what bias_act_bench.cpp measures.
  {
    std::vector<float> hy(static_cast<std::size_t>(n));
    std::vector<float> href(static_cast<std::size_t>(n));
    const Status st = K::launch_bias_act_f32(
        static_cast<const float*>(x.ptr), static_cast<const float*>(bias.ptr),
        static_cast<float*>(y.ptr), 8192, 4096, K::Activation::kGeluTanh, 4, stream->native());
    st.throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(hy.data(), y.ptr, nbytes, cudaMemcpyDeviceToHost));
    testing::reference_bias_act(hx.data(), hbias.data(), href.data(), 8192, 4096,
                                K::Activation::kGeluTanh);
    const auto r = testing::compare(hy.data(), href.data(), hy.size(),
                                    testing::kTolElementwise, testing::kAbsTolGeluCancellation);
    benchcfg::print_validation("fused/gelu_tanh/vw4 (pre-GB)", r.ok(), r.max_rel_err,
                               testing::kTolElementwise, "device tanhf vs host std::tanh");
    if (!r.ok()) {
      std::fprintf(stderr, "*** VALIDATION FAILURE -- refusing to run GB benchmarks ***\n");
      return 1;
    }
  }

  g_fx = std::make_unique<Fixture>(Fixture{std::move(*stream), std::move(alloc), x, bias, y, rl});

  benchmark::AddCustomContext("mcke_cmdline", cmdline);
  benchmark::AddCustomContext("mcke_timing",
                              "kBurst rows: Profiler::time_op (RESULTS.md-comparable). "
                              "kPerIter rows: GB native, host sync per iteration -- "
                              "NOT comparable, see gb_adapter.hpp.");
  benchmark::AddCustomContext("mcke_peak_gb_s", std::to_string(rl.peak_gb_s));
  benchmark::AddCustomContext("mcke_peak_tflops", std::to_string(rl.peak_tflops));

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  g_fx->alloc.deallocate(g_fx->x, g_fx->stream.native()).throw_if_error();
  g_fx->alloc.deallocate(g_fx->y, g_fx->stream.native()).throw_if_error();
  g_fx->alloc.deallocate(g_fx->bias, g_fx->stream.native()).throw_if_error();
  g_fx.reset();
  return 0;
}
