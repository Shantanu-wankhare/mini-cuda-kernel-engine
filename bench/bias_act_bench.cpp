// =============================================================================
//  bench/bias_act_bench.cpp
//
//  WHAT: Phase 3a. Measures fused `y = act(x + bias)` against the unfused pair,
//        sweeps the activation and the vector width, and runs two controlled
//        experiments whose job is to make the primary results interpretable.
//
//  WHY .cpp AND NOT .cu: it contains no __global__ and no <<<>>>. It calls
//  `launch_bias_act_f32`, an ordinary C++ function whose body lives in
//  kernels/bias_act.cu. That is the boundary kernels.hpp exists to define, and
//  tools/smoke_vector_add.cpp is the working precedent. Keeping the bench off
//  nvcc also keeps the edit-rebuild loop fast on Colab.
//
//  It also matters for a reason that is easy to miss: `-Xptxas=-v` is applied
//  only to the mcke_kernels library target. Any __global__ defined HERE would
//  have invisible regs/thread and smem/block, so the occupancy columns of
//  RESULTS.md would silently have a hole in them. Every kernel in Phase 3,
//  including the unfused baseline, lives in kernels/*.cu for that reason.
//
//  ---------------------------------------------------------------------------
//  CORRECTNESS BEFORE PERFORMANCE
//
//  Every configuration is validated against the CPU reference in
//  tests/reference.hpp *at the benchmark shape* before it is timed. Not at a
//  smaller proxy shape -- at the exact shape that produces the published number,
//  because RESULTS.md's standard is that every number must be explicable later,
//  and "verified correct at that shape, max rel err 3e-7" is part of the
//  explanation. A fast wrong kernel is worthless.
//
//  ---------------------------------------------------------------------------
//  THE TWO CONTROLLED EXPERIMENTS, AND WHY THEY ARE NOT PADDING
//
//  1. L2-RESIDENT CONTROL (512x512). The headline claim is "fusion is ~2x
//     because traffic halves". That claim is only true when the unfused pair's
//     intermediate actually goes to DRAM. At 512x512 the whole working set fits
//     in the T4's 4 MiB L2, so the second kernel reads `tmp` from cache and the
//     speedup should COLLAPSE to ~1.0-1.2x. Predicting that in advance and
//     measuring it turns "cache effects" from a post-hoc excuse into a result.
//
//  2. OCCUPANCY-STARVED WIDTH SWEEP. See kernels/bias_act.cu's banner: the width
//     sweep is predicted to be flat at full occupancy because instruction issue
//     is ~138x from being the limiter. Flat on its own reads as "vectorisation
//     is pointless", which is the wrong lesson. Re-run starved (~40 blocks, one
//     per SM) and the win should appear. Two rows, one rule.
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

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

// Primary shape, identical to the reduction and softmax benches so the three
// memory-bound kernels are directly comparable. 256 MiB per array is 64x the
// T4's 4 MiB L2, so nothing caches -- which is exactly what makes it the right
// shape for measuring DRAM traffic, and exactly why the L2 control needs to be
// a separate, much smaller shape.
constexpr std::int64_t kRows = 8192;
constexpr std::int64_t kCols = 4096;

constexpr int kWarmup = 5;
constexpr int kIters  = 20;

const char* act_name(K::Activation a) {
  switch (a) {
    case K::Activation::kNone:     return "none";
    case K::Activation::kRelu:     return "relu";
    case K::Activation::kGeluErf:  return "gelu_erf";
    case K::Activation::kGeluTanh: return "gelu_tanh";
  }
  return "?";
}

// Ideal (compulsory) traffic. Stated as named constants with the derivation, per
// RESULTS.md rule 4 -- anyone must be able to recompute the GB/s from these.
std::uint64_t fused_bytes(std::int64_t rows, std::int64_t cols) {
  // read x once + write y once + read the bias vector once.
  return static_cast<std::uint64_t>(2 * rows * cols + cols) * sizeof(float);
}
std::uint64_t unfused_bytes(std::int64_t rows, std::int64_t cols) {
  // k1: read x, read bias, write tmp.  k2: read tmp, write y.
  return static_cast<std::uint64_t>(4 * rows * cols + cols) * sizeof(float);
}

// FLOPs per element, so arithmetic_intensity() and memory_bound() are truthful.
// All of these are 12x to 400x below the 34.2 ridge point, which is why this
// bench reports GB/s and never TFLOP/s.
std::uint64_t act_flops(K::Activation a, std::int64_t n) {
  switch (a) {
    case K::Activation::kNone:     return static_cast<std::uint64_t>(n) * 1;   // the bias add
    case K::Activation::kRelu:     return static_cast<std::uint64_t>(n) * 2;   // add + max
    case K::Activation::kGeluTanh: return static_cast<std::uint64_t>(n) * 10;
    case K::Activation::kGeluErf:  return static_cast<std::uint64_t>(n) * 22;
  }
  return 0;
}

struct Buffers {
  Allocation x, bias, y, tmp;
  std::vector<float> hx, hbias, hy, href;
};

}  // namespace

int main(int argc, char** argv) {
  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to benchmark\n");
    return 0;
  }
  auto dev = query_device(0);
  dev.status().throw_if_error();
  set_device(0).throw_if_error();
  auto stream = rt::Stream::create();
  stream.status().throw_if_error();

  // Sets BOTH roofline fields or aborts -- see bench_common.hpp for why leaving
  // peak_tflops at 0 silently zeroes the %peak column and mislabels every
  // kernel as compute-bound.
  const Roofline rl = benchcfg::make_roofline(argc, argv);
  benchcfg::print_denominators(rl, *dev);

  // RawDeviceAllocator, deliberately not the Phase 2 pool: a bandwidth
  // measurement has no business exercising allocator logic, and the pool has its
  // own benchmark. Same reasoning as bench/stream_triad.cu.
  RawDeviceAllocator alloc;
  const std::int64_t n = kRows * kCols;
  const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(float);

  Buffers buf;
  auto need = [&](Allocation& a, std::size_t b) {
    auto r = alloc.allocate(b, stream->native());
    r.status().throw_if_error();
    a = *r;
  };
  need(buf.x,    nbytes);
  need(buf.y,    nbytes);
  need(buf.tmp,  nbytes);                      // the intermediate the fused path avoids
  need(buf.bias, static_cast<std::size_t>(kCols) * sizeof(float));

  buf.hx.resize(static_cast<std::size_t>(n));
  buf.hy.resize(static_cast<std::size_t>(n));
  buf.href.resize(static_cast<std::size_t>(n));
  buf.hbias.resize(static_cast<std::size_t>(kCols));
  testing::fill_random(buf.hx.data(), buf.hx.size(), 0xB1A5AC7ull, -3.0f, 3.0f);
  testing::fill_random(buf.hbias.data(), buf.hbias.size(), 0xB1A5B1A5ull, -1.0f, 1.0f);

  MCKE_CUDA_CHECK(cudaMemcpyAsync(buf.x.ptr, buf.hx.data(), nbytes,
                                  cudaMemcpyHostToDevice, stream->native()));
  MCKE_CUDA_CHECK(cudaMemcpyAsync(buf.bias.ptr, buf.hbias.data(),
                                  buf.hbias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream->native()));
  stream->synchronize().throw_if_error();

  std::printf("shape         %lld x %lld  (N = %lld, %.1f MiB per array)\n",
              (long long)kRows, (long long)kCols, (long long)n,
              double(nbytes) / (1024.0 * 1024.0));
  std::printf("ideal bytes   fused %llu   unfused %llu   (ratio %.3f)\n\n",
              (unsigned long long)fused_bytes(kRows, kCols),
              (unsigned long long)unfused_bytes(kRows, kCols),
              double(unfused_bytes(kRows, kCols)) / double(fused_bytes(kRows, kCols)));

  // ---------------------------------------------------------------------------
  // Correctness, at the benchmark shape, before any timing.
  // ---------------------------------------------------------------------------
  int validation_failures = 0;
  auto verify = [&](const char* what, K::Activation act, int vw, double tol,
                    const char* why) {
    MCKE_CUDA_CHECK(cudaMemsetAsync(buf.y.ptr, 0, nbytes, stream->native()));
    const Status st = K::launch_bias_act_f32(
        static_cast<const float*>(buf.x.ptr), static_cast<const float*>(buf.bias.ptr),
        static_cast<float*>(buf.y.ptr), kRows, kCols, act, vw, stream->native());
    st.throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(buf.hy.data(), buf.y.ptr, nbytes, cudaMemcpyDeviceToHost));
    testing::reference_bias_act(buf.hx.data(), buf.hbias.data(), buf.href.data(),
                                kRows, kCols, act);
    const auto r = testing::compare(buf.hy.data(), buf.href.data(),
                                    static_cast<std::size_t>(n), tol);
    benchcfg::print_validation(what, r.ok(), r.max_rel_err, tol, why);
    if (!r.ok()) { std::printf("   %s\n", r.to_string().c_str()); ++validation_failures; }
  };

  std::printf("=== correctness (at the benchmark shape, before timing) ====\n");
  // kNone and kRelu are validated with ZERO tolerance. x+bias is a single IEEE
  // add, and two floats summed in double is exact, so the reference rounds to
  // exactly the correctly-rounded float result -- bit-exactness is available
  // here and a free bit-exact test is worth taking. relu only adds a select.
  verify("fused/none/vw4",      K::Activation::kNone,     4, 0.0,
         "single IEEE add: bit-exact");
  verify("fused/relu/vw4",      K::Activation::kRelu,     4, 0.0,
         "add + select: bit-exact");
  // The GELUs cannot be exact: device erff/tanhf and host std::erf/std::tanh are
  // different correct implementations (~2 ulp vs ~1 ulp), so ~10 ulp of
  // disagreement is expected and is not a bug.
  verify("fused/gelu_tanh/vw4", K::Activation::kGeluTanh, 4, testing::kTolElementwise,
         "device tanhf vs host std::tanh, ~10 ulp");
  verify("fused/gelu_erf/vw4",  K::Activation::kGeluErf,  4, testing::kTolElementwise,
         "device erff vs host std::erf, ~10 ulp");
  verify("fused/gelu_tanh/vw1", K::Activation::kGeluTanh, 1, testing::kTolElementwise,
         "width must not change the answer");

  // Negative test: the launcher must REJECT an illegal width rather than
  // silently downgrade. Validation that is never exercised does not work.
  {
    const Status bad = K::launch_bias_act_f32(
        static_cast<const float*>(buf.x.ptr), static_cast<const float*>(buf.bias.ptr),
        static_cast<float*>(buf.y.ptr), 2, 4095, K::Activation::kRelu, 4,
        stream->native());
    const bool rejected = !bad.ok();
    std::printf("validation    %-28s %s  (cols=4095 %% 4 != 0 must be refused)\n",
                "reject/misaligned_width", rejected ? "OK  " : "FAIL");
    if (!rejected) ++validation_failures;
    // An in-place call violates the __restrict__ contract and must also be refused.
    const Status ip = K::launch_bias_act_f32(
        static_cast<const float*>(buf.x.ptr), static_cast<const float*>(buf.bias.ptr),
        static_cast<float*>(buf.x.ptr), kRows, kCols, K::Activation::kRelu, 4,
        stream->native());
    std::printf("validation    %-28s %s  (x == y violates __restrict__)\n",
                "reject/in_place", !ip.ok() ? "OK  " : "FAIL");
    if (ip.ok()) ++validation_failures;
  }
  if (validation_failures) {
    std::printf("\n*** %d VALIDATION FAILURE(S) -- timings below are meaningless ***\n",
                validation_failures);
  }
  std::printf("\n");

  // ---------------------------------------------------------------------------
  // Timing
  // ---------------------------------------------------------------------------
  Profiler prof;

  auto time_fused = [&](const std::string& name, const std::string& variant,
                        K::Activation act, int vw, std::int64_t rows,
                        std::int64_t cols, int max_row_blocks) {
    auto rec = prof.time_op(
        name, variant, *stream, act_flops(act, rows * cols), fused_bytes(rows, cols),
        kWarmup, kIters, [&](const rt::Stream& s) {
          return K::launch_bias_act_f32(
              static_cast<const float*>(buf.x.ptr),
              static_cast<const float*>(buf.bias.ptr),
              static_cast<float*>(buf.y.ptr), rows, cols, act, vw,
              s.native(), max_row_blocks);
        });
    rec.status().throw_if_error();
  };

  auto time_unfused = [&](const std::string& name, const std::string& variant,
                          K::Activation act, int vw, std::int64_t rows,
                          std::int64_t cols) {
    // BOTH launches inside ONE event bracket. The second kernel's launch
    // overhead is a genuine cost of not fusing and must not be excluded.
    auto rec = prof.time_op(
        name, variant, *stream, act_flops(act, rows * cols), unfused_bytes(rows, cols),
        kWarmup, kIters, [&](const rt::Stream& s) {
          MCKE_RETURN_IF_ERROR(K::launch_bias_add_f32(
              static_cast<const float*>(buf.x.ptr),
              static_cast<const float*>(buf.bias.ptr),
              static_cast<float*>(buf.tmp.ptr), rows, cols, vw, s.native()));
          return K::launch_activation_f32(
              static_cast<const float*>(buf.tmp.ptr),
              static_cast<float*>(buf.y.ptr), rows * cols, act, vw, s.native());
        });
    rec.status().throw_if_error();
  };

  // (a) activation sweep, fused vs unfused, at the widest width.
  for (K::Activation act : {K::Activation::kRelu, K::Activation::kGeluTanh,
                            K::Activation::kGeluErf}) {
    time_fused(std::string("bias_") + act_name(act), "fused_vw4", act, 4,
               kRows, kCols, 0);
    time_unfused(std::string("bias_") + act_name(act), "unfused_pair_vw4", act, 4,
                 kRows, kCols);
  }

  // (b) width sweep at full occupancy. PREDICTION: flat.
  for (int vw : {1, 2, 4})
    time_fused("bias_gelu_tanh", "fused_vw" + std::to_string(vw),
               K::Activation::kGeluTanh, vw, kRows, kCols, 0);

  // (c) the SAME width sweep, occupancy-starved to ~1 block per SM.
  //     PREDICTION: now the width matters, and that contrast is the real result.
  for (int vw : {1, 2, 4})
    time_fused("bias_gelu_tanh", "fused_vw" + std::to_string(vw) + "_lowocc",
               K::Activation::kGeluTanh, vw, kRows, kCols, dev->sm_count);

  // (d) L2-resident control. PREDICTION: the fusion win collapses to ~1.0-1.2x
  //     because the unfused pair's intermediate never leaves cache.
  time_fused("bias_gelu_tanh_L2", "fused_vw4_512x512", K::Activation::kGeluTanh, 4,
             512, 512, 0);
  time_unfused("bias_gelu_tanh_L2", "unfused_pair_vw4_512x512",
               K::Activation::kGeluTanh, 4, 512, 512);

  std::printf("%s\n", prof.summary_table(rl).c_str());

  // ---------------------------------------------------------------------------
  // The comparisons the table is FOR, computed rather than left to the reader.
  // ---------------------------------------------------------------------------
  auto find = [&](const std::string& name, const std::string& variant) -> const KernelRecord* {
    for (const auto& r : prof.records())
      if (r.name == name && r.variant == variant) return &r;
    return nullptr;
  };
  std::printf("=== fusion speedup (RESULTS.md section 3a) ==================\n");
  for (K::Activation act : {K::Activation::kRelu, K::Activation::kGeluTanh,
                            K::Activation::kGeluErf}) {
    const auto* f = find(std::string("bias_") + act_name(act), "fused_vw4");
    const auto* u = find(std::string("bias_") + act_name(act), "unfused_pair_vw4");
    if (f && u)
      std::printf("  %-12s fused %.3f ms   unfused %.3f ms   speedup %.2fx  "
                  "(predicted ~2.00x)\n",
                  act_name(act), f->median_ms, u->median_ms, u->median_ms / f->median_ms);
  }
  {
    const auto* f = find("bias_gelu_tanh_L2", "fused_vw4_512x512");
    const auto* u = find("bias_gelu_tanh_L2", "unfused_pair_vw4_512x512");
    if (f && u)
      std::printf("  %-12s fused %.3f ms   unfused %.3f ms   speedup %.2fx  "
                  "(L2-resident CONTROL, predicted ~1.0-1.2x)\n",
                  "512x512", f->median_ms, u->median_ms, u->median_ms / f->median_ms);
  }
  std::printf("\n=== vector width: full occupancy vs starved =================\n");
  for (int vw : {1, 2, 4}) {
    const auto* full = find("bias_gelu_tanh", "fused_vw" + std::to_string(vw));
    const auto* low  = find("bias_gelu_tanh", "fused_vw" + std::to_string(vw) + "_lowocc");
    if (full && low)
      std::printf("  vw=%d   full-occupancy %.3f ms (%.1f GB/s)   starved %.3f ms (%.1f GB/s)\n",
                  vw, full->median_ms, full->gb_per_s(), low->median_ms, low->gb_per_s());
  }
  std::printf("  PREDICTION: full-occupancy row is flat across widths (issue is ~138x\n"
              "  from being the limiter); starved row shows a large win, because vector\n"
              "  width buys memory-level parallelism and MLP is only scarce when\n"
              "  occupancy is scarce. If BOTH are flat, the starvation was insufficient.\n");
  {
    const auto* e = find("bias_gelu_erf", "fused_vw4");
    const auto* t = find("bias_gelu_tanh", "fused_vw4");
    if (e && t)
      std::printf("\n=== GELU form: erf vs tanh =================================\n"
                  "  erf %.3f ms vs tanh %.3f ms  -> %+.1f%%\n"
                  "  PREDICTION: within ~2%% (about 100 us of extra FP32-pipe work\n"
                  "  hidden under a ~1140 us memory floor). If erf is 5-9%% slower the\n"
                  "  arithmetic is NO LONGER fully hidden -- check ncu's top warp-stall\n"
                  "  reason for a shift from long_scoreboard to mio_throttle.\n",
                  e->median_ms, t->median_ms,
                  100.0 * (e->median_ms - t->median_ms) / t->median_ms);
  }

  const Status csv = prof.write_csv("phase3_bias_act.csv", rl);
  std::printf("\n%s\n", csv.ok() ? "wrote phase3_bias_act.csv"
                                 : csv.to_string().c_str());

  alloc.deallocate(buf.x, stream->native()).throw_if_error();
  alloc.deallocate(buf.y, stream->native()).throw_if_error();
  alloc.deallocate(buf.tmp, stream->native()).throw_if_error();
  alloc.deallocate(buf.bias, stream->native()).throw_if_error();
  return validation_failures ? 1 : 0;
}
