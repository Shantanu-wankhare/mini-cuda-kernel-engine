// =============================================================================
//  bench/softmax_bench.cpp
//
//  WHAT: Phase 3c. Three-pass stable softmax vs the online one-pass form.
//
//  WHY .cpp: no __global__, no <<<>>>. Same boundary as the other Phase 3
//  benches; see bench/bias_act_bench.cpp's banner.
//
//  ---------------------------------------------------------------------------
//  THE IDEAL-BYTES CONVENTION, STATED ONCE
//
//  Both variants use the COMPULSORY traffic as their ideal-byte count:
//      2 * N * 4   (read x once, write y once)
//  which is the physical minimum any correct softmax must move. It is identical
//  for both, so the GB/s column ranks them directly.
//
//  Their ALGORITHMIC traffic differs -- three-pass reads x three times (4N*4),
//  online reads it twice (3N*4) -- and that goes in a footnote, not the
//  denominator. The consequence to read correctly: neither variant will reach
//  100% of measured bandwidth, and how far below it lands is exactly the cost of
//  the extra passes. That is the honest way to present it. Putting algorithmic
//  traffic in the denominator would flatter both variants and hide the very
//  thing being compared.
//
//  ---------------------------------------------------------------------------
//  A REFERENCE-FREE NUMERICS METRIC
//
//  Every softmax row must sum to exactly 1. `max |sum(row) - 1|` needs no CPU
//  reference at all, costs one host pass, and directly exposes the one place the
//  online form is expected to be WORSE than three-pass: three-pass computes the
//  numerator and denominator from the identical expression expf(x - m) with the
//  same m, so they are bit-consistent; online builds its denominator through a
//  chain of rescaled partials and then computes a fresh numerator, so the two
//  agree only to within accumulated rounding.
//
//  Predicted: three-pass ~1e-7, online ~3e-7. Online being slightly less
//  accurate is the EXPECTED result, not a defect -- and it is worth reporting
//  because "the faster variant is also slightly less accurate" is the kind of
//  tradeoff that gets quietly dropped from write-ups.
// =============================================================================
#include <cmath>
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

constexpr std::int64_t kRows = 8192, kCols = 4096;
constexpr int kWarmup = 5, kIters = 20;

const char* variant_name(K::SoftmaxVariant v) {
  return v == K::SoftmaxVariant::kThreePass ? "three_pass_256t" : "online_one_pass_256t";
}

// Compulsory traffic only -- see the banner.
std::uint64_t ideal_bytes(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(2 * rows * cols) * sizeof(float);
}
// ~5 flops/element (a max, a subtract, an exp, an add, a multiply). AI ~= 0.6,
// 57x below the ridge point -- memory-bound, so this bench reports GB/s.
std::uint64_t ideal_flops(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(rows * cols) * 5;
}

// max over rows of |sum(row) - 1|. Reference-free.
double max_row_sum_error(const std::vector<float>& y, std::int64_t rows,
                         std::int64_t cols) {
  double worst = 0.0;
  for (std::int64_t r = 0; r < rows; ++r) {
    double s = 0.0;
    for (std::int64_t c = 0; c < cols; ++c) s += y[static_cast<std::size_t>(r * cols + c)];
    worst = std::max(worst, std::fabs(s - 1.0));
  }
  return worst;
}

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

  const Roofline rl = benchcfg::make_roofline(argc, argv);
  benchcfg::print_denominators(rl, *dev);

  RawDeviceAllocator alloc;
  const std::int64_t n = kRows * kCols;
  const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(float);
  auto grab = [&](std::size_t b) {
    auto r = alloc.allocate(b, stream->native());
    r.status().throw_if_error();
    return *r;
  };
  Allocation dx = grab(nbytes), dy = grab(nbytes);

  std::vector<float> hx(static_cast<std::size_t>(n));
  std::vector<float> hy(static_cast<std::size_t>(n));
  std::vector<float> href(static_cast<std::size_t>(n));
  // Range +/-8 rather than +/-1: wide enough that the max subtraction actually
  // does something and the exponentials span several orders of magnitude.
  testing::fill_random(hx.data(), hx.size(), 0x50F7A11ull, -8.0f, 8.0f);
  MCKE_CUDA_CHECK(cudaMemcpyAsync(dx.ptr, hx.data(), nbytes,
                                  cudaMemcpyHostToDevice, stream->native()));
  stream->synchronize().throw_if_error();

  std::printf("shape         %lld x %lld   compulsory bytes %llu (%.1f MiB)\n",
              (long long)kRows, (long long)kCols,
              (unsigned long long)ideal_bytes(kRows, kCols),
              double(ideal_bytes(kRows, kCols)) / (1024.0 * 1024.0));
  std::printf("algorithmic   three_pass 4N*4 = %.1f MiB   online 3N*4 = %.1f MiB\n",
              double(4 * n * 4) / (1024.0 * 1024.0),
              double(3 * n * 4) / (1024.0 * 1024.0));
  std::printf("L2 note       each row is %lld KiB; ~160 resident blocks give a ~%.1f MiB\n"
              "              working set, which FITS the T4's 4 MiB L2 -- so three_pass's\n"
              "              2nd and 3rd reads may never reach DRAM. If it beats its own\n"
              "              traffic model, that is the finding, not an error.\n\n",
              (long long)(kCols * 4 / 1024), 160.0 * double(kCols * 4) / (1024.0 * 1024.0));

  // ---------------------------------------------------------------------------
  // Correctness first.
  // ---------------------------------------------------------------------------
  int failures = 0;
  testing::reference_row_softmax(hx.data(), href.data(), kRows, kCols);

  auto verify = [&](K::SoftmaxVariant var) {
    MCKE_CUDA_CHECK(cudaMemsetAsync(dy.ptr, 0, nbytes, stream->native()));
    K::launch_row_softmax_f32(static_cast<const float*>(dx.ptr),
                              static_cast<float*>(dy.ptr), kRows, kCols, var,
                              stream->native()).throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(hy.data(), dy.ptr, nbytes, cudaMemcpyDeviceToHost));
    const auto r = testing::compare(hy.data(), href.data(),
                                    static_cast<std::size_t>(n), testing::kTolSoftmax);
    benchcfg::print_validation(variant_name(var), r.ok(), r.max_rel_err,
                               testing::kTolSoftmax,
                               "expf ~2ulp + tree sum + divide");
    if (!r.ok()) { std::printf("   %s\n", r.to_string().c_str()); ++failures; }
    const double sum_err = max_row_sum_error(hy, kRows, kCols);
    std::printf("              max |sum(row) - 1| = %.3e   (reference-free)\n", sum_err);
    return sum_err;
  };

  std::printf("=== correctness (at the benchmark shape, before timing) ====\n");
  const double err_three  = verify(K::SoftmaxVariant::kThreePass);
  const double err_online = verify(K::SoftmaxVariant::kOnlineOnePass);

  // The adversarial row: a monotone ramp to 90. Without the max subtraction
  // expf(90) is +inf and the whole row becomes NaN, so this is the test that
  // pins the overflow guard. It is also the worst case for the online form's
  // rescaling, since the running max updates on essentially every element.
  {
    std::vector<float> ramp(static_cast<std::size_t>(kCols));
    for (std::int64_t c = 0; c < kCols; ++c)
      ramp[static_cast<std::size_t>(c)] = 90.0f * float(c) / float(kCols - 1);
    MCKE_CUDA_CHECK(cudaMemcpy(dx.ptr, ramp.data(), ramp.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
    int nan_count = 0;
    for (auto var : {K::SoftmaxVariant::kThreePass, K::SoftmaxVariant::kOnlineOnePass}) {
      K::launch_row_softmax_f32(static_cast<const float*>(dx.ptr),
                                static_cast<float*>(dy.ptr), 1, kCols, var,
                                stream->native()).throw_if_error();
      stream->synchronize().throw_if_error();
      MCKE_CUDA_CHECK(cudaMemcpy(hy.data(), dy.ptr, ramp.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost));
      double s = 0.0;
      for (std::size_t i = 0; i < ramp.size(); ++i) {
        if (std::isnan(hy[i]) || std::isinf(hy[i])) ++nan_count;
        s += hy[i];
      }
      std::printf("validation    %-28s %s  (logits 0..90; sum=%.6f, "
                  "no max-subtraction would give NaN)\n",
                  (std::string("overflow/") + variant_name(var)).c_str(),
                  (nan_count == 0 && std::fabs(s - 1.0) < 1e-4) ? "OK  " : "FAIL", s);
      if (nan_count || std::fabs(s - 1.0) >= 1e-4) ++failures;
    }
    // restore the random input
    MCKE_CUDA_CHECK(cudaMemcpy(dx.ptr, hx.data(), nbytes, cudaMemcpyHostToDevice));
  }
  {  // in-place must be refused
    const Status ip = K::launch_row_softmax_f32(
        static_cast<const float*>(dx.ptr), static_cast<float*>(dx.ptr),
        kRows, kCols, K::SoftmaxVariant::kThreePass, stream->native());
    std::printf("validation    %-28s %s  (x == y violates __restrict__)\n",
                "reject/in_place", !ip.ok() ? "OK  " : "FAIL");
    if (ip.ok()) ++failures;
  }
  if (failures) std::printf("\n*** %d VALIDATION FAILURE(S) -- timings meaningless ***\n",
                            failures);
  std::printf("\n");

  // ---------------------------------------------------------------------------
  // Timing
  // ---------------------------------------------------------------------------
  Profiler prof;
  for (auto var : {K::SoftmaxVariant::kThreePass, K::SoftmaxVariant::kOnlineOnePass}) {
    auto rec = prof.time_op("row_softmax", variant_name(var), *stream,
                            ideal_flops(kRows, kCols), ideal_bytes(kRows, kCols),
                            kWarmup, kIters, [&](const rt::Stream& s) {
                              return K::launch_row_softmax_f32(
                                  static_cast<const float*>(dx.ptr),
                                  static_cast<float*>(dy.ptr), kRows, kCols, var,
                                  s.native());
                            });
    rec.status().throw_if_error();
  }
  std::printf("%s\n", prof.summary_table(rl).c_str());

  auto find = [&](const char* v) -> const KernelRecord* {
    for (const auto& r : prof.records()) if (r.variant == v) return &r;
    return nullptr;
  };
  const auto* t3 = find("three_pass_256t");
  const auto* on = find("online_one_pass_256t");
  std::printf("=== three-pass vs online (RESULTS.md section 3c) ===========\n");
  if (t3 && on) {
    std::printf("  three_pass %.3f ms (%.1f GB/s)   online %.3f ms (%.1f GB/s)   %.2fx\n",
                t3->median_ms, t3->gb_per_s(), on->median_ms, on->gb_per_s(),
                t3->median_ms / on->median_ms);
    std::printf("  PREDICTION 4/3 = 1.33x from traffic alone (3 reads -> 2), less any L2\n"
                "  reuse, so realistically 1.0-1.25x. NOT 3x -- \"one pass\" names the\n"
                "  statistics passes, not the memory passes.\n");
    std::printf("  Neither reaches 100%% of %.1f GB/s BY CONSTRUCTION: the denominator is\n"
                "  compulsory traffic (2N*4), and the extra passes are exactly the gap.\n",
                rl.peak_gb_s);
  }
  std::printf("\n=== numerics ==============================================\n");
  std::printf("  max |sum(row) - 1|   three_pass %.3e   online %.3e   (ratio %.2fx)\n",
              err_three, err_online,
              err_three > 0.0 ? err_online / err_three : 0.0);
  std::printf("  PREDICTION: online is 2-5x LESS accurate, and that is expected, not a\n"
              "  defect. Three-pass computes numerator and denominator from the identical\n"
              "  expression with the same m, so they are bit-consistent; online builds the\n"
              "  denominator through a chain of rescaled partials and then computes a fresh\n"
              "  numerator, so they agree only to within accumulated rounding.\n");

  const Status csv = prof.write_csv("phase3_softmax.csv", rl);
  std::printf("\n%s\n", csv.ok() ? "wrote phase3_softmax.csv" : csv.to_string().c_str());

  alloc.deallocate(dx, stream->native()).throw_if_error();
  alloc.deallocate(dy, stream->native()).throw_if_error();
  return failures ? 1 : 0;
}
