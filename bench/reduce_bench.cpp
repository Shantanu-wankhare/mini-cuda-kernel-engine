// =============================================================================
//  bench/reduce_bench.cpp
//
//  WHAT: Phase 3b. Row-wise reduction: shared-memory tree vs warp shuffle vs
//        two-pass, at two deliberately different shapes.
//
//  WHY .cpp: no __global__, no <<<>>> -- it calls the launchers. Same boundary
//  as bench/bias_act_bench.cpp; see that file's banner.
//
//  ---------------------------------------------------------------------------
//  WHY TWO SHAPES, AND WHY THE SECOND ONE IS NOT OPTIONAL
//
//  Both shapes move the SAME 33.5M elements, so the ideal-byte count and
//  therefore the GB/s denominator is identical -- only the row/column split
//  changes. That is what makes them a controlled comparison rather than two
//  unrelated measurements.
//
//    8192 x 4096   one block per row => 8192 blocks over 40 SMs = 51 waves.
//                  The machine is saturated. kTwoPass can only add overhead
//                  here: +0.4% traffic for its partials and ~5 us for the extra
//                  launch. PREDICTED 1-3% SLOWER.
//
//      64 x 524288 one block per row => 64 blocks on 40 SMs = 0.4 waves, with
//                  24 SMs sitting idle for the whole kernel. Splitting each row
//                  across blocks is the only way to fill them.
//                  PREDICTED 3-10x FASTER.
//
//  Benchmarked only at the first shape, kTwoPass looks like a variant that never
//  wins and someone reasonably deletes it. The second shape is the entire
//  justification for its existence -- and it is also the correction to what
//  kernels.hpp originally claimed, which was that kTwoPass is "for very long
//  rows". Row length is never the problem; a grid-stride loop handles any length
//  in one block. Too FEW ROWS to fill the machine is the problem.
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

constexpr std::int64_t kRowsA = 8192,  kColsA = 4096;      // saturated
constexpr std::int64_t kRowsB = 64,    kColsB = 524288;    // starved, same N
constexpr int kWarmup = 5;
constexpr int kIters  = 20;

const char* kind_name(K::ReduceKind k) {
  switch (k) {
    case K::ReduceKind::kSum:  return "sum";
    case K::ReduceKind::kMax:  return "max";
    case K::ReduceKind::kMean: return "mean";
  }
  return "?";
}
const char* variant_name(K::ReduceVariant v) {
  switch (v) {
    case K::ReduceVariant::kSmemTree:    return "smem_tree_256t";
    case K::ReduceVariant::kWarpShuffle: return "warp_shuffle_256t";
    case K::ReduceVariant::kTwoPass:     return "two_pass_256t";
  }
  return "?";
}

// Ideal (compulsory) traffic: read every element once, write one result per row.
//
// CONVENTION, stated once so nobody later "fixes" it: all three variants use
// this same count, INCLUDING kTwoPass. Its extra partial-staging traffic is
// algorithmic, not compulsory, so folding it into the denominator would make the
// GB/s column stop being a like-for-like comparison. It goes in a footnote.
std::uint64_t ideal_bytes(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(rows * cols + rows) * sizeof(float);
}
// n-1 combines for n elements. AI = 0.25 FLOP/byte -- 137x below the ridge
// point, which is why this bench reports GB/s and never TFLOP/s.
std::uint64_t ideal_flops(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::uint64_t>(rows * (cols - 1));
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
  const std::int64_t n = kRowsA * kColsA;                 // same N for both shapes
  const std::size_t xbytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t obytes = static_cast<std::size_t>(kRowsA) * sizeof(float);

  auto grab = [&](std::size_t b) {
    auto r = alloc.allocate(b, stream->native());
    r.status().throw_if_error();
    return *r;
  };
  Allocation dx  = grab(xbytes);
  Allocation dout = grab(obytes);
  // Workspace sized for the WORST case across both shapes, allocated once and
  // outside every timed region -- a cudaMalloc inside the loop would make the
  // kTwoPass row a measurement of the driver (Phase 2 measured cudaMalloc at up
  // to 720 us, with a cudaFree that synchronises the device).
  const std::size_t wsA = K::row_reduce_workspace_bytes(kRowsA, kColsA,
                                                        K::ReduceVariant::kTwoPass);
  const std::size_t wsB = K::row_reduce_workspace_bytes(kRowsB, kColsB,
                                                        K::ReduceVariant::kTwoPass);
  Allocation dws = grab(wsA > wsB ? wsA : wsB);

  std::vector<float> hx(static_cast<std::size_t>(n));
  std::vector<float> hout(static_cast<std::size_t>(kRowsA));
  std::vector<float> href(static_cast<std::size_t>(kRowsA));
  testing::fill_random(hx.data(), hx.size(), 0x5EED12345ull, -1.0f, 1.0f);
  MCKE_CUDA_CHECK(cudaMemcpyAsync(dx.ptr, hx.data(), xbytes,
                                  cudaMemcpyHostToDevice, stream->native()));
  stream->synchronize().throw_if_error();

  std::printf("shapes        A = %lld x %lld (saturated, %.0f waves)   "
              "B = %lld x %lld (starved, %.1f waves)\n",
              (long long)kRowsA, (long long)kColsA, double(kRowsA) / 160.0,
              (long long)kRowsB, (long long)kColsB, double(kRowsB) / 160.0);
  std::printf("ideal bytes   %llu (%.2f MiB), identical for all variants\n",
              (unsigned long long)ideal_bytes(kRowsA, kColsA),
              double(ideal_bytes(kRowsA, kColsA)) / (1024.0 * 1024.0));
  std::printf("two-pass split  shape A: %lld   shape B: %lld\n\n",
              (long long)(wsA / (kRowsA * sizeof(float))),
              (long long)(wsB / (kRowsB * sizeof(float))));

  // ---------------------------------------------------------------------------
  // Correctness first.
  // ---------------------------------------------------------------------------
  int failures = 0;
  auto verify = [&](K::ReduceKind kind, K::ReduceVariant var,
                    std::int64_t rows, std::int64_t cols, double tol, const char* why,
                    double abs_tol = 1e-8) {
    MCKE_CUDA_CHECK(cudaMemsetAsync(dout.ptr, 0, obytes, stream->native()));
    const Status st = K::launch_row_reduce_f32(
        static_cast<const float*>(dx.ptr), static_cast<float*>(dout.ptr), rows, cols,
        kind, var, static_cast<float*>(dws.ptr), dws.bytes, stream->native());
    st.throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(hout.data(), dout.ptr,
                               static_cast<std::size_t>(rows) * sizeof(float),
                               cudaMemcpyDeviceToHost));
    testing::reference_row_reduce(hx.data(), href.data(), rows, cols, kind);
    const auto r = testing::compare(hout.data(), href.data(),
                                    static_cast<std::size_t>(rows), tol, abs_tol);
    char label[64];
    std::snprintf(label, sizeof(label), "%s/%s", kind_name(kind), variant_name(var));
    benchcfg::print_validation(label, r.ok(), r.max_rel_err, tol, why);
    if (!r.ok()) { std::printf("   %s\n", r.to_string().c_str()); ++failures; }
  };

  std::printf("=== correctness (shape A, before timing) ===================\n");
  for (auto var : {K::ReduceVariant::kSmemTree, K::ReduceVariant::kWarpShuffle,
                   K::ReduceVariant::kTwoPass}) {
    // Sum needs the derived absolute floor: the accumulated rounding error
    // over 4096 zero-mean terms is set by the TERM magnitude (~1), not by the
    // row's own sum, and some rows nearly cancel by chance -- routine, not
    // adversarial. A pure relative test against the default 1e-8 floor
    // spuriously fails those rows even though the kernel is correct. See
    // tests/reference.hpp's derivation of kAbsTolReduceSum4096.
    //
    // Mean does NOT need it: both the value and the tolerance floor shrink by
    // the same factor of cols, so the plain relative test already has margin.
    verify(K::ReduceKind::kSum,  var, kRowsA, kColsA, testing::kTolReduce4096,
           "sqrt(4096)*f32 eps, term-magnitude absolute floor",
           testing::kAbsTolReduceSum4096);
    verify(K::ReduceKind::kMean, var, kRowsA, kColsA, testing::kTolReduce4096,
           "as sum, plus one divide");
    // MAX IS EXACT. It is a selection, not an arithmetic operation -- it returns
    // one of the inputs unchanged, identically in float and double. So this is a
    // free bit-exact test, and a free bit-exact test is worth taking. Any
    // mismatch is a genuine bug, almost always the -FLT_MAX vs -INFINITY
    // identity or a partial warp whose idle lanes were not padded.
    verify(K::ReduceKind::kMax,  var, kRowsA, kColsA, 0.0,
           "selection, not arithmetic: bit-exact");
  }
  // The frozen 7-arg overload must REFUSE kTwoPass rather than silently
  // allocating a workspace behind the caller's back.
  {
    const Status st = K::launch_row_reduce_f32(
        static_cast<const float*>(dx.ptr), static_cast<float*>(dout.ptr),
        kRowsA, kColsA, K::ReduceKind::kSum, K::ReduceVariant::kTwoPass,
        stream->native());
    std::printf("validation    %-28s %s  (7-arg form must refuse kTwoPass)\n",
                "reject/no_workspace", !st.ok() ? "OK  " : "FAIL");
    if (st.ok()) ++failures;
  }
  if (failures) std::printf("\n*** %d VALIDATION FAILURE(S) -- timings meaningless ***\n",
                            failures);
  std::printf("\n");

  // ---------------------------------------------------------------------------
  // Timing
  // ---------------------------------------------------------------------------
  Profiler prof;
  auto time_it = [&](K::ReduceKind kind, K::ReduceVariant var,
                     std::int64_t rows, std::int64_t cols, const char* shape_tag) {
    const std::string name = std::string("row_reduce_") + kind_name(kind);
    const std::string variant = std::string(variant_name(var)) + shape_tag;
    auto rec = prof.time_op(
        name, variant, *stream, ideal_flops(rows, cols), ideal_bytes(rows, cols),
        kWarmup, kIters, [&](const rt::Stream& s) {
          return K::launch_row_reduce_f32(
              static_cast<const float*>(dx.ptr), static_cast<float*>(dout.ptr),
              rows, cols, kind, var, static_cast<float*>(dws.ptr), dws.bytes,
              s.native());
        });
    rec.status().throw_if_error();
  };

  // Shape A: all three kinds x tree vs shuffle, plus two-pass for sum.
  for (auto var : {K::ReduceVariant::kSmemTree, K::ReduceVariant::kWarpShuffle})
    for (auto kind : {K::ReduceKind::kSum, K::ReduceKind::kMax, K::ReduceKind::kMean})
      time_it(kind, var, kRowsA, kColsA, "");
  time_it(K::ReduceKind::kSum, K::ReduceVariant::kTwoPass, kRowsA, kColsA, "");

  // Shape B: the starved regime that justifies kTwoPass existing.
  for (auto var : {K::ReduceVariant::kWarpShuffle, K::ReduceVariant::kTwoPass})
    time_it(K::ReduceKind::kSum, var, kRowsB, kColsB, "_starved");

  std::printf("%s\n", prof.summary_table(rl).c_str());

  auto find = [&](const std::string& n, const std::string& v) -> const KernelRecord* {
    for (const auto& r : prof.records())
      if (r.name == n && r.variant == v) return &r;
    return nullptr;
  };
  std::printf("=== tree vs shuffle (RESULTS.md section 3b) ================\n");
  for (auto kind : {K::ReduceKind::kSum, K::ReduceKind::kMax, K::ReduceKind::kMean}) {
    const std::string nm = std::string("row_reduce_") + kind_name(kind);
    const auto* t = find(nm, "smem_tree_256t");
    const auto* w = find(nm, "warp_shuffle_256t");
    if (t && w)
      std::printf("  %-5s tree %.3f ms (%.1f GB/s, 9 barriers)   "
                  "shuffle %.3f ms (%.1f GB/s, 1 barrier)   %+.1f%%\n",
                  kind_name(kind), t->median_ms, t->gb_per_s(),
                  w->median_ms, w->gb_per_s(),
                  100.0 * (t->median_ms - w->median_ms) / t->median_ms);
  }
  {
    const auto* t = find("row_reduce_sum", "smem_tree_256t");
    if (t)
      std::printf("  PREDICTION was 10-30%%. Note the CEILING: both variants move the\n"
                  "  same %.1f MiB and DRAM is the wall, so if the tree already reaches\n"
                  "  %.0f%% of %.1f GB/s the maximum possible win is only %.0f%%.\n",
                  double(ideal_bytes(kRowsA, kColsA)) / (1024.0 * 1024.0),
                  100.0 * t->gb_per_s() / rl.peak_gb_s, rl.peak_gb_s,
                  100.0 * (rl.peak_gb_s - t->gb_per_s()) / rl.peak_gb_s);
  }

  std::printf("\n=== two-pass: saturated vs starved =========================\n");
  {
    const auto* w_sat = find("row_reduce_sum", "warp_shuffle_256t");
    const auto* t_sat = find("row_reduce_sum", "two_pass_256t");
    if (w_sat && t_sat)
      std::printf("  8192x4096  shuffle %.3f ms   two_pass %.3f ms   %+.1f%%  "
                  "(predicted two_pass 1-3%% SLOWER)\n",
                  w_sat->median_ms, t_sat->median_ms,
                  100.0 * (t_sat->median_ms - w_sat->median_ms) / w_sat->median_ms);
    const auto* w_st = find("row_reduce_sum", "warp_shuffle_256t_starved");
    const auto* t_st = find("row_reduce_sum", "two_pass_256t_starved");
    if (w_st && t_st)
      std::printf("  64x524288  shuffle %.3f ms   two_pass %.3f ms   %.2fx  "
                  "(predicted two_pass 3-10x FASTER)\n",
                  w_st->median_ms, t_st->median_ms, w_st->median_ms / t_st->median_ms);
  }
  std::printf("  Same N and the same ideal byte count in both rows -- only the\n"
              "  row/column split changes, so this isolates the mapping and nothing else.\n");

  const Status csv = prof.write_csv("phase3_reduce.csv", rl);
  std::printf("\n%s\n", csv.ok() ? "wrote phase3_reduce.csv" : csv.to_string().c_str());

  alloc.deallocate(dx, stream->native()).throw_if_error();
  alloc.deallocate(dout, stream->native()).throw_if_error();
  alloc.deallocate(dws, stream->native()).throw_if_error();
  return failures ? 1 : 0;
}
