// =============================================================================
//  bench/gemm_bench.cpp
//
//  WHAT: Drives the Phase 3d GEMM ladder -- validates every variant, then times
//        them at a single pinned shape, and prints everything RESULTS.md sec 3d
//        needs so the table can be filled without re-running or remembering.
//
//  WHY .cpp AND NOT .cu: it contains no __global__ and no <<<>>>, only calls to
//  the launchers declared in kernels.hpp. That is exactly the boundary that
//  header exists to define, and keeping nvcc off the bench keeps the Colab
//  edit-rebuild loop fast.
//
//  ---------------------------------------------------------------------------
//  THE ONE MEASUREMENT DECISION THAT DECIDES WHETHER THIS TABLE MEANS ANYTHING
//
//  `ideal_bytes` is the OPERATION's compulsory traffic, (M*K + K*N + M*N)*4, and
//  it is the same for all eight rows. Arithmetic intensity is therefore ~683 for
//  every row (682.7), and `%peak` is uniformly against the measured 8.130 TFLOP/s FMA
//  peak. That is what makes the rows comparable to each other.
//
//  The tempting alternative -- give the naive rows "their" AI of 0.25, the
//  access-pattern intensity -- produces nonsense, and it is worth doing the
//  arithmetic once rather than arguing about it:
//
//      attainable_tflops(0.25) = min(0.25 * 235.4e9, 8.130e12) / 1e12 = 0.0589
//      naive at the predicted 2-4% of peak                            ~ 0.16-0.33
//      efficiency = 0.33 / 0.0589                                     = 550%
//
//  A `%peak` column reading 550%. Same silent-garbage failure class that
//  bench_common.hpp's make_roofline() was written to make unreachable.
//
//  TWO CONSEQUENCES, STATED HERE SO THEY ARE NOT MISREAD IN THE OUTPUT:
//   * summary_table's `bound` column says "compute" for the naive rows. That is
//     CORRECT. The operation is compute-bound; the naive kernel simply fails to
//     exploit it. Being memory-bound in practice is a property of the
//     implementation, and the roofline's x-axis is not.
//   * summary_table's `GB/s` column is MEANINGLESS for every row here. It is
//     compulsory bytes over elapsed time, so the naive row prints ~0.35 GB/s,
//     which is neither achieved DRAM bandwidth nor anything else. RESULTS.md
//     sec 3d has no GB/s column for this reason; ignore it in the console table.
//
//  ---------------------------------------------------------------------------
//  WHY beta == 0 IS THE ONLY THING TIMED
//
//  C is read-modify-write when beta != 0, so across 20 timed iterations each
//  iteration consumes the previous one's output: C grows geometrically and
//  reaches inf. Restoring C between iterations would put a 64 MiB device copy
//  INSIDE the timed region, which is worse. So beta != 0 is exercised in the
//  correctness section at small shapes (where the full CPU reference is
//  affordable) and never timed. This is also the standard GEMM benchmark
//  convention. Do not "fix" this later.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// -----------------------------------------------------------------------------
// The ladder, in PRESENTATION order -- which is deliberately not the enum's
// numeric order, because GemmVariant is append-only so that recorded values
// never shift. This table is the authority on ordering and on names.
//
// Note `kWarpTile` is presented as "warptile_dbuf": the enum was named before
// kWarpTileNoDbuf split double buffering out, and RESULTS.md sec 3d's row label
// is the clearer one. Renaming the enumerator would have shifted its value.
// -----------------------------------------------------------------------------
struct Row {
  K::GemmVariant variant;
  const char*    name;
  K::GemmTile    tile;
  bool           uses_tile;   // naive + cuBLAS own no tile
};

constexpr K::GemmTile kSmemTile{32, 32, 32, 1, 1};
constexpr K::GemmTile kRegTile{128, 128, 8, 8, 8};

const Row kLadder[] = {
    {K::GemmVariant::kNaiveUncoalesced, "naive_uncoalesced", {},         false},
    {K::GemmVariant::kNaive,            "naive",             {},         false},
    {K::GemmVariant::kTiledSmem,        "tiled_smem",        kSmemTile,  true},
    {K::GemmVariant::kTiledRegBlock,    "tiled_regblock",    kRegTile,   true},
    {K::GemmVariant::kWarpTileNoDbuf,   "warptile_nodbuf",   kRegTile,   true},
    {K::GemmVariant::kWarpTile,         "warptile_dbuf",     kRegTile,   true},
    {K::GemmVariant::kWarpTileVec4,     "warptile_vec4",     kRegTile,   true},
    {K::GemmVariant::kCublasRef,        "cublas",            {},         false},
};
constexpr int kLadderN = static_cast<int>(sizeof(kLadder) / sizeof(kLadder[0]));

// RESULTS.md rule 4: the ideal counts must be reconstructible from the table.
// 2*M*N*K is the standard convention and EXCLUDES the 2*M*N for the alpha/beta
// epilogue, which is 0.05% here and would only muddy comparisons with published
// GEMM numbers.
[[nodiscard]] std::uint64_t ideal_flops(std::int64_t m, std::int64_t n, std::int64_t k) {
  return 2ull * static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n) *
         static_cast<std::uint64_t>(k);
}
// Compulsory traffic at beta == 0: read A and B once, write C once. C is not
// read, because beta == 0 must not read C (see kernels/gemm.cu).
[[nodiscard]] std::uint64_t ideal_bytes(std::int64_t m, std::int64_t n, std::int64_t k) {
  return (static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) +
          static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(n) +
          static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n)) * sizeof(float);
}

// A MODEL of what each implementation actually pulls from memory, as opposed to
// the compulsory minimum above. Labelled a model, never a measurement --
// docs/PROFILING.md promises this reuse story ("watching it fall from ~2000x to
// ~1x across the GEMM variants is the single most instructive number in this
// project") and it has had no producer until now.
[[nodiscard]] double modelled_read_bytes(const Row& r, std::int64_t m, std::int64_t n,
                                         std::int64_t k) {
  const double dm = double(m), dn = double(n), dk = double(k);
  if (!r.uses_tile) return 2.0 * dm * dn * dk * 4.0;   // no reuse at all
  // Each element of A is re-read once per column-tile, each of B once per
  // row-tile.
  return (dm * dk * (dn / r.tile.bn) + dk * dn * (dm / r.tile.bm)) * 4.0;
}

// -----------------------------------------------------------------------------
// Arguments.
//
// UNKNOWN FLAGS ARE FATAL. make_roofline() silently ignores anything it does not
// recognise, and a second independent scan would too -- so `--M=1024` (capital
// M) would run the DEFAULT shape while the echoed command line claims 1024,
// producing a wrong RESULTS.md row with no error anywhere. That is a direct
// rule-2 violation, and rejecting unknown tokens is the only structural fix.
// -----------------------------------------------------------------------------
struct Args {
  std::int64_t m = 4096, n = 4096, k = 4096;
  int          warmup = 5, iters = 20;
  std::string  only;             // empty = run the whole ladder
  bool         skip_validation = false;
};

[[nodiscard]] bool starts_with(const char* s, const char* p) {
  return std::strncmp(s, p, std::strlen(p)) == 0;
}

Args parse_args(int argc, char** argv) {
  Args a;
  bool square_set = false;
  for (int i = 1; i < argc; ++i) {
    const char* s = argv[i];
    if      (starts_with(s, "--m="))      a.m = std::atoll(s + 4);
    else if (starts_with(s, "--n="))      a.n = std::atoll(s + 4);
    else if (starts_with(s, "--k="))      a.k = std::atoll(s + 4);
    else if (starts_with(s, "--warmup=")) a.warmup = std::atoi(s + 9);
    else if (starts_with(s, "--iters="))  a.iters = std::atoi(s + 8);
    else if (starts_with(s, "--only="))   a.only = s + 7;
    else if (std::strcmp(s, "--skip-validation") == 0) a.skip_validation = true;
    // Consumed by benchcfg::make_roofline, listed here so they are not "unknown".
    else if (starts_with(s, "--peak-gb-s=") || starts_with(s, "--peak-tflops=")) {}
    else if (s[0] >= '0' && s[0] <= '9') { a.m = a.n = a.k = std::atoll(s); square_set = true; }
    else {
      std::fprintf(stderr,
          "[mcke] FATAL: unrecognised argument '%s'.\n"
          "  usage: mcke_gemm_bench [N | --m=M --n=N --k=K] [--only=<variant>]\n"
          "                         [--warmup=W] [--iters=I] [--skip-validation]\n"
          "                         [--peak-gb-s=X] [--peak-tflops=X]\n"
          "  Rejecting rather than ignoring: a silently-ignored shape flag makes\n"
          "  the echoed command line disagree with what actually ran.\n", s);
      std::exit(2);
    }
  }
  (void)square_set;
  return a;
}

const char* limiter_str(K::OccupancyLimiter l) { return K::limiter_name(l); }

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

  const Args args = parse_args(argc, argv);
  const Roofline rl = benchcfg::make_roofline(argc, argv);
  benchcfg::print_denominators(rl, *dev);

  // --- RESULTS.md rule 2: the exact command line. Echoed rather than described,
  //     because this binary now produces DIFFERENT TABLES depending on its
  //     arguments -- which was not true of any earlier bench.
  std::printf("command       ");
  for (int i = 0; i < argc; ++i) std::printf("%s%s", argv[i], i + 1 < argc ? " " : "\n");

  // --- RESULTS.md rule 1: driver / runtime / cuBLAS versions. cuBLAS is a ROW
  //     in this table, so its version is part of the environment.
  int drv = 0, rtv = 0;
  MCKE_CUDA_CHECK(cudaDriverGetVersion(&drv));
  MCKE_CUDA_CHECK(cudaRuntimeGetVersion(&rtv));
  std::printf("versions      driver %d.%d  runtime %d.%d   (cuBLAS in PEDANTIC math "
              "mode -- no TF32)\n", drv / 1000, (drv % 1000) / 10, rtv / 1000,
              (rtv % 1000) / 10);
  std::printf("clocks        NOT LOCKED (no root on Colab) -- see the drift check below\n");

  const std::int64_t m = args.m, n = args.n, k = args.k;
  const std::uint64_t flops = ideal_flops(m, n, k);
  const std::uint64_t bytes = ideal_bytes(m, n, k);
  const double ai = double(flops) / double(bytes);

  // --- RESULTS.md rules 3 and 4.
  std::printf("shape         M=%lld N=%lld K=%lld   alpha=1 beta=0 (timed)\n",
              (long long)m, (long long)n, (long long)k);
  std::printf("ideal         flops = 2*M*N*K = %.4g   bytes = (MK+KN+MN)*4 = %.4g\n",
              double(flops), double(bytes));
  std::printf("              AI = %.1f FLOP/byte vs ridge %.1f -- %s\n",
              ai, rl.ridge_point_ai(), ai > rl.ridge_point_ai() ? "COMPUTE-bound"
                                                                : "memory-bound");
  std::printf("timing        %d warmup + %d timed iterations, median and min\n",
              args.warmup, args.iters);
  if (args.iters < 20 || args.warmup < 5) {
    std::printf("\n*** NON-COMPLIANT WITH RESULTS.md RULE 3 (needs >=5 warmup, >=20 "
                "timed). ***\n*** This run is for profiling only -- do NOT put it in "
                "RESULTS.md. ***\n\n");
  }
  std::printf("note          the GB/s column below is MEANINGLESS for a compute-bound\n"
              "              kernel (compulsory bytes / time). Read TFLOP/s and %%peak.\n");
  // A float4 load cannot be partially predicated, so warptile_vec4 falls back to
  // the scalar instantiation -- which IS warptile_dbuf -- when the shape cannot
  // support 128-bit loads. Say so, or the row reads as a vec4 measurement when it
  // is a duplicate of the row above it. (M is deliberately not part of the test:
  // it only bounds a row index and nothing in the vectorization touches it.)
  const bool vec4_ok = (k % 4 == 0) && (n % 4 == 0);
  std::printf("vec4          K%%4=%lld N%%4=%lld -> warptile_vec4 uses %s\n\n",
              (long long)(k % 4), (long long)(n % 4),
              vec4_ok ? "128-bit loads"
                      : "*** the SCALAR fallback: this row duplicates warptile_dbuf ***");

  // ---------------------------------------------------------------------------
  // Memory. One allocation set, sized for the benchmark shape; the small
  // validation shapes reuse the front of the same buffers.
  // ---------------------------------------------------------------------------
  RawDeviceAllocator alloc;
  auto grab = [&](std::size_t b) {
    auto r = alloc.allocate(b, stream->native());
    r.status().throw_if_error();
    return *r;
  };
  const std::size_t a_bytes = std::size_t(m) * std::size_t(k) * sizeof(float);
  const std::size_t b_bytes = std::size_t(k) * std::size_t(n) * sizeof(float);
  const std::size_t c_bytes = std::size_t(m) * std::size_t(n) * sizeof(float);
  Allocation da = grab(a_bytes), db = grab(b_bytes), dc = grab(c_bytes);
  Allocation dref = grab(c_bytes);   // cuBLAS's output, the full-shape oracle

  std::vector<float> ha(std::size_t(m) * std::size_t(k));
  std::vector<float> hb(std::size_t(k) * std::size_t(n));
  std::vector<float> hc(std::size_t(m) * std::size_t(n));
  std::vector<float> href(std::size_t(m) * std::size_t(n));

  auto upload = [&](std::int64_t vm, std::int64_t vn, std::int64_t vk) {
    MCKE_CUDA_CHECK(cudaMemcpyAsync(da.ptr, ha.data(),
                                    std::size_t(vm) * std::size_t(vk) * sizeof(float),
                                    cudaMemcpyHostToDevice, stream->native()));
    MCKE_CUDA_CHECK(cudaMemcpyAsync(db.ptr, hb.data(),
                                    std::size_t(vk) * std::size_t(vn) * sizeof(float),
                                    cudaMemcpyHostToDevice, stream->native()));
    stream->synchronize().throw_if_error();
  };

  const float* dA = static_cast<const float*>(da.ptr);
  const float* dB = static_cast<const float*>(db.ptr);
  float*       dC = static_cast<float*>(dc.ptr);

  auto run = [&](const Row& r, std::int64_t vm, std::int64_t vn, std::int64_t vk,
                 float alpha, float beta, float* out) {
    return K::launch_gemm_f32(dA, dB, out, vm, vn, vk, alpha, beta, r.variant,
                              r.tile, stream->native());
  };

  // Which rows exist in this build? Stage 5 implements 1, 2, 3 and 8; the
  // register-blocked variants return UnimplementedError until Stage 6. Probe
  // once, at a trivial shape, so the rest of the bench can skip cleanly instead
  // of printing a wall of identical errors.
  bool implemented[kLadderN] = {};
  int  selected  [kLadderN] = {};
  for (int i = 0; i < kLadderN; ++i) {
    selected[i] = args.only.empty() || args.only == kLadder[i].name;
    if (!selected[i]) continue;
    const Status st = run(kLadder[i], 1, 1, 1, 1.0f, 0.0f, dC);
    implemented[i] = st.ok();
    if (st.ok()) stream->synchronize().throw_if_error();
  }
  if (!args.only.empty()) {
    bool any = false;
    for (int i = 0; i < kLadderN; ++i) any = any || selected[i];
    if (!any) {
      std::fprintf(stderr, "[mcke] FATAL: --only=%s matches no variant.\n",
                   args.only.c_str());
      return 2;
    }
  }

  int failures = 0;

  // ---------------------------------------------------------------------------
  // Correctness, part 1: awkward shapes against the double-accumulating CPU
  // reference.
  //
  // Every one of these targets a specific way a tiled GEMM goes wrong:
  //   1x1x1        the degenerate case; also the tightest tolerance, since a
  //                K=1 GEMM is a single multiply and should be near-exact
  //   5x7x3        smaller than one block tile in every dimension
  //   129x65x257   none of M, N, K is a multiple of any tile dimension -- the
  //                remainder path in all three axes at once
  //   1x64x128     a single row;  64x1x128 a single column
  //   33x63x1      K=1 with awkward M, N
  //   253x253x253  not a multiple of 4, so a float4 path MUST fall back to
  //                scalar rather than fault
  // ---------------------------------------------------------------------------
  struct Shape { std::int64_t m, n, k; const char* why; };
  const Shape kShapes[] = {
      {1, 1, 1,       "degenerate"},
      {5, 7, 3,       "smaller than one tile"},
      {129, 65, 257,  "no extent is a tile multiple"},
      {1, 64, 128,    "single row"},
      {64, 1, 128,    "single column"},
      {33, 63, 1,     "K=1"},
      {253, 253, 253, "not a multiple of 4 (vec4 fallback)"},
      {256, 256, 256, "clean multiple, the control"},
  };

  if (!args.skip_validation) {
    std::printf("=== correctness: awkward shapes vs CPU reference ===========\n");
    for (const Shape& s : kShapes) {
      testing::fill_random(ha.data(), std::size_t(s.m) * std::size_t(s.k), 0x6E33Aull, -1.0f, 1.0f);
      testing::fill_random(hb.data(), std::size_t(s.k) * std::size_t(s.n), 0x8B10Cull, -1.0f, 1.0f);
      upload(s.m, s.n, s.k);
      const std::size_t cn = std::size_t(s.m) * std::size_t(s.n);
      const double rel = testing::tol_gemm(s.k);
      const double abs = testing::abs_tol_gemm(s.k, 1.0);

      for (int i = 0; i < kLadderN; ++i) {
        if (!selected[i] || !implemented[i]) continue;
        MCKE_CUDA_CHECK(cudaMemsetAsync(dc.ptr, 0, cn * sizeof(float), stream->native()));
        run(kLadder[i], s.m, s.n, s.k, 1.0f, 0.0f, dC).throw_if_error();
        stream->synchronize().throw_if_error();
        MCKE_CUDA_CHECK(cudaMemcpy(hc.data(), dc.ptr, cn * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        std::fill(href.begin(), href.begin() + std::int64_t(cn), 0.0f);
        testing::reference_gemm(ha.data(), hb.data(), href.data(), s.m, s.n, s.k, 1.0f, 0.0f);
        const auto r = testing::compare(hc.data(), href.data(), cn, rel, abs);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %lldx%lldx%lld", kLadder[i].name,
                      (long long)s.m, (long long)s.n, (long long)s.k);
        if (!r.ok()) {
          benchcfg::print_validation(label, false, r.max_rel_err, rel, s.why);
          std::printf("   %s\n", r.to_string().c_str());
          ++failures;
        }
      }
    }
    std::printf("              (only FAILURES are printed above; silence is success)\n");
    std::printf("              tolerances are derived per-K: rel=4*sqrt(K)*eps, "
                "abs=k*eps*amax^2/2\n");

    // -------------------------------------------------------------------------
    // Correctness, part 2: the beta != 0 read-modify-write path, small shape
    // only. Never timed -- see the file banner.
    // -------------------------------------------------------------------------
    {
      const std::int64_t bm = 129, bn = 65, bk = 257;
      const std::size_t cn = std::size_t(bm) * std::size_t(bn);
      testing::fill_random(ha.data(), std::size_t(bm) * std::size_t(bk), 0x6E33Aull, -1.0f, 1.0f);
      testing::fill_random(hb.data(), std::size_t(bk) * std::size_t(bn), 0x8B10Cull, -1.0f, 1.0f);
      testing::fill_random(hc.data(), cn, 0xC0FFEEull, -1.0f, 1.0f);
      upload(bm, bn, bk);
      const double rel = testing::tol_gemm(bk);
      // alpha*A@B can nearly cancel beta*C, so the absolute floor has to cover
      // the C term too -- the same cancellation trap as kAbsTolReduceSum4096.
      const double abs = testing::abs_tol_gemm(bk, 1.0) + 1e-6;
      for (int i = 0; i < kLadderN; ++i) {
        if (!selected[i] || !implemented[i]) continue;
        MCKE_CUDA_CHECK(cudaMemcpy(dc.ptr, hc.data(), cn * sizeof(float),
                                   cudaMemcpyHostToDevice));
        run(kLadder[i], bm, bn, bk, 0.5f, 2.0f, dC).throw_if_error();
        stream->synchronize().throw_if_error();
        std::vector<float> got(cn);
        MCKE_CUDA_CHECK(cudaMemcpy(got.data(), dc.ptr, cn * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        std::copy(hc.begin(), hc.begin() + std::int64_t(cn), href.begin());
        testing::reference_gemm(ha.data(), hb.data(), href.data(), bm, bn, bk, 0.5f, 2.0f);
        const auto r = testing::compare(got.data(), href.data(), cn, rel, abs);
        char label[64];
        std::snprintf(label, sizeof(label), "%s beta=2", kLadder[i].name);
        benchcfg::print_validation(label, r.ok(), r.max_rel_err, rel,
                                   "alpha=0.5 beta=2, read-modify-write");
        if (!r.ok()) { std::printf("   %s\n", r.to_string().c_str()); ++failures; }
      }
    }
    std::printf("\n");
  }

  // ---------------------------------------------------------------------------
  // Correctness, part 3: THE BENCHMARK SHAPE.
  //
  // bias_act_bench.cpp states the project standard -- validate at the exact
  // shape that produces the published number, not at a smaller proxy. A 4096^3
  // CPU reference is minutes, so two cheap substitutes stand in:
  //
  //   (a) cuBLAS as a full-shape oracle. It was just validated against the
  //       double-accumulating CPU reference at 129x65x257 -- deliberately
  //       NON-SQUARE, because the row-major/column-major swap in launch_cublas
  //       produces a transposed result that still passes at M == N for a
  //       disturbingly wide class of inputs. Having earned trust there, it
  //       becomes the reference here.
  //   (b) a random spot-check: 1024 output elements recomputed in double at
  //       O(K) each. Milliseconds, and it covers the tile-boundary and k-tail
  //       paths that the small shapes cannot reach.
  //
  // Without these, the eight published numbers would be for kernels verified
  // only at OTHER shapes -- and the double-buffering race in Stage 6 is exactly
  // the kind of bug that would not fire at K=256 and would corrupt only the
  // untested timing run.
  // ---------------------------------------------------------------------------
  testing::fill_random(ha.data(), ha.size(), 0x6E33Aull, -1.0f, 1.0f);
  testing::fill_random(hb.data(), hb.size(), 0x8B10Cull, -1.0f, 1.0f);
  upload(m, n, k);
  const std::size_t cn = std::size_t(m) * std::size_t(n);

  std::printf("=== correctness at the benchmark shape ====================\n");
  bool have_oracle = false;
  if (selected[kLadderN - 1] && implemented[kLadderN - 1]) {
    MCKE_CUDA_CHECK(cudaMemsetAsync(dref.ptr, 0, c_bytes, stream->native()));
    K::launch_gemm_f32(dA, dB, static_cast<float*>(dref.ptr), m, n, k, 1.0f, 0.0f,
                       K::GemmVariant::kCublasRef, K::GemmTile{},
                       stream->native()).throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(href.data(), dref.ptr, c_bytes, cudaMemcpyDeviceToHost));
    have_oracle = true;

    const auto sc = testing::spot_check_gemm(ha.data(), hb.data(), href.data(), m, n, k,
                                             1.0f, 1024, 0x5A11Eull,
                                             testing::tol_gemm(k),
                                             testing::abs_tol_gemm(k, 1.0));
    std::printf("spot-check    cublas   %zu/%zu elements OK   max_abs %.3e  max_rel %.3e\n",
                sc.checked - sc.mismatches, sc.checked, sc.max_abs_err, sc.max_rel_err);
    if (!sc.ok()) { std::printf("   *** cuBLAS FAILED its own spot-check ***\n"); ++failures; }
  }

  const double rel_big = testing::tol_gemm(k);
  const double abs_big = testing::abs_tol_gemm(k, 1.0);
  for (int i = 0; i < kLadderN - 1; ++i) {
    if (!selected[i] || !implemented[i]) continue;
    MCKE_CUDA_CHECK(cudaMemsetAsync(dc.ptr, 0, c_bytes, stream->native()));
    run(kLadder[i], m, n, k, 1.0f, 0.0f, dC).throw_if_error();
    stream->synchronize().throw_if_error();
    MCKE_CUDA_CHECK(cudaMemcpy(hc.data(), dc.ptr, c_bytes, cudaMemcpyDeviceToHost));
    if (have_oracle) {
      const auto r = testing::compare(hc.data(), href.data(), cn, rel_big, abs_big);
      benchcfg::print_validation(kLadder[i].name, r.ok(), r.max_rel_err, rel_big,
                                 "vs cuBLAS at the benchmark shape");
      if (!r.ok()) { std::printf("   %s\n", r.to_string().c_str()); ++failures; }
    } else {
      const auto sc = testing::spot_check_gemm(ha.data(), hb.data(), hc.data(), m, n, k,
                                               1.0f, 1024, 0x5A11Eull, rel_big, abs_big);
      std::printf("spot-check    %-18s %zu/%zu OK   max_rel %.3e\n", kLadder[i].name,
                  sc.checked - sc.mismatches, sc.checked, sc.max_rel_err);
      if (!sc.ok()) ++failures;
    }
  }
  if (failures)
    std::printf("\n*** %d VALIDATION FAILURE(S) -- the timings below are meaningless ***\n",
                failures);
  std::printf("\n");

  // ---------------------------------------------------------------------------
  // Occupancy: three legs.
  //
  // docs/ROADMAP.md asks for two -- hand-computed vs Nsight-measured. The middle
  // leg (cudaOccupancyMaxActiveBlocksPerMultiprocessor) is free and separates two
  // failure modes the two-leg version conflates: "my arithmetic is wrong" and
  // "the hardware is not achieving theoretical". Those need different fixes.
  //
  // regs/thread and smem/block come from cudaFuncGetAttributes on the SAME
  // kernel pointer that gets timed below, so these columns cannot describe a
  // stale build -- which is what parsing -Xptxas -v output from a previous
  // compile could silently do.
  // ---------------------------------------------------------------------------
  std::printf("=== occupancy: hand-computed vs CUDA occupancy API =========\n");
  std::printf("%-18s %5s %6s %7s %8s %8s %8s  %s\n", "variant", "regs", "smem",
              "spill", "hand", "api", "occ%", "binding limiter");
  for (int i = 0; i < kLadderN; ++i) {
    if (!selected[i] || !implemented[i]) continue;
    auto at = K::gemm_kernel_attrs(kLadder[i].variant, kLadder[i].tile);
    if (!at.ok()) { std::printf("%-18s  (no attributes: %s)\n", kLadder[i].name,
                                at.status().message().c_str()); continue; }
    const auto occ = K::occupancy_blocks_per_sm(*dev, at->threads_per_block,
                                                at->regs_per_thread,
                                                at->static_smem_bytes);
    std::printf("%-18s %5d %6zu %7zu %8d %8d %7.1f%%  %s%s\n",
                kLadder[i].name, at->regs_per_thread, at->static_smem_bytes,
                at->local_bytes, occ.blocks_per_sm, at->max_blocks_per_sm_api,
                occ.occupancy * 100.0, limiter_str(occ.limiter),
                occ.tied ? " (tied)" : "");
    if (occ.blocks_per_sm != at->max_blocks_per_sm_api)
      std::printf("      ^ HAND CALC DISAGREES WITH THE API. That is data, not a "
                  "nuisance: see the\n        allocation-granularity constants in "
                  "gemm_tile.hpp -- kSmemAllocUnit first.\n");
    if (at->local_bytes > 0)
      std::printf("      ^ REGISTER SPILLING (%zu B/thread of local memory). The "
                  "accumulator is not\n        in registers; local memory is DRAM. "
                  "This defeats register blocking entirely.\n", at->local_bytes);
  }
  std::printf("\n");

  // ---------------------------------------------------------------------------
  // Timing.
  //
  // cuBLAS runs FIRST and LAST. A passively-cooled 70 W T4 running minutes of
  // back-to-back GEMMs drops off boost clock, and that drift lands on whichever
  // rows come later -- flattering the naive rows and penalising the fast ones,
  // i.e. biased in exactly the direction that would make the ladder look better
  // than it is. Two cuBLAS measurements bracketing the run turn an invisible
  // systematic error into a printed number.
  // ---------------------------------------------------------------------------
  Profiler prof;
  auto time_one = [&](const Row& r, const char* label) {
    auto rec = prof.time_op("gemm", label, *stream, flops, bytes, args.warmup,
                            args.iters, [&](const rt::Stream& s) {
                              return K::launch_gemm_f32(dA, dB, dC, m, n, k, 1.0f, 0.0f,
                                                        r.variant, r.tile, s.native());
                            });
    rec.status().throw_if_error();
    return rec->median_ms;
  };

  const Row& cublas_row = kLadder[kLadderN - 1];
  double cublas_first = 0.0, cublas_last = 0.0;
  const bool do_cublas = selected[kLadderN - 1] && implemented[kLadderN - 1];
  if (do_cublas) cublas_first = time_one(cublas_row, "cublas");

  for (int i = 0; i < kLadderN - 1; ++i) {
    if (!selected[i]) continue;
    if (!implemented[i]) {
      std::printf("skipped       %-18s launcher returned an error for this build\n",
                  kLadder[i].name);
      continue;
    }
    if (i <= 1) {
      // ~0.5-7 s per launch at 4096^3; 25 launches is minutes for these two rows
      // alone. Say so, or a silent multi-minute stall looks like a hang.
      std::printf("running       %-18s (SLOW: expect %s minutes, ~%d launches)\n",
                  kLadder[i].name, i == 0 ? "2-3" : "under 1", args.warmup + args.iters);
      std::fflush(stdout);
    }
    time_one(kLadder[i], kLadder[i].name);
  }
  if (do_cublas) cublas_last = time_one(cublas_row, "cublas_drift_recheck");

  std::printf("\n%s\n", prof.summary_table(rl).c_str());

  // ---------------------------------------------------------------------------
  // Analysis. Compute the comparisons here rather than leaving them to a reader
  // with a calculator -- the same convention the other three Phase-3 benches use.
  // ---------------------------------------------------------------------------
  auto find = [&](const char* v) -> const KernelRecord* {
    for (const auto& r : prof.records()) if (r.variant == v) return &r;
    return nullptr;
  };

  if (do_cublas && cublas_first > 0.0) {
    const double drift = (cublas_last - cublas_first) / cublas_first * 100.0;
    std::printf("=== thermal drift check ====================================\n");
    std::printf("cublas first %.3f ms, last %.3f ms -> %+.1f%%  %s\n",
                cublas_first, cublas_last, drift,
                (drift > 3.0 || drift < -3.0)
                    ? "*** >3%: THE WHOLE TABLE IS DRIFTING, clocks moved during the run ***"
                    : "within 3%: clocks held, cross-row comparisons are sound");
  }

  std::printf("\n=== modelled DRAM traffic (a MODEL, not a measurement) =====\n");
  std::printf("%-18s %14s %12s   %s\n", "variant", "model bytes", "vs ideal", "note");
  for (int i = 0; i < kLadderN - 1; ++i) {
    if (!selected[i] || !implemented[i]) continue;
    const double mb = modelled_read_bytes(kLadder[i], m, n, k);
    std::printf("%-18s %14.3g %11.1fx\n", kLadder[i].name, mb, mb / double(bytes));
  }
  std::printf("This is the reuse story docs/PROFILING.md calls the most instructive\n"
              "number in the project, and `ncu --metrics dram__bytes_read.sum` is what\n"
              "turns it from a model into a measurement. Predict in advance: the naive\n"
              "model implies %.0f GB in ~%.0f ms = ~%.0f GB/s, which is FOUR TIMES the\n"
              "T4's DRAM peak -- physically impossible, so L1/L2 must already be\n"
              "absorbing most of it. That impossibility is the prediction being tested.\n",
              modelled_read_bytes(kLadder[0], m, n, k) / 1e9,
              find("naive") ? find("naive")->median_ms : 0.0,
              find("naive") ? modelled_read_bytes(kLadder[1], m, n, k) / 1e9 /
                                  (find("naive")->median_ms / 1e3) : 0.0);

  std::printf("\n=== step-by-step attribution ==============================\n");
  struct Step { const char* from; const char* to; const char* cause; };
  const Step kSteps[] = {
      {"naive_uncoalesced", "naive",             "thread->output transpose (coalescing)"},
      {"naive",             "tiled_smem",        "shared-memory staging"},
      {"tiled_smem",        "tiled_regblock",    "register blocking"},
      {"tiled_regblock",    "warptile_nodbuf",   "lane->output permutation"},
      {"warptile_nodbuf",   "warptile_dbuf",     "double-buffered smem"},
      {"warptile_dbuf",     "warptile_vec4",     "float4 global->smem loads"},
      {"warptile_vec4",     "cublas",            "THE REMAINING GAP"},
  };
  for (const Step& s : kSteps) {
    const KernelRecord* f = find(s.from);
    const KernelRecord* t = find(s.to);
    if (!f || !t) continue;
    std::printf("%-18s -> %-18s %6.2fx   %5.1f%% -> %5.1f%% of peak   [%s]\n",
                s.from, s.to, f->median_ms / t->median_ms,
                rl.efficiency(*f) * 100.0, rl.efficiency(*t) * 100.0, s.cause);
  }

  const Status csv = prof.write_csv("phase3_gemm.csv", rl);
  std::printf("\n%s\n", csv.ok() ? "wrote phase3_gemm.csv" : csv.to_string().c_str());

  alloc.deallocate(da, stream->native()).throw_if_error();
  alloc.deallocate(db, stream->native()).throw_if_error();
  alloc.deallocate(dc, stream->native()).throw_if_error();
  alloc.deallocate(dref, stream->native()).throw_if_error();
  return failures ? 1 : 0;
}
