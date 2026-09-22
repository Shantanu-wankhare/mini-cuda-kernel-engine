// =============================================================================
//  bench/alloc_bench_gb.cpp
//
//  WHAT: Google Benchmark microbenchmarks for the steady-state allocate/free
//        round trip of all three allocators, plus a self-check that verifies
//        gb_adapter.hpp's milliseconds -> seconds conversion.
//
//  WHY A NEW FILE rather than converting bench/alloc_bench.cpp:
//  alloc_bench.cpp's `int main()` with no argv is a deliberate convention, and
//  its stdout is transcribed into RESULTS.md section 2a/2b/2c. Converting it
//  would invalidate published rows for no gain. This file sits ALONGSIDE it and
//  measures a DIFFERENT thing -- see "what this measures" below.
//
//  WHY .cpp: plain host C++20, no device code, no __global__, no <<<>>>. It must
//  compile with clang++ on macOS, which is the entire point of it (see below).
//
//  ---------------------------------------------------------------------------
//  WHY THIS FILE EXISTS AT ALL -- it is the Phase 5 safety net
//
//  Stage 5b's failure modes are all SILENT (see gb_adapter.hpp's banner): a
//  missing `* 1e-3` reports every number 1000x too large; a missing
//  ->UseManualTime() reports launch latency as kernel time. Neither throws.
//
//  This is the ONLY Google Benchmark target in the project that builds and runs
//  on a machine with NO GPU. So it is where those guards get tested -- on the
//  MacBook, in a second, for free -- instead of on Colab, where a wrong number
//  costs a session and may not even look wrong. Hence bm_unit_conversion_*
//  below, which deliberately feeds a KNOWN duration through the adapter.
//
//  ---------------------------------------------------------------------------
//  WHAT THIS MEASURES, AND WHAT IT HONESTLY CANNOT
//
//  Measures: the steady-state cost of one allocate + one deallocate of a fixed
//  size, per allocator, with the block returning to the pool each time. This is
//  the case GB's "repeat identical work N times" model actually fits, and it is
//  where GB is a genuine upgrade over our own harness -- nanosecond-scale host
//  code needs DoNotOptimize and iteration auto-tuning, both of which GB has and
//  HostTimer/LatencyStats do not.
//
//  Does NOT measure, and must not be read as measuring:
//   * alloc_bench.cpp's TRACE REPLAY. That is a one-shot, path-dependent
//     sequence whose whole subject is how state evolves; "run it N times and
//     average" is meaningless for it. It stays where it is.
//   * TAIL LATENCY. GB reports mean/median/stddev/cv over repetitions. The
//     p99/p999/max that RESULTS.md section 2a publishes come from
//     LatencyStats's nearest-rank percentiles over individual operations, which
//     GB cannot produce. alloc_bench.cpp remains the source of truth there.
//   * THE REUSE POLICIES' ACTUAL PROBING COST. In a host-only build
//     rt::stream_query and rt::event_query return true UNCONDITIONALLY
//     (stream.hpp:85, :111) -- there is no async work to probe. So the policy
//     dimension below measures only the differing BOOKKEEPING (a push_back vs.
//     an event record that compiles to a no-op), not the driver-call cost that
//     makes the three differ on real hardware. The policy comparison is only
//     meaningful in a CUDA build. Labelled in the output accordingly.
//
//  ---------------------------------------------------------------------------
//  TWO MEASURED RESULTS THAT LOOK WRONG AND ARE NOT  (macOS host, 2026-09-22)
//
//  (a) "raw" BEATS both pools at 4 KiB: raw 34.5 ns vs buddy 84.6 ns vs
//      freelist 48.8 ns. That appears to refute Phase 2's entire thesis. It does
//      not, and the reason is that THIS IS A HOST-ONLY BUILD: with
//      MCKE_WITH_CUDA=0, RawDeviceAllocator is not calling cudaMalloc at all --
//      it is calling the system allocator, which is a fast user-space free-list
//      that macOS additionally caches. The pools exist to amortise cudaMalloc's
//      ~10-100 us driver round trip, and in this build there is no cudaMalloc to
//      amortise, so "raw" here is a different experiment wearing the same name.
//      Read these rows as "what does our bookkeeping cost on top of malloc",
//      never as "pools are slower than the driver". RESULTS.md section 2a's
//      raw-vs-pool numbers come from a CUDA build and are the real comparison.
//
//  (b) buddy gets FASTER as the block gets BIGGER: 84.6 ns at 4 KiB, 47.4 ns at
//      1 MiB, 31.3 ns at 8 MiB -- the opposite of the usual intuition. This one
//      is real, and it is buddy-allocator structure showing through: a request
//      is served by splitting the smallest adequate free block down to size, so
//      a 4 KiB allocation out of a 16 MiB slab walks log2(16Mi/4Ki) = 12 levels
//      of split on the way down and 12 levels of buddy-merge on the way back up,
//      while an 8 MiB allocation walks exactly one. Cost tracks the number of
//      LEVELS TRAVERSED, not the number of bytes -- nothing here touches the
//      memory it hands out.
//
//      Tested rather than asserted. Slab is 16 MiB = 2^24, so a request of 2^k
//      costs 24-k splits: 4 KiB -> 12, 1 MiB -> 4, 8 MiB -> 1. Fitting a line
//      through ONLY the two extremes (4 KiB and 8 MiB) gives
//          cost ~= 26.5 ns + 4.85 ns per level
//      which predicts the HELD-OUT middle point (1 MiB, 4 levels) at 45.8 ns
//      against a measured 47.4 ns -- 3.3% error. A bytes-based model cannot
//      produce that ordering at all, since it has cost increasing with size.
//
//      freelist is flat by comparison (48.8 ns at 4 KiB, 49.1 ns at 1 MiB),
//      which is exactly what a segregated size-class design predicts: O(1)
//      lookup regardless of size. So the two allocators' cost CURVES differ in
//      shape, not just in height -- buddy wins at large blocks, freelist wins
//      at small ones, and they cross near 1 MiB.
// =============================================================================

#include <benchmark/benchmark.h>

#include <cstdio>
#include <memory>
#include <string>

#include "gb_adapter.hpp"

#include "mcke/memory/allocator.hpp"
#include "mcke/memory/buddy_allocator.hpp"
#include "mcke/memory/freelist_allocator.hpp"
#include "mcke/memory/reuse_policy.hpp"
#include "mcke/runtime/stream.hpp"

namespace {

using namespace mcke;                  // NOLINT -- bench-local, matches the other benches

// Matched to bench/alloc_bench.cpp's constants so the two files are describing
// the same allocator configuration and a reader can compare them directly.
constexpr std::size_t kSlabBytes           = std::size_t{16} << 20;   // 16 MiB
constexpr std::size_t kLargeAllocThreshold = std::size_t{32} << 20;

// -----------------------------------------------------------------------------
// 1. THE SELF-CHECK. No allocator, no GPU, no timing -- pure verification that
//    gb_adapter.hpp's unit conversion is right.
//
// We hand set_iteration_time_from_ms a KNOWN duration every iteration. Under
// ->UseManualTime() GB reports Time = accumulated_manual_time / iterations, so
// feeding `ms` must print exactly `ms`. The MCKE_GB_GPU macro sets the display
// unit to microseconds, so bm_unit_conversion/2ms must read 2000 us.
//
// This is a REGRESSION TEST WEARING A BENCHMARK'S CLOTHES. If someone deletes
// the `* 1e-3`, this row reads 2000000 us instead of 2000 -- a 1000x jump that
// is impossible to miss, on a machine with no GPU, in under a second.
// Deliberately verified by breaking it once; see PROJECT_LOG.md.
// -----------------------------------------------------------------------------
void bm_unit_conversion(benchmark::State& state, double ms) {
  for (auto _ : state) {
    mcke::benchgb::set_iteration_time_from_ms(state, ms);
  }
  state.SetLabel("self-check: must read exactly " + std::to_string(ms) +
                 " ms; no GPU involved");
}

// Iterations pinned: there is nothing to auto-tune here, and a fixed count keeps
// the expected output byte-identical between runs and between machines.
MCKE_GB_GPU(bm_unit_conversion, 2ms, 2.0)->Iterations(16);
MCKE_GB_GPU(bm_unit_conversion, 0_5ms, 0.5)->Iterations(16);

// -----------------------------------------------------------------------------
// 2. ALLOCATOR MICROBENCHMARKS
// -----------------------------------------------------------------------------
enum class Which { kRaw, kBuddy, kFreeList };

// Built fresh per benchmark (not per iteration) and reserved up front, so the
// timed loop measures steady-state allocate/free and not first-touch slab
// reservation. Returned as a unique_ptr to the base so the timed loop is
// identical across all three -- a virtual call is part of the real cost anyway,
// since DeviceAllocator is used polymorphically by the runtime.
std::unique_ptr<DeviceAllocator> make_allocator(Which w, ReusePolicy pol, bool* ok) {
  *ok = true;
  switch (w) {
    case Which::kRaw:
      // No reserve(), no policy: RawDeviceAllocator has no pool and no parking
      // concept. It is the baseline the other two must beat.
      return std::make_unique<RawDeviceAllocator>();

    case Which::kBuddy: {
      BuddyConfig c;
      c.initial_slab_bytes    = kSlabBytes;
      c.min_block_bytes       = 256;
      c.large_alloc_threshold = kLargeAllocThreshold;
      c.reuse_policy          = pol;
      auto a = std::make_unique<BuddyAllocator>(c);
      *ok = a->reserve(kSlabBytes).ok();
      return a;
    }

    case Which::kFreeList: {
      FreeListConfig c;
      c.slab_bytes            = kSlabBytes;
      c.large_alloc_threshold = kLargeAllocThreshold;
      c.reuse_policy          = pol;
      auto a = std::make_unique<FreeListAllocator>(c);
      *ok = a->reserve(kSlabBytes).ok();
      return a;
    }
  }
  *ok = false;
  return nullptr;
}

const char* policy_name(ReusePolicy p) {
  switch (p) {
    case ReusePolicy::kSameStreamOnly:   return "same_stream_only";
    case ReusePolicy::kCoarseStreamPoll: return "coarse_stream_poll";
    case ReusePolicy::kPerFreeEvent:     return "per_free_event";
  }
  return "?";
}

// One allocate + one deallocate per GB iteration.
//
// NOTE ON TIMING MODE: this benchmark does NOT use manual time, and that is
// correct rather than an inconsistency with gb_adapter.hpp's GPU paths. The work
// here is synchronous HOST code, so GB's native wall-clock loop measures exactly
// the right thing. Manual time exists solely because a CUDA launch is async;
// there is no async here to misattribute.
void bm_alloc_free(benchmark::State& state, Which w, std::size_t bytes, ReusePolicy pol) {
  bool ok = false;
  auto alloc = make_allocator(w, pol, &ok);
  if (!ok || !alloc) {
    state.SkipWithError("allocator construction/reserve failed");
    return;
  }
  const rt::StreamHandle stream{};   // host-only: a null handle, never dereferenced

  for (auto _ : state) {
    auto a = alloc->allocate(bytes, stream);
    if (!a.ok()) {
      state.SkipWithError("allocate failed: " + a.status().to_string());
      return;
    }
    // Without this the compiler may observe that nothing reads the pointer and
    // sink or delete the whole round trip. At ~50 ns per iteration that is not
    // a hypothetical -- it is the difference between measuring the allocator and
    // measuring an empty loop. This is the single clearest thing GB gives us
    // that HostTimer does not.
    benchmark::DoNotOptimize(a->ptr);
    if (const Status st = alloc->deallocate(*a, stream); !st.ok()) {
      state.SkipWithError("deallocate failed: " + st.to_string());
      return;
    }
  }

  // No `bytes` counter here on purpose: the size is already in the benchmark
  // name, and GB aggregates every counter across repetitions -- so a constant
  // renders as `bytes=0` on the _stddev row and `bytes=0.00%` on the _cv row,
  // which reads like a measurement and is not one.
  state.SetLabel(std::string(alloc->name()) + " / " + policy_name(pol) +
                 (w == Which::kRaw ? " (policy N/A)"
                                   : " (host build: probing is a no-op)"));
}

// Registration. BENCHMARK_CAPTURE, not MCKE_GB_GPU: these are host benchmarks
// and must use GB's native timing, so ->UseManualTime() would be wrong here --
// it would report ~0 forever and hang the auto-tuner.
//
// Repetitions(5) rather than the default 1: the whole reason Phase 5 adopts GB
// is its run-to-run statistics, and a _stddev/_cv row needs repetitions to
// exist at all. DisplayAggregatesOnly keeps the output readable.
#define MCKE_GB_HOST(name_, ...)                                               \
  BENCHMARK_CAPTURE(bm_alloc_free, name_, __VA_ARGS__)                         \
      ->Repetitions(5)                                                         \
      ->DisplayAggregatesOnly(true)                                            \
      ->Unit(benchmark::kNanosecond)

// Three size classes spanning the interesting regimes:
//   4 KiB   -- small class, below freelist's small/large split (1 MiB)
//   1 MiB   -- exactly at the split
//   8 MiB   -- large class, still well under the 32 MiB bypass threshold
MCKE_GB_HOST(raw_4KiB,  Which::kRaw, std::size_t{4} << 10, ReusePolicy::kCoarseStreamPoll);
MCKE_GB_HOST(raw_1MiB,  Which::kRaw, std::size_t{1} << 20, ReusePolicy::kCoarseStreamPoll);
MCKE_GB_HOST(raw_8MiB,  Which::kRaw, std::size_t{8} << 20, ReusePolicy::kCoarseStreamPoll);

MCKE_GB_HOST(buddy_4KiB_same,   Which::kBuddy, std::size_t{4} << 10, ReusePolicy::kSameStreamOnly);
MCKE_GB_HOST(buddy_4KiB_poll,   Which::kBuddy, std::size_t{4} << 10, ReusePolicy::kCoarseStreamPoll);
MCKE_GB_HOST(buddy_4KiB_event,  Which::kBuddy, std::size_t{4} << 10, ReusePolicy::kPerFreeEvent);
MCKE_GB_HOST(buddy_1MiB_poll,   Which::kBuddy, std::size_t{1} << 20, ReusePolicy::kCoarseStreamPoll);
MCKE_GB_HOST(buddy_8MiB_poll,   Which::kBuddy, std::size_t{8} << 20, ReusePolicy::kCoarseStreamPoll);

MCKE_GB_HOST(freelist_4KiB_same,  Which::kFreeList, std::size_t{4} << 10, ReusePolicy::kSameStreamOnly);
MCKE_GB_HOST(freelist_4KiB_poll,  Which::kFreeList, std::size_t{4} << 10, ReusePolicy::kCoarseStreamPoll);
MCKE_GB_HOST(freelist_4KiB_event, Which::kFreeList, std::size_t{4} << 10, ReusePolicy::kPerFreeEvent);
MCKE_GB_HOST(freelist_1MiB_poll,  Which::kFreeList, std::size_t{1} << 20, ReusePolicy::kCoarseStreamPoll);
MCKE_GB_HOST(freelist_8MiB_poll,  Which::kFreeList, std::size_t{8} << 20, ReusePolicy::kCoarseStreamPoll);

}  // namespace

// -----------------------------------------------------------------------------
// main(). The ORDERING here is the whole point -- see gb_adapter.hpp's
// snapshot_cmdline comment. Reproduced in every *_gb bench.
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  // 1. Snapshot BEFORE Initialize, which is about to delete the --benchmark_*
  //    flags from argv. Echoing a post-Initialize argv would record a command
  //    line that does not reproduce the run (RESULTS.md rule 2).
  const std::string cmdline = mcke::benchgb::snapshot_cmdline(argc, argv);

  // 2. GB first. It consumes ONLY --benchmark_*, compacts argv, updates argc,
  //    and leaves our own flags alone.
  benchmark::Initialize(&argc, argv);

  // 3. NOT benchmark::ReportUnrecognizedArguments(argc, argv). In the other
  //    benches OUR flags (--only=, --m=, --peak-gb-s=) are the ones it would
  //    call unrecognised, so it would reject every real invocation. This file
  //    has no flags of its own, but the omission is deliberate and uniform.

  // 4. Context into the JSON artifact, not just stdout: our CSV has no context
  //    block, which is exactly how a CSV becomes uninterpretable six months on.
  benchmark::AddCustomContext("mcke_cmdline", cmdline);
  benchmark::AddCustomContext("mcke_timing",
                              "host wall-clock (GB native) for allocators; "
                              "manual-time self-check for the unit conversion");
  benchmark::AddCustomContext("mcke_with_cuda", MCKE_WITH_CUDA ? "1" : "0");
  benchmark::AddCustomContext(
      "mcke_caveat",
      "Reuse-policy rows are NOT a policy comparison in a host build: "
      "rt::stream_query/event_query are unconditionally true with no GPU, so "
      "only bookkeeping differs. Tail latency (p99/p999) comes from "
      "bench/alloc_bench.cpp, not from here.");

  // STDERR, not stdout. --benchmark_format=json writes the JSON document to
  // STDOUT, so a banner printed there makes the artifact unparseable -- and it
  // fails at the consumer, not here, which is the worst place to find it.
  // (Caught exactly that way during stage 5b: `--benchmark_format=json | python
  // -m json.tool` died on "Expecting value: line 1 column 1".) Stage 5f's
  // regen_results.sh parses this JSON, so stdout must stay machine-clean.
  std::fprintf(stderr, "# mcke_alloc_bench_gb -- host-only, no GPU required\n");
  std::fprintf(stderr, "# command: %s\n", cmdline.c_str());
  std::fprintf(stderr,
               "# NOTE: GB reports mean/median/stddev over REPETITIONS, of\n"
               "#       per-repetition means. For per-operation tail latency\n"
               "#       (p99/p999/max) use mcke_alloc_bench. See RESULTS.md 2a.\n");

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
