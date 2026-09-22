// =============================================================================
//  bench/gb_adapter.hpp
//
//  WHAT: the seam between Google Benchmark's iteration loop and CUDA-event
//        timing. NOTHING here measures anything new -- mcke::Profiler still does
//        all the measuring. This file exists so that:
//          (a) the milliseconds -> seconds conversion happens in exactly ONE
//              named place, where it can be tested;
//          (b) `->UseManualTime()` cannot be forgotten;
//          (c) the two timing MODES below are named and distinguishable in the
//              output, rather than silently interchangeable.
//
//  WHY .hpp, and why in bench/: bench-only host code, never shipped in the
//  runtime, so it does not belong under include/mcke/. Header-only because the
//  timed callable must inline -- the same argument profiler.hpp gives for
//  Profiler::time_op being a template rather than taking a std::function: an
//  indirect call inside the timed region is measurable for a 5 us kernel.
//  Peer of bench_common.hpp (bench-only setup), NOT of tests/reference.hpp
//  (correctness-only). Different jobs, different lifetimes.
//
//  ---------------------------------------------------------------------------
//  WHAT GOOGLE BENCHMARK IS FOR HERE, AND WHAT IT IS NOT  (DECISIONS.md Q7)
//
//  GB is adopted as a HARNESS and a REGRESSION-COMPARISON TOOL. It is NOT a
//  measurement tool in this project, and the distinction is not pedantry:
//
//   1. GB's default timing loop uses host wall-clock. A CUDA launch is async and
//      returns before the kernel runs, so host wall-clock measures LAUNCH
//      LATENCY. A 2 ms GEMM reads as roughly 8 us. Hence UseManualTime below.
//   2. GB's statistics (_mean/_median/_stddev/_cv) are computed over
//      REPETITIONS, of PER-REPETITION MEANS. Within one repetition, every
//      iteration is collapsed into a single mean before GB ever sees it. So the
//      per-launch distribution -- which profiler.hpp's header calls the SIGNAL
//      ("variance is the signal that tells you the GPU is throttling") -- is
//      averaged away. GB's "median" is a median OF MEANS.
//   3. GB has no `min` statistic at all. RESULTS.md rule 3 publishes median AND
//      min over >= 20 timed iterations. GB structurally cannot produce that min.
//
//  So: GB is STRICTLY LESS INFORMATIVE about kernel timing than
//  Profiler::time_op already is. What it genuinely adds, and what we adopt it
//  for, is run-to-run variance across repetitions (which time_op has no notion
//  of -- it reports within-one-process statistics only), a self-describing JSON
//  artifact with a context block, and GB's tools/compare.py Mann-Whitney U test
//  for commit-to-commit regressions.
//
//  ---------------------------------------------------------------------------
//  THE TWO MODES, AND WHY BOTH EXIST
//
//  kBurst   (DEFAULT) one GB iteration == one full Profiler::time_op burst
//           (warmup + N timed iters + ONE stream sync at the end).
//           SetIterationTime(median_ms * 1e-3).
//           The physics are IDENTICAL to every number already in RESULTS.md,
//           because it is literally the same code path. Use this for anything
//           whose number goes into RESULTS.md.
//
//  kPerIter one GB iteration == one launch, event-bracketed, with a HOST SYNC
//           PER ITERATION. This is GB's native model, and it CHANGES WHAT IS
//           MEASURED:
//             - profiler.hpp step 3 forbids exactly this ("a sync per iteration
//               would insert a host round-trip (~10 us) between kernels and
//               inflate short kernels enormously");
//             - the GPU idles between iterations, so clock/power state and L2
//               residency differ from back-to-back execution;
//             - launch pipelining is gone, so any launch-bound result --
//               graph_bench's entire subject -- is destroyed.
//           It is here ON PURPOSE: it is the instrument for the Phase 5
//           cross-check that quantifies what that sync costs as a function of
//           kernel duration. The delta is a deliverable, not an accident.
// =============================================================================
#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "mcke/core/status.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/stream.hpp"

namespace mcke::benchgb {

// -----------------------------------------------------------------------------
// THE unit conversion. This function is the reason this file exists.
//
// benchmark::State::SetIterationTime takes SECONDS, as a double.
// rt::Event::elapsed_ms (and therefore KernelRecord::median_ms) is in
// MILLISECONDS. The `* 1e-3` is the single most consequential expression in
// this header: omit it and every reported number is 1000x too large, with no
// error and no warning -- and GB's iteration auto-tuner then believes
// --benchmark_min_time was satisfied after one iteration, so it will not even
// look slow.
//
// It is a named function rather than an inline `* 1e-3` at each call site for
// one concrete reason: in a host-only build (MCKE_WITH_CUDA=0)
// rt::Event::elapsed_ms returns 0.0f unconditionally, so the GPU paths below
// cannot exercise this conversion on a machine with no GPU. Pulling it out
// makes it callable with a KNOWN value, which is what lets
// bench/alloc_bench_gb.cpp verify the 1000x guard on the MacBook instead of
// discovering it on Colab. A guard is only real once it has been seen to fail.
// -----------------------------------------------------------------------------
inline void set_iteration_time_from_ms(benchmark::State& state, double ms) {
  state.SetIterationTime(ms * 1e-3);   // ms -> s. See above before touching.
}

// -----------------------------------------------------------------------------
// Derived metrics -> GB counters.
//
// CONSISTENCY RULE, load-bearing: `flops`/`bytes` must be the ideal counts for
// ONE UNIT OF WHATEVER set_iteration_time_from_ms REPORTED. In kBurst mode that
// is one INNER iteration, not the whole burst. The algebra then cancels:
//
//   kIsIterationInvariantRate = value * iterations / total_manual_time
//                             = bytes * iters / (iters * median_seconds)
//                             = bytes / median_seconds                (correct)
//
// WHY NOT state.SetBytesProcessed(), which is the obvious convenience: it
// installs its counter with Counter::kIs1024 (verified in GB v1.9.5,
// benchmark.h:892). So it prints 219.2Gi/s where KernelRecord::gb_per_s() --
// which divides by 1e9 -- says 235.4. That is a silent 7.4% disagreement
// between GB's output and our CSV, on precisely the number RESULTS.md rule 5
// exists to pin down. Explicit Counters with kIs1000, always.
//
// ONE DIVERGENCE WE DOCUMENT RATHER THAN FIX: GB's rate counters divide by
// total manual time, i.e. by the MEAN; KernelRecord::gb_per_s() divides by the
// MEDIAN. On a right-skewed distribution mean > median, so GB's GB/s is
// systematically LOWER than our CSV's for identical data. That is arithmetic,
// not a bug -- but the two must never share a RESULTS.md table without a stated
// timing method.
// -----------------------------------------------------------------------------
inline void set_roofline_counters(benchmark::State& state, const Roofline& rl,
                                  std::uint64_t flops, std::uint64_t bytes) {
  using C = benchmark::Counter;
  const double ai = bytes ? static_cast<double>(flops) / static_cast<double>(bytes) : 0.0;

  state.counters["GB_s"] =
      C(static_cast<double>(bytes), C::kIsIterationInvariantRate, C::kIs1000);
  // Pre-divided by 1e12 so the printed figure reads in TFLOP/s rather than as
  // GB's own "1.37T" unit-prefixed rendering of raw FLOP/s.
  state.counters["TFLOP_s"] =
      C(static_cast<double>(flops) / 1e12, C::kIsIterationInvariantRate, C::kIs1000);
  // AI is a shape-derived constant, not a rate and not per-iteration: kDefaults
  // reports it verbatim. Any rate flag here would divide it by elapsed time.
  state.counters["AI"] = C(ai, C::kDefaults);
  // The denominator itself, printed alongside, so a JSON artifact read six
  // months from now is self-describing -- RESULTS.md rule 5 structurally rather
  // than by remembering.
  state.counters["attainable_TFLOP_s"] = C(rl.attainable_tflops(ai), C::kDefaults);
}

// -----------------------------------------------------------------------------
// MODE A (kBurst) -- the default. One GB iteration == one Profiler::time_op
// burst, so the measurement is bit-for-bit the same code path that produced
// every existing RESULTS.md row.
//
// WHY THE Profiler IS FUNCTION-LOCAL, and why that is not an oversight:
// Profiler has no clear(), and time_op APPENDS to its internal records_ on
// every call (profiler.hpp:263). A Profiler hoisted outside this loop would
// accumulate (repetitions x variants) duplicate records, and a later
// summary_table()/write_csv() would emit every repetition as though it were a
// separate measurement -- silently inflating the CSV. Function-local costs
// nothing (it is a vector that stays empty but for one entry) and needs ZERO
// changes to profiler.hpp. Do not "optimise" this by hoisting it.
//
// The one record worth keeping is handed back through `out` for the caller's
// own CSV, since GB's JSON is an additional artifact, not a replacement.
// -----------------------------------------------------------------------------
template <typename EnqueueFn>
void run_burst(benchmark::State& state, const std::string& name,
               const std::string& variant, const rt::Stream& stream,
               const Roofline& rl, std::uint64_t flops, std::uint64_t bytes,
               int warmup, int inner_iters, KernelRecord* out, EnqueueFn&& fn) {
  KernelRecord last{};
  for (auto _ : state) {
    Profiler local;                       // see the comment block above
    auto rec = local.time_op(name, variant, stream, flops, bytes,
                             warmup, inner_iters, fn);
    if (!rec.ok()) {
      state.SkipWithError(rec.status().to_string());
      return;
    }
    set_iteration_time_from_ms(state, rec->median_ms);
    last = *rec;
  }
  set_roofline_counters(state, rl, flops, bytes);
  // min over the last burst's inner iterations. GB has no min statistic and
  // cannot compute one, so it is carried across as a plain counter -- otherwise
  // a GB-only run could not satisfy RESULTS.md rule 3, which asks for both.
  state.counters["min_ms_last_burst"] =
      benchmark::Counter(last.min_ms, benchmark::Counter::kDefaults);
  state.SetLabel(variant + " [kBurst " + std::to_string(warmup) + "w+" +
                 std::to_string(inner_iters) + "i, one sync per burst]");
  if (out) *out = last;
}

// -----------------------------------------------------------------------------
// MODE B (kPerIter) -- GB's native model: one launch per GB iteration, one host
// sync per GB iteration. DIFFERENT PHYSICS from Mode A; see the banner. Never
// put a number from this mode in a RESULTS.md timing table.
// -----------------------------------------------------------------------------
template <typename EnqueueFn>
void run_per_iteration(benchmark::State& state, const std::string& variant,
                       const rt::Stream& stream, const Roofline& rl,
                       std::uint64_t flops, std::uint64_t bytes,
                       int warmup, EnqueueFn&& fn) {
  // Events are hoisted and RE-RECORDED rather than recreated per iteration:
  // stream.hpp:242 documents that re-recording simply overwrites the previous
  // recording, "which is what lets us reuse one event per graph node across
  // repeated executions". Creating a pair inside the loop would add two driver
  // calls to every iteration, i.e. would measure our own bookkeeping.
  auto s = rt::Event::create(rt::Event::Purpose::kTiming);
  if (!s.ok()) { state.SkipWithError(s.status().to_string()); return; }
  auto e = rt::Event::create(rt::Event::Purpose::kTiming);
  if (!e.ok()) { state.SkipWithError(e.status().to_string()); return; }

  // Warmup stays OURS and stays ITERATION-COUNTED. GB's
  // --benchmark_min_warmup_time is time-based, defaults to 0.0 (off), and knows
  // nothing about module load or clock ramp -- it cannot express RESULTS.md
  // rule 3's ">= 5 warmup iterations".
  for (int i = 0; i < warmup; ++i) {
    if (const Status st = fn(stream); !st.ok()) {
      state.SkipWithError(st.to_string());
      return;
    }
  }
  if (const Status st = stream.synchronize(); !st.ok()) {
    state.SkipWithError(st.to_string());
    return;
  }

  for (auto _ : state) {
    if (const Status st = s->record(stream); !st.ok()) { state.SkipWithError(st.to_string()); return; }
    if (const Status st = fn(stream);        !st.ok()) { state.SkipWithError(st.to_string()); return; }
    if (const Status st = e->record(stream); !st.ok()) { state.SkipWithError(st.to_string()); return; }
    // Sync on the STOP EVENT, not the stream: cudaEventSynchronize is the
    // narrower barrier and does not wait on unrelated work queued behind us.
    // It is still a full host round-trip -- which IS the point of this mode.
    if (const Status st = e->synchronize(); !st.ok()) { state.SkipWithError(st.to_string()); return; }
    auto ms = rt::Event::elapsed_ms(*s, *e);
    if (!ms.ok()) { state.SkipWithError(ms.status().to_string()); return; }
    set_iteration_time_from_ms(state, static_cast<double>(*ms));
  }
  set_roofline_counters(state, rl, flops, bytes);
  state.SetLabel(variant + " [kPerIter, HOST SYNC PER ITERATION -- "
                           "NOT comparable to RESULTS.md rows]");
}

// -----------------------------------------------------------------------------
// Registration macro, so ->UseManualTime() cannot be forgotten.
//
// Forgetting it is SILENT AND DIRECTIONAL: SetIterationTime is accumulated and
// then discarded, and GB falls back to host wall-clock -- which for an async
// launch is launch latency, so a 2 ms kernel reports as microseconds and looks
// like a spectacular result. The inverse mistake (UseManualTime with no
// SetIterationTime) is also silent: manual time stays ~0, GB keeps multiplying
// the iteration count chasing --benchmark_min_time, and the process appears to
// hang. Neither is diagnosed by GB. So the macro owns the call.
//
// THE ONE TELL, and it is worth knowing because it is the only one: GB appends
// "/manual_time" to the benchmark's reported NAME when manual timing is active.
// So `bm_x/2ms/iterations:16/manual_time` is wired correctly and
// `bm_x/2ms/iterations:16` is not. Measured on this file during stage 5b: with
// the flag, the 2 ms self-check reports 2000 us; without it, 0.292 us -- a
// ~6850x silent understatement, with that name suffix as the sole visible
// difference. If a GPU kernel ever reports a suspiciously wonderful number,
// check the name suffix before you believe it.
// -----------------------------------------------------------------------------
#define MCKE_GB_GPU(fn_, ...)                                                  \
  BENCHMARK_CAPTURE(fn_, __VA_ARGS__)->UseManualTime()->Unit(benchmark::kMicrosecond)

// -----------------------------------------------------------------------------
// main() helper: snapshot argv BEFORE GB eats part of it.
//
// benchmark::Initialize(&argc, argv) consumes ONLY --benchmark_* flags,
// compacts argv in place (`--(*argc); --i;`, verified in GB v1.9.5
// benchmark.cc) and leaves everything else untouched. That ordering is exactly
// what lets gemm_bench's and graph_bench's fatal-on-unknown-flag parsers keep
// working unchanged -- run them BEFORE Initialize and --benchmark_repetitions=20
// is an "unrecognised argument" and the process exits 2 before GB ever runs.
//
// But it also means that echoing argv AFTER Initialize prints a command line
// which omits the --benchmark_* flags -- i.e. a recorded command that does not
// reproduce the run. RESULTS.md rule 2 asks for "the exact command line", so we
// snapshot first. This is the same failure the fatal parsers exist to prevent,
// and it would otherwise walk straight back in through the front door.
// -----------------------------------------------------------------------------
[[nodiscard]] inline std::string snapshot_cmdline(int argc, char** argv) {
  std::string s;
  for (int i = 0; i < argc; ++i) {
    if (i) s += ' ';
    s += argv[i];
  }
  return s;
}

}  // namespace mcke::benchgb
