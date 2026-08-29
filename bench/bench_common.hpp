// =============================================================================
//  bench/bench_common.hpp
//
//  WHAT: Shared setup for every Phase-3 kernel benchmark — roofline denominators,
//        the environment banner, and the device/stream/allocator boilerplate.
//
//  WHY IT EXISTS — one specific trap, worth spelling out because it fails SILENTLY:
//
//  `Roofline` has two fields, `peak_gb_s` and `peak_tflops`. Every bench written
//  before Phase 3 sets only the first. With `peak_tflops == 0`:
//
//      attainable_tflops(ai) = min(ai * peak_gb_s, 0)  =  0      for every ai
//      efficiency(record)    = tflops / 0              -> guarded, returns 0
//      ridge_point_ai()      = 0 / peak_gb_s           =  0
//      memory_bound(ai)      = ai < 0                  =  false, always
//
//  So `summary_table()` prints a `%peak` column of 0.0% and a `bound` column
//  reading "compute" for a kernel sitting at AI = 0.25 — two confidently wrong
//  answers, no error, no warning. Four benches written by one person means this
//  gets forgotten at least once, so `make_roofline()` sets BOTH fields or does
//  not hand you a Roofline at all. The trap becomes unreachable rather than
//  merely documented.
//
//  WHY .hpp IN bench/: bench-only host code, never shipped. Distinct from
//  tests/reference.hpp, which is correctness-only — different jobs, different
//  lifetimes, no reason to couple them.
// =============================================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mcke/core/device.hpp"
#include "mcke/profiling/profiler.hpp"

namespace mcke::benchcfg {

// -----------------------------------------------------------------------------
// The MEASURED Colab T4 denominators, from RESULTS.md section 0.
//
// NOT spec-sheet figures. bench/stream_triad.cu and bench/fma_peak.cu exist
// precisely to produce these, and docs/PROFILING.md section 2 explains why a
// spec number is the wrong denominator: it ignores ECC overhead, refresh, and
// the clock the chip actually sustains under load, so dividing by it makes every
// real kernel look worse than it can physically be.
//
// The T4 spec bandwidth is 320.1 GB/s; the measured figure is 235.4. Using the
// former would understate every kernel in Phase 3 by 27%.
// -----------------------------------------------------------------------------
inline constexpr double kT4MeasuredGbS    = 235.4;   // bench/stream_triad
inline constexpr double kT4MeasuredTflops = 8.130;   // bench/fma_peak

// Overridable, because a run on Explorer or the RTX 5060 must NOT silently reuse
// T4 denominators — that would produce a "% of peak" against the wrong machine,
// which RESULTS.md rule 5 exists to prevent.
//   --peak-gb-s=<x>  --peak-tflops=<x>   or  MCKE_PEAK_GB_S / MCKE_PEAK_TFLOPS
[[nodiscard]] inline Roofline make_roofline(int argc, char** argv) {
  Roofline rl;
  rl.peak_gb_s   = kT4MeasuredGbS;
  rl.peak_tflops = kT4MeasuredTflops;

  if (const char* e = std::getenv("MCKE_PEAK_GB_S"))   rl.peak_gb_s   = std::atof(e);
  if (const char* e = std::getenv("MCKE_PEAK_TFLOPS")) rl.peak_tflops = std::atof(e);
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--peak-gb-s=", 12) == 0)   rl.peak_gb_s   = std::atof(argv[i] + 12);
    if (std::strncmp(argv[i], "--peak-tflops=", 14) == 0) rl.peak_tflops = std::atof(argv[i] + 14);
  }
  // Both must be positive. A zero here is exactly the silent-garbage case this
  // file exists to prevent, so refuse loudly rather than print a table of lies.
  if (rl.peak_gb_s <= 0.0 || rl.peak_tflops <= 0.0) {
    std::fprintf(stderr,
                 "[mcke] FATAL: roofline denominators must both be > 0 "
                 "(peak_gb_s=%.3f peak_tflops=%.3f).\n"
                 "       With peak_tflops == 0 the %%peak column reads 0%% and the\n"
                 "       bound column reads 'compute' for everything -- silently.\n",
                 rl.peak_gb_s, rl.peak_tflops);
    std::abort();
  }
  return rl;
}

// RESULTS.md rules 1 and 5 (record the environment; state the denominator for
// every % of peak) satisfied structurally, by printing them, rather than by
// remembering to write them down afterwards.
inline void print_denominators(const Roofline& rl, const DeviceInfo& d) {
  std::printf("=== environment ============================================\n");
  std::printf("device        %s (sm_%d%d), %d SMs, %zu KiB smem/SM\n",
              d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count,
              d.shared_mem_per_sm / 1024);
  std::printf("denominators  peak_gb_s=%.1f  peak_tflops=%.3f  (MEASURED, not spec)\n",
              rl.peak_gb_s, rl.peak_tflops);
  std::printf("ridge point   %.1f FLOP/byte -- below this a kernel is memory-bound\n",
              rl.ridge_point_ai());
  std::printf("              spec-formula bandwidth for reference: %.1f GB/s\n",
              d.peak_dram_gb_s());
  std::printf("build         %s\n",
              MCKE_WITH_CUDA ? "MCKE_WITH_CUDA=1" : "MCKE_WITH_CUDA=0 (no kernels!)");
  std::printf("============================================================\n\n");
}

// One place to state which tolerance a bench used and why, so the correctness
// line in the output is self-explaining rather than a bare OK.
inline void print_validation(const char* what, bool ok, double max_rel_err,
                             double tol, const char* why) {
  std::printf("validation    %-28s %s  (max_rel_err %.3g vs tol %.0e -- %s)\n",
              what, ok ? "OK  " : "FAIL", max_rel_err, tol, why);
}

}  // namespace mcke::benchcfg
