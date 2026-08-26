// =============================================================================
//  src/core/profiler.cpp
//
//  WHY .cpp: no device code — this is the derived-metric arithmetic and CSV
//  output. Host compiler, compiles on macOS.
// =============================================================================
#include "mcke/profiling/profiler.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace mcke {

// -----------------------------------------------------------------------------
// FP32 cores per SM is NOT exposed by the CUDA API. It is a per-architecture
// constant you have to look up, and getting it wrong is the most common cause of
// nonsense "% of peak" numbers.
//
// The subtlety on Turing/Ampere-consumer/Ada/Blackwell (sm_75, 86, 89, 120):
// there are 128 FP32 lanes per SM, but only 64 of them can issue an FMA in the
// same cycle as an INT32 op on some of these parts. The 128 figure is what
// NVIDIA quotes for peak FP32 TFLOPS, so we use it for consistency with the
// spec sheet — and note that a real kernel doing index arithmetic in INT32 may
// not be able to reach it. That caveat belongs in RESULTS.md next to any
// "% of peak" we report.
//
// sm_80 (A100) is 64 FP32 FMA cores/SM for the non-tensor path (its 19.5 TFLOPS
// f32 figure comes from 108 SMs * 64 * 2 * 1.41 GHz).
// -----------------------------------------------------------------------------
static int fp32_cores_per_sm(int cc_major, int cc_minor) {
  const int cc = cc_major * 10 + cc_minor;
  switch (cc) {
    case 70: case 72: return 64;    // Volta
    case 75:          return 64;    // Turing: 64 FP32 cores/SM
    case 80:          return 64;    // A100
    case 86: case 87: return 128;   // Ampere consumer (GA10x)
    case 89:          return 128;   // Ada (L4, 4090)
    case 90:          return 128;   // Hopper
    case 100: case 101: case 120:
                      return 128;   // Blackwell (incl. RTX 50-series, sm_120)
    default:          return 64;    // conservative fallback; will UNDER-report peak
  }
}

double theoretical_peak_f32_tflops(const DeviceInfo& d) {
  // NOTE: we do not have the SM clock in DeviceInfo (cudaDeviceProp::clockRate
  // is deprecated and reports the *base* boost clock, which real kernels exceed
  // or fall short of depending on power/thermals). Rather than bake in a lie, we
  // return 0 when we have no clock and force the caller to supply a measured
  // peak — see docs/PROFILING.md, where we obtain peak_tflops from an FMA-only
  // microbenchmark instead of a spec sheet. A measured ceiling is also the
  // honest one: it already includes whatever clock the GPU actually sustains.
  (void)fp32_cores_per_sm(d.cc_major, d.cc_minor);
  return 0.0;
}

std::string Profiler::summary_table(const Roofline& rl) const {
  std::ostringstream os;
  os << std::left << std::setw(22) << "kernel" << std::setw(26) << "variant"
     << std::right << std::setw(10) << "med_ms" << std::setw(10) << "min_ms"
     << std::setw(10) << "GB/s" << std::setw(10) << "TFLOP/s" << std::setw(8) << "AI"
     << std::setw(10) << "%peak" << "  bound\n";
  os << std::string(110, '-') << '\n';
  os << std::fixed << std::setprecision(3);
  for (const auto& r : records_) {
    const double ai = r.arithmetic_intensity();
    os << std::left << std::setw(22) << r.name << std::setw(26) << r.variant
       << std::right << std::setw(10) << r.median_ms
       << std::setw(10) << r.min_ms
       << std::setw(10) << std::setprecision(1) << r.gb_per_s()
       << std::setw(10) << std::setprecision(3) << r.tflops()
       << std::setw(8) << std::setprecision(2) << ai
       << std::setw(9) << std::setprecision(1) << rl.efficiency(r) * 100.0 << "%"
       << "  " << (rl.memory_bound(ai) ? "memory" : "compute") << '\n';
  }
  return os.str();
}

Status Profiler::write_csv(const std::string& path, const Roofline& rl) const {
  std::ofstream f(path);
  if (!f) return InternalError("write_csv: cannot open " + path);
  // Header names chosen to be plot-script friendly and self-documenting; the
  // ideal-vs-achieved distinction is explicit so a reader six months later
  // cannot misread which is which.
  f << "kernel,variant,median_ms,min_ms,iterations,ideal_flops,ideal_bytes,achieved_gb_s,"
       "achieved_tflops,arithmetic_intensity,attainable_tflops,efficiency_pct,bound\n";
  for (const auto& r : records_) {
    const double ai = r.arithmetic_intensity();
    f << r.name << ',' << r.variant << ',' << r.median_ms << ',' << r.min_ms << ','
      << r.iterations << ','
      << r.flops << ',' << r.bytes << ',' << r.gb_per_s() << ',' << r.tflops() << ','
      << ai << ',' << rl.attainable_tflops(ai) << ',' << rl.efficiency(r) * 100.0 << ','
      << (rl.memory_bound(ai) ? "memory" : "compute") << '\n';
  }
  return OkStatus();
}

}  // namespace mcke
