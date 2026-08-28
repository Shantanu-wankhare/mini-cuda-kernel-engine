// =============================================================================
//  src/core/host_timer.cpp
//
//  WHY .cpp: pure host arithmetic (sorting, percentile math, self-calibration).
//  No CUDA, no device code. Compiles and runs identically on macOS and any GPU
//  machine — the calibration numbers WILL differ (24 MHz on Apple Silicon vs. a
//  TSC-backed vDSO clock on x86-64), and that difference is the point: it is
//  what tells a reader which machine's latency numbers are load-bearing.
// =============================================================================
#include "mcke/profiling/host_timer.hpp"

#include <sstream>

namespace mcke {

ClockCalibration ClockCalibration::measure(int samples) {
  // Measure the clock with itself: back-to-back start/stop pairs with nothing
  // between them, so every nonzero delta IS the clock's own call + read
  // overhead, not any real work.
  std::vector<std::uint64_t> deltas;
  deltas.reserve(static_cast<std::size_t>(samples));
  HostTimer t;
  for (int i = 0; i < samples; ++i) {
    t.start();
    deltas.push_back(t.stop_ns());
  }
  std::sort(deltas.begin(), deltas.end());

  ClockCalibration c;
  // Smallest NONZERO delta: some pairs legitimately read 0 ns if the clock
  // ticked between neither call, which just means the granularity is coarser
  // than the call overhead — the tick itself is what we want.
  for (std::uint64_t d : deltas) {
    if (d > 0) { c.tick_ns = d; break; }
  }
  c.paired_median_ns = deltas.empty() ? 0 : deltas[deltas.size() / 2];
  c.floor_ns = std::max(c.tick_ns, c.paired_median_ns);
  return c;
}

std::string ClockCalibration::describe() const {
  std::ostringstream os;
  os << "tick=" << tick_ns << "ns  paired_median=" << paired_median_ns
     << "ns  ==> floor=" << floor_ns << "ns"
     << "  (percentiles at or below " << (2 * floor_ns)
     << "ns are instrument-limited, not physically meaningful)";
  return os.str();
}

std::string LatencyStats::log2_histogram(std::uint64_t floor_ns) const {
  if (samples_.empty()) return "  (no samples)\n";
  if (floor_ns == 0) floor_ns = 1;   // guard against a degenerate calibration

  // Bucket i covers (floor*2^(i-1), floor*2^i], with bucket 0 covering [0, floor].
  // Find the number of buckets needed to cover max().
  const std::uint64_t hi = samples_.back();
  int n_buckets = 1;
  while ((floor_ns << n_buckets) < hi && n_buckets < 40) ++n_buckets;

  std::vector<std::size_t> counts(static_cast<std::size_t>(n_buckets) + 1, 0);
  for (std::uint64_t v : samples_) {
    std::size_t b = 0;
    std::uint64_t edge = floor_ns;
    while (v > edge && b < counts.size() - 1) { edge <<= 1; ++b; }
    ++counts[b];
  }

  std::ostringstream os;
  std::uint64_t edge = floor_ns;
  for (std::size_t b = 0; b < counts.size(); ++b) {
    if (counts[b] == 0) { if (b > 0) edge <<= 1; continue; }
    os << "    <=" << edge << "ns: " << counts[b] << '\n';
    edge <<= 1;
  }
  return os.str();
}

}  // namespace mcke
