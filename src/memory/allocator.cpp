// =============================================================================
//  src/memory/allocator.cpp
//
//  WHY .cpp: no device code — this is host bookkeeping around raw_device_malloc.
//  Contains the baseline allocator and the stats formatter. Compiles and runs
//  on macOS in the host-only build (where raw_device_malloc is aligned_alloc),
//  which is exactly what lets us unit-test allocator *logic* without a GPU.
// =============================================================================
#include "mcke/memory/allocator.hpp"

#include <algorithm>   // std::max, for the peak-usage watermark
#include <sstream>

#include "mcke/core/device.hpp"

namespace mcke {

std::string AllocatorStats::to_string() const {
  std::ostringstream os;
  const double mib = 1024.0 * 1024.0;
  os << "allocs=" << alloc_calls << " frees=" << free_calls
     << " raw_mallocs=" << raw_malloc_calls << " raw_frees=" << raw_free_calls
     << " oom=" << oom_events << " deferred=" << deferred_reuses << '\n'
     << "  reserved=" << bytes_reserved / mib << " MiB"
     << "  in_use=" << bytes_in_use / mib << " MiB"
     << "  requested=" << bytes_requested / mib << " MiB"
     << "  peak=" << peak_bytes_in_use / mib << " MiB\n"
     << "  internal_waste=" << internal_waste() / mib << " MiB"
     << "  utilisation=" << utilisation() * 100.0 << "%"
     << "  largest_free_block=" << largest_free_block / mib << " MiB";
  return os.str();
}

// -----------------------------------------------------------------------------
// RawDeviceAllocator — the control group.
//
// Note it ignores `stream` entirely. That is correct *only* because cudaFree is
// a synchronising call: the driver guarantees all prior work on the device has
// completed before the memory is reusable. That guarantee is precisely what
// makes it slow, and precisely what our pool gives up (and must replace with
// explicit stream/event tracking).
// -----------------------------------------------------------------------------
StatusOr<Allocation> RawDeviceAllocator::allocate(std::size_t bytes, rt::StreamHandle stream) {
  (void)stream;
  ++stats_.alloc_calls;
  auto p = raw_device_malloc(bytes);
  if (!p.ok()) {
    ++stats_.oom_events;
    return p.status();
  }
  ++stats_.raw_malloc_calls;
  stats_.bytes_in_use    += bytes;
  stats_.bytes_requested += bytes;
  stats_.bytes_reserved  += bytes;
  stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);

  Allocation a;
  a.ptr = *p;
  a.bytes = bytes;
  a.requested_bytes = bytes;
  return a;
}

Status RawDeviceAllocator::deallocate(const Allocation& a, rt::StreamHandle stream) {
  (void)stream;
  ++stats_.free_calls;
  ++stats_.raw_free_calls;
  stats_.bytes_in_use    -= a.bytes;
  stats_.bytes_requested -= a.requested_bytes;
  stats_.bytes_reserved  -= a.bytes;
  return raw_device_free(a.ptr);
}

}  // namespace mcke
