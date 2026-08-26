// =============================================================================
//  tools/device_query.cpp
//
//  WHAT: Prints everything our planner knows about the GPU, plus the derived
//        roofline anchors.
//
//  WHY a separate tool and not a test: the numbers here (SM count, smem budget,
//  peak bandwidth) are *inputs* to every design decision in Phase 3. The first
//  thing to do on each new machine — Colab, Explorer, the 5060 — is run this and
//  paste the output into PROJECT_LOG.md. Otherwise, three months later, you have
//  a benchmark number with no idea which GPU produced it.
//
//  WHY .cpp: pure host code, no kernels. Runs (and reports "no GPU") in the
//  host-only macOS build too, which is how we know the tool itself works before
//  we ever get to a GPU.
// =============================================================================
#include <cstdio>

#include "mcke/core/device.hpp"
#include "mcke/profiling/profiler.hpp"

int main() {
  const int n = mcke::device_count();
  std::printf("MCKE device query — MCKE_WITH_CUDA=%d, visible devices=%d\n\n",
              MCKE_WITH_CUDA, n);

  if (n == 0) {
    std::printf(
        "No CUDA device available.\n"
        "  - On macOS this is expected: this binary is the host-only build.\n"
        "    Host-side logic (allocator, shapes, graph topology) is still tested\n"
        "    by `ctest`. Kernels are simply not present in this build.\n"
        "  - On a GPU machine, this means the driver is missing or no GPU is\n"
        "    visible: check `nvidia-smi` and CUDA_VISIBLE_DEVICES.\n");
    return 0;
  }

  for (int i = 0; i < n; ++i) {
    auto info = mcke::query_device(i);
    if (!info.ok()) {
      std::printf("device %d: %s\n", i, info.status().to_string().c_str());
      continue;
    }
    std::printf("%s\n", info->describe().c_str());

    // Roofline anchors. peak_gb_s comes from the spec formula; the compute peak
    // we deliberately leave to a measured microbenchmark (see profiler.cpp for
    // why a spec-sheet FLOPS number is the wrong denominator).
    mcke::Roofline rl;
    rl.peak_gb_s = info->peak_dram_gb_s();
    std::printf("  roofline: peak_bw=%.1f GB/s (spec formula)\n"
                "            peak_compute=<measure with bench/fma_peak — do NOT\n"
                "            take it from a spec sheet; see docs/PROFILING.md>\n\n",
                rl.peak_gb_s);

    // A useful sanity number for Phase 3 tile sizing, computed here so it ends
    // up in the log alongside the hardware it describes.
    std::printf("  Phase-3 planning aids:\n");
    std::printf("    threads in flight at 100%% occupancy : %d\n",
                info->sm_count * info->max_threads_per_sm);
    std::printf("    smem budget per SM                  : %zu KiB\n",
                info->shared_mem_per_sm / 1024);
    std::printf("    max f32 elements in smem per block  : %zu\n",
                info->shared_mem_per_block_optin / 4);
    std::printf("    regs per thread at 100%% occupancy   : %d\n",
                info->max_threads_per_sm > 0 ? info->regs_per_sm / info->max_threads_per_sm : 0);
    std::printf("\n");
  }
  return 0;
}
