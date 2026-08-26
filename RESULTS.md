# RESULTS.md

Running record of every measurement. `PROJECT_LOG.md` tells the story; this file
holds the numbers in comparable tables.

**Rules for adding a row (so the numbers stay defensible):**
1. Record the environment: GPU model, compute capability, driver version, CUDA
   toolkit version, and whether clocks were locked.
2. Record the exact command line.
3. Report **median** and **min** over ≥ 20 timed iterations after ≥ 5 warmups.
   Never a single run. Never the mean alone.
4. Report the **ideal** FLOPs/bytes used in the derived metrics, so anyone can
   recompute them.
5. State the **denominator** for any "% of peak": spec-sheet bandwidth, measured
   bandwidth, or measured FMA peak. These differ by 10-30%, and a % of peak
   without a stated denominator is meaningless.
6. If a result contradicts the prediction, keep the prediction in the table and
   write down why it was wrong. That is the valuable part.

---

## 0. Hardware inventory

Fill one row per machine, from `./build/bin/mcke_device_query`.

| Machine | GPU | CC | SMs | smem/SM | Peak BW (spec formula) | Measured BW | Measured FMA f32 peak | Driver / CUDA |
|---|---|---|---|---|---|---|---|---|
| MacBook Air (M-series) | — | — | — | — | — | — | — | host-only build |
| Colab | Tesla T4 | 7.5 | 40 | 64 KiB | 320.1 GB/s | 235.5 GB/s | 8.059 TFLOP/s | driver 580.82.07 / nvcc 12.8.93 |
| RTX 5060 laptop | _TBD_ | 12.0 | | | | | | needs CUDA ≥ 12.8 |
| Explorer | _TBD_ | | | | | | | |

> Colab clocks were **not locked** (no root access in the hosted notebook); the
> T4 was idle and cool at session start (51 degC), so throttling is unlikely for
> runs this short, but treat this row as indicative rather than authoritative —
> per `docs/ENVIRONMENTS.md`, Explorer is where the authoritative numbers come
> from. "Measured BW" here is `stream_triad`'s figure, not `vector_add`'s
> (240.6 GB/s) — see section 1 for why they differ and why that's expected.

> Note: "Peak BW (spec formula)" is `2 × memory_clock_khz × 1e3 × bus_bits/8`
> (the leading ×2 accounts for `cudaDeviceProp::memoryClockRate` reporting one
> edge of a double-data-rate clock — see `DeviceInfo::peak_dram_gb_s()` in
> `include/mcke/core/device.hpp` for how we found this was missing).
> "Measured BW" comes from `bench/stream_triad` (Phase 1). Use the **measured**
> figure as the denominator for bandwidth efficiency — it already accounts for
> ECC overhead and real sustained clocks, and is typically 80-90% of the spec
> number.

---

## 1. Phase 1 — Elementwise / bandwidth baseline

| Kernel | Variant | n | Ideal bytes | median ms | min ms | GB/s | % measured BW | Machine |
|---|---|---|---|---|---|---|---|---|
| stream_triad | grid_stride_256t | 64Mi | 768 MiB | 3.420 | pending rerun* | 235.5 | 100.0% (this IS the baseline) | Colab T4 |
| vector_add | grid_stride_256t | 64Mi | 768 MiB | 3.346 | pending rerun* | 240.6 | 102.2% | Colab T4 |

\* These rows predate the `Profiler::time_op` fix that added min-ms tracking
(`include/mcke/profiling/profiler.hpp` / `src/core/profiler.cpp`,
2026-08-26 — `KernelRecord::ms` split into `median_ms` + `min_ms`). The GB/s and
median-ms figures above are still correct and unaffected by that change; only
the min-ms column was genuinely missing data, not wrong data. Replace this
row's "pending rerun" with the real min ms once the next Colab run reports it.

**Prediction (recorded 2026-08-24, before any run):** 70-85% of measured
bandwidth. Below 50% indicates a problem — pageable-memory staging on the copies,
too small a grid, or clock throttling.

**Actual (2026-08-26, Colab T4):** 102.2% — higher than predicted, not lower.
Reason: `vector_add` and `stream_triad` turned out to be structurally almost
identical grid-stride kernels (same thread/block count heuristics, same 3-array
access pattern), so there was little room for one to lag the other; the ~2%
gap is consistent with ordinary run-to-run timing noise rather than a real
difference between them. The 70-85% guess assumed more daylight between the
two kernels than actually exists.

---

## 2. Phase 2 — Allocator

### 2a. Latency

| Allocator | Op | Median ns | p99 ns | raw cudaMalloc calls | Machine |
|---|---|---|---|---|---|
| raw (cudaMalloc) | allocate | | | = alloc_calls | |
| buddy | allocate | | | | |
| freelist | allocate | | | | |

**The headline claim to prove:** `raw_malloc_calls` stops growing after warm-up
while `alloc_calls` grows without bound. Report both, not just the speedup.

### 2b. Fragmentation, on a DL-shaped allocation trace

| Allocator | Peak reserved | Peak requested | Utilisation | Internal waste | Largest free block at end | OOM? |
|---|---|---|---|---|---|---|
| buddy | | | | | | |
| freelist | | | | | | |

**Predictions (2026-08-24):** buddy utilisation 55-70% on non-power-of-two
shapes; freelist 85-95%. On a long churny mixed-size trace, freelist's
`largest_free_block` collapses and it OOMs while still holding free bytes; buddy
holds up. Measured worst-case buddy internal fragmentation is already confirmed
at **50.0%** (4097 B request → 8192 B block, from `test_host_core`).

---

## 3. Phase 3 — Kernels

### 3a. GEMM, f32, square M=N=K

| Variant | M=N=K | Tile (BM,BN,BK,TM,TN) | regs/thread | smem/block | occupancy | median ms | TFLOP/s | % of measured FMA peak | Machine |
|---|---|---|---|---|---|---|---|---|---|
| naive | 4096 | — | | | | | | | |
| tiled_smem | 4096 | 32,32,32,1,1 | | | | | | | |
| tiled_regblock | 4096 | 128,128,8,8,8 | | | | | | | |
| warptile_dbuf | 4096 | 128,128,8,8,8 | | | | | | | |
| cuBLAS | 4096 | — | — | — | — | | | | |

**Predictions (2026-08-24):** naive 2-4% of peak (AI = 0.25, memory-bound);
tiled_smem 15-25%; tiled_regblock 45-65%; warptile+double-buffer 60-80%; cuBLAS
is the ceiling. Each row must attribute its gain to *one* change.

Ideal cost: `flops = 2·M·N·K`, `bytes = (M·K + K·N + M·N)·4`.

### 3b. Reduction / softmax

| Kernel | Variant | rows × cols | median ms | GB/s | % measured BW | __syncthreads count | Machine |
|---|---|---|---|---|---|---|---|
| row_reduce_sum | smem_tree | 8192 × 4096 | | | | log2(block) | |
| row_reduce_sum | warp_shuffle | 8192 × 4096 | | | | 1 | |
| row_softmax | three_pass | 8192 × 4096 | | | | | |
| row_softmax | online_one_pass | 8192 × 4096 | | | | | |

**Predictions:** warp-shuffle beats the smem tree by 10-30% (fewer barriers, no
bank conflicts, less smem → higher occupancy). Online softmax reduces global
traffic from 3 passes to 1-2 and should approach the reduce kernel's bandwidth.
Both are memory-bound: report **GB/s, not TFLOP/s**.

### 3c. Fusion

| Kernel | median ms | Ideal bytes | GB/s | Machine |
|---|---|---|---|---|
| bias_add then gelu (2 kernels) | | 4·N·4 | | |
| fused bias_gelu (1 kernel) | | 2·N·4 | | |

**Prediction:** ~2× speedup, because traffic halves and the op is bandwidth-bound.
If the measured speedup is well under 2×, find out why (launch overhead
dominating? cache hits making the second kernel's reads cheap?) — that
investigation is the deliverable, not the 2×.

---

## 4. Phase 4 — Scheduling

| Graph | Policy | streams | median ms | speedup vs sequential | peak memory | Machine |
|---|---|---|---|---|---|---|
| diamond (A→{B,C}→D) | sequential | 1 | | 1.00× | | |
| diamond | level_parallel | 2 | | | | |
| diamond | chain_greedy | 2 | | | | |
| chain ×16 | sequential | 1 | | 1.00× | | |
| chain ×16 | chain_greedy | 4 | | | | |

Also record: events recorded per iteration, host time in `run_async()`, and
memory saved by `kReuseByLiveness` vs `kAllocPerTensor`
(`ExecutionPlan::peak_memory_bytes()` vs `naive_memory_bytes()`).

**Prediction:** overlap helps only if the individual kernels leave SMs idle. If
B and C each already saturate the GPU, expect ~1.0× — and that non-result, with
the `nsys` timeline showing why, is a legitimate finding.

---

## 5. Nsight Compute deep dives

One subsection per kernel actually profiled. Required fields:
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`,
`achieved_occupancy`, top warp-stall reason, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
(→ sectors per request, the coalescing measure), and
`dram__bytes_read.sum` compared to our ideal bytes.

_(none yet)_
