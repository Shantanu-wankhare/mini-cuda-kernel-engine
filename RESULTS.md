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
| Colab | Tesla T4 | 7.5 | 40 | 64 KiB | 320.1 GB/s | 235.4 GB/s | 8.130 TFLOP/s | driver 580.82.07 / nvcc 12.8.93 |
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
| stream_triad | grid_stride_256t | 64Mi | 768 MiB | 3.421 | 3.417 | 235.4 | 100.0% (this IS the baseline) | Colab T4 |
| vector_add | grid_stride_256t | 64Mi | 768 MiB | 3.348 | 3.344 | 240.5 | 102.2% | Colab T4 |

Median and min sit within 0.1-0.2% of each other for both kernels — a clean,
idle, unthrottled T4 (41 degC at session start, no other tenants), not a wide
median-min gap that would have signalled clock ramp or a noisy shared GPU. This
is the first row filled in after the `Profiler::time_op` min-tracking fix
(`include/mcke/profiling/profiler.hpp` / `src/core/profiler.cpp`, 2026-08-26).

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

Produced by `./build/bin/mcke_alloc_bench` (source: `bench/alloc_bench.cpp`).
Three traces, all generated once per run from a fixed `std::mt19937_64` seed
(`0x9E3779B97F4A7C15`) using raw engine output rather than `<random>`
distributions — engines are specified bit-exactly by the standard, distributions
are not, so this is what makes a macOS run and a Colab run the *same* workload.

**Both pooling allocators are run under all three cross-stream reuse policies.**
That is not thoroughness for its own sake: it is the only way the deallocate
column stays interpretable on a GPU.

| Policy | What its deallocate pays for |
|---|---|
| `same_stream` | **pure allocator bookkeeping** — never probes for completion |
| `coarse_poll` | the above **+ one `cudaStreamQuery`** |
| `event` | the above **+ one `cudaEventRecord`** |

On the host build all three probes are free. On a GPU, rows 2 and 3 gain a real
driver round trip (~1–2 µs) while row 1 does not — so `same_stream` is the
honest coalescing-vs-no-coalescing baseline, and subtracting it from the other
two measures what each safety mechanism actually costs. Without row 1 there is
no baseline and the deallocate comparison silently becomes a driver-latency
benchmark.

### 2a. Latency

**Not yet collected — must come from a GPU.** In the host-only build
`raw_device_malloc` is `aligned_alloc` (~150 ns), not `cudaMalloc` (~10–100 µs),
so the headline pool-vs-raw speedup is a GPU phenomenon this machine physically
cannot demonstrate. The bench prints that caveat itself.

Additionally, the MacBook's host clock has a **41 ns tick** (24 MHz timebase;
`hw.tbfrequency`), and its own call overhead is also ~41 ns — so any median at or
below ~82 ns is instrument-limited, not a physical measurement. The bench marks
those with `*`. An x86-64 host reading the TSC via vDSO resolves ~1 ns, which is
why this table's authoritative numbers come from Colab.

| Trace | Allocator | Policy | Op | median ns | p90 | p99 | p999 | max | amortised ns | alloc_calls | raw_mallocs | Machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | | | | | | |

*The `amortised` column is one timestamp bracket around the whole bare loop
divided by op count — it recovers fast-path cost on a coarse clock, where the
per-op median is quantised. Per-op timing is kept anyway because averaging a
batch provably erases the p99 tail, which is the entire reason p99 is an exit
criterion.*

### 2b. Fragmentation — **authoritative, host-only build**

These are pure host bookkeeping with no GPU dependency, so the MacBook is the
right place for them and a Colab re-run should reproduce them exactly.

Three ratios, never one. `AllocatorStats::utilisation()` alone conflates
internal fragmentation with slab over-provisioning, and would read ~62% for an
allocator that wasted literally zero bytes:

- **`block_eff`** = peak_requested / peak_blocks — **internal fragmentation only.
  This is the number the roadmap's "buddy 55–70%" prediction actually meant.**
- **`reserv_eff`** = peak_blocks / peak_reserved — slab sizing, a property of the
  configured slab size, *not* of allocator design.
- **`utilisation`** = their product; what the header's accessor reports.

Measured 2026-08-26, MacBook Air (Apple Silicon), Apple clang 21.0.0,
`MCKE_WITH_CUDA=0`. Policy does not affect fragmentation, so one row per
allocator (all three policies produced identical figures except where noted).

**Trace `uniform_pow2`** — the control. Every size is `2^k ≥ kMinBlockBytes`,
so buddy's internal waste **must** be exactly zero; any other value is an
allocator bug, not a result.

| Allocator | peak_reserved | peak_blocks | peak_requested | block_eff | reserv_eff | utilisation | internal waste | largest_free @ end | OOM? |
|---|---|---|---|---|---|---|---|---|---|
| raw | 29.80 MiB | 29.80 MiB | 29.80 MiB | 100.0% | 100.0% | 100.0% | 0 B | n/a | no |
| buddy | 48.00 MiB | 29.80 MiB | 29.80 MiB | **100.0%** | 62.1% | 62.1% | **0 B** | 16–32 MiB | no |
| freelist | 80.00 MiB | 36.38 MiB | 29.04 MiB | 79.8% | 45.5% | 36.3% | 7.34 MiB | 8.00 MiB | no |

Buddy's 0 B is the control assertion holding. Freelist's 7.34 MiB comes from its
own ladder: `small_class_granularity = 512` cannot represent the 256 B class, so
**every 256 B request wastes 50%** — precisely where buddy (min block 256 B) is
perfect. The design that wins on non-power-of-two shapes loses on the smallest
one.

**Trace `dl_transformer`** — GPT-2-small shapes, f32, batch 1 × seq 512.

| Allocator | peak_reserved | peak_blocks | peak_requested | block_eff | reserv_eff | utilisation | internal waste | largest_free @ end | OOM? |
|---|---|---|---|---|---|---|---|---|---|
| raw | 45.04 MiB | 45.04 MiB | 45.04 MiB | 100.0% | 100.0% | 100.0% | 0 B | n/a | no |
| buddy | 80.00 MiB | 68.05 MiB | 43.54 MiB | **64.0%** | 85.1% | 54.4% | 24.51 MiB | **32.00 MiB** | no |
| freelist | 80.00 MiB | 68.04 MiB | 43.54 MiB | **64.0%** | 85.0% | 54.4% | 24.50 MiB | **16.00 MiB** | no |

**Trace `dl_transformer_bypass`** — the same workload with a 147 MiB embedding
table (50257 × 768 × f32) folded in, so the bypass-to-driver path is exercised
amid real churn. Kept as its own trace so its extra permanent driver allocation
never muddies the clean traces' flatline.

| Allocator | peak_reserved | peak_blocks | peak_requested | block_eff | reserv_eff | utilisation | internal waste | largest_free @ end | OOM? |
|---|---|---|---|---|---|---|---|---|---|
| raw | 192.27 MiB | 192.27 MiB | 192.27 MiB | 100.0% | 100.0% | 100.0% | 0 B | n/a | no |
| buddy | 227.24 MiB | 215.28 MiB | 190.77 MiB | 88.6% | 94.7% | 84.0% | 24.51 MiB | 32.00 MiB | no |
| freelist | 227.24 MiB | 215.27 MiB | 190.77 MiB | 88.6% | 94.7% | 84.0% | 24.50 MiB | 16.00 MiB | no |

The bypassed tensor is served exactly (zero reported waste, matching
`RawDeviceAllocator` for comparability), which is what lifts `block_eff` from
64.0% to 88.6% — the waste is unchanged in absolute terms (24.5 MiB), it is just
divided by a much larger total.

### 2c. Predictions vs. measurement

**Prediction (2026-08-24, before any implementation existed):** buddy
utilisation 55–70% on non-power-of-two shapes; **freelist 85–95%**. On a long
churny mixed-size trace, freelist's `largest_free_block` collapses and it OOMs
while still holding free bytes; buddy holds up.

**Buddy: CONFIRMED.** 64.0% `block_eff` on the DL trace, inside the predicted
55–70%.

**Freelist: CONTRADICTED, and the reason is the interesting part.** Measured
64.0%, not 85–95% — an exact tie with buddy. `FreeListConfig::small_large_split`
is 1 MiB, and *above* that the size-class ladder **is** a power-of-two ladder,
i.e. identical rounding to buddy. Multi-MiB tensors (weights, attention scores)
are essentially all the bytes in a transformer, so the two designs must round
identically and tie. The free-list's advantage exists only in the size range
where its ladder is genuinely finer than a power of two.

That range is real, and a unit test pins it directly
(`test_freelist_beats_buddy_on_dl_shapes`): on the roadmap's literal decode
sizes (768, 3072, 50257 × f32) freelist achieves **99.9%** against buddy's
**76.6%**, driven by the 50257-element logits row — 201,028 B, which buddy
rounds to 262,144 while freelist's 512 B ladder gives 201,216. Powers of two
cannot represent a vocabulary size.

So the honest correction to the prediction is: *a size-class ladder only beats a
buddy allocator in the size range where the ladder is actually finer than a
power of two.* On this workload, that is nowhere near where the bytes are.

**External fragmentation: CONFIRMED, and it is the cleanest result in Phase 2.**
From `test_freelist_external_fragmentation`, with both allocators given an
identical request sequence — fill a 64 KiB arena with 512 B blocks, free **all**
of them, then request 8 KiB:

| Allocator | free bytes held | largest contiguous free block | 8 KiB request |
|---|---|---|---|
| buddy | 65,536 B | 65,536 B (fully coalesced) | **succeeds** |
| freelist | 65,536 B | **512 B** | **OOM** |

Every single byte is free in both cases. Only one can serve the request. That
inequality — free bytes ≫ request, largest contiguous block < request — *is*
external fragmentation, stated as three numbers instead of a sentence.

**The tradeoff, in one pair of rows.** The coalescing that wins buddy the row
above is the same mechanism it pays for on the free path. Both directions are
measured, and neither allocator is categorically better:

| | buddy | freelist |
|---|---|---|
| largest contiguous free block after full drain | **wins** (32 MiB vs 16 MiB) | |
| internal fragmentation on sub-1 MiB non-pow2 shapes | | **wins** (99.9% vs 76.6%) |
| free-path cost | pays a coalesce cascade | O(1) push, no cascade |

### 2d. Stream-safety race — pending (GPU only)

`tests/test_stream_safety.cu`, `ctest -R stream_safety`. Cannot run host-only:
with `MCKE_WITH_CUDA=0` both stream handles are `nullptr`, so rule 1 in
`pending_reusable()` legitimately fires, `stream_query` is unconditionally true
so nothing ever parks, and there is no concurrency — there is no race to
construct.

| Arm | Expected | Observed | Aliased? | Refused reclaim? |
|---|---|---|---|---|
| naive_pool (no stream tracking) | CORRUPT 20/20 | _pending_ | | |
| buddy/freelist × 3 policies | CLEAN 20/20 | _pending_ | | |
| same-stream control | CLEAN 20/20, **same pointer** | _pending_ | | |
| raw(cudaMalloc) | stream idle after free | _pending_ | | |

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
