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
| Explorer | Tesla V100-SXM2-32GB | 7.0 | 80 | 96 KiB | 898.0 GB/s | 636.3 GB/s | 15.601 TFLOP/s | driver 545.23.08 / nvcc 12.3 |

> Explorer's clocks are effectively locked for the purposes of this project:
> the GEMM ladder's cuBLAS first-vs-last check (§3d) measured **+0.03% drift**
> across the full run, versus the Colab T4's +11.9% — the difference between an
> indicative number and an authoritative one. `gpu-interactive` partition,
> `--gres=gpu:v100-sxm2:1`, no thermal-throttling caveat needed on anything in
> this row.

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

**Re-confirmed 2026-08-29** (same Colab T4, different session, used here to
validate the Phase 2 machine against Phase 1's): `stream_triad` 235.3 GB/s
(73.5% of spec peak), `fma_peak` 8.126 TFLOP/s — both within 0.1% of the
2026-08-26 figures. This machine is the same class of result as the one that
produced the numbers above, so the Phase 2 figures below are directly
comparable to Phase 1's.

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

### 2a. Latency — **Colab Tesla T4, 2026-08-29, driver 580.82.07 / nvcc 12.8.93**

`./build/bin/mcke_alloc_bench` after the `settle_pending()` fix (commit
`e5c8b9d`). Clock: `tick=22ns paired_median=25ns floor=25ns` — an x86-64 TSC via
vDSO, ~1000× finer than the Mac's 41 ns ARM timebase, which is why this table
(not the host build) is the authoritative source for every number in it.

**`raw` is real `cudaMalloc`/`cudaFree` here** (unlike the host build, where it
is `aligned_alloc`), so this is the first table where the 100-1000× pool speedup
is actually demonstrable.

**Trace `uniform_pow2`** (99,816 allocs, small blocks 256 B – 1 MiB):

| Allocator | Op | median ns | p90 | p99 | p999 | max ns | amortised ns | alloc_calls | raw_mallocs |
|---|---|---|---|---|---|---|---|---|---|
| raw | allocate | 2913 | 5415 | 11098 | 78140 | 719836 | 3978.0 | 99816 | 99816 |
| raw | deallocate | 3487 | 7271 | 14003 | 60256 | 402893 | – | – | – |
| buddy/same_stream | allocate | 63 | 152 | 486 | 2346 | 29753 | 52.0 | 99816 | 2 |
| buddy/same_stream | deallocate | 42* | 84 | 141 | 441 | 71163 | – | – | – |
| buddy/coarse_poll | allocate | 56 | 77 | 429 | 524 | 17013 | 279.8 | 99816 | 2 |
| buddy/coarse_poll | deallocate | 472 | 548 | 654 | 4749 | 34177 | – | – | – |
| buddy/event | allocate | 62 | 140 | 471 | 597 | 26254 | 403.6 | 99816 | 2 |
| buddy/event | deallocate | 526 | 978 | 1301 | 7082 | 107915 | – | – | – |
| freelist/same_stream | allocate | 120 | 169 | 258 | 496 | 303011 | 109.3 | 99816 | 5 |
| freelist/same_stream | deallocate | 108 | 195 | 322 | 579 | 400805 | – | – | – |
| freelist/coarse_poll | allocate | 113 | 148 | 256 | 467 | 309729 | 345.2 | 99816 | 5 |
| freelist/coarse_poll | deallocate | 604 | 717 | 1061 | 7701 | 555192 | – | – | – |
| freelist/event | allocate | 125 | 187 | 286 | 515 | 307205 | 538.3 | 99816 | 5 |
| freelist/event | deallocate | 656 | 1214 | 1737 | 8430 | 525543 | – | – | – |

**Trace `dl_transformer`** (26,010 allocs, GPT-2-small shapes, multi-MiB tensors
dominate):

| Allocator | Op | median ns | p90 | p99 | p999 | max ns | amortised ns | alloc_calls | raw_mallocs |
|---|---|---|---|---|---|---|---|---|---|
| raw | allocate | 1855 | 66439 | 105048 | 137964 | 309106 | 32879.1 | 26010 | 26010 |
| raw | deallocate | 1838 | 121022 | 489471 | 573950 | 978282 | – | – | – |
| buddy/same_stream | allocate | 144 | 1412 | 1672 | 7614 | 438994 | 244.1 | 26010 | 3 |
| buddy/same_stream | deallocate | 56 | 63 | 94 | 173 | 17100 | – | – | – |
| buddy/coarse_poll | allocate | 101 | 1460 | 1659 | 7802 | 407527 | 677.3 | 26010 | 3 |
| buddy/coarse_poll | deallocate | 958 | 1080 | 1138 | 12560 | 35995 | – | – | – |
| buddy/event | allocate | 182 | 1509 | 1709 | 6381 | 364438 | 770.9 | 26010 | 3 |
| buddy/event | deallocate | 1028 | 1126 | 1177 | 13066 | 18882 | – | – | – |
| freelist/same_stream | allocate | 157 | 236 | 266 | 413 | 102191 | 126.4 | 26010 | 5 |
| freelist/same_stream | deallocate | 123 | 135 | 169 | 316 | 17586 | – | – | – |
| freelist/coarse_poll | allocate | 145 | 160 | 195 | 404 | 134252 | 564.1 | 26010 | 5 |
| freelist/coarse_poll | deallocate | 1063 | 1127 | 1186 | 12741 | 58471 | – | – | – |
| freelist/event | allocate | 166 | 241 | 327 | 592 | 104856 | 604.6 | 26010 | 5 |
| freelist/event | deallocate | 1055 | 1167 | 1327 | 13401 | 1879240 | – | – | – |

**Trace `dl_transformer_bypass`** (26,011 allocs, same shapes + one 147 MiB
embedding table taking the bypass path):

| Allocator | Op | median ns | p90 | p99 | p999 | max ns | amortised ns | alloc_calls | raw_mallocs |
|---|---|---|---|---|---|---|---|---|---|
| raw | allocate | 2275 | 86233 | 120732 | 152884 | 1970152 | 26905.3 | 26011 | 26011 |
| raw | deallocate | 2113 | 114587 | 336525 | 383271 | 1524228 | – | – | – |
| buddy/same_stream | allocate | 78 | 678 | 798 | 1349 | 273937 | 129.6 | 26011 | 4 |
| buddy/same_stream | deallocate | 36* | 42* | 107 | 130 | 199844 | – | – | – |
| buddy/coarse_poll | allocate | 61 | 692 | 822 | 2763 | 255691 | 343.7 | 26011 | 4 |
| buddy/coarse_poll | deallocate | 493 | 554 | 696 | 5974 | 206069 | – | – | – |
| buddy/event | allocate | 93 | 736 | 848 | 1802 | 240317 | 396.4 | 26011 | 4 |
| buddy/event | deallocate | 517 | 578 | 688 | 6893 | 183931 | – | – | – |
| freelist/same_stream | allocate | 90 | 131 | 156 | 332 | 88066 | 74.7 | 26011 | 6 |
| freelist/same_stream | deallocate | 76 | 86 | 117 | 192 | 163973 | – | – | – |
| freelist/coarse_poll | allocate | 91 | 106 | 124 | 194 | 77632 | 301.3 | 26011 | 6 |
| freelist/coarse_poll | deallocate | 565 | 606 | 641 | 9481 | 222100 | – | – | – |
| freelist/event | allocate | 94 | 138 | 164 | 304 | 72531 | 323.7 | 26011 | 6 |
| freelist/event | deallocate | 546 | 626 | 652 | 5768 | 195660 | – | – | – |

`*` = at or below the 25 ns instrument floor.

*The `amortised` column is one timestamp bracket around the whole bare loop
divided by op count — it recovers fast-path cost on a coarse clock, where the
per-op median is quantised. Per-op timing is kept anyway because averaging a
batch provably erases the p99 tail, which is the entire reason p99 is an exit
criterion.*

**The deallocate decomposition, exactly as designed** (see the policy table
above). `same_stream` never probes; `coarse_poll` and `event` each add one real
driver round trip. On `dl_transformer`: same_stream deallocate median 56 ns,
coarse_poll 958 ns, event 1028 ns — a ~900-1000 ns tax for the completion proof,
on top of whatever the allocator's own bookkeeping costs. That tax is close to
constant across allocators and traces (compare buddy 958 ns vs. freelist 1063 ns
coarse_poll on the same trace), which is exactly what "you're paying for a
`cudaStreamQuery`, not for the allocator" should look like.

**The headline claim, on real hardware:** `raw` allocate median ranges
1.8–2.9 µs and its p99 climbs into the 11-120 µs range depending on trace (real
`cudaMalloc` against the live driver state) — three orders of magnitude above
any pooled allocator's median. `raw_malloc_calls` is forced to equal
`alloc_calls` by construction; every pooled configuration in every trace stays
at 2-6 total driver allocations for 26k-99k logical allocate calls.

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
`MCKE_WITH_CUDA=0`, and **re-confirmed byte-for-byte on Colab Tesla T4,
2026-08-29** after the `settle_pending()` fix — every peak/waste/largest_free
figure below matched exactly across the two machines, which is the expected
outcome for numbers that are pure host bookkeeping with no GPU dependency.
Policy does not affect fragmentation, so one row per allocator (all three
policies produced identical figures).

**Trace `uniform_pow2`** — the control. Every size is `2^k ≥ kMinBlockBytes`,
so buddy's internal waste **must** be exactly zero; any other value is an
allocator bug, not a result.

| Allocator | peak_reserved | peak_blocks | peak_requested | block_eff | reserv_eff | utilisation | internal waste | largest_free @ end | OOM? |
|---|---|---|---|---|---|---|---|---|---|
| raw | 29.80 MiB | 29.80 MiB | 29.80 MiB | 100.0% | 100.0% | 100.0% | 0 B | n/a | no |
| buddy | 48.00 MiB | 29.80 MiB | 29.80 MiB | **100.0%** | 62.1% | 62.1% | **0 B** | 32.00 MiB | no |
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
| free-path cost (`same_stream` deallocate median, Colab T4, `dl_transformer`) | 56 ns | **wins**, 123 ns — but see below |

**On real hardware the free-path story is more nuanced than the host build
suggested.** Buddy's `same_stream` deallocate median is *lower* than
freelist's on `dl_transformer` (56 ns vs 123 ns) — freelist pays a
`std::unordered_map` insert (`live_`) on every allocate and an erase on every
free, which turns out to cost more than buddy's coalescing check in the common
case where the coalesce loop terminates after 0-1 iterations. The coalesce
*cascade* buddy pays for is real, but it shows up in the **tail**, not the
median: buddy's p99 deallocate is close to freelist's or worse in several rows
(e.g. `dl_transformer_bypass` freelist/event p999 5768 ns vs buddy/event
6893 ns), and its `max` occasionally spikes far higher (buddy/same_stream
`dl_transformer_bypass` max 199,844 ns — almost certainly one trial's
first-touch/cudaEventCreate cost, not the allocator itself, since it does not
recur at that magnitude elsewhere in the same row). The one-line qualitative
story ("buddy pays a cascade, freelist doesn't") is directionally right but the
median comparison alone would have said the opposite thing.

### 2d. Stream-safety race — **PASSED, Colab Tesla T4, 2026-08-29**

`ctest --test-dir build -R stream_safety` → `Passed` (0.50 s), after the
`settle_pending()` fix (an earlier run failed two arms — see below).

`tests/test_stream_safety.cu`. Cannot run host-only: with `MCKE_WITH_CUDA=0`
both stream handles are `nullptr`, so rule 1 in `pending_reusable()`
legitimately fires, `stream_query` is unconditionally true so nothing ever
parks, and there is no concurrency — there is no race to construct.

The reader is gated on a host-released mapped-pinned flag, not a timed spin: the
host only sets it after `cudaStreamSynchronize(S2)` proves the corrupting write
already landed, so the ordering is structural rather than probabilistic. Every
arm ran 20 trials.

| Arm | Expected | Observed | Aliased? | Refused cross-stream reclaim? | Result |
|---|---|---|---|---|---|
| naive_pool (no stream tracking) | CORRUPT 20/20 | **CORRUPT 20/20** (524,288 of 524,288 elements every trial) | yes | – | ok |
| buddy/same_stream_only | CLEAN 20/20 | CLEAN 20/20 | no | **yes** | ok |
| buddy/coarse_poll | CLEAN 20/20 | CLEAN 20/20 | no | yes | ok |
| buddy/per_free_event | CLEAN 20/20 | CLEAN 20/20 | no | yes | ok |
| freelist/same_stream_only | CLEAN 20/20 | CLEAN 20/20 | no | **yes** | ok |
| freelist/coarse_poll | CLEAN 20/20 | CLEAN 20/20 | no | yes | ok |
| freelist/per_free_event | CLEAN 20/20 | CLEAN 20/20 | no | yes | ok |
| buddy/same_stream **control** | CLEAN 20/20, same pointer | CLEAN 20/20 | **yes** | – | ok |
| raw(cudaMalloc) | stream idle after free | stream idle | – | – | ok |

**The two bold "yes" cells are the fix.** The first Colab run (before
`settle_pending()`) reported these two as `NO` — not because the allocators
were unsafe (both were CLEAN 20/20 in that run too) but because a block parked
during the per-trial warm-up under `kSameStreamOnly` had no same-stream
allocate *within the warm-up* to trigger its own reclaim, so it was still
counted as pending when the trial's own block should have been the only thing
there. `settle_pending()` (added to `DeviceAllocator`, see `PROJECT_LOG.md`)
flushes that residue before the measured window opens. The safety **property**
was correct in both runs; only this secondary mechanism diagnostic needed the
fix.

**The naive control matters as much as the failures.** Its 100%-of-elements
corruption on every single trial is what makes every other row's CLEAN mean
something — a race harness that never demonstrates the race is a harness that
proves nothing when the real allocators pass.

---

## 3. Phase 3 — Kernels

Section letters match `docs/ROADMAP.md` Phase 3 one-to-one: **3a fusion, 3b row
reduction, 3c row softmax, 3d GEMM.** They are build-order identifiers used in
commit messages and log entries, so they name work items, not presentation
order. (An earlier version of this file lettered them 3a=GEMM, 3b=reduce+softmax,
3c=fusion — three letters for four workloads, contradicting the roadmap. Fixed
here while §3 was still empty and the fix was free.)

**Denominators for every "% of peak" in this section** (RESULTS.md rule 5),
both measured on the Colab T4, never spec-sheet figures:
`peak_gb_s = 235.4` (from `bench/stream_triad`), `peak_tflops = 8.130` (from
`bench/fma_peak`), ridge point **34.5 FLOP/byte**. (Written as 34.2 until
2026-08-29; `Roofline::ridge_point_ai()` computes 8.130e12 / 235.4e9 = 34.54 from
the two frozen denominators, so the prose was simply inconsistent with the code
that produces every `%peak` in this file. The code was right.) Re-measured on the Colab
session that produced §3a–§3c below (2026-08-29): `stream_triad` 240.9 GB/s,
`fma_peak` 8.184 TFLOP/s — both within ~2.3% of the Phase-1/2 baseline, so
this is the same class of machine and the frozen denominators above are kept
rather than replaced (replacing them per-session would make GB/s numbers
across phases incomparable, which is the thing rule 5 exists to prevent).

> Every kernel in §3a–§3c has an arithmetic intensity between 0.08 and 2.8 —
> **12× to 400× below the ridge point.** They are all memory-bound, so those
> tables report **GB/s and never TFLOP/s**. Only §3d's GEMM (AI ≈ 682) is
> compute-bound and reports TFLOP/s. Reporting TFLOP/s for a reduction is the
> red flag `graph/op.hpp` already warns about.

### 3a. Fusion — bias + activation

Shape **8192 × 4096** (`N = 33,554,432`), identical to §3b/§3c so the three
memory-bound kernels are directly comparable. At 256 MiB the working set is 64×
the T4's 4 MiB L2, so nothing caches.

Ideal bytes: fused `(2N + cols)·4` = 268,451,840; unfused pair `(4N + cols)·4` =
536,887,296. The `cols·4` = 16 KiB bias term is 0.006% of the total — it is in
the formula because rule 4 requires the count to be reconstructible, not because
it moves any number.

| Kernel | Activation | vector_width | median ms | min ms | Ideal bytes | GB/s | % measured BW | Machine |
|---|---|---|---|---|---|---|---|---|
| bias_relu fused | relu | vw4 | 1.102 | 1.100 | 268,451,840 | 243.7 | 103.5% | Colab T4 |
| bias_relu unfused | relu | vw4 | 2.215 | 2.2 | 536,887,296 | 242.3 | 102.9% | Colab T4 |
| bias_gelu_tanh fused | gelu_tanh | vw4 | 1.083 | 1.1 | 268,451,840 | 247.8 | 105.3% | Colab T4 |
| bias_gelu_tanh unfused | gelu_tanh | vw4 | 2.226 | 2.2 | 536,887,296 | 241.2 | 102.5% | Colab T4 |
| bias_gelu_erf fused | gelu_erf | vw4 | 1.104 | 1.1 | 268,451,840 | 243.2 | 103.3% | Colab T4 |
| bias_gelu_erf unfused | gelu_erf | vw4 | 2.225 | 2.2 | 536,887,296 | 241.3 | 102.5% | Colab T4 |
| bias_gelu_tanh fused | gelu_tanh | vw1 | 1.1 | 1.1 | 268,451,840 | 238.8 | 101.5% | Colab T4 |
| bias_gelu_tanh fused | gelu_tanh | vw2 | 1.1 | 1.1 | 268,451,840 | 250.5 | 106.4% | Colab T4 |
| bias_gelu_tanh fused, starved (40 blocks) | gelu_tanh | vw1 | 1.3 | 1.3 | 268,451,840 | 208.7 | 88.7% | Colab T4 |
| bias_gelu_tanh fused, starved (40 blocks) | gelu_tanh | vw2 | 1.2 | 1.2 | 268,451,840 | 226.9 | 96.4% | Colab T4 |
| bias_gelu_tanh fused, starved (40 blocks) | gelu_tanh | vw4 | 1.2 | 1.2 | 268,451,840 | 224.6 | 95.4% | Colab T4 |
| bias_gelu_tanh fused, L2 control | gelu_tanh | vw4 | 0.008 | 0.0 | 2,099,200 | 271.1 | 115.2% | Colab T4 |
| bias_gelu_tanh unfused, L2 control | gelu_tanh | vw4 | 0.011 | 0.0 | 4,196,352 | 391.5 | 166.3% | Colab T4 |

Shape 8192×4096 unless noted; L2-control rows are 512×512 (256 KiB per array,
fits the T4's 4 MiB L2). All correctness checks passed at this shape,
including the two derived-tolerance fixes below.

**Results vs. predictions (2026-08-29 Colab T4, `b01e48d`):**
- **Fusion speedup landed almost exactly on the 2× prediction**: relu 2.01×,
  gelu_tanh 2.05×, gelu_erf 2.02× (1.102/2.215, 1.083/2.226, 1.104/2.225 ms).
- **The L2-resident control did NOT collapse to ~1.0–1.2× as predicted** — it
  measured **1.38×** (0.008/0.011 ms). The prediction assumed the unfused
  pair's second kernel reads its intermediate entirely from L2; in practice
  some of that traffic still costs real time (likely L2 latency plus kernel
  launch overhead dominating at sub-10-µs runtimes, not a clean DRAM-vs-L2
  swap). This is a genuine partial miss, not just noise — worth revisiting
  with `ncu` L2 hit-rate counters before trusting the mechanism, but the
  direction (control ratio << full-DRAM ratio) still supports the argument.
- **Vector width was flat at full occupancy and NOT flat when starved, as
  predicted**: full-occupancy vw1/vw2/vw4 = 238.8/250.5/242.8 GB/s (within
  noise of each other); starved vw1/vw2/vw4 = 208.7/226.9/224.6 GB/s — a real
  ~5–8% width-dependent gap once occupancy, not vector width, is the scarce
  resource. Confirms MLP is only worth buying when occupancy is low.
- **erf vs tanh landed inside the predicted ±2%**: 1.104 vs 1.083 ms = +1.9%,
  consistent with the FP32-pipe cost being hidden under the memory floor.
- Two GELU-only validation failures were found and fixed this session (not a
  kernel bug): `max_abs_err≈4.77e-7` (exactly 4 ULP) against the default
  `abs_tol=1e-8`, caused by GELU's O(1) `(1+tanh(z))`/`(1+erf(z))` intermediate
  turning a routine few-ULP libm disagreement into an absolute error
  independent of the near-zero output magnitude at the curve's knee. Fixed by
  adding `testing::kAbsTolGeluCancellation = 1e-6` (`tests/reference.hpp`) and
  passing it at the three GELU `verify()` call sites in `bias_act_bench.cpp`.

Row plan: `{relu, gelu_tanh, gelu_erf} × {fused, unfused}` at vw=4 (6 rows);
`{vw1, vw2, vw4}` fused + vw1 unfused at gelu_tanh (4); an **L2-resident control**
at 512×512 (2); a **deliberately occupancy-starved** vw sweep (see below).

**Predictions (2026-08-29, before any run):**
- Fused ≈ **2×** the unfused pair, because traffic halves and the op is
  bandwidth-bound.
- The **L2-resident control at 512×512 should collapse that to ~1.0–1.2×**,
  because the unfused pair's second kernel then reads its intermediate from L2
  rather than DRAM. This pre-answers, with a controlled experiment, the question
  the original prediction paragraph raised as a possible excuse.
- **`vector_width` buys 0–10% at full occupancy and may be inside the noise** —
  the T4 absorbs ~3.7 B/cycle/SM while an SM issues ~512 B/cycle of requests,
  so instruction issue is nowhere near the limiter. The *same sweep at ~6%
  occupancy* (40 blocks, grid-strided) should show a large win, because vector
  width buys memory-level parallelism and MLP is only scarce when occupancy is.
- **`kGeluErf` vs `kGeluTanh` should be invisible** (within ~2%): ~20 vs ~10
  instructions/element is ~100 µs of FP32-pipe work against a ~1140 µs memory
  floor. If erf shows up 5–9% slower instead, the arithmetic is no longer fully
  hidden — which is itself the finding, and `ncu`'s top warp-stall reason
  (`long_scoreboard` → `mio_throttle`) settles which happened.

Footnote to record: not fusing also costs a full 128 MiB intermediate tensor.
That is the memory-planner argument Phase 4 will make, measured here.

### 3b. Row reduction

Shape **8192 × 4096**. Ideal bytes `(rows·cols + rows)·4` = 134,250,496
(128.03 MiB) → 0.570 ms at 235.4 GB/s. AI = 0.25, matching `op.hpp`'s own note.

**Convention:** all three variants use this same ideal-byte count, including
`kTwoPass`, so the GB/s column ranks them like-for-like. `kTwoPass`'s extra
partial-staging traffic (~512 KiB, +0.4%) is *algorithmic*, not compulsory, and
belongs in a footnote rather than the denominator.

| Kernel | Variant | rows × cols | median ms | min ms | Ideal bytes | GB/s | % measured BW | __syncthreads | Machine |
|---|---|---|---|---|---|---|---|---|---|
| row_reduce_sum | smem_tree_256t | 8192 × 4096 | 0.519 | 0.517 | 128.03 MiB | 258.5 | 109.8% | **9** | Colab T4 |
| row_reduce_max | smem_tree_256t | 8192 × 4096 | 0.5 | 0.5 | 128.03 MiB | 256.4 | 108.9% | 9 | Colab T4 |
| row_reduce_mean | smem_tree_256t | 8192 × 4096 | 0.5 | 0.5 | 128.03 MiB | 257.5 | 109.4% | 9 | Colab T4 |
| row_reduce_sum | warp_shuffle_256t | 8192 × 4096 | 0.523 | 0.5 | 128.03 MiB | 256.9 | 109.1% | **1** | Colab T4 |
| row_reduce_max | warp_shuffle_256t | 8192 × 4096 | 0.5 | 0.5 | 128.03 MiB | 254.2 | 108.0% | 1 | Colab T4 |
| row_reduce_mean | warp_shuffle_256t | 8192 × 4096 | 0.5 | 0.5 | 128.03 MiB | 254.1 | 107.9% | 1 | Colab T4 |
| row_reduce_sum | two_pass_256t | 8192 × 4096 | 0.537 | 0.5 | 128.03 MiB | 250.2 | 106.3% | 1 | Colab T4 |
| row_reduce_sum | warp_shuffle_256t | **64 × 524288** | 0.937 | 0.9 | 128.03 MiB | 143.2 | 60.8% | 1 | Colab T4 |
| row_reduce_sum | two_pass_256t | **64 × 524288** | 0.538 | 0.5 | 128.03 MiB | 249.3 | 105.9% | 1 | Colab T4 |

The barrier count is `1 + log2(blockDim)` = **9** at 256 threads, not 8 — the
load-into-smem barrier before the tree starts is a real barrier. (Both this
column and `kernels.hpp` previously said 8.)

**Results vs. predictions (2026-08-29 Colab T4, `b01e48d`):**
- **Warp-shuffle did NOT beat the smem tree by 10–30% — it landed within
  ±1.3%** (sum -0.6%, max -0.9%, mean -1.3%, shuffle actually very slightly
  slower on all three). The absolute-GB/s check explains why the prediction
  had no room to be true: smem_tree already sits at ~108–110% of the 235.4
  GB/s denominator (the denominator is a conservative floor, not a hard cap),
  so both variants are already pinned to the DRAM wall and the 9-vs-1 barrier
  difference is invisible next to it. Honest negative result: barrier count
  does not matter here because bandwidth, not synchronization, is the limiter.
- **`kTwoPass` at the saturated shape (8192×4096) was 2.7% slower than
  warp_shuffle** (0.537 vs 0.523 ms) — inside the predicted 1–3% overhead
  band, confirming it as pure algorithmic tax with no upside when the machine
  is already full.
- **`kTwoPass` at the starved shape (64×524288) was 1.74× faster than
  warp_shuffle** (0.538 vs 0.937 ms) — real, but well short of the predicted
  3–10×. `warp_shuffle`'s starved-shape efficiency (60.8% of peak) is much
  higher than a naive one-block-per-row model suggests, likely because 64
  blocks still gives partial multi-block-per-SM overlap on a 40-SM part
  rather than true 24-SM idleness — worth an `ncu` occupancy check before
  trusting the mechanism, but the qualitative point (two-pass is the only
  variant that scales to few-rows/very-wide shapes) is confirmed.
- One validation failure was found and fixed this session (not a kernel bug):
  `kSum` at `max_abs_err≈1.5e-5` against the default `abs_tol=1e-8`, at a
  near-zero-mean row (`want≈0.2`, routine for `[-1,1]` random fill, not
  adversarial). Root cause: summation rounding error scales with the
  magnitude of the terms being summed (~1), not the final sum, so a
  near-cancelling row fails a pure-relative test even though the kernel is
  correct; `kMean` is unaffected because dividing by `cols` shrinks the error
  floor and the value together. Fixed by adding
  `testing::kAbsTolReduceSum4096 = 5e-5` and passing it only at the `kSum`
  `verify()` call in `reduce_bench.cpp`. A real diagnostic bug in
  `compare()`'s "worst offender" tracker was found and fixed alongside this
  (see `tests/reference.hpp`): it compared against `max_abs_err` after that
  field had already been unconditionally updated in the same iteration,
  so it reported the wrong element as "worst."

### 3c. Row softmax

Shape **8192 × 4096**. Ideal bytes: **compulsory** traffic `2N·4` = 268,435,456
for both variants (read x once, write y once), so the GB/s column ranks them
directly. Algorithmic traffic differs — three-pass reads x three times
(`4N·4` = 537 MB), online reads it twice (`3N·4` = 403 MB) — and goes in a
footnote.

| Kernel | Variant | rows × cols | median ms | min ms | Ideal bytes | GB/s | % measured BW | max abs(Σrow − 1) | Machine |
|---|---|---|---|---|---|---|---|---|---|
| row_softmax | three_pass_256t | 8192 × 4096 | 2.053 | 2.051 | 256.0 MiB | 130.7 | 55.5% | 1.701e-07 | Colab T4 |
| row_softmax | online_one_pass_256t | 8192 × 4096 | 1.767 | 1.8 | 256.0 MiB | 151.9 | 64.5% | 2.184e-07 | Colab T4 |

**Results vs. predictions (2026-08-29 Colab T4, `b01e48d`):**
- **Speedup landed at 1.16×**, inside the predicted 1.0–1.25× band (2.053 vs
  1.767 ms), well short of the naive 3× a reader might guess from "one-pass."
  Confirms the framing: "one-pass" names the statistics passes, not the
  memory passes — both variants still read x again to produce y.
- Neither variant showed the predicted L2 masking (three-pass did not beat
  its own 3-read traffic model) — both land well below 100% of the 235.4
  GB/s denominator (55.5% / 64.5%), consistent with the *extra* passes being
  real DRAM traffic rather than L2 hits, unlike the L2-control finding in
  §3a. This is worth an `ncu` `dram__bytes_read.sum` check before treating
  it as settled, since the a-priori argument (2.5 MiB working set fits the
  4 MiB L2) still holds on paper.
- **Numerics landed close to prediction**: online is 1.28× less accurate by
  the `Σrow−1` check (2.184e-07 vs 1.701e-07), both comfortably under 1e-5
  and under the predicted ~3e-7. Lower ratio than the 2–5× predicted, but
  the direction (online less accurate, both negligible) is confirmed.
- No validation tolerance issues here — softmax's max-subtraction cancels the
  magnitude sensitivity that caused the §3a/§3b tolerance bugs; both variants
  passed at the pre-existing `kTolSoftmax=1e-5` on the first Colab run.

**Predictions:**
- Speedup **4/3 ≈ 1.33×, not 3×.** "One-pass" names the *statistics* passes
  (max and sum computed together), not the memory passes — you still need x
  again to produce y. Netting out likely L2 reuse: **1.0–1.25×**.
- **Watch for L2 masking the result.** Each row is 16 KiB; ~160 resident blocks
  give a 2.5 MiB working set that *fits* the T4's 4 MiB L2, so three-pass's 2nd
  and 3rd reads may never touch DRAM. If three-pass beats its own traffic model,
  that is the finding, not an error — `dram__bytes_read.sum` vs `3·N·4` settles it.
- Online should reach **90–100% of `row_reduce`'s GB/s** — equally efficient per
  byte while moving 3× the bytes. That is what "approaches the reduce kernel's
  bandwidth" means, and it is a claim about GB/s, not about time.
- **Online is slightly LESS accurate**, and that is expected: its rescaling chain
  adds error, and its numerator and denominator are no longer computed from an
  identical expression. Predict online's max relative error at **2–5×**
  three-pass's, both under 1e-5. The `Σrow − 1` column is reference-free and
  exposes exactly that inconsistency (predict ~1e-7 three-pass, ~3e-7 online).

### 3d. GEMM, f32, square M=N=K

Shape **M=N=K=4096**, `alpha=1, beta=0`, row-major. Command:
`./build/bin/mcke_gemm_bench 4096` (the bench echoes its own argv, per rule 2).

| Variant | M=N=K | Tile (BM,BN,BK,TM,TN) | regs/thread | smem/block | spill B | occupancy (hand / API) | median ms | min ms | TFLOP/s | % of measured FMA peak | Machine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive_uncoalesced | 4096 | — | 32 | 0 | 0 | 4 / 4 (100%) | 1143.68 | 1140.86 | 0.120 | 1.48% | Colab T4 |
| naive | 4096 | — | 32 | 0 | 0 | 4 / 4 (100%) | 343.44 | 341.76 | 0.400 | 4.92% | Colab T4 |
| tiled_smem | 4096 | 32,32,32,1,1 | 43 | 8192 | 0 | 1 / 1 (100%, tied) | 163.19 | 160.49 | 0.842 | 10.36% | Colab T4 |
| tiled_regblock | 4096 | 128,128,8,8,8 | 115 | 8320 | 0 | 2 / 2 (50%) | 41.85 | 41.51 | 3.284 | 40.39% | Colab T4 |
| warptile_nodbuf | 4096 | 128,128,8,8,8 | 115 | 8320 | 0 | 2 / 2 (50%) | 42.62 | 42.00 | 3.224 | 39.66% | Colab T4 |
| warptile_dbuf | 4096 | 128,128,8,8,8 | 128 | 16640 | 0 | 2 / 2 (50%) | 41.97 | 41.64 | 3.275 | 40.28% | Colab T4 |
| warptile_vec4 | 4096 | 128,128,8,8,8 | 128 | 16640 | 0 | 2 / 2 (50%) | 39.70 | 39.36 | 3.462 | 42.58% | Colab T4 |
| cuBLAS (first) | 4096 | — | — | — | — | — | 33.12 | 32.56 | 4.150 | 51.05% | Colab T4 |
| cuBLAS (drift recheck, last) | 4096 | — | — | — | — | — | 37.07 | 36.06 | 3.708 | 45.61% | Colab T4 |
| naive_uncoalesced | 4096 | — | 32 | 0 | 0 | 8 / 8 (100%, tied) | 302.9 | 302.8 | 0.454 | 2.90% | Explorer V100 |
| naive | 4096 | — | 32 | 0 | 0 | 8 / 8 (100%, tied) | 78.9 | 78.9 | 1.741 | 11.16% | Explorer V100 |
| tiled_smem | 4096 | 32,32,32,1,1 | 42 | 8192 | 0 | 1 / 1 (50%) | 48.9 | 48.8 | 2.812 | 18.02% | Explorer V100 |
| tiled_regblock | 4096 | 128,128,8,8,8 | 120 | 8320 | 0 | 2 / 2 (25%) | 11.7 | 11.7 | 11.732 | 75.20% | Explorer V100 |
| warptile_nodbuf | 4096 | 128,128,8,8,8 | 121 | 8320 | 0 | 2 / 2 (25%) | 11.7 | 11.7 | 11.710 | 75.06% | Explorer V100 |
| warptile_dbuf | 4096 | 128,128,8,8,8 | 128 | 16640 | 0 | 2 / 2 (25%) | 10.9 | 10.7 | 12.631 | 80.96% | Explorer V100 |
| warptile_vec4 | 4096 | 128,128,8,8,8 | 128 | 16640 | 0 | 2 / 2 (25%) | 10.8 | 10.7 | 12.739 | 81.66% | Explorer V100 |
| cuBLAS (first) | 4096 | — | — | — | — | — | 9.831 | 9.554 | 13.980 | 89.60% | Explorer V100 |
| cuBLAS (drift recheck, last) | 4096 | — | — | — | — | — | 9.834 | 9.804 | 13.975 | 89.57% | Explorer V100 |

**The Explorer run is the clean baseline the Colab run couldn't be**: cuBLAS
first-vs-last drift was **+0.03%** — clocks held throughout, no caveat needed on
any number in this block. Driver 545.23.08, CUDA 12.3, Tesla V100-SXM2-32GB
(sm_70), on the `gpu-interactive` partition. Measured denominators for this
chip: `peak_gb_s = 636.3` (`stream_triad`), `peak_tflops = 15.601` (`fma_peak`),
ridge point **24.5 FLOP/byte** — a different roofline entirely from the T4's
34.5, so these numbers are never mixed into a single "%peak" comparison with the
T4 rows above; they sit in the same table only because the `Machine` column
already exists to keep architectures distinguishable, per this file's own
"don't mix V100 and A100 in one comparison" rule in §0.

**Same source code, same tile sizes (untuned for this chip), a different
occupancy story on almost every row** — and the differences are all explained
by the architecture, not by anything wrong with the kernels:

- **`naive`/`naive_uncoalesced` are a genuine tie here (8/8) where they were not
  tied on the T4 (4/4, uniquely threads-cap-bound).** These kernels use exactly
  32 registers/thread — which `mcke_device_query` printed as this chip's own
  "regs per thread at 100% occupancy: 32" *before* any kernel ran. The T4 has
  64 regs/thread of headroom at 100% occupancy for the same 65536-register file,
  because it has half as many threads/SM to spread them across; the V100 has
  exactly none left at 32. Same kernel, same register count, different binding
  story purely from the SM's own thread-to-register ratio.
- **`tiled_smem` flips from a 100%-occupancy tie on the T4 to 50%,
  uniquely register-bound, here.** The V100's 2048 threads/SM means one
  1024-thread block only fills half the SM's thread capacity — so where the T4's
  1024-thread cap made *itself* the tie-breaker, here the thread cap is not even
  close to binding, and 42 registers/thread turns out to allow exactly one
  resident block, not two.
- **`tiled_regblock`'s 2 blocks is 50% occupancy on the T4 and 25% here** — the
  same absolute number of active warps (16) reads as a smaller fraction of a
  bigger SM. And despite that *lower* occupancy percentage, this row hits
  **75.2%** of the V100's own peak versus the T4's 40.4% of its own — occupancy
  percentage alone predicts none of this; what matters is whether 16 warps is
  enough to keep this specific, ALU-heavy kernel fed, and on this chip it clearly
  is.

**`warptile_nodbuf`'s regression did not reproduce.** On the T4 it measured
**−0.73 pp** relative to `tiled_regblock` (a contradiction of the +5–12%
prediction, per §3d above). Here it is **−0.10 pp** — flat, indistinguishable
from run-to-run noise at this iteration count. Registers, shared memory, tile,
and occupancy are all identical to `tiled_regblock` on both chips, and the
verified bank-conflict cut (`test_gemm_bank_conflict_math`) is an
architecture-independent integer property — so this is evidence *against* a bug
in the lane permutation itself, and evidence *for* the T4-specific hypothesis
already on record: the extra lane-index arithmetic cost and the bank-conflict
saving were roughly cancelling on Turing. Whether they cancel for the same
reason here, or the saving and the cost are both just smaller on Volta, is
still an `ncu` question — but "the kernel is subtly broken" is no longer a live
hypothesis for this regression.

**Double buffering buys far more here than on the T4**: **+5.9 pp**
(`tiled_regblock`→`warptile_nodbuf`→`warptile_dbuf`, 75.1%→81.0%) versus the
T4's **+0.6 pp**. Consistent with the mechanism this row was always supposed to
demonstrate — hiding DRAM latency behind compute — mattering more when there
are fewer resident warps already doing that job: 25% occupancy here (2 blocks)
versus 50% there, for the exact same block count, on a chip with proportionally
more thread-slots per SM.

**`warptile_vec4` buys less here** (+0.7 pp vs. the T4's +2.3 pp): cutting
global load instruction count from 4-per-thread to 1 matters most when
instruction issue rate is the binding constraint, and at 81% of a 15.6 TFLOP/s
peak this kernel is evidently not issue-bound on this chip the way it may have
been on the T4.

**cuBLAS's own efficiency is the headline number of this whole run: 89.6% of
measured peak, both before and after the ladder**, versus the T4's throttled
51.0%/45.6%. This is the number to treat as authoritative for "how close does
hand-written CUDA get to a vendor library" — and at 81.7%, `warptile_vec4` sits
within **8 percentage points** of it (a 1.10× gap) versus the T4's 1.20× gap,
even before any Explorer-side tuning of tile sizes for this architecture.

**smem/block matches the pad prediction exactly.** `tiled_regblock`'s 8320 B is
8192 + 128 = the `kGemmAPad=4` cost (`BK·4 floats·4 B = 128 B`) predicted in the
design review. `warptile_dbuf`/`warptile_vec4` double that to 16640, confirming
the padded footprint carries through both buffers.

**Occupancy hand-calc agreed with the CUDA API on every row, including the two
edge cases that were the actual test:** `tiled_smem` is a genuine **tie** between
registers and the threads/SM cap at 43 regs/thread (both give exactly 1 block) —
predicted in `gemm_tile.hpp`'s test suite before this ever ran on hardware.
`warptile_dbuf` landed at **exactly** 128 registers, the boundary this project
predicted for staying at 2 blocks rather than falling to 1 (25%) — one register
over and occupancy would have halved. Neither kernel spilled (`spill B` = 0 on
every row), so register blocking is doing what it is supposed to.

Ideal cost: `flops = 2·M·N·K` = 1.374e11, `bytes = (M·K + K·N + M·N)·4` = 2.01e8,
**AI ≈ 682.7** — deep in compute-bound territory, 20× past the ridge point. This is
the only section in the document that reports TFLOP/s.

**Two columns deliberately absent, and one deliberately ignored.** There is no
GB/s column: `bytes` here is the *operation's* compulsory traffic, so GB/s would
be compulsory-bytes-over-time, which for the naive row is ~0.35 — neither
achieved bandwidth nor anything else. The console `summary_table` still prints
it (it is generic); ignore it for this section. Likewise its `bound` column reads
`compute` for the naive rows, which is **correct**: the operation is
compute-bound and the naive kernel simply fails to exploit that.

> **Why all eight rows use the same `bytes`.** AI is a property of the operation,
> not of an implementation. The naive kernel's often-quoted "AI = 0.25" is its
> *access-pattern* intensity, a different quantity. Substituting it gives
> `attainable_tflops(0.25) = 0.0589`, so a naive kernel at the predicted 2–4% of
> peak would report **`%peak = 550%`**. Uniform compulsory bytes is what keeps
> the rows comparable to each other.

**Predictions — recorded 2026-08-29, before the run, per rule 6.** The 2026-08-24
originals are preserved in the third column; where they were revised, the reason
is a design-review finding, not a measurement.

| Row | Predicted % of FMA peak | Actual | Verdict |
|---|---|---|---|
| naive_uncoalesced | 0.2–0.6% | **1.48%** | miss — 2.5–7.4× above the predicted range |
| naive | 2–4% | **4.92%** | miss — just above the range |
| tiled_smem | 15–25% | **10.36%** | miss — below the range |
| tiled_regblock | 45–65% | **40.39%** | miss — below the range |
| warptile_nodbuf | +5–12% over regblock | **−1.8%** (0.98×) | **contradicted — regressed, did not improve** |
| warptile_dbuf | 60–80% | **40.28%** | miss — well below the range |
| warptile_vec4 | +10–20% over dbuf | **+5.7%** (1.057×) | miss — below the range |
| cuBLAS | 75–85% | **51.05%** (first) / 45.61% (last) | miss — well below the range |

**Every prediction missed, in the same direction, and one was contradicted
outright.** Per rule 6, the predictions stay in the table above rather than
being smoothed into the revised numbers, and the reasoning for each miss follows
— separating what the run actually shows from what remains a hypothesis pending
`ncu` on Explorer or the 5060 (Colab does not expose profiling counters).

**1. Thermal drift is real, measured, and it explains part — but not
most — of the shortfall.** `cublas` bracketed the run: 33.12 ms first, 37.07 ms
last, **+11.9%**, past the 3% threshold this project treats as "the table is
drifting." The run order is slow-to-fast (`naive_uncoalesced` through
`warptile_vec4`, in that order, ~46 s of kernel time total), so the clock had
already started dropping off boost by the time the fast rows ran — meaning
`warptile_vec4`'s 42.58% and even `tiled_regblock`'s 40.39% are measured under a
*warmer* chip than the frozen 8.130 TFLOP/s denominator (from a cool Phase-1
session) assumes. Comparing `warptile_vec4` against the **hot** `cublas_last`
figure instead of the cool `peak_tflops` denominator gives `3.462 / 3.708 =
93.4%` — a very different, much more encouraging number. **This does not
explain the tiled_smem or warptile_nodbuf misses**, which are large enough (and
in tiled_smem's case, off by an entire predicted band) that an 11.9% clock
effect cannot be the whole story.

**2. `tiled_smem`'s "zero cross-block overlap" caveat, written into the kernel's
own banner before the run, may be the dominant effect.** At 1024 threads/block
this kernel occupies the *entire* SM with one resident block — 100% occupancy,
but no second block to hide the two `__syncthreads` stalls per k-tile. The
15–25% prediction modelled only two ceilings (shared bandwidth ≈25%, DRAM reuse
≈23%); it did not model barrier-stall time, because that requires a third,
occupancy-dependent term the two-roofline argument does not capture. The
measured 10.36% is consistent with barrier stalls being the actual binding
constraint rather than either roofline. `smsp__average_warps_issue_stalled_
barrier_per_issue_active.ratio` on this kernel specifically (not just the
nodbuf→dbuf transition it was originally proposed for) would confirm this.

**3. `warptile_nodbuf` regressing instead of improving is the most interesting
open result in this table, and it is not explained by anything measured here.**
The lane permutation was verified, as an integer property, to cut the
B-fragment's shared-bank conflict from 4-way to 2-way (5 phases → 3,
`test_gemm_bank_conflict_math`) — that part of the design is not in question.
Register count, shared memory, and occupancy are all identical to
`tiled_regblock` (115 regs, 8320 B, 2 blocks/SM), so the regression is not an
occupancy or spilling effect. Two live hypotheses, neither confirmed:
  - the `kWarp4x8` lane-index arithmetic (`warp/2`, `lane/8`, etc.) costs more
    integer ALU work than the row-major map's plain division, and at this
    occupancy (16 of 32 warps/SM active) there may not be enough independent
    work to hide that extra latency — i.e. the kernel was never actually
    shared-memory-bandwidth-bound at 40% of peak, so cutting shared-load phases
    bought nothing;
  - some other resource (LSU issue rate, or a scheduling artifact of the warp
    grid touching output tiles in a different order) is the true binding
    constraint, and bank conflicts were a red herring at this occupancy level.
  This is exactly the kind of finding `ncu`'s stall-reason breakdown
  (`smsp__average_warps_issue_stalled_*`) is built to distinguish, and it is the
  single most important thing to check on the next Explorer or 5060 session.

**4. The naive rows landed slightly ABOVE their predicted range rather than
below it**, the opposite direction of every other row. `naive_uncoalesced` at
1.48% is 2.5–7.4× the predicted 0.2–0.6%, and it ran *first*, when the chip was
coolest — so this one miss plausibly goes the other way from the thermal
story: a cool, boosted clock outperforming a prediction calibrated loosely
against "should be memory-bound and slow." The transpose's net effect (A and C
worse, B better — see the banner) was always acknowledged as unmeasured by
anything but sectors-per-request, which needs `ncu` to check directly.

**Conclusion for this trip:** correctness is unambiguous (every variant agreed
with the CPU reference at the awkward shapes, with cuBLAS as a validated
full-shape oracle, and the beta≠0 path), and the occupancy hand-calculation
matched the CUDA API on every row including both edge cases it was designed to
catch. But the **performance numbers in this table are Colab-indicative, not
authoritative** — the drift check itself says so, per this project's own rule.
The step ladder up through `tiled_regblock` (3.33×, 2.10×, 3.90×) is the clean
part of the story; everything from `warptile_nodbuf` onward needs an `ncu` pass
on Explorer or the 5060 before the "remaining gap to cuBLAS" writeup this
section owes can be more than a hypothesis.

**Three corrections made before running, all recorded rather than quietly fixed:**

1. **Warp tiling cannot remove the B-fragment bank conflict.** With `Bs[BK][BN]`
   and `TN=8` a lane's bank is `(c·8 + i) mod 32`, which has **period 4** in the
   column group — at most 4 distinct banks are reachable by *any* assignment of
   lanes. The best a pure remap achieves is 2×16 lanes (A 1-way, B 4-way = 5
   phases) → 4×8 lanes (A 1-way, B 2-way = 3 phases): a **1.67× cut, not
   elimination**. Padding does not help, because the period comes from the
   intra-row stride `TN`, not the row stride. Hence the revised prediction.
2. **The transposed-A shared store is an 8-way conflict** in the obvious layout
   (bank = `arow mod 32`; lanes 0–7 all hit bank 0 at 8 distinct addresses, paid
   `K/BK = 512×` per block). **The fix is pad = 4, not the reflexive pad = 1.**
   With stride `BM+p` the bank is `(p·acol + arow) mod 32`, which must be
   injective over `acol ∈ 0..7, arow ∈ 0..3`; `p=1` gives `acol+arow`, where
   (0,1) and (1,0) collide — **still 4-way**. `p=4` walks 0,4,…,28 with a 0..3
   offset and covers all 32 banks exactly once. Verified as an integer property
   in `test_gemm_bank_conflict_math` (8-way → 4-way → conflict-free for p = 0, 1,
   4). Cost: 128 B per buffer, occupancy unchanged.

   Together with `tiled_smem` needing *no* padding, this is one lesson with two
   halves — what matters is the **leading dimension mod 32**, and "always pad"
   and "never pad" are equally wrong.

   The pad is also what makes row 7 single-variable: the `float4` loader must
   decompose differently (`arow = tid/2, acol = 4·(tid%2)`), and under stride 132
   *that* store is conflict-free too. Had the 8-way conflict been left in place,
   vectorizing would have improved it to 2-way as a **side effect** and the row
   would have carried two causes. One fix, two problems.
3. **The blocks-per-SM cap is 16 on sm_75, not 32.** Volta is 32, Turing halved
   it. Every authoritative number here comes from a T4, so the previous comment
   in `kernels.hpp` had the fourth occupancy limiter wrong on the exact hardware
   being measured. `DeviceInfo::max_blocks_per_sm` now reads it from the driver.

**Two rows have disclosed multiple causes** — stated rather than papered over:
`warptile_dbuf` changes both the barrier count (2→1) and the global-load issue
point (~512 FFMAs earlier, overlapping DRAM latency with compute), and the
second is probably the larger half; isolating them would need a third row.
`naive_uncoalesced` makes A reads and C writes worse but B reads *better* (a
broadcast), so its slowdown is a net — which is why the attribution belongs to
`sectors_per_request`, not the wall clock.

**`tiled_smem` has two independent limiters landing two points apart**, and the
15–25% prediction cannot distinguish them:
- *Shared bandwidth*: 2 shared loads per FMA = 0.25 FLOP/B; at 32 banks × 4 B ×
  1.59 GHz × 40 SM = 8.14 TB/s that ceilings at **2.03 TFLOP/s = 25%**.
- *DRAM*: A and B each re-read 4096/32 = 128× = 17.2 GB; at 235.4 GB/s that is
  73 ms = **1.88 TFLOP/s = 23%**.

Only `dram__bytes_read.sum` settles which roof was hit. Predicted in advance.

**The occupancy pair is the most valuable result in this table, and it is now
measured, not hypothetical:** `tiled_smem` runs at **100% occupancy** (1024
threads/block, one resident block, a genuine tie between the threads-per-SM cap
and registers, confirmed by the CUDA API) and `tiled_regblock` at **50%**
(2 blocks, register-bound) — yet `tiled_regblock` measured **3.90× faster**
(41.85 ms vs 163.19 ms), well past the "roughly 2–3×" this section originally
guessed. That is the measured answer to the standing question in
`LEARNING_LOG.md` about how a lower-occupancy kernel can be faster: occupancy
governs latency hiding, and this kernel is not latency-bound.

`occupancy` is reported as **hand-computed / CUDA-API**, two of three legs; the
third is Nsight's measured `sm__warps_active.avg.pct_of_peak_sustained_active` in
§5. The middle leg is free and separates *"my arithmetic is wrong"* from *"the
hardware is not achieving theoretical"* — different bugs, different fixes. The
hand calculation models sm_75's per-warp 256-register allocation granule and
4-warp rounding; a naive `regs_per_sm / (R · threads)` diverges from the API
(e.g. 7 vs 5 at R=17, 512 threads) and would make the comparison uninformative.

**Pre-registered ncu metric per transition** (rule 6 works better when the
prediction is specific enough to be wrong):

| Transition | Metric | Prediction |
|---|---|---|
| naive_uncoalesced → naive | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | 32 → 4 |
| naive → tiled_smem | `dram__bytes_read.sum` | falls ~32× |
| tiled_smem → tiled_regblock | shared-load instructions per FFMA | 2.0 → 0.25 |
| tiled_regblock → warptile_nodbuf | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | falls ~1.67× |
| warptile_nodbuf → warptile_dbuf | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | roughly halves |
| warptile_dbuf → warptile_vec4 | `smsp__inst_executed_op_global_ld.sum` | falls 4×, DRAM bytes unchanged |

**Modelled DRAM traffic** (a *model*, printed by the bench; the measurement is
`dram__bytes_read.sum`). This is the reuse story `docs/PROFILING.md` calls the
single most instructive number in the project:

| Variant | Model read bytes | Amplification vs compulsory |
|---|---|---|
| naive (either) | `2·M·N·K·4` = 5.5e11 | ~2731× |
| tiled_smem | `(M·K·N/32 + K·N·M/32)·4` | ~85× |
| regblock / warptile | `(M·K·N/128 + K·N·M/128)·4` | ~21× |
| cuBLAS | unknown — measure | ~1×? |

> Predicted in advance: the naive model implies 5.5e11 bytes in ~570 ms = **965
> GB/s, four times the T4's DRAM peak**. That is physically impossible, so L1/L2
> must already be absorbing most of it. The impossibility *is* the prediction
> being tested.

**Validation.** A 4096³ CPU reference is minutes, so the published shape is
covered two other ways rather than left unverified (`bias_act_bench` sets the
project standard: validate at the shape that produces the number). cuBLAS is
first validated against the double-accumulating CPU reference at **129×65×257 —
deliberately non-square**, because the row-major/column-major operand swap
produces a transposed result that still passes at M=N for a wide class of inputs;
having earned trust there it becomes the full-shape oracle for the other seven
rows. On top of that, 1024 random output elements are recomputed in `double` at
O(K) each. `beta ≠ 0` is validated at small shapes only and **never timed** —
across 20 timed iterations C would compound to `inf`.

**What was verified on the Mac before the trip**, since none of it needs a GPU:

- **The double-buffered k-loop schedule**, exhaustively for every tile count 0–200:
  every k-tile computed exactly once, from the buffer last published with it.
  This is the bug the trip could not have caught — K=4096 with BK=8 is **512
  tiles, an even count, so the benchmark shape never exercises the odd tail**,
  and a dropped final tile is a ~0.2% error that no tolerance would flag. (K=257
  → 33 tiles, which does exercise it.) Confirmed live: forcing `has_tail = false`
  produces 103 failures.
- **The bank arithmetic** above, as an integer property.
- **`sizeof(As) + sizeof(Bs)` against the host-side `smem_bytes()`** via
  `static_assert` inside each kernel — so the RESULTS.md smem column and the
  occupancy calculator cannot describe an allocation that does not exist. Works
  on the Mac because the fake-CUDA harness maps `__shared__`→`static`, which
  preserves `sizeof`. Confirmed live.
- **`reference_gemm` itself**, bit-for-bit against an independent Python oracle
  at 11 shapes including a non-square one.

**Rows 4–7 are one kernel template differing by one argument each**
(`LaneMap`, `DBUF`, `VW`), so the one-variable rule is enforced by the type
system rather than by discipline — there is no way for two changes to enter one
row when the rows share every line of code.

**Deliberately not built:** an **L2 block swizzle** (grouping `blockIdx` into 8×8
super-tiles). It is a grid-index remap only — provably one variable, no change to
the kernel body — and typically worth 5–10% on a 4 MiB L2. It is the cleanest
remaining rung between `warptile_vec4` and cuBLAS, and is named here with that
estimate as part of the required "remaining gap to cuBLAS" explanation rather
than measured.

---

## 4. Phase 4 — Scheduling — **Colab Tesla T4, 2026-09-10, driver 580.82.07, `mcke_graph_bench --streams=4`**

Numbers below are from a single fresh run (fresh Colab runtime, fresh
`git clone` at commit `d4e6826`, clean rebuild) — this table replaces an
earlier version of itself sourced from a stale/cached notebook cell, which is
why the numbers differ slightly (all well within run-to-run noise) from any
previously-reported figures for the same five graphs.

| Graph | Policy | streams | median ms | min ms | speedup vs sequential | peak memory | naive memory | numerics gate |
|---|---|---|---|---|---|---|---|---|
| fanout4x4 | sequential | 1/1 | 4.222 | 4.205 | 1.00× | 184,557,568 B | 285,220,864 B (1.55×) | — |
| fanout4x4 | level_parallel | 4/4 | 2.211 | 2.192 | 1.91× | 218,112,000 B | 285,220,864 B (1.31×) | — |
| fanout4x4 | chain_greedy | 4/4 | 2.175 | 2.152 | 1.94× | 218,112,000 B | 285,220,864 B (1.31×) | **PASS** (9 configs × 20 repeats, 71,305,216 elements) |
| diamond_starved | sequential | 1/1 | 3.488 | 3.470 | 1.00× | 536,887,296 B | 536,887,296 B (1.00×) | — |
| diamond_starved | level_parallel | 2/4 | 3.459 | 3.433 | 1.01× | 536,887,296 B | 536,887,296 B (1.00×) | — |
| diamond_starved | chain_greedy | 2/4 | 3.453 | 3.433 | 1.01× | 536,887,296 B | 536,887,296 B (1.00×) | **PASS** (9 configs × 20 repeats, 134,221,824 elements) |
| transformer_block | sequential | 1/1 | 21.802 | 20.930 | 1.00× | 201,342,976 B | 218,120,192 B (1.08×) | — |
| transformer_block | level_parallel | 1/4 | 21.912 | 21.745 | 0.99× | 201,342,976 B | 218,120,192 B (1.08×) | — |
| transformer_block | chain_greedy | 1/4 | 21.974 | 18.985 | 0.99× | 201,342,976 B | 218,120,192 B (1.08×) | **PASS** (9 configs × 20 repeats, 54,530,048 elements) |
| diamond_gemm_2048 | sequential | 1/1 | 14.477 | 13.933 | 1.00× | 83,886,080 B | 83,886,080 B (1.00×) | — |
| diamond_gemm_2048 | level_parallel | 2/4 | 14.327 | 14.161 | 1.01× | 83,886,080 B | 83,886,080 B (1.00×) | — |
| diamond_gemm_2048 | chain_greedy | 3/4 | 14.541 | 14.264 | 1.00× | 83,886,080 B | 83,886,080 B (1.00×) | **PASS** (9 configs × 20 repeats, 20,971,520 elements) |
| chain16 | sequential | 1/1 | 8.809 | 8.800 | 1.00× | 268,451,840 B | 1,140,867,072 B (4.25×) | — |
| chain16 | level_parallel | 1/4 | 9.090 | 9.080 | 0.97× | 268,451,840 B | 1,140,867,072 B (4.25×) | — |
| chain16 | chain_greedy | 1/4 | 9.080 | 9.068 | 0.97× | 268,451,840 B | 1,140,867,072 B (4.25×) | **PASS** (9 configs × 20 repeats, 285,216,768 elements) |

`chain16`'s `level_parallel`/`chain_greedy` land slightly *under* 1.0×
(0.97×) despite genuine 4-way streaming (4.25× memory savings from liveness
reuse) — 16 sequential GEMMs of the same size leave no per-kernel idle SMs to
overlap into, so the parallel schedules just pay event overhead with nothing
to win back. `diamond_gemm_2048` sits at a flat ~1.00–1.01× for the same
reason (each GEMM alone saturates the T4). Both are further confirmation of
the same prediction fanout4x4 (1.94×) breaks: overlap only pays when kernels
are small/independent enough to leave the GPU idle.

**Result:** all five gated graphs (`fanout4x4`, `diamond_starved`,
`transformer_block`, `diamond_gemm_2048`, `chain16`) pass the numerics gate —
`kLevelParallel` and `kChainGreedy` are bit-identical to `kSequential` across
all 9 `(schedule × memory policy)` combinations, 20 repeats each. This
followed two real bugs found and fixed (see `PROJECT_LOG.md` Session 7): a
use-after-free in `GraphExecutor::set_input()`'s numerics-gate replay path,
and a mismatched-index (raw position vs. `TensorId`) comparison bug in
`validate_numerics()`'s per-tensor diagnostic.

**`fanout4x4`** is the one graph where overlap actually pays off: 4-way
fan-out has real independent work to interleave, and `chain_greedy` gets
1.94× with only 1.31× the naive memory footprint (vs. 1.55× for the
allocate-per-tensor baseline) — the liveness-based reuse buys memory back
without giving up the overlap win.

**`diamond_starved` and `transformer_block` show ~1.00×** — the predicted
non-result. `diamond_starved`'s branches are memory-bandwidth-starved on
purpose (already saturating the GPU independently, so there's nothing to
overlap), and `transformer_block`'s ops are large GEMMs that already occupy
all SMs; `level_parallel`/`chain_greedy` even round-trip slightly under 1.0×
(0.99×) from event-management overhead with no overlap benefit to offset it.
This confirms the wave-sweep finding below: overlap only helps when the
underlying kernels leave the GPU idle.

Also record: events recorded per iteration, host time in `run_async()`, and
memory saved by `kReuseHappensBefore` vs `kAllocPerTensor`
(`ExecutionPlan::peak_memory_bytes()` vs `naive_memory_bytes()`) — captured
per-graph in the table above; `launch_bound_ratio` (enqueue/device time) was
≤0.029 for every graph/policy, i.e. every run here is device-bound, not
launch-bound.

### Wave sweep — the 2026-09-07 "2.00× at N=1024" anomaly, investigated

The original prediction (recorded before any run): *"Speedup should fall
monotonically and cross ~1.5× near one wave."* A single 2026-09-07 run instead
showed a **non-monotone** curve with a **2.00× point at N=1024** — a number
that is structurally impossible under the obvious model (`diamond_gemm(n)` is
two independent GEMMs B, C feeding one dependent GEMM D; if each takes time
`t`, sequential = 3t and the best case is `max(t_B,t_C) + t_D = 2t`, a **1.5×
ceiling**). Rather than reword the prediction to fit one run, this was
investigated with four tools added to `bench/graph_bench.cpp`: a `--gemm-n=N`
flag plus `diamond_gemm_custom`/`single_gemm_custom` graphs (run a diamond or
an isolated single GEMM at an arbitrary N instead of only the pinned 2048),
per-node timing printed under `--profile` (so a GEMM's own time under
`kSequential` can be compared against its time under `kChainGreedy`
directly), and the wave-sweep table now measuring `blocks/SM` itself via
`cudaFuncGetAttributes` + `occupancy_blocks_per_sm` rather than asserting a
copy-pasted Phase 3d number.

**Per-node timing at N=1024** (`--gemm-n=1024 --profile --only=diamond_gemm_custom`,
Colab T4, 2026-09-10):

| Policy | B_gemm min ms | C_gemm min ms | D_gemm min ms | graph median ms |
|---|---|---|---|---|
| sequential | 0.890 | 0.890 | 0.889 | 2.693 |
| chain_greedy | 0.748 | 0.753 | 0.705 | 2.046 |

Each GEMM's own **minimum** time drops from ~0.89 ms to ~0.75 ms under
concurrency — a genuine ~16% per-kernel speedup, not just a scheduling
artifact. **Mechanism:** at N=1024 a single `tiled_regblock` GEMM launches 64
blocks, and this T4 was measured (below) at 2 blocks/SM × 40 SMs = 80
blocks/full wave — one GEMM alone (64/80 = 0.8 waves) cannot fill the machine.
Running two concurrently (128 blocks) gives the scheduler more independent
work to hide latency behind, so `t` itself is lower under concurrency and the
constant-`t` assumption behind the 1.5× ceiling doesn't hold in this
sub-wave regime. This is genuine partial support for hypothesis (a)
("concurrency raises per-kernel efficiency") — but ~16% cannot by itself
explain a reported 2.00× speedup.

**Isolated single-GEMM cross-check** (`--gemm-n=1024 --only=single_gemm_custom`):
one `tiled_regblock` GEMM at N=1024, alone, sequential: **median 0.927 ms**.
3× that is 2.78 ms, matching `diamond_gemm_custom`'s own sequential number
(2.69 ms, above) to within 3% — **hypothesis (b) is refuted**: the diamond's
sequential measurement is not an anomaly, it is exactly consistent with three
independent GEMMs. The 2026-09-07 run's sequential number at N=1024 (3.704 ms)
was itself the outlier, not a real property of the graph.

**Reproducibility — four independent runs, two different Colab VM instances**
(`--only=wave_sweep`, each 5 warmup + 20 timed iterations):

| N | waves | run 1 | run 2 | run 3 | run 4 (fresh VM) |
|---|---|---|---|---|---|
| 256 | 0.05 | 1.49× | 1.46× | 1.42× | 1.47× |
| 512 | 0.20 | 1.35× | 1.34× | 1.35× | 1.35× |
| 768 | 0.45 | 1.13× | 1.06× | 1.07× | **1.87×** |
| 1024 | 0.80 | 1.09× | **1.55×** | **1.66×** | 1.05× |
| 1536 | 1.80 | 1.02× | 1.03× | 1.01× | 1.02× |
| 2048 | 3.20 | 1.02× | 1.01× | 1.02× | 1.04× |
| 4096 | 12.80 | 0.99× | 0.99× | 0.99× | 0.99× |

Every point except N=768 and N=1024 is stable to within ~2% across all four
runs — including run 4, on a **completely fresh Colab VM instance** (fresh
clone, fresh rebuild), which rules out anything specific to one physical GPU
or one runtime's thermal state. N=768 (0.45 waves, 36+36=72 combined blocks)
and N=1024 (0.80 waves, 64+64=128 combined blocks) are the two points where
two concurrent GEMMs' combined block count is close enough to the 80-slot
full-wave capacity that **which of the two kernels' blocks land in the first
available SM slots is sensitive to launch-order/scheduling jitter** — small
run-to-run differences in that residency pattern change how much the two
GEMMs actually overlap, hence the wide spread (1.05×–1.87× at N=1024 alone).
At N=512 (32 combined blocks) there's enough headroom that residency order
doesn't matter; at N=1536+ each single GEMM already dominates a full wave on
its own, so the marginal overlap is small and stable regardless of order.

**Conclusion:** the original prediction's *wording* ("falls monotonically")
was wrong — the real curve has a **noisy bump** in the 0.45–0.8 wave range
sitting on top of an otherwise smooth decay from ~1.45× (0.05 waves) to ~1.0×
(≥1.8 waves).

*Stated precisely, because an earlier version of this paragraph said the bump
was "visible across all four runs" and the table does not support that.* The
bump appears in **3 of 4 runs**, at **exactly one of the two adjacent sample
points, never both** (runs 2 and 3 at N=1024; run 4 at N=768), and is
**absent from run 1**: interpolating the smooth decay between 0.20 waves
(1.35×) and 1.80 waves (1.02×) predicts roughly 1.22× at 0.45 waves and 1.10×
at 0.80, and run 1's 1.13× / 1.09× sit at or just below that line — i.e. run 1
*is* the monotone curve the original prediction described.

That is a weaker claim about universality and a **stronger** one about
mechanism. A bump that migrates between adjacent sample points and sometimes
fails to appear at all is much better evidence for launch-order/residency
sensitivity than a bump reproducing at a fixed N would be: a fixed bump would
point at something structural about that specific block count, whereas a
mobile, intermittent one is what jitter looks like.

Hypothesis (a) (concurrency raises per-kernel
efficiency in the sub-wave regime) is real but small (~16%) and does not
alone explain any single run's peak; hypothesis (b) (the sequential
measurement itself was anomalous) explains the specific 2026-09-07 data
point; the dominant honest explanation is hypothesis (c) — a real,
reproducible scheduling-jitter sensitivity localized to the 0.45–0.8 wave
range, not a general property of the whole curve. No single number at
N=768/1024 should be treated as *the* speedup for this graph at that size —
report the range.

```
occupancy: 2 blocks/SM (measured via cudaFuncGetAttributes on this build's
           actual tiled_regblock kernel) x 40 SMs = 80 blocks/full wave
```

This matches the Phase 3d figure exactly, so the "2 blocks/SM, 80/160-block
full wave" claim used throughout Phase 3/4 is now measured by this binary
directly, not just carried over as prose.

---

## 5. Nsight Compute deep dives

One subsection per kernel actually profiled. Required fields:
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`,
`achieved_occupancy`, top warp-stall reason, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
(→ sectors per request, the coalescing measure), and
`dram__bytes_read.sum` compared to our ideal bytes.

Note on names: `achieved_occupancy` is the legacy metric spelling; the modern
counter is `sm__warps_active.avg.pct_of_peak_sustained_active`, which is what
`docs/PROFILING.md` §3 lists and what to actually pass to `ncu --metrics`.

### 5a. GEMM ladder (Phase 3d) — blocked on ncu access, 2026-08-31

One `ncu` run per variant via `mcke_gemm_bench --only=<variant>`. Profiling all
eight in one process does not work: `--kernel-name regex:gemm` also matches
cuBLAS's own `turing_sgemm_*`, so `--launch-count 3` would profile three launches
of whichever kernel happened to come first out of ~200.

**Attempted on Explorer (Tesla V100-SXM2, `gpu-interactive` partition,
`ncu` 2023.3.0.0 shipped with the `cuda/12.3.0` module) and blocked:**

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

A driver-level restriction (`NVreg_RestrictProfilingToAdminUsers`), not fixable
from inside a job — Explorer's own documentation does not mention Nsight
Compute at all (only Nsight *Systems*), and does not address this permission
model either. Fixing it needs an RC ticket (`rchelp@northeastern.edu`), which
has not been filed yet — deliberately deferred (see `DECISIONS.md`,
2026-08-31) rather than blocking Phase 3's writeup on it. **Not attempted on
Colab** — `docs/ENVIRONMENTS.md` already documents that Colab's container
lacks the profiling permissions this needs. **Not yet attempted on the RTX
5060.**

Everything in this table below the header remains unmeasured. The rest of the
Phase-3d exit writeup (immediately below) works from what §3d's non-`ncu`
methodology already established, and marks each remaining question as open
rather than guessed at.

| Variant | sm__throughput % | dram__throughput % | occupancy (hand / API / ncu) | top stall reason | sectors/request | dram_bytes_read vs compulsory |
|---|---|---|---|---|---|---|
| naive_uncoalesced | | | | | | |
| naive | | | | | | |
| tiled_smem | | | | | | |
| tiled_regblock | | | | | | |
| warptile_nodbuf | | | | | | |
| warptile_dbuf | | | | | | |
| warptile_vec4 | | | | | | |
| cuBLAS | | | — | | | |

The two comparisons this table exists for, both pinned in `docs/PROFILING.md` §4
and predicted in §3d above: hand-computed occupancy vs measured, and modelled
bytes vs `dram__bytes_read.sum`.

_(no kernels profiled yet — blocked, see above)_

### 5b. Phase 3 exit writeup: the remaining gap to cuBLAS

`docs/ROADMAP.md`'s Phase-3 exit criterion is "a written explanation of the
remaining gap to cuBLAS." Written honestly given the current evidence: some of
this gap is measured and explained; the rest is a narrowed set of open
questions, not a guess dressed up as an answer.

**What is confirmed, from two independent architectures and no `ncu`
required:**

- The attribution order is portable across generations: the same seven
  transitions (coalescing → shared staging → register blocking → lane
  permutation → double buffer → vectorized loads → cuBLAS) hold on both the
  T4 and the V100, with register blocking the single largest jump on both
  chips (T4: 4.17×/10.4%→40.4%; V100: 4.17×/18.0%→75.2% — the SAME ratio,
  independently).
- `warptile_nodbuf`'s regression is very likely **not a bug**: identical
  registers/smem/occupancy to `tiled_regblock` on both chips, a verified
  architecture-independent bank-conflict reduction
  (`test_gemm_bank_conflict_math`), and a result that goes from a confirmed
  −0.73pp regression on the T4 to flat (−0.10pp, noise-level) on the V100.
  A real bug in the permutation would not selectively vanish on a different
  chip while every other row's relative ordering holds.
- Double buffering and vectorized loads trade off in the *opposite* direction
  on the two chips (dbuf: +0.6pp T4 vs. +5.9pp V100; vec4: +2.3pp T4 vs.
  +0.7pp V100) in a way that is consistent with occupancy-dependent latency
  hiding mattering more on the V100 (25% occupancy, same absolute warp count
  as the T4's 50%) and instruction-issue rate mattering more on the T4. This
  is a plausible, internally consistent story — not yet a confirmed one,
  since confirming it needs `smsp__average_warps_issue_stalled_*` from `ncu`
  on both chips.
- The gap that remains is smaller in relative terms on the more capable,
  stably-clocked chip: `warptile_vec4` sits at 1.20× cuBLAS's time on the T4
  and 1.10× on the V100, at 89.6% cuBLAS efficiency with essentially zero
  thermal drift — the most trustworthy ceiling measurement this project has
  produced.

**What is still open, and requires `ncu` (or a future attempt on the RTX
5060) specifically:**

- Whether `tiled_smem`'s shortfall against its own 15–25% prediction (10.4%
  T4, 18.0% V100 — both below the modelled shared-bandwidth/DRAM-reuse
  ceilings) is barrier-stall time from the single-resident-block occupancy,
  as hypothesized in §3d, or something else. `smsp__average_warps_issue_
  stalled_barrier_per_issue_active.ratio` on this kernel specifically would
  settle it.
- The actual mechanism behind the dbuf/vec4 trade-off above — plausible, not
  measured.
- The final, quantified piece of `warptile_vec4`'s gap to cuBLAS: an **L2
  block swizzle** was named in §3d as the cleanest remaining, unbuilt lever
  (grid-index remap only, provably one variable, typically 5–10% on a
  resident-L2 GEMM) but was never measured, so it cannot yet be credited with
  any specific fraction of the remaining 1.10×–1.20×.

**Conclusion:** the exit criterion is partially met. The *shape* of the
remaining gap is explained and cross-validated on two architectures; its exact
*cause*, and how much of it an L2 swizzle would close, are not — and are not
being guessed at here rather than measured.

---

### 5c. Phase 4 exit writeup: what the graph engine bought, and what it cost

`docs/ROADMAP.md`'s Phase-4 exit criteria are: speedup per policy per graph;
events recorded per iteration; peak memory with and without liveness reuse; and
an `nsys` timeline showing actual overlap **or explaining its absence**. Three of
the four are met and tabulated in §4. The fourth is explained below and
partially open.

#### The headline: overlap paid off on exactly one of five graphs, and that is the result

| Graph | best speedup | why |
|---|---|---|
| `fanout4x4` | **1.94×** (chain_greedy) | four independent branches, each deliberately starved to 10 blocks — genuine idle SMs to interleave into |
| `diamond_gemm_2048` | 1.01× | each GEMM alone is 3.2 waves; the machine is already full |
| `diamond_starved` | 1.01× | SM-idle but **bandwidth**-saturated |
| `transformer_block` | 0.99× | width 1 — there is nothing to overlap, and events cost a little |
| `chain16` | 0.97× | width 1, same |

The prediction recorded before any of this ran was *"overlap helps only if the
individual kernels leave SMs idle. If B and C each already saturate the GPU,
expect ~1.0× — and that non-result, with the timeline showing why, is a
legitimate finding."* That is what happened, four times out of five.

**`diamond_starved` is the row that carries the phase.** Its branches are
`bias_act` kernels capped to 40 blocks — roughly 6% occupancy, so by every
occupancy metric the machine is idle. It still measured **1.01×**. Phase 3a had
already measured that same starved kernel, at the same 8192×4096 shape, at
**224.6 GB/s — 95.4% of the 235.4 GB/s measured DRAM ceiling**. So the SMs were
idle and the *memory system was not*, and overlap had nothing left to win.

> **Correction to the prediction recorded in `bench/graph_bench.cpp`.** That
> banner computed the ceiling as 235.4/208.7 = ~1.13×, reading the **vw1** row
> of §3a's starvation sweep. That is the wrong row: `BiasActOp` is constructed
> with `vector_width = 0`, which means "pick the widest legal width", and at
> 4096 columns that resolves to **vw4** — whose starved figure is 224.6 GB/s,
> not 208.7. The correct ceiling estimate is therefore **235.4/224.6 = 1.048×**,
> and the measured 1.01× sits within 4% of it rather than 12% below a looser
> bound. The prediction was right in kind and loose by a factor of ~2.6 in the
> headroom it claimed; using a kernel's *measured* bandwidth means using the
> configuration that actually runs.

Set against `fanout4x4`'s 1.94×, the pair says the thing neither row says alone:
**"SMs are idle" is not "the machine is idle."** Overlap pays only when the
*bottleneck* resource is idle, and occupancy does not tell you which resource
that is. This is the same lesson Phase 3d reached from the other direction, when
`tiled_smem` at 100% occupancy lost to `tiled_regblock` at 50%.

On D3's recorded prediction: the banner deliberately framed its number as a
**ceiling estimate and not a floor**, on the grounds that the arithmetic assumes
two concurrent kernels share DRAM cleanly and additively — which was the
assumption under test. That framing is what makes the measured 1.01× readable as
a confirmation rather than a miss; had ~1.13× (or the corrected ~1.048×) been
recorded as a lower bound, a correct result would have looked like a failure.
The band {≈ceiling, ≈1.0×, <1.0×} was stated in advance and 1.01× is in it.

#### Event counts: the header's central claim, verified and narrowed

`executor.hpp` originally claimed `kChainGreedy` "minimises event count." That is
not defensible as stated, and the corrected version — asserted as exact integers
by `tests/test_graph_host.cpp`, with no GPU — is:

| Graph | width | sequential | level_parallel | chain_greedy |
|---|---|---|---|---|
| diamond | 2 | 0 / 0 | 2 rec / 2 wait | 2 / 2 |
| chain16 | 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| fanout4×4 | 4 | 0 / 0 | **12 / 36** | **0 / 0** |

**Chain-greedy's event count scales with the number of cross-stream EDGES;
level-parallel's scales with LEVEL BOUNDARIES × STREAMS USED.** On the diamond
they tie at 2/2 — so the original "minimises" claim is false on the very graph
the header used to illustrate it. On `fanout4×4` chain-greedy pays nothing while
level-parallel manufactures 12 records and 36 waits on a graph with **zero
cross-stream data edges**, because its barrier is between levels rather than
between dependencies. It is a greedy heuristic that happens to be optimal on
every graph benchmarked here, not a proven minimum.

Two costs are reported separately from those figures, and must stay separate:
the per-iteration **fork/join** (K records + 2(K−1) waits) is a fixed cost of
the `run_async`/`synchronize` contract rather than anything attributable to a
policy, and folding it in would misattribute it.

**The event cost turned out to be immeasurable here.** `launch_bound_ratio`
(host enqueue time / device time) was **≤ 0.029 for every graph and every
policy** — every run is device-bound by a factor of at least 34. That is why
`chain16`'s and `transformer_block`'s parallel policies land at 0.97–0.99×
rather than dramatically worse: the events genuinely cost something, but on
these shapes the cost is a rounding error against the kernels. A launch-bound
graph is where chain-greedy's zero-event schedule would actually show up as
wall-clock, and none of the five graphs is launch-bound.

#### Memory: liveness reuse, and its direct tension with parallelism

| Graph | naive | reused | ratio |
|---|---|---|---|
| chain16 | 1,140,867,072 B | 268,451,840 B | **4.25×** |
| fanout4×4 (K=1) | 285,220,864 B | 184,557,568 B | 1.55× |
| fanout4×4 (K=4) | 285,220,864 B | 218,112,000 B | 1.31× |
| transformer_block | 218,120,192 B | 201,342,976 B | 1.08× |
| diamond_gemm / diamond_starved | — | — | 1.00× |

`chain16`'s 4.25× is exactly the hand-computed figure asserted in a host unit
test: 17 tensors of 64 MiB collapse to **four** buffers — the graph input (never
dies: it is filled by an async H2D whose completion the planner does not track),
the graph output (must outlive execution), and two ping-pong buffers for the 15
intermediates, because `t_i` and `t_{i+1}` overlap while `t_i` and `t_{i+2}` do
not.

The 1.00× rows are correct, not failures: on a diamond, A's output is read by
both branches so it spans both, and both branches' outputs live until the join.
Five tensors, five buffers, nothing to reuse.

**`fanout4x4` at 1.55× (K=1) versus 1.31× (K=4) is the measured form of a
prediction made before the run:** peak memory rises with stream count, because
concurrency destroys the *ordering* that makes reuse legal. Parallelism and
memory reuse are in direct tension. The practical consequence, now recorded in
the header: peak is a function of **(graph, schedule policy, memory policy,
num_streams)** — not of (graph, memory policy), as the original accessor
comments implied. A §4 row omitting the last two coordinates would be
uninterpretable later.

#### Design decisions, with the alternatives that were rejected

**SSA DAG with derived edges, not a declared edge list.** Each tensor records
its single producer and each node lists its input tensors; the dependency edge
is *implied*. The alternative — asking the user to declare "node 5 depends on
node 3" — permits an edge list that disagrees with the actual dataflow, and when
it does the result is a race that appears only under load. Deriving edges makes
that class of bug unrepresentable, and hands liveness def/use chains for free.

The stronger consequence emerged while implementing `finalize()`: because
`add_node` is the only way to create a non-input tensor, sets `producer` exactly
once, and can only reference tensors that already exist, **cycles and
multiple-producers are unrepresentable too.** `finalize()` cannot meaningfully
"verify acyclicity" as its original comment claimed — Kahn will always succeed.
Both checks remain, reclassified `kInternal` to say so, guarding a future
alias/mutation API. Two of the seven documented failure modes turned out to be
structurally impossible, which understates rather than overstates the design.

**Events, not stream-synchronize, for cross-stream dependencies.** A
`cudaStreamSynchronize` between dependent nodes would be correct and would
destroy the entire point: it blocks the *host*, so the CPU cannot run ahead to
enqueue the next node, and the runtime becomes synchronous while still looking
asynchronous. `cudaEventRecord`/`cudaStreamWaitEvent` creates a device-side
ordering the scheduler honours without the host waiting at all. Hence the
invariant that `Op::launch` must not synchronise, and the deliberate restriction
of host barriers to exactly two places (`GraphExecutor::synchronize()` and
benchmark timing).

That invariant also drove the workspace design. `OpContext` originally handed
ops a `DeviceAllocator*`, inviting `allocate()` on the per-iteration path —
where Phase 2 had already measured `cudaMalloc` at up to 720 µs with a
`cudaFree` that **synchronises the whole device**. One such call inside
`run_async` would silently undo the phase. It now carries a plain
`{void*, size_t}` into a per-stream arena, safe with no analysis because
same-stream issue is in order.

**A plan-time arena with static offsets, not runtime allocate/deallocate.**
This was the phase's largest decision and it was taken for a correctness reason
first, not a performance one. `Storage::note_use()` records a single stream, and
under a parallel schedule it records an *arbitrary* one — it is called by the
host at **enqueue** time, and the last-issued consumer routinely finishes first.
Feeding that to Phase 2's cross-stream reuse policies defeats `kCoarseStreamPoll`
*and* `kPerFreeEvent` equally: precision in the reclaim policy buys nothing when
its input is wrong. A static arena never asks the question. It also makes peak
memory deterministic rather than a function of pool fragmentation, which is what
makes the numbers above comparable across policies at all.

The consequence for Phase 2 is worth stating plainly, since it looks like a
demotion and is not: the pooling allocator's job becomes serving the arena, the
per-stream workspaces, and graph I/O. `run_async()` makes **zero** allocator
calls. That is precisely why production runtimes plan memory, and the project can
now point at the mechanism rather than assert it.

**Three schedule policies, all implemented, all measured.** `kSequential` exists
as the bit-exact correctness baseline, not as a strawman — every other policy is
compared against it by `memcmp`. `kLevelParallel`'s inter-level barrier is
implemented as a *true* barrier rather than weakened to per-edge waits: its cost
**is** the point of the policy, and softening it would turn it into
chain-greedy-without-the-heuristic and destroy the comparison. `kChainGreedy`
balances by **estimated roofline cost**, not node count — every graph here mixes
a ~10 ms GEMM with a ~0.2 ms softmax, and counting nodes calls them equal.

#### The correctness result that mattered more than any speedup

Liveness computed over a **topological order is unsound** under a multi-stream
schedule. A topological order is one arbitrary linear extension of the
dependency partial order; the executor's happens-before is that partial order
*plus stream-serialisation edges*. Both extend the dependencies, and they extend
them **differently** — so interval non-overlap asserts an ordering the executor
never established.

The minimal case is two unequal chains joined at the end (the
transformer-with-residual shape). Tensors `a` and `d` have disjoint live ranges
by topological position, so a linear-scan planner shares their buffer — and under
`kChainGreedy` nothing orders the two streams before the join, while under
`kLevelParallel` the inter-level barrier orders the *producers* and leaves a
reader and a writer unordered **inside the same level**. The barrier closes the
write-after-write and leaves the read/write open, which is exactly why "put a
barrier between levels" feels sufficient and is not.

This was found and fixed **before any GPU time was spent on it**, and it is
demonstrated rather than merely avoided: `kReuseTopoNaive` ships as a
deliberately-unsafe arm, and a host-side happens-before race checker
(Fidge–Mattern vector clocks, the ThreadSanitizer algorithm applied to a GPU
schedule) flags it with the offending tensor pair, the shared byte range, and the
two unordered clocks — on a laptop, deterministically. Across the fuzz matrix,
4,050 sound plans verified race-free and the naive arm raced in 32 cases; that
count is asserted `> 0`, because a demonstration that never fires is not a
demonstration.

**Why the static checker and the runtime numerics gate are both kept.** They
catch different classes. The checker proves the *plan* is race-free,
exhaustively and without hardware. The gate proves the *executor faithfully
implements the plan*, and catches what the checker cannot model — and it did
exactly that: the two real bugs of Session 7 (a use-after-free in `set_input()`'s
replay path, and a raw-position-vs-`TensorId` mix-up in the gate's own
diagnostic) were found by the gate, not the checker. Neither substitutes for the
other.

#### What is still open

- **The `nsys` timeline.** The one report that exists profiled the **pre-fix**
  binary and is flagged non-compliant/profiling-only in its own output; it must
  be regenerated against the fixed build before it can be cited. The *absence*
  of overlap on four of five graphs is nevertheless already explained above and
  quantified by `launch_bound_ratio ≤ 0.029` — the ROADMAP's criterion allows
  explaining the absence, and that part is met. What a clean timeline would add
  is direct visual confirmation of `fanout4x4`'s 1.94×, which is currently
  inferred from wall-clock plus event counts rather than seen.
- **The 0.45–0.8-wave jitter bump** is characterised (3 of 4 runs, migrating
  between adjacent sample points, absent from one run) but its mechanism —
  block-to-SM residency order under near-full-wave conditions — is a hypothesis
  consistent with the data, not a measurement. Confirming it needs per-SM
  residency information that `ncu` would provide and Explorer currently refuses
  (`ERR_NVGPUCTRPERM`, §5a).
- **`kReuseWithSyncEdges`**, the memory-vs-parallelism knob, is deliberately not
  built. Adding sync edges to make more reuse legal cannot be a post-pass:
  it changes the schedule, which changes happens-before, which changes what
  reuse is legal. Scheduling and allocation become a fixpoint — the
  register-allocator/instruction-scheduler co-design problem — and an enum value
  that always errors is worse than one that does not exist.
- **Launch-bound behaviour is unmeasured** in the sense that matters: no graph
  here is launch-bound, so chain-greedy's zero-event advantage never showed up
  in wall-clock. The experiment that would expose it is a chain of many tiny
  kernels; the tooling (`--gemm-n=`, per-node `--profile`) now exists for it.

**Conclusion:** three of four exit criteria are fully met with measured numbers
on real hardware. The fourth — the timeline — is met in its "or explaining its
absence" form and open in its "showing actual overlap" form. The phase's most
valuable output is not a speedup: it is that the central correctness trap was
found by construction on a laptop, demonstrated with a shipped unsafe arm, and
proven absent by two independent checkers before a GPU was involved.
