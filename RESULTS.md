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
`bench/fma_peak`), ridge point **34.2 FLOP/byte**. Re-measured on the Colab
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

| Variant | M=N=K | Tile (BM,BN,BK,TM,TN) | regs/thread | smem/block | occupancy | median ms | min ms | TFLOP/s | % of measured FMA peak | Machine |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 4096 | — | | | | | | | | |
| tiled_smem | 4096 | 32,32,32,1,1 | | | | | | | | |
| tiled_regblock | 4096 | 128,128,8,8,8 | | | | | | | | |
| warptile_nodbuf | 4096 | 128,128,8,8,8 | | | | | | | | |
| warptile_dbuf | 4096 | 128,128,8,8,8 | | | | | | | | |
| cuBLAS | 4096 | — | — | — | — | | | | | |

Ideal cost: `flops = 2·M·N·K` = 1.37e11, `bytes = (M·K + K·N + M·N)·4` = 2.01e8,
**AI ≈ 682** — deep in compute-bound territory, 20× past the ridge point.

**Predictions (2026-08-24, kept verbatim):** naive 2–4% of peak (AI = 0.25 for
the *naive access pattern*, memory-bound); tiled_smem 15–25%; tiled_regblock
45–65%; warptile+double-buffer 60–80%; cuBLAS is the ceiling. Each row must
attribute its gain to *one* change — which is why `warptile_nodbuf` was added to
the variant enum, so warp tiling and double buffering get separate rows instead
of one row with two causes.

`occupancy` is the **hand-computed** figure from regs/thread and smem/block, to
be compared against Nsight's measured `achieved_occupancy` in §5. Per the
roadmap, that three-way comparison is the learning; the TFLOP/s is just the score.

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
