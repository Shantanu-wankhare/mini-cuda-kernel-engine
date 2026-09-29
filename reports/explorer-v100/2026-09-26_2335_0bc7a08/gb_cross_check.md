# Google Benchmark adapter cross-check — Explorer V100 (2026-09-29)

Phase 5 stage 5h bonus item. `gb_adapter.hpp`'s two claims tested against the
same real dataset committed alongside this file
(`reports/explorer-v100/2026-09-26_2335_0bc7a08/`), on the second real GPU
architecture this project has run the adapter on (the first was implicit in
its own host-only self-tests; no Colab run has happened yet as of this
commit).

Same build as the rest of this dataset (`git 0bc7a08`), reconfigured with
`-DMCKE_GOOGLE_BENCHMARK=ON` (fetched Google Benchmark v1.9.5 at the pinned
SHA `192ef100`, confirmed by the configure log) and rebuilt `-j1` (this
allocation only ever had 1 CPU). Not run through `scripts/regen_results.sh` —
this is a supplementary check, not one of the fenced RESULTS.md tables, so it
has no `manifest.json` of its own; the environment/denominators/git state are
identical to the sibling dataset in this same directory.

## Claim 1: does `kBurst` reproduce `Profiler::time_op`?

`kBurst`'s medians against the `time_op` medians already committed in this
same dataset (`reduce_bench.stdout.log`, `bias_act_bench.stdout.log`):

| Config | `time_op` (this dataset) | GB `kBurst` median | Diff |
|---|---|---|---|
| `row_reduce_sum`/`warp_shuffle_256t` (saturated, 8192×4096) | 0.181 ms | 0.185 ms | +2.2% |
| `row_reduce_sum`/`warp_shuffle_256t_starved` (64×524288) | 0.784 ms | 0.816 ms | +4.1% |
| `bias_gelu_tanh`/`fused_vw4` (full, 8192×4096) | 0.333 ms | 0.332 ms | +0.3% |
| `bias_gelu_tanh_L2`/`fused_vw4_512x512` (L2-resident) | 0.006 ms | 6.14 µs | ~0% |

All four agree within ordinary run-to-run noise (comparable in size to the
~0.3–1% noise already seen between the original T4 §3d numbers and their own
fresh Colab-side re-measurements). None show the 1000× (missing `* 1e-3`) or
near-zero (missing `->UseManualTime()`) signature that would indicate a broken
adapter. Every benchmark name below carries `/manual_time` (or
`manual_time_{mean,median,stddev,cv}` for the aggregate rows), confirming
manual timing was active throughout, not silently falling back to host
wall-clock. **Claim 1 holds on a second real GPU.**

## Claim 2 (informal, not part of the adapter's design claims): does `kPerIter`
show the per-iteration host-sync overhead `profiler.hpp`'s own header
comment predicts (~10 µs, "inflate short kernels enormously")?

| Config | `kBurst` median | `kPerIter` median | Delta |
|---|---|---|---|
| reduce, saturated | 185 µs | 181 µs | **−2.2%** (faster) |
| reduce, starved | 816 µs | 790 µs | **−3.2%** (faster) |
| bias_act, full | 332 µs | 332 µs | ~0% |
| bias_act, L2-resident (shortest kernel, ~6 µs) | 6.14 µs | 6.19 µs | +0.8% |

**Genuinely surprising, and recorded as an honest result rather than
explained away under time pressure:** no config shows the predicted
overhead. Two are *faster* under `kPerIter`, one is flat, and the shortest
kernel — where a fixed per-iteration cost should be most visible relative to
the kernel's own duration — shows only 0.8%, nowhere near "enormous." This
does not falsify `profiler.hpp`'s reasoning (a sync per iteration is still
architecturally a different measurement than one sync per burst, and remains
the wrong hot-path pattern for a graph executor regardless of its measured
cost here) — but it is evidence that on this specific system (V100,
driver 545.23.08, CUDA 12.3, a `cudaEventSynchronize` on a single timing
event specifically, not a full stream sync) the per-iteration round trip is
much cheaper than the general "~10 µs host round trip" the comment describes.
**Needs a Colab T4 comparison point before drawing any general conclusion** —
this is the first real-GPU exercise of `kPerIter` anywhere in the project;
5g hasn't run yet.

> **Correction (2026-09-28, main chat) — Claim 2's inference is the wrong
> mechanism; its data stands.** `kPerIter` records the stop event and *only
> then* synchronizes (`bench/gb_adapter.hpp`), so the host round trip happens
> **outside** the device-event bracket. Its reported Time cannot contain the
> round trip at all. The −3.2%…+0.8% deltas above therefore say nothing about
> how cheap the round trip is. They measure only the second-order effects that
> *can* reach the bracket: launch latency leaking in, since the start event is
> timestamped on an idle stream, and idle-state effects such as clocks and L2.
> What they do show is still useful: those effects are small on V100, ≤3% and
> about 50 ns on the 6 µs kernel, so per-launch **event** timing is robust to
> a per-iteration sync. The original design had the same blind spot: the
> adapter's own banner claimed this mode "quantifies what that sync costs". The
> adapter now also reports `host_us` (wall time per iteration) and
> `roundtrip_us` (host minus device), which measure the round trip directly.
> Stage 5g's T4 run is the first to collect them. The text above is kept
> as written.

## Raw output

<details>
<summary><code>./build/bin/mcke_reduce_bench_gb</code></summary>

```
=== environment ============================================
device        Tesla V100-SXM2-32GB (sm_70), 80 SMs, 96 KiB smem/SM
denominators  peak_gb_s=235.4  peak_tflops=8.130  (MEASURED, not spec)
ridge point   34.5 FLOP/byte -- below this a kernel is memory-bound
              spec-formula bandwidth for reference: 898.0 GB/s
build         MCKE_WITH_CUDA=1
============================================================

validation    warp_shuffle/sum (pre-GB)    OK    (max_rel_err 0.000764 vs tol 1e-05 -- sqrt(4096)*f32 eps)
2026-09-29T00:49:08-04:00
Running ./build/bin/mcke_reduce_bench_gb
Run on (28 X 3299.92 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x28)
  L1 Instruction 32 KiB (x28)
  L2 Unified 1024 KiB (x28)
  L3 Unified 19712 KiB (x2)
Load Average: 7.34, 7.41, 7.46
mcke_cmdline: ./build/bin/mcke_reduce_bench_gb
mcke_peak_gb_s: 235.400000
mcke_peak_tflops: 8.130000
mcke_timing: kBurst rows: Profiler::time_op (RESULTS.md-comparable). kPerIter rows: GB native, host sync per iteration -- NOT comparable, see gb_adapter.hpp.
***WARNING*** ASLR is enabled, the results may have unreproducible noise in them.
-------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                     Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------------------------
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               185 us         4696 us            1 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               184 us         4669 us            1 AI=0.249878 GB_s=728.356G/s TFLOP_s=0.182/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               185 us         4676 us            1 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.183296 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               185 us         4672 us            1 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.183296 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               184 us         4673 us            1 AI=0.249878 GB_s=728.356G/s TFLOP_s=0.182/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               185 us         4675 us            1 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               185 us         4680 us            1 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               184 us         4674 us            1 AI=0.249878 GB_s=728.356G/s TFLOP_s=0.182/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               184 us         4675 us            1 AI=0.249878 GB_s=728.356G/s TFLOP_s=0.182/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time               185 us         4680 us            1 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time_mean          185 us         4677 us           10 AI=0.249878 GB_s=725.941G/s TFLOP_s=0.181397/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.184115 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time_median        185 us         4675 us           10 AI=0.249878 GB_s=724.332G/s TFLOP_s=0.180994/s attainable_TFLOP_s=0.0588213 min_ms_last_burst=0.18432 warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time_stddev      0.529 us         7.49 us           10 AI=0 GB_s=2.078G/s TFLOP_s=519.247u/s attainable_TFLOP_s=0 min_ms_last_burst=431.76u warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/saturated/iterations:1/repeats:10/manual_time_cv           0.29 %          0.16 %            10 AI=0.00% GB_s=0.29% TFLOP_s=0.29% attainable_TFLOP_s=0.00% min_ms_last_burst=0.23% warp_shuffle_256t [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 853 us        21352 us            1 AI=0.249999 GB_s=157.35G/s TFLOP_s=0.0393373/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.843776 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 853 us        21250 us            1 AI=0.249999 GB_s=157.35G/s TFLOP_s=0.0393373/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.816128 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 813 us        20373 us            1 AI=0.249999 GB_s=165.078G/s TFLOP_s=0.0412694/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.80896 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 817 us        20417 us            1 AI=0.249999 GB_s=164.251G/s TFLOP_s=0.0410626/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.806912 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 817 us        20468 us            1 AI=0.249999 GB_s=164.251G/s TFLOP_s=0.0410626/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.805888 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 816 us        20385 us            1 AI=0.249999 GB_s=164.457G/s TFLOP_s=0.0411141/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.807968 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 816 us        20400 us            1 AI=0.249999 GB_s=164.457G/s TFLOP_s=0.0411141/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.797696 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 814 us        20354 us            1 AI=0.249999 GB_s=164.871G/s TFLOP_s=0.0412175/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.801792 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 815 us        20392 us            1 AI=0.249999 GB_s=164.664G/s TFLOP_s=0.0411658/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.805888 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time                 816 us        20446 us            1 AI=0.249999 GB_s=164.457G/s TFLOP_s=0.0411141/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.804864 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time_mean            823 us        20584 us           10 AI=0.249999 GB_s=163.119G/s TFLOP_s=0.0407795/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.809987 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time_median          816 us        20408 us           10 AI=0.249999 GB_s=164.457G/s TFLOP_s=0.0411141/s attainable_TFLOP_s=0.0588498 min_ms_last_burst=0.8064 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time_stddev         15.8 us          380 us           10 AI=0 GB_s=3.05138G/s TFLOP_s=762.842u/s attainable_TFLOP_s=1.38833n min_ms_last_burst=0.0127892 warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_burst/starved/iterations:1/repeats:10/manual_time_cv             1.92 %          1.85 %            10 AI=0.00% GB_s=1.87% TFLOP_s=1.87% attainable_TFLOP_s=0.00% min_ms_last_burst=1.58% warp_shuffle_256t_starved [kBurst 5w+20i, one sync per burst]
bm_reduce_periter/saturated/iterations:20/manual_time                       181 us          188 us           20 AI=0.249878 GB_s=740.498G/s TFLOP_s=0.185034/s attainable_TFLOP_s=0.0588213 warp_shuffle_256t [kPerIter, HOST SYNC PER ITERATION -- NOT comparable to RESULTS.md rows]
bm_reduce_periter/starved/iterations:20/manual_time                         790 us          795 us           20 AI=0.249999 GB_s=169.827G/s TFLOP_s=0.0424566/s attainable_TFLOP_s=0.0588498 warp_shuffle_256t_starved [kPerIter, HOST SYNC PER ITERATION -- NOT comparable to RESULTS.md rows]
```

</details>

<details>
<summary><code>./build/bin/mcke_bias_act_bench_gb</code></summary>

```
=== environment ============================================
device        Tesla V100-SXM2-32GB (sm_70), 80 SMs, 96 KiB smem/SM
denominators  peak_gb_s=235.4  peak_tflops=8.130  (MEASURED, not spec)
ridge point   34.5 FLOP/byte -- below this a kernel is memory-bound
              spec-formula bandwidth for reference: 898.0 GB/s
build         MCKE_WITH_CUDA=1
============================================================

validation    fused/gelu_tanh/vw4 (pre-GB) OK    (max_rel_err 0.000824 vs tol 1e-06 -- device tanhf vs host std::tanh)
2026-09-29T00:49:10-04:00
Running ./build/bin/mcke_bias_act_bench_gb
Run on (28 X 3299.91 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x28)
  L1 Instruction 32 KiB (x28)
  L2 Unified 1024 KiB (x28)
  L3 Unified 19712 KiB (x2)
Load Average: 7.34, 7.41, 7.46
mcke_cmdline: ./build/bin/mcke_bias_act_bench_gb
mcke_peak_gb_s: 235.400000
mcke_peak_tflops: 8.130000
mcke_timing: kBurst rows: Profiler::time_op (RESULTS.md-comparable). kPerIter rows: GB native, host sync per iteration -- NOT comparable, see gb_adapter.hpp.
***WARNING*** ASLR is enabled, the results may have unreproducible noise in them.
-----------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                         Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------------------------------------------------
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      333 us         8410 us            1 AI=1.24992 GB_s=806.646G/s TFLOP_s=1.00825/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.330752 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8348 us            1 AI=1.24992 GB_s=809.136G/s TFLOP_s=1.01136/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.331776 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      333 us         8346 us            1 AI=1.24992 GB_s=806.646G/s TFLOP_s=1.00825/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.330752 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8346 us            1 AI=1.24992 GB_s=809.136G/s TFLOP_s=1.01136/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.331776 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      333 us         8353 us            1 AI=1.24992 GB_s=806.646G/s TFLOP_s=1.00825/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.331776 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8350 us            1 AI=1.24992 GB_s=809.058G/s TFLOP_s=1.01126/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.330752 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8348 us            1 AI=1.24992 GB_s=809.136G/s TFLOP_s=1.01136/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.330816 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8346 us            1 AI=1.24992 GB_s=809.136G/s TFLOP_s=1.01136/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.330752 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8348 us            1 AI=1.24992 GB_s=808.98G/s TFLOP_s=1.01116/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.331776 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time                      332 us         8348 us            1 AI=1.24992 GB_s=808.98G/s TFLOP_s=1.01116/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.331776 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time_mean                 332 us         8354 us           10 AI=1.24992 GB_s=808.35G/s TFLOP_s=1.01038/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.33127 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time_median               332 us         8348 us           10 AI=1.24992 GB_s=809.019G/s TFLOP_s=1.01121/s attainable_TFLOP_s=0.294232 min_ms_last_burst=0.331296 fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time_stddev             0.484 us         19.7 us           10 AI=0 GB_s=1.17726G/s TFLOP_s=1.47148m/s attainable_TFLOP_s=5.55333n min_ms_last_burst=533.293u fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/full/iterations:1/repeats:10/manual_time_cv                  0.15 %          0.24 %            10 AI=0.00% GB_s=0.15% TFLOP_s=0.15% attainable_TFLOP_s=0.00% min_ms_last_burst=0.16% fused_vw4 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          231 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          217 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          216 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          212 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          214 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          215 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          217 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          218 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          217 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time              6.14 us          221 us            1 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time_mean         6.14 us          218 us           10 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time_median       6.14 us          217 us           10 AI=1.24878 GB_s=341.667G/s TFLOP_s=0.426667/s attainable_TFLOP_s=0.293963 min_ms_last_burst=5.12m fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time_stddev      0.000 us         5.23 us           10 AI=0 GB_s=0/s TFLOP_s=0/s attainable_TFLOP_s=0 min_ms_last_burst=0 fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_burst/l2_resident/iterations:1/repeats:10/manual_time_cv           0.00 %          2.40 %            10 AI=0.00% GB_s=0.00% TFLOP_s=0.00% attainable_TFLOP_s=0.00% min_ms_last_burst=0.00% fused_vw4_512x512 [kBurst 5w+20i, one sync per burst]
bm_bias_act_periter/full/iterations:20/manual_time                              332 us          338 us           20 AI=1.24992 GB_s=807.776G/s TFLOP_s=1.00966/s attainable_TFLOP_s=0.294232 fused_vw4 [kPerIter, HOST SYNC PER ITERATION -- NOT comparable to RESULTS.md rows]
bm_bias_act_periter/l2_resident/iterations:20/manual_time                      6.19 us         13.2 us           20 AI=1.24878 GB_s=339.193G/s TFLOP_s=0.423578/s attainable_TFLOP_s=0.293963 fused_vw4_512x512 [kPerIter, HOST SYNC PER ITERATION -- NOT comparable to RESULTS.md rows]
```

</details>
