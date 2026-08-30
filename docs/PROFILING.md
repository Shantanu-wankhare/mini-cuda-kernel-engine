# PROFILING.md — what to measure, and with which tool

## 1. Pick the right tool

| Question | Tool | Cost |
|---|---|---|
| How long did this kernel take? | CUDA events (`Profiler::time_op`) | ~0.5 µs, always on |
| Is the GPU idle? Did streams overlap? Is the CPU starving the GPU? | **Nsight Systems** (`nsys`) | low overhead, whole-timeline |
| Why is *this one kernel* slow? Occupancy, stalls, memory hierarchy, SASS | **Nsight Compute** (`ncu`) | replays each kernel many times — very slow |
| How many registers / how much shared memory does this kernel use? | `nvcc -Xptxas -v` at build time | free |

Using `ncu` to answer a scheduling question, or `nsys` to answer an occupancy
question, is the classic way to lose an afternoon.

## 2. The denominators — establish these FIRST (Phase 1)

Every "% of peak" needs a stated denominator. Get both by measurement, not from a
spec sheet:

- **Measured bandwidth** — `bench/stream_triad`: `a[i] = b[i] + s*c[i]` over a
  working set ≫ L2. Typically 80-90% of the spec-formula number
  (`DeviceInfo::peak_dram_gb_s()`); the gap is ECC, refresh, and real clocks.
- **Measured f32 FMA peak** — `bench/fma_peak`: a long dependency-free chain of
  `fmaf` in registers, no memory traffic, enough independent chains to saturate
  the pipes. This already includes whatever clock the GPU sustains under load,
  which a spec sheet does not.

**Ridge point** = measured_FMA_peak / measured_bandwidth [FLOP/byte]. Kernels with
AI below it are memory-bound: for those, **report GB/s, not TFLOP/s**.

Reference AIs for our kernels (f32):

| Kernel | flops | ideal bytes | AI | Verdict |
|---|---|---|---|---|
| vector add | N | 12N | 0.083 | memory-bound |
| bias+GELU fused | ~8N | 8N | ~1.0 | memory-bound |
| row reduce (sum) | N | 4N | 0.25 | memory-bound |
| softmax (1-pass) | ~5N | 8N | ~0.6 | memory-bound |
| GEMM M=N=K=4096 | 2N³ = 1.37e11 | 3N²·4 = 2.0e8 | **~682** | compute-bound |

That table is the whole reason GEMM gets four optimisation variants and vector
add gets one.

## 3. Nsight Systems — timeline / scheduling

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true \
  --output=reports/nsys_graph_$(date +%Y%m%d_%H%M) \
  ./build/bin/mcke_graph_bench --policy=chain_greedy
```

What to look for, in order:
1. **Gaps between kernels.** A gap with a busy CPU means launch-bound: the host
   cannot enqueue fast enough. Fix with fewer/larger kernels, a precomputed plan,
   or CUDA Graphs.
2. **Are kernels on different streams actually concurrent?** If the rows are
   staircased rather than overlapping, suspect (a) default-flag streams
   serialising against the NULL stream, (b) an unnecessary
   `cudaStreamSynchronize`, (c) an allocator call synchronising, or (d) the
   kernels genuinely saturating the SMs so the scheduler won't co-resident them.
3. **memcpy overlapping compute?** Requires pinned host memory *and* a separate
   copy engine (`DeviceInfo::async_engine_count`).

NVTX ranges (`MCKE_USE_NVTX=ON`) are what make this readable — without them you
get anonymous kernel bars and cannot tell which graph node is which.

## 4. Nsight Compute — single-kernel deep dive

**One variant per `ncu` invocation.** This is not a style preference — profiling
the whole ladder in one process does not work:

```bash
# Build with line info (default in RelWithDebInfo here) so SASS maps to source.
# --only= isolates ONE variant; --skip-validation drops the correctness passes,
# whose kernel launches would otherwise be profiled too.
ncu --set full \
    --kernel-name-base function --kernel-name regex:gemm \
    --launch-skip 1 --launch-count 3 \
    --export reports/ncu_gemm_tiled_smem_$(date +%Y%m%d_%H%M) \
    ./build/bin/mcke_gemm_bench 1024 \
        --only=tiled_smem --skip-validation --warmup=1 --iters=3
```

Three things in that command are load-bearing, and an earlier version of this
section got all three wrong (it read
`ncu ... ./build/bin/mcke_gemm_bench 4096`, which was fiction — no bench parsed
a shape argument at all until Phase 3d):

- **`--only=<variant>`.** `--kernel-name regex:gemm` also matches cuBLAS's own
  kernels, which are named `turing_sgemm_*` / `volta_sgemm_*` — "sgemm" contains
  "gemm". With eight variants × 25 launches, `--launch-count 3` would profile
  three launches of whichever kernel happened to come first out of ~200, and the
  report would be labelled with whatever you assumed it was.
- **A smaller shape.** `--set full` replays each kernel a dozen-plus times to
  collect every counter. Replaying a 4096³ naive GEMM at ~3 s per launch is tens
  of minutes; 1024³ is 64× less work and the *ratios* this table cares about
  (sectors per request, bank conflicts, stall reasons) are shape-independent.
- **`--warmup=1 --iters=3`.** Fewer launches to skip and replay. The bench prints
  a loud `NON-COMPLIANT WITH RULE 3` banner in this mode, deliberately: these
  timings must never be copied into `RESULTS.md`, only the counters.

For the *timing* numbers that do go in `RESULTS.md`, run the bench without `ncu`
at the pinned 4096³ shape and full iteration counts.

### Metrics that actually change decisions

| Metric | Reads as | What to do about it |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | compute pipe utilisation | high + low DRAM → compute-bound, optimise arithmetic/ILP |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM utilisation | high → you are at the bandwidth roof; reduce traffic (fuse, tile, reuse) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy | compare against your *hand-computed* theoretical occupancy; a gap means tail effects or launch config |
| `launch__registers_per_thread` | register pressure | high → fewer resident warps; if it exceeds the budget you get spills |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / ...requests.sum` | **sectors per request** | 4 is ideal for f32 (128 B / 32 B); 32 means fully uncoalesced |
| `smsp__inst_executed_op_shared_ld.sum` + `l1tex__data_bank_conflicts_pipe_lsu_mem_shared*` | shared-memory bank conflicts | non-zero → pad the smem tile leading dimension (`+1` or `+4` floats) |
| `smsp__average_warps_issue_stalled_*` | top stall reason | `long_scoreboard` = waiting on global memory; `barrier` = `__syncthreads` imbalance; `mio_throttle` = too many shared/LSU instructions |
| `dram__bytes_read.sum` + `dram__bytes_write.sum` | **actual** traffic | divide by our *ideal* bytes → cache efficiency. >1 means we re-read data that should have been reused |

### The two comparisons that teach the most

1. **Hand-computed occupancy vs. `achieved_occupancy`.** Compute by hand from
   `-Xptxas -v` output: blocks/SM limited by (registers per SM / (regs per thread
   × threads per block)), by (smem per SM / smem per block), and by the hardware
   block/warp caps. Take the minimum. Then compare. A mismatch usually means tail
   effect (too few blocks for the SM count) or a launch-bound kernel.
2. **Ideal bytes vs. `dram__bytes`.** Our cost model says the GEMM must read
   `(MK + KN + MN)·4` bytes. A naive GEMM reads ~`2·M·N·K·4`. The ratio *is* the
   reuse factor that tiling buys, and watching it fall from ~2000× to ~1× across
   the four GEMM variants is the single most instructive number in this project.

## 5. Locking clocks (Explorer / any machine you control)

Timing variance on an unlocked GPU is dominated by clock behaviour, not by your
code.

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS | head -40      # see what's available
sudo nvidia-smi -pm 1                             # persistence mode
sudo nvidia-smi -lgc 1200,1200                    # lock SM clock (needs privileges)
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw \
           --format=csv -l 1                      # watch during a run
```

On Explorer you may not have `sudo`; in that case log the observed clocks
alongside the result instead. On the RTX 5060 laptop, expect throttling within
seconds — log `clocks.sm` at start and end of the run and treat any laptop number
as indicative only.

## 6. Reproducibility checklist for a RESULTS.md row

- [ ] GPU model + compute capability (`mcke_device_query`)
- [ ] Driver version + CUDA toolkit version (`nvidia-smi`, `nvcc --version`)
- [ ] Clocks locked? Observed `clocks.sm` at start/end
- [ ] Exact command line, including problem size
- [ ] Warmup count and timed-iteration count
- [ ] Median **and** min (and note if they differ by >10% — that's a story)
- [ ] Ideal FLOPs and ideal bytes used for derived metrics
- [ ] Stated denominator for any % of peak
- [ ] Correctness verified in the same run (never report a time for an unverified
      kernel)
