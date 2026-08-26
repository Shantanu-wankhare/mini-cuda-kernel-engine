# LEARNING_LOG.md

**This file is mine, not the project's.** `PROJECT_LOG.md` records what the
project did; this records what *I* understood, half-understood, or had to
re-derive. It is allowed to be messy and to contain wrong first attempts —
those are the useful part.

Conventions:
- `❓ Open` — I don't understand this yet and want to dig in (possibly in a
  separate chat).
- `🔁 Re-derived` — I thought I knew it, had to work it out again from scratch.
- `✅ Solid` — I can explain this cold, without notes.
- `⚠️ Trap` — something that looks fine and silently isn't.

---

## Session 1 — 2026-08-24 (architecture/planning, macOS, no GPU)

### Concepts introduced this session

| Concept | Status | My one-line version | Notes to self |
|---|---|---|---|
| `.cu` vs `.cpp` for CUDA API calls | | | The claim: a `.cpp` can call `cudaMalloc` fine; only `__global__`/`__device__` code and `<<<>>>` need nvcc. Test this belief by trying to move a launch into a `.cpp` and reading the error. |
| `MCKE_WITH_CUDA` vs `__CUDACC__` | | | Build-wide vs per-translation-unit. Why does confusing them cause *link* errors specifically? |
| Stream-ordered allocation | | | Freeing on the host does not mean the GPU is done with the memory. Reuse across streams needs an event. |
| Buddy allocator XOR trick | | | `buddy_of(g) = ((g-1) ^ 1) + 1`. Derive it again on paper: why does converting to 1-based indexing make siblings differ in the low bit? |
| Event vs. stream-synchronize | | | Event = GPU-side wait, no host stall. Stream-sync = host blocks, pipeline drains. |
| `cudaStreamNonBlocking` | | | Without it, streams implicitly serialise against the legacy NULL stream. |
| Roofline: AI, ridge point | | | `AI = flops/bytes`; `ridge = peak_flops/peak_bw`. Below the ridge you are memory-bound and TFLOP/s is a meaningless metric. |
| Ideal vs. actual bytes | | | Cost model uses *compulsory* traffic. The gap to Nsight's `dram__bytes` is the cache-efficiency measurement. |
| Kahn vs DFS topological sort | | | Kahn gives waves = independent node groups = stream-assignment candidates. |
| SSA-style graph (derived edges) | | | One producer per tensor; edges inferred from def/use. |

*(Fill in the middle two columns yourself — the point is to write it in your own
words, not to read mine.)*

### 🔁 Things I want to re-derive by hand before Phase 1

1. **Buddy heap indexing.** On paper, for a 1 KiB arena with 256 B min blocks:
   draw the tree, number the nodes, and verify `offset_of`, `buddy_of`, and
   `level_for_size(300)` by hand. Then check against the `static_assert`s in
   `include/mcke/memory/buddy_math.hpp`.
2. **Why 50% waste at 4097 bytes.** The test printed exactly 50.0%. Convince
   myself this is the worst case and that it happens at `2^n + 1`.
3. **Peak bandwidth formula.** `memory_clock_khz * 1e3 * bus_width_bytes`.
   Why is there no extra ×2 for DDR here? (Because CUDA already reports the
   effective data rate.) Verify by computing a T4's number and comparing to the
   spec sheet — if I get 2× the spec, I've double-counted.
4. **Arithmetic intensity of vector add.** 1 FLOP per 12 bytes = 0.083. Where
   does the 12 come from? What would it be for `a[i] += b[i]`?

### ❓ Open questions (worth a separate deep-dive chat)

- **Why is `cudaFree` synchronising, mechanically?** What does the driver
  actually have to do — unmap pages, wait for TLB invalidation? And why is
  `cudaFreeAsync` able to avoid it?
- **What makes an async CUDA error "sticky"?** Why can't the context recover
  after an illegal address, when a failed `cudaMalloc` is perfectly recoverable?
- **FP32 cores per SM is not queryable.** Why not? It feels like the most basic
  fact about a GPU. (And why is the sm_75 figure 64 while sm_86/89 is 128 — what
  changed in the SM partition layout?)
- **Occupancy is not throughput.** I keep reading that 100% occupancy is not the
  goal. What is the actual mechanism by which a *lower*-occupancy kernel can be
  faster? (Suspect: more registers per thread → more ILP and fewer spills.)
- **Warp divergence vs. warp scheduling.** If a warp stalls on memory, another
  warp issues. So how does latency hiding relate to occupancy quantitatively —
  is there a formula (Little's law?) for "how many warps do I need to hide 400
  cycles of DRAM latency"?
- **Why 32 and not 64 threads per warp?** Has NVIDIA ever changed it, and what
  breaks if they do?

### ⚠️ Traps flagged for me this session (I have not personally hit these yet)

- Timing a kernel with `std::chrono` around the launch → measures launch
  overhead (~5 µs), not the kernel. Must use CUDA events.
- Default-flag streams silently serialising against the NULL stream → "why is
  there no overlap?"
- Double-counting DDR in the bandwidth formula → reporting half the real % of
  peak.
- Reporting TFLOP/s for a reduction (AI ≈ 0.25, hopelessly memory-bound) —
  apparently a visible red flag in an interview.
- `std::array` in a struct that crosses into device code — its `operator[]` is
  host-constexpr and needs `--expt-relaxed-constexpr`. Use raw C arrays.
- First-iteration timings can be 10× off (module load + clock ramp) → always
  warm up.

### End-of-phase Q&A — Phase 0 (Architecture)

**Q1. Why does this project keep CUDA types out of the graph and allocator
layers?**
Because roughly 60% of the interesting code — allocator bookkeeping, shape math,
topological sorting, liveness analysis, stream assignment — is pure host logic
whose correctness has nothing to do with a GPU. Confining CUDA types to a thin
`mcke::rt` boundary lets that half be compiled and unit-tested anywhere,
including a machine with no CUDA at all. It also enforces a real architectural
discipline: if a scheduling decision needs a CUDA type, that is a signal the
abstraction is leaking. Concretely, `DeviceInfo` is a plain-old-data snapshot of
the fields we use from `cudaDeviceProp`, so tile-selection logic can be tested
against a hand-written device description.

**Q2. Why does every `allocate`/`deallocate` call take a stream?**
Because host-side `free` says nothing about whether the GPU has finished with the
memory. A kernel using a block may still be queued when the host frees it; if the
pool hands those bytes to a different stream, two kernels race on the same memory
and produce silent, non-deterministic wrong answers. Putting the stream in the
signature makes the requirement impossible to forget. The reuse rules: same
stream is safe immediately (streams are in-order); a different stream needs proof
of completion, either an event the consumer waits on (no host stall) or polling
`cudaEventQuery` before reusing. `cudaMalloc`/`cudaFree` sidestep all of this by
synchronising — which is exactly why they are slow and why we are replacing them.

**Q3. Why a DAG instead of a flat list of ops?**
Three things are impossible with a flat list because a list encodes a total
order and thereby destroys independence information. (1) Concurrency: you cannot
know ops 3 and 4 may overlap. (2) Memory reuse: liveness analysis needs the
dependency structure to prove tensor *t* is dead so its bytes can be recycled —
worth 2-4× peak memory on deep chains. (3) Fusion: "is this op's only consumer an
activation?" is a graph query. The cost is a topological sort (microseconds) and
maintaining acyclicity. Additionally, edges are *derived* from tensor def/use
rather than declared, which makes "the declared dependency disagrees with the
actual data flow" — a race that appears only under load — unrepresentable.

**Q4. Why must every op report ideal FLOPs and ideal bytes?**
Because they are the roofline axes, and without them a timing number is
uninterpretable. `AI = flops/bytes` places the kernel relative to the ridge
point `peak_flops/peak_bandwidth`; that tells you whether the kernel is memory-
or compute-bound, and therefore whether the next optimisation should target
traffic or arithmetic. It converts "4.2 ms" into "63% of attainable at AI = 0.5,
memory-bound, 37% left on the table." The bytes figure must be *compulsory*
traffic (each input read once, each output written once), not measured traffic —
because the ratio between ideal bytes and Nsight's `dram__bytes__sum` is itself
the cache-efficiency measurement, and folding actual traffic into the model
destroys that signal.

**Q5. Why is `RelWithDebInfo` the default build type rather than `Release`?**
Nsight Compute maps SASS instructions back to source lines using debug line
tables (`-lineinfo`). Without them the profiler still gives you per-kernel
metrics but cannot attribute a stall reason to a line of code, which is most of
its value. `RelWithDebInfo` keeps full optimisation, so the timings are
representative — you pay only in binary size. `Release` would force a rebuild
every time you wanted to profile, and profiling a differently-built binary is
how people end up chasing phantom regressions.

---

## Template for future sessions

```
## Session N — YYYY-MM-DD (topic, environment)

### Concepts introduced
| Concept | Status | My one-line version | Notes to self |

### 🔁 Re-derived
### ❓ Open questions
### ⚠️ Traps I actually hit (with the symptom I saw)
### End-of-phase Q&A (at phase boundaries only)
```
