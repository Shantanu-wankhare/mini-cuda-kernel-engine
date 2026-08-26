# MCKE — Mini CUDA Kernel Engine

A GPU compute runtime built from first principles in **C++20 and CUDA**, with no
deep-learning framework dependencies. Custom device memory management, a
computation-graph scheduler with asynchronous multi-stream execution, and
hand-tuned kernels — each component measured against a stated baseline.

> **Status:** Phase 0 (architecture) complete. Host-side logic builds and tests
> on any machine; the CUDA path has not yet been compiled on a GPU. See
> [PROJECT_LOG.md](PROJECT_LOG.md) for the honest state of every component and
> [docs/ROADMAP.md](docs/ROADMAP.md) for the plan.

## Why this exists

Frameworks hide four things that determine GPU performance: how memory is
allocated, how work is scheduled, how kernels access memory, and how you know any
of it is fast. This project rebuilds each one, small enough to understand
completely and instrumented enough to prove.

The design goal for every component is not "it works" but **"here is the
measurement, the baseline it beats, and the reason it beats it."**

## Architecture

```
                      ┌───────────────────────────────────────────┐
   host, portable     │  Graph            Op / Node / TensorDesc  │
   C++20 — no CUDA    │  GraphExecutor    topo sort, liveness,    │
   types, testable    │                   stream assignment       │
   without a GPU      ├───────────────────────────────────────────┤
                      │  Tensor  =  Storage (owning) + view       │
                      ├───────────────────────────────────────────┤
                      │  DeviceAllocator                          │
                      │    ├─ BuddyAllocator    (split/merge)     │
                      │    ├─ FreeListAllocator (size classes)    │
                      │    └─ RawDeviceAllocator (baseline)       │
   ─ ─ ─ ─ ─ ─ ─ ─ ─  ├───────────────────────────────────────────┤ ← CUDA types
   the boundary       │  mcke::rt   Stream · Event · cuda_check    │   stop here
   ─ ─ ─ ─ ─ ─ ─ ─ ─  ├───────────────────────────────────────────┤
   device, nvcc       │  kernels/*.cu   GEMM · reduce · softmax ·  │
                      │                 fused bias+GELU           │
                      └───────────────────────────────────────────┘
```

Everything above the boundary is plain C++20 and compiles on a machine with no
CUDA at all. That is a deliberate architectural constraint, not a convenience:
if a scheduling decision needs a CUDA type, the abstraction is leaking.

### The five design decisions worth knowing about

| Decision | Alternative rejected | Reason |
|---|---|---|
| Slab-based pool allocator with **stream-ordered** reuse | `cudaMalloc` per tensor | `cudaMalloc`/`cudaFree` cost 10-100 µs and `cudaFree` synchronises — it would destroy the async overlap the whole runtime is built for |
| **DAG** with edges derived from tensor def/use | flat op list; user-declared edges | A list destroys the independence information needed for overlap, memory reuse, and fusion. Derived edges make "declared dependency ≠ actual data flow" unrepresentable |
| **Events** for cross-stream dependencies | `cudaStreamSynchronize` | Events are a GPU-side wait; stream-sync is a host barrier that drains the pipeline |
| **Kahn's** topological sort | DFS post-order | Kahn's waves *are* the parallelism structure; DFS discards it |
| Every op reports **ideal FLOPs and bytes** | time only | They are the roofline axes. Without them, "4.2 ms" is uninterpretable |

Each is argued in full in the header that implements it — see
[`memory/allocator.hpp`](include/mcke/memory/allocator.hpp),
[`graph/graph.hpp`](include/mcke/graph/graph.hpp),
[`runtime/stream.hpp`](include/mcke/runtime/stream.hpp),
[`graph/op.hpp`](include/mcke/graph/op.hpp).

## Quick start

```bash
# Host-only (no GPU needed) — builds and tests the allocator, shape, graph logic
cmake -B build-host -DMCKE_ENABLE_CUDA=OFF && cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

```bash
# GPU machine
cmake -B build -DMCKE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/bin/mcke_device_query    # run this first on any new machine
./build/bin/mcke_smoke
```

Per-machine setup (Colab, SLURM, RTX 50-series) is in
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

## Results

All measurements live in [RESULTS.md](RESULTS.md), with the environment, command
line, iteration counts, ideal FLOPs/bytes, and the stated denominator for every
"% of peak". Predictions are recorded *before* the runs, and kept when they turn
out wrong.

## Documentation map

| File | Audience | Contents |
|---|---|---|
| `README.md` | reader / reviewer | what and why, architecture, headline results |
| `docs/ROADMAP.md` | anyone | phase plan, which hardware suits which phase, exit criteria |
| `docs/PROFILING.md` | anyone profiling | tool selection, metric tables, Nsight command lines |
| `docs/ENVIRONMENTS.md` | anyone building | per-machine setup and gotchas |
| `RESULTS.md` | reader / reviewer | every measurement, reproducibly |
| `PROJECT_LOG.md` | project record | dated session log: built / learned / measured / next |
| `CLAUDE.md` | AI agents | conventions, build commands, session protocol |
| `LEARNING_LOG.md` | the owner | personal: confusions, re-derivations, open questions, interview Q&A |
| header comments | implementer | the *why* for each design decision, next to the code |

Design rationale lives in the headers rather than in a separate design doc,
because a rationale that sits next to the code it explains is the only kind that
stays true.

## Non-goals

- Not a framework. No autograd, no operator coverage, no Python bindings.
- Not a cuBLAS competitor. cuBLAS is the *reference ceiling*; the goal is to know
  the gap and be able to explain it.
- Not portable to AMD/Metal. The point is to learn the NVIDIA execution model in
  detail.
