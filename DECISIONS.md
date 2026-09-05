# DECISIONS.md — running AskUserQuestion log

Every `AskUserQuestion` asked in any MCKE chat gets appended here live, per
`CLAUDE.md` §2. Newest at the bottom. Purpose: a re-openable record for when
the owner wants to revisit a decision after understanding the topic better —
not a design doc, just the question, the options as offered, and the pick.

---

## 2026-08-29 — Phase 3 GEMM ladder planning (main chat)

### Q1. Vectorized loads vs. the ladder's one-cause-per-rung rule

**Question:** Vectorized (float4) global→shared loads are worth a large
fraction of the gap to cuBLAS, but they are a *separate* optimization from
register blocking and warp tiling. How should the ladder handle them without
giving a row two attributable causes?

**Options offered:**
1. **Scalar throughout + a 7th isolating row (Recommended).** Build
   naive→tiled_smem→regblock→warptile all with scalar global→shared loads, so
   every row has exactly one cause. Then append `kWarpTileVec4` to the enum
   (same appending discipline as `kWarpTileNoDbuf`) as a 7th row that changes
   only the load width. Costs one extra kernel instantiation; keeps the
   one-variable rule intact. Risk: regblock/warptile may land below their
   predicted ranges because loads are unvectorized — itself an honest,
   explainable result.
2. **float4 from tiled_regblock onward.** Numbers land closer to the
   predictions and closer to cuBLAS sooner, but the tiled_regblock row then
   carries two causes (register blocking + vectorized loads) and the
   attribution is compromised for that step.
3. **Scalar only, no vectorization row.** Six rows exactly as pinned in
   `RESULTS.md`. Vectorization becomes a named, quantified item in the
   required "explain the remaining gap to cuBLAS" writeup rather than a
   measured row. Smallest scope; leaves a known lever unmeasured.

**Picked:** Option 1 (scalar throughout + isolating 7th row).

---

### Q2. Naive kernel's thread→output mapping and memory coalescing

**Question:** The naive kernel's thread→output mapping decides whether
B-reads and C-writes coalesce, and the two choices differ by roughly 5-10x.
I'll use the coalesced mapping as the baseline (an uncoalesced baseline would
silently inflate every speedup above it). Should the uncoalesced version also
be measured?

**Options offered:**
1. **Yes — add `naive_uncoalesced` (Recommended).** ~3 extra lines (swap the
   roles of `threadIdx.x`/`y`). The single cleanest coalescing demonstration
   in the project; directly feeds the documented sectors-per-request metric
   (4 ideal vs. 32 fully-uncoalesced); justifies the choice of coalesced-naive
   as the honest baseline rather than leaving that choice asserted.
2. **No — six rows as pinned.** Keep `RESULTS.md` §3d to its six planned
   rows. Coalescing gets explained in prose against the tiled_smem step
   instead of measured directly.

**Picked:** Option 1 (add `naive_uncoalesced`).

---

### Q3. Where does `tests/test_kernels_numerics.cpp` fit now?

**Question:** `tests/test_kernels_numerics.cpp` (the ctest numerics gate with
the awkward-shape list) is still outstanding from the Phase 3 plan. Where
does it fit now?

**Options offered:**
1. **Fold the shapes into `gemm_bench`, defer the ctest (Recommended).**
   `gemm_bench`'s correctness section already runs before timing and would
   carry the awkward-shape matrix (non-tile-multiples like 129×65×257, M=1,
   N=1, K=1, sub-tile 5×7×3, plus a `beta≠0` read-modify-write case). Gets
   full coverage on Colab trip 2 with no new target. The standalone ctest —
   which needs a GPU and so can't gate the Mac build anyway — is revisited in
   Phase 3 cleanup.
2. **Build the ctest now, in Stage 5.** Add
   `tests/test_kernels_numerics.cpp` as a proper GPU-only ctest target
   covering all four kernel families at the awkward-shape list. More
   rigorous and reusable; adds a CMake target and roughly a stage's worth of
   extra work before the GEMM kernels are written.
3. **Drop it.** Per-bench validation is considered sufficient; remove the
   item from the plan rather than leaving it perpetually outstanding.

**Picked:** Option 1 (fold the shapes into `gemm_bench`, defer the ctest).

**Why recommended:** `gemm_bench` already validates before it times anything,
so the awkward-shape matrix costs no new target; and the standalone ctest needs
a GPU, so it could never have gated the Mac build — which is where the fast
feedback loop actually lives.

**Outcome (Stage 5/6):** the shape list landed in `bench/gemm_bench.cpp` as
`1×1×1`, `5×7×3`, `129×65×257`, `1×64×128`, `64×1×128`, `33×63×1`, `253³`
(exercises the `float4` scalar fallback) and `256³` as a control, plus a
`beta=2` read-modify-write case. Two things were added beyond the option as
described, because the option only covered the small shapes and the *published*
numbers come from 4096³: cuBLAS is validated at the non-square `129×65×257`
first and then used as a full-shape oracle, and 1024 random output elements are
recomputed in `double`. The standalone ctest remains deferred to Phase 3
cleanup.

---

**Date:** 2026-08-31

**Question:** Rewriting these 6 commits requires changing their SHAs and force-pushing to origin/main. How do you want to proceed?

**Options offered:**
1. Rewrite + force-push — Strip the Co-Authored-By trailer from all 6 commit messages via git filter-branch/rebase, then force-push main to origin. Rewrites SHAs from a8dcbb5 onward, including the PR #1 merge commit. Safe here since Shantanu is the only contributor, but any other clone of this repo would need to be re-fetched/reset.
2. Leave history as-is — Keep the existing commits (with the trailer) on GitHub untouched. Only new commits going forward will omit it, thanks to the settings change already made.

**Picked:** Option 1 (Rewrite + force-push).

**Why recommended:** Sole contributor on this repo, so force-push carries no collaborator risk, and it fully removes the attribution from GitHub as requested rather than leaving it partially done.

---

## 2026-08-31 — Explorer V100 session (fork chat)

### Q. How to proceed on ncu stall-reason data after Explorer confirmed `ERR_NVGPUCTRPERM`

**Question:** `ncu` on the `gpu-interactive` allocation failed with
`ERR_NVGPUCTRPERM` — a driver-level permission restriction, not fixable from
inside the job. How do you want to proceed?

**Options offered:**
1. **File an RC ticket now (Recommended).** Email `rchelp@northeastern.edu` (or
   the ServiceNow link from Explorer's own login banner) asking for GPU
   performance-counter permission on `gpu-interactive`, citing
   `ERR_NVGPUCTRPERM`. Routine request at HPC centers; Explorer is the
   authoritative environment regardless, so worth having working here.
2. **Try the RTX 5060 laptop instead.** Own hardware, likely admin rights to set
   `NVreg_RestrictProfilingToAdminUsers=0` directly — no ticket wait. Tradeoff:
   thermal throttling means laptop numbers are correctness/stall-reason data
   only, never headline timing, per `CLAUDE.md`'s existing rule for that
   machine.
3. **Hold on ncu, move to writeup.** Bank what Colab and Explorer already gave —
   correctness, occupancy hand-calc vs. the CUDA API on two architectures, the
   drift comparison, the `warptile_nodbuf` non-reproduction — and write the
   Phase-3 exit criterion's honest current state: the remaining gap to cuBLAS
   is measured but only partially diagnosed, pending profiling access. Revisit
   `ncu` later without blocking on it now.

**Picked:** Option 3 (hold on `ncu`, move to writeup).

**Revisited 2026-08-31, later the same day:** owner now has time and wants to
pursue `ncu` access rather than continue holding. No code or cluster state
changed in between — this supersedes the pick above by resuming toward
**Option 1** (file an RC ticket), which was the original recommendation.
Ticket text drafted; owner to submit via `rchelp@northeastern.edu`.

---

## 2026-09-04 — Phase 4 graph engine planning (main chat)

Context: design review (two parallel passes) found that liveness computed over a
topological order is **unsound** under a multi-stream schedule. All three
questions below follow from that finding.

### Q4. How should the Phase 4 memory planner allocate?

**Question:** Liveness over a topological order gives non-overlap in topological
index, but the executor runs nodes concurrently. Two tensors that don't overlap
in topo index CAN be alive simultaneously in wall-clock time if they sit on
different streams with no dependency between them — so a naive planner assigns
them the same buffer and races. (Verified by hand on a 5-node graph: races under
BOTH parallel policies, not under `kSequential`.) How should the memory planner
allocate?

**Options offered:**
1. **Plan-time arena, static offsets (Recommended).** Offsets computed at plan
   time from a happens-before relation (vector clocks) rather than topo order;
   one arena; intermediates are `Tensor::slice()` views; `run_async()` makes zero
   allocator calls. Removes the broken multi-stream `Storage::note_use()` from
   the hot path rather than patching it; makes race-freedom statically decidable;
   deterministic peak. Phase 2's allocator still serves the arena, workspaces and
   graph I/O, and `kAllocPerTensor` still exercises it per-tensor as the baseline.
2. **Runtime allocate/deallocate per tensor lifetime.** Leans on Phase 2's
   stream-ordered reuse policies. Exercises the allocators harder and handles
   dynamic shapes, but requires fixing `note_use()` first, makes peak
   pool-state-dependent, and makes race-freedom un-checkable statically.

**Picked:** Option 1 (plan-time arena).

**Why recommended:** `Tensor::slice()`'s own Phase-0 comment already says "this
is how the Phase-4 memory planner hands two graph nodes disjoint slices of one
buffer" — the header anticipated this design. The decisive extra argument is that
a static plan has no data-dependent control flow, which makes race-freedom
*decidable on a laptop*.

### Q5. How much verification rigour for the race trap?

**Question:** The plan can be checked without a GPU, since a static plan has no
data-dependent control flow. How much rigour?

**Options offered:**
1. **Host race checker + ship the deliberately-unsafe arm (Recommended).** A
   host-side happens-before race detector (vector clocks; the ThreadSanitizer
   algorithm applied to a GPU schedule), fuzzed over random DAGs × policies ×
   stream counts; plus keep a `kReuseTopoNaive` arm so the trap is demonstrated
   rather than merely avoided — the analogue of Phase 2's naive arm in
   `test_stream_safety.cu` and Phase 3's `naive_uncoalesced` row.
2. **Safe planner + the runtime numerics gate only.** Much less code, but the
   gate catches races only probabilistically, only under load, and only on
   hardware.

**Picked:** Option 1 — **with a framing correction from the owner:** it is
"checker **plus** gate, not checker **instead of** gate, because they catch
different bug classes."

**Why the correction matters (recorded because it changes the design):** the
checker proves the *plan* is race-free; the gate proves the *executor faithfully
implements the plan*, and catches what the checker cannot model — the shared
cuBLAS handle, uninitialised reads, driver behaviour. The option text implied a
substitution. It isn't one, and the plan now says so explicitly.

### Q6. Which benchmark graph set?

**Question:** `RESULTS.md` §4 pins only diamond and chain×16. The reviews argue
that is not enough to make the phase informative. Which set?

**Options offered:**
1. **Add the wave sweep + a fanout graph (Recommended).** Pinned graphs plus
   (a) a diamond wave-sweep at M=N=K ∈ {256,512,1024,2048,4096}; (b) a fanout 4×4
   graph, because every pinned graph has width ≤ 2 so the K=4 columns are
   decorative and the two parallel policies never meaningfully differ;
   (c) a bandwidth-saturated diamond reusing Phase 3a's `max_row_blocks`
   starvation lever. No new kernels or ops.
2. **Pinned graphs + the transformer block only.** Smallest GPU time; leaves the
   phase's central claim largely untested.
3. **Everything the reviews proposed** — also per-shape chain variants,
   fused-vs-unfused transformer variants, per-launch-overhead microbenchmark.

**Picked:** Option 1 — **with an amendment from the owner on how (c)'s
prediction is recorded.**

**The amendment, and it is a real catch:** record (c) as "~1.13×, with **below
1.0× explicitly possible** if the two streams interfere in L2 or degrade the
memory controller's access pattern" — **not as a floor.** The 1.13× comes from
235.4 / 208.7 (machine peak over one starved kernel's achieved bandwidth), which
*assumes concurrent kernels share DRAM cleanly and additively* — and that
assumption is exactly what the experiment tests. Baking it in as a lower bound
would be circular. Any of {≈1.13×, ≈1.0×, <1.0×} is a real result; the mechanism
is the deliverable, not the direction.
