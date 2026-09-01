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
