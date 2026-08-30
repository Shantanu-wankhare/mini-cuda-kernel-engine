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
