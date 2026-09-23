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

## 2026-09-15 — Phase 5 benchmark/profiling planning (main chat)

### Q7. What role should Google Benchmark actually play?

**Question:** Google Benchmark's timing loop uses host wall-clock, which is the
wrong instrument for async CUDA launches — so "adopt Google Benchmark" has to
mean something more specific. What role should it play relative to the existing
`Profiler::time_op` + CUDA-event harness that every `RESULTS.md` §1–§4 number
came from?

**Options offered:**
1. **Wrap: GB owns repetitions, Profiler owns timing (Recommended).** New
   `bench/gb_adapter.hpp` registers benchmarks with `->UseManualTime()` and feeds
   `state.SetIterationTime()` from our CUDA-event deltas. GB contributes what
   Profiler genuinely lacks: `--benchmark_repetitions=N` with run-to-run
   stddev/CV, and JSON output. Profiler's 13-column CSV stays the source of truth
   for `RESULTS.md`, so no existing number is invalidated. The delta between the
   two harnesses (one sync at the end vs per-iteration sync) gets measured and
   written up as a Phase 5 result.
2. **Replace: rewrite all 6 CUDA benches on GB.** One harness, no duplication,
   closest to the roadmap bullet's plain reading. Cost: every §1–§4 number
   becomes non-comparable and needs re-measuring on both T4 and V100;
   `%peak`/`bound`/ideal-bytes derivation must be reimplemented as
   `benchmark::Counter`s; and `RESULTS.md` rule 3 ("median AND min over ≥20
   iterations") is not what GB natively reports.
3. **Alongside: GB for new benches only, leave the 6 alone.** Lowest risk and
   least work. But the roadmap's stated motivation — "now we genuinely need its
   statistics" — goes unmet precisely where the statistics are needed, i.e. the
   wave-sweep-style run-to-run variance in §4.
4. **Decline GB; add repetitions + stddev to Profiler instead.** Honest if GB's
   real value-add is mostly duplicated by ~120 lines in `profiler.cpp`. Keeps the
   zero-external-dependency, builds-offline property that `CMakeLists.txt:310`
   currently brags about. Costs: no JSON, no standard tooling, and a deliberate
   roadmap deviation that has to be argued for in writing.

**Picked:** Option 1.

**Why recommended:** GB's statistics are computed over *repetitions, of
per-repetition means*. The within-run per-launch distribution that
`Profiler::time_op` reports a median and a min over is averaged away inside each
repetition before GB ever sees it — so GB is *strictly less informative* about
kernel timing than what we already have, and it structurally cannot produce the
`min` that rule 3 publishes. What GB does add is run-to-run variance, a
self-describing JSON artifact, and `tools/compare.py`'s Mann–Whitney U
regression test. That is a harness and a comparison tool, not a measurement tool,
and the design should say so rather than let the roadmap bullet imply otherwise.

### Q8. What does "one command regenerates every table" mean?

**Question:** The Phase 5 exit criterion is "one command regenerates every table
in `RESULTS.md`." It reads light but isn't: there are ~28 tables, only ~13 are
bench-generated, 5 come from unit tests, 3 are hand-authored predictions, 1 is
blocked on `ncu` perms, and §3d interleaves Colab T4 and Explorer V100 rows under
different denominators — so no single machine can regenerate it.

**Options offered:**
1. **Auto-stitch the generated tables, mark the rest (Recommended).**
   `scripts/regen_results.sh` runs everything runnable on the current machine into
   `reports/<machine>/*.json`; `tools/render_results.py` rewrites only tables
   inside `<!-- BEGIN GENERATED: §3a -->` fences, merging per-machine JSON so
   §3d's T4+V100 rows stitch from two runs. The ~15 prose/derived/test/blocked
   tables stay hand-written and get an explicit `<!-- AUTHORED -->` marker so it's
   visible which is which.
2. **Artifacts only: one command produces every number, not every table.**
   `regen_results.sh` emits every CSV/JSON/plot plus a `manifest.json` recording
   GPU, driver, clocks and exact argv. `RESULTS.md` stays entirely hand-edited.
   Much less machinery and zero risk of a script clobbering hand-written analysis
   — but the exit criterion is being reinterpreted, which needs saying out loud.
3. **Full auto: every table generated, including test- and prose-derived ones.**
   Most faithful to the criterion as written. Requires teaching
   `test_host_core`/`test_graph_host`/`test_stream_safety` to emit
   machine-readable output, and committing §3d's prediction/verdict columns as
   data. Largest surface, and the authored analysis in §2c/§5b/§5c is the part of
   `RESULTS.md` with the most value per line — putting a generator anywhere near
   it is the main risk.

**Picked:** Option 1.

**Why recommended:** the fence protocol makes the generated/authored split a
property of the file rather than tribal knowledge, and it is the only option that
handles §3d's two-machine merge without one machine's run deleting the other's
rows. Option 3's cost lands entirely on the prose that is hardest to reconstruct.

### Q9. Where should the `ncu` work run?

**Question:** `scripts/profile_ncu.sh` is a Phase 5 deliverable, but `RESULTS.md`
§5a is 8 empty rows blocked on `ERR_NVGPUCTRPERM` on Explorer, with the RC ticket
still open. Where should the ncu work actually run?

**Options offered:**
1. **RTX 5060 laptop — the owner has admin there (Recommended).**
   `ERR_NVGPUCTRPERM` is a driver setting (`NVreg_RestrictProfilingToAdminUsers`)
   changeable on a machine you own. And §5a's columns — sectors/request, bank
   conflicts, stall reasons, dram_bytes vs compulsory — are *ratios*, which
   `docs/PROFILING.md` §4 already argues are shape-insensitive and are likewise
   largely throttle-insensitive, so the laptop's thermal problem doesn't
   disqualify it for counters the way it does for timings. Cost: sm_120 bring-up
   (CUDA ≥ 12.8) is a fresh environment this project has never touched.
2. **Write the script, validate on Colab, keep §5a blocked.** Colab's ncu
   permissions are documented as unavailable too, so the script would likely ship
   syntactically checked but never actually executed end-to-end. Honest, cheap,
   and leaves §5a exactly as blocked as it is today.
3. **Explorer only — wait on the RC ticket.** Keeps the roadmap's "Env:
   [Explorer]" as written and keeps all authoritative numbers on one machine with
   locked clocks. But it makes a Phase 5 deliverable depend on a third party's
   response time with no known ETA.

**Picked:** Option 2 — the owner chose **not** to take the recommendation.

**What this means for the plan:** §5a stays blocked and stays visibly blocked.
`scripts/profile_ncu.sh` gets written to the metric set in `docs/PROFILING.md`
§4, is run on Colab, and the expected `ERR_NVGPUCTRPERM` failure is detected
explicitly and recorded rather than buried in driver output. Phase 5 does **not**
open the RTX 5060 as a new environment; that stays available for Phase 6's arch
A/B, where sm_120 is the point rather than a side effect.

## 2026-09-22 — Phase 5 stage 5f planning: regenerating RESULTS.md (main chat)

### Q10. How should a new benchmark run reach RESULTS.md?

**Question:** RESULTS.md's prose quotes specific table numbers in well over 50
places across §0–§5c, and only 1 of the 13 tables slated for generation has any
raw data today (the T4 half of §3d). If the renderer simply rewrote tables with
each new run's numbers, the surrounding prose would go silently stale. How
should a new benchmark run reach RESULTS.md?

**Options offered:**
1. **Pinned datasets + explicit promotion (Recommended).** Each fence names the
   exact dataset directory it renders from, visible in the file. A GPU run
   creates a NEW dataset dir and changes nothing in RESULTS.md. Rendering from
   pins is one deterministic command (the exit criterion). Promoting a run =
   edit the pin and re-render; a `--stale-prose` report lists every nearby prose
   number that no longer appears in the table. Side benefit: stage 5g's run
   doubles as a reproducibility check against the published numbers before
   anything is promoted.
2. **Live overwrite.** regen_results.sh runs the benches and immediately
   rewrites the fenced tables. Simplest, but every run silently desyncs the
   prose — §3a already shows the failure (prose cites 242.8 GB/s for vw4, the
   table shows 247.8, because the bench measures that config twice per run and
   the two picked different measurements).
3. **Separate generated file.** Generated tables go to their own file(s);
   RESULTS.md links to them and stays fully hand-written. Zero risk to prose,
   but revisits Q8 and reinterprets the exit criterion.

**Picked:** Option 1.

**Why recommended:** it is the only option under which "one command regenerates
every table" and "the prose stays true" can both hold. Promotion becomes a
deliberate, reviewable act instead of a side effect of running a benchmark.

### Q11. Where should the renderer get columns that aren't in any CSV?

**Question:** Many columns of the generated tables are printed to stdout only:
GEMM regs/smem/spill/occupancy, softmax's max|Σrow−1|, the graph numerics-gate
result, §0's hardware facts, the whole §2b fragmentation table. Where should the
renderer get them?

**Options offered:**
1. **Parse captured stdout (Recommended).** regen_results.sh runs each bench with
   CWD = the dataset dir and saves stdout next to its CSVs (which is also the
   provenance record rule 2 asks for); the renderer parses the fixed printf
   formats and fails loudly on any mismatch. Zero C++ changes; the GEMM parser is
   testable today against the committed stdout log.
2. **JSON sidecars from each bench.** Structured output from ~8 C++ programs.
   More robust long-term, but none of it can run on the Mac, so it adds untested
   code to exactly the 5g run meant to validate everything else.
3. **CSV columns only.** Generated tables carry only CSV fields; the rest moves
   to authored side tables. Simplest, but splits tables readers use together.

**Picked:** Option 1 — **with a correction from the owner:** not every bench
writes bare filenames to CWD. `alloc_bench.cpp` creates `reports/` and writes
`reports/alloc_*.csv` (line 1040), so with CWD = dataset dir its CSVs land in
`<dataset>/reports/`, and the loader must look there. (The option text had said
"every bench already writes bare filenames to CWD"; that was imprecise.)

**Why recommended:** zero untestable C++ in the data path, and the one real
stdout fixture that exists (the Session-5 GEMM log) makes the hardest parser
testable before any GPU is involved.

### Q12. How should the pre-existing errors found during exploration be handled?

**Question:** Exploration turned up pre-existing errors, all verified: §5b says
register blocking gave "the SAME ratio" (4.17×) on both chips, but T4's is 3.90×;
§3d prose says −0.10 pp where its table gives −0.14; the §2a star footnote says
≤25 ns but the code stars ≤50 ns; §0 says measured BW is "typically 80–90% of
spec" when this project measured 73.5% and 70.9%; a "§0 rule" reference is
dangling; Profiler::summary_table has a sticky-precision bug (why many published
ms cells have 1 decimal); graph_bench exits 0 when its numerics gate FAILS; and
fma_peak runs 3 warmup + 10 timed, below rule 3, while producing the denominator
for every compute %peak.

**Options offered:**
1. **Fix docs + 2 code bugs now; test fma_peak in 5g (Recommended).** Named
   corrections in RESULTS.md, keeping superseded text; two small code fixes
   (summary_table precision, graph_bench's exit code); fma_peak NOT changed
   blind — 5g runs it at both 3+10 and 5+20 first.
2. **Record all, fix in 5i.**
3. **Fix everything now, fma_peak included** — changes the denominator of every
   compute %peak, blind.

**Picked:** Option 1. (Owner also noted that the sticky-precision bug shows in
§3b too — `warp_shuffle_256t` median 0.523, min 0.5.)

**Why recommended:** the doc errors are cheap and certain; the two code bugs are
both silent-failure bugs of exactly the kind this project exists to catch;
fma_peak is the one change whose consequences can't be seen until it is measured.

### Q13. Should generated tables reproduce editorial emphasis?

**Question:** Some published tables contain editorial marks — bold cells (§2b's
100.0%/64.0%, §3b's 9 vs 1 barrier counts, §4's **PASS**) and an inline note
("(this IS the baseline)"). Descriptive labels ("starved (40 blocks)",
"cuBLAS (first)") would be generated either way.

**Options offered:**
1. **Data only in fences; emphasis in prose (Recommended).** A generator
   shouldn't decide what's notable — after a new run the notable cell may change.
   Bold dropped from generated regions; notes become a footnote just below the
   fence.
2. **Encode emphasis in each table's spec.** Tables look exactly as today, but
   emphasis can go silently wrong after a promotion.

**Picked:** Option 1 — **with an exception from the owner:** formatting that
depends only on a cell's *own value* is fine to generate. Specifically, in §4's
numerics-gate column, **FAIL must always render bold; PASS can be plain.**

**Why the exception is right:** it is not editorial — it is a rule about a value,
not a judgement about which cell matters, so it cannot go stale on promotion.
And a FAIL is exactly the cell no reader should be able to miss.
