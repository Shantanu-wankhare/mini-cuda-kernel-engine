# reports/colab-t4/2026-08-30_session5/

Raw artifacts from runs on **Google Colab, Tesla T4 (sm_75)**. One directory per
machine so that a run on one machine can never overwrite another machine's
evidence — the layout Phase 5 stage 5f's regeneration driver builds on.

## `phase3_gemm.csv` + `gemm_bench.stdout.log`

The CSV and the complete stdout of **one** run of the Phase 3d GEMM ladder. These
are the source of the nine **Colab T4** rows of `RESULTS.md` §3d — every value
there matches this CSV exactly.

| | |
|---|---|
| Session | PROJECT_LOG.md Session 5, **2026-08-30** ("the GEMM ladder, Colab trip 2") |
| Command | `./build/bin/mcke_gemm_bench 4096` |
| GPU | Tesla T4, sm_75, 40 SMs |
| Driver / runtime | driver 580.82.07 (reports as CUDA 13.0) / runtime 12.8; cuBLAS in `PEDANTIC` math mode (no TF32) |
| Denominators | `peak_gb_s = 235.4`, `peak_tflops = 8.130` (measured, `RESULTS.md` §0) |
| Timing | 5 warmup + 20 timed iterations; median and min |
| Clocks | **not locked** (no root on Colab) |

**Caveat that travels with these numbers:** the bench's own drift check measured
cuBLAS at 33.116 ms first and 37.069 ms last — **+11.9%**, past the project's 3%
threshold. Clocks moved during the run, so treat the T4 rows as indicative; the
Explorer V100 rows in §3d (+0.03% drift) are the authoritative ones.

**Why they live here and not at the repo root:** they sat untracked at the root
until 2026-09-22, one `git clean` away from being lost, as the only surviving raw
evidence behind any published `RESULTS.md` table. And `gemm_bench` writes
`phase3_gemm.csv` to its working directory, so the next run from the root would
have silently overwritten them. The stdout file was renamed from `gemm_run.log`
so the pair reads as one run; contents are byte-identical to the originals
(sha256 `58a4c53c…` for the CSV, `fd6e36f7…` for the log).

The CSV uses `Profiler::write_csv`'s frozen 13-column schema, which is why this
provenance lives in a README rather than as a header row in the file.

## Moved into a per-run directory (2026-09-22, Phase 5 stage 5f)

These files lived at `reports/colab-t4/` until stage 5f introduced one directory
per run (`reports/<machine>/<run-id>/`, DECISIONS.md Q10), so that a dataset is a
self-contained unit a RESULTS.md table can pin by path. Moved with `git mv`
(history preserved) and verified byte-identical by sha256. The stdout log was
renamed a second time, `phase3_gemm_stdout.log` → `gemm_bench.stdout.log`, to
match the `<bench>.stdout.log` convention `scripts/regen_results.sh` uses.

`manifest.json` here is **reconstructed by hand** (`"reconstructed": true`): this
run predates the regeneration driver, so nothing was captured at run time.
Fields that could not be recovered are `null`, not guessed; the git commit is
marked **inferred**.
