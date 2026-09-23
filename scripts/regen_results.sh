#!/usr/bin/env bash
# =============================================================================
#  scripts/regen_results.sh — run every benchmark this machine can run, and
#  capture the results as ONE self-contained dataset directory.
#
#  Phase 5 stage 5f. Half of the exit criterion "one command regenerates every
#  table in RESULTS.md"; the other half is tools/render_results.py.
#
#  WHAT IT DOES NOT DO, and this is the design (DECISIONS.md Q10): it does NOT
#  touch RESULTS.md. A run produces a NEW directory,
#      reports/<machine-tag>/<run-id>/
#  and nothing else. RESULTS.md's tables each name ("pin") the exact dataset
#  they render from, so a fresh run can be compared against the published
#  numbers (render_results.py --preview) before anyone decides to promote it.
#  Rewriting tables as a side effect of running a benchmark would silently
#  desync the dozens of places RESULTS.md's prose quotes those numbers.
#
#  WHY .sh: it sequences external programs and files. Anything that needs real
#  data structures -- the manifest, the validity rules -- is a small python3
#  block below, not hand-built JSON in bash.
#
#  HOW DATA IS CAPTURED -- no bench was modified for this (DECISIONS.md Q11):
#  every bench is run with its WORKING DIRECTORY set to the dataset dir. The
#  benches write bare filenames to CWD (phase3_gemm.csv, phase4_graph.csv, ...)
#  except alloc_bench, which writes reports/alloc_*.csv under its CWD -- so its
#  CSVs land in <dataset>/reports/ (owner's correction during 5f planning).
#  Each bench's stdout and stderr are saved as <bench>.stdout.log /
#  <bench>.stderr.log: they are both the source of the columns no CSV carries
#  and the "exact command line + output" record RESULTS.md rule 2 asks for.
#
#  A DATASET IS INVALID -- and tools/render_results.py will refuse to render
#  from it -- if any bench exits nonzero, prints "no CUDA device" (which exits
#  0!), prints a FAILing validation line, or prints a numerics-gate FAIL. A
#  failed correctness check must never be able to become a published table.
#
#  USAGE:
#    scripts/regen_results.sh                       # everything this machine can run
#    scripts/regen_results.sh --only=device_query,alloc_bench
#    scripts/regen_results.sh --dry-run             # print the plan, run nothing
#    MCKE_MACHINE_TAG=colab-t4 scripts/regen_results.sh
#    MCKE_PEAK_GB_S=636.3 MCKE_PEAK_TFLOPS=15.601 scripts/regen_results.sh --tag=explorer-v100
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
# shellcheck source=scripts/machine_tag.sh
source scripts/machine_tag.sh

usage() {
  cat <<'USAGE'
usage: scripts/regen_results.sh [--only=a,b,...] [--build-dir=DIR] [--tag=TAG] [--dry-run]
  benches: device_query stream_triad fma_peak smoke alloc_bench bias_act_bench
           reduce_bench softmax_bench gemm_bench graph_bench
  Creates reports/<tag>/<run-id>/ and never modifies RESULTS.md.
USAGE
}

# name | binary | args | needs_gpu | uses_bench_common_denominators
BENCHES=(
  "device_query|mcke_device_query||0|0"
  "stream_triad|mcke_stream_triad||1|0"
  "fma_peak|mcke_fma_peak||1|0"
  "smoke|mcke_smoke||1|0"
  "alloc_bench|mcke_alloc_bench||0|0"
  "bias_act_bench|mcke_bias_act_bench||1|1"
  "reduce_bench|mcke_reduce_bench||1|1"
  "softmax_bench|mcke_softmax_bench||1|1"
  "gemm_bench|mcke_gemm_bench|4096|1|1"
  "graph_bench|mcke_graph_bench|--streams=4|1|1"
)

ONLY=""; BUILD_DIR=""; DRY=0; TAG=""; TAG_FROM_FLAG=0
for a in "$@"; do
  case "$a" in
    --only=*)      ONLY="${a#--only=}" ;;
    --build-dir=*) BUILD_DIR="${a#--build-dir=}" ;;
    --tag=*)       TAG="${a#--tag=}"; TAG_FROM_FLAG=1 ;;
    --dry-run)     DRY=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "regen_results.sh: unknown argument '$a'" >&2; usage >&2; exit 2 ;;
  esac
done

# --- build directory -------------------------------------------------------
if [ -z "$BUILD_DIR" ]; then
  if   [ -x build/bin/mcke_gemm_bench ];       then BUILD_DIR=build
  elif [ -x build-host/bin/mcke_device_query ]; then BUILD_DIR=build-host
  else echo "FATAL: no build found (build/ or build-host/). Build first, or pass --build-dir=." >&2; exit 1
  fi
fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"   # absolute: benches run with CWD = dataset dir

# --- selection ---------------------------------------------------------------
known_names=()
for row in "${BENCHES[@]}"; do known_names+=("${row%%|*}"); done
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a wanted <<< "$ONLY"
  for w in "${wanted[@]}"; do
    found=0
    for k in "${known_names[@]}"; do [ "$w" = "$k" ] && found=1; done
    if [ "$found" -eq 0 ]; then
      echo "FATAL: --only names unknown bench '$w'. Known: ${known_names[*]}" >&2; exit 2
    fi
  done
fi
selected() {
  [ -z "$ONLY" ] && return 0
  local w
  for w in "${wanted[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

# --- identity ----------------------------------------------------------------
if [ "$TAG_FROM_FLAG" = 1 ]; then
  TAG_REASON="explicit --tag"
else
  TAG="$(mcke_machine_tag)"               # honours MCKE_MACHINE_TAG itself
  TAG_REASON="$(mcke_machine_tag_reason)"
fi
SHA="$(git rev-parse --short HEAD)"
DIRTY=""
[ -n "$(git status --porcelain --untracked-files=no)" ] && DIRTY="-dirty"
RUN_ID="$(date +%Y-%m-%d_%H%M)_${SHA}${DIRTY}"
DS="$REPO/reports/$TAG/$RUN_ID"

# --- denominators guard --------------------------------------------------------
# bench_common.hpp FALLS BACK TO THE T4's measured denominators when
# MCKE_PEAK_GB_S / MCKE_PEAK_TFLOPS are unset. On any other machine that makes
# every %peak in those benches' CSVs a percentage of the WRONG GPU -- silently,
# which is exactly what RESULTS.md rule 5 exists to prevent. So on a non-T4 tag
# the benches that use those denominators are refused unless both are set.
# stream_triad/fma_peak are exempt: they MEASURE the denominators; they don't use
# them (run those first on a new machine, then pass what they print).
case "$TAG" in *-t4|t4) DENOM_OK=1 ;; *) DENOM_OK=0 ;; esac
[ -n "${MCKE_PEAK_GB_S:-}" ] && [ -n "${MCKE_PEAK_TFLOPS:-}" ] && DENOM_OK=1

# --- plan ----------------------------------------------------------------------
plan_names=(); plan_bins=(); plan_args=(); plan_gpu=(); skipped=()
for row in "${BENCHES[@]}"; do
  IFS='|' read -r name bin args needs_gpu uses_denom <<< "$row"
  selected "$name" || continue
  if [ ! -x "$BUILD_DIR/bin/$bin" ]; then
    if [ -n "$ONLY" ]; then
      echo "FATAL: --only asked for '$name', but $BUILD_DIR/bin/$bin is not built." >&2; exit 1
    fi
    skipped+=("$name (not built in $BUILD_DIR)"); continue
  fi
  if [ "$uses_denom" = 1 ] && [ "$DENOM_OK" = 0 ]; then
    echo "FATAL: '$name' uses bench_common.hpp's roofline denominators, which default to the" >&2
    echo "       T4's (235.4 GB/s / 8.130 TFLOP/s). Tag '$TAG' is not a T4, so set both" >&2
    echo "       MCKE_PEAK_GB_S and MCKE_PEAK_TFLOPS to THIS machine's measured values" >&2
    echo "       (run --only=stream_triad,fma_peak first to measure them)." >&2
    exit 1
  fi
  plan_names+=("$name"); plan_bins+=("$BUILD_DIR/bin/$bin"); plan_args+=("$args"); plan_gpu+=("$needs_gpu")
done
if [ "${#plan_names[@]}" -eq 0 ]; then
  echo "FATAL: nothing to run (selection empty after skipping unbuilt benches)." >&2; exit 1
fi

echo "=== regen_results.sh ==="
echo "machine tag   $TAG   [$TAG_REASON]"
echo "dataset       ${DS#"$REPO"/}"
echo "build dir     $BUILD_DIR"
echo "git           $SHA${DIRTY}"
echo "denominators  MCKE_PEAK_GB_S=${MCKE_PEAK_GB_S:-<unset: T4 default>}  MCKE_PEAK_TFLOPS=${MCKE_PEAK_TFLOPS:-<unset: T4 default>}"
for i in "${!plan_names[@]}"; do
  echo "  run  ${plan_names[$i]}: (cd <dataset> && ${plan_bins[$i]#"$REPO"/} ${plan_args[$i]})"
done
for s in "${skipped[@]+"${skipped[@]}"}"; do echo "  skip $s"; done

if [ "$DRY" = 1 ]; then
  echo "(dry run: nothing created, nothing run)"
  exit 0
fi
if [ -e "$DS" ]; then
  echo "FATAL: $DS already exists -- refusing to overwrite a dataset." >&2; exit 1
fi
mkdir -p "$DS"

# --- capture -------------------------------------------------------------------
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gpu_snapshot() {   # $1 = output file
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw \
               --format=csv,noheader > "$1" 2>/dev/null || true
  fi
}
gpu_snapshot "$DS/.gpu_start.csv"
# FULL text, not `tail -1`: the last line of `nvcc --version` is a build string
# ("Build cuda_12.8.r12.8/compiler..."); the version RESULTS.md §0 quotes
# ("nvcc 12.8.93") is on the "release ..., V12.8.93" line above it.
{ command -v nvcc  >/dev/null 2>&1 && nvcc --version; } > "$DS/.nvcc.txt"  || true
{ command -v cmake >/dev/null 2>&1 && cmake --version | head -1; } > "$DS/.cmake.txt" || true

: > "$DS/.runs.tsv"
for i in "${!plan_names[@]}"; do
  name="${plan_names[$i]}"; bin="${plan_bins[$i]}"
  read -r -a argv <<< "${plan_args[$i]}"
  echo "--- $name"
  t0="$(date +%s)"
  set +e
  ( cd "$DS" && "$bin" "${argv[@]+"${argv[@]}"}" ) > "$DS/$name.stdout.log" 2> "$DS/$name.stderr.log"
  rc=$?
  set -e
  t1="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$bin ${plan_args[$i]}" "$rc" "$((t1 - t0))" "${plan_gpu[$i]}" >> "$DS/.runs.tsv"
  echo "    exit $rc, $((t1 - t0)) s"
done
gpu_snapshot "$DS/.gpu_end.csv"
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- manifest + validity (python: real data structures, one source of the rules) --
set +e
DS="$DS" REPO="$REPO" TAG="$TAG" TAG_REASON="$TAG_REASON" RUN_ID="$RUN_ID" SHA="$SHA" \
DIRTY="$DIRTY" START="$START" END="$END" BUILD_DIR="$BUILD_DIR" \
python3 - <<'PYEOF'
import json, os, re, socket, platform, sys

ds = os.environ["DS"]
def read(name):
    p = os.path.join(ds, name)
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""

def gpu_row(text):
    line = text.strip().splitlines()[0] if text.strip() else ""
    if not line:
        return None
    keys = ["name", "driver", "clocks_sm", "clocks_mem", "temperature", "power"]
    return dict(zip(keys, [f.strip() for f in line.split(",")]))

cache = {}
cache_path = os.path.join(os.environ["BUILD_DIR"], "CMakeCache.txt")
if os.path.exists(cache_path):
    for line in open(cache_path, encoding="utf-8", errors="replace"):
        m = re.match(r"^(MCKE_ENABLE_CUDA|MCKE_USE_NVTX|MCKE_CUDA_ARCH|MCKE_GOOGLE_BENCHMARK|"
                     r"CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER):[A-Z]+=(.*)$", line)
        if m:
            cache[m.group(1)] = m.group(2)

# The validity rules. Each is a way a bench can "succeed" by exit code while
# its numbers must not be published.
RULES = [
    (re.compile(r"^validation\s+.*\bFAIL\b", re.M),   "a validation line FAILED"),
    (re.compile(r"VALIDATION FAILURE"),               "the bench reported VALIDATION FAILURE(S)"),
    (re.compile(r"\*\*\* FAIL \*\*\*"),               "a numerics gate FAILED"),
    (re.compile(r"CORRECTNESS FAILURE"),              "the bench reported CORRECTNESS FAILURE(S)"),
]
runs, invalid = [], []
for line in read(".runs.tsv").splitlines():
    bench, cmd, rc, secs, needs_gpu = line.split("\t")
    out = read(f"{bench}.stdout.log")
    why = []
    if int(rc) != 0:
        why.append(f"exit code {rc}")
    if needs_gpu == "1" and "no CUDA device" in out:
        why.append("printed 'no CUDA device' (did no work, but exits 0)")
    for rx, msg in RULES:
        hits = rx.findall(out)
        if hits:
            why.append(f"{msg} ({len(hits)}x)")
    runs.append({"bench": bench, "argv": cmd.split(), "exit_code": int(rc), "seconds": int(secs),
                 "stdout": f"{bench}.stdout.log", "stderr": f"{bench}.stderr.log",
                 "invalid_reasons": why})
    invalid += [f"{bench}: {w}" for w in why]

# Every file the benches wrote (CSVs, alloc's reports/*.csv), for the record.
outputs = sorted(
    os.path.relpath(os.path.join(root, f), ds)
    for root, _, files in os.walk(ds) for f in files
    if not f.startswith(".") and not f.endswith((".stdout.log", ".stderr.log")) and f != "manifest.json")

manifest = {
    "schema": "mcke-dataset/1",
    "status": "INVALID" if invalid else "VALID",
    "invalid_reasons": invalid,
    "reconstructed": False,
    "machine_tag": os.environ["TAG"],
    "machine_tag_reason": os.environ["TAG_REASON"],
    "run_id": os.environ["RUN_ID"],
    "started": os.environ["START"], "finished": os.environ["END"],
    "host": {"hostname": socket.gethostname(), "uname": " ".join(platform.uname())},
    "gpu": {"start": gpu_row(read(".gpu_start.csv")), "end": gpu_row(read(".gpu_end.csv"))},
    "toolchain": {"nvcc": read(".nvcc.txt").strip() or None, "cmake": read(".cmake.txt").strip() or None,
                  "cxx": cache.get("CMAKE_CXX_COMPILER")},
    "git": {"sha": os.environ["SHA"], "dirty": bool(os.environ["DIRTY"])},
    "build": {"dir": os.environ["BUILD_DIR"], "cache": cache},
    "denominators": {"MCKE_PEAK_GB_S": os.environ.get("MCKE_PEAK_GB_S"),
                     "MCKE_PEAK_TFLOPS": os.environ.get("MCKE_PEAK_TFLOPS"),
                     "note": "null = unset: bench_common.hpp used its T4 defaults (235.4 / 8.130)"},
    "outputs": outputs,
    "runs": runs,
}
with open(os.path.join(ds, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
for name in (".runs.tsv", ".gpu_start.csv", ".gpu_end.csv", ".nvcc.txt", ".cmake.txt"):
    p = os.path.join(ds, name)
    if os.path.exists(p):
        os.remove(p)   # folded into manifest.json; the dataset stays one readable file + outputs
print(f"\nstatus        {manifest['status']}")
for r in invalid:
    print(f"  INVALID: {r}")
sys.exit(1 if invalid else 0)
PYEOF
status=$?
set -e

echo
echo "Nothing in RESULTS.md changed. To compare this run against a published table:"
echo "  python3 tools/render_results.py --preview id=<table-id> source=${DS#"$REPO"/}"
echo "To PROMOTE it, edit that table's source= pin in RESULTS.md, re-render, then run"
echo "  python3 tools/render_results.py --stale-prose"
exit "$status"
