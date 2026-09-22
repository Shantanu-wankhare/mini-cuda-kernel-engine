#!/usr/bin/env bash
# =============================================================================
#  scripts/typecheck_cuda.sh
#
#  Type-check the MCKE_WITH_CUDA=1 code path on a machine with NO CUDA, using
#  the fake headers in scripts/fakecuda/.
#
#  WHY: the MacBook cannot run nvcc, so every CUDA branch would otherwise reach
#  Colab completely unverified -- which is exactly how the Phase 0 -> Phase 1
#  transition burned a session. This catches missing includes, wrong signatures,
#  and broken launcher logic locally, for free.
#
#  It does NOT validate device codegen or kernel correctness. See the prelude's
#  banner for the precise boundary of what this proves.
#
#  Usage:  ./scripts/typecheck_cuda.sh                 # everything
#          ./scripts/typecheck_cuda.sh kernels/gemm.cu # one file
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
FAKE="scripts/fakecuda"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CXX="${CXX:-clang++}"

# ---------------------------------------------------------------------------
# Google Benchmark headers (Phase 5). bench/*_gb.cpp include <benchmark/...>,
# and this script's whole point is to compile bench/ WITHOUT a CMake build, so
# it has to find them itself.
#
# Optional, and degrades to SKIP rather than to FAIL -- deliberately mirroring
# MCKE_GOOGLE_BENCHMARK=AUTO in CMakeLists.txt. If Google Benchmark is absent
# then CMake does not build the *_gb targets either, so failing here would
# report a problem that cannot exist in any buildable configuration.
#
# Header-only for our purposes: we are running -fsyntax-only, so no library is
# needed, just the include path.
# ---------------------------------------------------------------------------
GB_INC=""
for cand in \
    "${MCKE_BENCHMARK_INCLUDE:-}" \
    build-gb/_deps/benchmark-src/include \
    build/_deps/benchmark-src/include \
    build-host/_deps/benchmark-src/include \
    /opt/homebrew/include \
    /usr/local/include; do
  if [ -n "$cand" ] && [ -f "$cand/benchmark/benchmark.h" ]; then
    GB_INC="$cand"
    break
  fi
done

FLAGS=(-std=c++20 -Wall -Wextra -fsyntax-only -I include -I tests -I bench -I kernels -I "$FAKE" -DMCKE_WITH_CUDA=1)
if [ -n "$GB_INC" ]; then
  # -isystem, not -I: Google Benchmark's headers are not ours to keep warning
  # -clean, and -Wall -Wextra -Wpedantic above would otherwise apply to them.
  # Same reasoning as the INTERFACE_SYSTEM_INCLUDE_DIRECTORIES fixup in
  # CMakeLists.txt, and for the same reason.
  FLAGS+=(-isystem "$GB_INC")
  echo "  (Google Benchmark headers: $GB_INC)"
else
  echo "  (Google Benchmark headers not found -- bench/*_gb.cpp will be SKIPPED."
  echo "   Set MCKE_BENCHMARK_INCLUDE=<dir> or configure with"
  echo "   -DMCKE_GOOGLE_BENCHMARK=ON to cover them.)"
fi
# Second pass with NVTX on. profiler.hpp's NvtxRange gains a member field under
# MCKE_USE_NVTX, and src/graph/executor.cpp wraps every node launch in one, so
# without this pass the entire Nsight-Systems path -- Phase 4's exit criterion --
# would only ever be compiled for the first time on Explorer.
NVTX_FLAGS=("${FLAGS[@]}" -DMCKE_USE_NVTX)

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  # Host .cpp files first (no language extensions needed), then every .cu.
  # Glob bench/*.cpp rather than naming files, so a new bench is covered the
  # moment it exists instead of when someone remembers to add it here.
  # src/*/*.cpp rather than naming the subdirectories: Phase 4 added src/graph/,
  # and the previous list (src/core + src/memory only) silently excluded it --
  # so the MCKE_WITH_CUDA=1 path of every graph file would have reached Colab
  # completely unverified, which is the exact failure this script exists to
  # prevent. Same reasoning for tests/test_*.cpp.
  targets=(src/*/*.cpp bench/*.cpp tests/test_*.cpp)
  while IFS= read -r f; do targets+=("$f"); done < <(find kernels bench tests tools -name '*.cu' | sort)
fi

fail=0
for f in "${targets[@]}"; do
  # A *_gb translation unit cannot compile without Google Benchmark's headers.
  # Skipping is correct rather than lenient: with GB absent, CMake does not
  # build these targets either (MCKE_GOOGLE_BENCHMARK=AUTO), so a failure here
  # would report a problem unreachable in any buildable configuration. Announced
  # per file, never silent -- a silent skip is how coverage rots, which is the
  # exact failure the src/*/*.cpp glob above was widened to fix.
  case "$f" in
    *_gb.cpp|*_gb.cu)
      if [ -z "$GB_INC" ]; then
        echo "  SKIP  $f  (no Google Benchmark headers)"
        continue
      fi
      ;;
  esac

  case "$f" in
    *.cu)
      # Strip the <<<grid,block,smem,stream>>> launch syntax, which no host
      # compiler can parse, then compile with the language prelude forced in.
      #
      # perl with a NON-GREEDY .*? rather than sed's [^>]*: a launch config
      # legitimately contains '>' characters, e.g.
      #     kernel<<<static_cast<unsigned>(rows), 256, 0, s>>>(...)
      # and a [^>]* class stops at the first one, leaving a mangled line that
      # reports as a syntax error in the kernel rather than a limitation here.
      out="$TMP/$(echo "$f" | tr '/' '_').cpp"
      perl -pe 's/<<<.*?>>>//g' "$f" > "$out"
      if "$CXX" "${FLAGS[@]}" -include "$FAKE/cuda_lang_prelude.h" "$out" 2>"$TMP/err"; then
        echo "  ok    $f"
      else
        # Constants referenced only inside a stripped <<<>>> now look unused.
        if grep -qv 'unused-variable\|unused-const-variable\|warning' "$TMP/err"; then
          echo "  FAIL  $f"; sed -n '1,20p' "$TMP/err"; fail=1
        else
          echo "  ok    $f  (unused-* warnings are sed artifacts)"
        fi
      fi
      ;;
    *)
      if "$CXX" "${FLAGS[@]}" "$f" 2>"$TMP/err"; then echo "  ok    $f"
      else echo "  FAIL  $f"; sed -n '1,20p' "$TMP/err"; fail=1; fi
      # NVTX pass, host .cpp only: the ranges live in executor.cpp and the
      # macro changes NvtxRange's layout, so this catches an ODR-shaped
      # mistake and a missing include before Explorer does.
      if ! "$CXX" "${NVTX_FLAGS[@]}" "$f" 2>"$TMP/errn"; then
        echo "  FAIL  $f  (with -DMCKE_USE_NVTX)"; sed -n '1,20p' "$TMP/errn"; fail=1
      fi
      ;;
  esac
done
[ $fail -eq 0 ] && echo "CUDA-path type-check: all clean" || echo "CUDA-path type-check: FAILURES"
exit $fail
