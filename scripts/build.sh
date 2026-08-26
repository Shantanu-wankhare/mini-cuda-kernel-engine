#!/usr/bin/env bash
# =============================================================================
#  scripts/build.sh — one command that does the right thing on any of our boxes.
#
#  WHY a shell script and not just documenting the cmake line: because the cmake
#  line differs per machine (CUDA on/off, arch, parallelism) and typing it from
#  memory at 11pm on a Colab session is how you end up benchmarking a Debug
#  build. Encoding the detection once removes a whole class of "why is this 8x
#  slower" incident.
#
#  WHY bash and not Python: it is a thin wrapper over cmake, must run before any
#  Python environment exists, and has no logic worth testing.
#
#  Usage:  ./scripts/build.sh [host|cuda] [extra cmake args...]
# =============================================================================
set -euo pipefail

MODE="${1:-auto}"
shift || true

if [[ "$MODE" == "auto" ]]; then
  if command -v nvcc >/dev/null 2>&1; then MODE=cuda; else MODE=host; fi
  echo "[build.sh] auto-detected mode: $MODE"
fi

# nvcc is memory-hungry; on a 2-core Colab VM, -j$(nproc) with several .cu files
# can OOM the container. Cap it.
if command -v nproc >/dev/null 2>&1; then NPROC=$(nproc); else NPROC=$(sysctl -n hw.ncpu); fi
JOBS=$(( NPROC > 8 ? 8 : NPROC ))

case "$MODE" in
  host)
    BUILD_DIR=build-host
    cmake -B "$BUILD_DIR" -DMCKE_ENABLE_CUDA=OFF \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo "$@"
    ;;
  cuda)
    BUILD_DIR=build
    # RelWithDebInfo, not Release: -lineinfo is required for Nsight Compute to
    # map SASS back to source, and optimisation is still fully on.
    cmake -B "$BUILD_DIR" -DMCKE_ENABLE_CUDA=ON \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DMCKE_CUDA_PTXAS_VERBOSE=ON "$@"
    ;;
  *)
    echo "usage: $0 [host|cuda] [cmake args...]" >&2; exit 2 ;;
esac

cmake --build "$BUILD_DIR" -j "$JOBS"

echo
echo "[build.sh] built into $BUILD_DIR/ with $JOBS jobs"
echo "[build.sh] next: $BUILD_DIR/bin/mcke_device_query"
if [[ "$MODE" == "cuda" ]]; then
  echo "[build.sh] register/smem usage is in the ptxas lines above — record it"
  echo "[build.sh]   for occupancy calculations (docs/PROFILING.md section 4)."
fi
