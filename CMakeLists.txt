# =============================================================================
#  MCKE — Mini CUDA Kernel Engine
#  Top-level build script.
#
#  WHY CMake (and not a Makefile):
#   - We build on four different machines (macOS/no-GPU, Colab, an RTX 5060
#     Windows/WSL box, and a SLURM HPC cluster). CMake is the only build system
#     with first-class, portable CUDA support: since 3.18 CUDA is a real
#     "language" (like CXX), so nvcc flags, architectures, and separable
#     compilation are handled by the generator instead of by us.
#   - `CMAKE_CUDA_ARCHITECTURES native` (3.24+) auto-detects the local GPU, so
#     the *same* command line works on a T4 (sm_75) and a 5060 (sm_120).
#
#  WHY 3.24 as the floor: that is the version that introduced `native` for
#  CUDA architectures. Colab and Explorer both ship newer CMake; on macOS
#  install with `brew install cmake`.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(mcke
        VERSION 0.1.0
        DESCRIPTION "Mini CUDA Kernel Engine: a from-scratch GPU compute runtime"
        LANGUAGES CXX)          # CUDA is enabled conditionally, see below.

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
# The single most important design decision in this build file:
#   MCKE_ENABLE_CUDA=OFF must produce a *working, testable* library.
# The MacBook Air has no nvcc and never will. Rather than let it be a
# pure text editor, we split the codebase into:
#   (a) host logic  — allocator bookkeeping, shape math, topological sort,
#                     liveness analysis, scheduling. Pure C++20, no CUDA.
#   (b) device logic — kernels and the thin CUDA driver/runtime calls.
# (a) is ~60% of the interesting code and 100% unit-testable on macOS.
option(MCKE_ENABLE_CUDA        "Compile CUDA kernels and the CUDA backend" ON)
option(MCKE_BUILD_TESTS        "Build unit tests"                          ON)
option(MCKE_BUILD_BENCH        "Build benchmark harnesses"                 ON)
option(MCKE_CUDA_PTXAS_VERBOSE "Pass -Xptxas -v (register/smem usage)"     OFF)
option(MCKE_CUDA_LINEINFO      "Pass -lineinfo (needed by Nsight Compute)" ON)
set(MCKE_CUDA_ARCH "native" CACHE STRING
    "CUDA architectures: 'native', or e.g. '75;80;89;120'")

# -----------------------------------------------------------------------------
# Language standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++20, not -std=gnu++20: portability.

# Put binaries somewhere predictable so the SLURM / Colab scripts can find them.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Export compile_commands.json so clangd on the Mac gives us real completion
# even for files we cannot compile locally.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # RelWithDebInfo, not Release: we *want* debug line tables. Nsight Compute
  # can only map SASS instructions back to source lines if they exist, and the
  # optimizer is still fully enabled, so timings are representative.
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------------------
# CUDA enablement
# -----------------------------------------------------------------------------
if(MCKE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)   # gives us CUDA::cudart, CUDA::cublas, ...
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES ${MCKE_CUDA_ARCH})
    # Separable compilation OFF by default: it forces device-link steps and can
    # cost performance (no cross-TU inlining of __device__ functions). We keep
    # every kernel + its device helpers in one .cu translation unit instead.
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "MCKE: CUDA enabled — toolkit ${CUDAToolkit_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "MCKE: no CUDA compiler found; falling back to host-only build")
    set(MCKE_ENABLE_CUDA OFF)
  endif()
endif()

# -----------------------------------------------------------------------------
# Core library: host-side runtime (portable C++20, no CUDA required)
# -----------------------------------------------------------------------------
add_library(mcke_core STATIC
  src/core/device.cpp
  src/core/profiler.cpp
  src/core/host_timer.cpp
  src/memory/allocator.cpp
  src/memory/buddy_allocator.cpp
  src/memory/freelist_allocator.cpp
  # Phase 3+ adds: src/graph/graph.cpp, src/graph/executor.cpp, src/graph/ops_*.cpp
)
target_include_directories(mcke_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mcke_core PUBLIC cxx_std_20)
target_compile_options(mcke_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(MCKE_ENABLE_CUDA)
  # MCKE_WITH_CUDA is the *compile-time* switch every header keys off.
  # PUBLIC because headers change shape based on it — anything linking us
  # must see the same value or we get ODR violations.
  target_compile_definitions(mcke_core PUBLIC MCKE_WITH_CUDA=1)
  target_link_libraries(mcke_core PUBLIC CUDA::cudart)
else()
  target_compile_definitions(mcke_core PUBLIC MCKE_WITH_CUDA=0)
endif()

# -----------------------------------------------------------------------------
# Kernel library: device code (CUDA only)
# -----------------------------------------------------------------------------
if(MCKE_ENABLE_CUDA)
  add_library(mcke_kernels STATIC
    kernels/elementwise.cu
    kernels/bias_act.cu
    kernels/reduce.cu
    kernels/softmax.cu
    kernels/gemm.cu
  )
  target_link_libraries(mcke_kernels PUBLIC mcke_core)
  # cuBLAS, for the kCublasRef ceiling variant in kernels/gemm.cu.
  #
  # PRIVATE, not PUBLIC, and the distinction matters. On a STATIC library PRIVATE
  # still propagates the LINK requirement transitively (CMake emits it under
  # $<LINK_ONLY:>), so every consumer still links cuBLAS correctly. What PRIVATE
  # does NOT propagate is cuBLAS's INTERFACE_INCLUDE_DIRECTORIES -- which with
  # PUBLIC would flow through the `mcke` umbrella onto mcke_smoke and all four
  # benches, making <cublas_v2.h> reachable from translation units that have no
  # business seeing it. mcke/runtime/cublas_check.hpp exists specifically to
  # confine that header to one TU; PUBLIC here would quietly undo it.
  target_link_libraries(mcke_kernels PRIVATE CUDA::cublas)
  target_compile_options(mcke_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>   # let __device__ code call constexpr host fns
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${MCKE_CUDA_LINEINFO}>>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${MCKE_CUDA_PTXAS_VERBOSE}>>:-Xptxas=-v>
  )
  # Keep register pressure visible in CI logs; occupancy work in Phase 3
  # depends on knowing regs/thread and smem/block per kernel.
endif()

# Convenience umbrella target: link this from apps/tests/benches.
add_library(mcke INTERFACE)
target_link_libraries(mcke INTERFACE mcke_core $<$<BOOL:${MCKE_ENABLE_CUDA}>:mcke_kernels>)

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
add_executable(mcke_device_query tools/device_query.cpp)
target_link_libraries(mcke_device_query PRIVATE mcke)

if(MCKE_ENABLE_CUDA)
  add_executable(mcke_smoke tools/smoke_vector_add.cpp)
  target_link_libraries(mcke_smoke PRIVATE mcke)
endif()

# -----------------------------------------------------------------------------
# Benchmarks
#
# stream_triad and fma_peak are Phase 1 deliverables: they establish the
# MEASURED bandwidth and MEASURED FMA-peak denominators that every later
# "% of peak" figure in RESULTS.md is computed against (see docs/PROFILING.md
# section 2 for why a spec-sheet number is the wrong denominator). They are
# standalone diagnostic .cu files, not part of the mcke_kernels library, so
# they are built directly here rather than through the kernels/ launcher
# pattern used by the runtime's actual ops.
# -----------------------------------------------------------------------------
if(MCKE_BUILD_BENCH AND MCKE_ENABLE_CUDA)
  add_executable(mcke_stream_triad bench/stream_triad.cu)
  target_link_libraries(mcke_stream_triad PRIVATE mcke)

  add_executable(mcke_fma_peak bench/fma_peak.cu)
  target_link_libraries(mcke_fma_peak PRIVATE mcke)
endif()

# Phase 3 kernel benches verify against the shared CPU reference in
# tests/reference.hpp before they time anything -- correctness first, always --
# so the bench targets need tests/ on their include path just as the test
# targets do.
set(MCKE_TESTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Phase 3 kernel benches. Each is a .cpp, NOT a .cu: they contain no __global__
# and no <<<>>>, only calls to the launchers declared in kernels.hpp -- which is
# exactly the boundary that header exists to define, and tools/smoke_vector_add.cpp
# is the precedent. Keeping nvcc off the benches also keeps the Colab edit-rebuild
# loop fast.
#
# Gated on MCKE_ENABLE_CUDA (unlike alloc_bench, which is deliberately ungated):
# in a host-only build kernels/*.cu is never compiled, so every launcher would be
# an undefined symbol at link time, and there is nothing useful for these to do
# without a GPU anyway.
if(MCKE_BUILD_BENCH AND MCKE_ENABLE_CUDA)
  foreach(_b bias_act reduce softmax gemm)
    add_executable(mcke_${_b}_bench bench/${_b}_bench.cpp)
    target_link_libraries(mcke_${_b}_bench PRIVATE mcke)
    # tests/ for reference.hpp (the CPU reference each bench validates against
    # before timing), bench/ for bench_common.hpp.
    target_include_directories(mcke_${_b}_bench PRIVATE
      ${MCKE_TESTS_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/bench)
    # mcke_core's warning flags are PRIVATE and are NOT inherited by consumers.
    # These files do pointer and size arithmetic where -Wextra earns its keep.
    target_compile_options(mcke_${_b}_bench PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:MSVC>:/W4>)
  endforeach()
endif()

# alloc_bench is deliberately NOT gated on MCKE_ENABLE_CUDA: it is a plain .cpp
# with no device code, and its entire point (Phase 2c) is that allocator LOGIC,
# host-side allocate/deallocate latency, and fragmentation bookkeeping are all
# measurable on the MacBook with no GPU at all. It links mcke_core rather than
# the `mcke` umbrella for the same reason the tests do -- the umbrella pulls in
# mcke_kernels, which only exists in a CUDA build.
if(MCKE_BUILD_BENCH)
  add_executable(mcke_alloc_bench bench/alloc_bench.cpp)
  target_link_libraries(mcke_alloc_bench PRIVATE mcke_core)
  # mcke_core's -Wall -Wextra -Wpedantic are PRIVATE to that target and are NOT
  # inherited by anything linking it (unlike a normal header-only interface).
  # This file does substantial pointer and size arithmetic, where -Wextra earns
  # its keep, so opt in explicitly rather than silently building unwarned.
  target_compile_options(mcke_alloc_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(MCKE_BUILD_TESTS)
  enable_testing()
  # Deliberately zero external test dependencies for now (no GTest). A 40-line
  # assertion header keeps `cmake -B build && ctest` working on a laptop with
  # no network. Google Benchmark arrives in Phase 5, where we actually need
  # its statistics, not before.
  add_executable(mcke_test_host_core tests/test_host_core.cpp)
  target_link_libraries(mcke_test_host_core PRIVATE mcke_core)
  target_include_directories(mcke_test_host_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME host_core COMMAND mcke_test_host_core)

  # Phase 2d: the stream-safety race. The repo's FIRST .cu test target.
  #
  # Needs CUDA for a reason that is not incidental: in a host-only build every
  # rt::StreamHandle is nullptr, so "two streams" are the same stream, rule 1 in
  # pending_reusable() legitimately fires, stream_query is unconditionally true
  # so nothing ever parks, and there is no concurrency -- there is simply no race
  # to construct. See the file's banner.
  #
  # Links the `mcke` umbrella rather than mcke_core (unlike every other test
  # target) because it contains device code of its own.
  if(MCKE_ENABLE_CUDA)
    add_executable(mcke_test_stream_safety tests/test_stream_safety.cu)
    target_link_libraries(mcke_test_stream_safety PRIVATE mcke)
    target_include_directories(mcke_test_stream_safety PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME stream_safety COMMAND mcke_test_stream_safety)
    # 77 == "no CUDA device at runtime". A CUDA-capable TOOLCHAIN does not imply
    # a GPU on the machine running ctest; without this, a build host that cannot
    # execute the test would report it as PASSED -- precisely the false pass this
    # test exists to prevent.
    #
    # The timeout is generous because the test builds a fresh allocator per trial
    # (160 trials x one slab reserve) so that every trial starts from an
    # identical state; a slow or shared GPU should not fail for scheduling
    # reasons.
    set_tests_properties(stream_safety PROPERTIES
      SKIP_RETURN_CODE 77
      TIMEOUT 600)
  endif()
endif()
