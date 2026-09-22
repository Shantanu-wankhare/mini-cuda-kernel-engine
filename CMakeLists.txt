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
option(MCKE_USE_NVTX           "Emit NVTX ranges for Nsight Systems timelines" OFF)
set(MCKE_CUDA_ARCH "native" CACHE STRING
    "CUDA architectures: 'native', or e.g. '75;80;89;120'")

# Google Benchmark (Phase 5). A TRI-STATE, not an option(), and deliberately so:
# option() is a bool, and a bool cannot express what three machines need.
#
#   AUTO (default) : use it if it is ALREADY available (system package, module,
#                    or FETCHCONTENT_SOURCE_DIR_BENCHMARK). Otherwise silently
#                    skip the *_gb targets. NEVER touches the network.
#   ON             : require it; fetch if necessary; failure is fatal.
#   OFF            : never, even if installed.
#
# WHY AUTO is the default, rather than just always calling FetchContent:
# FetchContent clones at CONFIGURE time. With no route to github.com,
# `cmake -B build` aborts -- and not just the bench targets, the WHOLE PROJECT.
# The MacBook offline hits that directly. Explorer is subtler: its compute nodes
# DO reach the internet, but only through an HTTP proxy that the default
# `explorer` module sets -- and `module purge`, a common first line of a job
# script, silently removes it (Explorer's docs, software/systemwide/modules.md).
# (Corrected 2026-09-22: an earlier version of this comment claimed Explorer's
# compute nodes have no outbound network at all. They do; the proxy is the
# catch.) The zero-dependency, works-on-a-laptop-with-no-network property
# this project has had since Phase 0 is worth more than the convenience of an
# unconditional fetch, so absence degrades to "skip" rather than to "fail".
set(MCKE_GOOGLE_BENCHMARK "AUTO" CACHE STRING
    "Google Benchmark: AUTO (use if present, else skip) | ON (require/fetch) | OFF")
set_property(CACHE MCKE_GOOGLE_BENCHMARK PROPERTY STRINGS AUTO ON OFF)

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
  src/graph/graph.cpp
  src/graph/happens_before.cpp
  src/graph/memory_plan.cpp
  src/graph/schedule.cpp
  src/graph/ops_gemm.cpp
  src/graph/ops_bias_act.cpp
  src/graph/ops_reduce.cpp
  src/graph/ops_softmax.cpp
  src/graph/executor.cpp
  src/tensor/tensor.cpp
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
# NVTX — named, coloured bands in an Nsight Systems timeline.
#
# Phase 4's exit criterion is an nsys timeline showing actual overlap (or
# explaining its absence), and without NVTX that timeline is a wall of
# `void mcke::kernels::gemm_regblock_kernel<...>` bars you cannot attribute to
# graph nodes. profiler.hpp has had the NvtxRange class since Phase 0; nothing
# ever defined the macro, so the whole path was dead code.
#
# PUBLIC IS MANDATORY, NOT STYLISTIC. NvtxRange gains a member field under this
# macro, so a translation unit that disagrees about it sees a differently-sized
# class — an ODR violation with no diagnostic.
#
# DO NOT LINK libnvToolsExt. NVTX v3 (`nvtx3/nvToolsExt.h`, CUDA >= 10) is
# HEADER-ONLY: it lazily dlopen()s the injection library that nsys itself sets
# via NVTX_INJECTION64_PATH. libnvToolsExt is the deprecated v2 library, still
# shipped in CUDA 12.x and removed later — linking it is how a build breaks on a
# toolkit upgrade.
if(MCKE_ENABLE_CUDA AND MCKE_USE_NVTX)
  target_compile_definitions(mcke_core PUBLIC MCKE_USE_NVTX)
  if(TARGET CUDA::nvtx3)
    # CUDA::nvtx3 is an INTERFACE (headers-only) target and first appears in
    # CMake 3.25. Our floor is 3.24, so the fallback below is required rather
    # than defensive.
    target_link_libraries(mcke_core PUBLIC CUDA::nvtx3)
  else()
    target_include_directories(mcke_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  endif()
  target_link_libraries(mcke_core PUBLIC ${CMAKE_DL_LIBS})   # dlopen on older glibc
  message(STATUS "MCKE: NVTX ranges ON (header-only nvtx3; no library linked)")
elseif(MCKE_USE_NVTX)
  message(WARNING "MCKE: MCKE_USE_NVTX ignored — CUDA is off")
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
#
# LINK_GROUP:RESCAN, not a plain list -- and this is not decoration.
# src/graph/ops_*.cpp (inside mcke_core) call straight into mcke_kernels'
# launchers: GemmOp::launch -> kernels::launch_gemm_f32, etc. That is a REAL
# link-time dependency of mcke_core ON mcke_kernels. But the only edge
# actually DECLARED to CMake runs the other way -- `mcke_kernels PUBLIC
# mcke_core`, above, which exists purely to propagate mcke_core's include
# dirs / MCKE_WITH_CUDA define to kernel translation units, not because any
# kernels/*.cu needs a mcke_core SYMBOL. CMake's link-line ordering has no
# visibility into actual object-file symbol needs -- only into declared
# target_link_libraries edges -- so it is free to place mcke_kernels before
# mcke_core in the final link command (which satisfies the edge it DOES
# know about). When it does, `ld` has already passed mcke_kernels.a by the
# time it discovers mcke_core.a's undefined references to launch_gemm_f32
# et al., and the link fails -- exactly the failure this project's first
# CUDA build of the graph layer hit, on two targets (mcke_test_graph_host
# and mcke_graph_bench) that construct real Op subclasses.
#
# LINK_GROUP:RESCAN (CMake >= 3.24, our stated floor) wraps both archives in
# the --start-group/--end-group idiom, so the linker rescans the pair until
# nothing is missing, regardless of which order CMake happens to emit them
# in. This fixes the root cause once, for every consumer of `mcke`, instead
# of patching each affected executable's link line individually.
add_library(mcke INTERFACE)
if(MCKE_ENABLE_CUDA)
  target_link_libraries(mcke INTERFACE $<LINK_GROUP:RESCAN,mcke_core,mcke_kernels>)
else()
  target_link_libraries(mcke INTERFACE mcke_core)
endif()

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
# Google Benchmark acquisition (Phase 5)
#
# Adopted as a HARNESS and a regression-comparison tool, NOT as a measurement
# tool -- see DECISIONS.md Q7. GB's iteration loop times with host wall-clock,
# which for an async CUDA launch measures launch latency, and its statistics are
# computed over REPETITIONS, of per-repetition MEANS. The per-launch median and
# min that RESULTS.md rule 3 publishes are averaged away inside each repetition
# before GB ever sees them. So Profiler::time_op keeps doing the measuring;
# bench/gb_adapter.hpp feeds GB our CUDA-event numbers via SetIterationTime().
#
# Deliberately NOT FetchContent's FIND_PACKAGE_ARGS, even though it exists at our
# 3.24 floor: FetchContent_MakeAvailable FALLS THROUGH to the clone when
# find_package misses, so on an offline machine it is a configure-time
# FATAL_ERROR for the whole project. The explicit find_package below cannot do
# that. FETCHCONTENT_SOURCE_DIR_BENCHMARK=<path> is the documented offline path.
# -----------------------------------------------------------------------------
set(MCKE_HAVE_GOOGLE_BENCHMARK OFF)

if(MCKE_BUILD_BENCH AND NOT MCKE_GOOGLE_BENCHMARK STREQUAL "OFF")
  # Every knob forced OFF *before* the subproject is added. All defaults below
  # verified against google/benchmark v1.9.5 (192ef100), not assumed:
  #   BENCHMARK_ENABLE_TESTING       ON  -> the OUTER gate (CMakeLists.txt:348).
  #                                         This is the one that actually matters:
  #                                         the GTest acquisition is NESTED inside
  #                                         it, so OFF here makes the rest
  #                                         unreachable. Forced anyway, below.
  #   BENCHMARK_ENABLE_GTEST_TESTS   ON  -> nested at :350; reaching it runs
  #                                         find_package(GTest CONFIG REQUIRED)
  #                                         at :356 -- a HARD configure failure
  #                                         on a box with no GTest.
  #   BENCHMARK_USE_BUNDLED_GTEST    ON  -> nested at :353; pulls the vendored
  #                                         googletest tree via include(GoogleTest)
  #   BENCHMARK_ENABLE_WERROR        ON  -> the genuinely dangerous one: GB
  #                                         builds ITS OWN sources with -Werror,
  #                                         so one new warning from a future
  #                                         AppleClang breaks OUR build inside a
  #                                         dependency we do not control.
  #   BENCHMARK_ENABLE_INSTALL/_DOCS/_TOOLS  ON -> we never install(); these
  #                                         would graft GB's install rules onto
  #                                         ours.
  # The rest already default OFF and are pinned so nobody flips them later.
  # LTO is pinned OFF on principle: link-time optimisation of a MEASUREMENT
  # library changes what is being measured. LIBPFM is pinned OFF because it
  # reports HOST CPU counters, which for a GPU benchmark is worse than nothing --
  # it is plausible-looking noise.
  #
  # CACHE ... FORCE rather than a plain set(): GB v1.9.5 declares
  # cmake_minimum_required(3.13...3.22), so CMP0077 is NEW and a normal variable
  # WOULD be honoured today. Pinning an older GB tag later would silently
  # re-enable GTest. CACHE...FORCE is correct under both policy behaviours.
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  foreach(_bmopt BENCHMARK_ENABLE_TESTING BENCHMARK_ENABLE_GTEST_TESTS
                 BENCHMARK_USE_BUNDLED_GTEST BENCHMARK_DOWNLOAD_DEPENDENCIES
                 BENCHMARK_ENABLE_INSTALL BENCHMARK_INSTALL_DOCS
                 BENCHMARK_INSTALL_TOOLS BENCHMARK_ENABLE_WERROR
                 BENCHMARK_FORCE_WERROR BENCHMARK_ENABLE_LTO
                 BENCHMARK_ENABLE_LIBPFM BENCHMARK_ENABLE_DOXYGEN
                 BENCHMARK_ENABLE_ASSEMBLY_TESTS)
    set(${_bmopt} OFF CACHE BOOL "" FORCE)
  endforeach()

  # Path 1 -- already available: `brew install google-benchmark`, or
  # -Dbenchmark_DIR=<dir>. Zero network, zero build time. (Explorer has NO
  # Google Benchmark module -- checked against its module list 2026-09-22 -- so
  # there it is path 2: FetchContent through the `explorer` module's proxy, or
  # FETCHCONTENT_SOURCE_DIR_BENCHMARK.)
  find_package(benchmark QUIET CONFIG)

  if(benchmark_FOUND)
    set(MCKE_HAVE_GOOGLE_BENCHMARK ON)
    message(STATUS "MCKE: Google Benchmark ${benchmark_VERSION} found (system package)")

  # NOTE the test is "non-empty", NOT "DEFINED". FetchContent_Declare itself
  # creates FETCHCONTENT_SOURCE_DIR_<name> as an EMPTY cache entry, so once any
  # configure in this build dir has reached the fetch path, DEFINED is true
  # forever after -- and AUTO would silently start fetching in a build dir that
  # had once been configured with =ON. A bare if(VAR) is false for an empty
  # string, which is exactly the question we mean to ask.
  elseif(MCKE_GOOGLE_BENCHMARK STREQUAL "ON" OR FETCHCONTENT_SOURCE_DIR_BENCHMARK)
    # Path 2 -- fetch. Only on explicit opt-in, or when the user has pointed
    # FETCHCONTENT_SOURCE_DIR_BENCHMARK at a local checkout (the offline recipe).
    include(FetchContent)
    FetchContent_Declare(benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      # A full 40-char SHA, not the tag name. A tag is mutable, and this
      # project's whole thesis is that a number is only meaningful with its
      # exact provenance -- "which benchmark library" is part of that. This is
      # v1.9.5. Cost of a raw SHA: GIT_SHALLOW is unreliable with one, so this
      # is a full ~25 MB clone, once, cached in the build tree thereafter.
      GIT_TAG        192ef10025eb2c4cdd392bc502f0c852196baa48
      GIT_PROGRESS   TRUE
      UPDATE_DISCONNECTED TRUE)      # do not re-fetch on every reconfigure
    FetchContent_MakeAvailable(benchmark)
    set(MCKE_HAVE_GOOGLE_BENCHMARK ON)

    # GB's CMakeLists.txt:144 does an UNGUARDED `set(CMAKE_CXX_STANDARD 17)` in
    # its own directory scope, and declares no target_compile_features anywhere.
    # Two consequences, both real:
    #   (good) benchmark::benchmark imposes NOTHING on us; our
    #          target_compile_features(mcke_core PUBLIC cxx_std_20) stands.
    #   (bad)  libbenchmark.a is compiled as C++17 while OUR TUs compile
    #          benchmark.h as C++20 -- every inline function and class template
    #          in that header is compiled twice under two different standards.
    #          That is an ODR violation with no diagnostic.
    # -DCMAKE_CXX_STANDARD=20 on the command line does NOT fix it; GB's
    # unguarded set() clobbers it. Overriding the target property afterwards
    # does, because CXX_STANDARD is merely INITIALISED from the variable at
    # target-creation time.
    set_property(TARGET benchmark PROPERTY CXX_STANDARD 20)

    # The SYSTEM keyword on FetchContent_Declare/add_subdirectory is CMake 3.25;
    # our floor is 3.24 (see the top of this file -- CUDA_ARCHITECTURES native
    # and LINK_GROUP:RESCAN set it). So this hand-rolled equivalent is REQUIRED,
    # not defensive: without it, benchmark.h is a normal include and the
    # -Wall -Wextra -Wpedantic that every bench target sets explicitly (because
    # mcke_core's are PRIVATE and not inherited) apply to GB's headers too.
    get_target_property(_bm_inc benchmark INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(benchmark PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_bm_inc}")
    # Distinguish the two ways we get here: a real clone vs a local checkout
    # supplied by FETCHCONTENT_SOURCE_DIR_BENCHMARK. Saying "fetched" for the
    # latter would be a small lie in exactly the place someone reads to answer
    # "where did this library come from?" -- which is provenance, and provenance
    # is the point.
    if(FETCHCONTENT_SOURCE_DIR_BENCHMARK)
      message(STATUS "MCKE: Google Benchmark from local checkout "
                     "${FETCHCONTENT_SOURCE_DIR_BENCHMARK} (offline path)")
    else()
      message(STATUS "MCKE: Google Benchmark fetched (v1.9.5, 192ef100)")
    endif()

  else()
    message(STATUS
      "MCKE: Google Benchmark not found -- *_gb bench targets skipped. "
      "This is the AUTO path and is NOT an error. To enable: "
      "-DMCKE_GOOGLE_BENCHMARK=ON (fetches), or "
      "-DFETCHCONTENT_SOURCE_DIR_BENCHMARK=<path> (offline), or "
      "install it (brew install google-benchmark).")
  endif()
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
  foreach(_b bias_act reduce softmax gemm graph)
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

# Google Benchmark bench targets (Phase 5, stage 5b).
#
# alloc_bench_gb is the ONLY Google Benchmark target that builds without a GPU,
# and that is its job as much as the allocator numbers are: stage 5b's failure
# modes are all silent (a missing `* 1e-3` reports 1000x; a missing
# ->UseManualTime() reports launch latency as kernel time), so the guards are
# tested HERE, on the MacBook, rather than discovered on Colab. See the
# bm_unit_conversion self-check in that file.
#
# Links mcke_core, NOT the `mcke` umbrella -- same reason as mcke_alloc_bench
# directly above: the umbrella pulls in mcke_kernels, which does not exist in a
# host-only build. Ungated on MCKE_ENABLE_CUDA for the same reason.
if(MCKE_BUILD_BENCH AND MCKE_HAVE_GOOGLE_BENCHMARK)
  add_executable(mcke_alloc_bench_gb bench/alloc_bench_gb.cpp)
  # benchmark::benchmark, NOT benchmark::benchmark_main: gb_adapter.hpp requires
  # our own main() so that argv can be snapshotted BEFORE benchmark::Initialize
  # strips the --benchmark_* flags out of it (RESULTS.md rule 2).
  target_link_libraries(mcke_alloc_bench_gb PRIVATE mcke_core benchmark::benchmark)
  target_include_directories(mcke_alloc_bench_gb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
  # As with every other bench target: mcke_core's warning flags are PRIVATE and
  # are not inherited, so opt in explicitly.
  target_compile_options(mcke_alloc_bench_gb PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )
endif()

# Pilot GPU _gb targets (Phase 5, stage 5c). Only reduce and bias_act -- NOT
# gemm or graph, whose bespoke tables (the wave sweep, fork/join event counts,
# enqueue_us) ARE the Phase 3d/4 deliverables and Google Benchmark's
# row-per-benchmark output cannot express them.
#
# Gated on BOTH MCKE_ENABLE_CUDA and MCKE_HAVE_GOOGLE_BENCHMARK, unlike
# alloc_bench_gb: these call the launchers declared in kernels.hpp, which do
# not exist in a host-only build. Link the `mcke` UMBRELLA (not mcke_core
# alone) for the same LINK_GROUP:RESCAN reason bias_act_bench/reduce_bench
# above do -- mcke_core calls into mcke_kernels.
if(MCKE_BUILD_BENCH AND MCKE_ENABLE_CUDA AND MCKE_HAVE_GOOGLE_BENCHMARK)
  foreach(_b bias_act reduce)
    add_executable(mcke_${_b}_bench_gb bench/${_b}_bench_gb.cpp)
    target_link_libraries(mcke_${_b}_bench_gb PRIVATE mcke benchmark::benchmark)
    target_include_directories(mcke_${_b}_bench_gb PRIVATE
      ${MCKE_TESTS_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/bench)
    target_compile_options(mcke_${_b}_bench_gb PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:MSVC>:/W4>)
  endforeach()
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

  # Phase 4: the graph layer. Host-only ON PURPOSE, and worth saying why rather
  # than treating it as a limitation.
  #
  # The graph engine's hardest correctness property -- that buffer reuse must be
  # keyed on happens-before rather than on topological order -- is a property of
  # the PLAN, which is pure data with no data-dependent control flow. So it is
  # decidable without a device, exhaustively, deterministically, and in
  # milliseconds. Everything that needs a GPU (actual overlap, event timing, the
  # nsys timeline) is measurement; everything that needs to be CORRECT is here.
  #
  # Links the `mcke` UMBRELLA, not mcke_core alone -- unlike every other
  # host-only test target, and for a subtler reason than test_stream_safety's
  # (which needs it because it contains .cu device code directly).
  #
  # tests/test_graph_host.cpp constructs real Op subclasses (GemmOp, BiasActOp,
  # ReduceOp, SoftmaxOp) to test infer_shapes()/cost() -- never launch(). But
  # under the Itanium ABI, a class's vtable is emitted in the translation unit
  # holding its first out-of-line virtual function, which for every one of
  # these is launch(), defined in src/graph/ops_*.cpp. So merely CONSTRUCTING a
  # GemmOp pulls in ops_gemm.cpp.o's vtable -- and with MCKE_WITH_CUDA=1,
  # ops_gemm.cpp's launch() body genuinely calls kernels::launch_gemm_f32,
  # which lives in mcke_kernels.
  #
  # This was invisible on macOS: MCKE_WITH_CUDA=0 makes every ops_*.cpp
  # launch() body a plain `return UnimplementedError(...)` that never
  # references a kernel launcher, so no undefined symbol existed to catch.
  # Colab -- the first MCKE_WITH_CUDA=1 build of this target -- is what
  # surfaced it: "undefined reference to launch_gemm_f32" et al. at link time.
  add_executable(mcke_test_graph_host tests/test_graph_host.cpp)
  target_link_libraries(mcke_test_graph_host PRIVATE mcke)
  target_include_directories(mcke_test_graph_host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME graph_host COMMAND mcke_test_graph_host)

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
