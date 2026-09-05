// =============================================================================
//  bench/graph_bench.cpp
//
//  WHAT: Phase 4's measurement driver -- schedule policies, overlap, event
//        counts, peak memory, and the launch-bound-vs-compute-bound experiment.
//
//  WHY .cpp: no __global__, no <<<>>>. It drives GraphExecutor, which drives the
//  launchers. Same boundary every other bench respects.
//
//  ---------------------------------------------------------------------------
//  WHAT IS ALREADY KNOWN BEFORE THIS BINARY RUNS
//
//  Every scheduling INTEGER in RESULTS.md section 4 -- stream assignments, event
//  counts, wait dedup, peak memory -- is computed and asserted on a laptop by
//  tests/test_graph_host.cpp. This program does not discover them; it prints
//  them next to TIMES, which are the only thing that genuinely needs a GPU.
//  That split is deliberate: GPU minutes are the scarce resource, and a number
//  that can be wrong deterministically should be caught deterministically.
//
//  ---------------------------------------------------------------------------
//  THE PREDICTION THIS EXISTS TO TEST
//
//  Overlap buys nothing if each kernel already saturates the machine. Stated
//  that way it is almost a tautology, so the interesting question is WHICH
//  resource is saturated -- and the two diamond graphs below are built to
//  separate them:
//
//    D2 (wave sweep)  compute-starved: GEMMs so small they leave SMs idle.
//                     Speedup should fall monotonically from ~2x to ~1x as the
//                     grid grows past one wave. The CURVE is the deliverable.
//    D3 (bandwidth)   SM-starved but memory-saturated: two bias_act kernels
//                     capped to 40 blocks. Phase 3a measured that kernel at
//                     88.7% of the measured DRAM bandwidth at ~6% occupancy.
//
//  D3's prediction, recorded before the run: 235.4/208.7 = ~1.13x is a rough
//  CEILING ESTIMATE, NOT A FLOOR. That arithmetic assumes two concurrent
//  kernels share DRAM cleanly and additively -- which is exactly the assumption
//  under test, so treating it as a lower bound would be circular. BELOW 1.0x is
//  a live possibility: two interleaved streams can thrash L2 or degrade the
//  memory controller's access pattern relative to one kernel's clean sweep. Any
//  of {~1.13x, ~1.0x, <1.0x} is a real result; the MECHANISM is the deliverable,
//  not the direction.
//
//  D2 and D3 together are the actual lesson: "SMs are idle" is not "the machine
//  is idle". Neither row says that alone.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "mcke/core/device.hpp"
#include "mcke/graph/executor.hpp"
#include "mcke/memory/allocator.hpp"
#include "mcke/profiling/host_timer.hpp"
#include "mcke/profiling/profiler.hpp"
#include "mcke/runtime/cuda_check.hpp"

#include "bench_common.hpp"

using namespace mcke;

namespace {

constexpr int kWarmup = 5;
constexpr int kIters  = 20;

// -----------------------------------------------------------------------------
// Graph builders. Every one of these is also built by the host tests, so a
// failure here is a KERNEL or TIMING problem, never a graph-construction one.
// -----------------------------------------------------------------------------
struct Built { Graph g; TensorId input = 0; std::vector<TensorId> outputs; };

OpPtr gemm_op(std::int64_t /*n*/) {
  GemmParams p;
  p.variant = kernels::GemmVariant::kTiledRegBlock;
  p.tile    = kernels::GemmTile{};      // (128,128,8,8,8), the instantiated tile
  return std::make_unique<GemmOp>(p);
}
OpPtr bias_act_op(int max_row_blocks = 0) {
  BiasActParams p;
  p.act = BiasActParams::Act::kGeluTanh;
  p.max_row_blocks = max_row_blocks;
  return std::make_unique<BiasActOp>(p);
}

// D1/D2: diamond whose two middle nodes are GEMMs of a chosen size.
Built diamond_gemm(std::int64_t n) {
  Built b;
  auto a = b.g.add_input(Shape{n, n}, DType::kF32, "A"); a.status().throw_if_error();
  auto w = b.g.add_input(Shape{n, n}, DType::kF32, "W"); w.status().throw_if_error();
  b.input = *a;
  auto l = b.g.add_node(gemm_op(n), {*a, *w}, "B_gemm"); l.status().throw_if_error();
  auto r = b.g.add_node(gemm_op(n), {*a, *w}, "C_gemm"); r.status().throw_if_error();
  auto d = b.g.add_node(gemm_op(n), {(*l)[0], (*r)[0]}, "D_gemm"); d.status().throw_if_error();
  b.g.mark_output((*d)[0]).throw_if_error();
  b.outputs.push_back((*d)[0]);
  b.g.finalize().throw_if_error();
  return b;
}

// D3: diamond whose two middle nodes are DELIBERATELY STARVED bias_act kernels
// -- SM-idle but bandwidth-saturated. The starvation lever is Phase 3a's own.
Built diamond_starved(std::int64_t rows, std::int64_t cols, int blocks) {
  Built b;
  auto x = b.g.add_input(Shape{rows, cols}, DType::kF32, "X"); x.status().throw_if_error();
  auto bi = b.g.add_input(Shape{cols}, DType::kF32, "bias"); bi.status().throw_if_error();
  b.input = *x;
  auto l = b.g.add_node(bias_act_op(blocks), {*x, *bi}, "B_starved"); l.status().throw_if_error();
  auto r = b.g.add_node(bias_act_op(blocks), {*x, *bi}, "C_starved"); r.status().throw_if_error();
  auto d = b.g.add_node(bias_act_op(0), {(*l)[0], *bi}, "D_join"); d.status().throw_if_error();
  (void)r;
  b.g.mark_output((*d)[0]).throw_if_error();
  b.g.mark_output((*r)[0]).throw_if_error();
  b.outputs = {(*d)[0], (*r)[0]};
  b.g.finalize().throw_if_error();
  return b;
}

Built chain(int n, std::int64_t rows, std::int64_t cols) {
  Built b;
  auto x = b.g.add_input(Shape{rows, cols}, DType::kF32, "x"); x.status().throw_if_error();
  auto bi = b.g.add_input(Shape{cols}, DType::kF32, "bias"); bi.status().throw_if_error();
  b.input = *x;
  TensorId cur = *x;
  for (int i = 0; i < n; ++i) {
    auto r = b.g.add_node(bias_act_op(), {cur, *bi}, "n" + std::to_string(i));
    r.status().throw_if_error();
    cur = (*r)[0];
  }
  b.g.mark_output(cur).throw_if_error();
  b.outputs.push_back(cur);
  b.g.finalize().throw_if_error();
  return b;
}

// F1: the only pinned graph with width > 2, and therefore the only one on which
// the two parallel policies differ at all (chain_greedy 0 events vs
// level_parallel 12 records + 36 waits on 4x4).
Built fanout(int branches, int depth, std::int64_t rows, std::int64_t cols, int blocks) {
  Built b;
  auto x = b.g.add_input(Shape{rows, cols}, DType::kF32, "x"); x.status().throw_if_error();
  auto bi = b.g.add_input(Shape{cols}, DType::kF32, "bias"); bi.status().throw_if_error();
  b.input = *x;
  for (int k = 0; k < branches; ++k) {
    TensorId cur = *x;
    for (int d = 0; d < depth; ++d) {
      auto r = b.g.add_node(bias_act_op(blocks), {cur, *bi},
                            "b" + std::to_string(k) + "n" + std::to_string(d));
      r.status().throw_if_error();
      cur = (*r)[0];
    }
    b.g.mark_output(cur).throw_if_error();
    b.outputs.push_back(cur);
  }
  b.g.finalize().throw_if_error();
  return b;
}

// T1: the transformer block -- GEMM -> bias+GELU -> GEMM -> softmax. Width 1, so
// every policy should measure 1.00x. It is here for the per-node breakdown and
// the nsys timeline, not for a speedup number.
Built transformer_block(std::int64_t tokens, std::int64_t d_model, std::int64_t d_ff) {
  Built b;
  auto x  = b.g.add_input(Shape{tokens, d_model}, DType::kF32, "x");  x.status().throw_if_error();
  auto w1 = b.g.add_input(Shape{d_model, d_ff}, DType::kF32, "W1");   w1.status().throw_if_error();
  auto b1 = b.g.add_input(Shape{d_ff}, DType::kF32, "b1");            b1.status().throw_if_error();
  auto w2 = b.g.add_input(Shape{d_ff, d_model}, DType::kF32, "W2");   w2.status().throw_if_error();
  b.input = *x;
  auto g1 = b.g.add_node(gemm_op(d_ff), {*x, *w1}, "gemm1");          g1.status().throw_if_error();
  auto ba = b.g.add_node(bias_act_op(), {(*g1)[0], *b1}, "bias_gelu");ba.status().throw_if_error();
  auto g2 = b.g.add_node(gemm_op(d_model), {(*ba)[0], *w2}, "gemm2"); g2.status().throw_if_error();
  auto sm = b.g.add_node(std::make_unique<SoftmaxOp>(SoftmaxParams{}), {(*g2)[0]}, "softmax");
  sm.status().throw_if_error();
  b.g.mark_output((*sm)[0]).throw_if_error();
  b.outputs.push_back((*sm)[0]);
  b.g.finalize().throw_if_error();
  return b;
}

// -----------------------------------------------------------------------------
// Arguments. Unknown flags are FATAL, inherited from gemm_bench: a silently
// ignored --streams=4 would produce a wrong RESULTS row with no error anywhere,
// which is a direct rule-2 violation.
// -----------------------------------------------------------------------------
struct Args {
  int  streams = 4;
  int  warmup = kWarmup, iters = kIters;
  bool skip_gate = false;
  bool profile = false;
  std::string only;     // graph name filter
};

bool starts_with(const char* s, const char* p) {
  return std::strncmp(s, p, std::strlen(p)) == 0;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const char* s = argv[i];
    if      (starts_with(s, "--streams=")) a.streams = std::atoi(s + 10);
    else if (starts_with(s, "--warmup="))  a.warmup  = std::atoi(s + 9);
    else if (starts_with(s, "--iters="))   a.iters   = std::atoi(s + 8);
    else if (starts_with(s, "--only="))    a.only    = s + 7;
    else if (std::strcmp(s, "--skip-gate") == 0) a.skip_gate = true;
    else if (std::strcmp(s, "--profile") == 0)   a.profile = true;
    else if (starts_with(s, "--peak-gb-s=") || starts_with(s, "--peak-tflops=")) {}
    else {
      std::fprintf(stderr,
          "[mcke] FATAL: unrecognised argument '%s'.\n"
          "  usage: mcke_graph_bench [--streams=K] [--warmup=W] [--iters=I]\n"
          "                          [--only=<graph>] [--skip-gate] [--profile]\n"
          "                          [--peak-gb-s=X] [--peak-tflops=X]\n", s);
      std::exit(2);
    }
  }
  return a;
}

const char* policy_name(SchedulePolicy p) {
  switch (p) {
    case SchedulePolicy::kSequential:    return "sequential";
    case SchedulePolicy::kLevelParallel: return "level_parallel";
    case SchedulePolicy::kChainGreedy:   return "chain_greedy";
  }
  return "?";
}

struct Row {
  std::string graph, policy;
  std::size_t streams_used = 0, intra_events = 0, intra_waits = 0, forkjoin = 0;
  double median_ms = 0, min_ms = 0, speedup = 0, enqueue_us = 0, concurrency = 0;
  std::size_t peak = 0, naive = 0;
};

}  // namespace

int main(int argc, char** argv) {
  if (device_count() == 0) {
    std::printf("no CUDA device; nothing to benchmark\n");
    return 0;
  }
  auto dev = query_device(0);
  dev.status().throw_if_error();
  set_device(0).throw_if_error();

  const Args args = parse_args(argc, argv);
  const Roofline rl = benchcfg::make_roofline(argc, argv);
  benchcfg::print_denominators(rl, *dev);

  std::printf("command       ");
  for (int i = 0; i < argc; ++i) std::printf("%s%s", argv[i], i + 1 < argc ? " " : "\n");
  int drv = 0, rtv = 0;
  MCKE_CUDA_CHECK(cudaDriverGetVersion(&drv));
  MCKE_CUDA_CHECK(cudaRuntimeGetVersion(&rtv));
  std::printf("versions      driver %d.%d  runtime %d.%d\n",
              drv / 1000, (drv % 1000) / 10, rtv / 1000, (rtv % 1000) / 10);
  std::printf("clocks        NOT LOCKED unless stated -- see the drift check\n");
  std::printf("timing        %d warmup + %d timed iterations, median and min\n",
              args.warmup, args.iters);
  if (args.iters < 20 || args.warmup < 5)
    std::printf("\n*** NON-COMPLIANT WITH RESULTS.md RULE 3 (needs >=5 warmup, >=20 timed).\n"
                "*** Profiling only -- do NOT put this run in RESULTS.md.\n");
  std::printf("\nNOTE: every scheduling INTEGER below (streams, events, waits, peak bytes)\n"
              "      is also asserted on the host by tests/test_graph_host.cpp. This run\n"
              "      contributes the TIMES.\n\n");

  RawDeviceAllocator alloc;
  std::vector<Row> rows;

  // ---------------------------------------------------------------------------
  // One graph, all three policies.
  // ---------------------------------------------------------------------------
  auto run_graph = [&](const char* name, Built (*make)(), bool want_gate) {
    if (!args.only.empty() && args.only != name) return;
    std::printf("=== %s ===============================================\n", name);

    double seq_ms = 0.0;
    for (auto pol : {SchedulePolicy::kSequential, SchedulePolicy::kLevelParallel,
                     SchedulePolicy::kChainGreedy}) {
      Built b = make();
      const std::size_t in_bytes = b.g.tensor(b.input).shape.bytes(DType::kF32);

      ExecutorOptions opts;
      opts.schedule = pol;
      opts.memory   = MemoryPolicy::kReuseHappensBefore;
      opts.num_streams = (pol == SchedulePolicy::kSequential) ? 1 : args.streams;
      opts.profile = args.profile;

      // Feed every graph input before planning is not possible (binding happens
      // in plan()), so: plan, then fill.
      std::vector<TensorId> inputs;
      for (std::size_t t = 0; t < b.g.num_tensors(); ++t)
        if (b.g.tensor(static_cast<TensorId>(t)).is_graph_input)
          inputs.push_back(static_cast<TensorId>(t));

      GraphExecutor ex(std::move(b.g), alloc, *dev, opts);
      const Status planned = ex.plan();
      if (!planned.ok()) { std::printf("  plan FAILED: %s\n", planned.message().c_str()); return; }

      // Deterministic host data for every input.
      for (TensorId t : inputs) {
        auto bound = ex.output(t);   // not an output; use the plan's shape instead
        (void)bound;
      }
      {
        std::vector<float> host;
        for (TensorId t : inputs) {
          const std::size_t nb = ex.plan_ref().memory_plan().bytes_of[t];
          host.assign(nb / sizeof(float), 0.0f);
          for (std::size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<float>((i * 2654435761u) % 1000) * 0.001f - 0.5f;
          ex.set_input(t, host.data(), host.size() * sizeof(float)).throw_if_error();
        }
      }
      (void)in_bytes;

      const ExecutionPlan& pl = ex.plan_ref();

      for (int i = 0; i < args.warmup; ++i) ex.run_sync().throw_if_error();

      // Host enqueue cost, separately from device time: the launch-bound signal.
      std::vector<double> wall, enq;
      for (int i = 0; i < args.iters; ++i) {
        // Two readings from ONE start: the host enqueue cost alone, then the
        // full wall time. Their ratio is the launch-bound signal, and taking
        // both from the same start avoids attributing clock jitter between two
        // timers to the GPU.
        HostTimer t0;
        t0.start();
        ex.run_async().throw_if_error();
        const double e_us = static_cast<double>(t0.stop_ns()) / 1e3;
        ex.synchronize().throw_if_error();
        wall.push_back(static_cast<double>(t0.stop_ns()) / 1e6);
        enq.push_back(e_us);
      }
      std::sort(wall.begin(), wall.end());
      std::sort(enq.begin(), enq.end());

      Row r;
      r.graph = name;
      r.policy = policy_name(pol);
      r.streams_used = pl.num_streams();
      r.intra_events = pl.intra_events();
      r.intra_waits  = pl.intra_waits();
      r.forkjoin     = pl.forkjoin_events();
      r.median_ms    = wall[wall.size() / 2];
      r.min_ms       = wall.front();
      r.enqueue_us   = enq[enq.size() / 2];
      r.peak         = pl.peak_memory_bytes();
      r.naive        = pl.naive_memory_bytes();
      if (pol == SchedulePolicy::kSequential) seq_ms = r.median_ms;
      r.speedup = seq_ms > 0 ? seq_ms / r.median_ms : 0.0;

      if (args.profile) {
        ex.collect_timings().throw_if_error();
        double sum = 0.0;
        for (const auto& nt : ex.node_timings()) sum += nt.median_ms;
        // > 1.0 means real overlap: the per-node times add up to more than the
        // graph took. The cleanest overlap evidence available without nsys.
        r.concurrency = r.median_ms > 0 ? sum / r.median_ms : 0.0;
      }
      rows.push_back(r);

      std::printf("  %-15s streams %zu/%d  events %zu rec + %zu wait (+%zu fork/join)\n",
                  r.policy.c_str(), r.streams_used, opts.num_streams,
                  r.intra_events, r.intra_waits, r.forkjoin);
      std::printf("  %-15s median %8.3f ms  min %8.3f ms  speedup %5.2fx  "
                  "enqueue %7.1f us\n", "", r.median_ms, r.min_ms, r.speedup, r.enqueue_us);
      std::printf("  %-15s peak %zu B vs naive %zu B (%.2fx)%s\n", "", r.peak, r.naive,
                  r.peak ? double(r.naive) / double(r.peak) : 0.0,
                  args.profile ? "" : "");
      if (args.profile)
        std::printf("  %-15s concurrency factor %.2f (>1.0 = real overlap)\n", "",
                    r.concurrency);
      // The launch-bound signal, one number, no profiler needed.
      std::printf("  %-15s launch_bound_ratio %.3f (enqueue/device; >1 = launch-bound)\n",
                  "", r.median_ms > 0 ? (r.enqueue_us / 1000.0) / r.median_ms : 0.0);
    }

    // --- The numerics gate, on this graph. Correctness before any conclusion.
    if (want_gate && !args.skip_gate) {
      Built b = make();
      std::vector<TensorId> inputs;
      for (std::size_t t = 0; t < b.g.num_tensors(); ++t)
        if (b.g.tensor(static_cast<TensorId>(t)).is_graph_input)
          inputs.push_back(static_cast<TensorId>(t));
      ExecutorOptions o;
      o.validate_numerics = true;
      o.num_streams = args.streams;
      GraphExecutor ex(std::move(b.g), alloc, *dev, o);
      ex.plan().throw_if_error();
      std::vector<float> host;
      for (TensorId t : inputs) {
        const std::size_t nb = ex.plan_ref().memory_plan().bytes_of[t];
        host.assign(nb / sizeof(float), 0.0f);
        for (std::size_t i = 0; i < host.size(); ++i)
          host[i] = static_cast<float>((i * 2654435761u) % 1000) * 0.001f - 0.5f;
        ex.set_input(t, host.data(), host.size() * sizeof(float)).throw_if_error();
      }
      auto res = ex.validate_numerics(/*repeats=*/20);
      res.status().throw_if_error();
      std::printf("  numerics gate  %s  (%d configs x %d repeats, %zu elements)\n",
                  res->passed ? "PASS" : "*** FAIL ***", res->configs_compared,
                  res->repeats, res->elements_compared);
      if (!res->passed) std::printf("    %s\n", res->detail.c_str());
    }
    std::printf("\n");
  };

  run_graph("diamond_gemm_2048", [] { return diamond_gemm(2048); }, true);
  run_graph("chain16",           [] { return chain(16, 4096, 4096); }, true);
  run_graph("fanout4x4",         [] { return fanout(4, 4, 2048, 2048, 10); }, true);
  run_graph("diamond_starved",   [] { return diamond_starved(8192, 4096, 40); }, true);
  run_graph("transformer_block", [] { return transformer_block(4096, 1024, 4096); }, true);

  // --- D2: the wave sweep. The speedup-vs-waves CURVE is the deliverable; a
  //     single point at either end would only confirm what is already believed.
  if (args.only.empty() || args.only == "wave_sweep") {
    std::printf("=== wave sweep (diamond of GEMMs, chain_greedy vs sequential) ===\n");
    std::printf("  %6s %10s %12s %12s %8s\n", "N", "blocks", "seq ms", "greedy ms", "speedup");
    for (std::int64_t n : {256, 512, 1024, 2048, 4096}) {
      double t[2] = {0, 0};
      int idx = 0;
      for (auto pol : {SchedulePolicy::kSequential, SchedulePolicy::kChainGreedy}) {
        Built b = diamond_gemm(n);
        std::vector<TensorId> inputs;
        for (std::size_t q = 0; q < b.g.num_tensors(); ++q)
          if (b.g.tensor(static_cast<TensorId>(q)).is_graph_input)
            inputs.push_back(static_cast<TensorId>(q));
        ExecutorOptions o;
        o.schedule = pol;
        o.num_streams = (pol == SchedulePolicy::kSequential) ? 1 : args.streams;
        GraphExecutor ex(std::move(b.g), alloc, *dev, o);
        if (!ex.plan().ok()) { std::printf("  N=%lld plan failed\n", (long long)n); break; }
        std::vector<float> host;
        for (TensorId q : inputs) {
          const std::size_t nb = ex.plan_ref().memory_plan().bytes_of[q];
          host.assign(nb / sizeof(float), 0.25f);
          ex.set_input(q, host.data(), host.size() * sizeof(float)).throw_if_error();
        }
        for (int i = 0; i < args.warmup; ++i) ex.run_sync().throw_if_error();
        std::vector<double> ms;
        for (int i = 0; i < args.iters; ++i) {
          HostTimer h;
          h.start();
          ex.run_sync().throw_if_error();
          ms.push_back(static_cast<double>(h.stop_ns()) / 1e6);
        }
        std::sort(ms.begin(), ms.end());
        t[idx++] = ms[ms.size() / 2];
      }
      const std::int64_t blocks = ((n + 127) / 128) * ((n + 127) / 128);
      std::printf("  %6lld %10lld %12.3f %12.3f %8.2fx\n", (long long)n,
                  (long long)blocks, t[0], t[1], t[1] > 0 ? t[0] / t[1] : 0.0);
    }
    std::printf("  ^ tiled_regblock measured 2 blocks/SM on both T4 and V100, so a full\n"
                "    wave is 80 blocks (T4) / 160 (V100). Speedup should fall monotonically\n"
                "    and cross ~1.5x near one wave.\n\n");
  }

  // --- CSV. A NEW file, not a widened phase3 schema: Phase 3 rows depend on
  //     Profiler::write_csv's columns and must not shift.
  if (FILE* f = std::fopen("phase4_graph.csv", "w")) {
    std::fprintf(f, "graph,policy,streams_used,intra_events,intra_waits,forkjoin_events,"
                    "median_ms,min_ms,speedup,enqueue_us,concurrency,peak_bytes,naive_bytes\n");
    for (const Row& r : rows)
      std::fprintf(f, "%s,%s,%zu,%zu,%zu,%zu,%.6f,%.6f,%.4f,%.2f,%.4f,%zu,%zu\n",
                   r.graph.c_str(), r.policy.c_str(), r.streams_used, r.intra_events,
                   r.intra_waits, r.forkjoin, r.median_ms, r.min_ms, r.speedup,
                   r.enqueue_us, r.concurrency, r.peak, r.naive);
    std::fclose(f);
    std::printf("wrote phase4_graph.csv\n");
  }
  return 0;
}
