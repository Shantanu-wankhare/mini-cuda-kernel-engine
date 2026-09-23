#!/usr/bin/env python3
# =============================================================================
#  tools/render_results.py — regenerate RESULTS.md's generated tables from the
#  benchmark datasets they are PINNED to.
#
#  Phase 5 stage 5f: the other half of the exit criterion "one command
#  regenerates every table in RESULTS.md" (scripts/regen_results.sh is the
#  first half: it produces datasets).
#
#  THE MODEL (DECISIONS.md Q8, Q10) -- read this before editing RESULTS.md:
#
#   * A generated table lives inside a FENCE that names its data:
#
#       <!-- BEGIN GENERATED id=s3d-gemm source=reports/colab-t4/<run>,PENDING (...) -->
#       (blank line, the table, blank line)
#       <!-- END GENERATED id=s3d-gemm -->
#
#     `source=` lists one dataset directory per SLOT the table's spec declares
#     (e.g. s3d-gemm has a t4 slot and a v100 slot). `PENDING` = not captured yet.
#   * ANY PENDING slot -> the fence is left byte-for-byte untouched. The renderer
#     never deletes what it cannot regenerate.
#   * Only lines strictly between BEGIN and END are ever written. Headings,
#     prose, footnotes, and every AUTHORED table stay exactly as they are.
#   * A benchmark run NEVER changes RESULTS.md by itself. Promotion is: edit a
#     fence's source= to a new dataset, run --stale-prose to see which prose
#     numbers that would orphan, fix the prose, then render.
#   * Fences contain DATA ONLY (Q13): no editorial bold, no inline notes -- a
#     generator must not decide what is notable, because after a new run the
#     notable cell may be a different one. The one exception (the owner's): a
#     format that depends only on a cell's own value -- a FAILing numerics gate
#     always renders **bold**.
#
#  MODES
#    (default)            render and write every fence that is not PENDING
#    --check              exit 1 if any fence would change (nothing written)
#    --diff               print a unified diff of what would change
#    --audit              every markdown table must be fenced or AUTHORED-marked
#    --preview ID SRC..   render one table from any dataset(s) and compare it,
#                         cell by cell, to the published text -- the
#                         reproducibility check for a fresh run (stage 5g)
#    --stale-prose        prose numbers that the pinned renders would orphan
#    --self-test          no GPU, no network; includes golden test G1
#
#  WHY .py, stdlib only: analysis tooling, never runtime (CLAUDE.md section 5),
#  and it must run wherever a dataset is -- Colab, Explorer, this Mac.
# =============================================================================
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_outputs as bo  # noqa: E402  (sibling module; path set just above)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "RESULTS.md")


class FenceError(ValueError):
    pass


# -----------------------------------------------------------------------------
# Number formatting.
#
# ROUND_HALF_UP on the DECIMAL STRING, never float round(): the CSV cell
# "45.605" is published as 45.61, but round(45.605, 2) gives 45.6 in Python --
# both because round() is banker's rounding and because 45.605 has no exact
# binary representation. Decimal(str) rounds the digits that were written.
# -----------------------------------------------------------------------------
def dp(value, places: int) -> str:
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def commas(value) -> str:
    return f"{int(Decimal(str(value))):,}"


def trim_zero(value: str) -> str:
    """'100.0' -> '100', '50.0' -> '50', '62.5' -> '62.5' (occupancy percentages)."""
    return value[:-2] if value.endswith(".0") else value


# Display names. A NEW machine must be added here on purpose: guessing a
# human-readable name from a tag would put a made-up label in a published table.
MACHINE_LABEL = {"colab-t4": "Colab T4", "explorer-v100": "Explorer V100"}
MACHINE_SHORT = {"colab-t4": "Colab", "explorer-v100": "Explorer"}


def machine_label(ds: bo.Dataset, short: bool = False) -> str:
    table = MACHINE_SHORT if short else MACHINE_LABEL
    if ds.tag not in table:
        raise FenceError(f"{ds.path}: no display name for machine tag {ds.tag!r}; add it to "
                         f"MACHINE_LABEL/MACHINE_SHORT in tools/render_results.py on purpose")
    return table[ds.tag]


def row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


# -----------------------------------------------------------------------------
# Table specs. One per fence id. Each returns the table's lines (header,
# separator, rows) from the datasets in its slots.
# -----------------------------------------------------------------------------
@dataclass
class Spec:
    id: str
    slots: list[str]
    render: Callable[[dict[str, bo.Dataset]], list[str]]


SPECS: dict[str, Spec] = {}


def spec(id_: str, slots: list[str]):
    def deco(fn):
        SPECS[id_] = Spec(id_, slots, fn)
        return fn
    return deco


# §0 -------------------------------------------------------------------------
S0_HEADER = ["| Machine | GPU | CC | SMs | smem/SM | Peak BW (spec formula) | Measured BW | "
             "Measured FMA f32 peak | Driver / CUDA |", "|---|---|---|---|---|---|---|---|---|"]
# Machines this project measures WITHOUT a GPU dataset, by design: the MacBook
# has no GPU, and the RTX 5060 was deliberately not brought up in Phase 5
# (DECISIONS.md Q9). These are statements, not data, so they are constants.
S0_MACBOOK = row("MacBook Air (M-series)", "—", "—", "—", "—", "—", "—", "—", "host-only build")
S0_RTX5060 = row("RTX 5060 laptop", "_TBD_", "12.0", "", "", "", "", "", "needs CUDA ≥ 12.8")


def s0_gpu_row(ds: bo.Dataset) -> str:
    dq = bo.parse_device_query(ds.stdout("device_query"))
    triad = bo.parse_achieved_gbs(ds.stdout("stream_triad"))
    fma = bo.parse_fma_peak(ds.stdout("fma_peak"))
    gpu = (ds.manifest.get("gpu") or {}).get("start") or {}
    driver = gpu.get("driver")
    nvcc = bo.nvcc_release(ds.manifest)
    if not driver or not nvcc:
        raise FenceError(f"{ds.path}: manifest lacks driver ({driver!r}) or nvcc release ({nvcc!r})")
    return row(machine_label(ds, short=True), dq["name"], dq["cc"], dq["sm_count"], dq["smem_per_sm"],
               f"{dq['peak_bw_spec']} GB/s", f"{triad} GB/s", f"{fma} TFLOP/s",
               f"driver {driver} / nvcc {nvcc}")


@spec("s0-hardware", ["t4", "v100"])
def _s0(d):
    return S0_HEADER + [S0_MACBOOK, s0_gpu_row(d["t4"]), S0_RTX5060, s0_gpu_row(d["v100"])]


# §1 -------------------------------------------------------------------------
# n and ideal bytes are compile-time constants of the two programs and are not
# printed: stream_triad.cu:78 (n = 64 Mi) / :104 (3 arrays x 4 B), and
# smoke_vector_add.cpp:36 (default n = 64 Mi; regen_results.sh passes no argv).
S1_ROWS = [("stream_triad", "stream_triad", "grid_stride_256t", "64Mi", "768 MiB"),
           ("smoke", "vector_add", "grid_stride_256t", "64Mi", "768 MiB")]


@spec("s1-elementwise", ["t4"])
def _s1(d):
    ds = d["t4"]
    out = ["| Kernel | Variant | n | Ideal bytes | median ms | min ms | GB/s | % measured BW | Machine |",
           "|---|---|---|---|---|---|---|---|---|"]
    baseline = Decimal(bo.parse_achieved_gbs(ds.stdout("stream_triad")))
    for bench, kernel, variant, n, nbytes in S1_ROWS:
        text = ds.stdout(bench)
        r = bo.parse_summary_row(text, kernel, variant)
        gbs = bo.parse_achieved_gbs(text)
        pct = dp(Decimal(gbs) / baseline * 100, 1) + "%"
        out.append(row(kernel, variant, n, nbytes, dp(r["median_ms"], 3), dp(r["min_ms"], 3),
                       gbs, pct, machine_label(ds)))
    return out


# §2a ------------------------------------------------------------------------
def s2a(trace: str):
    def render(d):
        ds = d["t4"]
        rows = bo.read_csv(ds.file(f"reports/alloc_latency_{trace}.csv"))
        out = ["| Allocator | Op | median ns | p90 | p99 | p999 | max ns | amortised ns | alloc_calls | raw_mallocs |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            floor = int(r["clock_floor_ns"])
            def star(v: str) -> str:
                # ClockCalibration::is_below_floor (host_timer.hpp:134): <= 2x the floor.
                return f"{v}*" if floor > 0 and int(v) <= 2 * floor else v
            lat = [star(r[k]) for k in ("median_ns", "p90_ns", "p99_ns", "p999_ns", "max_ns")]
            if r["op"] == "allocate":
                tail = [dp(r["amortised_ns"], 1), r["alloc_calls"], r["raw_malloc_calls"]]
            else:
                tail = ["–", "–", "–"]   # per-op latency only; the counters are per allocate
            out.append(row(r["allocator"], r["op"], *lat, *tail))
        return out
    return render


for _t, _id in (("uniform_pow2", "s2a-lat-uniform"), ("dl_transformer", "s2a-lat-dl"),
                ("dl_transformer_bypass", "s2a-lat-bypass")):
    SPECS[_id] = Spec(_id, ["t4"], s2a(_t))


# §2b ------------------------------------------------------------------------
def s2b(trace: str):
    def render(d):
        ds = d["any"]
        tables = bo.parse_alloc_fragmentation(ds.stdout("alloc_bench"))
        if trace not in tables:
            raise FenceError(f"{ds.path}: alloc_bench printed no fragmentation table for {trace!r}")
        by_family: dict[str, list[list[str]]] = {}
        for cells in tables[trace]:
            by_family.setdefault(cells[0].split("/")[0], []).append(cells[1:])
        out = ["| Allocator | peak_reserved | peak_blocks | peak_requested | block_eff | reserv_eff | "
               "utilisation | internal waste | largest_free @ end | OOM? |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for fam in ("raw", "buddy", "freelist"):
            variants = by_family.get(fam)
            if not variants:
                raise FenceError(f"{ds.path}: no {fam} rows in the {trace} fragmentation table")
            # One row per allocator is the published claim: reuse POLICY does not
            # affect fragmentation. Asserted, not assumed -- if a policy ever did
            # change these numbers, collapsing the rows would hide exactly that.
            if any(v != variants[0] for v in variants):
                raise FenceError(f"{ds.path}: {trace}: {fam} policies DISAGREE on fragmentation "
                                 f"({variants}) -- the one-row-per-allocator table would hide it")
            cells = list(variants[0])
            if fam == "raw":
                cells[7] = "n/a"   # raw has no pool: "0 B largest free" would read as a full pool
            out.append(row(fam, *cells))
        return out
    return render


for _t, _id in (("uniform_pow2", "s2b-frag-uniform"), ("dl_transformer", "s2b-frag-dl"),
                ("dl_transformer_bypass", "s2b-frag-bypass")):
    SPECS[_id] = Spec(_id, ["any"], s2b(_t))


# §3a ------------------------------------------------------------------------
# (kernel, variant, occurrence, label, activation, width). The row marked NEW is
# deliberate (stage 5f plan): the published width comparison put sweep (b)'s
# vw1/vw2 next to sweep (a)'s vw4, measurements from two different sweeps. The
# repeat is sweep (b)'s own vw4 -- the 242.8 GB/s §3a's prose already quotes.
S3A_ROWS = [
    ("bias_relu", "fused_vw4", 0, "bias_relu fused", "relu", "vw4"),
    ("bias_relu", "unfused_pair_vw4", 0, "bias_relu unfused", "relu", "vw4"),
    ("bias_gelu_tanh", "fused_vw4", 0, "bias_gelu_tanh fused", "gelu_tanh", "vw4"),
    ("bias_gelu_tanh", "unfused_pair_vw4", 0, "bias_gelu_tanh unfused", "gelu_tanh", "vw4"),
    ("bias_gelu_erf", "fused_vw4", 0, "bias_gelu_erf fused", "gelu_erf", "vw4"),
    ("bias_gelu_erf", "unfused_pair_vw4", 0, "bias_gelu_erf unfused", "gelu_erf", "vw4"),
    ("bias_gelu_tanh", "fused_vw1", 0, "bias_gelu_tanh fused", "gelu_tanh", "vw1"),
    ("bias_gelu_tanh", "fused_vw2", 0, "bias_gelu_tanh fused", "gelu_tanh", "vw2"),
    ("bias_gelu_tanh", "fused_vw4", 1, "bias_gelu_tanh fused (width-sweep repeat)", "gelu_tanh", "vw4"),  # NEW
    ("bias_gelu_tanh", "fused_vw1_lowocc", 0, "bias_gelu_tanh fused, starved ({sm} blocks)", "gelu_tanh", "vw1"),
    ("bias_gelu_tanh", "fused_vw2_lowocc", 0, "bias_gelu_tanh fused, starved ({sm} blocks)", "gelu_tanh", "vw2"),
    ("bias_gelu_tanh", "fused_vw4_lowocc", 0, "bias_gelu_tanh fused, starved ({sm} blocks)", "gelu_tanh", "vw4"),
    ("bias_gelu_tanh_L2", "fused_vw4_512x512", 0, "bias_gelu_tanh fused, L2 control", "gelu_tanh", "vw4"),
    ("bias_gelu_tanh_L2", "unfused_pair_vw4_512x512", 0, "bias_gelu_tanh unfused, L2 control", "gelu_tanh", "vw4"),
]


@spec("s3a-bias-act", ["t4"])
def _s3a(d):
    ds = d["t4"]
    recs = bo.ProfilerRecords.load(ds, "phase3_bias_act.csv")
    # The starved rows cap the grid at one block per SM (bias_act_bench.cpp:307,
    # max_row_blocks = dev->sm_count), so the label's block count IS the SM count.
    sm = bo.parse_environment_block(ds.stdout("bias_act_bench"))["sm_count"]
    out = ["| Kernel | Activation | vector_width | median ms | min ms | Ideal bytes | GB/s | % measured BW | Machine |",
           "|---|---|---|---|---|---|---|---|---|"]
    for kernel, variant, occ, label, act, vw in S3A_ROWS:
        r = recs.get(kernel, variant, occ)
        out.append(row(label.format(sm=sm), act, vw, dp(r["median_ms"], 3), dp(r["min_ms"], 3),
                       commas(r["ideal_bytes"]), dp(r["achieved_gb_s"], 1),
                       dp(r["efficiency_pct"], 1) + "%", machine_label(ds)))
    return out


# §3b ------------------------------------------------------------------------
# __syncthreads per kernel, from the kernel source -- NOT parsed from stdout:
# reduce_bench prints them as hard-coded literals ("9 barriers"), so "parsing"
# them would dress a constant up as a measurement. kernels/reduce.cu:76
# (smem tree: 1 + log2(256) = 9), :124 (warp shuffle: exactly one), and the
# two-pass kernels both call the same shuffle block_reduce (:173-205): one per
# kernel, two kernels.
S3B_BARRIERS = {"smem_tree_256t": 9, "warp_shuffle_256t": 1, "two_pass_256t": 1}
S3B_ROWS = [  # (kernel, csv variant, shape)
    ("row_reduce_sum", "smem_tree_256t", "A"), ("row_reduce_max", "smem_tree_256t", "A"),
    ("row_reduce_mean", "smem_tree_256t", "A"), ("row_reduce_sum", "warp_shuffle_256t", "A"),
    ("row_reduce_max", "warp_shuffle_256t", "A"), ("row_reduce_mean", "warp_shuffle_256t", "A"),
    ("row_reduce_sum", "two_pass_256t", "A"),
    ("row_reduce_sum", "warp_shuffle_256t_starved", "B"), ("row_reduce_sum", "two_pass_256t_starved", "B"),
]


@spec("s3b-reduce", ["t4"])
def _s3b(d):
    ds = d["t4"]
    recs = bo.ProfilerRecords.load(ds, "phase3_reduce.csv")
    text = ds.stdout("reduce_bench")
    shapes = {k: bo.parse_rows_cols(text, k) for k in ("A", "B")}
    out = ["| Kernel | Variant | rows × cols | median ms | min ms | Ideal bytes | GB/s | % measured BW | "
           "__syncthreads | Machine |", "|---|---|---|---|---|---|---|---|---|---|"]
    for kernel, variant, shape in S3B_ROWS:
        r = recs.get(kernel, variant)
        base = variant.removesuffix("_starved")
        rr, cc = shapes[shape]
        out.append(row(kernel, base, f"{rr} × {cc}", dp(r["median_ms"], 3), dp(r["min_ms"], 3),
                       dp(Decimal(r["ideal_bytes"]) / (1 << 20), 2) + " MiB", dp(r["achieved_gb_s"], 1),
                       dp(r["efficiency_pct"], 1) + "%", S3B_BARRIERS[base], machine_label(ds)))
    return out


# §3c ------------------------------------------------------------------------
@spec("s3c-softmax", ["t4"])
def _s3c(d):
    ds = d["t4"]
    recs = bo.ProfilerRecords.load(ds, "phase3_softmax.csv")
    text = ds.stdout("softmax_bench")
    rr, cc = bo.parse_rows_cols(text, "shape")
    err = bo.parse_softmax_numerics(text)
    out = ["| Kernel | Variant | rows × cols | median ms | min ms | Ideal bytes | GB/s | % measured BW | "
           "max abs(Σrow − 1) | Machine |", "|---|---|---|---|---|---|---|---|---|---|"]
    for variant in ("three_pass_256t", "online_one_pass_256t"):
        r = recs.get("row_softmax", variant)
        out.append(row("row_softmax", variant, f"{rr} × {cc}", dp(r["median_ms"], 3), dp(r["min_ms"], 3),
                       dp(Decimal(r["ideal_bytes"]) / (1 << 20), 1) + " MiB", dp(r["achieved_gb_s"], 1),
                       dp(r["efficiency_pct"], 1) + "%", err[variant], machine_label(ds)))
    return out


# §3d ------------------------------------------------------------------------
# Tile shapes are not printed by gemm_bench; they are kSmemTile / kRegTile at
# gemm_bench.cpp:92-93. CAVEAT: these describe the CURRENT source. A dataset
# from an older commit is only described correctly if that commit used the
# same tiles (true for every dataset that exists as of 2026-09-22).
S3D_LADDER = [("naive_uncoalesced", "—"), ("naive", "—"), ("tiled_smem", "32,32,32,1,1"),
              ("tiled_regblock", "128,128,8,8,8"), ("warptile_nodbuf", "128,128,8,8,8"),
              ("warptile_dbuf", "128,128,8,8,8"), ("warptile_vec4", "128,128,8,8,8")]
S3D_CUBLAS = [("cublas", "cuBLAS (first)"), ("cublas_drift_recheck", "cuBLAS (drift recheck, last)")]


def s3d_rows(ds: bo.Dataset) -> list[str]:
    recs = bo.ProfilerRecords.load(ds, "phase3_gemm.csv")
    text = ds.stdout("gemm_bench")
    m, n, k = bo.parse_gemm_shape(text)
    if not (m == n == k):
        raise FenceError(f"{ds.path}: §3d's M=N=K column assumes a square GEMM, got {m}x{n}x{k}")
    occ = bo.parse_gemm_occupancy(text)
    label = machine_label(ds)
    out = []
    for variant, tile in S3D_LADDER:
        r, o = recs.get("gemm", variant), occ.get(variant)
        if o is None:
            raise FenceError(f"{ds.path}: no occupancy row for {variant}")
        occ_cell = f"{o.hand} / {o.api} ({trim_zero(o.occ_pct)}%{', tied' if o.tied else ''})"
        out.append(row(variant, m, tile, o.regs, o.smem, o.spill, occ_cell, dp(r["median_ms"], 2),
                       dp(r["min_ms"], 2), dp(r["achieved_tflops"], 3), dp(r["efficiency_pct"], 2) + "%", label))
    for variant, name in S3D_CUBLAS:
        r = recs.get("gemm", variant)
        out.append(row(name, m, "—", "—", "—", "—", "—", dp(r["median_ms"], 2), dp(r["min_ms"], 2),
                       dp(r["achieved_tflops"], 3), dp(r["efficiency_pct"], 2) + "%", label))
    return out


@spec("s3d-gemm", ["t4", "v100"])
def _s3d(d):
    return (["| Variant | M=N=K | Tile (BM,BN,BK,TM,TN) | regs/thread | smem/block | spill B | "
             "occupancy (hand / API) | median ms | min ms | TFLOP/s | % of measured FMA peak | Machine |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            + s3d_rows(d["t4"]) + s3d_rows(d["v100"]))


# §4 -------------------------------------------------------------------------
S4_GRAPHS = ["fanout4x4", "diamond_starved", "transformer_block", "diamond_gemm_2048", "chain16"]
S4_POLICIES = ["sequential", "level_parallel", "chain_greedy"]
GRAPH_COLUMNS = ["graph", "policy", "streams_used", "intra_events", "intra_waits", "forkjoin_events",
                 "median_ms", "min_ms", "speedup", "enqueue_us", "concurrency", "peak_bytes", "naive_bytes"]


@spec("s4-graph", ["t4"])
def _s4(d):
    ds = d["t4"]
    rows = bo.read_csv(ds.file("phase4_graph.csv"), GRAPH_COLUMNS)
    by = {(r["graph"], r["policy"]): r for r in rows}
    policies, gates = bo.parse_graph(ds.stdout("graph_bench"))
    out = ["| Graph | Policy | streams | median ms | min ms | speedup vs sequential | peak memory | "
           "naive memory | numerics gate |", "|---|---|---|---|---|---|---|---|---|"]
    for g in S4_GRAPHS:
        for p in S4_POLICIES:
            r = by.get((g, p))
            if r is None:
                raise FenceError(f"{ds.path}: phase4_graph.csv has no row for {g}/{p}")
            s = policies.get(g, {}).get(p)
            if s is None or s.streams_used != int(r["streams_used"]):
                raise FenceError(f"{ds.path}: {g}/{p}: stdout streams line {s} disagrees with CSV "
                                 f"streams_used={r['streams_used']}")
            gate = "—"
            if p == S4_POLICIES[-1]:
                gt = gates.get(g)
                if gt is None:
                    gate = "not run"
                else:
                    detail = f"({gt.configs} configs × {gt.repeats} repeats, {gt.elements:,} elements)"
                    # Q13's one exception: a FAIL is always bold -- a rule about the
                    # cell's own value, so it cannot go stale on promotion.
                    gate = f"PASS {detail}" if gt.passed else f"**FAIL** {detail}"
            peak, naive = int(r["peak_bytes"]), int(r["naive_bytes"])
            ratio = dp(Decimal(naive) / Decimal(peak), 2) if peak else "—"
            out.append(row(g, p, f"{s.streams_used}/{s.streams_avail}", dp(r["median_ms"], 3),
                           dp(r["min_ms"], 3), dp(r["speedup"], 2) + "×", f"{peak:,} B",
                           f"{naive:,} B ({ratio}×)", gate))
    return out


# -----------------------------------------------------------------------------
# Fences
# -----------------------------------------------------------------------------
BEGIN_RE = re.compile(r"^<!-- BEGIN GENERATED id=([a-z0-9-]+) source=(\S+)(?: \(.*\))? -->$")
END_RE = re.compile(r"^<!-- END GENERATED id=([a-z0-9-]+) -->$")
AUTHORED_RE = re.compile(r"^<!-- AUTHORED: .+ -->$")


@dataclass
class Fence:
    id: str
    sources: list[str]
    begin: int   # 0-based line index of the BEGIN line
    end: int     # 0-based line index of the END line


def parse_fences(lines: list[str]) -> list[Fence]:
    fences, open_ = [], None
    seen: set[str] = set()
    for i, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        if line.startswith("<!-- BEGIN GENERATED"):
            m = BEGIN_RE.match(line)
            if not m:
                raise FenceError(f"line {i+1}: malformed BEGIN fence: {line!r}")
            if open_:
                raise FenceError(f"line {i+1}: BEGIN id={m.group(1)} inside open fence id={open_[0]} "
                                 f"(opened at line {open_[1]+1}); fences do not nest")
            open_ = (m.group(1), i, m.group(2).split(","))
        elif line.startswith("<!-- END GENERATED"):
            m = END_RE.match(line)
            if not m:
                raise FenceError(f"line {i+1}: malformed END fence: {line!r}")
            if not open_:
                raise FenceError(f"line {i+1}: END id={m.group(1)} with no open fence")
            if m.group(1) != open_[0]:
                raise FenceError(f"line {i+1}: END id={m.group(1)} closes BEGIN id={open_[0]} "
                                 f"(line {open_[1]+1})")
            fid, b, sources = open_
            if fid not in SPECS:
                raise FenceError(f"line {b+1}: unknown table id {fid!r} (known: {sorted(SPECS)})")
            if fid in seen:
                raise FenceError(f"line {b+1}: duplicate fence id {fid!r}")
            if len(sources) != len(SPECS[fid].slots):
                raise FenceError(f"line {b+1}: id={fid} needs {len(SPECS[fid].slots)} source(s) "
                                 f"{SPECS[fid].slots}, got {len(sources)}: {sources}")
            seen.add(fid)
            fences.append(Fence(fid, sources, b, i))
            open_ = None
    if open_:
        raise FenceError(f"line {open_[1]+1}: fence id={open_[0]} is never closed")
    return fences


def load_slots(f: Fence, root: str) -> dict[str, bo.Dataset] | None:
    if any(s == "PENDING" for s in f.sources):
        return None
    return {slot: bo.Dataset.load(os.path.join(root, src)) for slot, src in zip(SPECS[f.id].slots, f.sources)}


def eol_of(line: str) -> str:
    return "\r\n" if line.endswith("\r\n") else "\n"


def render_text(text: str, root: str = REPO) -> tuple[str, list[tuple[str, str]]]:
    """Returns (new_text, [(fence_id, status)]). status: PENDING / unchanged / changed."""
    lines = text.splitlines(keepends=True)
    fences = parse_fences(lines)
    report, out, cursor = [], [], 0
    for f in fences:
        out.extend(lines[cursor:f.begin + 1])
        inner = lines[f.begin + 1:f.end]
        slots = load_slots(f, root)
        if slots is None:
            out.extend(inner)
            report.append((f.id, "PENDING"))
        else:
            nl = eol_of(lines[f.begin])
            table = SPECS[f.id].render(slots)
            new_inner = [nl] + [t + nl for t in table] + [nl]
            out.extend(new_inner)
            report.append((f.id, "unchanged" if new_inner == inner else "changed"))
        cursor = f.end
    out.extend(lines[cursor:])
    return "".join(out), report


# -----------------------------------------------------------------------------
# Table comparison -- precision-aware. Used by --preview (does a fresh run
# reproduce the published table?) and by the self-test.
#
# Two cells "agree" if their text is identical once every number is replaced
# by '#', and each number differs by no more than half a unit in the last
# place the PUBLISHED cell shows. That last clause is what makes a 3-decimal
# render of a value published at 1 decimal (the sticky-precision cells) count
# as a reproduction rather than a difference.
# -----------------------------------------------------------------------------
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def table_cells(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|") or re.match(r"^\|[\s:|-]+\|$", s):
            continue
        rows.append([c.strip().replace("**", "") for c in s.strip("|").split("|")])
    return rows[1:] if rows else rows   # drop the header row


def cell_delta(pub: str, new: str) -> str | None:
    if pub == new:
        return None
    if NUM_RE.sub("#", pub) != NUM_RE.sub("#", new):
        return f"{pub!r} -> {new!r}"
    worst = None
    for a, b in zip(NUM_RE.findall(pub), NUM_RE.findall(new)):
        da, db = Decimal(a.replace(",", "")), Decimal(b.replace(",", ""))
        places = len(a.split(".")[1]) if "." in a else 0
        if abs(da - db) > Decimal(5).scaleb(-(places + 1)):
            rel = f"{(db - da) / da * 100:+.1f}%" if da else "n/a"
            worst = f"{pub} -> {new} ({rel})"
    return worst


def compare_tables(published: list[str], rendered: list[str]) -> list[str]:
    """Rows are aligned by (first cell, occurrence): the k-th published row whose
    first cell is X pairs with the k-th rendered row whose first cell is X.

    NOT by matching every text-bearing cell -- an earlier version did, and a row
    whose only change was a label or a status (an inline note moving out per
    Q13, or a numerics gate going PASS -> FAIL) then failed to pair at all and
    was reported as "missing" + "new", hiding that row's numeric comparison.
    A gate flipping to FAIL is exactly the change this report must show as a
    CELL difference, not bury."""
    pub, new = table_cells(published), table_cells(rendered)

    def keys(rows):
        seen: dict[str, int] = {}
        out = []
        for r in rows:
            n = seen.get(r[0], 0)
            out.append((r[0], n))
            seen[r[0]] = n + 1
        return out

    pk, nk = keys(pub), keys(new)
    nindex = {k: j for j, k in enumerate(nk)}
    report, used = [], set()
    for i, (prow, k) in enumerate(zip(pub, pk)):
        j = nindex.get(k)
        if j is None or len(new[j]) != len(prow):
            report.append(f"published row {i+1} has no counterpart: {' | '.join(prow)}")
            continue
        used.add(j)
        for c, (a, b) in enumerate(zip(prow, new[j])):
            if (d := cell_delta(a, b)) is not None:
                report.append(f"row {i+1} ({prow[0]}), col {c+1}: {d}")
    for j, nrow in enumerate(new):
        if j not in used:
            report.append(f"NEW row in render: {' | '.join(nrow)}")
    return report


# -----------------------------------------------------------------------------
# --audit and --stale-prose
# -----------------------------------------------------------------------------
SEP_RE = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


def audit(lines: list[str]) -> list[str]:
    fences = parse_fences(lines)
    inside = {i for f in fences for i in range(f.begin, f.end + 1)}
    problems = []
    for i in range(len(lines) - 1):
        if lines[i].startswith("|") and SEP_RE.match(lines[i + 1].strip()) and i not in inside:
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j < 0 or not AUTHORED_RE.match(lines[j].strip()):
                problems.append(f"line {i+1}: table is neither fenced nor marked "
                                f"<!-- AUTHORED: ... -->: {lines[i].strip()[:70]}")
    return problems


INTERESTING = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,})(?![\w])")


def section_span(lines: list[str], at: int) -> tuple[int, int]:
    start = next((i for i in range(at, -1, -1) if re.match(r"^#{2,3} ", lines[i])), 0)
    level = len(lines[start]) - len(lines[start].lstrip("#")) if lines[start].startswith("#") else 2
    end = next((i for i in range(at + 1, len(lines))
                if re.match(rf"^#{{2,{level}}} ", lines[i])), len(lines))
    return start, end


def stale_prose(text: str, root: str = REPO) -> list[str]:
    lines = text.splitlines(keepends=True)
    fences = parse_fences(lines)
    inside = {i for f in fences for i in range(f.begin, f.end + 1)}
    report = []
    for f in fences:
        slots = load_slots(f, root)
        if slots is None:
            continue
        old = set(INTERESTING.findall("".join(lines[f.begin + 1:f.end])))
        new = set(INTERESTING.findall("\n".join(SPECS[f.id].render(slots))))
        gone = old - new
        if not gone:
            continue
        s, e = section_span(lines, f.begin)
        for i in range(s, e):
            if i in inside:
                continue
            hits = sorted(n for n in gone if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", lines[i]))
            if hits:
                report.append(f"line {i+1} ({f.id}): quotes {', '.join(hits)} -- not in the new render")
    return report


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def self_test() -> int:
    fails = 0
    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))
        fails += 0 if ok else 1

    print("render_results.py self-test (no GPU, no network)")

    # Rounding: the exact case that motivated Decimal.
    check("half-up rounding: 45.605 -> 45.61 (round() would give 45.6)", dp("45.605", 2) == "45.61")
    check("half-up rounding keeps trailing zeros: 39.7017 -> 39.70", dp("39.7017", 2) == "39.70")

    # G1: golden test against REAL published data. The Session-5 dataset's T4
    # rows must reproduce RESULTS.md §3d's published T4 rows byte for byte.
    results = open(RESULTS, encoding="utf-8").read().splitlines()
    published = [ln for ln in results if ln.startswith("| ") and ln.endswith("| Colab T4 |")
                 and ln.split("|")[2].strip() == "4096"]
    try:
        ds = bo.Dataset.load(os.path.join(REPO, "reports/colab-t4/2026-08-30_session5"))
        rendered = s3d_rows(ds)
        check(f"G1: §3d T4 rows ({len(rendered)}) reproduce RESULTS.md byte-for-byte",
              rendered == published,
              "\n" + "\n".join(difflib.unified_diff(published, rendered, "published", "rendered", lineterm="")))
    except Exception as e:  # noqa: BLE001 -- report, don't crash, in a self-test
        check("G1: §3d T4 rows reproduce RESULTS.md", False, repr(e))

    # G6: fence grammar -- every malformation is fatal, with a line number.
    doc_ok = "a\n<!-- BEGIN GENERATED id=s1-elementwise source=PENDING -->\n\n| x |\n\n<!-- END GENERATED id=s1-elementwise -->\nb\n"
    bad = {
        "unterminated": "<!-- BEGIN GENERATED id=s1-elementwise source=PENDING -->\n| x |\n",
        "mismatched end": "<!-- BEGIN GENERATED id=s1-elementwise source=PENDING -->\n<!-- END GENERATED id=s3c-softmax -->\n",
        "nested": ("<!-- BEGIN GENERATED id=s1-elementwise source=PENDING -->\n"
                   "<!-- BEGIN GENERATED id=s3c-softmax source=PENDING -->\n"),
        "unknown id": "<!-- BEGIN GENERATED id=nope source=PENDING -->\n<!-- END GENERATED id=nope -->\n",
        "slot arity": ("<!-- BEGIN GENERATED id=s3d-gemm source=PENDING -->\n"
                       "<!-- END GENERATED id=s3d-gemm -->\n"),
        "malformed": "<!-- BEGIN GENERATED id=s1-elementwise -->\n",
        "orphan end": "<!-- END GENERATED id=s1-elementwise -->\n",
    }
    for name, doc in bad.items():
        try:
            parse_fences(doc.splitlines(keepends=True))
            check(f"G6: {name} fence rejected", False, "was accepted")
        except FenceError as e:
            check(f"G6: {name} fence rejected with a line number", "line " in str(e), str(e))

    # G4: all-PENDING document is byte-identical after rendering (CRLF too).
    for eol in ("\n", "\r\n"):
        doc = doc_ok.replace("\n", eol)
        new, rep = render_text(doc)
        check(f"G4: PENDING fence leaves the document byte-identical ({'CRLF' if eol == chr(13)+chr(10) else 'LF'})",
              new == doc and rep == [("s1-elementwise", "PENDING")])

    # G9: an INVALID dataset is refused.
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "manifest.json"), "w").write(
            '{"status": "INVALID", "invalid_reasons": ["graph_bench: a numerics gate FAILED (1x)"]}')
        try:
            bo.Dataset.load(tmp)
            check("G9: INVALID dataset refused", False, "was loaded")
        except bo.BenchOutputError as e:
            check("G9: INVALID dataset refused, reason surfaced", "numerics gate FAILED" in str(e), str(e))

    # G7: stale-prose flags a prose number that a new render would orphan.
    with tempfile.TemporaryDirectory() as tmp:
        dsdir = os.path.join(tmp, "reports", "colab-t4", "fake")
        os.makedirs(dsdir)
        open(os.path.join(dsdir, "manifest.json"), "w").write('{"status": "VALID", "machine_tag": "colab-t4"}')
        for bench, gbs, med, mn in (("stream_triad", "231.9", "3.472", "3.470"), ("smoke", "240.5", "3.348", "3.344")):
            k = "stream_triad" if bench == "stream_triad" else "vector_add"
            open(os.path.join(dsdir, f"{bench}.stdout.log"), "w").write(
                f"{k:22}{'grid_stride_256t':30}{med:>10}{mn:>10}{gbs:>10}{'0.100':>10}{'0.17':>8}{'100.0':>9}%  memory\n"
                f"achieved {gbs} GB/s\n")
        doc = ("## 1. Phase 1\n\n<!-- BEGIN GENERATED id=s1-elementwise source=reports/colab-t4/fake -->\n\n"
               "| Kernel | Variant | n | Ideal bytes | median ms | min ms | GB/s | % measured BW | Machine |\n"
               "|---|---|---|---|---|---|---|---|---|\n"
               "| stream_triad | grid_stride_256t | 64Mi | 768 MiB | 3.421 | 3.417 | 235.4 | 100.0% | Colab T4 |\n\n"
               "<!-- END GENERATED id=s1-elementwise -->\n\nThe triad reached 235.4 GB/s here.\n\n## 2. Next\n\nAlso 235.4.\n")
        rep = stale_prose(doc, root=tmp)
        # Line numbers are 1-based in the report: heading(1) blank(2) BEGIN(3) blank(4)
        # header(5) sep(6) row(7) blank(8) END(9) blank(10) -> the prose is line 11,
        # and the other section's "Also 235.4." is line 15.
        check("G7: stale-prose catches the orphaned 235.4 in its own section",
              len(rep) == 1 and "235.4" in rep[0] and "line 11 " in rep[0], str(rep))
        check("G7: ...and ignores the same number in a DIFFERENT section", all("line 15 " not in r for r in rep))

        # G3: render, then render again -> the second pass changes nothing.
        once, _ = render_text(doc, root=tmp)
        twice, rep2 = render_text(once, root=tmp)
        check("G3: rendering is idempotent", once == twice and rep2 == [("s1-elementwise", "unchanged")])
        check("G3: ...and only the fenced region changed",
              once.split("<!-- BEGIN")[0] == doc.split("<!-- BEGIN")[0]
              and once.split("<!-- END GENERATED id=s1-elementwise -->")[1]
              == doc.split("<!-- END GENERATED id=s1-elementwise -->")[1])

    # Comparison: a 3-dp render of a value published at 1 dp reproduces it.
    check("compare: 0.523 vs published 0.5 counts as a reproduction", cell_delta("0.5", "0.523") is None)
    check("compare: 243.7 -> 241.2 is reported with its relative change",
          (cell_delta("243.7", "241.2") or "").endswith("(-1.0%)"))

    # Row alignment: a gate going PASS -> FAIL must surface as a CELL difference on
    # its own row -- not as "missing row" + "new row", which is how an earlier
    # version of compare_tables buried it.
    pub = ["| Graph | Policy | gate |", "|---|---|---|",
           "| g | sequential | — |", "| g | chain_greedy | PASS (9 configs × 20 repeats, 1 elements) |"]
    new = ["| Graph | Policy | gate |", "|---|---|---|",
           "| g | sequential | — |", "| g | chain_greedy | **FAIL** (9 configs × 20 repeats, 1 elements) |"]
    rep = compare_tables(pub, new)
    check("compare: PASS -> FAIL is reported as a cell difference on its own row",
          len(rep) == 1 and rep[0].startswith("row 2 (g), col 3:"), str(rep))

    print("all self-tests PASSED" if not fails else f"*** {fails} self-test(s) FAILED ***")
    return 1 if fails else 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Render RESULTS.md's generated tables from pinned datasets. "
                                            "See the comment block at the top of this file.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--diff", action="store_true")
    g.add_argument("--audit", action="store_true")
    g.add_argument("--stale-prose", action="store_true")
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--preview", nargs="+", metavar=("ID", "SOURCE"),
                   help="render table ID from the given dataset dir(s), one per slot, and compare")
    p.add_argument("--file", default=RESULTS)
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    text = open(a.file, encoding="utf-8", newline="").read()
    lines = text.splitlines(keepends=True)
    try:
        if a.audit:
            problems = audit(lines)
            for pr in problems:
                print(pr)
            print(f"audit: {len(problems)} unclassified table(s)")
            return 1 if problems else 0
        if a.stale_prose:
            rep = stale_prose(text)
            for r in rep:
                print(r)
            print(f"stale-prose: {len(rep)} line(s) quote numbers the pinned renders would orphan")
            return 1 if rep else 0
        if a.preview:
            fid, srcs = a.preview[0], a.preview[1:]
            if fid not in SPECS or len(srcs) != len(SPECS[fid].slots):
                p.error(f"--preview {fid}: need {len(SPECS.get(fid, Spec('', [], None)).slots)} source(s)")
            slots = {s: bo.Dataset.load(os.path.join(REPO, src)) for s, src in zip(SPECS[fid].slots, srcs)}
            rendered = SPECS[fid].render(slots)
            print("\n".join(rendered))
            f = next((f for f in parse_fences(lines) if f.id == fid), None)
            if f is None:
                print(f"\n(no fence id={fid} in {a.file}; nothing to compare against)")
                return 0
            rep = compare_tables([ln.rstrip("\r\n") for ln in lines[f.begin + 1:f.end]], rendered)
            print(f"\n=== {fid}: fresh render vs published ({len(rep)} difference(s) beyond published precision) ===")
            for r in rep:
                print("  " + r)
            return 0
        new, rep = render_text(text)
        for fid, status in rep:
            print(f"  {status:9}  {fid}")
        changed = new != text
        if a.check:
            return 1 if changed else 0
        if a.diff:
            sys.stdout.writelines(difflib.unified_diff(lines, new.splitlines(keepends=True), a.file, a.file + " (rendered)"))
            return 0
        if changed:
            for r in stale_prose(text):
                print(f"  WARNING stale prose: {r}")
            d = os.path.dirname(os.path.abspath(a.file))
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=d, delete=False) as tf:
                tf.write(new)
            os.replace(tf.name, a.file)   # atomic: never a half-written RESULTS.md
            print(f"wrote {a.file}")
        return 0
    except (FenceError, bo.BenchOutputError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
