#!/usr/bin/env python3
# =============================================================================
#  tools/bench_outputs.py — read one benchmark DATASET: its manifest, its CSVs,
#  and the table-shaped data its benches print only to stdout.
#
#  Phase 5 stage 5f. The data-access half of tools/render_results.py.
#
#  WHY A SEPARATE MODULE: render_results.py decides what a TABLE looks like;
#  this file decides what a BENCH SAID. Keeping them apart means every parser
#  here can be tested against a bench's real output (or its exact printf format)
#  without any knowledge of RESULTS.md, and a change in a bench's output format
#  breaks exactly one function in exactly one file.
#
#  WHY PARSE STDOUT AT ALL (DECISIONS.md Q11): many columns RESULTS.md publishes
#  are printed but never written to a file -- GEMM registers/occupancy, softmax
#  numerics, the graph numerics gate, device facts, the fragmentation table.
#  The alternative was adding structured output to ~8 C++ programs that cannot
#  run on the machine this was written on. Parsing fixed printf formats needed
#  zero C++ changes, and the one real stdout log that exists (Session 5's GEMM
#  run) makes the hardest parser testable immediately.
#
#  THE ONE RULE EVERY PARSER HERE FOLLOWS: fail loudly on a format it does not
#  recognise, with the offending text. Never return a partial or default value.
#  A parser that silently returns nothing when a bench changes its output would
#  turn a format change into a missing or wrong table cell -- the confidently
#  wrong, no-error failure this project's profiling discipline exists to stop.
#  Each parser names the printf it reads (file:line), so a mismatch points
#  straight at the code that changed.
#
#  WHY .py and stdlib only: analysis tooling, never runtime (CLAUDE.md section
#  5), and it must run anywhere a dataset is -- Colab, Explorer, this Mac --
#  with no pip install.
# =============================================================================
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field


class BenchOutputError(ValueError):
    """A bench's output did not match the format this module expects."""


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
@dataclass
class Dataset:
    path: str
    manifest: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str, allow_invalid: bool = False) -> "Dataset":
        mpath = os.path.join(path, "manifest.json")
        if not os.path.isfile(mpath):
            raise BenchOutputError(
                f"{path}: no manifest.json -- not a dataset. Datasets are made by "
                f"scripts/regen_results.sh (or, for pre-5f runs, a hand-written "
                f"manifest marked \"reconstructed\": true).")
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("status") != "VALID" and not allow_invalid:
            reasons = manifest.get("invalid_reasons") or ["(no reasons recorded)"]
            raise BenchOutputError(
                f"{path}: dataset status is {manifest.get('status')!r}, refusing to render "
                f"from it -- a failed correctness check must never become a published "
                f"table. Reasons: " + "; ".join(reasons))
        return cls(path=path, manifest=manifest)

    def file(self, rel: str) -> str:
        p = os.path.join(self.path, rel)
        if not os.path.isfile(p):
            raise BenchOutputError(f"{self.path}: expected file {rel!r} is missing")
        return p

    def stdout(self, bench: str) -> str:
        with open(self.file(f"{bench}.stdout.log"), encoding="utf-8", errors="replace") as f:
            return f.read()

    @property
    def tag(self) -> str:
        return self.manifest.get("machine_tag", "")


# -----------------------------------------------------------------------------
# CSVs
# -----------------------------------------------------------------------------
PROFILER_COLUMNS = [
    "kernel", "variant", "median_ms", "min_ms", "iterations", "ideal_flops", "ideal_bytes",
    "achieved_gb_s", "achieved_tflops", "arithmetic_intensity", "attainable_tflops",
    "efficiency_pct", "bound",
]


def read_csv(path: str, expect_columns: list[str] | None = None) -> list[dict]:
    """Rows as dicts of STRINGS. Values stay strings on purpose: rounding is done
    later from the decimal string (see render_results.fmt), and a float round
    trip here would re-introduce binary-representation error first."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if expect_columns is not None and header != expect_columns:
            raise BenchOutputError(
                f"{path}: header does not match the expected schema.\n"
                f"  expected: {expect_columns}\n  actual:   {header}")
        return list(reader)


class ProfilerRecords:
    """Profiler::write_csv rows, addressable by (kernel, variant, occurrence).

    `occurrence` exists because a bench may time the SAME configuration twice
    in one run -- bias_act_bench records (bias_gelu_tanh, fused_vw4) once in its
    activation sweep and again in its width sweep. Last-wins lookup would pick
    one silently; RESULTS.md §3a's table and prose did in fact pick different
    ones (247.8 vs 242.8 GB/s). Here the caller must say which.
    """

    def __init__(self, rows: list[dict], source: str):
        self.source = source
        self._rows = rows
        self._index: dict[tuple[str, str, int], dict] = {}
        seen: dict[tuple[str, str], int] = {}
        for r in rows:
            k = (r["kernel"], r["variant"])
            n = seen.get(k, 0)
            self._index[(k[0], k[1], n)] = r
            seen[k] = n + 1
        self._counts = seen

    @classmethod
    def load(cls, ds: Dataset, filename: str) -> "ProfilerRecords":
        path = ds.file(filename)
        return cls(read_csv(path, PROFILER_COLUMNS), path)

    def get(self, kernel: str, variant: str, occurrence: int = 0) -> dict:
        try:
            return self._index[(kernel, variant, occurrence)]
        except KeyError:
            raise BenchOutputError(
                f"{self.source}: no record ({kernel!r}, {variant!r}) occurrence {occurrence}. "
                f"Present: {sorted(self._counts.items())}") from None

    def count(self, kernel: str, variant: str) -> int:
        return self._counts.get((kernel, variant), 0)


# -----------------------------------------------------------------------------
# stdout parsers. Each cites the printf it reads.
# -----------------------------------------------------------------------------
def _need(m, what: str, text: str):
    if m is None:
        snippet = text[:400].replace("\n", "\n    ")
        raise BenchOutputError(f"could not find {what}. Start of the text searched:\n    {snippet}")
    return m


def parse_environment_block(text: str) -> dict:
    """bench_common.hpp print_denominators: 'device  <name> (sm_XY), N SMs, K KiB smem/SM'."""
    m = _need(re.search(r"^device\s+(.+?) \(sm_(\d)(\d+)\), (\d+) SMs, (\d+) KiB smem/SM$",
                        text, re.M), "the print_denominators 'device' line (bench_common.hpp:89)", text)
    return {"name": m.group(1), "cc": f"{m.group(2)}.{m.group(3)}",
            "sm_count": int(m.group(4)), "smem_per_sm_kib": int(m.group(5))}


def parse_gemm_shape(text: str) -> tuple[int, int, int]:
    """gemm_bench.cpp:246 'shape         M=%lld N=%lld K=%lld   alpha=1 beta=0 (timed)'."""
    m = _need(re.search(r"^shape\s+M=(\d+) N=(\d+) K=(\d+)", text, re.M),
              "gemm_bench's 'shape M= N= K=' line (gemm_bench.cpp:246)", text)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


@dataclass
class Occupancy:
    regs: int
    smem: int
    spill: int
    hand: int
    api: int
    occ_pct: str      # verbatim, e.g. "100.0"
    limiter: str      # may contain spaces ("threads/SM cap", "shared memory")
    tied: bool


def parse_gemm_occupancy(text: str) -> dict[str, Occupancy | None]:
    """gemm_bench.cpp:528 header / :538 rows / :533 the cuBLAS no-attributes row.

    Row format "%-18s %5d %6zu %7zu %8d %8d %7.1f%%  %s%s": name, regs/thread,
    static smem bytes, local (spill) bytes, hand-computed blocks/SM, API
    blocks/SM, occupancy%, limiter, optional " (tied)". The limiter can contain
    spaces, so it is taken to end of line. cuBLAS maps to None: its kernels are
    not ours to introspect, which is why RESULTS.md prints em-dashes for it.
    """
    hdr = _need(re.search(r"^variant\s+regs\s+smem\s+spill\s+hand\s+api\s+occ%.*$", text, re.M),
                "gemm_bench's occupancy table header (gemm_bench.cpp:528)", text)
    out: dict[str, Occupancy | None] = {}
    row = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(.+?)(\s+\(tied\))?\s*$")
    noattr = re.compile(r"^(\S+)\s+\(no attributes:")
    for line in text[hdr.end():].splitlines()[1:]:
        if not line.strip():
            break
        if line.lstrip().startswith("^"):          # warning continuation lines (:544, :548)
            continue
        if m := noattr.match(line):
            out[m.group(1)] = None
            continue
        m = row.match(line)
        if not m:
            raise BenchOutputError(f"unrecognised gemm occupancy row (gemm_bench.cpp:538): {line!r}")
        out[m.group(1)] = Occupancy(int(m.group(2)), int(m.group(3)), int(m.group(4)),
                                    int(m.group(5)), int(m.group(6)), m.group(7),
                                    m.group(8).strip(), bool(m.group(9)))
    if not out:
        raise BenchOutputError("gemm occupancy table header found but no rows followed it")
    return out


def parse_summary_row(text: str, kernel: str, variant: str) -> dict:
    """profiler.cpp summary_table row: kernel variant med_ms min_ms GB/s TFLOP/s AI %peak bound.
    Names contain no spaces, so a whitespace split is exact. NOTE: before the Phase 5
    stage 5f fix, rows after the first carried 1-decimal med/min (sticky
    setprecision); stream_triad and mcke_smoke each print a single row, so their
    first-row values were always 3 dp."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 9 and parts[0] == kernel and parts[1] == variant and parts[7].endswith("%"):
            return {"median_ms": parts[2], "min_ms": parts[3], "gb_s": parts[4],
                    "tflops": parts[5], "ai": parts[6], "peak_pct": parts[7][:-1], "bound": parts[8]}
    raise BenchOutputError(f"no summary_table row for {kernel}/{variant} (profiler.cpp:summary_table)")


def parse_achieved_gbs(text: str) -> str:
    """stream_triad.cu:124 'achieved %.1f GB/s  (spec-formula peak ...)' and
    smoke_vector_add.cpp:108 'achieved %.1f GB/s of %.1f GB/s peak ...'."""
    m = _need(re.search(r"^achieved ([\d.]+) GB/s", text, re.M), "the 'achieved %.1f GB/s' line", text)
    return m.group(1)


def parse_fma_peak(text: str) -> str:
    """fma_peak.cu 'measured f32 FMA peak: %.3f TFLOP/s'."""
    m = _need(re.search(r"^measured f32 FMA peak: ([\d.]+) TFLOP/s", text, re.M),
              "fma_peak's 'measured f32 FMA peak' line", text)
    return m.group(1)


def parse_device_query(text: str) -> dict:
    """tools/device_query.cpp + DeviceInfo::describe() (src/core/device.cpp:27-47),
    plus device_query.cpp:51 '  roofline: peak_bw=%.1f GB/s (spec formula)'."""
    head = _need(re.search(r"^\[device 0\] (.+?)  sm_(\d)(\d+)$", text, re.M),
                 "DeviceInfo::describe()'s '[device 0] <name>  sm_XY' line", text)
    def field_(label: str) -> str:
        m = _need(re.search(rf"^  {re.escape(label)}\s*: (.+)$", text, re.M),
                  f"describe() field {label!r}", text)
        return m.group(1).strip()
    bw = _need(re.search(r"roofline: peak_bw=([\d.]+) GB/s", text), "device_query's roofline line", text)
    return {"name": head.group(1), "cc": f"{head.group(2)}.{head.group(3)}",
            "sm_count": int(field_("SMs")),
            "smem_per_sm": field_("shared mem / SM"),       # e.g. "64 KiB"
            "peak_bw_spec": bw.group(1)}


def parse_softmax_numerics(text: str) -> dict[str, str]:
    """softmax_bench.cpp:248 '  max |sum(row) - 1|   three_pass %.3e   online %.3e   (ratio %.2fx)'."""
    m = _need(re.search(r"max \|sum\(row\) - 1\|\s+three_pass (\S+)\s+online (\S+)", text),
              "softmax_bench's 'max |sum(row) - 1|' summary line (softmax_bench.cpp:248)", text)
    return {"three_pass_256t": m.group(1), "online_one_pass_256t": m.group(2)}


def parse_rows_cols(text: str, which: str) -> tuple[int, int]:
    """reduce_bench.cpp:138 'shapes  A = R x C (...)   B = R x C (...)' (which='A'/'B'),
    softmax_bench.cpp 'shape  R x C   compulsory bytes ...' (which='shape')."""
    if which == "shape":
        m = _need(re.search(r"^shape\s+(\d+) x (\d+)\s", text, re.M), "the 'shape R x C' line", text)
    else:
        m = _need(re.search(rf"\b{which} = (\d+) x (\d+) \(", text), f"reduce's shape {which} line", text)
    return int(m.group(1)), int(m.group(2))


@dataclass
class GraphPolicy:
    streams_used: int
    streams_avail: int


@dataclass
class GraphGate:
    passed: bool
    configs: int
    repeats: int
    elements: int


def parse_graph(text: str) -> tuple[dict[str, dict[str, GraphPolicy]], dict[str, GraphGate]]:
    """graph_bench.cpp: section header '=== <graph> ===...' (:315), per-policy
    '  %-15s streams %zu/%d  events ...' (:410), and the gate line
    '  numerics gate  PASS|*** FAIL ***  (%d configs x %d repeats, %zu elements)' (:454)."""
    policies: dict[str, dict[str, GraphPolicy]] = {}
    gates: dict[str, GraphGate] = {}
    graph = None
    sec = re.compile(r"^=== (\S+) ===")
    pol = re.compile(r"^  (\S+)\s+streams (\d+)/(\d+)  events ")
    gate = re.compile(r"^  numerics gate  (PASS|\*\*\* FAIL \*\*\*)  \((\d+) configs x (\d+) repeats, (\d+) elements\)")
    for line in text.splitlines():
        if m := sec.match(line):
            graph = m.group(1)
        elif (m := pol.match(line)) and graph:
            policies.setdefault(graph, {})[m.group(1)] = GraphPolicy(int(m.group(2)), int(m.group(3)))
        elif (m := gate.match(line)) and graph:
            gates[graph] = GraphGate(m.group(1) == "PASS", int(m.group(2)), int(m.group(3)), int(m.group(4)))
    if not policies:
        raise BenchOutputError("graph_bench stdout: no '<policy> streams N/K' lines found (graph_bench.cpp:410)")
    return policies, gates


_BYTES = r"[\d.]+ (?:B|KiB|MiB|GiB)"


def parse_alloc_fragmentation(text: str) -> dict[str, list[list[str]]]:
    """alloc_bench.cpp:854/:859 fragmentation table, one per trace, cells kept VERBATIM
    (human_bytes strings like '29.80 MiB' and '%.1f%%' percentages). No numeric round
    trip: §2b's published cells ARE these strings, so reproducing them byte-for-byte
    needs no rounding logic at all. Traces are delimited by alloc_bench's
    '############### trace = <name> ###' banners."""
    out: dict[str, list[list[str]]] = {}
    trace = None
    in_table = False
    row = re.compile(rf"^(\S+)\s+({_BYTES})\s+({_BYTES})\s+({_BYTES})\s+([\d.]+%)\s+([\d.]+%)\s+"
                     rf"([\d.]+%)\s+({_BYTES})\s+({_BYTES}|n/a)\s+(yes|no)\s*$")
    for line in text.splitlines():
        if m := re.match(r"^#+ trace = (\S+) #", line):
            trace, in_table = m.group(1), False
        elif line.startswith("=== 2b. Fragmentation"):
            in_table = True
            out.setdefault(trace or "?", [])
        elif in_table:
            if line.startswith("allocator "):
                continue
            if not line.strip():
                in_table = False
                continue
            m = row.match(line)
            if not m:
                raise BenchOutputError(f"unrecognised fragmentation row (alloc_bench.cpp:859): {line!r}")
            out[trace or "?"].append(list(m.groups()))
    if not out:
        raise BenchOutputError("alloc_bench stdout: no '=== 2b. Fragmentation' tables found")
    return out


def nvcc_release(manifest: dict) -> str | None:
    """'Cuda compilation tools, release 12.8, V12.8.93' -> '12.8.93'. The FULL nvcc
    --version text is stored in the manifest; its last line is a build string, not
    the version (caught while writing this parser -- an earlier regen_results.sh
    captured only `tail -1`)."""
    text = (manifest.get("toolchain") or {}).get("nvcc") or ""
    m = re.search(r"release [\d.]+, V([\d.]+)", text)
    return m.group(1) if m else None
