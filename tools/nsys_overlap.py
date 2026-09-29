#!/usr/bin/env python3
# =============================================================================
#  tools/nsys_overlap.py
#
#  WHAT: computes cross-stream kernel-overlap statistics from an nsys GPU
#  trace -- the analysis that produced RESULTS.md section 5c's three headline
#  numbers ("4,404 overlapping kernel pairs", "measured maximum of 4
#  concurrently active streams", "local concurrency factor 3.46x"), which
#  existed only as unrecorded ad-hoc shell work until this file. Phase 5
#  stage 5d.
#
#  WHY .py: analysis tooling, never part of the runtime (CLAUDE.md section 5
#  reserves .py for "test harnesses and plotting"). Not a .sh: the algorithm
#  needs real data structures (spans, a sweep-line, snapshot windows) and is
#  what Python is for, not bash.
#
#  ---------------------------------------------------------------------------
#  HONESTY NOTE -- read this before trusting the extraction path
#
#  This file was written and unit-tested on a MacBook with NO `nsys` binary
#  and NO GPU. Two consequences:
#
#   1. `extract_csv_via_nsys()` (the part that shells out to `nsys stats`) has
#      never actually been run. It cannot be, from this machine.
#   2. The nsys `cuda_gpu_trace` report's exact CSV column names are not
#      published anywhere fetchable (checked; NVIDIA's own User Guide does not
#      list them, and no local `nsys` binary exists to ask directly). The
#      candidate column names in `_COLUMN_CANDIDATES` below are an informed
#      guess from RESULTS.md section 5c's own account of the command that was
#      run, not a verified schema.
#
#  Both are designed to FAIL LOUDLY with the actual header row printed, rather
#  than silently mis-parse and report a plausible-looking wrong number --
#  which would be exactly the kind of "confidently wrong, no error" failure
#  this project's whole profiling discipline exists to prevent (see
#  bench_common.hpp's peak_tflops==0 trap for the precedent).
#
#  The OVERLAP-COUNTING ALGORITHM (compute_overlaps, find_max_concurrency_
#  windows) has none of this uncertainty: it is pure Python over a list of
#  (stream, start_ns, end_ns) tuples, and IS fully verified here, on this
#  machine, with hand-checkable synthetic cases -- run `--self-test`.
#
#  The first real test of the extraction+parsing path is Phase 5 stage 5g, on
#  Colab, against the ALREADY-COMMITTED reports/nsys_phase4_fanout4x4.nsys-rep.
#  That is why this script's job in 5g is to REPRODUCE numbers already in
#  RESULTS.md, not to produce new ones: if it does not reproduce 4,404 / 4 /
#  ~3.46x, the column-detection guessed wrong, not the arithmetic -- and the
#  fix is a one-line edit to `_COLUMN_CANDIDATES`, not a rewrite of the sweep.
#
#  ---------------------------------------------------------------------------
#  USAGE
#
#    # From a .nsys-rep (needs `nsys` on PATH; shells out to `nsys stats`):
#    python3 tools/nsys_overlap.py reports/colab-t4/nsys_fanout4x4.nsys-rep
#
#    # From an already-exported CSV (what scripts/profile_nsys.sh produces, or
#    # `nsys stats --report cuda_gpu_trace --format csv` run by hand):
#    python3 tools/nsys_overlap.py --csv reports/colab-t4/nsys_fanout4x4_gputrace.csv
#
#    # Self-test only -- no nsys, no CSV, no GPU. What this Mac CAN run:
#    python3 tools/nsys_overlap.py --self-test
# =============================================================================
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Sequence


# -----------------------------------------------------------------------------
# Column-name candidates for the `cuda_gpu_trace` CSV. See the honesty note
# above: these are an informed guess, not a verified schema. If stage 5g's run
# fails with "no candidate column matched", add the real header text seen in
# the printed diagnostic to the relevant list below -- that is the intended
# fix, not a redesign.
# -----------------------------------------------------------------------------
_START_CANDIDATES = ["Start (ns)", "Start:ts_ns", "Start"]
_DURATION_CANDIDATES = ["Duration (ns)", "Duration:dur_ns", "Duration"]
_END_CANDIDATES = ["End (ns)", "End"]  # used only if present; else Start+Duration
_STREAM_CANDIDATES = ["Strm", "StrmId", "Stream Id", "Stream"]
_NAME_CANDIDATES = ["Name", "Kernel Name", "GridStr"]


@dataclass(frozen=True)
class KernelSpan:
    stream: int
    start_ns: int
    end_ns: int
    name: str = ""


@dataclass
class OverlapStats:
    total_spans: int
    overlapping_pairs: int
    max_concurrent_streams: int
    # sum(duration) / wall-clock span, for the best window achieving
    # max_concurrent_streams. None if max_concurrent_streams < 2 (no window to
    # report -- nothing overlapped at all).
    local_concurrency_factor: float | None
    example_window: list[KernelSpan] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"total kernels                 {self.total_spans}",
            f"cross-stream overlapping pairs {self.overlapping_pairs}",
            f"max concurrent streams        {self.max_concurrent_streams}",
        ]
        if self.local_concurrency_factor is not None:
            lines.append(
                f"local concurrency factor      {self.local_concurrency_factor:.2f}x "
                f"(best window, {len(self.example_window)} kernels)"
            )
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Extraction: .nsys-rep -> CSV text, via the real `nsys` binary.
#
# UNTESTED FROM THIS MACHINE (see honesty note). Isolated in its own function,
# doing nothing but the subprocess call, so the untested part is exactly one
# function long and everything downstream of it is independently verified.
# -----------------------------------------------------------------------------
def extract_csv_via_nsys(nsys_rep_path: str) -> str:
    try:
        proc = subprocess.run(
            ["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv",
             "--output", "-", nsys_rep_path],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "`nsys` not found on PATH. This function can only run on a machine "
            "with the Nsight Systems CLI installed (Colab, Explorer) -- it has "
            "never been exercised on the MacBook this script was written on. "
            "If you have a CSV already (e.g. from scripts/profile_nsys.sh), "
            "pass it with --csv instead."
        ) from None
    if proc.returncode != 0:
        raise RuntimeError(
            f"`nsys stats` exited {proc.returncode}.\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _find_column(fieldnames: Sequence[str], candidates: list[str], kind: str) -> str | None:
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _find_header_line(lines: list[str]) -> int:
    """Scan forward for the line that is actually the CSV header, rather than
    assuming line 0 is it.

    Found in the field (Explorer V100, 2026-09-29): `nsys stats` printed its
    own informational preamble to stdout BEFORE the real CSV table --
    "NOTICE: Existing SQLite export found... Consider using --force-export"
    followed by a blank line and "Processing [...] with [...]..." -- and
    `scripts/profile_nsys.sh`'s plain `>` redirect captured all of it, not
    just the table. The column-name GUESSES were exactly right on the first
    try (`Start (ns)`, `Duration (ns)`, `Strm`, `Name`, all literally present);
    the bug was assuming line 0 is the header at all. This scan is robust to
    ANY future preamble nsys chooses to print, not just this one wording --
    it looks for the first line that, parsed as CSV, contains a recognisable
    Start column AND a recognisable Stream column, and treats that as ground
    truth for where the table begins.
    """
    for i, line in enumerate(lines):
        cells = next(csv.reader([line]), [])
        if (_find_column(cells, _START_CANDIDATES, "start") is not None
                and _find_column(cells, _STREAM_CANDIDATES, "stream") is not None):
            return i
    preview = "\n".join(f"  {i}: {l!r}" for i, l in enumerate(lines[:15]))
    raise ValueError(
        "could not find a CSV header line (one containing both a recognisable "
        "Start column and a recognisable Stream column) anywhere in the input.\n"
        f"looked for start in:  {_START_CANDIDATES}\n"
        f"looked for stream in: {_STREAM_CANDIDATES}\n"
        f"first {min(15, len(lines))} line(s) of the input, for diagnosis:\n{preview}"
    )


def parse_gpu_trace_csv(text: str) -> list[KernelSpan]:
    """Parse `nsys stats --report cuda_gpu_trace --format csv` output.

    Fails loudly (ValueError, with the actual header printed) if no candidate
    column name matches -- see the honesty note at the top of this file for
    why that is the deliberate failure mode here, not a bug to silence.
    """
    lines = text.splitlines()
    header_idx = _find_header_line(lines)
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    if reader.fieldnames is None:
        raise ValueError("empty CSV: no header row found")
    fields = list(reader.fieldnames)

    start_col = _find_column(fields, _START_CANDIDATES, "start")
    stream_col = _find_column(fields, _STREAM_CANDIDATES, "stream")
    if start_col is None or stream_col is None:
        raise ValueError(
            "could not identify required columns in the nsys CSV.\n"
            f"  actual header seen: {fields}\n"
            f"  looked for start in:  {_START_CANDIDATES}\n"
            f"  looked for stream in: {_STREAM_CANDIDATES}\n"
            "This is the exact failure mode the module docstring warns about: "
            "add the real column name(s) above to the candidate lists at the "
            "top of tools/nsys_overlap.py and re-run -- do not guess further "
            "downstream, the header above is ground truth."
        )
    end_col = _find_column(fields, _END_CANDIDATES, "end")
    duration_col = None if end_col else _find_column(fields, _DURATION_CANDIDATES, "duration")
    if end_col is None and duration_col is None:
        raise ValueError(
            "found Start/Stream columns but neither an End nor a Duration "
            f"column.\n  actual header seen: {fields}\n"
            f"  looked for end in:      {_END_CANDIDATES}\n"
            f"  looked for duration in: {_DURATION_CANDIDATES}"
        )
    name_col = _find_column(fields, _NAME_CANDIDATES, "name")

    spans: list[KernelSpan] = []
    for row in reader:
        start = int(float(row[start_col]))
        stream = int(float(row[stream_col]))
        if end_col:
            end = int(float(row[end_col]))
        else:
            end = start + int(float(row[duration_col]))
        name = row[name_col] if name_col else ""
        spans.append(KernelSpan(stream=stream, start_ns=start, end_ns=end, name=name))
    return spans


# -----------------------------------------------------------------------------
# The algorithm. Pure, no I/O, no nsys dependency -- see the self-tests below
# for hand-checkable verification of every branch.
#
# APPROACH: a sweep line over (time, is_end, stream) events, END events sorted
# BEFORE START events at an identical timestamp. That tie-break is a real
# decision, not an arbitrary one: two half-open intervals [s1,e1) and [s2,e2)
# that merely TOUCH (e1 == s2) are defined here as NOT overlapping -- one
# kernel's last active instant is the next one's first, which is a boundary,
# not shared execution. Processing the END first means the ending span's
# stream is no longer "active" by the time the touching START is processed, so
# it is correctly excluded. Two spans that genuinely start at the same
# instant DO count as overlapping each other: this falls out for free, because
# each START is applied to the running counters before the next same-timestamp
# START is processed, so simultaneous starts see each other as already active.
#
# Counting pairs in O(n log n), not O(n^2): maintain `active[stream]` (a count
# of currently-open spans on that stream) and `total_active`. When a span on
# stream S starts, the number of NEW cross-stream overlapping pairs it forms
# is `total_active - active[S]` (everything currently open, minus whatever is
# already open on its OWN stream, since same-stream pairs are excluded by
# definition -- a real stream is serial, so this should be 0 in genuine nsys
# data, but the code does not assume that).
# -----------------------------------------------------------------------------
def compute_overlaps(spans: Sequence[KernelSpan], max_windows_tracked: int = 64) -> OverlapStats:
    if not spans:
        return OverlapStats(0, 0, 0, None, [])

    # event tuple: (time, kind, seq, span_index). kind=0 for END, kind=1 for
    # START -- chosen so that sorting ascending on (time, kind) puts an END at
    # a given timestamp BEFORE a START at that same timestamp, which is the
    # tie-break the "touching intervals don't overlap" rule above depends on.
    # `kind` is NOT itself the boolean "is this an end event" -- that would
    # make kind=1 (START) truthy and kind=0 (END) falsy, exactly backwards.
    # The loop below computes `is_end = (kind == 0)` explicitly rather than
    # testing `kind` directly, specifically to avoid that inversion. (Caught by
    # the self-test itself: the first version of this function tested `kind`
    # directly and ran the END branch on every START event and vice versa --
    # every self-test failed with a KeyError, which is exactly the point of
    # having them.) `seq` is a stable tiebreak so sort never compares
    # KernelSpan objects.
    events: list[tuple[int, int, int, int]] = []
    for i, s in enumerate(spans):
        events.append((s.start_ns, 1, i, i))   # START
        events.append((s.end_ns, 0, i, i))     # END, sorts first on ties
    events.sort(key=lambda e: (e[0], e[1]))

    active_count: dict[int, int] = {}
    active_spans: dict[int, set[int]] = {}   # stream -> set of active span indices
    total_active = 0
    distinct_active_streams = 0

    overlapping_pairs = 0
    max_concurrent_streams = 0
    # Windows tied for the current max, captured as (set of active span indices)
    # at the instant the max was (re-)achieved. Capped so a trace with the same
    # steady-state concurrency thousands of times over doesn't blow up memory --
    # fanout4x4-scale traces (5,684 kernels) hit this dozens of times, not
    # thousands, so 64 is generous headroom, not a tight fit.
    candidate_windows: list[frozenset[int]] = []

    for time_ns, kind, seq, idx in events:
        span = spans[idx]
        stream = span.stream
        is_end = (kind == 0)
        if is_end:
            active_count[stream] -= 1
            active_spans[stream].discard(idx)
            total_active -= 1
            if active_count[stream] == 0:
                distinct_active_streams -= 1
        else:
            prior_active_here = active_count.get(stream, 0)
            overlapping_pairs += total_active - prior_active_here
            if prior_active_here == 0:
                distinct_active_streams += 1
            active_count[stream] = prior_active_here + 1
            active_spans.setdefault(stream, set()).add(idx)
            total_active += 1

            if distinct_active_streams > max_concurrent_streams:
                max_concurrent_streams = distinct_active_streams
                candidate_windows = []
            if distinct_active_streams == max_concurrent_streams and max_concurrent_streams >= 2:
                if len(candidate_windows) < max_windows_tracked:
                    snapshot = frozenset(
                        i for ids in active_spans.values() for i in ids
                    )
                    candidate_windows.append(snapshot)

    local_concurrency_factor = None
    best_window: list[KernelSpan] = []
    if candidate_windows:
        best_ratio = -1.0
        for window in candidate_windows:
            window_spans = [spans[i] for i in window]
            total_duration = sum(s.end_ns - s.start_ns for s in window_spans)
            wall_span = (max(s.end_ns for s in window_spans)
                         - min(s.start_ns for s in window_spans))
            if wall_span <= 0:
                continue
            ratio = total_duration / wall_span
            if ratio > best_ratio:
                best_ratio = ratio
                best_window = sorted(window_spans, key=lambda s: s.stream)
        if best_window:
            local_concurrency_factor = best_ratio

    return OverlapStats(
        total_spans=len(spans),
        overlapping_pairs=overlapping_pairs,
        max_concurrent_streams=max_concurrent_streams,
        local_concurrency_factor=local_concurrency_factor,
        example_window=best_window,
    )


# -----------------------------------------------------------------------------
# Self-tests. Every case here is hand-computed in the comment beside it -- the
# standard this project applies everywhere else (buddy-allocator math,
# GEMM bank arithmetic) applied to this script. Run with --self-test; needs
# nothing but Python.
# -----------------------------------------------------------------------------
def _assert_stats(name: str, stats: OverlapStats, pairs: int, max_streams: int,
                  factor: float | None = None, tol: float = 1e-6) -> None:
    ok = stats.overlapping_pairs == pairs and stats.max_concurrent_streams == max_streams
    if factor is not None:
        ok = ok and stats.local_concurrency_factor is not None \
            and abs(stats.local_concurrency_factor - factor) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}: pairs={stats.overlapping_pairs} "
          f"max_streams={stats.max_concurrent_streams} "
          f"factor={stats.local_concurrency_factor}")
    if not ok:
        raise SystemExit(
            f"self-test FAILED: {name} expected pairs={pairs} max_streams={max_streams} "
            f"factor={factor}, got pairs={stats.overlapping_pairs} "
            f"max_streams={stats.max_concurrent_streams} "
            f"factor={stats.local_concurrency_factor}"
        )


def run_self_test() -> None:
    print("nsys_overlap.py self-test (no nsys, no GPU required)")

    # 1. Two streams, disjoint in time -> no overlap, concurrency never exceeds 1.
    spans = [KernelSpan(0, 0, 10), KernelSpan(1, 20, 30)]
    _assert_stats("disjoint", compute_overlaps(spans), pairs=0, max_streams=1)

    # 2. Two streams, identical windows -> exactly one overlapping pair,
    #    concurrency 2, and the concurrency factor is exactly 2.0x (both
    #    kernels' full duration falls inside the shared wall-clock window).
    spans = [KernelSpan(0, 0, 10), KernelSpan(1, 0, 10)]
    _assert_stats("full-overlap", compute_overlaps(spans), pairs=1, max_streams=2, factor=2.0)

    # 3. Same-stream "overlap" must NOT count as a cross-stream pair, and must
    #    not count as concurrency > 1 either (it is one stream, however many
    #    spans on it happen to be marked active).
    spans = [KernelSpan(0, 0, 10), KernelSpan(0, 5, 15)]
    _assert_stats("same-stream-only", compute_overlaps(spans), pairs=0, max_streams=1)

    # 4. Touching boundary: A=[0,10) on stream0, B=[5,15) on stream1 overlap;
    #    C=[10,20) on stream2 TOUCHES A (A ends exactly when C starts) and must
    #    NOT count as overlapping A. B and C DO overlap (10 < 15).
    #    Expected pairs: A-B yes, A-C no (touch only), B-C yes -> 2 pairs.
    #    Concurrency: at t=5, {0,1} active = 2. At t=10, A's END is processed
    #    BEFORE C's START (tie-break), so streams active momentarily = {1,2} = 2.
    #    Max concurrency across the whole trace = 2, never 3.
    spans = [KernelSpan(0, 0, 10), KernelSpan(1, 5, 15), KernelSpan(2, 10, 20)]
    _assert_stats("touching-boundary", compute_overlaps(spans), pairs=2, max_streams=2)

    # 5. Four streams, identical windows -- the fanout4x4 shape: C(4,2)=6 cross-
    #    stream pairs, max concurrency 4, concurrency factor exactly 4.0x.
    spans = [KernelSpan(s, 100, 200) for s in range(4)]
    _assert_stats("four-way-full-overlap", compute_overlaps(spans),
                 pairs=6, max_streams=4, factor=4.0)

    # 6. The RESULTS.md section 5c worked example, reconstructed EXACTLY from
    #    the four (start, end) pairs printed in that table, to confirm the
    #    concurrency-factor arithmetic against a number already in the
    #    document rather than only against arithmetic I invented myself:
    #      stream 14: 799727180 -> 800191268  (464.1 us)
    #      stream 15: 799751052 -> 800239683  (488.6 us)
    #      stream 16: 799768203 -> 800281858  (513.7 us)
    #      stream 17: 799792715 -> 800296226  (503.5 us)
    #    RESULTS.md states: sum of durations 1,969,885 ns, wall span 569,046 ns,
    #    factor 3.46x. Recomputed here independently as a cross-check.
    spans = [
        KernelSpan(14, 799_727_180, 800_191_268),
        KernelSpan(15, 799_751_052, 800_239_683),
        KernelSpan(16, 799_768_203, 800_281_858),
        KernelSpan(17, 799_792_715, 800_296_226),
    ]
    stats = compute_overlaps(spans)
    total_dur = sum(s.end_ns - s.start_ns for s in spans)
    wall = max(s.end_ns for s in spans) - min(s.start_ns for s in spans)
    assert total_dur == 1_969_885, f"duration sum mismatch: {total_dur}"
    assert wall == 569_046, f"wall span mismatch: {wall}"
    _assert_stats("results-md-5c-worked-example", stats, pairs=6, max_streams=4,
                 factor=1_969_885 / 569_046, tol=1e-3)

    # 7. parse_gpu_trace_csv must skip past nsys's own informational preamble
    #    to find the real header -- the exact bytes that broke on Explorer
    #    V100 (2026-09-29): a leading blank line, then two NOTICE lines, then
    #    a blank line, then a "Processing [...] with [...]..." line, THEN the
    #    real header. Reproduced verbatim (shortened to 2 data rows) rather
    #    than a made-up preamble, so this test would have caught the actual
    #    bug instead of a hypothetical one.
    preamble_csv = (
        "\n"
        "NOTICE: Existing SQLite export found: reports/x.sqlite\n"
        "        It is assumed file was previously exported from: reports/x.nsys-rep\n"
        "        Consider using --force-export=true if needed.\n"
        "\n"
        "Processing [reports/x.sqlite] with [cuda_gpu_trace.py]... \n"
        "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,"
        "StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,"
        "Device,Ctx,GreenCtx,Strm,Name\n"
        "100,50,1,,,,,,,,,,,,,,GPU0,1,,0,kernel_a\n"
        "200,50,2,,,,,,,,,,,,,,GPU0,1,,1,kernel_b\n"
    )
    parsed = parse_gpu_trace_csv(preamble_csv)
    ok = (len(parsed) == 2
          and parsed[0] == KernelSpan(stream=0, start_ns=100, end_ns=150, name="kernel_a")
          and parsed[1] == KernelSpan(stream=1, start_ns=200, end_ns=250, name="kernel_b"))
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  nsys-preamble-header-skip: parsed={parsed}")
    if not ok:
        raise SystemExit(f"self-test FAILED: nsys-preamble-header-skip, got {parsed}")

    print("all self-tests PASSED")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Cross-stream kernel-overlap statistics from an nsys GPU "
                    "trace. See the comment block at the top of this file for "
                    "the honesty note on what is and isn't verified.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("nsys_rep", nargs="?", help=".nsys-rep file (requires `nsys` on PATH)")
    src.add_argument("--csv", help="already-exported cuda_gpu_trace CSV")
    p.add_argument("--self-test", action="store_true",
                   help="run the built-in synthetic tests and exit (no nsys/GPU needed)")
    p.add_argument("--json", help="also write the summary as JSON to this path "
                                 "(consumed by Phase 5 stage 5f's regen_results.sh)")
    args = p.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    # RuntimeError/ValueError from here down are the ANTICIPATED failure modes
    # this file's honesty note describes (no nsys binary, wrong column names) --
    # printed as a clean fatal message, not a Python traceback, matching this
    # project's convention elsewhere (e.g. bench_common.hpp's abort()) for a
    # failure that is actionable rather than a genuine bug in this script.
    try:
        if args.csv:
            with open(args.csv, encoding="utf-8") as f:
                text = f.read()
        elif args.nsys_rep:
            text = extract_csv_via_nsys(args.nsys_rep)
        else:
            p.error("give a .nsys-rep path, --csv=<file>, or --self-test")
            return 2
        spans = parse_gpu_trace_csv(text)
    except (RuntimeError, ValueError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    stats = compute_overlaps(spans)
    print(stats.summary())
    if stats.example_window:
        print("\nexample max-concurrency window:")
        for s in stats.example_window:
            print(f"  stream {s.stream:>3}  {s.start_ns:>14,} -> {s.end_ns:>14,}"
                 f"  ({(s.end_ns - s.start_ns) / 1000:.1f} us)  {s.name}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "total_spans": stats.total_spans,
                "overlapping_pairs": stats.overlapping_pairs,
                "max_concurrent_streams": stats.max_concurrent_streams,
                "local_concurrency_factor": stats.local_concurrency_factor,
            }, f, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
