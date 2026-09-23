#!/usr/bin/env python3
# =============================================================================
#  tools/plot_roofline.py
#
#  WHAT: reads one or more of Profiler::write_csv's frozen 13-column CSVs and
#  places every kernel on one roofline plot -- Phase 5's second exit criterion
#  ("a roofline plot with all kernels on it", docs/ROADMAP.md).
#
#  WHY .py: plotting only, never part of the runtime (CLAUDE.md section 5).
#  WHY matplotlib, a new dependency: there is no stdlib way to draw a log-log
#  scatter plot with reference lines. tools/gen_reference.py's stdlib-only
#  convention is for numeric correctness code that must run everywhere
#  (including inside scripts/typecheck_cuda.sh's environment); a plotting
#  script has a different job and a different lifetime, the same distinction
#  Phase 5 stage 5a already drew for Google Benchmark on the C++ side. numpy
#  is NOT used -- every computation here is a handful of scalar formulas
#  (Roofline's own formulas, in fact -- see below), and matplotlib accepts
#  plain Python lists.
#
#  ---------------------------------------------------------------------------
#  THE ONE DESIGN DECISION WORTH EXPLAINING BEFORE THE CODE: where do the
#  roofline's own two numbers (peak_gb_s, peak_tflops) come from?
#
#  NOT inferred from the CSV. A CSV's `attainable_tflops`/`bound` columns were
#  computed against whatever denominators PRODUCED that file (bench_common.hpp
#  bakes in Colab T4's numbers unless overridden) -- trusting them here would
#  silently draw "the roofline for whatever machine happened to make this
#  file", which is exactly the kind of hidden assumption bench_common.hpp's own
#  header comment was written to prevent ("a Roofline has two fields... With
#  peak_tflops == 0 ... two confidently wrong answers, no error, no warning").
#  The same discipline applies here: state the denominator, don't guess it.
#
#  So: `--peak-gb-s=` and `--peak-tflops=` (or `MCKE_PEAK_GB_S`/
#  `MCKE_PEAK_TFLOPS`, matching bench_common.hpp's own env vars for the same
#  two numbers) are the primary interface, and both are required together --
#  mirroring `benchcfg::make_roofline`'s exact contract. `--preset=t4` /
#  `--preset=v100` are named shortcuts for the two machines this project has
#  already measured (RESULTS.md section 0), not a directory-name guess. A
#  directory name records WHERE a run happened (scripts/machine_tag.sh's
#  <env>-<gpu> tags, reconciled in stage 5f), and detection is deliberately
#  best-effort; a denominator is a MEASUREMENT, and deriving one from a path
#  would turn a harmless misnamed directory into a silently wrong roofline.
#  Give none of the above and this script refuses to plot, loudly, the same
#  way `benchcfg::make_roofline` aborts on a zero denominator.
#
#  Every plotted point's (x, y) still comes straight from the CSV
#  (arithmetic_intensity, achieved_tflops) -- only the ROOFLINE LINES and the
#  memory/compute-bound marker styling are computed fresh from the chosen
#  denominators, using the exact formulas in profiler.hpp's `Roofline` struct,
#  reproduced here rather than reimplemented differently by accident:
#      ridge_point_ai      = (peak_tflops * 1e12) / (peak_gb_s * 1e9)
#      attainable_tflops(ai) = min(ai * peak_gb_s * 1e9 / 1e12, peak_tflops)
#      memory_bound(ai)    = ai < ridge_point_ai
#
#  USAGE:
#    python3 tools/plot_roofline.py --preset=t4 reports/colab-t4/<run-id>/*.csv \
#        -o reports/colab-t4/<run-id>/roofline.svg
#
#    python3 tools/plot_roofline.py --peak-gb-s=636.3 --peak-tflops=15.601 \
#        reports/explorer-v100/<run-id>/*.csv -o reports/explorer-v100/<run-id>/roofline.svg
#
#    python3 tools/plot_roofline.py --self-test    # no CSV, no matplotlib call
# =============================================================================
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Named presets -- RESULTS.md section 0's measured denominators, transcribed
# once, here, rather than left to be retyped (and mistyped) on every
# invocation. Cite the source line so a future correction to RESULTS.md's own
# numbers doesn't leave this file silently stale.
# -----------------------------------------------------------------------------
_PRESETS = {
    # RESULTS.md section 0: Colab Tesla T4. Identical to
    # bench/bench_common.hpp's kT4MeasuredGbS / kT4MeasuredTflops.
    "t4": (235.4, 8.130),
    # RESULTS.md section 0 and section 3d's intro: Explorer Tesla V100-SXM2-32GB.
    "v100": (636.3, 15.601),
}


@dataclass(frozen=True)
class Roofline:
    peak_gb_s: float
    peak_tflops: float

    def ridge_point_ai(self) -> float:
        return (self.peak_tflops * 1e12) / (self.peak_gb_s * 1e9)

    def attainable_tflops(self, ai: float) -> float:
        bw_bound = ai * self.peak_gb_s * 1e9 / 1e12
        return min(bw_bound, self.peak_tflops)

    def memory_bound(self, ai: float) -> bool:
        return ai < self.ridge_point_ai()


@dataclass
class Point:
    kernel: str
    variant: str
    ai: float
    achieved_tflops: float
    source_csv: str


# -----------------------------------------------------------------------------
# Label placement. A first real render (Phase 5 stage 5e, against the
# committed GEMM CSV) showed labels for closely-spaced points overlapping into
# unreadable mush -- four of the nine GEMM variants land within a few percent
# of each other in TFLOP/s once the ladder nears cuBLAS, and a naive fixed
# (5, 2) point offset for every label puts them on top of one another. This
# WILL get worse once stage 5g adds bias_act/reduce/softmax CSVs to the same
# plot, so fixed now rather than left to be discovered illegible later.
#
# No new dependency (no adjustText/scipy): a simple, deterministic clustering
# in LOG-SPACE (matching the plot's own log-log axes, so "close" means what it
# visually means here) followed by a vertical stack of increasing point-space
# offsets. This is intentionally testable without matplotlib -- see
# _test_label_offsets in the self-test.
# -----------------------------------------------------------------------------
def compute_label_offsets(points: list[Point], cluster_tol: float = 0.08
                         ) -> list[tuple[float, float, bool]]:
    """Returns one (dx, dy, needs_leader) per point in `points`, same order.
    (dx, dy) is a text offset in POINTS (matplotlib text-offset units).
    Points whose (log10 ai, log10 tflops) are within `cluster_tol` of each
    other are stacked vertically instead of all using the same offset --
    `needs_leader` is True for every stacked member below the first, because a
    render with 4+ members in one cluster showed labels drifting far enough
    from their actual marker (which may sit within a couple of PIXELS of its
    neighbours) to look disconnected from any point at all. A leader line
    (drawn by the caller when this is True) is what keeps a heavily stacked
    label attributable.
    """
    n = len(points)
    log_coords = []
    for p in points:
        lx = math.log10(p.ai) if p.ai > 0 else 0.0
        ly = math.log10(p.achieved_tflops) if p.achieved_tflops > 0 else 0.0
        log_coords.append((lx, ly))

    # Union-find style clustering: two points join the same cluster if within
    # tol of ANY existing cluster member (chained), not just pairwise to the
    # first -- cheap for the handful of points this project ever plots (dozens,
    # not thousands), so an O(n^2) pass is fine and simple beats fast here.
    cluster_of = list(range(n))

    def find(i: int) -> int:
        while cluster_of[i] != i:
            i = cluster_of[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            cluster_of[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if (abs(log_coords[i][0] - log_coords[j][0]) < cluster_tol
                    and abs(log_coords[i][1] - log_coords[j][1]) < cluster_tol):
                union(i, j)

    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)

    offsets: list[tuple[float, float, bool] | None] = [None] * n
    for group in members.values():
        # Stack top-to-bottom in descending y (matches how a reader's eye
        # scans a cluster from its highest point down), each 11pt apart --
        # enough to clear a default-size annotation without a real text-extent
        # measurement, which would need a live matplotlib figure to compute.
        group_sorted = sorted(group, key=lambda i: -log_coords[i][1])
        for rank, idx in enumerate(group_sorted):
            offsets[idx] = (6.0, 4.0 - 11.0 * rank, rank > 0)
    return offsets  # type: ignore[return-value]


_REQUIRED_COLUMNS = [
    "kernel", "variant", "ideal_flops", "ideal_bytes",
    "achieved_tflops", "arithmetic_intensity",
]


def load_points(csv_paths: list[str]) -> list[Point]:
    """Parse Profiler::write_csv's frozen schema. Cross-checks the stored
    `arithmetic_intensity` column against a fresh ideal_flops/ideal_bytes
    recomputation, and refuses (loudly) rather than silently plot a corrupted
    or hand-edited file -- the same "don't trust a number you can recompute
    without also checking it" discipline as everywhere else in this project.
    """
    points: list[Point] = []
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: empty CSV, no header row")
            missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"{path}: missing column(s) {missing} from the frozen "
                    f"13-column schema (src/core/profiler.cpp write_csv). "
                    f"Actual header: {list(reader.fieldnames)}"
                )
            for row_num, row in enumerate(reader, start=2):  # header is line 1
                flops = float(row["ideal_flops"])
                bytes_ = float(row["ideal_bytes"])
                recomputed_ai = flops / bytes_ if bytes_ else 0.0
                stored_ai = float(row["arithmetic_intensity"])
                if bytes_ and abs(recomputed_ai - stored_ai) > 1e-6 * max(1.0, stored_ai):
                    raise ValueError(
                        f"{path}:{row_num}: arithmetic_intensity column ({stored_ai}) "
                        f"disagrees with ideal_flops/ideal_bytes ({recomputed_ai}) for "
                        f"{row['kernel']}/{row['variant']} -- CSV looks hand-edited or "
                        f"corrupted. Refusing to plot rather than draw a point that "
                        f"doesn't match its own inputs."
                    )
                points.append(Point(
                    kernel=row["kernel"], variant=row["variant"],
                    ai=stored_ai, achieved_tflops=float(row["achieved_tflops"]),
                    source_csv=os.path.basename(path),
                ))
    return points


def _warn_if_offline_from_roofline(points: list[Point], rl: Roofline) -> None:
    """Non-fatal: if a point's achieved_tflops sits ABOVE what `rl` says is
    attainable at its own arithmetic intensity, either the point exceeded its
    own machine's roofline (a real bug worth knowing about) or -- far more
    likely -- these denominators are for a DIFFERENT machine than the one that
    produced this CSV (e.g. plotting a Colab T4 CSV against V100 peaks by
    mistake). Warn once per offending row rather than abort: a deliberate
    cross-machine "what would this look like on a bigger GPU" comparison is a
    legitimate use of this script, just not a silent one.
    """
    for p in points:
        attainable = rl.attainable_tflops(p.ai)
        if attainable > 0 and p.achieved_tflops > attainable * 1.02:  # 2% slack for fp noise
            print(f"WARNING: {p.kernel}/{p.variant} ({p.source_csv}) achieved "
                 f"{p.achieved_tflops:.3f} TFLOP/s, ABOVE the {attainable:.3f} TFLOP/s "
                 f"this roofline says is attainable at AI={p.ai:.1f}. Likely cause: "
                 f"--peak-gb-s/--peak-tflops don't match the machine that produced "
                 f"this CSV. Continuing anyway.", file=sys.stderr)


def render(points: list[Point], rl: Roofline, out_path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed; headless-safe (Colab, Explorer, CI)
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required to render a plot (not to parse CSVs or run "
            "--self-test). Install with `pip install matplotlib`, or on "
            "Explorer `module load anaconda3` (module_list.md)."
        ) from e

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- roofline lines, log-log, drawn from the CHOSEN denominators only ---
    ridge_ai = rl.ridge_point_ai()
    ai_values = [p.ai for p in points if p.ai > 0]
    x_lo = min(ai_values + [ridge_ai]) / 4 if ai_values else ridge_ai / 4
    x_hi = max(ai_values + [ridge_ai]) * 4 if ai_values else ridge_ai * 4

    # Memory-bound diagonal: y = ai * peak_gb_s / 1000 (TFLOP/s), up to the ridge.
    mem_x = [x_lo, ridge_ai]
    mem_y = [rl.attainable_tflops(x) for x in mem_x]
    ax.plot(mem_x, mem_y, "k--", linewidth=1, label=f"memory roof ({rl.peak_gb_s:.1f} GB/s)")
    # Compute-bound horizontal: y = peak_tflops, from the ridge onward.
    ax.plot([ridge_ai, x_hi], [rl.peak_tflops, rl.peak_tflops], "k-", linewidth=1,
           label=f"compute roof ({rl.peak_tflops:.3f} TFLOP/s)")
    ax.axvline(ridge_ai, color="gray", linestyle=":", linewidth=0.8)
    # Anchored near the BOTTOM of the plot (y in AXES fraction, not data) via a
    # blended transform, deliberately NOT at (ridge_ai, peak_tflops) -- the top
    # of the plot is exactly where the "compute roof" legend box sits (upper
    # left), and a first real render put this label right on top of it,
    # because for a GEMM-only plot the ridge point (~34.5 FLOP/byte) happens
    # to fall inside the legend's horizontal span. The bottom of the axes is
    # never contested by either legend (top-left, bottom-right).
    import matplotlib.transforms as mtransforms
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(ridge_ai, 0.03, f"ridge {ridge_ai:.1f} FLOP/byte", transform=blended,
           fontsize=8, color="gray", ha="left", va="bottom",
           rotation=90 if (x_hi / x_lo) > 100 else 0)

    # --- one colour per kernel, one marker shape per bound classification ---
    kernels = sorted({p.kernel for p in points})
    cmap = plt.get_cmap("tab10")
    colour = {k: cmap(i % 10) for i, k in enumerate(kernels)}

    label_offsets = compute_label_offsets(points)
    for p, (dx, dy, needs_leader) in zip(points, label_offsets):
        marker = "o" if rl.memory_bound(p.ai) else "s"  # circle=memory-bound, square=compute-bound
        ax.scatter(p.ai, p.achieved_tflops, color=colour[p.kernel], marker=marker,
                  s=40, zorder=3, edgecolors="black", linewidths=0.4)
        # A thin leader line for heavily-stacked labels only (needs_leader) --
        # see compute_label_offsets's docstring for why: without one, a label
        # ranked 3rd or 4th in a tight cluster drifts far enough from its own
        # marker (which may be a couple of PIXELS from its neighbours) to read
        # as unattached to any point at all.
        arrowprops = dict(arrowstyle="-", color="gray", lw=0.5, shrinkA=0, shrinkB=3) \
            if needs_leader else None
        ax.annotate(p.variant, (p.ai, p.achieved_tflops), fontsize=6,
                   textcoords="offset points", xytext=(dx, dy), arrowprops=arrowprops)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)

    # Kernel legend (colour) separate from the roofline-line legend (already
    # added via label= above) -- otherwise matplotlib interleaves them into one
    # confusing list. `o`/`s` markers explained once in a text note instead of
    # a third legend, since it's one fact, not a list.
    from matplotlib.lines import Line2D
    kernel_handles = [Line2D([0], [0], marker="o", linestyle="", color=colour[k],
                             markeredgecolor="black", markeredgewidth=0.4, label=k)
                      for k in kernels]
    roof_handles, roof_labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles=roof_handles, labels=roof_labels, loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=kernel_handles, loc="lower right", fontsize=8, title="kernel")
    # Bottom-LEFT, deliberately: the roofline legend already occupies the top
    # left and the kernel legend the bottom right (both `ax.legend` calls
    # above), so this is the one corner neither claims. A first real render
    # had this at bottom-right, directly under the kernel legend box.
    ax.text(0.01, 0.02, "○ memory-bound   □ compute-bound", transform=ax.transAxes,
           ha="left", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path} ({len(points)} points from {len(set(p.source_csv for p in points))} CSV file(s))")


# -----------------------------------------------------------------------------
# Self-test: the parts that don't need matplotlib or a real CSV file at all --
# the Roofline formulas (cross-checked against profiler.hpp's own worked
# numbers, RESULTS.md section 3d) and the CSV-corruption guard.
# -----------------------------------------------------------------------------
def run_self_test() -> None:
    print("plot_roofline.py self-test (no matplotlib call, no GPU required)")

    # 1. Roofline formulas against RESULTS.md section 3 (T4): ridge point
    #    34.5 FLOP/byte is documented directly; reproduce it from the same
    #    two measured denominators bench_common.hpp hard-codes.
    rl = Roofline(peak_gb_s=235.4, peak_tflops=8.130)
    ridge = rl.ridge_point_ai()
    assert abs(ridge - 34.5) < 0.05, f"T4 ridge point mismatch: {ridge}"
    print(f"  PASS  T4 ridge point = {ridge:.2f} FLOP/byte (RESULTS.md says 34.5)")

    # 2. GEMM's AI (682.667, from the committed CSV) is deep compute-bound:
    #    attainable_tflops must equal peak_tflops exactly (the min() branch),
    #    and the committed CSV's own attainable_tflops column (8.13) must
    #    agree, since this file WAS generated with these exact denominators.
    ai_gemm = 682.667
    att = rl.attainable_tflops(ai_gemm)
    assert abs(att - 8.130) < 1e-6, f"expected compute-bound ceiling, got {att}"
    assert not rl.memory_bound(ai_gemm)
    print(f"  PASS  GEMM AI={ai_gemm} -> attainable {att:.3f} TFLOP/s (compute-bound, matches CSV's 8.13)")

    # 3. A memory-bound example: bias+GELU's AI ~= 1.0 (profiler.hpp's own
    #    header table lists it as "~1.0"), which must sit on the diagonal, not
    #    the ceiling, and be classified memory-bound.
    ai_bias = 1.0
    att_bias = rl.attainable_tflops(ai_bias)
    expected = ai_bias * rl.peak_gb_s * 1e9 / 1e12
    assert abs(att_bias - expected) < 1e-9
    assert rl.memory_bound(ai_bias)
    print(f"  PASS  bias+GELU AI={ai_bias} -> attainable {att_bias:.4f} TFLOP/s (memory-bound, on the diagonal)")

    # 4. The CSV corruption guard: a row whose stored arithmetic_intensity
    #    disagrees with ideal_flops/ideal_bytes must be REFUSED, not silently
    #    plotted with a wrong x-coordinate.
    import tempfile
    bad_csv = (
        "kernel,variant,median_ms,min_ms,iterations,ideal_flops,ideal_bytes,"
        "achieved_gb_s,achieved_tflops,arithmetic_intensity,attainable_tflops,"
        "efficiency_pct,bound\n"
        "gemm,naive,1.0,1.0,20,1000,10,1.0,1.0,999.0,8.13,10.0,compute\n"  # AI should be 100, says 999
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write(bad_csv)
        bad_path = tf.name
    try:
        load_points([bad_path])
        raise SystemExit("self-test FAILED: corrupted CSV was NOT rejected")
    except ValueError as e:
        assert "disagrees" in str(e)
        print(f"  PASS  corrupted CSV correctly refused: {str(e)[:60]}...")
    finally:
        os.unlink(bad_path)

    # 5. compute_label_offsets: two nearly-identical points must be assigned
    #    DISTINCT, vertically-stacked offsets (not the same offset, which is
    #    what produced the overlapping-label render this function replaced);
    #    a far-away third point must get its own default offset, unaffected
    #    by the other two's cluster.
    # close_b has the HIGHER tflops of the pair, so it ranks first (no leader
    # line needed); close_a, ranked second, must get one.
    close_a = Point("gemm", "a", ai=700.0, achieved_tflops=3.28, source_csv="x")
    close_b = Point("gemm", "b", ai=705.0, achieved_tflops=3.30, source_csv="x")  # same log-cluster as a
    far_c = Point("gemm", "c", ai=1.0, achieved_tflops=0.01, source_csv="x")      # nowhere near a/b
    offsets = compute_label_offsets([close_a, close_b, far_c])
    assert offsets[0] != offsets[1], f"clustered points got the SAME offset: {offsets[0]}"
    assert offsets[0][0] == offsets[1][0], "clustered points should share dx (stack vertically)"
    assert offsets[0][1] != offsets[1][1], "clustered points must differ in dy to avoid overlapping"
    assert offsets[0][2], "close_a (lower tflops -> rank 1) must get a leader line"
    assert not offsets[1][2], "close_b (higher tflops -> rank 0) needs no leader line"
    assert offsets[2] == (6.0, 4.0, False), f"the lone far point should get the plain default, got {offsets[2]}"
    print(f"  PASS  compute_label_offsets: clustered pair got distinct offsets "
         f"{offsets[0]} vs {offsets[1]} (leader line: {offsets[0][2]}/{offsets[1][2]}); "
         f"isolated point got the default {offsets[2]}")

    print("all self-tests PASSED")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Plot every kernel in one or more Profiler CSVs onto one roofline. "
                    "See the comment block at the top of this file for how the "
                    "roofline's own denominators are chosen -- they are never "
                    "inferred from the CSV.")
    p.add_argument("csv_files", nargs="*", help="one or more Profiler::write_csv output files")
    p.add_argument("--preset", choices=sorted(_PRESETS), help="named denominators (see top of file)")
    p.add_argument("--peak-gb-s", type=float, default=None)
    p.add_argument("--peak-tflops", type=float, default=None)
    p.add_argument("-o", "--out", default="roofline.svg", help="output path (.svg/.png/...)")
    p.add_argument("--title", default="MCKE roofline")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if not args.csv_files:
        p.error("give one or more CSV files, or --self-test")
        return 2

    # Denominator resolution: explicit flags > --preset > env vars > refuse.
    # Mirrors benchcfg::make_roofline's own precedence and its own refusal to
    # hand back a zero-or-missing denominator silently.
    peak_gb_s = args.peak_gb_s
    peak_tflops = args.peak_tflops
    if args.preset:
        preset_gb_s, preset_tflops = _PRESETS[args.preset]
        peak_gb_s = peak_gb_s if peak_gb_s is not None else preset_gb_s
        peak_tflops = peak_tflops if peak_tflops is not None else preset_tflops
    if peak_gb_s is None:
        env = os.environ.get("MCKE_PEAK_GB_S")
        peak_gb_s = float(env) if env else None
    if peak_tflops is None:
        env = os.environ.get("MCKE_PEAK_TFLOPS")
        peak_tflops = float(env) if env else None

    if not peak_gb_s or not peak_tflops or peak_gb_s <= 0 or peak_tflops <= 0:
        print(
            "FATAL: roofline denominators must both be > 0 "
            f"(peak_gb_s={peak_gb_s} peak_tflops={peak_tflops}).\n"
            "       Give --preset=t4 / --preset=v100, or both --peak-gb-s= and\n"
            "       --peak-tflops= explicitly, or MCKE_PEAK_GB_S/MCKE_PEAK_TFLOPS.\n"
            "       This is deliberately not inferred from the CSV -- see the\n"
            "       module comment at the top of this file for why (the exact\n"
            "       trap bench_common.hpp's own header describes).",
            file=sys.stderr,
        )
        return 1

    rl = Roofline(peak_gb_s=peak_gb_s, peak_tflops=peak_tflops)
    print(f"denominators  peak_gb_s={rl.peak_gb_s:.1f}  peak_tflops={rl.peak_tflops:.3f}  "
         f"ridge={rl.ridge_point_ai():.1f} FLOP/byte")

    try:
        points = load_points(args.csv_files)
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    if not points:
        print("FATAL: no data rows found in the given CSV file(s).", file=sys.stderr)
        return 1

    _warn_if_offline_from_roofline(points, rl)

    try:
        render(points, rl, args.out, args.title)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
