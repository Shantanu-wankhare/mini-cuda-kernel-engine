#!/usr/bin/env python3
# =============================================================================
#  tools/gen_reference.py
#
#  WHAT: Generates tests/data/reference_vectors.txt -- an INDEPENDENT oracle for
#        every CPU reference in tests/reference.hpp.
#
#  WHY .py: it is a test-data generator, not part of the runtime. CLAUDE.md is
#  explicit that Python is for "test harnesses and plotting only, never part of
#  the runtime", so this script is run BY HAND and its output is COMMITTED. It is
#  not a build step, and the C++ side has no Python dependency at any point.
#
#  ---------------------------------------------------------------------------
#  THE PROBLEM THIS SOLVES, WHICH IS NOT "TEST THE KERNELS"
#
#  tests/reference.hpp is the oracle every CUDA kernel in Phase 3 is judged
#  against. So what judges the oracle? Today: nothing. If reference_gemm and the
#  GEMM kernel share the same misunderstanding -- say, both treat B as
#  column-major -- they agree perfectly and every validation passes. The bug is
#  invisible precisely because the two things that could disagree don't.
#
#  Breaking that requires an oracle derived from a DIFFERENT source. Hence this
#  file: the operations re-derived from their mathematical definitions, in a
#  different language, by a different person-hour.
#
#  ---------------------------------------------------------------------------
#  WHY PURE PYTHON RATHER THAN NUMPY (a deliberate deviation from the plan)
#
#  The plan called this "the NumPy cross-check". Pure Python + math.fsum turned
#  out to be strictly better here, for two reasons:
#
#    1. INDEPENDENCE. numpy's `@` dispatches to a BLAS, which is the same family
#       of code cuBLAS belongs to and which reassociates summation freely. An
#       oracle should not share an implementation lineage with the thing it
#       judges. math.fsum is exactly-rounded summation with a published
#       algorithm (Shewchuk) -- a genuinely different derivation.
#    2. NO DEPENDENCY. The repo's tests already advertise "zero external test
#       dependencies, works on a laptop with no network". Requiring numpy to
#       regenerate the vectors would quietly break that.
#
#  math.erf / math.tanh / math.exp are libm, the same functions reference.hpp
#  calls, so the transcendental cases test the FORMULA (is the GELU constant
#  0.044715? is sqrt(2/pi) right?) rather than libm itself. That is the bug class
#  worth catching; a libm disagreement would be a few ULP and is what tolerances
#  are for.
#
#  ---------------------------------------------------------------------------
#  THE MOVE THAT MAKES THIS A TEST RATHER THAN A TOLERANCE ARGUMENT
#
#  For the structural cases (GEMM shapes, the reduction), inputs are drawn from
#  MULTIPLES OF 0.25 in [-4, 4]. Every such value is exact in f32; every product
#  of two is a multiple of 0.0625 with magnitude <= 16; and a sum of up to 257 of
#  them is at most 4112, i.e. 65792 units of 0.0625 -- well under 2^24. So the
#  entire computation is EXACT in both f32 and f64, and Python, reference.hpp and
#  the GPU must agree BIT FOR BIT. Any difference is a bug, full stop, with no
#  rounding argument available to explain it away.
#
#  Cases involving exp/erf/tanh cannot be exact, so those carry a stated
#  tolerance instead and are marked `approx`.
#
#  ---------------------------------------------------------------------------
#  FILE FORMAT: 8-hex-digit IEEE-754 bit patterns, one token per float.
#
#  Not decimal. Hex bit patterns round-trip exactly (no parse/print rounding to
#  argue about), diff cleanly in git, and parse with strtoul -- so the C++ reader
#  needs no float parsing and no locale handling.
#
#  Inputs are written into the file EXPLICITLY rather than regenerated on the C++
#  side from fill_random. If both sides re-derived inputs from a shared
#  generator, a misunderstanding of that generator would be shared too -- which
#  is the exact failure mode this file exists to break. Independent oracle means
#  independent everything.
#
#  ---------------------------------------------------------------------------
#  USAGE
#      python3 tools/gen_reference.py > tests/data/reference_vectors.txt
#  Then run the host test suite; tests/test_host_core.cpp reads the file and
#  compares against tests/reference.hpp.
# =============================================================================

import math
import struct
import sys

# -----------------------------------------------------------------------------
# f32 plumbing. Python floats are f64, so every value that is supposed to be an
# f32 must be round-tripped through struct explicitly -- otherwise the oracle
# would silently carry more precision than the thing it judges, and "expected"
# values would be unreachable by any f32 implementation.
# -----------------------------------------------------------------------------
def f32(x):
    """Round a Python float to the nearest f32 and return it as a Python float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def hexf(x):
    """f32 -> 8 hex digits of its IEEE-754 bit pattern."""
    return "%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


# -----------------------------------------------------------------------------
# Deterministic input generation.
#
# A tiny explicit LCG rather than random.Random: the values must be reproducible
# from this file alone, forever, without depending on a Python version's
# generator internals. It is 3 lines and it removes a whole class of "why did the
# vectors change" questions.
# -----------------------------------------------------------------------------
class Rng:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next_u32(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 33) & 0xFFFFFFFF

    def quarter(self, lo=-4.0, hi=4.0):
        """A multiple of 0.25 in [lo, hi] -- EXACTLY representable in f32.
        This is what makes the structural cases bit-exact rather than approximate."""
        steps = int((hi - lo) / 0.25) + 1
        return lo + 0.25 * (self.next_u32() % steps)

    def unit(self):
        """An arbitrary f32 in [-1, 1); used only for `approx` cases."""
        return f32(self.next_u32() / 2147483648.0 - 1.0)


# -----------------------------------------------------------------------------
# The oracle: each operation re-derived from its definition.
#
# Deliberately written in the most literal way possible. This code is not trying
# to be fast or clever -- it is trying to be OBVIOUSLY correct to a reader
# comparing it against the mathematical definition, because that is the only
# property that makes it worth more than the code it checks.
# -----------------------------------------------------------------------------
def gemm(a, b, c, m, n, k, alpha, beta):
    """C = alpha * A[m,k] @ B[k,n] + beta * C, all ROW-MAJOR.

    Row-major is stated in every index expression below rather than assumed:
    a[i*k + p] walks a row of A, b[p*n + j] walks down a column of B. If
    reference.hpp ever disagrees about which is which, these vectors catch it --
    and a row/column-major mix-up is the single most likely GEMM bug in the
    project, because cuBLAS is column-major and needs an operand swap."""
    out = []
    for i in range(m):
        for j in range(n):
            # fsum, not sum(): exactly-rounded, so the oracle's own summation
            # error is zero and any disagreement belongs to the thing under test.
            acc = math.fsum(a[i * k + p] * b[p * n + j] for p in range(k))
            out.append(f32(alpha * acc + beta * c[i * n + j]))
    return out


def bias_act(x, bias, rows, cols, act):
    """y[r][c] = act(x[r][c] + bias[c])"""
    out = []
    for r in range(rows):
        for c in range(cols):
            v = x[r * cols + c] + bias[c]
            if act == "none":
                y = v
            elif act == "relu":
                y = v if v > 0.0 else 0.0
            elif act == "gelu_erf":
                # The exact definition: 0.5x(1 + erf(x/sqrt(2))).
                y = 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
            elif act == "gelu_tanh":
                # Hendrycks & Gimpel. The two constants here are the whole point
                # of this case: sqrt(2/pi) and 0.044715 are exactly the kind of
                # magic number that gets mistyped and never noticed, because a
                # wrong one still produces a plausible sigmoid-ish curve.
                y = 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
                                               (v + 0.044715 * v ** 3)))
            else:
                raise ValueError(act)
            out.append(f32(y))
    return out


def row_reduce(x, rows, cols, kind):
    out = []
    for r in range(rows):
        row = x[r * cols:(r + 1) * cols]
        if kind == "max":
            out.append(f32(max(row)))
        elif kind == "sum":
            out.append(f32(math.fsum(row)))
        elif kind == "mean":
            out.append(f32(math.fsum(row) / cols))
        else:
            raise ValueError(kind)
    return out


def row_softmax(x, rows, cols):
    out = []
    for r in range(rows):
        row = x[r * cols:(r + 1) * cols]
        m = max(row)                       # the max subtraction is mandatory,
        e = [math.exp(v - m) for v in row]  # not an optimisation -- see gemm.cu
        s = math.fsum(e)
        out.extend(f32(v / s) for v in e)
    return out


# -----------------------------------------------------------------------------
# Emission
# -----------------------------------------------------------------------------
def emit_case(name, kind, ints, floats, arrays, mode):
    print("case %s" % name)
    print("  kind %s" % kind)
    print("  mode %s" % mode)
    for key, val in ints:
        print("  int %s %d" % (key, val))
    for key, val in floats:
        print("  float %s %s" % (key, hexf(val)))
    for key, vals in arrays:
        print("  array %s %d" % (key, len(vals)))
        # 8 values per line: short enough to read in a diff, long enough that the
        # file stays under a few hundred lines.
        for i in range(0, len(vals), 8):
            print("    " + " ".join(hexf(v) for v in vals[i:i + 8]))
    print("end")
    print("")


def main():
    print("# GENERATED BY tools/gen_reference.py -- DO NOT EDIT BY HAND.")
    print("# Regenerate with: python3 tools/gen_reference.py > tests/data/reference_vectors.txt")
    print("#")
    print("# An INDEPENDENT oracle for tests/reference.hpp: the same operations")
    print("# re-derived in Python from their definitions, so that a shared")
    print("# misunderstanding between the C++ reference and the CUDA kernels")
    print("# cannot validate clean. Every float is an 8-hex-digit IEEE-754 f32.")
    print("#")
    print("# mode exact  : inputs are multiples of 0.25, so every product and sum")
    print("#               is exact in f32 -- agreement must be BIT-FOR-BIT.")
    print("# mode approx : involves exp/erf/tanh; compare with a tolerance.")
    print("")

    rng = Rng(0x9E3779B97F4A7C15)

    # --- GEMM. The awkward-shape matrix, matching bench/gemm_bench.cpp.
    #     All EXACT: this is where a row/column-major error would hide, and an
    #     exact test leaves no room to argue that a mismatch is "just rounding".
    for (m, n, k) in [(1, 1, 1), (5, 7, 3), (2, 3, 4), (9, 5, 17), (13, 13, 13)]:
        a = [rng.quarter() for _ in range(m * k)]
        b = [rng.quarter() for _ in range(k * n)]
        c = [rng.quarter() for _ in range(m * n)]
        for (alpha, beta, tag) in [(1.0, 0.0, "ab"), (0.5, 2.0, "beta")]:
            emit_case(
                "gemm_%s_%dx%dx%d" % (tag, m, n, k), "gemm",
                [("m", m), ("n", n), ("k", k)],
                [("alpha", alpha), ("beta", beta)],
                [("a", a), ("b", b), ("c", c),
                 ("expect", gemm(a, b, c, m, n, k, alpha, beta))],
                "exact")

    # --- NON-SQUARE, deliberately. With m == n a transposed result passes for a
    #     wide class of inputs, so a square-only test would not detect the
    #     row/column-major swap it exists to detect.
    m, n, k = 3, 7, 5
    a = [rng.quarter() for _ in range(m * k)]
    b = [rng.quarter() for _ in range(k * n)]
    c = [0.0] * (m * n)
    emit_case("gemm_nonsquare_3x7x5", "gemm",
              [("m", m), ("n", n), ("k", k)],
              [("alpha", 1.0), ("beta", 0.0)],
              [("a", a), ("b", b), ("c", c),
               ("expect", gemm(a, b, c, m, n, k, 1.0, 0.0))],
              "exact")

    # --- bias + activation. relu and none are exact; the GELUs are not, and
    #     their whole purpose is to pin the two magic constants.
    rows, cols = 4, 6
    xq = [rng.quarter(-4.0, 4.0) for _ in range(rows * cols)]
    bq = [rng.quarter(-2.0, 2.0) for _ in range(cols)]
    for act, mode in [("none", "exact"), ("relu", "exact"),
                      ("gelu_erf", "approx"), ("gelu_tanh", "approx")]:
        emit_case("bias_act_%s" % act, "bias_act",
                  [("rows", rows), ("cols", cols), ("act", ["none", "relu", "gelu_erf",
                                                            "gelu_tanh"].index(act))],
                  [],
                  [("x", xq), ("bias", bq),
                   ("expect", bias_act(xq, bq, rows, cols, act))],
                  mode)

    # --- row reduce. Exact by construction, including a deliberately
    #     all-negative row: the max identity must be -inf, not 0, or an
    #     all-negative row reduces to a value that is not in the row at all.
    rows, cols = 3, 9
    xr = [rng.quarter(-4.0, 4.0) for _ in range(rows * cols)]
    for c in range(cols):
        xr[c] = -abs(xr[c]) - 0.25          # row 0 entirely negative
    for kind, idx in [("sum", 0), ("max", 1), ("mean", 2)]:
        emit_case("row_reduce_%s" % kind, "row_reduce",
                  [("rows", rows), ("cols", cols), ("kind", idx)], [],
                  [("x", xr), ("expect", row_reduce(xr, rows, cols, kind))],
                  "exact")

    # --- softmax, including a row with a large value: without the max
    #     subtraction exp(90) overflows to inf and the row becomes NaN.
    rows, cols = 3, 8
    xs = [rng.unit() * 4.0 for _ in range(rows * cols)]
    xs[cols] = 90.0                          # row 1 has a value that would overflow
    xs = [f32(v) for v in xs]
    emit_case("row_softmax", "row_softmax",
              [("rows", rows), ("cols", cols)], [],
              [("x", xs), ("expect", row_softmax(xs, rows, cols))],
              "approx")


if __name__ == "__main__":
    sys.exit(main())
