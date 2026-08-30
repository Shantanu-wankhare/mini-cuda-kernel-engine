// =============================================================================
//  tests/reference.hpp
//
//  WHAT: CPU reference implementations of every Phase-3 kernel, plus the
//        floating-point comparison used to judge a GPU result against them.
//
//  WHY THIS EXISTS: before Phase 3 the repo's only correctness check was
//  `tools/smoke_vector_add.cpp`'s exact `!=` on a single IEEE add. That is
//  perfectly defensible for `a + b` — one operation, bit-reproducible on host
//  and device — and it is completely unusable for anything in this phase. A GEMM
//  accumulates K products in a different ORDER than a host loop does, and
//  floating-point addition is not associative, so the two results differ in the
//  last bits by construction. Softmax runs `expf`, whose implementation is not
//  required to be bit-identical between libm and the device. Demanding equality
//  there would produce a test that fails on correct code, which is worse than no
//  test at all: it trains you to ignore it.
//
//  WHY .hpp IN tests/: header-only, host-only, no device code — includable from
//  a plain `clang++` build on a machine with no CUDA. `tests/test_access.hpp` is
//  the precedent for a shared test-only header living here. Note the bench
//  targets include it too (they need the same references to verify a kernel
//  before timing it), which is why CMake adds `-I tests` to them as well. If
//  this grows a third consumer it should be promoted to a top-level `testing/`.
//
//  ---------------------------------------------------------------------------
//  TWO DECISIONS THAT DETERMINE WHETHER A MISMATCH MEANS ANYTHING
//
//  1. THE REFERENCE ACCUMULATES IN `double`, NOT `float`.
//
//     The point of a reference is to be a more trustworthy answer than the thing
//     under test. If it accumulated in float it would carry its own rounding
//     error of the same magnitude as the kernel's, and a disagreement would tell
//     you only "these two are both approximate", not "the kernel is wrong". With
//     a double accumulator the reference's error is ~2^-53 against the kernel's
//     ~2^-24 — nine orders of magnitude smaller, so any observed difference is
//     attributable to the kernel.
//
//     The cost is that the comparison gets STRICTER, so the tolerance has to be
//     chosen deliberately rather than by feel — see below.
//
//  2. THE TOLERANCE IS MIXED ABSOLUTE + RELATIVE:
//         |got - want| <= abs_tol + rel_tol * |want|
//
//     A pure relative test divides by `want` and therefore explodes whenever the
//     expected value is near zero — and near-zero outputs are not an edge case
//     here, they are routine: ReLU produces exact zeros for half its inputs,
//     GELU produces tiny values for moderately negative x, and a softmax row has
//     entries that underflow toward zero. A pure absolute test is equally wrong
//     in the other direction: 1e-6 is a fine tolerance next to 1.0 and a
//     catastrophic one next to 1e8. The mixed form (the same one `numpy.allclose`
//     uses) is the standard answer and handles both ends.
//
//  ---------------------------------------------------------------------------
//  HOW TO CHOOSE rel_tol, RATHER THAN GUESSING
//
//  f32 has ~1.19e-7 machine epsilon. Summing N values in f32 accumulates error
//  that grows like sqrt(N)*eps for data of mixed sign (errors partially cancel,
//  a random walk) and like N*eps in the adversarial same-sign worst case. So the
//  tolerance should scale with the DEPTH of the accumulation, not be a constant:
//
//      one op (bias + activation)      : a few ulp        -> 1e-6
//      sum over 4096 (row reduce)      : sqrt(4096)*eps
//                                        = 64 * 1.19e-7   -> 1e-5
//      softmax over 4096               : as above, plus expf's own ~1 ulp -> 1e-5
//      GEMM with K=256 (validation K)  : sqrt(256)*eps
//                                        = 16 * 1.19e-7   -> 1e-5
//
//  Each kernel's bench states which tolerance it used and why. A tolerance
//  chosen because "it made the test pass" is not a tolerance, it is a decision
//  to stop looking.
// =============================================================================
#pragma once

#include <algorithm>   // std::max
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>      // numeric_limits<double>::infinity for the max identity
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mcke/kernels/kernels.hpp"

namespace mcke::testing {

// -----------------------------------------------------------------------------
// Deterministic test data.
//
// std::mt19937_64 rather than std::rand: the standard specifies the engine's
// recurrence bit-exactly, so the same seed produces the same stream under
// libc++ on macOS and libstdc++ on Colab. A correctness failure that reproduces
// on both machines is debuggable; one that does not is a ghost.
//
// Deliberately NOT std::uniform_real_distribution: engines are specified
// exactly, DISTRIBUTIONS ARE NOT — libc++ and libstdc++ legitimately consume
// different numbers of engine outputs and return different values. Using one
// would silently break cross-machine reproducibility while looking correct.
// -----------------------------------------------------------------------------
inline void fill_random(float* p, std::size_t n, std::uint64_t seed,
                        float lo = -1.0f, float hi = 1.0f) {
  std::mt19937_64 eng(seed);
  for (std::size_t i = 0; i < n; ++i) {
    // Take the top 24 bits (f32's mantissa width) and scale into [lo, hi].
    const double u = static_cast<double>(eng() >> 40) / static_cast<double>(1u << 24);
    p[i] = static_cast<float>(lo + u * (hi - lo));
  }
}

// -----------------------------------------------------------------------------
// Activations. Both GELU forms, because "which GELU" is a real question every
// kernel author has to answer and Phase 3a measures whether the choice is even
// visible on a bandwidth-bound kernel (prediction: it is not).
// -----------------------------------------------------------------------------
inline double gelu_erf(double x) {
  // The exact definition: 0.5x(1 + erf(x/sqrt(2))).
  return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));  // 1/sqrt(2)
}

inline double gelu_tanh(double x) {
  // Hendrycks & Gimpel's tanh approximation. Differs from the exact form in the
  // 3rd decimal place; cheaper on device (tanhf vs erff).
  const double k = 0.79788456080286535588;   // sqrt(2/pi)
  return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
}

inline double apply_activation(double v, kernels::Activation act) {
  switch (act) {
    case kernels::Activation::kNone:     return v;
    case kernels::Activation::kRelu:     return v > 0.0 ? v : 0.0;
    case kernels::Activation::kGeluErf:  return gelu_erf(v);
    case kernels::Activation::kGeluTanh: return gelu_tanh(v);
  }
  return v;
}

// -----------------------------------------------------------------------------
// References. All row-major, all accumulating in double.
// -----------------------------------------------------------------------------

// y[r][c] = act(x[r][c] + bias[c])
inline void reference_bias_act(const float* x, const float* bias, float* y,
                               std::int64_t rows, std::int64_t cols,
                               kernels::Activation act) {
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t c = 0; c < cols; ++c) {
      const double v = static_cast<double>(x[r * cols + c]) + static_cast<double>(bias[c]);
      y[r * cols + c] = static_cast<float>(apply_activation(v, act));
    }
}

// out[r] = reduce(x[r][:])
inline void reference_row_reduce(const float* x, float* out,
                                 std::int64_t rows, std::int64_t cols,
                                 kernels::ReduceKind kind) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    if (kind == kernels::ReduceKind::kMax) {
      // Identity for max must be -inf, not 0 — an all-negative row would
      // otherwise reduce to 0, which is not an element of the row at all.
      double m = -std::numeric_limits<double>::infinity();
      for (std::int64_t c = 0; c < cols; ++c) m = std::max(m, static_cast<double>(row[c]));
      out[r] = static_cast<float>(m);
    } else {
      double s = 0.0;
      for (std::int64_t c = 0; c < cols; ++c) s += static_cast<double>(row[c]);
      if (kind == kernels::ReduceKind::kMean) s /= static_cast<double>(cols);
      out[r] = static_cast<float>(s);
    }
  }
}

// y[r][:] = softmax(x[r][:]), numerically stable (max subtracted before exp).
inline void reference_row_softmax(const float* x, float* y,
                                  std::int64_t rows, std::int64_t cols) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    float* orow = y + r * cols;
    // Subtracting the row max is MANDATORY, not an optimisation: expf overflows
    // f32 for arguments above ~88, and a single large logit would otherwise turn
    // the whole row into inf/nan. The subtraction is exactly cancelled by the
    // division, so it changes nothing mathematically and everything numerically.
    double m = -std::numeric_limits<double>::infinity();
    for (std::int64_t c = 0; c < cols; ++c) m = std::max(m, static_cast<double>(row[c]));
    double sum = 0.0;
    for (std::int64_t c = 0; c < cols; ++c) sum += std::exp(static_cast<double>(row[c]) - m);
    for (std::int64_t c = 0; c < cols; ++c)
      orow[c] = static_cast<float>(std::exp(static_cast<double>(row[c]) - m) / sum);
  }
}

// C = alpha * A[m,k] @ B[k,n] + beta * C, all row-major.
//
// O(m*n*k), so ONLY call this at small shapes. A 4096-cubed reference is 1.4e11
// FLOPs single-threaded — minutes. Correctness is established at small sizes and
// performance measured at large ones; they are separate concerns and conflating
// them is what makes people skip the correctness half.
inline void reference_gemm(const float* a, const float* b, float* c,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           float alpha, float beta) {
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::int64_t p = 0; p < k; ++p)
        acc += static_cast<double>(a[i * k + p]) * static_cast<double>(b[p * n + j]);
      const double prev = static_cast<double>(c[i * n + j]);
      c[i * n + j] = static_cast<float>(static_cast<double>(alpha) * acc +
                                        static_cast<double>(beta) * prev);
    }
}

// -----------------------------------------------------------------------------
// Comparison
// -----------------------------------------------------------------------------

struct CompareResult {
  std::size_t n           = 0;
  std::size_t mismatches  = 0;
  double      max_abs_err = 0.0;
  double      max_rel_err = 0.0;
  std::size_t worst_index = 0;
  double      worst_got   = 0.0;
  double      worst_want  = 0.0;
  bool        any_nan     = false;   // tracked separately: NaN fails every comparison
  bool        any_inf     = false;

  [[nodiscard]] bool ok() const { return mismatches == 0 && !any_nan && !any_inf; }

  [[nodiscard]] std::string to_string() const {
    std::ostringstream os;
    os << (ok() ? "OK" : "FAIL") << ": " << mismatches << " / " << n << " mismatched";
    if (any_nan) os << ", NaN present";
    if (any_inf) os << ", Inf present";
    os << "; max_abs=" << max_abs_err << " max_rel=" << max_rel_err;
    if (mismatches) os << "; worst at [" << worst_index << "] got " << worst_got
                       << " want " << worst_want;
    return os.str();
  }
};

// |got - want| <= abs_tol + rel_tol * |want|   (the numpy.allclose form)
//
// NaN is reported separately rather than folded into the mismatch count: a NaN
// compares false against everything including itself, so without the explicit
// check a fully-NaN output and a slightly-off output look identical in the
// mismatch count, when they are completely different bugs. NaN in a softmax
// almost always means the max-subtraction was skipped.
inline CompareResult compare(const float* got, const float* want, std::size_t n,
                             double rel_tol, double abs_tol = 1e-8) {
  CompareResult r;
  r.n = n;
  // Tracks the worst FLAGGED mismatch specifically -- deliberately separate
  // from r.max_abs_err, which is a running max over EVERY element regardless
  // of whether it failed tolerance. An earlier version of this function
  // compared against r.max_abs_err directly, which is updated unconditionally
  // just above -- so "abs_err >= r.max_abs_err" degenerated to "did this
  // element just set a new global max", not "is this the worst mismatch".
  // Concretely: an element with a large |want| can have the single largest
  // abs_err in the whole array while still passing tolerance easily (its
  // rel_tol*|want| term is large); that element would silently claim
  // worst_index even though it never failed, starving the diagnostic of the
  // element that actually caused the failure.
  double worst_mismatch_rel = -1.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(got[i]);
    const double w = static_cast<double>(want[i]);
    if (std::isnan(g) || std::isnan(w)) { r.any_nan = true; ++r.mismatches; continue; }
    if (std::isinf(g) != std::isinf(w)) { r.any_inf = true; ++r.mismatches; continue; }

    const double abs_err = std::fabs(g - w);
    const double rel_err = std::fabs(w) > 0.0 ? abs_err / std::fabs(w) : 0.0;
    if (abs_err > r.max_abs_err) r.max_abs_err = abs_err;
    if (rel_err > r.max_rel_err) r.max_rel_err = rel_err;

    if (abs_err > abs_tol + rel_tol * std::fabs(w)) {
      ++r.mismatches;
      if (rel_err > worst_mismatch_rel) {
        worst_mismatch_rel = rel_err;
        r.worst_index = i;
        r.worst_got = g;
        r.worst_want = w;
      }
    }
  }
  return r;
}

// Recommended tolerances, derived rather than guessed (see the banner). Exposed
// as named constants so a bench states WHICH tolerance it used and a reader can
// check the reasoning instead of trusting a magic number at the call site.
inline constexpr double kTolElementwise = 1e-6;   // one op: a few ulp
inline constexpr double kTolReduce4096  = 1e-5;   // sqrt(4096) * f32 eps
inline constexpr double kTolSoftmax     = 1e-5;   // as reduce, plus expf's own ulp
inline constexpr double kTolGemmK256    = 1e-5;   // sqrt(256) * f32 eps, with headroom

// -----------------------------------------------------------------------------
// Absolute floors for two kernels where the DEFAULT abs_tol=1e-8 is the wrong
// size -- found by running on real hardware, not derived in advance. Both are
// instances of the same trap this file's own banner warns about (near-zero
// "want" values are routine, not an edge case), just showing up in a form that
// was not anticipated until measured:
//
// GELU: y = 0.5*x*(1+tanh(z)) has a genuine O(1) intermediate, (1+tanh(z)) or
// (1+erf(z)). A normal few-ULP disagreement between device tanhf/erff and host
// std::tanh/std::erf (measured: 4.77e-7 = exactly 4 ULP at magnitude ~2) shows
// up as an ABSOLUTE error in y that does NOT scale down with y's own
// magnitude -- and for x near the curve's knee, y itself is tiny (~1e-4), so a
// routine libm disagreement reads as a huge RELATIVE error (measured: 8e-4
// against a 1e-6 relative tolerance). Not a kernel bug.
//
// ROW-SUM over `cols` terms of O(1) magnitude: the rounding error is set by
// the magnitude of the TERMS being summed, not by the magnitude of the final
// result. For zero-mean random data, some rows nearly cancel by chance
// (routine, not adversarial) -- measured worst case: want=0.2, abs error
// 1.53e-5, at cols=4096 with data in [-1,1]. kMean is unaffected because both
// the value and the tolerance floor shrink together by the same factor of
// cols; kSum has nothing to shrink, so the same absolute error that passes
// easily for kMean fails a rel_tol*|want| test for kSum's occasionally-tiny
// sums.
//
// Both floors are sized from what was ACTUALLY MEASURED on a Tesla T4, with a
// >= 2x safety margin -- deriving a tight theoretical bound for pairwise
// float32 summation error is a genuine rabbit hole (see Higham, "The accuracy
// of floating point summation"), and a measured-plus-margin floor is the
// pragmatic answer for a fixed, known benchmark shape.
inline constexpr double kAbsTolGeluCancellation = 1e-6;   // >= 2x the observed 4.77e-7
inline constexpr double kAbsTolReduceSum4096    = 5e-5;   // >= 3x the observed 1.53e-5

}  // namespace mcke::testing
