// =============================================================================
//  mcke/kernels/softmax_online.hpp
//
//  WHAT: The online-softmax running state `(m, d)` and its combine operator —
//        the single piece of arithmetic in Phase 3 that is genuinely easy to get
//        subtly wrong.
//
//  WHY .hpp AND NOT .cuh, when kernels/reduce_ops.cuh is a .cuh: because this is
//  MCKE_HOST_DEVICE. It uses nothing but fmaxf and expf, so it compiles as plain
//  C++ on a machine with no CUDA — which means the recurrence can be unit-tested
//  EXHAUSTIVELY on the MacBook, against the three-pass result, before it is ever
//  compiled by nvcc. (`dtype_size` in core/dtype.hpp is the precedent for a
//  MCKE_HOST_DEVICE function living in the shipped include tree.)
//
//  That testability is the entire reason this is a separate header rather than
//  twenty lines inside softmax.cu. Debugging a rescaling recurrence through a
//  GPU kernel, on a metered Colab session, is a bad way to spend an afternoon.
//
//  ---------------------------------------------------------------------------
//  THE ALGORITHM (Milakov & Gimelshein, "Online normalizer calculation for
//  softmax")
//
//  Naive stable softmax needs the row maximum before it can exponentiate
//  anything, so it reads the row once for the max and again for the sum. The
//  online formulation carries a running pair:
//
//      m = the largest element seen so far
//      d = sum of exp(x_i - m) over elements seen so far, measured against THAT m
//
//  and updates both in a single traversal. When a new element arrives:
//
//      m_new = max(m, x)
//      d_new = d * exp(m - m_new) + exp(x - m_new)
//
//  Read the rescale factor as the whole idea: everything already accumulated was
//  measured against the OLD, smaller max, so before adding a new term you scale
//  the old sum down by exp(m_old - m_new), which is <= 1. Two cases fall out:
//    * x <= m  : m_new = m, exp(m - m_new) = 1, so d += exp(x - m). Same as the
//                three-pass inner loop; no rescale happens at all.
//    * x >  m  : m_new = x, so d = d*exp(m - x) + 1. One rescale.
//
//  ---------------------------------------------------------------------------
//  WHY THE *COMBINE* MATTERS MORE THAN THE PER-ELEMENT UPDATE
//
//  Combining two independently-accumulated partial states:
//
//      m = max(m_a, m_b)
//      d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
//
//  This operator is ASSOCIATIVE and COMMUTATIVE. That is not a footnote — it is
//  the whole reason the algorithm is useful. Associativity is exactly what a tree
//  reduction requires, so `(m, d)` pairs can be merged in any order: lane by lane
//  within a warp, warp by warp across a block, block by block across a grid,
//  tile by tile across a sequence. Every parallel decomposition is legal.
//
//  AND THAT IS WHY FLASHATTENTION IS POSSIBLE. FlashAttention never materialises
//  the N x N attention matrix; it walks key/value blocks one tile at a time,
//  keeping only a running (m, d) plus a running rescaled output accumulator, and
//  merges each new tile with exactly this operator. The realisation that softmax
//  needs no global pass — only an associative merge — is what turns an O(N^2)
//  memory algorithm into an O(N) one. When you meet FlashAttention later, this
//  function is the part you will already have written.
// =============================================================================
#pragma once

#include <cfloat>
#include <cmath>

#include "mcke/core/config.hpp"

namespace mcke::kernels {

struct OnlineState {
  float m;   // running max
  float d;   // running sum of exp(x_i - m)
};

// The identity element: an empty partial contributes nothing.
//
// -FLT_MAX, NOT -INFINITY, and this is the single most important line in the
// file. With -INFINITY, combining two EMPTY partials computes
//     d_a * exp(m_a - m) = 0 * exp(-inf - (-inf)) = 0 * exp(NaN) = NaN
// and one NaN poisons an entire block reduction. With -FLT_MAX the same
// expression underflows cleanly: exp(-FLT_MAX - (-FLT_MAX)) = exp(0) = 1, and
// 0 * 1 = 0.
//
// This is only reachable when a thread has no data at all, i.e. when
// cols < blockDim.x — which is precisely why the validation shape list contains
// cols = 1 and cols = 17. It is the same convention MaxOp uses in
// kernels/reduce_ops.cuh, for the same reason.
[[nodiscard]] MCKE_HOST_DEVICE inline OnlineState online_identity() {
  return OnlineState{-FLT_MAX, 0.0f};
}

// Fold one element into a running state.
[[nodiscard]] MCKE_HOST_DEVICE inline OnlineState online_update(OnlineState s, float x) {
  const float m_new = fmaxf(s.m, x);
  // Both exponents are <= 0 by construction, so neither expf can overflow.
  return OnlineState{m_new, s.d * expf(s.m - m_new) + expf(x - m_new)};
}

// Merge two independently-accumulated partial states. Associative and
// commutative — see the banner for why that is the point.
[[nodiscard]] MCKE_HOST_DEVICE inline OnlineState online_combine(OnlineState a,
                                                                OnlineState b) {
  // Belt and braces on top of the -FLT_MAX identity: an empty partial short-
  // circuits, so no exp of a huge negative difference is evaluated at all.
  if (b.d == 0.0f) return a;
  if (a.d == 0.0f) return b;
  const float m = fmaxf(a.m, b.m);
  return OnlineState{m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

}  // namespace mcke::kernels
