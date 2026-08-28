# QUESTIONS.md

The owner's parking lot for questions that come up mid-conversation but that
the owner explicitly does **not** want answered right now — usually because
answering would derail whatever is currently being worked on, or because the
honest answer only makes real sense once more of the project exists to point
at. This is distinct from `LEARNING_LOG.md` (concepts already explained,
revisited later for retention) and from `PROJECT_LOG.md` (what happened,
when). This file is specifically for **unresolved, deliberately deferred**
questions.

**Convention:** when the owner ends a message with `(question.md)`, capture
the question that message just asked into this file — verbatim — rather than
answering it in that turn. Do not answer it there either, unless the owner
says otherwise; the whole point of the tag is "not now."

**If you are an assistant/agent reading this file in a later session:**
1. Do not assume any entry here was answered elsewhere unless it is marked
   `[ANSWERED]` with a pointer to where.
2. Read the attached context for an entry before attempting an answer — it
   exists so the question is answerable without re-reading the original
   conversation it came from.
3. Confirm with the owner that now is actually a good time before diving in.
   These were deferred on purpose, often until "the project is done" or a
   specific later phase.

Each entry: the question verbatim, when/where it came up, the context needed
to answer it properly, and a pointer to the file(s)/concept it's about.

---

## Open questions

### 1. How do we unit-test allocator internals in isolation, with no "real" caller driving them?

**Asked:** 2026-08-26 (approx.), in a forked chat, during a Stage 5 planning
discussion about whether to exercise `FreeListAllocator`'s large-allocation
bypass path (see `include/mcke/memory/freelist_allocator.hpp` — any request
bigger than `large_alloc_threshold` skips the custom pool and goes straight to
the plain system allocator) on real GPU hardware inside `bench/alloc_bench.cpp`.

**Context:** the assistant had said the bypass boundary logic "has already
been tested separately in a small isolated check on your Mac (confirming the
exact size cutoff behaves correctly)" — referring to `test_buddy_bypass()` (and
the analogous free-list test) in `tests/test_host_core.cpp`. The owner wants to
understand, later, what that sentence actually means mechanically.

**The question, verbatim:**
> in this how did we set up the testing, and how do we do a isolated check?
> and if it not in actual use then how do we give it values (inputs) and how
> do we see what it give and how it gives?

In plain terms: when a piece of code (like the bypass size-cutoff check) isn't
being driven by the real program (no graph, no real tensors, nobody actually
calling it as part of normal use) — how do we call it directly with made-up
inputs of our choosing, and how do we inspect what it returns / how it
behaved, without a real caller wired up?

**Where to look when this comes back up:** `tests/test_host_core.cpp` —
specifically `test_buddy_bypass()` and its free-list counterpart. Worth
walking through, from scratch:
- what a "unit test" is, conceptually, for someone who hasn't written one —
  a small standalone program whose only job is to call one piece of code with
  chosen inputs and check the output against what's expected.
- how the test constructs an allocator object directly (`mcke::BuddyAllocator
  a(fixed_config());`) with no graph, no tensors, nothing else running.
- how it picks specific byte counts that straddle the `large_alloc_threshold`
  boundary (e.g. exactly at the threshold vs. one byte over) and calls
  `allocate()` with each, by hand.
- what it checks afterward (the `CHECK_EQ`/`CHECK` macros — pattern defined at
  the top of `tests/test_host_core.cpp`) — e.g. that a request just over the
  threshold gets `slab_id == kBypassSlabId` and that `stats().raw_malloc_calls`
  went up by exactly one.
- the general idea that this is *isolated* because nothing about the rest of
  the runtime (graph, tensors, kernels) needs to exist or run for this test to
  make sense — the allocator is a self-contained object you can poke directly.

### 2. Why does some code behave identically on any machine, while other code can genuinely behave differently depending on the hardware?

**Asked:** 2026-08-26 (approx.), same forked chat, immediately after question 1,
following up on the assistant's explanation of why the free-list bypass logic
didn't need GPU testing while the earlier stream-safety logic did.

**Context — the exact passage that prompted this** (quoted in full because the
question is "go deeper on this," so the passage itself is the starting point,
not just a summary of it):

> Yes, that's exactly it, and here's the actual reasoning rather than just the
> conclusion: the bypass rule is nothing but "is this number bigger than that
> number?" — pure arithmetic. That kind of check gives the identical answer on
> a Mac, a GPU server, a toaster, anything, because there's no timing, no
> hardware quirk, no concurrency involved. Once proven on the Mac, running the
> same comparison on a GPU is very unlikely to reveal anything new.
>
> Compare that to the stream-safety question from before: that one was
> fundamentally about real concurrent execution — whether the GPU is actually
> still busy right now is something a Mac literally cannot simulate for real,
> it can only have us pretend. That's why GPU testing was essential there and
> much lower-value here. It's not a blanket rule ("skip GPU always") — it's
> specific to why each piece of logic could possibly go wrong.

**The question:** the owner understood this specific case, but wants to go
in-depth later into the *general* principle behind it: what makes a piece of
code guaranteed to behave the same on any machine, versus what kinds of code
can genuinely behave differently depending on the hardware it runs on — and
why, mechanically, that difference exists.

**Where to look when this comes back up:** this is a general systems/computer-
architecture question, not tied to one file, but this project has clean
worked examples of both categories to anchor the explanation in:
- **"same everywhere" category** — pure arithmetic/logic with no I/O, no
  timing, no hardware-specific instructions: e.g. the bypass size comparison
  itself, or `mcke::buddy::level_for_size` in `include/mcke/memory/buddy_math.hpp`
  (all `constexpr`, tested at compile time for exactly this reason).
- **"can genuinely differ" category** — real concurrency and timing (the
  `#if MCKE_WITH_CUDA` split in `include/mcke/runtime/stream.hpp`: the host
  build's `stream_query`/`event_query` are hard-coded `true` because there is
  no real asynchronous execution to ask about, while the CUDA build asks the
  driver a genuinely unpredictable question about real hardware state);
  hardware-dependent performance characteristics (`docs/PROFILING.md`'s
  measured-vs-spec bandwidth discussion; the clock-granularity difference
  between this Mac's ARM timer and an x86 machine's, noted in
  `include/mcke/profiling/host_timer.hpp`).
- **worth mentioning as a distinct third bucket, not to conflate with
  concurrency:** environment/toolchain-level differences — floating-point
  rounding or fused-multiply-add behavior varying by compiler/architecture,
  compiler optimization differences, OS scheduling — which aren't about
  hardware timing or concurrency per se but still break the "runs identically
  everywhere" assumption.

Good candidate for a `LEARNING_LOG.md` entry once actually explained, since
it's exactly the kind of from-scratch concept that log is for.
