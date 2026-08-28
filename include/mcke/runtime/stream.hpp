// =============================================================================
//  mcke/runtime/stream.hpp
//
//  WHAT: RAII wrappers for the two primitives that make GPU execution
//        asynchronous: `Stream` (an ordered queue of work) and `Event` (a
//        marker you can wait on, or time between).
//
//  MENTAL MODEL — this is the concept the whole Phase-4 scheduler rests on:
//    * A stream is an in-order queue. Work in one stream runs in issue order.
//    * Two different streams have NO ordering relationship. That is where
//      overlap (and therefore speed) comes from — and where race conditions
//      come from.
//    * An event is a token you `record` into stream A and `wait` on from
//      stream B. That is the *only* cheap way to express a cross-stream
//      dependency. It creates a one-directional edge: "B must not proceed past
//      this point until A has reached the recorded point."
//
//  DESIGN DECISION — event-wait vs. stream-synchronize.
//    cudaStreamSynchronize(A) blocks the *CPU* until A drains. It is a
//    host-side barrier: the CPU stops issuing work, the pipeline empties, and
//    you lose all overlap. Correct, but it converts your async runtime into a
//    synchronous one.
//    cudaStreamWaitEvent(B, e) blocks nothing on the CPU. It inserts a wait
//    *into the GPU's queue for B*. The CPU keeps racing ahead enqueuing work.
//    Rule we follow: cross-node dependencies inside the graph ALWAYS use
//    events; cudaStreamSynchronize appears exactly twice in the runtime — at
//    the end of `GraphExecutor::run_sync()` and in benchmark timing code.
//
//  DESIGN DECISION — cudaStreamNonBlocking.
//    Streams created with the default flag have an implicit dependency on the
//    legacy NULL stream: any work on stream 0 acts as a full barrier against
//    them. Since library code (and cudaMemcpy without a stream argument) uses
//    the NULL stream, the default flag silently serialises everything. We
//    always pass cudaStreamNonBlocking. This single flag is a classic
//    "why is my overlap not happening" bug.
//
//  WHY header-only: these are ~10 thin inline wrappers around C calls. Putting
//  them in a .cpp would add a call boundary the compiler cannot see through,
//  for zero benefit. The cost is that including this header pulls in CUDA
//  headers — acceptable, because it is guarded and only kernel/runtime TUs
//  include it.
// =============================================================================
#pragma once

#include <cstdint>
#include <utility>

#include "mcke/core/config.hpp"
#include "mcke/core/status.hpp"
#include "mcke/runtime/cuda_check.hpp"

namespace mcke::rt {

#if MCKE_WITH_CUDA
using StreamHandle = cudaStream_t;
using EventHandle  = cudaEvent_t;
#else
using StreamHandle = void*;
using EventHandle  = void*;
#endif

enum class StreamPriority { kHigh, kNormal, kLow };

// -----------------------------------------------------------------------------
// Non-blocking completion query on a *raw handle*.
//
// WHY this exists as a free function when Stream::query() already does the same
// thing: our allocator stores bare `StreamHandle`s, not `Stream` objects. It has
// to — a `Stream` is a move-only RAII owner with a private constructor, so an
// allocator cannot store or reconstruct one, and it must not own the caller's
// stream anyway. Without this free function the allocator's pending-free
// reclaim logic is literally unwritable.
//
// Returns true if every operation enqueued on `h` so far has completed. Note the
// asymmetry with the CUDA API: cudaStreamQuery returns cudaErrorNotReady (not an
// error!) for "still running", so anything other than cudaSuccess must be read
// as "not done" rather than propagated as a failure.
//
// In a host-only build this is unconditionally true — there is no async work.
// That is exactly why the allocator needs a test seam to exercise the "not yet
// safe to reuse" branch on a machine with no GPU (see buddy_allocator.hpp).
[[nodiscard]] inline bool stream_query([[maybe_unused]] StreamHandle h) {
#if MCKE_WITH_CUDA
  return cudaStreamQuery(h) == cudaSuccess;
#else
  return true;
#endif
}

// Raw-handle event operations, for the same reason stream_query exists: the
// allocator holds bare handles, not the move-only RAII wrappers.
//
// event_record marks "this point in this stream" so that a later event_query can
// answer "has the work that was queued before the free finished?" — which is a
// strictly more precise question than stream_query's "is this whole stream idle?".
[[nodiscard]] inline bool event_record([[maybe_unused]] EventHandle e,
                                      [[maybe_unused]] StreamHandle s) {
#if MCKE_WITH_CUDA
  return cudaEventRecord(e, s) == cudaSuccess;
#else
  return true;
#endif
}

[[nodiscard]] inline bool event_query([[maybe_unused]] EventHandle e) {
#if MCKE_WITH_CUDA
  // As with cudaStreamQuery, cudaErrorNotReady means "still running", not
  // "failed" — so anything non-success must be read as "not done".
  return cudaEventQuery(e) == cudaSuccess;
#else
  return true;
#endif
}

inline void event_synchronize([[maybe_unused]] EventHandle e) {
#if MCKE_WITH_CUDA
  (void)cudaEventSynchronize(e);
#endif
}

class Event;

// -----------------------------------------------------------------------------
// Stream
// -----------------------------------------------------------------------------
class Stream {
 public:
  // Non-owning wrapper around the legacy default stream. Used only for interop
  // and for the "single stream baseline" execution policy in Phase 1.
  static Stream default_stream() { return Stream(StreamHandle{}, /*owned=*/false); }

  [[nodiscard]] static StatusOr<Stream> create(StreamPriority prio = StreamPriority::kNormal) {
#if MCKE_WITH_CUDA
    // Priority range is device-dependent and, counter-intuitively, *lower
    // numbers are higher priority*. Querying instead of hard-coding matters:
    // on some GPUs the range is [-5,0], on others [-2,0].
    int lo = 0, hi = 0;
    MCKE_CUDA_RETURN_IF_ERROR(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    int p = (prio == StreamPriority::kHigh) ? hi : (prio == StreamPriority::kLow ? lo : (lo + hi) / 2);
    cudaStream_t s{};
    MCKE_CUDA_RETURN_IF_ERROR(
        cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, p));
    return Stream(s, /*owned=*/true);
#else
    (void)prio;
    return Stream(nullptr, /*owned=*/false);
#endif
  }

  Stream(const Stream&)            = delete;   // a stream is a resource, not a value
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& o) noexcept : handle_(o.handle_), owned_(o.owned_) {
    o.owned_ = false;
  }
  Stream& operator=(Stream&& o) noexcept {
    if (this != &o) { destroy(); handle_ = o.handle_; owned_ = o.owned_; o.owned_ = false; }
    return *this;
  }
  ~Stream() { destroy(); }

  [[nodiscard]] StreamHandle native() const noexcept { return handle_; }

  // Host-side barrier. Use sparingly — see the header comment.
  Status synchronize() const {
#if MCKE_WITH_CUDA
    MCKE_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(handle_));
#endif
    return OkStatus();
  }

  // Non-blocking poll: has everything enqueued so far completed?
  // This is what the allocator uses to decide whether a freed block whose
  // last use was on this stream is safe to hand out again (Phase 2).
  [[nodiscard]] bool query() const {
#if MCKE_WITH_CUDA
    return cudaStreamQuery(handle_) == cudaSuccess;
#else
    return true;
#endif
  }

  // Insert a GPU-side wait: work enqueued into *this* stream after this call
  // will not begin until `e` has been reached in whichever stream recorded it.
  Status wait(const Event& e);

 private:
  Stream(StreamHandle h, bool owned) : handle_(h), owned_(owned) {}
  void destroy() {
#if MCKE_WITH_CUDA
    // Note: cudaStreamDestroy is itself asynchronous — it returns immediately
    // and the stream is torn down once its pending work completes. So we do NOT
    // need to synchronize first, and doing so would be a needless stall.
    if (owned_ && handle_) cudaStreamDestroy(handle_);
#endif
    owned_ = false;
  }

  StreamHandle handle_{};
  bool         owned_ = false;
};

// -----------------------------------------------------------------------------
// Event
// -----------------------------------------------------------------------------
class Event {
 public:
  enum class Purpose {
    kDependency,  // used only for ordering -> created with cudaEventDisableTiming
    kTiming       // used with elapsed_ms() -> full timing support
  };

  [[nodiscard]] static StatusOr<Event> create(Purpose purpose = Purpose::kDependency) {
#if MCKE_WITH_CUDA
    // cudaEventDisableTiming is not a micro-optimisation to skip: a timing
    // event forces the GPU to record a timestamp, which adds a small pipeline
    // hiccup and, on some architectures, prevents the event from being resolved
    // entirely on-device. In a graph with hundreds of dependency edges those
    // add up. Only events we intend to *measure* with get timing enabled.
    unsigned flags = (purpose == Purpose::kTiming) ? cudaEventDefault
                                                   : cudaEventDisableTiming;
    cudaEvent_t ev{};
    MCKE_CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&ev, flags));
    return Event(ev);
#else
    (void)purpose;
    return Event(nullptr);
#endif
  }

  Event(const Event&)            = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& o) noexcept : handle_(o.handle_) { o.handle_ = EventHandle{}; }
  Event& operator=(Event&& o) noexcept {
    if (this != &o) { destroy(); handle_ = o.handle_; o.handle_ = EventHandle{}; }
    return *this;
  }
  ~Event() { destroy(); }

  [[nodiscard]] EventHandle native() const noexcept { return handle_; }

  // Mark "this point in this stream". An event may be re-recorded; the previous
  // recording is simply overwritten, which is what lets us reuse one event per
  // graph node across repeated executions instead of allocating per iteration.
  Status record(const Stream& s) {
#if MCKE_WITH_CUDA
    MCKE_CUDA_RETURN_IF_ERROR(cudaEventRecord(handle_, s.native()));
#else
    (void)s;
#endif
    return OkStatus();
  }

  Status synchronize() const {
#if MCKE_WITH_CUDA
    MCKE_CUDA_RETURN_IF_ERROR(cudaEventSynchronize(handle_));
#endif
    return OkStatus();
  }

  [[nodiscard]] bool query() const {
#if MCKE_WITH_CUDA
    return cudaEventQuery(handle_) == cudaSuccess;
#else
    return true;
#endif
  }

  // Elapsed GPU time between two *completed* timing events, in milliseconds.
  //
  // WHY this and not std::chrono around the launch: the launch returns before
  // the kernel runs, so a host timer measures launch overhead (~5 us), not the
  // kernel. Events are timestamped by the GPU itself in the stream's order, so
  // they measure device time with ~0.5 us resolution. This is the correct way
  // to time a kernel, and the only one we use in benchmarks.
  [[nodiscard]] static StatusOr<float> elapsed_ms(const Event& start, const Event& stop) {
#if MCKE_WITH_CUDA
    float ms = 0.f;
    MCKE_CUDA_RETURN_IF_ERROR(cudaEventElapsedTime(&ms, start.native(), stop.native()));
    return ms;
#else
    (void)start; (void)stop;
    return 0.0f;
#endif
  }

 private:
  explicit Event(EventHandle h) : handle_(h) {}
  void destroy() {
#if MCKE_WITH_CUDA
    if (handle_) cudaEventDestroy(handle_);
#endif
    handle_ = EventHandle{};
  }
  EventHandle handle_{};
};

inline Status Stream::wait(const Event& e) {
#if MCKE_WITH_CUDA
  // Third argument is a flags word, reserved-must-be-0 on all current
  // architectures except for cudaEventWaitExternal.
  MCKE_CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(handle_, e.native(), 0));
#else
  (void)e;
#endif
  return OkStatus();
}

}  // namespace mcke::rt
