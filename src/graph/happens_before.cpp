// =============================================================================
//  src/graph/happens_before.cpp
//
//  Vector-clock construction for the scheduled plan's happens-before relation.
//  See mcke/graph/happens_before.hpp for why this relation, and not topological
//  order, is what a memory planner must ask.
//
//  WHY .cpp AND NOT HEADER-INLINE: build() validates its inputs and returns
//  Status, so it needs <sstream> and a fair amount of error prose. The two hot
//  queries (precedes/concurrent) stay inline in the header where they are two
//  array reads and a comparison.
// =============================================================================
#include "mcke/graph/happens_before.hpp"

#include <algorithm>
#include <sstream>

namespace mcke {

StatusOr<HappensBefore> HappensBefore::build(
    std::size_t num_streams,
    const std::vector<std::uint16_t>& stream_of,
    const std::vector<std::uint32_t>& issue_of,
    const std::vector<std::uint32_t>& record_of,
    const std::vector<std::vector<std::uint32_t>>& waits_of,
    const std::vector<std::uint32_t>& order) {
  const std::size_t n = stream_of.size();
  if (num_streams == 0)
    return InvalidArgumentError("HappensBefore::build: num_streams must be > 0");
  if (issue_of.size() != n || record_of.size() != n || waits_of.size() != n)
    return InvalidArgumentError("HappensBefore::build: per-node arrays disagree in length");
  if (order.size() != n)
    return InvalidArgumentError("HappensBefore::build: order length != node count");

  for (std::size_t i = 0; i < n; ++i)
    if (stream_of[i] >= num_streams)
      return InvalidArgumentError("HappensBefore::build: node " + std::to_string(i) +
                                  " is on stream " + std::to_string(stream_of[i]) +
                                  " but only " + std::to_string(num_streams) + " exist");

  // event id -> the node that records it. Built by inverting record_of, with a
  // duplicate check: two nodes recording the same event would silently make the
  // relation weaker than it looks (the second record would move the event's
  // timestamp under a wait that had already been issued against the first).
  std::vector<std::uint32_t> recorder;
  for (std::size_t i = 0; i < n; ++i) {
    if (record_of[i] == kNoEventId) continue;
    const std::size_t e = record_of[i];
    if (e >= recorder.size()) recorder.resize(e + 1, kNoEventId);
    if (recorder[e] != kNoEventId)
      return InvalidArgumentError("HappensBefore::build: event " + std::to_string(e) +
                                  " is recorded by both node " +
                                  std::to_string(recorder[e]) + " and node " +
                                  std::to_string(i));
    recorder[e] = static_cast<std::uint32_t>(i);
  }

  HappensBefore hb;
  hb.num_streams_ = num_streams;
  hb.stream_of_   = stream_of;
  hb.issue_of_    = issue_of;
  hb.clock_.assign(n * num_streams, -1);

  // Per stream, issue index -> node, so a node can find its same-stream
  // predecessor in O(1). Also validates that issue indices are a dense 0..k-1
  // per stream, which the planner is responsible for and which a silent gap
  // would quietly weaken the relation.
  std::vector<std::vector<std::uint32_t>> by_stream(num_streams);
  for (std::size_t i = 0; i < n; ++i) {
    auto& col = by_stream[stream_of[i]];
    const std::size_t want = issue_of[i];
    if (col.size() <= want) col.resize(want + 1, kNoEventId);
    if (col[want] != kNoEventId)
      return InvalidArgumentError("HappensBefore::build: nodes " +
                                  std::to_string(col[want]) + " and " +
                                  std::to_string(i) + " share issue index " +
                                  std::to_string(want) + " on stream " +
                                  std::to_string(stream_of[i]));
    col[want] = static_cast<std::uint32_t>(i);
  }
  for (std::size_t s = 0; s < num_streams; ++s)
    for (std::size_t k = 0; k < by_stream[s].size(); ++k)
      if (by_stream[s][k] == kNoEventId)
        return InvalidArgumentError("HappensBefore::build: stream " + std::to_string(s) +
                                    " has no node at issue index " + std::to_string(k) +
                                    " (issue indices must be dense per stream)");

  // The construction itself. Walking `order` is what makes one pass sufficient:
  // it is a linear extension of the dependency edges, and a node can only wait
  // on an event recorded by one of its predecessors, so every clock this node
  // reads is already final.
  for (std::uint32_t nid : order) {
    if (nid >= n)
      return InvalidArgumentError("HappensBefore::build: order contains out-of-range node " +
                                  std::to_string(nid));
    std::int64_t* c = &hb.clock_[static_cast<std::size_t>(nid) * num_streams];

    auto absorb = [&](std::uint32_t other) {
      const std::int64_t* o = &hb.clock_[static_cast<std::size_t>(other) * num_streams];
      for (std::size_t s = 0; s < num_streams; ++s) c[s] = std::max(c[s], o[s]);
    };

    // (1) same-stream issue order. Absorbing the predecessor's whole clock is
    // what makes the relation transitive: its own position is already recorded
    // in its clock at clock[prev][stream(prev)].
    const std::uint32_t iss = issue_of[nid];
    if (iss > 0) absorb(by_stream[stream_of[nid]][iss - 1]);

    // (2) event edges.
    for (std::uint32_t e : waits_of[nid]) {
      if (e >= recorder.size() || recorder[e] == kNoEventId)
        return InvalidArgumentError("HappensBefore::build: node " + std::to_string(nid) +
                                    " waits on event " + std::to_string(e) +
                                    " which no node records");
      absorb(recorder[e]);
    }

    c[stream_of[nid]] = static_cast<std::int64_t>(iss);
  }

  return hb;
}

std::string HappensBefore::describe_pair(std::uint32_t u, std::uint32_t v) const {
  std::ostringstream os;
  os << "node " << u << " (stream " << stream_of_[u] << ", issue " << issue_of_[u] << ") and "
     << "node " << v << " (stream " << stream_of_[v] << ", issue " << issue_of_[v] << "): ";
  if (precedes(u, v))      os << u << " happens-before " << v;
  else if (precedes(v, u)) os << v << " happens-before " << u;
  else                     os << "CONCURRENT -- neither orders the other";
  os << "; clock[" << v << "][s" << stream_of_[u] << "] = "
     << clock_[static_cast<std::size_t>(v) * num_streams_ + stream_of_[u]]
     << " vs issue[" << u << "] = " << issue_of_[u];
  return os.str();
}

}  // namespace mcke
