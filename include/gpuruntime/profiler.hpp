#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace gpuruntime {

/// Single complete ("X") trace event compatible with chrome://tracing and Perfetto.
struct TraceEvent {
    std::string name;
    std::string category = "runtime";
    std::uint64_t pid = 1;
    std::uint64_t tid = 0;
    double ts_us = 0.0;
    double dur_us = 0.0;
    std::uint64_t flops = 0;
    std::uint64_t bytes = 0;
};

/// Aggregated latency distribution and roofline throughput metrics for an event name.
struct EventStats {
    std::string name;
    std::size_t count = 0;
    double min_us = 0.0;
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double max_us = 0.0;
    double total_us = 0.0;
    std::uint64_t total_flops = 0;
    std::uint64_t total_bytes = 0;
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;
    double arithmetic_intensity = 0.0;
};

/// Thread-safe event profiler with Chrome Trace JSON export and percentile / roofline stats.
class TraceProfiler {
public:
    using Clock = std::chrono::steady_clock;

    TraceProfiler();

    /// Records a pre-constructed TraceEvent (useful for synthetic spans or external timings).
    void record_event(TraceEvent event);

    /// Records a timed span relative to the profiler epoch.
    void record_span(std::string name,
                     std::string category,
                     Clock::time_point start,
                     Clock::time_point end,
                     std::uint64_t flops = 0,
                     std::uint64_t bytes = 0,
                     std::optional<std::uint64_t> explicit_tid = std::nullopt);

    [[nodiscard]] std::vector<TraceEvent> events() const;
    [[nodiscard]] std::vector<EventStats> summarize() const;
    [[nodiscard]] std::optional<EventStats> stats_for(const std::string& name) const;

    /// Exports recorded events in Chrome Trace Event Format (`ph: "X"`).
    [[nodiscard]] std::string export_chrome_trace_json() const;

    /// Writes Chrome Trace JSON directly to disk. Throws std::runtime_error on I/O failure.
    void write_chrome_trace_json(const std::string& filepath) const;

    /// Formats a human-readable ASCII summary table of latency percentiles & roofline metrics.
    [[nodiscard]] std::string format_summary_table() const;

    /// Clears recorded events and resets the profiler epoch.
    void clear();

private:
    std::uint64_t resolve_thread_id_locked(std::thread::id id) const;

    mutable std::mutex mutex_;
    Clock::time_point epoch_;
    std::vector<TraceEvent> events_;
    mutable std::unordered_map<std::thread::id, std::uint64_t> thread_id_map_;
    mutable std::uint64_t next_tid_ = 0;
};

/// RAII scoped trace event that automatically records its elapsed lifetime on destruction.
class ScopedTraceEvent {
public:
    ScopedTraceEvent(TraceProfiler* profiler,
                     std::string name,
                     std::string category = "runtime",
                     std::uint64_t flops = 0,
                     std::uint64_t bytes = 0,
                     std::optional<std::uint64_t> explicit_tid = std::nullopt) noexcept;
    ~ScopedTraceEvent() noexcept;

    ScopedTraceEvent(const ScopedTraceEvent&) = delete;
    ScopedTraceEvent& operator=(const ScopedTraceEvent&) = delete;
    ScopedTraceEvent(ScopedTraceEvent&&) = delete;
    ScopedTraceEvent& operator=(ScopedTraceEvent&&) = delete;

private:
    TraceProfiler* profiler_ = nullptr;
    std::string name_;
    std::string category_;
    std::uint64_t flops_ = 0;
    std::uint64_t bytes_ = 0;
    std::optional<std::uint64_t> explicit_tid_;
    TraceProfiler::Clock::time_point start_;
};

}  // namespace gpuruntime
