#include "gpuruntime/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpuruntime {

namespace {

double percentile_sorted(const std::vector<double>& sorted, double p) noexcept {
    if (sorted.empty()) {
        return 0.0;
    }
    if (sorted.size() == 1) {
        return sorted.front();
    }
    const double clamped_p = std::clamp(p, 0.0, 1.0);
    const double pos = clamped_p * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(pos));
    const auto upper = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lower);
    return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
}

std::string escape_json_string(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (unsigned char ch : input) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(kHex[(ch >> 4) & 0x0F]);
                    out.push_back(kHex[ch & 0x0F]);
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return out;
}

EventStats compute_stats_for_group(const std::string& name,
                                   const std::vector<const TraceEvent*>& group) {
    EventStats stats{};
    stats.name = name;
    stats.count = group.size();
    if (group.empty()) {
        return stats;
    }

    std::vector<double> durations;
    durations.reserve(group.size());
    for (const TraceEvent* ev : group) {
        durations.push_back(ev->dur_us);
        stats.total_us += ev->dur_us;
        stats.total_flops += ev->flops;
        stats.total_bytes += ev->bytes;
    }

    std::sort(durations.begin(), durations.end());
    stats.min_us = durations.front();
    stats.max_us = durations.back();
    stats.mean_us = stats.total_us / static_cast<double>(durations.size());
    stats.p50_us = percentile_sorted(durations, 0.50);
    stats.p95_us = percentile_sorted(durations, 0.95);
    stats.p99_us = percentile_sorted(durations, 0.99);

    if (stats.total_us > 0.0) {
        const double total_sec = stats.total_us * 1e-6;
        stats.gflops = (static_cast<double>(stats.total_flops) / total_sec) * 1e-9;
        stats.bandwidth_gbps = (static_cast<double>(stats.total_bytes) / total_sec) * 1e-9;
    }
    if (stats.total_bytes > 0) {
        stats.arithmetic_intensity =
            static_cast<double>(stats.total_flops) / static_cast<double>(stats.total_bytes);
    }

    return stats;
}

}  // namespace

TraceProfiler::TraceProfiler() : epoch_(Clock::now()) {}

void TraceProfiler::record_event(TraceEvent event) {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back(std::move(event));
}

void TraceProfiler::record_span(std::string name,
                                std::string category,
                                Clock::time_point start,
                                Clock::time_point end,
                                std::uint64_t flops,
                                std::uint64_t bytes,
                                std::optional<std::uint64_t> explicit_tid) {
    const auto thread_id = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mutex_);

    const double ts_us =
        std::chrono::duration<double, std::micro>(start - epoch_).count();
    const double dur_us =
        std::max(0.0, std::chrono::duration<double, std::micro>(end - start).count());
    const std::uint64_t tid =
        explicit_tid.has_value() ? *explicit_tid : resolve_thread_id_locked(thread_id);

    TraceEvent ev{};
    ev.name = std::move(name);
    ev.category = std::move(category);
    ev.pid = 1;
    ev.tid = tid;
    ev.ts_us = std::max(0.0, ts_us);
    ev.dur_us = dur_us;
    ev.flops = flops;
    ev.bytes = bytes;
    events_.push_back(std::move(ev));
}

std::vector<TraceEvent> TraceProfiler::events() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_;
}

std::vector<EventStats> TraceProfiler::summarize() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, std::vector<const TraceEvent*>> grouped;
    for (const auto& ev : events_) {
        grouped[ev.name].push_back(&ev);
    }

    std::vector<EventStats> results;
    results.reserve(grouped.size());
    for (const auto& [name, group] : grouped) {
        results.push_back(compute_stats_for_group(name, group));
    }
    return results;
}

std::optional<EventStats> TraceProfiler::stats_for(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<const TraceEvent*> group;
    for (const auto& ev : events_) {
        if (ev.name == name) {
            group.push_back(&ev);
        }
    }
    if (group.empty()) {
        return std::nullopt;
    }
    return compute_stats_for_group(name, group);
}

std::string TraceProfiler::export_chrome_trace_json() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed << std::setprecision(3);
    oss << "{\n  \"traceEvents\": [\n";

    for (std::size_t i = 0; i < events_.size(); ++i) {
        const auto& ev = events_[i];
        double gflops = 0.0;
        double gbps = 0.0;
        if (ev.dur_us > 0.0) {
            const double sec = ev.dur_us * 1e-6;
            gflops = (static_cast<double>(ev.flops) / sec) * 1e-9;
            gbps = (static_cast<double>(ev.bytes) / sec) * 1e-9;
        }

        oss << "    {\"name\":\"" << escape_json_string(ev.name)
            << "\",\"cat\":\"" << escape_json_string(ev.category)
            << "\",\"ph\":\"X\",\"pid\":" << ev.pid
            << ",\"tid\":" << ev.tid
            << ",\"ts\":" << ev.ts_us
            << ",\"dur\":" << ev.dur_us;

        if (ev.flops > 0 || ev.bytes > 0) {
            oss << ",\"args\":{\"flops\":" << ev.flops
                << ",\"bytes\":" << ev.bytes
                << ",\"gflops\":" << gflops
                << ",\"gb_per_s\":" << gbps << "}";
        }
        oss << "}";
        if (i + 1 < events_.size()) {
            oss << ",";
        }
        oss << "\n";
    }

    oss << "  ]\n}\n";
    return oss.str();
}

void TraceProfiler::write_chrome_trace_json(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open Chrome trace output file: " + filepath);
    }
    const std::string json = export_chrome_trace_json();
    ofs.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!ofs.good()) {
        throw std::runtime_error("Failed while writing Chrome trace output file: " + filepath);
    }
}

std::string TraceProfiler::format_summary_table() const {
    const auto summary = summarize();
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed << std::setprecision(2);

    oss << std::left << std::setw(28) << "Event"
        << std::right << std::setw(7) << "Count"
        << std::setw(11) << "Min(us)"
        << std::setw(11) << "Mean(us)"
        << std::setw(11) << "P50(us)"
        << std::setw(11) << "P95(us)"
        << std::setw(11) << "P99(us)"
        << std::setw(11) << "Max(us)"
        << std::setw(10) << "GFLOP/s"
        << std::setw(10) << "GB/s"
        << "\n";
    oss << std::string(121, '-') << "\n";

    for (const auto& s : summary) {
        std::string label = s.name;
        if (label.size() > 27) {
            label = label.substr(0, 24) + "...";
        }
        oss << std::left << std::setw(28) << label
            << std::right << std::setw(7) << s.count
            << std::setw(11) << s.min_us
            << std::setw(11) << s.mean_us
            << std::setw(11) << s.p50_us
            << std::setw(11) << s.p95_us
            << std::setw(11) << s.p99_us
            << std::setw(11) << s.max_us
            << std::setw(10) << s.gflops
            << std::setw(10) << s.bandwidth_gbps
            << "\n";
    }
    return oss.str();
}

void TraceProfiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
    epoch_ = Clock::now();
}

std::uint64_t TraceProfiler::resolve_thread_id_locked(std::thread::id id) const {
    const auto it = thread_id_map_.find(id);
    if (it != thread_id_map_.end()) {
        return it->second;
    }
    const std::uint64_t assigned = next_tid_++;
    thread_id_map_.emplace(id, assigned);
    return assigned;
}

ScopedTraceEvent::ScopedTraceEvent(TraceProfiler* profiler,
                                   std::string name,
                                   std::string category,
                                   std::uint64_t flops,
                                   std::uint64_t bytes,
                                   std::optional<std::uint64_t> explicit_tid) noexcept
    : profiler_(profiler),
      name_(std::move(name)),
      category_(std::move(category)),
      flops_(flops),
      bytes_(bytes),
      explicit_tid_(explicit_tid),
      start_(TraceProfiler::Clock::now()) {}

ScopedTraceEvent::~ScopedTraceEvent() noexcept {
    if (profiler_ != nullptr) {
        const auto end = TraceProfiler::Clock::now();
        try {
            profiler_->record_span(std::move(name_),
                                   std::move(category_),
                                   start_,
                                   end,
                                   flops_,
                                   bytes_,
                                   explicit_tid_);
        } catch (...) {
            // Destructor must not propagate exceptions.
        }
    }
}

}  // namespace gpuruntime
