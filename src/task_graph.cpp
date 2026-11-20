#include "gpuruntime/task_graph.hpp"

#include "gpuruntime/profiler.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace gpuruntime {

namespace {

class StreamEvent {
public:
    void signal() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            signaled_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }

    void wait() {
        if (signaled_.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return signaled_.load(std::memory_order_acquire); });
    }

private:
    std::atomic<bool> signaled_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

std::string escape_dot_label(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

const char* dot_fill_color(TaskKind kind) noexcept {
    switch (kind) {
        case TaskKind::ComputeKernel:
            return "#d6eaf8";
        case TaskKind::TransferH2D:
            return "#d5f5e3";
        case TaskKind::TransferD2H:
            return "#fdebd0";
        case TaskKind::HostCallback:
            return "#e8daef";
    }
    return "#f2f3f4";
}

}  // namespace

const char* task_kind_to_string(TaskKind kind) noexcept {
    switch (kind) {
        case TaskKind::ComputeKernel:
            return "ComputeKernel";
        case TaskKind::TransferH2D:
            return "TransferH2D";
        case TaskKind::TransferD2H:
            return "TransferD2H";
        case TaskKind::HostCallback:
            return "HostCallback";
    }
    return "Unknown";
}

double TaskNode::effective_weight() const noexcept {
    if (estimated_cost_us > 0.0) {
        return estimated_cost_us;
    }
    if (estimated_flops > 0 || estimated_bytes > 0) {
        return std::max(1.0,
                        static_cast<double>(estimated_flops) * 1e-6 +
                            static_cast<double>(estimated_bytes) * 1e-6);
    }
    return 1.0;
}

NodeId TaskGraph::add_node(std::string name,
                           TaskKind kind,
                           std::size_t preferred_stream,
                           std::function<void()> work,
                           std::uint64_t estimated_flops,
                           std::uint64_t estimated_bytes,
                           double estimated_cost_us) {
    TaskNode spec{};
    spec.name = std::move(name);
    spec.kind = kind;
    spec.preferred_stream = preferred_stream;
    spec.estimated_flops = estimated_flops;
    spec.estimated_bytes = estimated_bytes;
    spec.estimated_cost_us = estimated_cost_us;
    spec.work = std::move(work);
    return add_node(std::move(spec));
}

NodeId TaskGraph::add_node(TaskNode spec) {
    const NodeId id = nodes_.size();
    spec.id = id;
    if (spec.name.empty()) {
        spec.name = "node_" + std::to_string(id);
    }
    nodes_.push_back(std::move(spec));
    adj_.emplace_back();
    rev_adj_.emplace_back();
    return id;
}

void TaskGraph::add_dependency(NodeId producer, NodeId consumer) {
    validate_node_id(producer);
    validate_node_id(consumer);

    auto& succ = adj_[producer];
    if (std::find(succ.begin(), succ.end(), consumer) == succ.end()) {
        succ.push_back(consumer);
        rev_adj_[consumer].push_back(producer);
    }
}

void TaskGraph::validate_node_id(NodeId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("TaskGraph: NodeId out of range");
    }
}

const TaskNode& TaskGraph::node(NodeId id) const {
    validate_node_id(id);
    return nodes_[id];
}

const std::vector<NodeId>& TaskGraph::successors(NodeId id) const {
    validate_node_id(id);
    return adj_[id];
}

const std::vector<NodeId>& TaskGraph::predecessors(NodeId id) const {
    validate_node_id(id);
    return rev_adj_[id];
}

bool TaskGraph::has_cycle() const {
    return !try_topological_sort().has_value();
}

std::optional<std::vector<NodeId>> TaskGraph::try_topological_sort() const {
    const std::size_t n = nodes_.size();
    std::vector<std::size_t> in_degree(n, 0);
    for (std::size_t u = 0; u < n; ++u) {
        in_degree[u] = rev_adj_[u].size();
    }

    std::queue<NodeId> ready;
    for (std::size_t u = 0; u < n; ++u) {
        if (in_degree[u] == 0) {
            ready.push(u);
        }
    }

    std::vector<NodeId> order;
    order.reserve(n);
    while (!ready.empty()) {
        const NodeId u = ready.front();
        ready.pop();
        order.push_back(u);

        for (NodeId v : adj_[u]) {
            if (--in_degree[v] == 0) {
                ready.push(v);
            }
        }
    }

    if (order.size() != n) {
        return std::nullopt;
    }
    return order;
}

std::vector<NodeId> TaskGraph::topological_sort() const {
    auto maybe_order = try_topological_sort();
    if (!maybe_order.has_value()) {
        throw CycleDetectedError(
            "TaskGraph contains a directed cycle; cannot construct valid execution schedule");
    }
    return std::move(*maybe_order);
}

std::vector<std::vector<NodeId>> TaskGraph::levelize() const {
    const std::size_t n = nodes_.size();
    std::vector<std::size_t> in_degree(n, 0);
    std::queue<NodeId> ready;
    for (std::size_t u = 0; u < n; ++u) {
        in_degree[u] = rev_adj_[u].size();
        if (in_degree[u] == 0) {
            ready.push(u);
        }
    }

    std::vector<std::vector<NodeId>> levels;
    std::size_t visited = 0;

    while (!ready.empty()) {
        const std::size_t wave_size = ready.size();
        std::vector<NodeId> wave;
        wave.reserve(wave_size);

        for (std::size_t i = 0; i < wave_size; ++i) {
            const NodeId u = ready.front();
            ready.pop();
            wave.push_back(u);
            ++visited;

            for (NodeId v : adj_[u]) {
                if (--in_degree[v] == 0) {
                    ready.push(v);
                }
            }
        }
        levels.push_back(std::move(wave));
    }

    if (visited != n) {
        throw CycleDetectedError(
            "TaskGraph contains a directed cycle; levelization aborted");
    }
    return levels;
}

CriticalPathResult TaskGraph::critical_path() const {
    const auto topo = topological_sort();
    CriticalPathResult result{};
    if (topo.empty()) {
        return result;
    }

    const std::size_t n = nodes_.size();
    constexpr NodeId kNoParent = std::numeric_limits<NodeId>::max();

    std::vector<double> dp(n, 0.0);
    std::vector<NodeId> parent(n, kNoParent);
    for (std::size_t u = 0; u < n; ++u) {
        dp[u] = nodes_[u].effective_weight();
    }

    for (NodeId u : topo) {
        for (NodeId v : adj_[u]) {
            const double candidate = dp[u] + nodes_[v].effective_weight();
            if (candidate > dp[v]) {
                dp[v] = candidate;
                parent[v] = u;
            }
        }
    }

    NodeId best_end = topo.front();
    for (NodeId u : topo) {
        if (dp[u] > dp[best_end]) {
            best_end = u;
        }
    }

    result.total_cost = dp[best_end];
    for (NodeId cur = best_end; cur != kNoParent; cur = parent[cur]) {
        result.path.push_back(cur);
        result.total_flops += nodes_[cur].estimated_flops;
        result.total_bytes += nodes_[cur].estimated_bytes;
    }
    std::reverse(result.path.begin(), result.path.end());
    return result;
}

std::string TaskGraph::to_dot(const CriticalPathResult* highlight_cp) const {
    std::unordered_set<NodeId> cp_nodes;
    std::unordered_set<std::uint64_t> cp_edges;
    const auto edge_key = [](NodeId u, NodeId v) -> std::uint64_t {
        return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
    };

    if (highlight_cp != nullptr) {
        for (std::size_t i = 0; i < highlight_cp->path.size(); ++i) {
            cp_nodes.insert(highlight_cp->path[i]);
            if (i + 1 < highlight_cp->path.size()) {
                cp_edges.insert(edge_key(highlight_cp->path[i], highlight_cp->path[i + 1]));
            }
        }
    }

    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed << std::setprecision(1);
    oss << "digraph TaskGraph {\n";
    oss << "  rankdir=LR;\n";
    oss << "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n";

    for (const auto& n : nodes_) {
        const bool on_cp = cp_nodes.contains(n.id);
        oss << "  n" << n.id << " [label=\"" << escape_dot_label(n.name)
            << "\\n[" << task_kind_to_string(n.kind)
            << " | stream " << n.preferred_stream << "]";
        if (n.estimated_flops > 0 || n.estimated_bytes > 0) {
            oss << "\\nFLOPs=" << n.estimated_flops << " Bytes=" << n.estimated_bytes;
        }
        oss << "\", fillcolor=\"" << dot_fill_color(n.kind) << "\"";
        if (on_cp) {
            oss << ", color=\"#c0392b\", penwidth=2.5";
        }
        oss << "];\n";
    }

    for (std::size_t u = 0; u < nodes_.size(); ++u) {
        for (NodeId v : adj_[u]) {
            const bool on_cp = cp_edges.contains(edge_key(u, v));
            oss << "  n" << u << " -> n" << v;
            if (on_cp) {
                oss << " [color=\"#c0392b\", penwidth=2.5]";
            }
            oss << ";\n";
        }
    }

    oss << "}\n";
    return oss.str();
}

StreamScheduler::StreamScheduler(std::size_t num_streams) {
    if (num_streams == 0) {
        throw std::invalid_argument("StreamScheduler requires at least 1 stream");
    }

    streams_.reserve(num_streams);
    try {
        for (std::size_t s = 0; s < num_streams; ++s) {
            auto worker = std::make_unique<StreamWorker>();
            worker->stream_id = s;
            StreamWorker* raw = worker.get();
            worker->thread = std::thread([this, raw]() { worker_loop(*raw); });
            streams_.push_back(std::move(worker));
        }
    } catch (...) {
        for (auto& worker : streams_) {
            {
                std::lock_guard<std::mutex> lock(worker->mutex);
                worker->stop = true;
            }
            worker->cv.notify_all();
        }
        for (auto& worker : streams_) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
        throw;
    }
}

StreamScheduler::~StreamScheduler() {
    for (auto& worker : streams_) {
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->stop = true;
        }
        worker->cv.notify_all();
    }
    for (auto& worker : streams_) {
        if (worker->thread.joinable()) {
            worker->thread.join();
        }
    }
}

void StreamScheduler::worker_loop(StreamWorker& worker) {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(worker.mutex);
            worker.cv.wait(lock, [&worker]() { return worker.stop || !worker.queue.empty(); });
            if (worker.stop && worker.queue.empty()) {
                return;
            }
            task = std::move(worker.queue.front());
            worker.queue.pop_front();
        }
        if (task) {
            task();
        }
    }
}

ExecutionReport StreamScheduler::execute(const TaskGraph& graph, TraceProfiler* profiler) {
    std::lock_guard<std::mutex> exec_lock(execute_mutex_);

    const auto topo = graph.topological_sort();
    const std::size_t n = graph.node_count();

    ExecutionReport report{};
    report.num_streams = streams_.size();
    report.node_start_us.assign(n, 0.0);
    report.node_end_us.assign(n, 0.0);
    report.node_stream.assign(n, 0);

    if (n == 0) {
        return report;
    }

    std::vector<std::shared_ptr<StreamEvent>> events;
    events.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        events.push_back(std::make_shared<StreamEvent>());
    }

    std::mutex error_mutex;
    std::exception_ptr first_error = nullptr;
    std::atomic<bool> aborted{false};

    using Clock = std::chrono::steady_clock;
    const auto wall_start = Clock::now();

    // Proof of deadlock freedom:
    // Enqueuing tasks onto per-stream FIFO queues strictly in topological order ensures
    // every edge in (DAG dependencies U intra-stream FIFO order) strictly increases the
    // node's topological rank. Hence the combined schedule graph is strictly acyclic.
    for (NodeId u : topo) {
        const TaskNode& task_node = graph.node(u);
        const std::size_t target_stream = task_node.preferred_stream % streams_.size();
        report.node_stream[u] = target_stream;

        std::vector<std::shared_ptr<StreamEvent>> wait_events;
        const auto& preds = graph.predecessors(u);
        wait_events.reserve(preds.size());
        for (NodeId p : preds) {
            wait_events.push_back(events[p]);
        }

        auto done_event = events[u];
        auto& worker = *streams_[target_stream];

        {
            std::lock_guard<std::mutex> lock(worker.mutex);
            worker.queue.emplace_back([&, u, target_stream,
                                       wait_events = std::move(wait_events),
                                       done_event = std::move(done_event)]() {
                for (const auto& ev : wait_events) {
                    ev->wait();
                }

                if (!aborted.load(std::memory_order_acquire)) {
                    const TaskNode& node_ref = graph.node(u);
                    const auto t0 = Clock::now();
                    try {
                        ScopedTraceEvent trace(profiler,
                                               node_ref.name,
                                               task_kind_to_string(node_ref.kind),
                                               node_ref.estimated_flops,
                                               node_ref.estimated_bytes,
                                               static_cast<std::uint64_t>(target_stream));
                        if (node_ref.work) {
                            node_ref.work();
                        }
                    } catch (...) {
                        std::lock_guard<std::mutex> err_lock(error_mutex);
                        if (!first_error) {
                            first_error = std::current_exception();
                        }
                        aborted.store(true, std::memory_order_release);
                    }
                    const auto t1 = Clock::now();
                    report.node_start_us[u] =
                        std::chrono::duration<double, std::micro>(t0 - wall_start).count();
                    report.node_end_us[u] =
                        std::chrono::duration<double, std::micro>(t1 - wall_start).count();
                }

                done_event->signal();
            });
        }
        worker.cv.notify_one();
    }

    for (const auto& ev : events) {
        ev->wait();
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    const auto wall_end = Clock::now();
    report.total_wall_time_us =
        std::chrono::duration<double, std::micro>(wall_end - wall_start).count();
    report.tasks_executed = n;

    double sum_us = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_us += std::max(0.0, report.node_end_us[i] - report.node_start_us[i]);
    }
    report.sum_task_time_us = sum_us;
    if (report.total_wall_time_us > 0.0) {
        report.parallel_speedup = sum_us / report.total_wall_time_us;
    }

    return report;
}

}  // namespace gpuruntime
