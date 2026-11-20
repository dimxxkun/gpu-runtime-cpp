#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gpuruntime {

class TraceProfiler;

using NodeId = std::size_t;

/// Classifies heterogeneous runtime operations scheduled across GPU/host streams.
enum class TaskKind : std::uint8_t {
    ComputeKernel = 0,
    TransferH2D = 1,
    TransferD2H = 2,
    HostCallback = 3,
};

[[nodiscard]] const char* task_kind_to_string(TaskKind kind) noexcept;

/// Thrown when a cycle is detected in a TaskGraph during validation or scheduling.
class CycleDetectedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Descriptor for a single node in a heterogeneous DAG execution graph.
struct TaskNode {
    NodeId id = 0;
    std::string name;
    TaskKind kind = TaskKind::ComputeKernel;
    std::size_t preferred_stream = 0;
    std::uint64_t estimated_flops = 0;
    std::uint64_t estimated_bytes = 0;
    double estimated_cost_us = 0.0;
    std::function<void()> work;

    [[nodiscard]] double effective_weight() const noexcept;
};

/// Result of longest-weighted-path (critical path) analysis on a TaskGraph.
struct CriticalPathResult {
    std::vector<NodeId> path;
    double total_cost = 0.0;
    std::uint64_t total_flops = 0;
    std::uint64_t total_bytes = 0;
};

/// Directed Acyclic Graph representing compute kernels, DMA transfers, and host callbacks.
class TaskGraph {
public:
    TaskGraph() = default;

    NodeId add_node(std::string name,
                    TaskKind kind,
                    std::size_t preferred_stream,
                    std::function<void()> work,
                    std::uint64_t estimated_flops = 0,
                    std::uint64_t estimated_bytes = 0,
                    double estimated_cost_us = 0.0);

    NodeId add_node(TaskNode spec);

    /// Adds a directed dependency edge `producer -> consumer`.
    void add_dependency(NodeId producer, NodeId consumer);

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] const TaskNode& node(NodeId id) const;
    [[nodiscard]] const std::vector<NodeId>& successors(NodeId id) const;
    [[nodiscard]] const std::vector<NodeId>& predecessors(NodeId id) const;

    /// Returns true if the graph contains at least one directed cycle.
    [[nodiscard]] bool has_cycle() const;

    /// Kahn's topological sort; returns std::nullopt if the graph contains a cycle.
    [[nodiscard]] std::optional<std::vector<NodeId>> try_topological_sort() const;

    /// Kahn's topological sort; throws CycleDetectedError if the graph contains a cycle.
    [[nodiscard]] std::vector<NodeId> topological_sort() const;

    /// Partitions the DAG into parallelizable Kahn levels (waves).
    [[nodiscard]] std::vector<std::vector<NodeId>> levelize() const;

    /// Computes the weighted critical path across the DAG.
    [[nodiscard]] CriticalPathResult critical_path() const;

    /// Exports the task graph in Graphviz DOT format with optional critical-path highlighting.
    [[nodiscard]] std::string to_dot(const CriticalPathResult* highlight_cp = nullptr) const;

private:
    void validate_node_id(NodeId id) const;

    std::vector<TaskNode> nodes_;
    std::vector<std::vector<NodeId>> adj_;
    std::vector<std::vector<NodeId>> rev_adj_;
};

/// Execution telemetry returned by StreamScheduler::execute().
struct ExecutionReport {
    double total_wall_time_us = 0.0;
    double sum_task_time_us = 0.0;
    double parallel_speedup = 1.0;
    std::size_t tasks_executed = 0;
    std::size_t num_streams = 0;
    std::vector<double> node_start_us;
    std::vector<double> node_end_us;
    std::vector<std::size_t> node_stream;
};

/// Multi-stream virtual GPU scheduler with dedicated per-stream FIFO workers and
/// explicit cross-stream event synchronization.
class StreamScheduler {
public:
    explicit StreamScheduler(std::size_t num_streams);
    ~StreamScheduler();

    StreamScheduler(const StreamScheduler&) = delete;
    StreamScheduler& operator=(const StreamScheduler&) = delete;
    StreamScheduler(StreamScheduler&&) = delete;
    StreamScheduler& operator=(StreamScheduler&&) = delete;

    /// Validates and executes a TaskGraph across virtual streams, optionally emitting
    /// per-stream trace spans to `profiler`.
    ExecutionReport execute(const TaskGraph& graph, TraceProfiler* profiler = nullptr);

    [[nodiscard]] std::size_t num_streams() const noexcept { return streams_.size(); }

private:
    struct StreamWorker {
        std::size_t stream_id = 0;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<std::function<void()>> queue;
        bool stop = false;
        std::thread thread;
    };

    void worker_loop(StreamWorker& worker);

    std::vector<std::unique_ptr<StreamWorker>> streams_;
    std::mutex execute_mutex_;
};

}  // namespace gpuruntime
