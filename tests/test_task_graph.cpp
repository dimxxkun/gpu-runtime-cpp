#include "gpuruntime/profiler.hpp"
#include "gpuruntime/task_graph.hpp"
#include "test_common.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void test_diamond_dag_ordering() {
    gpuruntime::TaskGraph graph;

    std::atomic<int> step_counter{0};
    int a_done = -1;
    int b_start = -1;
    int b_done = -1;
    int c_start = -1;
    int c_done = -1;
    int d_start = -1;

    int val_a = 0;
    int val_b = 0;
    int val_c = 0;
    int val_d = 0;

    const auto a = graph.add_node("h2d_input", gpuruntime::TaskKind::TransferH2D, 0, [&]() {
        val_a = 10;
        a_done = step_counter.fetch_add(1, std::memory_order_acq_rel);
    });
    const auto b = graph.add_node("kernel_left", gpuruntime::TaskKind::ComputeKernel, 1, [&]() {
        b_start = step_counter.fetch_add(1, std::memory_order_acq_rel);
        val_b = val_a * 3;
        b_done = step_counter.fetch_add(1, std::memory_order_acq_rel);
    });
    const auto c = graph.add_node("kernel_right", gpuruntime::TaskKind::ComputeKernel, 2, [&]() {
        c_start = step_counter.fetch_add(1, std::memory_order_acq_rel);
        val_c = val_a + 7;
        c_done = step_counter.fetch_add(1, std::memory_order_acq_rel);
    });
    const auto d = graph.add_node("d2h_reduce", gpuruntime::TaskKind::TransferD2H, 0, [&]() {
        d_start = step_counter.fetch_add(1, std::memory_order_acq_rel);
        val_d = val_b + val_c;
    });

    graph.add_dependency(a, b);
    graph.add_dependency(a, c);
    graph.add_dependency(b, d);
    graph.add_dependency(c, d);

    const auto levels = graph.levelize();
    TEST_CHECK_EQ(levels.size(), std::size_t{3});
    TEST_CHECK_EQ(levels[0].size(), std::size_t{1});
    TEST_CHECK_EQ(levels[1].size(), std::size_t{2});
    TEST_CHECK_EQ(levels[2].size(), std::size_t{1});

    gpuruntime::StreamScheduler scheduler(3);
    const auto report = scheduler.execute(graph);

    TEST_CHECK_EQ(report.tasks_executed, std::size_t{4});
    TEST_CHECK_EQ(val_d, 47);
    TEST_CHECK(a_done < b_start);
    TEST_CHECK(a_done < c_start);
    TEST_CHECK(b_done < d_start);
    TEST_CHECK(c_done < d_start);
}

void test_cycle_detection() {
    // 2-node cycle
    {
        gpuruntime::TaskGraph g;
        const auto n0 = g.add_node("n0", gpuruntime::TaskKind::ComputeKernel, 0, nullptr);
        const auto n1 = g.add_node("n1", gpuruntime::TaskKind::ComputeKernel, 1, nullptr);
        g.add_dependency(n0, n1);
        g.add_dependency(n1, n0);

        TEST_CHECK(g.has_cycle());
        TEST_CHECK(!g.try_topological_sort().has_value());
        TEST_CHECK_THROWS(g.topological_sort(), gpuruntime::CycleDetectedError);
        TEST_CHECK_THROWS(g.levelize(), gpuruntime::CycleDetectedError);
        TEST_CHECK_THROWS(g.critical_path(), gpuruntime::CycleDetectedError);

        gpuruntime::StreamScheduler sched(2);
        TEST_CHECK_THROWS(sched.execute(g), gpuruntime::CycleDetectedError);
    }

    // 3-node cycle reachable from acyclic root
    {
        gpuruntime::TaskGraph g;
        const auto root = g.add_node("root", gpuruntime::TaskKind::HostCallback, 0, nullptr);
        const auto a = g.add_node("a", gpuruntime::TaskKind::ComputeKernel, 0, nullptr);
        const auto b = g.add_node("b", gpuruntime::TaskKind::ComputeKernel, 1, nullptr);
        const auto c = g.add_node("c", gpuruntime::TaskKind::ComputeKernel, 2, nullptr);
        g.add_dependency(root, a);
        g.add_dependency(a, b);
        g.add_dependency(b, c);
        g.add_dependency(c, a);

        TEST_CHECK(g.has_cycle());
        TEST_CHECK_THROWS(g.topological_sort(), gpuruntime::CycleDetectedError);
    }

    // Self-loop
    {
        gpuruntime::TaskGraph g;
        const auto self = g.add_node("self", gpuruntime::TaskKind::ComputeKernel, 0, nullptr);
        g.add_dependency(self, self);
        TEST_CHECK(g.has_cycle());
        TEST_CHECK_THROWS(g.topological_sort(), gpuruntime::CycleDetectedError);
    }
}

void test_parallel_multistream_execution() {
    constexpr std::size_t kStreams = 4;
    gpuruntime::TaskGraph graph;

    std::atomic<int> in_flight{0};
    std::atomic<int> peak_in_flight{0};
    std::vector<std::uint64_t> branch_results(kStreams, 0);
    std::uint64_t final_sum = 0;

    const auto root = graph.add_node("init", gpuruntime::TaskKind::TransferH2D, 0, nullptr);

    std::vector<gpuruntime::NodeId> workers;
    workers.reserve(kStreams);
    for (std::size_t s = 0; s < kStreams; ++s) {
        const auto w = graph.add_node(
            "stream_kernel_" + std::to_string(s),
            gpuruntime::TaskKind::ComputeKernel,
            s,
            [&, s]() {
                const int cur = in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
                int prev = peak_in_flight.load(std::memory_order_relaxed);
                while (cur > prev && !peak_in_flight.compare_exchange_weak(
                                         prev, cur, std::memory_order_relaxed)) {
                }

                // Short deterministic work + sleep so streams measurably overlap.
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                std::uint64_t acc = 0;
                for (std::uint64_t i = 1; i <= 1000; ++i) {
                    acc += i * (s + 1);
                }
                branch_results[s] = acc;

                in_flight.fetch_sub(1, std::memory_order_acq_rel);
            },
            2000,
            4096);
        graph.add_dependency(root, w);
        workers.push_back(w);
    }

    const auto join = graph.add_node("join_callback", gpuruntime::TaskKind::HostCallback, 0, [&]() {
        for (std::uint64_t v : branch_results) {
            final_sum += v;
        }
    });
    for (gpuruntime::NodeId w : workers) {
        graph.add_dependency(w, join);
    }

    gpuruntime::TraceProfiler profiler;
    gpuruntime::StreamScheduler scheduler(kStreams);
    const auto report = scheduler.execute(graph, &profiler);

    TEST_CHECK_EQ(report.tasks_executed, kStreams + 2);
    TEST_CHECK(peak_in_flight.load() >= 2);

    // Sum_{s=0..3} (s+1) * (1000*1001/2) = 10 * 500500 = 5005000
    TEST_CHECK_EQ(final_sum, std::uint64_t{5005000});
    TEST_CHECK_EQ(profiler.events().size(), kStreams + 2);

    // Exception propagation must unwind cleanly without deadlocking downstream waiters.
    gpuruntime::TaskGraph failing;
    const auto f0 = failing.add_node("fail_step", gpuruntime::TaskKind::ComputeKernel, 0, []() {
        throw std::runtime_error("simulated kernel launch failure");
    });
    const auto f1 = failing.add_node("downstream", gpuruntime::TaskKind::TransferD2H, 1, []() {});
    failing.add_dependency(f0, f1);
    TEST_CHECK_THROWS(scheduler.execute(failing), std::runtime_error);
}

void test_critical_path_and_dot_export() {
    gpuruntime::TaskGraph graph;

    const auto a = graph.add_node("upload", gpuruntime::TaskKind::TransferH2D, 0, nullptr, 0, 1024, 10.0);
    const auto b = graph.add_node("heavy_gemm", gpuruntime::TaskKind::ComputeKernel, 1, nullptr, 100000, 4096, 50.0);
    const auto c = graph.add_node("light_norm", gpuruntime::TaskKind::ComputeKernel, 2, nullptr, 5000, 1024, 20.0);
    const auto d = graph.add_node("download", gpuruntime::TaskKind::TransferD2H, 0, nullptr, 0, 1024, 15.0);

    graph.add_dependency(a, b);
    graph.add_dependency(a, c);
    graph.add_dependency(b, d);
    graph.add_dependency(c, d);

    const auto cp = graph.critical_path();
    TEST_CHECK_EQ(cp.path.size(), std::size_t{3});
    TEST_CHECK_EQ(cp.path[0], a);
    TEST_CHECK_EQ(cp.path[1], b);
    TEST_CHECK_EQ(cp.path[2], d);
    TEST_CHECK_NEAR(cp.total_cost, 75.0, 1e-9);
    TEST_CHECK_EQ(cp.total_flops, std::uint64_t{100000});
    TEST_CHECK_EQ(cp.total_bytes, std::uint64_t{6144});

    const std::string dot = graph.to_dot(&cp);
    TEST_CHECK(dot.find("digraph TaskGraph") != std::string::npos);
    TEST_CHECK(dot.find("heavy_gemm") != std::string::npos);
    TEST_CHECK(dot.find("n0 -> n1") != std::string::npos);
    TEST_CHECK(dot.find("n1 -> n3") != std::string::npos);
    TEST_CHECK(dot.find("#c0392b") != std::string::npos);
}

}  // namespace

int run_task_graph_tests() {
    gpuruntime_test::SuiteRunner runner{"task_graph"};
    runner.run("diamond_dag_ordering", test_diamond_dag_ordering);
    runner.run("cycle_detection", test_cycle_detection);
    runner.run("parallel_multistream_execution", test_parallel_multistream_execution);
    runner.run("critical_path_and_dot_export", test_critical_path_and_dot_export);
    return runner.exit_code();
}
