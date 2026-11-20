#include "gpuruntime/kernels.hpp"
#include "gpuruntime/memory_pool.hpp"
#include "gpuruntime/profiler.hpp"
#include "gpuruntime/ring_buffer.hpp"
#include "gpuruntime/task_graph.hpp"
#include "gpuruntime/tensor_view.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void bench_memory_pool(bool quick, gpuruntime::TraceProfiler& profiler) {
    std::cout << "\n[1/4] DeviceMemoryPool vs. System Aligned Allocator (256B alignment)\n";
    std::cout << std::string(72, '-') << "\n";

    constexpr std::size_t kPoolBytes = 32ULL * 1024ULL * 1024ULL;
    const std::size_t iterations = quick ? 20000 : 120000;
    constexpr std::size_t kWindow = 64;
    constexpr std::size_t kSizes[8] = {256, 512, 1024, 4096, 8192, 16384, 32768, 65536};

    std::mt19937 rng(42);
    std::vector<std::size_t> req_sizes(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        req_sizes[i] = kSizes[rng() % 8];
    }

    // Benchmark DeviceMemoryPool
    gpuruntime::DeviceMemoryPool pool(kPoolBytes, 256, gpuruntime::MemoryKind::DeviceSimulated);
    std::vector<void*> live_pool(kWindow, nullptr);

    const auto pool_t0 = Clock::now();
    {
        gpuruntime::ScopedTraceEvent trace(&profiler, "bench_memory_pool", "allocator");
        for (std::size_t i = 0; i < iterations; ++i) {
            const std::size_t slot = i % kWindow;
            if (live_pool[slot] != nullptr) {
                pool.deallocate_raw(live_pool[slot]);
                live_pool[slot] = nullptr;
            }
            live_pool[slot] = pool.allocate_raw(req_sizes[i], 256);
        }
        for (void*& ptr : live_pool) {
            if (ptr != nullptr) {
                pool.deallocate_raw(ptr);
                ptr = nullptr;
            }
        }
    }
    const auto pool_t1 = Clock::now();
    const double pool_ms = elapsed_ms(pool_t0, pool_t1);
    const auto pool_stats = pool.stats();

    // Benchmark system aligned operator new/delete
    std::vector<void*> live_sys(kWindow, nullptr);
    std::vector<std::size_t> live_sys_sz(kWindow, 0);

    const auto sys_t0 = Clock::now();
    {
        gpuruntime::ScopedTraceEvent trace(&profiler, "bench_system_aligned_new", "allocator");
        for (std::size_t i = 0; i < iterations; ++i) {
            const std::size_t slot = i % kWindow;
            if (live_sys[slot] != nullptr) {
                ::operator delete(live_sys[slot], live_sys_sz[slot], std::align_val_t{256});
                live_sys[slot] = nullptr;
                live_sys_sz[slot] = 0;
            }
            live_sys[slot] = ::operator new(req_sizes[i], std::align_val_t{256});
            live_sys_sz[slot] = req_sizes[i];
        }
        for (std::size_t slot = 0; slot < kWindow; ++slot) {
            if (live_sys[slot] != nullptr) {
                ::operator delete(live_sys[slot], live_sys_sz[slot], std::align_val_t{256});
                live_sys[slot] = nullptr;
                live_sys_sz[slot] = 0;
            }
        }
    }
    const auto sys_t1 = Clock::now();
    const double sys_ms = elapsed_ms(sys_t0, sys_t1);

    const double pool_mops = (static_cast<double>(iterations) / (pool_ms * 1e-3)) * 1e-6;
    const double sys_mops = (static_cast<double>(iterations) / (sys_ms * 1e-3)) * 1e-6;
    const double speedup = (pool_ms > 0.0) ? (sys_ms / pool_ms) : 0.0;

    std::cout << std::fixed << std::setprecision(2)
              << "  Iterations          : " << iterations << " alloc+free pairs\n"
              << "  DeviceMemoryPool    : " << pool_ms << " ms (" << pool_mops << " M pairs/s)\n"
              << "  System aligned new  : " << sys_ms << " ms (" << sys_mops << " M pairs/s)\n"
              << "  Speedup             : " << speedup << "x\n"
              << "  Fast-path hit rate  : " << (pool_stats.cache_hit_rate * 100.0) << "%\n"
              << "  Coalesce merges     : " << pool_stats.coalesce_merges << "\n"
              << "  Peak watermark      : " << (pool_stats.peak_allocated_bytes / 1024) << " KiB\n"
              << "  Post-free frag ratio: " << pool_stats.fragmentation_ratio
              << " (free blocks=" << pool_stats.free_block_count << ")\n";
}

void bench_ring_buffer(bool quick, gpuruntime::TraceProfiler& profiler) {
    std::cout << "\n[2/4] Lock-Free SPSC Command Queue Throughput (64B GpuCommand)\n";
    std::cout << std::string(72, '-') << "\n";

    constexpr std::size_t kQueueCap = 4096;
    const std::uint64_t total_msgs = quick ? 200000ULL : 2000000ULL;

    // Single-item SPSC benchmark
    {
        auto queue = std::make_unique<gpuruntime::LockFreeCommandQueue<gpuruntime::GpuCommand, kQueueCap>>();
        std::uint64_t consumer_checksum = 0;

        const auto t0 = Clock::now();
        {
            gpuruntime::ScopedTraceEvent trace(
                &profiler,
                "spsc_single_transfer",
                "ring_buffer",
                0,
                total_msgs * sizeof(gpuruntime::GpuCommand));

            std::thread producer([&]() {
                for (std::uint64_t i = 0; i < total_msgs; ++i) {
                    gpuruntime::GpuCommand cmd{};
                    cmd.opcode = gpuruntime::CommandOpcode::LaunchKernel;
                    cmd.sequence_id = i;
                    cmd.arg0 = i ^ 0x9e3779b97f4a7c15ULL;
                    while (!queue->try_push(cmd)) {
                        std::this_thread::yield();
                    }
                }
            });

            std::thread consumer([&]() {
                gpuruntime::GpuCommand cmd{};
                std::uint64_t received = 0;
                while (received < total_msgs) {
                    if (queue->try_pop(cmd)) {
                        consumer_checksum += cmd.sequence_id;
                        ++received;
                    } else {
                        std::this_thread::yield();
                    }
                }
            });

            producer.join();
            consumer.join();
        }
        const auto t1 = Clock::now();
        const double ms = elapsed_ms(t0, t1);
        const double mops = (static_cast<double>(total_msgs) / (ms * 1e-3)) * 1e-6;
        const std::uint64_t expected = (total_msgs * (total_msgs - 1)) / 2;

        std::cout << std::fixed << std::setprecision(2)
                  << "  Single-item mode    : " << total_msgs << " cmds in " << ms
                  << " ms (" << mops << " M ops/s, integrity="
                  << (consumer_checksum == expected ? "OK" : "FAIL") << ")\n";
    }

    // Batch-32 SPSC benchmark
    {
        auto queue = std::make_unique<gpuruntime::LockFreeCommandQueue<gpuruntime::GpuCommand, kQueueCap>>();
        constexpr std::size_t kBatch = 32;
        std::uint64_t consumer_checksum = 0;

        const auto t0 = Clock::now();
        {
            gpuruntime::ScopedTraceEvent trace(
                &profiler,
                "spsc_batch32_transfer",
                "ring_buffer",
                0,
                total_msgs * sizeof(gpuruntime::GpuCommand));

            std::thread producer([&]() {
                std::array<gpuruntime::GpuCommand, kBatch> batch{};
                std::uint64_t sent = 0;
                while (sent < total_msgs) {
                    const std::size_t chunk =
                        static_cast<std::size_t>(std::min<std::uint64_t>(kBatch, total_msgs - sent));
                    for (std::size_t i = 0; i < chunk; ++i) {
                        batch[i].opcode = gpuruntime::CommandOpcode::LaunchKernel;
                        batch[i].sequence_id = sent + i;
                    }
                    std::size_t offset = 0;
                    while (offset < chunk) {
                        const std::size_t n =
                            queue->try_push_batch(std::span<const gpuruntime::GpuCommand>(
                                batch.data() + offset, chunk - offset));
                        offset += n;
                        if (n == 0) {
                            std::this_thread::yield();
                        }
                    }
                    sent += chunk;
                }
            });

            std::thread consumer([&]() {
                std::array<gpuruntime::GpuCommand, kBatch> batch{};
                std::uint64_t received = 0;
                while (received < total_msgs) {
                    const std::size_t want =
                        static_cast<std::size_t>(std::min<std::uint64_t>(kBatch, total_msgs - received));
                    const std::size_t n =
                        queue->try_pop_batch(std::span<gpuruntime::GpuCommand>(batch.data(), want));
                    if (n > 0) {
                        for (std::size_t i = 0; i < n; ++i) {
                            consumer_checksum += batch[i].sequence_id;
                        }
                        received += n;
                    } else {
                        std::this_thread::yield();
                    }
                }
            });

            producer.join();
            consumer.join();
        }
        const auto t1 = Clock::now();
        const double ms = elapsed_ms(t0, t1);
        const double mops = (static_cast<double>(total_msgs) / (ms * 1e-3)) * 1e-6;
        const std::uint64_t expected = (total_msgs * (total_msgs - 1)) / 2;

        std::cout << std::fixed << std::setprecision(2)
                  << "  Batch-32 mode       : " << total_msgs << " cmds in " << ms
                  << " ms (" << mops << " M ops/s, integrity="
                  << (consumer_checksum == expected ? "OK" : "FAIL") << ")\n";
    }
}

void bench_task_graph_scheduler(bool quick, gpuruntime::TraceProfiler& profiler) {
    std::cout << "\n[3/4] Multi-Stream Heterogeneous DAG Scheduler Scaling (1..4 streams)\n";
    std::cout << std::string(72, '-') << "\n";

    constexpr std::size_t kBranches = 4;
    const std::size_t dim = quick ? 96 : 160;
    const std::size_t elems = dim * dim;

    std::vector<float> host_in(elems, 1.0f);
    std::vector<std::vector<float>> dev_a(kBranches, std::vector<float>(elems, 0.0f));
    std::vector<std::vector<float>> dev_b(kBranches, std::vector<float>(elems, 0.5f));
    std::vector<std::vector<float>> dev_c(kBranches, std::vector<float>(elems, 0.0f));
    std::vector<float> final_out(elems, 0.0f);

    gpuruntime::TaskGraph graph;
    const auto root = graph.add_node(
        "prepare_inputs",
        gpuruntime::TaskKind::HostCallback,
        0,
        [&]() {
            for (std::size_t i = 0; i < elems; ++i) {
                host_in[i] = static_cast<float>((i % 17) + 1) * 0.05f;
            }
        },
        0,
        elems * sizeof(float));

    std::vector<gpuruntime::NodeId> gemm_nodes;
    gemm_nodes.reserve(kBranches);

    for (std::size_t b = 0; b < kBranches; ++b) {
        const auto h2d = graph.add_node(
            "h2d_branch_" + std::to_string(b),
            gpuruntime::TaskKind::TransferH2D,
            b,
            [&, b]() { std::memcpy(dev_a[b].data(), host_in.data(), elems * sizeof(float)); },
            0,
            elems * sizeof(float));

        const auto gemm = graph.add_node(
            "sgemm_branch_" + std::to_string(b),
            gpuruntime::TaskKind::ComputeKernel,
            b,
            [&, b, dim]() {
                gpuruntime::TensorView<const float> va(dev_a[b].data(), {dim, dim});
                gpuruntime::TensorView<const float> vb(dev_b[b].data(), {dim, dim});
                gpuruntime::TensorView<float> vc(dev_c[b].data(), {dim, dim});
                gpuruntime::sgemm(va, vb, vc, 1.0f, 0.0f);
            },
            gpuruntime::gemm_flop_count(dim, dim, dim),
            3ULL * elems * sizeof(float));

        graph.add_dependency(root, h2d);
        graph.add_dependency(h2d, gemm);
        gemm_nodes.push_back(gemm);
    }

    const auto reduce = graph.add_node(
        "reduce_and_d2h",
        gpuruntime::TaskKind::TransferD2H,
        0,
        [&]() {
            std::fill(final_out.begin(), final_out.end(), 0.0f);
            for (std::size_t b = 0; b < kBranches; ++b) {
                for (std::size_t i = 0; i < elems; ++i) {
                    final_out[i] += dev_c[b][i];
                }
            }
        },
        kBranches * elems,
        (kBranches + 1) * elems * sizeof(float));

    for (gpuruntime::NodeId g_id : gemm_nodes) {
        graph.add_dependency(g_id, reduce);
    }

    const auto cp = graph.critical_path();
    std::cout << "  DAG nodes=" << graph.node_count()
              << ", levels=" << graph.levelize().size()
              << ", critical path length=" << cp.path.size() << " nodes\n";

    double baseline_us = 0.0;
    for (std::size_t streams : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        gpuruntime::StreamScheduler scheduler(streams);
        // Only record trace spans for the 4-stream run so Chrome Trace shows full concurrency.
        gpuruntime::TraceProfiler* prof = (streams == 4) ? &profiler : nullptr;
        const auto report = scheduler.execute(graph, prof);
        if (streams == 1) {
            baseline_us = report.total_wall_time_us;
        }
        const double wall_speedup =
            (report.total_wall_time_us > 0.0) ? (baseline_us / report.total_wall_time_us) : 1.0;

        std::cout << std::fixed << std::setprecision(2)
                  << "  " << streams << " stream(s)         : wall="
                  << (report.total_wall_time_us * 1e-3) << " ms"
                  << " | sum_tasks=" << (report.sum_task_time_us * 1e-3) << " ms"
                  << " | scaling vs 1-stream=" << wall_speedup << "x\n";
    }
}

void bench_sgemm_roofline(bool quick, gpuruntime::TraceProfiler& profiler) {
    std::cout << "\n[4/4] Cache-Blocked L1/L2 Tiled SGEMM Roofline ("
              << gpuruntime::active_simd_backend() << " backend)\n";
    std::cout << std::string(72, '-') << "\n";

    const std::size_t m = quick ? 256 : 512;
    const std::size_t n = quick ? 256 : 512;
    const std::size_t k = quick ? 256 : 512;
    const std::size_t runs = quick ? 3 : 5;

    gpuruntime::DeviceMemoryPool pool(16ULL * 1024ULL * 1024ULL, 256);
    auto buf_a = pool.allocate(m * k * sizeof(float), 256);
    auto buf_b = pool.allocate(k * n * sizeof(float), 256);
    auto buf_c = pool.allocate(m * n * sizeof(float), 256);

    float* ptr_a = buf_a.as<float>();
    float* ptr_b = buf_b.as<float>();
    float* ptr_c = buf_c.as<float>();

    for (std::size_t i = 0; i < m * k; ++i) {
        ptr_a[i] = static_cast<float>((i % 31) - 15) * 0.05f;
    }
    for (std::size_t i = 0; i < k * n; ++i) {
        ptr_b[i] = static_cast<float>((i % 29) - 14) * 0.05f;
    }

    gpuruntime::TensorView<const float> view_a(ptr_a, {m, k});
    gpuruntime::TensorView<const float> view_b(ptr_b, {k, n});
    gpuruntime::TensorView<float> view_c(ptr_c, {m, n});

    const std::uint64_t flops = gpuruntime::gemm_flop_count(m, n, k);
    const std::uint64_t bytes =
        (static_cast<std::uint64_t>(m) * k + static_cast<std::uint64_t>(k) * n +
         2ULL * static_cast<std::uint64_t>(m) * n) *
        sizeof(float);

    // Warm-up run
    gpuruntime::sgemm(view_a, view_b, view_c, 1.0f, 0.0f);

    double best_ms = 1e18;
    for (std::size_t r = 0; r < runs; ++r) {
        const auto t0 = Clock::now();
        {
            gpuruntime::ScopedTraceEvent trace(&profiler, "sgemm_tiled", "kernel", flops, bytes);
            gpuruntime::sgemm(view_a, view_b, view_c, 1.0f, 0.0f);
        }
        const auto t1 = Clock::now();
        best_ms = std::min(best_ms, elapsed_ms(t0, t1));
    }

    const double gflops = (static_cast<double>(flops) / (best_ms * 1e-3)) * 1e-9;
    const double gbps = (static_cast<double>(bytes) / (best_ms * 1e-3)) * 1e-9;

    std::cout << std::fixed << std::setprecision(2)
              << "  Matrix shape        : " << m << " x " << n << " x " << k << " (FP32)\n"
              << "  Best latency        : " << best_ms << " ms (" << runs << " runs)\n"
              << "  Throughput          : " << gflops << " GFLOP/s (" << gbps << " GB/s)\n"
              << "  Checksum C[0,0]     : " << view_c(0, 0) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    std::string trace_path = "trace.json";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quick") {
            quick = true;
        } else if (arg == "--trace" && (i + 1) < argc) {
            trace_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: gpuruntime_bench [--quick] [--trace <output.json>]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    std::cout << "========================================================================\n"
              << "  gpu-runtime-cpp Host-Side Runtime Benchmark Suite"
              << (quick ? " [QUICK MODE]" : "") << "\n"
              << "  SIMD Backend: " << gpuruntime::active_simd_backend()
              << " | Hardware Concurrency: " << std::thread::hardware_concurrency() << "\n"
              << "========================================================================\n";

    gpuruntime::TraceProfiler profiler;

    bench_memory_pool(quick, profiler);
    bench_ring_buffer(quick, profiler);
    bench_task_graph_scheduler(quick, profiler);
    bench_sgemm_roofline(quick, profiler);

    std::cout << "\nProfiler Summary Table:\n";
    std::cout << profiler.format_summary_table();

    profiler.write_chrome_trace_json(trace_path);
    std::cout << "\nChrome Trace written to: " << trace_path
              << " (open in chrome://tracing or https://ui.perfetto.dev)\n";
    return 0;
}
