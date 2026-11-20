#include "gpuruntime/ring_buffer.hpp"
#include "test_common.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace {

void test_empty_full_and_wraparound() {
    gpuruntime::LockFreeCommandQueue<std::uint64_t, 8> q;
    TEST_CHECK(q.empty_approx());
    TEST_CHECK(!q.full_approx());
    TEST_CHECK_EQ(q.capacity(), std::size_t{8});

    std::uint64_t out = 0;
    TEST_CHECK(!q.try_pop(out));
    TEST_CHECK(!q.try_pop().has_value());

    // Fill to exact capacity (8 elements).
    for (std::uint64_t i = 0; i < 8; ++i) {
        TEST_CHECK(q.try_push(i + 100));
    }
    TEST_CHECK(q.full_approx());
    TEST_CHECK_EQ(q.size_approx(), std::size_t{8});
    TEST_CHECK(!q.try_push(999));

    // Pop 3, push 3, repeat across many wrap-arounds.
    std::uint64_t next_push = 108;
    std::uint64_t next_expect = 100;

    for (int round = 0; round < 200; ++round) {
        for (int k = 0; k < 3; ++k) {
            TEST_CHECK(q.try_pop(out));
            TEST_CHECK_EQ(out, next_expect++);
        }
        for (int k = 0; k < 3; ++k) {
            TEST_CHECK(q.try_push(next_push++));
        }
        TEST_CHECK(q.full_approx());
    }

    // Drain remaining 8 elements.
    for (int k = 0; k < 8; ++k) {
        auto val = q.try_pop();
        TEST_CHECK(val.has_value());
        TEST_CHECK_EQ(*val, next_expect++);
    }
    TEST_CHECK(q.empty_approx());
    TEST_CHECK(!q.try_pop(out));
}

std::atomic<int> g_tracked_live_objects{0};

struct TrackedItem {
    std::uint64_t id = 0;
    TrackedItem() noexcept { g_tracked_live_objects.fetch_add(1, std::memory_order_relaxed); }
    explicit TrackedItem(std::uint64_t v) noexcept : id(v) {
        g_tracked_live_objects.fetch_add(1, std::memory_order_relaxed);
    }
    TrackedItem(const TrackedItem& o) noexcept : id(o.id) {
        g_tracked_live_objects.fetch_add(1, std::memory_order_relaxed);
    }
    TrackedItem(TrackedItem&& o) noexcept : id(o.id) {
        g_tracked_live_objects.fetch_add(1, std::memory_order_relaxed);
    }
    TrackedItem& operator=(const TrackedItem&) noexcept = default;
    TrackedItem& operator=(TrackedItem&&) noexcept = default;
    ~TrackedItem() noexcept { g_tracked_live_objects.fetch_sub(1, std::memory_order_relaxed); }
};

void test_batch_operations_and_lifetime() {
    g_tracked_live_objects.store(0, std::memory_order_relaxed);

    {
        gpuruntime::LockFreeCommandQueue<TrackedItem, 16> q;

        std::vector<TrackedItem> src;
        src.reserve(20);
        for (std::uint64_t i = 0; i < 20; ++i) {
            src.emplace_back(i);
        }

        // Only 16 of 20 fit.
        const std::size_t pushed1 = q.try_push_batch(std::span<const TrackedItem>(src));
        TEST_CHECK_EQ(pushed1, std::size_t{16});

        // Pop 10.
        std::vector<TrackedItem> dst(10);
        const std::size_t popped1 = q.try_pop_batch(std::span<TrackedItem>(dst));
        TEST_CHECK_EQ(popped1, std::size_t{10});
        for (std::size_t i = 0; i < 10; ++i) {
            TEST_CHECK_EQ(dst[i].id, static_cast<std::uint64_t>(i));
        }

        // Push remaining 4 (wrapping across ring boundary).
        const std::size_t pushed2 =
            q.try_push_batch(std::span<const TrackedItem>(src.data() + 16, 4));
        TEST_CHECK_EQ(pushed2, std::size_t{4});

        // Leave 10 items in queue on destruction to verify destructor cleans up live slots.
        TEST_CHECK_EQ(q.size_approx(), std::size_t{10});
    }

    TEST_CHECK_EQ(g_tracked_live_objects.load(std::memory_order_relaxed), 0);
}

void test_multithreaded_one_million_spsc() {
    constexpr std::size_t kCapacity = 1024;
    constexpr std::uint64_t kTotalMessages = 1000000ULL;

    auto queue = std::make_unique<gpuruntime::LockFreeCommandQueue<gpuruntime::GpuCommand, kCapacity>>();
    std::atomic<bool> ordering_ok{true};
    std::atomic<std::uint64_t> received_count{0};

    std::thread producer([&]() {
        for (std::uint64_t seq = 0; seq < kTotalMessages; ++seq) {
            gpuruntime::GpuCommand cmd{};
            cmd.opcode = gpuruntime::CommandOpcode::LaunchKernel;
            cmd.stream_id = static_cast<std::uint32_t>(seq & 3U);
            cmd.sequence_id = seq;
            cmd.arg0 = seq * 2654435761ULL;
            cmd.arg1 = ~cmd.arg0;
            while (!queue->try_push(cmd)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        gpuruntime::GpuCommand cmd{};
        std::uint64_t expected_seq = 0;
        while (expected_seq < kTotalMessages) {
            if (queue->try_pop(cmd)) {
                const std::uint64_t expected_arg0 = expected_seq * 2654435761ULL;
                if (cmd.sequence_id != expected_seq ||
                    cmd.stream_id != static_cast<std::uint32_t>(expected_seq & 3U) ||
                    cmd.arg0 != expected_arg0 ||
                    cmd.arg1 != ~expected_arg0) {
                    ordering_ok.store(false, std::memory_order_relaxed);
                    break;
                }
                ++expected_seq;
            } else {
                std::this_thread::yield();
            }
        }
        received_count.store(expected_seq, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    TEST_CHECK(ordering_ok.load());
    TEST_CHECK_EQ(received_count.load(), kTotalMessages);
    TEST_CHECK(queue->empty_approx());
}

}  // namespace

int run_ring_buffer_tests() {
    gpuruntime_test::SuiteRunner runner{"ring_buffer"};
    runner.run("empty_full_and_wraparound", test_empty_full_and_wraparound);
    runner.run("batch_operations_and_lifetime", test_batch_operations_and_lifetime);
    runner.run("multithreaded_one_million_spsc", test_multithreaded_one_million_spsc);
    return runner.exit_code();
}
