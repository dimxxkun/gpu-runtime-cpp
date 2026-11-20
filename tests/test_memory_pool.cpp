#include "gpuruntime/memory_pool.hpp"
#include "test_common.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

void test_alignment_guarantees() {
    gpuruntime::DeviceMemoryPool pool(1024ULL * 1024ULL, 64, gpuruntime::MemoryKind::PinnedHost);

    const std::array<std::size_t, 3> alignments = {64, 256, 4096};
    std::vector<gpuruntime::BufferHandle> handles;
    handles.reserve(30);

    for (std::size_t iter = 0; iter < 10; ++iter) {
        for (std::size_t align : alignments) {
            const std::size_t req_size = 100 + iter * 37;
            auto buf = pool.allocate(req_size, align);
            TEST_CHECK(static_cast<bool>(buf));
            const auto addr = reinterpret_cast<std::uintptr_t>(buf.data());
            TEST_CHECK_EQ(addr % align, std::size_t{0});
            std::memset(buf.data(), static_cast<int>((iter + 1) & 0xFF), req_size);
            handles.push_back(std::move(buf));
        }
    }

    handles.clear();

    const auto st = pool.stats();
    TEST_CHECK_EQ(st.active_allocations, std::size_t{0});
    TEST_CHECK_EQ(st.allocated_bytes, std::size_t{0});
    TEST_CHECK_EQ(st.free_bytes, pool.capacity());
    TEST_CHECK_EQ(st.free_block_count, std::size_t{1});
    TEST_CHECK_EQ(st.largest_free_block_bytes, pool.capacity());
    TEST_CHECK_NEAR(st.fragmentation_ratio, 0.0, 1e-12);
}

void test_coalescing_full_recovery() {
    constexpr std::size_t kBlockSize = 4096;
    constexpr std::size_t kCount = 16;
    constexpr std::size_t kCapacity = kBlockSize * kCount;

    gpuruntime::DeviceMemoryPool pool(kCapacity, 256);

    std::vector<gpuruntime::PooledAllocation> blocks;
    blocks.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        blocks.push_back(pool.allocate(kBlockSize, 256));
    }

    // Pool is 100% full; another allocation must fail.
    TEST_CHECK(!pool.try_allocate(256).has_value());

    // Free even blocks first to induce checkerboard fragmentation.
    for (std::size_t i = 0; i < kCount; i += 2) {
        blocks[i].reset();
    }
    const auto mid_stats = pool.stats();
    TEST_CHECK_EQ(mid_stats.free_block_count, kCount / 2);
    TEST_CHECK(mid_stats.fragmentation_ratio > 0.5);

    // Free odd blocks; bidirectional coalescing must restore a single contiguous free block.
    for (std::size_t i = 1; i < kCount; i += 2) {
        blocks[i].reset();
    }

    const auto end_stats = pool.stats();
    TEST_CHECK_EQ(end_stats.active_allocations, std::size_t{0});
    TEST_CHECK_EQ(end_stats.free_block_count, std::size_t{1});
    TEST_CHECK_EQ(end_stats.largest_free_block_bytes, kCapacity);
    TEST_CHECK_NEAR(end_stats.fragmentation_ratio, 0.0, 1e-12);
    TEST_CHECK(end_stats.coalesce_merges >= kCount - 1);

    // Full-capacity single allocation must now succeed.
    auto whole = pool.allocate(kCapacity, 4096);
    TEST_CHECK(static_cast<bool>(whole));
    TEST_CHECK_EQ(reinterpret_cast<std::uintptr_t>(whole.data()) % 4096, std::size_t{0});
}

void test_oom_and_error_handling() {
    gpuruntime::DeviceMemoryPool pool(8192, 256);

    TEST_CHECK_THROWS(pool.allocate(16384), std::bad_alloc);
    TEST_CHECK(!pool.try_allocate(16384).has_value());
    TEST_CHECK_THROWS(pool.allocate(256, 300), std::invalid_argument);
    TEST_CHECK_THROWS(pool.allocate(256, 8192), std::invalid_argument);

    auto a = pool.allocate(4096);
    auto b = pool.allocate(4096);
    TEST_CHECK_THROWS(pool.allocate(256), std::bad_alloc);

    // Move semantics transfer ownership cleanly.
    gpuruntime::PooledAllocation moved_a(std::move(a));
    TEST_CHECK(!static_cast<bool>(a));
    TEST_CHECK(static_cast<bool>(moved_a));

    moved_a.reset();
    auto c = pool.allocate(4096);
    TEST_CHECK(static_cast<bool>(c));

    // Double free via raw API throws std::invalid_argument.
    void* raw = pool.allocate_raw(256);
    TEST_CHECK(raw == nullptr);  // Still full (b + c = 8192)
    b.reset();
    raw = pool.allocate_raw(256);
    TEST_CHECK(raw != nullptr);
    pool.deallocate_raw(raw);
    TEST_CHECK_THROWS(pool.deallocate_raw(raw), std::invalid_argument);

    // Zero-byte allocation returns empty handle without throwing.
    auto empty = pool.allocate(0);
    TEST_CHECK(!static_cast<bool>(empty));
}

void test_randomized_canary_stress() {
    constexpr std::size_t kPoolBytes = 2ULL * 1024ULL * 1024ULL;
    gpuruntime::DeviceMemoryPool pool(kPoolBytes, 256);

    struct LiveEntry {
        gpuruntime::PooledAllocation buf;
        std::size_t size = 0;
        std::uint8_t canary = 0;
    };

    std::mt19937 rng(1337);
    std::vector<LiveEntry> live;
    const std::array<std::size_t, 3> aligns = {64, 256, 4096};

    auto verify_entry = [](const LiveEntry& entry) {
        const auto* bytes = entry.buf.as<const std::uint8_t>();
        for (std::size_t i = 0; i < entry.size; ++i) {
            const auto expected = static_cast<std::uint8_t>((entry.canary + (i & 0x3F)) & 0xFF);
            if (bytes[i] != expected) {
                TEST_CHECK_EQ(static_cast<int>(bytes[i]), static_cast<int>(expected));
            }
        }
    };

    for (std::size_t step = 0; step < 2000; ++step) {
        const bool should_alloc = live.empty() || (live.size() < 64 && (rng() % 100) < 60);
        if (should_alloc) {
            const std::size_t sz = 64 + (rng() % 32768);
            const std::size_t al = aligns[rng() % aligns.size()];
            auto maybe_buf = pool.try_allocate(sz, al);
            if (maybe_buf.has_value()) {
                TEST_CHECK_EQ(reinterpret_cast<std::uintptr_t>(maybe_buf->data()) % al,
                              std::size_t{0});
                const auto canary = static_cast<std::uint8_t>((step * 13 + 7) & 0xFF);
                auto* bytes = maybe_buf->as<std::uint8_t>();
                for (std::size_t i = 0; i < sz; ++i) {
                    bytes[i] = static_cast<std::uint8_t>((canary + (i & 0x3F)) & 0xFF);
                }
                live.push_back(LiveEntry{std::move(*maybe_buf), sz, canary});
            }
        } else {
            const std::size_t idx = rng() % live.size();
            verify_entry(live[idx]);
            live[idx] = std::move(live.back());
            live.pop_back();
        }
    }

    for (const auto& entry : live) {
        verify_entry(entry);
    }
    live.clear();

    const auto st = pool.stats();
    TEST_CHECK_EQ(st.active_allocations, std::size_t{0});
    TEST_CHECK_EQ(st.free_block_count, std::size_t{1});
    TEST_CHECK_EQ(st.largest_free_block_bytes, kPoolBytes);
    TEST_CHECK_NEAR(st.fragmentation_ratio, 0.0, 1e-12);
    TEST_CHECK(st.cache_hit_rate > 0.0);
}

void test_concurrent_multithread_stress() {
    constexpr std::size_t kPoolBytes = 8ULL * 1024ULL * 1024ULL;
    constexpr std::size_t kThreads = 4;
    constexpr std::size_t kItersPerThread = 500;

    gpuruntime::DeviceMemoryPool pool(kPoolBytes, 256);
    std::atomic<bool> corruption_detected{false};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t tid = 0; tid < kThreads; ++tid) {
        threads.emplace_back([&, tid]() {
            std::mt19937 rng(static_cast<std::uint32_t>(100 + tid * 997));
            std::vector<std::pair<gpuruntime::PooledAllocation, std::uint8_t>> local;
            local.reserve(16);

            for (std::size_t i = 0; i < kItersPerThread; ++i) {
                if (local.size() < 12 && (local.empty() || (rng() % 2 == 0))) {
                    const std::size_t sz = 256 + (rng() % 16384);
                    const std::size_t al = ((rng() % 4) == 0) ? 4096 : 256;
                    auto maybe_buf = pool.try_allocate(sz, al);
                    if (maybe_buf.has_value()) {
                        if (reinterpret_cast<std::uintptr_t>(maybe_buf->data()) % al != 0) {
                            corruption_detected.store(true, std::memory_order_relaxed);
                        }
                        const auto tag =
                            static_cast<std::uint8_t>(((tid + 1) * 31 + i) & 0xFF);
                        std::memset(maybe_buf->data(), tag, sz);
                        local.emplace_back(std::move(*maybe_buf), tag);
                    }
                } else {
                    const std::size_t idx = rng() % local.size();
                    const auto& [buf, tag] = local[idx];
                    const auto* ptr = buf.as<const std::uint8_t>();
                    for (std::size_t b = 0; b < buf.size(); ++b) {
                        if (ptr[b] != tag) {
                            corruption_detected.store(true, std::memory_order_relaxed);
                            break;
                        }
                    }
                    local[idx] = std::move(local.back());
                    local.pop_back();
                }
            }
            local.clear();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    TEST_CHECK(!corruption_detected.load());
    const auto st = pool.stats();
    TEST_CHECK_EQ(st.active_allocations, std::size_t{0});
    TEST_CHECK_EQ(st.allocated_bytes, std::size_t{0});
    TEST_CHECK_EQ(st.free_block_count, std::size_t{1});
    TEST_CHECK_EQ(st.largest_free_block_bytes, kPoolBytes);
}

}  // namespace

int run_memory_pool_tests() {
    gpuruntime_test::SuiteRunner runner{"memory_pool"};
    runner.run("alignment_guarantees", test_alignment_guarantees);
    runner.run("coalescing_full_recovery", test_coalescing_full_recovery);
    runner.run("oom_and_error_handling", test_oom_and_error_handling);
    runner.run("randomized_canary_stress", test_randomized_canary_stress);
    runner.run("concurrent_multithread_stress", test_concurrent_multithread_stress);
    return runner.exit_code();
}
