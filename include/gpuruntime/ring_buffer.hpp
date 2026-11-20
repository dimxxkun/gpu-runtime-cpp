#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace gpuruntime {

/// Opcodes for host-to-GPU command submission packets.
enum class CommandOpcode : std::uint16_t {
    NoOp = 0,
    LaunchKernel = 1,
    CopyH2D = 2,
    CopyD2H = 3,
    RecordEvent = 4,
    WaitEvent = 5,
    HostCallback = 6,
    Fence = 7,
};

/// Compact 64-byte cache-line-aligned command descriptor submitted from host to GPU stream.
struct alignas(64) GpuCommand {
    CommandOpcode opcode = CommandOpcode::NoOp;
    std::uint16_t flags = 0;
    std::uint32_t stream_id = 0;
    std::uint64_t sequence_id = 0;
    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    std::uint64_t byte_count = 0;
    std::uint64_t arg0 = 0;
    std::uint64_t arg1 = 0;
    std::uint64_t reserved = 0;
};

static_assert(sizeof(GpuCommand) == 64, "GpuCommand should occupy exactly one 64-byte cache line");

/// Wait-free / lock-free Single-Producer Single-Consumer (SPSC) ring buffer.
///
/// Concurrency & memory ordering contract:
/// - Exactly ONE producer thread may invoke try_push / try_emplace / try_push_batch.
/// - Exactly ONE consumer thread may invoke try_pop / try_pop_batch.
/// - `head_` (write cursor) and `tail_` (read cursor) reside on separate 64-byte cache lines
///   alongside their thread-private cached shadow indices (`cached_tail_`, `cached_head_`)
///   to eliminate false sharing and minimize cross-core coherency traffic.
/// - Monotonically increasing unsigned indices with power-of-two masking allow all `Capacity`
///   slots to be utilized without a wasted sentinel slot.
template <typename T, std::size_t Capacity>
class LockFreeCommandQueue {
public:
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "LockFreeCommandQueue Capacity must be a power of two >= 2");
    static_assert(std::is_move_constructible_v<T> || std::is_copy_constructible_v<T>,
                  "LockFreeCommandQueue element type T must be movable or copyable");

    static constexpr std::size_t kCapacity = Capacity;
    static constexpr std::size_t kMask = Capacity - 1;
    static constexpr std::size_t kCacheLineSize = 64;

    LockFreeCommandQueue() noexcept = default;

    ~LockFreeCommandQueue() noexcept {
        // Safe during single-threaded teardown once producer & consumer have stopped.
        const std::size_t r = tail_.load(std::memory_order_relaxed);
        const std::size_t w = head_.load(std::memory_order_relaxed);
        for (std::size_t idx = r; idx != w; ++idx) {
            std::destroy_at(live_slot_ptr(idx & kMask));
        }
    }

    LockFreeCommandQueue(const LockFreeCommandQueue&) = delete;
    LockFreeCommandQueue& operator=(const LockFreeCommandQueue&) = delete;
    LockFreeCommandQueue(LockFreeCommandQueue&&) = delete;
    LockFreeCommandQueue& operator=(LockFreeCommandQueue&&) = delete;

    /// Enqueues a single item by constructing it in-place. Returns false if the queue is full.
    template <typename... Args>
    [[nodiscard]] bool try_emplace(Args&&... args) {
        // Producer is the sole writer to head_, so relaxed load is sufficient.
        const std::size_t w = head_.load(std::memory_order_relaxed);

        if ((w - cached_tail_) == Capacity) {
            // Acquire synchronizes-with consumer's release store to tail_, guaranteeing
            // the consumer has finished reading & destroying the recycled slot.
            cached_tail_ = tail_.load(std::memory_order_acquire);
            if ((w - cached_tail_) == Capacity) {
                return false;
            }
        }

        std::construct_at(raw_slot_ptr(w & kMask), std::forward<Args>(args)...);

        // Release publishes the fully constructed element in storage_[w & kMask]
        // before exposing the incremented head_ to the consumer.
        head_.store(w + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_push(const T& item) {
        return try_emplace(item);
    }

    [[nodiscard]] bool try_push(T&& item) {
        return try_emplace(std::move(item));
    }

    /// Amortized batch enqueue: commits all pushed items with a single atomic release store.
    /// Returns the number of elements actually enqueued (0 <= n <= items.size()).
    [[nodiscard]] std::size_t try_push_batch(std::span<const T> items) {
        if (items.empty()) {
            return 0;
        }

        const std::size_t w = head_.load(std::memory_order_relaxed);
        std::size_t available = Capacity - (w - cached_tail_);

        if (available < items.size()) {
            // Refresh consumer progress with acquire ordering to observe newly freed slots.
            cached_tail_ = tail_.load(std::memory_order_acquire);
            available = Capacity - (w - cached_tail_);
            if (available == 0) {
                return 0;
            }
        }

        const std::size_t count = std::min(items.size(), available);
        std::size_t constructed = 0;
        try {
            for (; constructed < count; ++constructed) {
                std::construct_at(raw_slot_ptr((w + constructed) & kMask), items[constructed]);
            }
        } catch (...) {
            for (std::size_t i = 0; i < constructed; ++i) {
                std::destroy_at(live_slot_ptr((w + i) & kMask));
            }
            throw;
        }

        // Single release store publishes the entire batch to the consumer.
        head_.store(w + count, std::memory_order_release);
        return count;
    }

    /// Dequeues a single item into `out`. Returns false if the queue is empty.
    [[nodiscard]] bool try_pop(T& out) {
        // Consumer is the sole writer to tail_, so relaxed load is sufficient.
        const std::size_t r = tail_.load(std::memory_order_relaxed);

        if (cached_head_ == r) {
            // Acquire synchronizes-with producer's release store to head_, guaranteeing
            // visibility of all writes performed during element construction.
            cached_head_ = head_.load(std::memory_order_acquire);
            if (cached_head_ == r) {
                return false;
            }
        }

        T* slot = live_slot_ptr(r & kMask);
        out = std::move(*slot);
        std::destroy_at(slot);

        // Release publishes completion of slot read + destruction so producer can safely reuse it.
        tail_.store(r + 1, std::memory_order_release);
        return true;
    }

    /// Convenience optional-returning pop.
    [[nodiscard]] std::optional<T> try_pop() {
        const std::size_t r = tail_.load(std::memory_order_relaxed);

        if (cached_head_ == r) {
            cached_head_ = head_.load(std::memory_order_acquire);
            if (cached_head_ == r) {
                return std::nullopt;
            }
        }

        T* slot = live_slot_ptr(r & kMask);
        std::optional<T> result(std::move(*slot));
        std::destroy_at(slot);

        tail_.store(r + 1, std::memory_order_release);
        return result;
    }

    /// Amortized batch dequeue: releases all consumed slots with a single atomic release store.
    /// Returns the number of elements actually dequeued (0 <= n <= out.size()).
    [[nodiscard]] std::size_t try_pop_batch(std::span<T> out) {
        if (out.empty()) {
            return 0;
        }

        const std::size_t r = tail_.load(std::memory_order_relaxed);
        std::size_t available = cached_head_ - r;

        if (available < out.size()) {
            // Refresh producer progress with acquire ordering to observe newly pushed slots.
            cached_head_ = head_.load(std::memory_order_acquire);
            available = cached_head_ - r;
            if (available == 0) {
                return 0;
            }
        }

        const std::size_t count = std::min(out.size(), available);
        std::size_t popped = 0;
        try {
            for (; popped < count; ++popped) {
                T* slot = live_slot_ptr((r + popped) & kMask);
                out[popped] = std::move(*slot);
                std::destroy_at(slot);
            }
        } catch (...) {
            if (popped > 0) {
                tail_.store(r + popped, std::memory_order_release);
            }
            throw;
        }

        // Single release store hands the entire batch of slots back to the producer.
        tail_.store(r + count, std::memory_order_release);
        return count;
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

    /// Approximate occupancy snapshot (racy under concurrent producer/consumer; diagnostic use).
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::size_t w = head_.load(std::memory_order_acquire);
        const std::size_t r = tail_.load(std::memory_order_acquire);
        return w - r;
    }

    [[nodiscard]] bool empty_approx() const noexcept { return size_approx() == 0; }
    [[nodiscard]] bool full_approx() const noexcept { return size_approx() == Capacity; }

private:
    struct alignas(alignof(T)) Slot {
        std::byte bytes[sizeof(T)];
    };

    [[nodiscard]] T* raw_slot_ptr(std::size_t slot_index) noexcept {
        return reinterpret_cast<T*>(storage_[slot_index].bytes);
    }

    [[nodiscard]] T* live_slot_ptr(std::size_t slot_index) noexcept {
        return std::launder(reinterpret_cast<T*>(storage_[slot_index].bytes));
    }

    // Producer-owned cache line
    alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};
    std::size_t cached_tail_{0};

    // Consumer-owned cache line
    alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};
    std::size_t cached_head_{0};

    // Ring buffer storage isolated on its own cache-line boundary
    alignas(std::max(kCacheLineSize, alignof(T))) std::array<Slot, Capacity> storage_;
};

}  // namespace gpuruntime
