#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace gpuruntime {

class DeviceMemoryPool;

/// Distinguishes simulated GPU device memory vs. host-side pinned staging memory.
enum class MemoryKind : std::uint8_t {
    DeviceSimulated = 0,
    PinnedHost = 1,
};

/// Configuration for DeviceMemoryPool initialization.
struct PoolConfig {
    std::size_t capacity_bytes = 64ULL * 1024ULL * 1024ULL;
    std::size_t default_alignment = 256;
    MemoryKind kind = MemoryKind::DeviceSimulated;
};

/// Telemetry snapshot for allocator occupancy, fragmentation, and cache hit rate.
struct PoolStats {
    std::size_t total_capacity_bytes = 0;
    std::size_t allocated_bytes = 0;
    std::size_t free_bytes = 0;
    std::size_t peak_allocated_bytes = 0;
    std::size_t largest_free_block_bytes = 0;
    std::size_t free_block_count = 0;
    std::size_t active_allocations = 0;
    std::uint64_t total_alloc_calls = 0;
    std::uint64_t total_free_calls = 0;
    std::uint64_t fast_path_hits = 0;
    std::uint64_t fallback_hits = 0;
    std::uint64_t coalesce_merges = 0;
    double fragmentation_ratio = 0.0;
    double cache_hit_rate = 0.0;
};

/// Move-only RAII handle owning a sub-allocation from a DeviceMemoryPool.
class PooledAllocation {
public:
    PooledAllocation() noexcept = default;
    PooledAllocation(DeviceMemoryPool* pool, void* ptr, std::size_t size, std::size_t alignment) noexcept;
    ~PooledAllocation() noexcept;

    PooledAllocation(const PooledAllocation&) = delete;
    PooledAllocation& operator=(const PooledAllocation&) = delete;

    PooledAllocation(PooledAllocation&& other) noexcept;
    PooledAllocation& operator=(PooledAllocation&& other) noexcept;

    [[nodiscard]] void* data() const noexcept { return ptr_; }

    template <typename T>
    [[nodiscard]] T* as() const noexcept {
        return static_cast<T*>(ptr_);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

    void reset() noexcept;

private:
    DeviceMemoryPool* pool_ = nullptr;
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t alignment_ = 0;
};

using BufferHandle = PooledAllocation;

/// Thread-safe segregated free-list & address-ordered coalescing GPU/pinned memory pool.
///
/// Design notes:
/// - Block metadata is maintained strictly on the host side (out-of-band), mirroring
///   real GPU caching allocators where device memory cannot be dereferenced for
///   intrusive headers and buffers require strict 256B-4096B alignment.
/// - Power-of-two size-class bins plus a 32-bit occupancy bitmask enable O(1) fast-path
///   allocation via C++20 `std::countr_zero`, with best-fit fallback and O(1) bidirectional
///   neighbor coalescing on deallocation.
class DeviceMemoryPool {
public:
    static constexpr std::size_t kMinAlignment = 16;
    static constexpr std::size_t kDefaultAlignment = 256;
    static constexpr std::size_t kMaxAlignment = 4096;
    static constexpr std::size_t kNumBins = 32;

    explicit DeviceMemoryPool(PoolConfig config);
    explicit DeviceMemoryPool(std::size_t capacity_bytes,
                              std::size_t default_alignment = kDefaultAlignment,
                              MemoryKind kind = MemoryKind::DeviceSimulated);
    ~DeviceMemoryPool();

    DeviceMemoryPool(const DeviceMemoryPool&) = delete;
    DeviceMemoryPool& operator=(const DeviceMemoryPool&) = delete;
    DeviceMemoryPool(DeviceMemoryPool&&) = delete;
    DeviceMemoryPool& operator=(DeviceMemoryPool&&) = delete;

    /// Allocates an aligned buffer and returns an RAII handle. Throws std::bad_alloc on OOM.
    [[nodiscard]] PooledAllocation allocate(std::size_t bytes, std::size_t alignment = 0);

    /// Allocates an aligned buffer, returning std::nullopt on OOM instead of throwing.
    [[nodiscard]] std::optional<PooledAllocation> try_allocate(std::size_t bytes, std::size_t alignment = 0);

    /// Raw allocation API returning nullptr on OOM (or if bytes == 0).
    [[nodiscard]] void* allocate_raw(std::size_t bytes, std::size_t alignment = 0);

    /// Releases a pointer previously obtained from allocate_raw(). Passing nullptr is a no-op.
    void deallocate_raw(void* ptr);

    [[nodiscard]] PoolStats stats() const;
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t default_alignment() const noexcept { return default_alignment_; }
    [[nodiscard]] MemoryKind kind() const noexcept { return kind_; }

private:
    static constexpr std::size_t kInvalidBin = kNumBins;

    struct Block {
        std::size_t offset = 0;
        std::size_t size = 0;
        std::size_t requested_bytes = 0;
        bool is_free = true;

        Block* prev_phys = nullptr;
        Block* next_phys = nullptr;

        std::size_t bin_index = kInvalidBin;
        Block* prev_free = nullptr;
        Block* next_free = nullptr;
    };

    static constexpr bool is_power_of_two(std::size_t value) noexcept {
        return value != 0 && (value & (value - 1)) == 0;
    }

    static constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static std::size_t bin_for_size(std::size_t size) noexcept;
    static bool block_fits(const Block* block, std::size_t alloc_size, std::size_t alignment) noexcept;

    Block* acquire_block_node();
    void recycle_block_node(Block* block);

    void insert_free_block(Block* block) noexcept;
    void remove_free_block(Block* block) noexcept;
    Block* find_best_fit_in_bin(std::size_t bin, std::size_t alloc_size, std::size_t alignment) const noexcept;

    void* base_ptr_ = nullptr;
    std::size_t base_alignment_ = kMaxAlignment;
    std::size_t capacity_ = 0;
    std::size_t default_alignment_ = kDefaultAlignment;
    MemoryKind kind_ = MemoryKind::DeviceSimulated;

    mutable std::mutex mutex_;

    std::array<Block*, kNumBins> bins_{};
    std::uint32_t non_empty_bins_mask_ = 0;

    std::vector<std::unique_ptr<Block>> block_storage_;
    std::vector<Block*> recycled_blocks_;
    std::unordered_map<std::uintptr_t, Block*> active_blocks_;

    std::size_t allocated_bytes_ = 0;
    std::size_t free_bytes_ = 0;
    std::size_t peak_allocated_bytes_ = 0;
    std::size_t free_block_count_ = 0;
    std::size_t active_allocations_ = 0;
    std::uint64_t total_alloc_calls_ = 0;
    std::uint64_t total_free_calls_ = 0;
    std::uint64_t fast_path_hits_ = 0;
    std::uint64_t fallback_hits_ = 0;
    std::uint64_t coalesce_merges_ = 0;
};

}  // namespace gpuruntime
