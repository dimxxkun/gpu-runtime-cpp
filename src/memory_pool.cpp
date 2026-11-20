#include "gpuruntime/memory_pool.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gpuruntime {

PooledAllocation::PooledAllocation(DeviceMemoryPool* pool,
                                   void* ptr,
                                   std::size_t size,
                                   std::size_t alignment) noexcept
    : pool_(pool), ptr_(ptr), size_(size), alignment_(alignment) {}

PooledAllocation::~PooledAllocation() noexcept {
    reset();
}

PooledAllocation::PooledAllocation(PooledAllocation&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      ptr_(std::exchange(other.ptr_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      alignment_(std::exchange(other.alignment_, 0)) {}

PooledAllocation& PooledAllocation::operator=(PooledAllocation&& other) noexcept {
    if (this != &other) {
        reset();
        pool_ = std::exchange(other.pool_, nullptr);
        ptr_ = std::exchange(other.ptr_, nullptr);
        size_ = std::exchange(other.size_, 0);
        alignment_ = std::exchange(other.alignment_, 0);
    }
    return *this;
}

void PooledAllocation::reset() noexcept {
    if (pool_ != nullptr && ptr_ != nullptr) {
        try {
            pool_->deallocate_raw(ptr_);
        } catch (...) {
            // Destructor and reset() must remain noexcept.
        }
    }
    pool_ = nullptr;
    ptr_ = nullptr;
    size_ = 0;
    alignment_ = 0;
}

DeviceMemoryPool::DeviceMemoryPool(PoolConfig config)
    : base_alignment_(kMaxAlignment),
      default_alignment_(config.default_alignment),
      kind_(config.kind) {
    if (config.capacity_bytes == 0) {
        throw std::invalid_argument("DeviceMemoryPool capacity must be > 0");
    }
    if (!is_power_of_two(default_alignment_) ||
        default_alignment_ < kMinAlignment ||
        default_alignment_ > kMaxAlignment) {
        throw std::invalid_argument(
            "DeviceMemoryPool default_alignment must be a power of two in [16, 4096]");
    }
    if (config.capacity_bytes > std::numeric_limits<std::size_t>::max() - default_alignment_) {
        throw std::invalid_argument("DeviceMemoryPool capacity overflow");
    }

    capacity_ = align_up(config.capacity_bytes, default_alignment_);
    base_alignment_ = std::max(kMaxAlignment, default_alignment_);
    base_ptr_ = ::operator new(capacity_, std::align_val_t{base_alignment_});

    try {
        free_bytes_ = capacity_;

        Block* initial = acquire_block_node();
        initial->offset = 0;
        initial->size = capacity_;
        initial->requested_bytes = 0;
        initial->is_free = true;
        insert_free_block(initial);
    } catch (...) {
        ::operator delete(base_ptr_, std::align_val_t{base_alignment_});
        base_ptr_ = nullptr;
        throw;
    }
}

DeviceMemoryPool::DeviceMemoryPool(std::size_t capacity_bytes,
                                   std::size_t default_alignment,
                                   MemoryKind kind)
    : DeviceMemoryPool(PoolConfig{capacity_bytes, default_alignment, kind}) {}

DeviceMemoryPool::~DeviceMemoryPool() {
    if (base_ptr_ != nullptr) {
        ::operator delete(base_ptr_, std::align_val_t{base_alignment_});
        base_ptr_ = nullptr;
    }
}

std::size_t DeviceMemoryPool::bin_for_size(std::size_t size) noexcept {
    if (size <= 1) {
        return 0;
    }
    const std::size_t bit_idx = static_cast<std::size_t>(std::bit_width(size) - 1);
    return std::min(bit_idx, kNumBins - 1);
}

bool DeviceMemoryPool::block_fits(const Block* block,
                                  std::size_t alloc_size,
                                  std::size_t alignment) noexcept {
    const std::size_t aligned_offset = align_up(block->offset, alignment);
    const std::size_t prefix = aligned_offset - block->offset;
    return prefix <= block->size && (block->size - prefix) >= alloc_size;
}

DeviceMemoryPool::Block* DeviceMemoryPool::acquire_block_node() {
    if (!recycled_blocks_.empty()) {
        Block* node = recycled_blocks_.back();
        recycled_blocks_.pop_back();
        *node = Block{};
        return node;
    }
    auto owned = std::make_unique<Block>();
    Block* raw = owned.get();
    block_storage_.push_back(std::move(owned));
    return raw;
}

void DeviceMemoryPool::recycle_block_node(Block* block) {
    *block = Block{};
    recycled_blocks_.push_back(block);
}

void DeviceMemoryPool::insert_free_block(Block* block) noexcept {
    const std::size_t bin = bin_for_size(block->size);
    block->is_free = true;
    block->bin_index = bin;
    block->prev_free = nullptr;
    block->next_free = bins_[bin];
    if (bins_[bin] != nullptr) {
        bins_[bin]->prev_free = block;
    }
    bins_[bin] = block;
    non_empty_bins_mask_ |= (std::uint32_t{1} << bin);
    ++free_block_count_;
}

void DeviceMemoryPool::remove_free_block(Block* block) noexcept {
    const std::size_t bin = block->bin_index;
    if (block->prev_free != nullptr) {
        block->prev_free->next_free = block->next_free;
    } else {
        bins_[bin] = block->next_free;
    }
    if (block->next_free != nullptr) {
        block->next_free->prev_free = block->prev_free;
    }
    if (bins_[bin] == nullptr) {
        non_empty_bins_mask_ &= ~(std::uint32_t{1} << bin);
    }
    block->prev_free = nullptr;
    block->next_free = nullptr;
    block->bin_index = kInvalidBin;
    --free_block_count_;
}

DeviceMemoryPool::Block* DeviceMemoryPool::find_best_fit_in_bin(
    std::size_t bin,
    std::size_t alloc_size,
    std::size_t alignment) const noexcept {
    Block* best = nullptr;
    for (Block* cur = bins_[bin]; cur != nullptr; cur = cur->next_free) {
        if (block_fits(cur, alloc_size, alignment)) {
            if (best == nullptr || cur->size < best->size) {
                best = cur;
                if (best->size == alloc_size) {
                    break;
                }
            }
        }
    }
    return best;
}

PooledAllocation DeviceMemoryPool::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return PooledAllocation{};
    }
    const std::size_t req_align = (alignment == 0) ? default_alignment_ : alignment;
    void* ptr = allocate_raw(bytes, req_align);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    const std::size_t effective_align = std::max(req_align, default_alignment_);
    return PooledAllocation(this, ptr, bytes, effective_align);
}

std::optional<PooledAllocation> DeviceMemoryPool::try_allocate(std::size_t bytes,
                                                               std::size_t alignment) {
    if (bytes == 0) {
        return PooledAllocation{};
    }
    const std::size_t req_align = (alignment == 0) ? default_alignment_ : alignment;
    void* ptr = allocate_raw(bytes, req_align);
    if (ptr == nullptr) {
        return std::nullopt;
    }
    const std::size_t effective_align = std::max(req_align, default_alignment_);
    return PooledAllocation(this, ptr, bytes, effective_align);
}

void* DeviceMemoryPool::allocate_raw(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }

    const std::size_t req_align = (alignment == 0) ? default_alignment_ : alignment;
    if (!is_power_of_two(req_align) || req_align > kMaxAlignment) {
        throw std::invalid_argument(
            "Requested alignment must be a power of two <= 4096");
    }

    const std::size_t effective_align = std::max(req_align, default_alignment_);
    if (bytes > capacity_) {
        return nullptr;
    }
    const std::size_t alloc_size = align_up(bytes, default_alignment_);
    if (alloc_size > capacity_) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const std::size_t target_bin = bin_for_size(alloc_size);
    Block* chosen = nullptr;
    bool fast_hit = false;

    // 1. O(1) fast path: check the head of the exact size-class bin.
    if (bins_[target_bin] != nullptr &&
        block_fits(bins_[target_bin], alloc_size, effective_align)) {
        chosen = bins_[target_bin];
        fast_hit = true;
    }

    // 2. Intra-bin best-fit fallback before splitting a larger size class.
    if (chosen == nullptr && bins_[target_bin] != nullptr) {
        chosen = find_best_fit_in_bin(target_bin, alloc_size, effective_align);
    }

    // 3. O(1) bitmask jump to lowest non-empty higher bin, with best-fit scan fallback.
    if (chosen == nullptr && (target_bin + 1) < kNumBins) {
        std::uint32_t higher_mask =
            non_empty_bins_mask_ & (~std::uint32_t{0} << (target_bin + 1));
        if (higher_mask != 0) {
            const std::size_t first_higher_bin =
                static_cast<std::size_t>(std::countr_zero(higher_mask));
            if (bins_[first_higher_bin] != nullptr &&
                block_fits(bins_[first_higher_bin], alloc_size, effective_align)) {
                chosen = bins_[first_higher_bin];
                fast_hit = true;
            } else {
                while (higher_mask != 0) {
                    const std::size_t bin =
                        static_cast<std::size_t>(std::countr_zero(higher_mask));
                    chosen = find_best_fit_in_bin(bin, alloc_size, effective_align);
                    if (chosen != nullptr) {
                        break;
                    }
                    higher_mask &= (higher_mask - 1);
                }
            }
        }
    }

    if (chosen == nullptr) {
        return nullptr;
    }

    remove_free_block(chosen);

    const std::size_t aligned_offset = align_up(chosen->offset, effective_align);
    const std::size_t prefix_size = aligned_offset - chosen->offset;

    // Split leading alignment padding back into a free block so it can coalesce later.
    if (prefix_size > 0) {
        Block* prefix = acquire_block_node();
        prefix->offset = chosen->offset;
        prefix->size = prefix_size;
        prefix->prev_phys = chosen->prev_phys;
        prefix->next_phys = chosen;
        if (chosen->prev_phys != nullptr) {
            chosen->prev_phys->next_phys = prefix;
        }
        chosen->prev_phys = prefix;

        chosen->offset = aligned_offset;
        chosen->size -= prefix_size;

        insert_free_block(prefix);
    }

    // Split trailing remainder if large enough to form an aligned free block.
    const std::size_t suffix_size = chosen->size - alloc_size;
    if (suffix_size >= default_alignment_) {
        Block* suffix = acquire_block_node();
        suffix->offset = chosen->offset + alloc_size;
        suffix->size = suffix_size;
        suffix->prev_phys = chosen;
        suffix->next_phys = chosen->next_phys;
        if (chosen->next_phys != nullptr) {
            chosen->next_phys->prev_phys = suffix;
        }
        chosen->next_phys = suffix;

        chosen->size = alloc_size;

        insert_free_block(suffix);
    }

    chosen->is_free = false;
    chosen->requested_bytes = bytes;

    allocated_bytes_ += chosen->size;
    free_bytes_ -= chosen->size;
    peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);
    ++active_allocations_;
    ++total_alloc_calls_;
    if (fast_hit) {
        ++fast_path_hits_;
    } else {
        ++fallback_hits_;
    }

    auto* byte_ptr = static_cast<std::byte*>(base_ptr_) + chosen->offset;
    void* result = static_cast<void*>(byte_ptr);
    active_blocks_.emplace(reinterpret_cast<std::uintptr_t>(result), chosen);
    return result;
}

void DeviceMemoryPool::deallocate_raw(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const auto key = reinterpret_cast<std::uintptr_t>(ptr);
    const auto it = active_blocks_.find(key);
    if (it == active_blocks_.end()) {
        throw std::invalid_argument(
            "DeviceMemoryPool::deallocate_raw: unknown or already freed pointer");
    }

    Block* blk = it->second;
    active_blocks_.erase(it);

    allocated_bytes_ -= blk->size;
    free_bytes_ += blk->size;
    --active_allocations_;
    ++total_free_calls_;

    blk->is_free = true;
    blk->requested_bytes = 0;

    // Coalesce with immediate lower-address neighbor if free.
    if (blk->prev_phys != nullptr && blk->prev_phys->is_free) {
        Block* prev = blk->prev_phys;
        remove_free_block(prev);
        prev->size += blk->size;
        prev->next_phys = blk->next_phys;
        if (blk->next_phys != nullptr) {
            blk->next_phys->prev_phys = prev;
        }
        recycle_block_node(blk);
        blk = prev;
        ++coalesce_merges_;
    }

    // Coalesce with immediate higher-address neighbor if free.
    if (blk->next_phys != nullptr && blk->next_phys->is_free) {
        Block* next = blk->next_phys;
        remove_free_block(next);
        blk->size += next->size;
        blk->next_phys = next->next_phys;
        if (next->next_phys != nullptr) {
            next->next_phys->prev_phys = blk;
        }
        recycle_block_node(next);
        ++coalesce_merges_;
    }

    insert_free_block(blk);
}

PoolStats DeviceMemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::size_t largest_free = 0;
    if (non_empty_bins_mask_ != 0) {
        const std::size_t top_bin =
            static_cast<std::size_t>(std::bit_width(non_empty_bins_mask_) - 1);
        for (const Block* cur = bins_[top_bin]; cur != nullptr; cur = cur->next_free) {
            largest_free = std::max(largest_free, cur->size);
        }
    }

    PoolStats s{};
    s.total_capacity_bytes = capacity_;
    s.allocated_bytes = allocated_bytes_;
    s.free_bytes = free_bytes_;
    s.peak_allocated_bytes = peak_allocated_bytes_;
    s.largest_free_block_bytes = largest_free;
    s.free_block_count = free_block_count_;
    s.active_allocations = active_allocations_;
    s.total_alloc_calls = total_alloc_calls_;
    s.total_free_calls = total_free_calls_;
    s.fast_path_hits = fast_path_hits_;
    s.fallback_hits = fallback_hits_;
    s.coalesce_merges = coalesce_merges_;

    if (free_bytes_ > 0) {
        s.fragmentation_ratio =
            1.0 - (static_cast<double>(largest_free) / static_cast<double>(free_bytes_));
    } else {
        s.fragmentation_ratio = 0.0;
    }

    if (total_alloc_calls_ > 0) {
        s.cache_hit_rate =
            static_cast<double>(fast_path_hits_) / static_cast<double>(total_alloc_calls_);
    } else {
        s.cache_hit_rate = 0.0;
    }

    return s;
}

}  // namespace gpuruntime
