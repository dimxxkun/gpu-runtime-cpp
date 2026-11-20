#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gpuruntime {

/// Zero-copy N-dimensional strided tensor view supporting slicing, transposition,
/// permutation, contiguity-checked reshaping, and NumPy-style broadcasting.
template <typename T>
class TensorView {
public:
    using value_type = std::remove_cv_t<T>;

    TensorView() noexcept = default;

    TensorView(T* data, std::vector<std::size_t> shape)
        : data_(data),
          shape_(std::move(shape)),
          strides_(compute_contiguous_strides(shape_)) {}

    TensorView(T* data, std::vector<std::size_t> shape, std::vector<std::size_t> strides)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {
        if (shape_.size() != strides_.size()) {
            throw std::invalid_argument("TensorView: shape and strides rank mismatch");
        }
    }

    /// Implicit conversion from TensorView<U> to TensorView<const U>.
    template <typename U>
        requires(std::is_same_v<T, const U> && !std::is_const_v<U>)
    TensorView(const TensorView<U>& other)
        : data_(other.data()), shape_(other.shape()), strides_(other.strides()) {}

    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }

    [[nodiscard]] std::size_t dim(std::size_t axis) const {
        validate_axis(axis);
        return shape_[axis];
    }

    [[nodiscard]] std::size_t stride(std::size_t axis) const {
        validate_axis(axis);
        return strides_[axis];
    }

    [[nodiscard]] std::size_t numel() const noexcept {
        if (shape_.empty()) {
            return (data_ != nullptr) ? 1 : 0;
        }
        std::size_t total = 1;
        for (std::size_t d : shape_) {
            total *= d;
        }
        return total;
    }

    [[nodiscard]] bool empty() const noexcept { return numel() == 0; }

    /// Returns true if elements are laid out strictly in row-major contiguous order.
    [[nodiscard]] bool is_contiguous() const noexcept {
        if (empty()) {
            return true;
        }
        if (shape_.size() == 1) {
            return shape_[0] <= 1 || strides_[0] == 1;
        }
        std::size_t expected_stride = 1;
        for (std::size_t i = shape_.size(); i-- > 0;) {
            if (shape_[i] == 1) {
                continue;
            }
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }

    /// Computes linear element offset from multi-dimensional coordinates (bounds-checked).
    [[nodiscard]] std::size_t linear_offset(std::span<const std::size_t> indices) const {
        if (indices.size() != shape_.size()) {
            throw std::out_of_range("TensorView: index rank mismatch");
        }
        std::size_t offset = 0;
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("TensorView: index out of bounds along axis " +
                                        std::to_string(i));
            }
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    [[nodiscard]] T& at(std::span<const std::size_t> indices) const {
        if (data_ == nullptr) {
            throw std::out_of_range("TensorView: dereferencing null view");
        }
        return data_[linear_offset(indices)];
    }

    template <typename... Indices>
        requires(sizeof...(Indices) > 0 && (std::is_convertible_v<Indices, std::size_t> && ...))
    [[nodiscard]] T& operator()(Indices... indices) const {
        if (data_ == nullptr) {
            throw std::out_of_range("TensorView: dereferencing null view");
        }
        const std::array<std::size_t, sizeof...(Indices)> idx{
            static_cast<std::size_t>(indices)...};
        return data_[linear_offset(idx)];
    }

    /// Unchecked fast-path 2D accessor for tight compute kernels.
    [[nodiscard]] T& unchecked_2d(std::size_t row, std::size_t col) const noexcept {
        return data_[row * strides_[0] + col * strides_[1]];
    }

    /// Unchecked fast-path 1D accessor for tight compute kernels.
    [[nodiscard]] T& unchecked_1d(std::size_t idx) const noexcept {
        return data_[idx * strides_[0]];
    }

    /// Zero-copy strided slice along `axis` over `[start, end)` with stride `step`.
    [[nodiscard]] TensorView slice(std::size_t axis,
                                   std::size_t start,
                                   std::size_t end,
                                   std::size_t step = 1) const {
        validate_axis(axis);
        if (step == 0) {
            throw std::invalid_argument("TensorView::slice: step must be >= 1");
        }
        if (start > end || end > shape_[axis]) {
            throw std::out_of_range("TensorView::slice: invalid slice range");
        }

        auto new_shape = shape_;
        auto new_strides = strides_;
        new_shape[axis] = (start == end) ? 0 : ((end - start + step - 1) / step);
        new_strides[axis] = strides_[axis] * step;

        T* new_data = (data_ != nullptr) ? (data_ + start * strides_[axis]) : nullptr;
        return TensorView(new_data, std::move(new_shape), std::move(new_strides));
    }

    /// Zero-copy swap of two axes.
    [[nodiscard]] TensorView transpose(std::size_t axis0, std::size_t axis1) const {
        validate_axis(axis0);
        validate_axis(axis1);
        auto new_shape = shape_;
        auto new_strides = strides_;
        std::swap(new_shape[axis0], new_shape[axis1]);
        std::swap(new_strides[axis0], new_strides[axis1]);
        return TensorView(data_, std::move(new_shape), std::move(new_strides));
    }

    /// Zero-copy general axis permutation.
    [[nodiscard]] TensorView permute(std::span<const std::size_t> order) const {
        if (order.size() != shape_.size()) {
            throw std::invalid_argument("TensorView::permute: order rank mismatch");
        }
        std::vector<bool> seen(shape_.size(), false);
        std::vector<std::size_t> new_shape(shape_.size());
        std::vector<std::size_t> new_strides(shape_.size());

        for (std::size_t i = 0; i < order.size(); ++i) {
            const std::size_t axis = order[i];
            if (axis >= shape_.size() || seen[axis]) {
                throw std::invalid_argument("TensorView::permute: invalid axis permutation");
            }
            seen[axis] = true;
            new_shape[i] = shape_[axis];
            new_strides[i] = strides_[axis];
        }
        return TensorView(data_, std::move(new_shape), std::move(new_strides));
    }

    /// Zero-copy reshape. Throws std::runtime_error if the view is non-contiguous.
    [[nodiscard]] TensorView reshape(std::vector<std::size_t> new_shape) const {
        if (!is_contiguous()) {
            throw std::runtime_error(
                "TensorView::reshape: cannot zero-copy reshape a non-contiguous view");
        }

        std::size_t new_numel = new_shape.empty() ? (data_ != nullptr ? 1 : 0) : 1;
        for (std::size_t d : new_shape) {
            new_numel *= d;
        }
        if (new_numel != numel()) {
            throw std::invalid_argument(
                "TensorView::reshape: total element count mismatch");
        }
        return TensorView(data_, std::move(new_shape));
    }

    /// Infers the NumPy/PyTorch right-aligned broadcasted shape of two shapes.
    [[nodiscard]] static std::vector<std::size_t> broadcast_shape(
        std::span<const std::size_t> a,
        std::span<const std::size_t> b) {
        const std::size_t out_rank = std::max(a.size(), b.size());
        std::vector<std::size_t> result(out_rank, 1);

        for (std::size_t i = 0; i < out_rank; ++i) {
            const std::size_t dim_a =
                (i < a.size()) ? a[a.size() - 1 - i] : 1;
            const std::size_t dim_b =
                (i < b.size()) ? b[b.size() - 1 - i] : 1;

            if (dim_a == dim_b) {
                result[out_rank - 1 - i] = dim_a;
            } else if (dim_a == 1) {
                result[out_rank - 1 - i] = dim_b;
            } else if (dim_b == 1) {
                result[out_rank - 1 - i] = dim_a;
            } else {
                throw std::invalid_argument(
                    "TensorView::broadcast_shape: incompatible dimensions");
            }
        }
        return result;
    }

    /// Zero-copy broadcast to `target_shape` by setting broadcasted axis strides to 0.
    [[nodiscard]] TensorView broadcast_to(std::span<const std::size_t> target_shape) const {
        if (target_shape.size() < shape_.size()) {
            throw std::invalid_argument(
                "TensorView::broadcast_to: target rank cannot be smaller than source rank");
        }

        const std::size_t target_rank = target_shape.size();
        const std::size_t rank_diff = target_rank - shape_.size();
        std::vector<std::size_t> new_shape(target_shape.begin(), target_shape.end());
        std::vector<std::size_t> new_strides(target_rank, 0);

        for (std::size_t i = 0; i < shape_.size(); ++i) {
            const std::size_t dst_idx = rank_diff + i;
            if (shape_[i] == target_shape[dst_idx]) {
                new_strides[dst_idx] = strides_[i];
            } else if (shape_[i] == 1) {
                new_strides[dst_idx] = 0;
            } else {
                throw std::invalid_argument(
                    "TensorView::broadcast_to: incompatible dimension at axis " +
                    std::to_string(dst_idx));
            }
        }

        return TensorView(data_, std::move(new_shape), std::move(new_strides));
    }

private:
    static std::vector<std::size_t> compute_contiguous_strides(
        const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> strides(shape.size(), 1);
        for (std::size_t i = shape.size(); i-- > 1;) {
            strides[i - 1] = strides[i] * shape[i];
        }
        return strides;
    }

    void validate_axis(std::size_t axis) const {
        if (axis >= shape_.size()) {
            throw std::out_of_range("TensorView: axis out of range");
        }
    }

    T* data_ = nullptr;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
};

}  // namespace gpuruntime
