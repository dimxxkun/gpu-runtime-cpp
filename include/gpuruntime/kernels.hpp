#pragma once

#include "gpuruntime/tensor_view.hpp"

#include <cstddef>
#include <cstdint>

namespace gpuruntime {

/// Returns the active SIMD backend compiled into the host kernels ("AVX2+FMA", "AVX2", or "Scalar").
[[nodiscard]] const char* active_simd_backend() noexcept;

/// Computes FLOP count for an M x N x K GEMM (`2 * M * N * K`).
[[nodiscard]] constexpr std::uint64_t gemm_flop_count(std::size_t m,
                                                      std::size_t n,
                                                      std::size_t k) noexcept {
    return 2ULL * static_cast<std::uint64_t>(m) *
           static_cast<std::uint64_t>(n) *
           static_cast<std::uint64_t>(k);
}

/// Cache-blocked L1/L2 tiled SGEMM (`C := alpha * A * B + beta * C`).
///
/// Supports arbitrary strided 2D TensorViews (including transposed views) via
/// BLIS-style MC/NC/KC tile packing with a 4x8 register-blocked micro-kernel
/// (AVX2 SIMD when `__AVX2__` is enabled, with a portable scalar fallback).
void sgemm(TensorView<const float> a,
           TensorView<const float> b,
           TensorView<float> c,
           float alpha = 1.0f,
           float beta = 0.0f);

/// Numerically stable single-pass online softmax (Milakov & Gimelshein, 2018)
/// along the last dimension of a 1D `[N]` or 2D `[Rows, Cols]` tensor view.
void online_softmax(TensorView<const float> input, TensorView<float> output);

/// Root Mean Square Layer Normalization (RMSNorm) along the last dimension
/// of a 1D `[D]` or 2D `[Rows, D]` tensor view with scale `weight [D]`.
void rms_norm(TensorView<const float> input,
              TensorView<const float> weight,
              TensorView<float> output,
              float eps = 1e-6f);

}  // namespace gpuruntime
