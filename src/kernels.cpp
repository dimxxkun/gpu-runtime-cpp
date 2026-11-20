#include "gpuruntime/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace gpuruntime {

namespace {

constexpr std::size_t kMC = 64;
constexpr std::size_t kKC = 128;
constexpr std::size_t kNC = 256;
constexpr std::size_t kMR = 4;
constexpr std::size_t kNR = 8;

void pack_panel_a(const TensorView<const float>& a,
                  std::size_t row_start,
                  std::size_t mc,
                  std::size_t col_start,
                  std::size_t kc,
                  float alpha,
                  float* dst) noexcept {
    for (std::size_t ir = 0; ir < mc; ir += kMR) {
        const std::size_t mr = std::min(kMR, mc - ir);
        float* panel_dst = dst + ir * kc;
        for (std::size_t k = 0; k < kc; ++k) {
            float* col_dst = panel_dst + k * kMR;
            for (std::size_t r = 0; r < mr; ++r) {
                col_dst[r] = alpha * a.unchecked_2d(row_start + ir + r, col_start + k);
            }
            for (std::size_t r = mr; r < kMR; ++r) {
                col_dst[r] = 0.0f;
            }
        }
    }
}

void pack_panel_b(const TensorView<const float>& b,
                  std::size_t row_start,
                  std::size_t kc,
                  std::size_t col_start,
                  std::size_t nc,
                  float* dst) noexcept {
    for (std::size_t jr = 0; jr < nc; jr += kNR) {
        const std::size_t nr = std::min(kNR, nc - jr);
        float* panel_dst = dst + jr * kc;
        for (std::size_t k = 0; k < kc; ++k) {
            float* row_dst = panel_dst + k * kNR;
            for (std::size_t c = 0; c < nr; ++c) {
                row_dst[c] = b.unchecked_2d(row_start + k, col_start + jr + c);
            }
            for (std::size_t c = nr; c < kNR; ++c) {
                row_dst[c] = 0.0f;
            }
        }
    }
}

void micro_kernel_4x8(std::size_t kc,
                      const float* packed_a,
                      const float* packed_b,
                      TensorView<float>& c_view,
                      std::size_t row0,
                      std::size_t col0,
                      std::size_t mr,
                      std::size_t nr) noexcept {
    alignas(32) float acc[kMR][kNR];

#if defined(__AVX2__)
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    for (std::size_t k = 0; k < kc; ++k) {
        const float* a_ptr = packed_a + k * kMR;
        const __m256 b_vec = _mm256_loadu_ps(packed_b + k * kNR);
        const __m256 a0 = _mm256_broadcast_ss(a_ptr + 0);
        const __m256 a1 = _mm256_broadcast_ss(a_ptr + 1);
        const __m256 a2 = _mm256_broadcast_ss(a_ptr + 2);
        const __m256 a3 = _mm256_broadcast_ss(a_ptr + 3);
#if defined(__FMA__)
        c0 = _mm256_fmadd_ps(a0, b_vec, c0);
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);
#else
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b_vec));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b_vec));
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b_vec));
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b_vec));
#endif
    }

    _mm256_store_ps(acc[0], c0);
    _mm256_store_ps(acc[1], c1);
    _mm256_store_ps(acc[2], c2);
    _mm256_store_ps(acc[3], c3);
#else
    for (std::size_t r = 0; r < kMR; ++r) {
        for (std::size_t c = 0; c < kNR; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    for (std::size_t k = 0; k < kc; ++k) {
        const float* a_ptr = packed_a + k * kMR;
        const float* b_ptr = packed_b + k * kNR;
        for (std::size_t r = 0; r < kMR; ++r) {
            const float ar = a_ptr[r];
            for (std::size_t c = 0; c < kNR; ++c) {
                acc[r][c] += ar * b_ptr[c];
            }
        }
    }
#endif

    if (mr == kMR && nr == kNR && c_view.stride(1) == 1) {
        const std::size_t row_stride = c_view.stride(0);
        float* base = c_view.data() + row0 * row_stride + col0;
        for (std::size_t r = 0; r < kMR; ++r) {
            float* c_row = base + r * row_stride;
            for (std::size_t c = 0; c < kNR; ++c) {
                c_row[c] += acc[r][c];
            }
        }
    } else {
        for (std::size_t r = 0; r < mr; ++r) {
            for (std::size_t c = 0; c < nr; ++c) {
                c_view.unchecked_2d(row0 + r, col0 + c) += acc[r][c];
            }
        }
    }
}

template <typename ReadFn, typename WriteFn>
void online_softmax_row(std::size_t cols, ReadFn read_at, WriteFn write_at) noexcept {
    if (cols == 0) {
        return;
    }

    // Single-pass online running max & normalizer (Milakov & Gimelshein, 2018).
    float running_max = -std::numeric_limits<float>::infinity();
    double running_sum = 0.0;

    for (std::size_t j = 0; j < cols; ++j) {
        const float x = read_at(j);
        if (x > running_max) {
            if (std::isfinite(running_max)) {
                running_sum =
                    running_sum * std::exp(static_cast<double>(running_max) - static_cast<double>(x)) +
                    1.0;
            } else {
                running_sum = 1.0;
            }
            running_max = x;
        } else if (std::isfinite(running_max)) {
            running_sum += std::exp(static_cast<double>(x) - static_cast<double>(running_max));
        }
    }

    if (running_sum <= 0.0 || !std::isfinite(running_sum)) {
        for (std::size_t j = 0; j < cols; ++j) {
            write_at(j, 0.0f);
        }
        return;
    }

    const double inv_sum = 1.0 / running_sum;
    for (std::size_t j = 0; j < cols; ++j) {
        const double shifted =
            static_cast<double>(read_at(j)) - static_cast<double>(running_max);
        write_at(j, static_cast<float>(std::exp(shifted) * inv_sum));
    }
}

template <typename ReadFn, typename WriteFn>
void rms_norm_row(std::size_t cols,
                  const TensorView<const float>& weight,
                  float eps,
                  ReadFn read_at,
                  WriteFn write_at) noexcept {
    if (cols == 0) {
        return;
    }

    double sum_sq = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
        const double x = static_cast<double>(read_at(j));
        sum_sq += x * x;
    }

    const double mean_sq = sum_sq / static_cast<double>(cols);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

    for (std::size_t j = 0; j < cols; ++j) {
        const double x = static_cast<double>(read_at(j));
        const double w = static_cast<double>(weight.unchecked_1d(j));
        write_at(j, static_cast<float>(x * inv_rms * w));
    }
}

}  // namespace

const char* active_simd_backend() noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    return "AVX2+FMA";
#elif defined(__AVX2__)
    return "AVX2";
#else
    return "Scalar";
#endif
}

void sgemm(TensorView<const float> a,
           TensorView<const float> b,
           TensorView<float> c,
           float alpha,
           float beta) {
    if (a.rank() != 2 || b.rank() != 2 || c.rank() != 2) {
        throw std::invalid_argument("sgemm: A, B, and C must all be 2D TensorViews");
    }

    const std::size_t m = a.dim(0);
    const std::size_t k = a.dim(1);
    const std::size_t n = b.dim(1);

    if (b.dim(0) != k || c.dim(0) != m || c.dim(1) != n) {
        throw std::invalid_argument("sgemm: incompatible matrix dimensions");
    }

    if (m == 0 || n == 0) {
        return;
    }

    if (beta == 0.0f) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                c.unchecked_2d(i, j) = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                c.unchecked_2d(i, j) *= beta;
            }
        }
    }

    if (k == 0 || alpha == 0.0f) {
        return;
    }

    // Scratch packing buffers sized for one L1/L2 tile (kMC*kKC = 32 KB, kKC*kNC = 128 KB).
    std::vector<float> packed_a(kMC * kKC, 0.0f);
    std::vector<float> packed_b(kKC * kNC, 0.0f);

    for (std::size_t j = 0; j < n; j += kNC) {
        const std::size_t nc = std::min(kNC, n - j);
        for (std::size_t p = 0; p < k; p += kKC) {
            const std::size_t kc = std::min(kKC, k - p);
            pack_panel_b(b, p, kc, j, nc, packed_b.data());

            for (std::size_t i = 0; i < m; i += kMC) {
                const std::size_t mc = std::min(kMC, m - i);
                pack_panel_a(a, i, mc, p, kc, alpha, packed_a.data());

                for (std::size_t jr = 0; jr < nc; jr += kNR) {
                    const std::size_t nr = std::min(kNR, nc - jr);
                    const float* b_panel = packed_b.data() + jr * kc;
                    for (std::size_t ir = 0; ir < mc; ir += kMR) {
                        const std::size_t mr = std::min(kMR, mc - ir);
                        const float* a_panel = packed_a.data() + ir * kc;
                        micro_kernel_4x8(kc, a_panel, b_panel, c, i + ir, j + jr, mr, nr);
                    }
                }
            }
        }
    }
}

void online_softmax(TensorView<const float> input, TensorView<float> output) {
    if (input.rank() != 1 && input.rank() != 2) {
        throw std::invalid_argument("online_softmax: only 1D and 2D TensorViews are supported");
    }
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("online_softmax: input and output shapes must match");
    }
    if (input.empty()) {
        return;
    }

    if (input.rank() == 1) {
        const std::size_t cols = input.dim(0);
        online_softmax_row(
            cols,
            [&input](std::size_t j) noexcept { return input.unchecked_1d(j); },
            [&output](std::size_t j, float val) noexcept { output.unchecked_1d(j) = val; });
    } else {
        const std::size_t rows = input.dim(0);
        const std::size_t cols = input.dim(1);
        for (std::size_t r = 0; r < rows; ++r) {
            online_softmax_row(
                cols,
                [&input, r](std::size_t j) noexcept { return input.unchecked_2d(r, j); },
                [&output, r](std::size_t j, float val) noexcept {
                    output.unchecked_2d(r, j) = val;
                });
        }
    }
}

void rms_norm(TensorView<const float> input,
              TensorView<const float> weight,
              TensorView<float> output,
              float eps) {
    if (input.rank() != 1 && input.rank() != 2) {
        throw std::invalid_argument("rms_norm: only 1D and 2D TensorViews are supported");
    }
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("rms_norm: input and output shapes must match");
    }
    if (weight.rank() != 1) {
        throw std::invalid_argument("rms_norm: weight must be a 1D TensorView");
    }
    const std::size_t last_dim = input.dim(input.rank() - 1);
    if (weight.dim(0) != last_dim) {
        throw std::invalid_argument("rms_norm: weight length must match input last dimension");
    }
    if (eps < 0.0f) {
        throw std::invalid_argument("rms_norm: eps must be >= 0");
    }
    if (input.empty()) {
        return;
    }

    if (input.rank() == 1) {
        rms_norm_row(
            last_dim,
            weight,
            eps,
            [&input](std::size_t j) noexcept { return input.unchecked_1d(j); },
            [&output](std::size_t j, float val) noexcept { output.unchecked_1d(j) = val; });
    } else {
        const std::size_t rows = input.dim(0);
        for (std::size_t r = 0; r < rows; ++r) {
            rms_norm_row(
                last_dim,
                weight,
                eps,
                [&input, r](std::size_t j) noexcept { return input.unchecked_2d(r, j); },
                [&output, r](std::size_t j, float val) noexcept {
                    output.unchecked_2d(r, j) = val;
                });
        }
    }
}

}  // namespace gpuruntime
