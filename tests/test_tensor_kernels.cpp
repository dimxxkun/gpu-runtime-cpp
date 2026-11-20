#include "gpuruntime/kernels.hpp"
#include "gpuruntime/profiler.hpp"
#include "gpuruntime/tensor_view.hpp"
#include "test_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void naive_sgemm_fp64_ref(gpuruntime::TensorView<const float> a,
                          gpuruntime::TensorView<const float> b,
                          gpuruntime::TensorView<float> c,
                          float alpha,
                          float beta) {
    const std::size_t m = a.dim(0);
    const std::size_t k = a.dim(1);
    const std::size_t n = b.dim(1);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                acc += static_cast<double>(a(i, p)) * static_cast<double>(b(p, j));
            }
            const double prior = (beta == 0.0f) ? 0.0 : (static_cast<double>(beta) * c(i, j));
            c(i, j) = static_cast<float>(static_cast<double>(alpha) * acc + prior);
        }
    }
}

void test_tensor_view_operations() {
    std::vector<float> buf(24);
    std::iota(buf.begin(), buf.end(), 0.0f);

    gpuruntime::TensorView<float> t(buf.data(), {2, 3, 4});
    TEST_CHECK(t.is_contiguous());
    TEST_CHECK_EQ(t.numel(), std::size_t{24});
    TEST_CHECK_EQ(t.stride(0), std::size_t{12});
    TEST_CHECK_EQ(t.stride(1), std::size_t{4});
    TEST_CHECK_EQ(t.stride(2), std::size_t{1});
    TEST_CHECK_EQ(t(1, 2, 3), 23.0f);

    // Slice with step > 1 -> non-contiguous
    auto sliced = t.slice(2, 1, 4, 2);  // shape [2, 3, 2], values at col 1 and 3
    TEST_CHECK_EQ(sliced.dim(2), std::size_t{2});
    TEST_CHECK(!sliced.is_contiguous());
    TEST_CHECK_EQ(sliced(0, 0, 0), 1.0f);
    TEST_CHECK_EQ(sliced(0, 0, 1), 3.0f);
    TEST_CHECK_EQ(sliced(1, 2, 1), 23.0f);

    // Non-contiguous reshape must throw std::runtime_error
    TEST_CHECK_THROWS(sliced.reshape({12}), std::runtime_error);

    // Transpose -> non-contiguous, rejects reshape, reads transposed coordinates zero-copy
    auto transposed = t.transpose(0, 2);  // shape [4, 3, 2]
    TEST_CHECK(!transposed.is_contiguous());
    TEST_CHECK_EQ(transposed(3, 2, 1), 23.0f);
    TEST_CHECK_THROWS(transposed.reshape({24}), std::runtime_error);

    // Contiguous reshape succeeds zero-copy
    auto reshaped = t.reshape({4, 6});
    TEST_CHECK(reshaped.is_contiguous());
    TEST_CHECK_EQ(reshaped.data(), t.data());
    TEST_CHECK_EQ(reshaped(3, 5), 23.0f);

    // Permute validation
    const std::array<std::size_t, 3> perm = {2, 0, 1};
    auto permuted = t.permute(perm);
    TEST_CHECK_EQ(permuted.dim(0), std::size_t{4});
    TEST_CHECK_EQ(permuted.dim(1), std::size_t{2});
    TEST_CHECK_EQ(permuted.dim(2), std::size_t{3});
    TEST_CHECK_EQ(permuted(3, 1, 2), 23.0f);

    // Broadcasting shape inference & zero-copy stride-0 view
    const std::array<std::size_t, 2> s1 = {1, 4};
    const std::array<std::size_t, 3> s2 = {2, 3, 1};
    const auto bshape = gpuruntime::TensorView<float>::broadcast_shape(s1, s2);
    TEST_CHECK_EQ(bshape.size(), std::size_t{3});
    TEST_CHECK_EQ(bshape[0], std::size_t{2});
    TEST_CHECK_EQ(bshape[1], std::size_t{3});
    TEST_CHECK_EQ(bshape[2], std::size_t{4});

    std::vector<float> row = {10.0f, 20.0f, 30.0f, 40.0f};
    gpuruntime::TensorView<const float> row_view(row.data(), {1, 4});
    auto bview = row_view.broadcast_to(bshape);
    TEST_CHECK_EQ(bview.stride(0), std::size_t{0});
    TEST_CHECK_EQ(bview.stride(1), std::size_t{0});
    TEST_CHECK_EQ(bview.stride(2), std::size_t{1});
    TEST_CHECK_EQ(bview(0, 0, 2), 30.0f);
    TEST_CHECK_EQ(bview(1, 2, 2), 30.0f);

    // Bounds checking
    const std::array<std::size_t, 3> oob = {2, 0, 0};
    TEST_CHECK_THROWS(t.linear_offset(oob), std::out_of_range);

    const std::array<std::size_t, 2> bad_a = {2, 3};
    const std::array<std::size_t, 2> bad_b = {2, 4};
    TEST_CHECK_THROWS(gpuruntime::TensorView<float>::broadcast_shape(bad_a, bad_b),
                      std::invalid_argument);
}

void test_sgemm_parity_against_fp64() {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    struct GemmCase {
        std::size_t m;
        std::size_t n;
        std::size_t k;
        float alpha;
        float beta;
        bool transpose_a;
        bool transpose_b;
    };

    const std::vector<GemmCase> cases = {
        {1, 1, 1, 1.0f, 0.0f, false, false},
        {7, 13, 19, 1.0f, 0.0f, false, false},
        {33, 65, 17, 1.35f, 0.65f, false, false},
        {131, 97, 73, 1.0f, 0.0f, false, false},
        {70, 52, 66, 0.75f, 0.25f, true, false},
        {45, 77, 38, -1.2f, 0.5f, false, true},
    };

    for (const auto& tc : cases) {
        std::vector<float> a_data(tc.m * tc.k);
        std::vector<float> b_data(tc.k * tc.n);
        std::vector<float> c_init(tc.m * tc.n);

        for (float& x : a_data) {
            x = dist(rng);
        }
        for (float& x : b_data) {
            x = dist(rng);
        }
        for (float& x : c_init) {
            x = (tc.beta == 0.0f) ? std::numeric_limits<float>::quiet_NaN() : dist(rng);
        }

        std::vector<float> c_got = c_init;
        std::vector<float> c_ref = c_init;

        gpuruntime::TensorView<const float> va =
            tc.transpose_a ? gpuruntime::TensorView<const float>(a_data.data(), {tc.k, tc.m})
                                 .transpose(0, 1)
                           : gpuruntime::TensorView<const float>(a_data.data(), {tc.m, tc.k});

        gpuruntime::TensorView<const float> vb =
            tc.transpose_b ? gpuruntime::TensorView<const float>(b_data.data(), {tc.n, tc.k})
                                 .transpose(0, 1)
                           : gpuruntime::TensorView<const float>(b_data.data(), {tc.k, tc.n});

        gpuruntime::TensorView<float> vc_got(c_got.data(), {tc.m, tc.n});
        gpuruntime::TensorView<float> vc_ref(c_ref.data(), {tc.m, tc.n});

        gpuruntime::sgemm(va, vb, vc_got, tc.alpha, tc.beta);
        naive_sgemm_fp64_ref(va, vb, vc_ref, tc.alpha, tc.beta);

        for (std::size_t i = 0; i < tc.m * tc.n; ++i) {
            TEST_CHECK(std::isfinite(c_got[i]));
            TEST_CHECK_NEAR(c_got[i], c_ref[i], 1e-3);
        }
    }
}

void test_online_softmax_stability() {
    // 1. Standard 2D parity against FP64 reference
    {
        std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, -2.5f, 0.0f, 2.5f, 5.0f};
        std::vector<float> out(in.size(), 0.0f);

        gpuruntime::online_softmax(gpuruntime::TensorView<const float>(in.data(), {2, 4}),
                                   gpuruntime::TensorView<float>(out.data(), {2, 4}));

        for (std::size_t r = 0; r < 2; ++r) {
            double max_v = -1e300;
            for (std::size_t j = 0; j < 4; ++j) {
                max_v = std::max(max_v, static_cast<double>(in[r * 4 + j]));
            }
            double denom = 0.0;
            for (std::size_t j = 0; j < 4; ++j) {
                denom += std::exp(static_cast<double>(in[r * 4 + j]) - max_v);
            }
            double row_sum = 0.0;
            for (std::size_t j = 0; j < 4; ++j) {
                const double expected =
                    std::exp(static_cast<double>(in[r * 4 + j]) - max_v) / denom;
                TEST_CHECK_NEAR(out[r * 4 + j], expected, 1e-6);
                row_sum += out[r * 4 + j];
            }
            TEST_CHECK_NEAR(row_sum, 1.0, 1e-6);
        }
    }

    // 2. Extreme logits where naive exp(x) would overflow/underflow to NaN
    {
        std::vector<float> extreme_pos = {1000.0f, 1001.0f, 1002.0f};
        std::vector<float> out_pos(3, 0.0f);
        gpuruntime::online_softmax(gpuruntime::TensorView<const float>(extreme_pos.data(), {3}),
                                   gpuruntime::TensorView<float>(out_pos.data(), {3}));
        double sum_pos = 0.0;
        for (float v : out_pos) {
            TEST_CHECK(std::isfinite(v));
            sum_pos += v;
        }
        TEST_CHECK_NEAR(sum_pos, 1.0, 1e-6);
        TEST_CHECK(out_pos[2] > out_pos[1] && out_pos[1] > out_pos[0]);

        std::vector<float> extreme_huge = {1e30f, 1e30f, -1e30f};
        std::vector<float> out_huge(3, 0.0f);
        gpuruntime::online_softmax(gpuruntime::TensorView<const float>(extreme_huge.data(), {3}),
                                   gpuruntime::TensorView<float>(out_huge.data(), {3}));
        TEST_CHECK_NEAR(out_huge[0], 0.5, 1e-6);
        TEST_CHECK_NEAR(out_huge[1], 0.5, 1e-6);
        TEST_CHECK_NEAR(out_huge[2], 0.0, 1e-6);
    }
}

void test_rms_norm_parity() {
    std::vector<float> input = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, 1.5f, -0.5f, 2.5f};
    std::vector<float> weight = {1.0f, 0.5f, 2.0f, 1.5f};
    std::vector<float> output(input.size(), 0.0f);
    constexpr float kEps = 1e-5f;

    gpuruntime::rms_norm(gpuruntime::TensorView<const float>(input.data(), {2, 4}),
                         gpuruntime::TensorView<const float>(weight.data(), {4}),
                         gpuruntime::TensorView<float>(output.data(), {2, 4}),
                         kEps);

    for (std::size_t r = 0; r < 2; ++r) {
        double sum_sq = 0.0;
        for (std::size_t j = 0; j < 4; ++j) {
            const double x = input[r * 4 + j];
            sum_sq += x * x;
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / 4.0 + kEps);
        for (std::size_t j = 0; j < 4; ++j) {
            const double expected = input[r * 4 + j] * inv_rms * weight[j];
            TEST_CHECK_NEAR(output[r * 4 + j], expected, 1e-6);
        }
    }
}

void test_profiler_percentiles_and_json() {
    gpuruntime::TraceProfiler profiler;

    // 101 deterministic samples: 0, 1, 2, ..., 100 us
    for (int i = 0; i <= 100; ++i) {
        gpuruntime::TraceEvent ev{};
        ev.name = "kernel_\"quoted\"";
        ev.category = "compute";
        ev.pid = 1;
        ev.tid = static_cast<std::uint64_t>(i % 4);
        ev.ts_us = static_cast<double>(i * 10);
        ev.dur_us = static_cast<double>(i);
        ev.flops = 1000;
        ev.bytes = 250;
        profiler.record_event(std::move(ev));
    }

    const auto maybe_stats = profiler.stats_for("kernel_\"quoted\"");
    TEST_CHECK(maybe_stats.has_value());
    TEST_CHECK_EQ(maybe_stats->count, std::size_t{101});
    TEST_CHECK_NEAR(maybe_stats->min_us, 0.0, 1e-9);
    TEST_CHECK_NEAR(maybe_stats->mean_us, 50.0, 1e-9);
    TEST_CHECK_NEAR(maybe_stats->p50_us, 50.0, 1e-9);
    TEST_CHECK_NEAR(maybe_stats->p95_us, 95.0, 1e-9);
    TEST_CHECK_NEAR(maybe_stats->p99_us, 99.0, 1e-9);
    TEST_CHECK_NEAR(maybe_stats->max_us, 100.0, 1e-9);
    TEST_CHECK_NEAR(maybe_stats->arithmetic_intensity, 4.0, 1e-9);

    const std::string json = profiler.export_chrome_trace_json();
    TEST_CHECK(json.find("\"traceEvents\"") != std::string::npos);
    TEST_CHECK(json.find("\"ph\":\"X\"") != std::string::npos);
    TEST_CHECK(json.find("kernel_\\\"quoted\\\"") != std::string::npos);
    TEST_CHECK(json.find("\"gflops\"") != std::string::npos);
    TEST_CHECK(json.find("\"gb_per_s\"") != std::string::npos);
}

}  // namespace

int run_tensor_kernels_tests() {
    gpuruntime_test::SuiteRunner runner{"tensor_kernels"};
    runner.run("tensor_view_operations", test_tensor_view_operations);
    runner.run("sgemm_parity_against_fp64", test_sgemm_parity_against_fp64);
    runner.run("online_softmax_stability", test_online_softmax_stability);
    runner.run("rms_norm_parity", test_rms_norm_parity);
    runner.run("profiler_percentiles_and_json", test_profiler_percentiles_and_json);
    return runner.exit_code();
}
