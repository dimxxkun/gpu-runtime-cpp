#pragma once

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gpuruntime_test {

class AssertionFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct SuiteRunner {
    std::string suite_name;
    int passed = 0;
    int failed = 0;

    template <typename Fn>
    void run(const std::string& test_name, Fn&& fn) {
        try {
            fn();
            ++passed;
            std::cout << "  [PASS] " << suite_name << "::" << test_name << "\n";
        } catch (const std::exception& ex) {
            ++failed;
            std::cerr << "  [FAIL] " << suite_name << "::" << test_name
                      << " -> " << ex.what() << "\n";
        } catch (...) {
            ++failed;
            std::cerr << "  [FAIL] " << suite_name << "::" << test_name
                      << " -> unknown exception\n";
        }
    }

    [[nodiscard]] int exit_code() const noexcept { return failed == 0 ? 0 : 1; }
};

}  // namespace gpuruntime_test

#define TEST_CHECK(cond)                                                            \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::ostringstream _test_oss;                                           \
            _test_oss << __FILE__ << ":" << __LINE__ << " CHECK failed: " #cond;    \
            throw ::gpuruntime_test::AssertionFailure(_test_oss.str());             \
        }                                                                           \
    } while (false)

#define TEST_CHECK_EQ(lhs, rhs)                                                     \
    do {                                                                            \
        const auto _test_lhs = (lhs);                                               \
        const auto _test_rhs = (rhs);                                               \
        if (!(_test_lhs == _test_rhs)) {                                            \
            std::ostringstream _test_oss;                                           \
            _test_oss << __FILE__ << ":" << __LINE__                                \
                      << " CHECK_EQ failed: " #lhs " (" << _test_lhs                \
                      << ") != " #rhs " (" << _test_rhs << ")";                     \
            throw ::gpuruntime_test::AssertionFailure(_test_oss.str());             \
        }                                                                           \
    } while (false)

#define TEST_CHECK_NEAR(lhs, rhs, tol)                                              \
    do {                                                                            \
        const double _test_lhs = static_cast<double>(lhs);                          \
        const double _test_rhs = static_cast<double>(rhs);                          \
        const double _test_tol = static_cast<double>(tol);                          \
        if (!(std::abs(_test_lhs - _test_rhs) <= _test_tol)) {                      \
            std::ostringstream _test_oss;                                           \
            _test_oss << __FILE__ << ":" << __LINE__                                \
                      << " CHECK_NEAR failed: |" << _test_lhs << " - "              \
                      << _test_rhs << "| > " << _test_tol;                          \
            throw ::gpuruntime_test::AssertionFailure(_test_oss.str());             \
        }                                                                           \
    } while (false)

#define TEST_CHECK_THROWS(expr, exc_type)                                           \
    do {                                                                            \
        bool _test_caught = false;                                                  \
        try {                                                                       \
            static_cast<void>(expr);                                                \
        } catch (const exc_type&) {                                                 \
            _test_caught = true;                                                    \
        }                                                                           \
        if (!_test_caught) {                                                        \
            std::ostringstream _test_oss;                                           \
            _test_oss << __FILE__ << ":" << __LINE__                                \
                      << " CHECK_THROWS failed: expected " #exc_type                \
                      << " from " #expr;                                            \
            throw ::gpuruntime_test::AssertionFailure(_test_oss.str());             \
        }                                                                           \
    } while (false)

// Suite entrypoints implemented in individual test translation units.
int run_memory_pool_tests();
int run_ring_buffer_tests();
int run_task_graph_tests();
int run_tensor_kernels_tests();
