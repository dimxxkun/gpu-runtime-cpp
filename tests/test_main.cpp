#include "test_common.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc == 1) {
        int rc = 0;
        rc |= run_memory_pool_tests();
        rc |= run_ring_buffer_tests();
        rc |= run_task_graph_tests();
        rc |= run_tensor_kernels_tests();
        return rc;
    }

    const std::string suite = argv[1];
    if (suite == "memory_pool") {
        return run_memory_pool_tests();
    }
    if (suite == "ring_buffer") {
        return run_ring_buffer_tests();
    }
    if (suite == "task_graph") {
        return run_task_graph_tests();
    }
    if (suite == "tensor_kernels") {
        return run_tensor_kernels_tests();
    }

    std::cerr << "Unknown test suite: " << suite << "\n"
              << "Available suites: memory_pool, ring_buffer, task_graph, tensor_kernels\n";
    return 1;
}
