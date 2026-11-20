cmake_minimum_required(VERSION 3.20)

project(gpu_runtime_cpp
    VERSION 1.0.0
    DESCRIPTION "Modern C++20 Host-Side GPU Runtime & Heterogeneous Systems Library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_NATIVE_SIMD "Enable native host SIMD instructions (-march=native)" OFF)
option(BUILD_TESTING "Build unit test suite" ON)

find_package(Threads REQUIRED)

add_library(gpuruntime_warnings INTERFACE)
if(MSVC)
    target_compile_options(gpuruntime_warnings INTERFACE /W4 /WX)
else()
    target_compile_options(gpuruntime_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
endif()

if(ENABLE_SANITIZERS)
    if(MSVC)
        target_compile_options(gpuruntime_warnings INTERFACE /fsanitize=address)
    else()
        target_compile_options(gpuruntime_warnings INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -g
        )
        target_link_options(gpuruntime_warnings INTERFACE
            -fsanitize=address,undefined
        )
    endif()
endif()

if(ENABLE_NATIVE_SIMD AND NOT MSVC)
    target_compile_options(gpuruntime_warnings INTERFACE -march=native)
endif()

add_library(gpuruntime STATIC
    src/memory_pool.cpp
    src/task_graph.cpp
    src/kernels.cpp
    src/profiler.cpp
)

target_include_directories(gpuruntime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(gpuruntime
    PUBLIC
        Threads::Threads
    PRIVATE
        gpuruntime_warnings
)

add_executable(gpuruntime_bench
    src/bench_main.cpp
)

target_link_libraries(gpuruntime_bench
    PRIVATE
        gpuruntime
        gpuruntime_warnings
)

if(BUILD_TESTING)
    enable_testing()

    add_executable(gpuruntime_tests
        tests/test_main.cpp
        tests/test_memory_pool.cpp
        tests/test_ring_buffer.cpp
        tests/test_task_graph.cpp
        tests/test_tensor_kernels.cpp
    )

    target_include_directories(gpuruntime_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(gpuruntime_tests
        PRIVATE
            gpuruntime
            gpuruntime_warnings
    )

    add_test(NAME memory_pool COMMAND gpuruntime_tests memory_pool)
    add_test(NAME ring_buffer COMMAND gpuruntime_tests ring_buffer)
    add_test(NAME task_graph COMMAND gpuruntime_tests task_graph)
    add_test(NAME tensor_kernels COMMAND gpuruntime_tests tensor_kernels)
endif()
