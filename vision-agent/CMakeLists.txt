cmake_minimum_required(VERSION 3.16)
project(vision-agent VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(VA_WITH_REALSENSE "Build with RealSense camera support" OFF)
option(VA_WITH_NEON "Build ARM NEON assembly kernels" OFF)
option(VA_WITH_GPIO "Build GPIO motor HAL (Raspberry Pi)" OFF)
option(VA_BUILD_TESTS "Build unit tests" ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
    if(VA_WITH_NEON)
        add_compile_options(-march=armv8-a+simd)
    endif()
endif()

add_library(va_core STATIC
    src/util/timing.c
    src/preprocess/preprocess.c
    src/perception/perception.c
    src/fusion/occupancy.c
    src/agent/fsm.c
    src/ipc/ipc_shm.c
    src/ipc/ipc_client.c
)
target_include_directories(va_core PUBLIC src)
target_link_libraries(va_core PRIVATE rt)

if(VA_WITH_REALSENSE)
    find_package(realsense2 REQUIRED)
    target_sources(va_core PRIVATE src/capture/capture_realsense.c)
    target_link_libraries(va_core PRIVATE realsense2::realsense2)
    target_compile_definitions(va_core PUBLIC VA_HAS_REALSENSE=1)
else()
    target_sources(va_core PRIVATE src/capture/capture_stub.c)
endif()

if(VA_WITH_GPIO)
    target_sources(va_core PRIVATE src/hal/motor_gpio.c)
    target_compile_definitions(va_core PUBLIC VA_HAS_GPIO=1)
else()
    target_sources(va_core PRIVATE src/hal/motor_stub.c)
endif()

add_executable(vision-agent src/main.c)
target_link_libraries(vision-agent PRIVATE va_core)

if(VA_BUILD_TESTS)
    enable_testing()

    add_executable(test_preprocess tests/test_preprocess.c)
    target_link_libraries(test_preprocess PRIVATE va_core)
    add_test(NAME preprocess COMMAND test_preprocess)

    add_executable(test_occupancy tests/test_occupancy.c)
    target_link_libraries(test_occupancy PRIVATE va_core)
    add_test(NAME occupancy COMMAND test_occupancy)

    add_executable(test_fsm tests/test_fsm.c)
    target_link_libraries(test_fsm PRIVATE va_core)
    add_test(NAME fsm COMMAND test_fsm)

    add_executable(test_ipc tests/test_ipc.c)
    target_link_libraries(test_ipc PRIVATE va_core)
    add_test(NAME ipc COMMAND test_ipc)
endif()
