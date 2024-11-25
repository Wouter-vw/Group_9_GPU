# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(sputniPIC CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug has less optimizations for profiling
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(APPLE)
    # Check for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
        # Check for Clang
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
    endif()
endif()

# Find all source files including .cu files
file(GLOB_RECURSE SOURCES
        "src/*.cu"
        "src/*.cpp"
)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Define CUDA-related macros as empty to avoid compilation errors
# TODO: Remove after adding CUDA
add_definitions(
        -D__host__=""
        -D__device__=""
)

# Create executable
add_executable(sputniPIC ${SOURCES})
