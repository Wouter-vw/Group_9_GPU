# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and languages
project(sputniPIC LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Visual studio has different flag notation
if(MSVC)
    # Debug has less optimizations for profiling
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /O0")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

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

# Create executable
add_executable(sputniPIC ${SOURCES})

# Set properties for CUDA
set_target_properties(sputniPIC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_custom_target(test2D
        COMMAND sputniPIC "${CMAKE_SOURCE_DIR}/inputfiles/GEM_2D.inp"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS sputniPIC
)

add_custom_target(test3D
        COMMAND sputniPIC "${CMAKE_SOURCE_DIR}/inputfiles/GEM_3D.inp"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS sputniPIC
)