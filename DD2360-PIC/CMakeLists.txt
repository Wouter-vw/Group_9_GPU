cmake_minimum_required(VERSION 3.10)

# Set CUDA architectures before project declaration
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust for your GPU

# Project name and language (just CUDA since we're compiling everything with it)
project(sputniPIC LANGUAGES CUDA)

# Set CUDA standard to match C++11
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")
else()
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

if(APPLE)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fno-omit-frame-pointer")
    elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
    endif()
endif()

# Find source files
file(GLOB_RECURSE SOURCES
        "src/*.cu"
        "src/*.cpp"
)

# Tell CMake to treat all source files as CUDA sources
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create executable
add_executable(sputniPIC ${SOURCES})

# Set CUDA properties
set_target_properties(sputniPIC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
