cmake_minimum_required(VERSION 3.10)
project(sputniPIC LANGUAGES CXX CUDA)

# Set C++ standard for both CUDA and CXX
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Visual studio optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)

if(APPLE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
    endif()
endif()

# Find source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create executable
add_executable(sputniPIC ${SOURCES})

# Set all source files to be compiled with CUDA
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

# Set CUDA properties
set_target_properties(sputniPIC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Optional: Add CUDA specific compile flags if needed
target_compile_options(sputniPIC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)