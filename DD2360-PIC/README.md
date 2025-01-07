# SputniPIC

## Pre-requisites

- CMake >= 3.14
- Make
- CUDA Toolkit

Modify 'CMakeLists.txt' to include an appropriate CUDA architecture (if needed) for the target device. Currently, has only be set for architecture 75.

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75) 
```

## Running

1. Build the CMake project and compile

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B ./build
cd build
make
```

or `cmake -DCMAKE_BUILD_TYPE=Debug -S . -B ./build` for a debug build.

2. Run the executable

```bash
./sputniPIC ../inputfiles/GEM_2D.in
```