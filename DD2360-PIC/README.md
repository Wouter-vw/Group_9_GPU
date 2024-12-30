# SputniPIC

## Pre-requisites

- CMake
- Make
- CUDA Toolkit

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