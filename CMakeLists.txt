cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)
project(test_fp8 CUDA)

set(CMAKE_CUDA_STANDARD 20)
add_executable(test_fp8 library.cu)
set(CMAKE_CUDA_FLAGS "-O3 -Xptxas")

set_target_properties(test_fp8 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
