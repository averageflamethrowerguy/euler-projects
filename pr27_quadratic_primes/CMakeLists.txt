cmake_minimum_required(VERSION 3.20)
project(pr27_quadratic_primes CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr27_quadratic_primes main.cu DevArray.cuh)

set_target_properties(pr27_quadratic_primes PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
