cmake_minimum_required(VERSION 3.20)
project(pr29_distinct_powers CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr29_distinct_powers main.cu DevArray.cuh)

set_target_properties(pr29_distinct_powers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
