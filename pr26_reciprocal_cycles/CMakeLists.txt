cmake_minimum_required(VERSION 3.20)
project(pr26_reciprocal_cycles CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr26_reciprocal_cycles main.cu DevArray.cuh)

set_target_properties(pr26_reciprocal_cycles PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
