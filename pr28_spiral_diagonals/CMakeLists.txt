cmake_minimum_required(VERSION 3.20)
project(pr28_spiral_diagonals CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr28_spiral_diagonals main.cu DevArray.cuh)

set_target_properties(pr28_spiral_diagonals PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
