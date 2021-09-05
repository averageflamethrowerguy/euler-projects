cmake_minimum_required(VERSION 3.20)
project(pr23_non_abundent_sums CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr23_non_abundent_sums main.cu)

set_target_properties(pr23_non_abundent_sums PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
