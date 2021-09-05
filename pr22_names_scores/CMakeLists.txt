cmake_minimum_required(VERSION 3.20)
project(pr22_names_scores CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-arch=compute_86 -code=sm_86,compute_86")

add_executable(pr22_names_scores main.cu)

set_target_properties(pr22_names_scores PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
