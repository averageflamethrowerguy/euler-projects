cmake_minimum_required(VERSION 3.20)
project(pr24_lexographic_permutations_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-arch=compute_86 -code=sm_86,compute_86")

add_executable(pr24_lexographic_permutations_cuda main.cu)

set_target_properties(pr24_lexographic_permutations_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
