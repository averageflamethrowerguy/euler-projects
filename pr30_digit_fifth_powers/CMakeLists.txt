cmake_minimum_required(VERSION 3.20)
project(pr30_digit_fifth_powers CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr30_digit_fifth_powers main.cu DevArray.cuh)

set_target_properties(pr30_digit_fifth_powers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
