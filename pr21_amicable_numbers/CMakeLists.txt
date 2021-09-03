cmake_minimum_required(VERSION 3.20)
project(pr21_amicable_numbers CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr21_amicable_numbers main.cu)

set_target_properties(pr21_amicable_numbers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
