cmake_minimum_required(VERSION 3.20)
project(pr25_1000_digit_fib_number CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pr25_1000_digit_fib_number main.cu)

set_target_properties(pr25_1000_digit_fib_number PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
