cmake_minimum_required(VERSION 3.20)
project(pr5_evenly_divisible C)

set(CMAKE_C_STANDARD 99)

add_executable(pr5_evenly_divisible main.c)
target_link_libraries(pr5_evenly_divisible m)
