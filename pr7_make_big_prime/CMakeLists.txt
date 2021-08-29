cmake_minimum_required(VERSION 3.20)
project(pr7_make_big_prime C)

set(CMAKE_C_STANDARD 99)

add_executable(pr7_make_big_prime main.c)
target_link_libraries(pr7_make_big_prime m)
