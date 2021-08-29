cmake_minimum_required(VERSION 3.20)
project(pr3_largest_prime_factor C)

set(CMAKE_C_STANDARD 99)

add_executable(pr3_largest_prime_factor main.c)
target_link_libraries(pr3_largest_prime_factor m)