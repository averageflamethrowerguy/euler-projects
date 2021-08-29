cmake_minimum_required(VERSION 3.20)
project(pr10_get_many_primes C)

set(CMAKE_C_STANDARD 99)

add_executable(pr10_get_many_primes main.c)
target_link_libraries(pr10_get_many_primes m)