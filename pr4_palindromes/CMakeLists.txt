cmake_minimum_required(VERSION 3.20)
project(pr4_palindromes C)

set(CMAKE_C_STANDARD 99)

add_executable(pr4_palindromes main.c)
target_link_libraries(pr4_palindromes m)
