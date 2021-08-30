cmake_minimum_required(VERSION 3.20)
project(pr12_triangle_number)

set(CMAKE_CXX_STANDARD 14)

add_executable(pr12_triangle_number main.cpp)
target_link_libraries(pr12_triangle_number m)