cmake_minimum_required(VERSION 3.22)
project(TEM)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -march=native -w")


add_executable(temgraph tem_graph.h main_tem_graph.cpp)


