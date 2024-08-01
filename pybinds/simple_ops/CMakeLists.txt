cmake_minimum_required(VERSION 3.0.0)
project(Projectpybind11 VERSION 0.1.0)
include(CTest)
enable_testing()

add_subdirectory(pybind11)
pybind11_add_module(SumFunction main.cpp)

pybind11_add_module(numpy_demo2 main.cpp)

pybind11_add_module(numpy_demo1 main.cpp)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)