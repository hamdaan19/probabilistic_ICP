cmake_minimum_required(VERSION 3.0.2)
project(probabilistic_ICP)

set (CMAKE_CXX_FLAGS "-std=c++17")
set (CMAKE_BUILD_TYPE "Debug")

find_package(Eigen3 3.3.7 REQUIRED)
find_package(PCL 1.3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(
    transform
    src/transform_cloud.cpp
)

target_link_libraries(
    transform
    ${EIGEN3_LIBS}
    ${PCL_LIBRARIES}
)