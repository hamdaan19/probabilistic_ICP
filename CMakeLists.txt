cmake_minimum_required(VERSION 3.0.2)
project(Probabilistic_icp)

set (CMAKE_CXX_FLAGS "-std=c++17")

find_package(Eigen3 3.3.7 REQUIRED)
find_package(PCL 1.3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(
    scan_matching_2d
    src/scan_matching_2d.cpp
)

target_link_libraries(
    scan_matching_2d
    ${EIGEN3_LIBS}
    ${PCL_LIBRARIES}
)