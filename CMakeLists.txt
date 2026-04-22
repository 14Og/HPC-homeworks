cmake_minimum_required(VERSION 3.16)
project(HW1 C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(FLAGS_COMMON
    "-Werror"
    "-Wall"
    "-Wextra"
    "-Wpedantic"
)
string(REPLACE ";" " " FLAGS_COMMON "${FLAGS_COMMON}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_COMMON}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_COMMON}")

# add_subdirectory(demos)
add_subdirectory(tasks)
