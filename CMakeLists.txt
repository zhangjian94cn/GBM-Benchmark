cmake_minimum_required(VERSION 3.0.0)
project(test VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

include(cmake/ExternalLibs.cmake)
include(cmake/Utils.cmake)
include(cmake/Version.cmake)

add_subdirectory(cpp) 