cmake_minimum_required(VERSION 3.21.0)

project( TEST )

# set(Intel_C_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icc")
# set(Intel_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc")

# set(Intel_C_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/icx")
# set(Intel_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/icpx")

# set(CMAKE_C_COMPILER   ${Intel_C_COMPILER}  )
# set(CMAKE_CXX_COMPILER ${Intel_CXX_COMPILER})


SET(CMAKE_C_COMPILER "icc")
SET(CMAKE_CXX_COMPILER "icpc")

# SET(CMAKE_C_FLAGS  "-xhost -fopenmp")
# SET(CMAKE_CXX_FLAGS  "-xhost  -fopenmp")

SET(CMAKE_C_FLAGS  "-xhost -O3 -fopenmp")
SET(CMAKE_CXX_FLAGS  "-xhost -O3 -fopenmp")

# SET(CMAKE_C_FLAGS  "-fast  -fopenmp")
# SET(CMAKE_CXX_FLAGS  "-fast  -fopenmp")

set(INCLUDE_DIR 
    "include",
    "common",
    "io",
    "/opt/intel/oneapi/vtune/latest/sdk/include/"
)
include_directories (${INCLUDE_DIR}) 

# cnpy
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
add_library(cnpy SHARED "io/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

find_package( TBB REQUIRED)



# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

# exe
add_executable( TEST_tree 
    main.cpp 
    group_tree.cpp 
    common/common.cpp
    frontend/simple/xgboost_json.cpp
    frontend/simple/xgboost_json.h
    include/frontend.h
    include/group_tree.h
    common/common.h
    common/pred_transform.h
)
target_link_libraries(TEST_tree 
cnpy 
TBB::tbb
/opt/intel/oneapi/vtune/latest/sdk/lib64/libittnotify.a)