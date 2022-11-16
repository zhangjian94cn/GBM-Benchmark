cmake_minimum_required(VERSION 3.21.0)

project( TEST )

# set(Intel_C_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icc")
# set(Intel_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc")

# set(Intel_C_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/icx")
# set(Intel_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/linux/bin/icpx")

# set(CMAKE_C_COMPILER   ${Intel_C_COMPILER}  )
# set(CMAKE_CXX_COMPILER ${Intel_CXX_COMPILER})


# SET(CMAKE_C_COMPILER "icc")
# SET(CMAKE_CXX_COMPILER "icpc")

# add_subdirectory(./dmlc-core)
SET(CMAKE_CXX_FLAGS  "-mavx512f -mavx512bw")
# SET(CMAKE_CXX_FLAGS  "-mavx512f -mavx512bw -O3")


set(INCLUDE_DIR 
    "include",
    "common",
    # "frontend"
)
include_directories (${INCLUDE_DIR}) 

# dmlc-core

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
