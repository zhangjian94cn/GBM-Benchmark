include(FetchContent)
include(cmake/FetchContentMakeAvailable.cmake)


# RapidJSON (header-only library)
add_library(rapidjson INTERFACE)
find_package(RapidJSON)
if(RapidJSON_FOUND)
  target_include_directories(rapidjson INTERFACE ${RAPIDJSON_INCLUDE_DIRS})
else()
  message(STATUS "Did not find RapidJSON in the system root. Fetching RapidJSON now...")
  FetchContent_Declare(
    RapidJSON
    GIT_REPOSITORY      https://github.com/Tencent/rapidjson
    GIT_TAG             v1.1.0
  )
  FetchContent_Populate(RapidJSON)
  message(STATUS "RapidJSON was downloaded at ${rapidjson_SOURCE_DIR}.")
  target_include_directories(rapidjson INTERFACE $<BUILD_INTERFACE:${rapidjson_SOURCE_DIR}/include>)
endif()
add_library(RapidJSON::rapidjson ALIAS rapidjson)