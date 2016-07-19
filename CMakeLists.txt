cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(vigra_math VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vigra_cmake")

include(VigraAddExternal)
vigra_add_external(vigra_core REPO https://github.com/vigralibs/vigra_core)

add_library(vigra_math INTERFACE)
TARGET_LINK_LIBRARIES(vigra_math INTERFACE vigra_core)

target_include_directories(vigra_math INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

enable_testing()
add_subdirectory(test)
