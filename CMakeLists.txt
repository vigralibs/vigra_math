cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(math VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vigralibs_cmake")

include(VigraAddExternal)
vigra_add_external(core REPO https://github.com/vigralibs/core)

add_library(math INTERFACE)
TARGET_LINK_LIBRARIES(math INTERFACE core)

target_include_directories(math INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

enable_testing()
add_subdirectory(test)
