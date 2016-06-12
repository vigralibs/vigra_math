cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(math VERSION 1.0 LANGUAGES CXX)

add_library(math INTERFACE)

target_include_directories(math INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)
