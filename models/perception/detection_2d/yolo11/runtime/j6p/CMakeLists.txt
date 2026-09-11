cmake_minimum_required(VERSION 3.16)

project(yolo11_j6p_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED J6P_DEPS_ROOT)
  message(FATAL_ERROR "请通过 -DJ6P_DEPS_ROOT 指定官方 OE samples/ucp_tutorial/deps_aarch64 目录.")
endif()

find_path(
  J6P_UCP_INCLUDE_DIR
  NAMES hobot/dnn/hb_dnn.h
  HINTS "${J6P_DEPS_ROOT}/ucp/include"
  NO_DEFAULT_PATH)
find_path(
  J6P_OPENCV_INCLUDE_DIR
  NAMES opencv2/opencv.hpp
  HINTS "${J6P_DEPS_ROOT}/opencv/include/opencv4"
        "${J6P_DEPS_ROOT}/opencv/include"
        "${J6P_DEPS_ROOT}/ucp/include/opencv4"
  NO_DEFAULT_PATH)
find_library(
  J6P_DNN_LIBRARY
  NAMES dnn
  HINTS "${J6P_DEPS_ROOT}/ucp/lib"
  NO_DEFAULT_PATH)
find_library(
  J6P_UCP_LIBRARY
  NAMES hbucp
  HINTS "${J6P_DEPS_ROOT}/ucp/lib"
  NO_DEFAULT_PATH)
find_library(
  J6P_OPENCV_LIBRARY
  NAMES opencv_world
  HINTS "${J6P_DEPS_ROOT}/opencv/lib" "${J6P_DEPS_ROOT}/ucp/lib"
  NO_DEFAULT_PATH)

foreach(required_path
        J6P_UCP_INCLUDE_DIR
        J6P_OPENCV_INCLUDE_DIR
        J6P_DNN_LIBRARY
        J6P_UCP_LIBRARY
        J6P_OPENCV_LIBRARY)
  if(NOT ${required_path})
    message(FATAL_ERROR "官方 J6P 依赖缺失: ${required_path}.")
  endif()
endforeach()

add_executable(yolo11_j6p src/main.cc)
target_include_directories(
  yolo11_j6p PRIVATE "${J6P_UCP_INCLUDE_DIR}" "${J6P_OPENCV_INCLUDE_DIR}")
target_link_directories(yolo11_j6p PRIVATE "${J6P_DEPS_ROOT}/ucp/lib")
target_link_libraries(
  yolo11_j6p PRIVATE "${J6P_DNN_LIBRARY}" "${J6P_UCP_LIBRARY}"
                      "${J6P_OPENCV_LIBRARY}" pthread dl)
target_compile_options(yolo11_j6p PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_options(
  yolo11_j6p PRIVATE "-Wl,-rpath-link,${J6P_DEPS_ROOT}/ucp/lib"
                      "-Wl,-rpath-link,${J6P_DEPS_ROOT}/opencv/lib"
                      "-Wl,--unresolved-symbols=ignore-in-shared-libs"
                      "-Wl,-rpath,$ORIGIN/../lib")

install(TARGETS yolo11_j6p RUNTIME DESTINATION bin)
