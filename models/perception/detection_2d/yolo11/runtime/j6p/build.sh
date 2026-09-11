#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
J6P_DEPS_ROOT="${J6P_DEPS_ROOT:-}"
TOOLCHAIN_ROOT="${J6P_TOOLCHAIN_ROOT:-/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu}"

if [[ -z "${J6P_DEPS_ROOT}" ]]; then
  echo "错误: 请设置 J6P_DEPS_ROOT，指向官方 OE samples/ucp_tutorial/deps_aarch64 目录." >&2
  exit 2
fi

if [[ ! -f "${J6P_DEPS_ROOT}/ucp/include/hobot/dnn/hb_dnn.h" ]]; then
  echo "错误: 未找到官方 UCP 头文件: ${J6P_DEPS_ROOT}/ucp/include/hobot/dnn/hb_dnn.h" >&2
  exit 2
fi

CXX_PATH="${J6P_CXX:-${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++}"
if [[ ! -x "${CXX_PATH}" ]]; then
  echo "错误: 未找到 AArch64 交叉编译器: ${CXX_PATH}" >&2
  exit 2
fi

OPENCV_INCLUDE=""
for candidate in \
  "${J6P_DEPS_ROOT}/opencv/include/opencv4" \
  "${J6P_DEPS_ROOT}/opencv/include" \
  "${J6P_DEPS_ROOT}/ucp/include/opencv4"; do
  if [[ -f "${candidate}/opencv2/opencv.hpp" ]]; then
    OPENCV_INCLUDE="${candidate}"
    break
  fi
done
if [[ -z "${OPENCV_INCLUDE}" ]]; then
  echo "错误: 官方 deps_aarch64 中未找到 opencv2/opencv.hpp." >&2
  exit 2
fi

OPENCV_LIB_DIR="${J6P_DEPS_ROOT}/opencv/lib"
if [[ ! -f "${OPENCV_LIB_DIR}/libopencv_world.so" ]]; then
  OPENCV_LIB_DIR="${J6P_DEPS_ROOT}/ucp/lib"
fi
if [[ ! -f "${OPENCV_LIB_DIR}/libopencv_world.so" ]]; then
  echo "错误: 官方 deps_aarch64 中未找到 libopencv_world.so." >&2
  exit 2
fi

mkdir -p "${BUILD_DIR}"
echo "[1/3] 核对官方 UCP、OpenCV 和交叉编译器依赖."
echo "编译器: ${CXX_PATH}"
echo "UCP: ${J6P_DEPS_ROOT}/ucp"
echo "OpenCV: ${OPENCV_INCLUDE}"

echo "[2/3] 编译 yolo11_j6p."
"${CXX_PATH}" \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -I"${J6P_DEPS_ROOT}/ucp/include" \
  -I"${OPENCV_INCLUDE}" \
  "${SCRIPT_DIR}/src/main.cc" \
  -L"${J6P_DEPS_ROOT}/ucp/lib" \
  -L"${OPENCV_LIB_DIR}" \
  -Wl,-rpath-link,"${J6P_DEPS_ROOT}/ucp/lib" \
  -Wl,-rpath-link,"${OPENCV_LIB_DIR}" \
  -Wl,--unresolved-symbols=ignore-in-shared-libs \
  -Wl,-rpath,'$ORIGIN/../lib' \
  -ldnn \
  -lhbucp \
  -lopencv_world \
  -lpthread \
  -ldl \
  -o "${BUILD_DIR}/yolo11_j6p"

echo "[3/3] 构建完成: ${BUILD_DIR}/yolo11_j6p"
