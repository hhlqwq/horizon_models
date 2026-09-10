#!/usr/bin/env bash

set -Eeuo pipefail

# 输出错误信息并退出.
die() {
  echo "错误: $*" >&2
  exit 1
}

# 将输入路径转换为绝对路径.
resolve_path() {
  local path="$1"
  local parent

  parent="$(cd "$(dirname "${path}")" && pwd)"
  echo "${parent}/$(basename "${path}")"
}

# 检查编译输入和工具链.
check_prerequisites() {
  command -v hb_compile >/dev/null 2>&1 || die "未找到 hb_compile."
  command -v hb_model_info >/dev/null 2>&1 || die "未找到 hb_model_info."
  [[ -f "${ONNX_MODEL}" ]] || die "ONNX 文件不存在: ${ONNX_MODEL}"
}

# 生成不含真实校准数据的 J6P 冒烟编译配置.
write_config() {
  cat >"${CONFIG_PATH}" <<EOF
calibration_parameters:
  quant_config: {}
compiler_parameters:
  compile_mode: latency
  core_num: 1
  jobs: 16
  optimize_level: O2
input_parameters:
  input_layout_train: NCHW
  input_name: images
  input_shape: 1x3x640x640
  input_type_rt: featuremap
  input_type_train: featuremap
model_parameters:
  march: nash-p
  onnx_model: '${ONNX_MODEL}'
  output_model_file_prefix: yolo11s_640x640_nash_p_smoke
  working_dir: '${MODEL_OUTPUT_DIR}'
EOF
}

# 执行编译并检查 HBM 产物.
compile_model() {
  local hbm_path

  echo "[1/3] 生成 J6P 冒烟编译配置: ${CONFIG_PATH}"
  write_config
  echo "[2/3] 执行 hb_compile, march=nash-p."
  hb_compile --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_PATH}"

  hbm_path="$(find "${MODEL_OUTPUT_DIR}" -type f -name '*.hbm' -print -quit)"
  [[ -n "${hbm_path}" ]] || die "编译命令结束但未找到 HBM 产物."

  echo "[3/3] 检查 HBM 元信息."
  hb_model_info "${hbm_path}" | tee "${MODEL_INFO_PATH}"
  sha256sum "${hbm_path}" | tee "${OUTPUT_DIR}/hbm.sha256"
  echo "编译完成: ${hbm_path}"
}

[[ $# -ge 1 ]] || die "用法: $0 ONNX_MODEL [OUTPUT_DIR]"

readonly ONNX_MODEL="$(resolve_path "$1")"
readonly OUTPUT_DIR="${2:-./artifacts/j6p_compile_smoke/compile}"
mkdir -p "${OUTPUT_DIR}"
readonly OUTPUT_DIR_ABS="$(cd "${OUTPUT_DIR}" && pwd)"
readonly MODEL_OUTPUT_DIR="${OUTPUT_DIR_ABS}/model_output"
readonly CONFIG_PATH="${OUTPUT_DIR_ABS}/compile_config.yaml"
readonly LOG_PATH="${OUTPUT_DIR_ABS}/hb_compile.log"
readonly MODEL_INFO_PATH="${OUTPUT_DIR_ABS}/hb_model_info.txt"

mkdir -p "${MODEL_OUTPUT_DIR}"
check_prerequisites
compile_model
