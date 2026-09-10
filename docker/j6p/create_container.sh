#!/usr/bin/env bash

set -Eeuo pipefail

readonly DEFAULT_IMAGE="openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1"
readonly DEFAULT_IMAGE_ID="sha256:2fd28b8ba255fecac9670467af7f4453a64ee3e39fb1eec8243cd12068a1a7c6"

# 输出错误信息并退出。
die() {
  echo "错误: $*" >&2
  exit 1
}

# 检查创建容器所需的本地条件。
check_prerequisites() {
  command -v docker >/dev/null 2>&1 || die "未找到 docker 命令。"
  [[ -n "${HORIZON_MODELS_ROOT:-}" ]] || die "必须设置 HORIZON_MODELS_ROOT。"
  [[ -d "${HORIZON_MODELS_ROOT}" ]] || die "项目目录不存在: ${HORIZON_MODELS_ROOT}"
  [[ -d "${HORIZON_MODELS_ROOT}/.git" ]] || die "项目目录不是 Git 仓库: ${HORIZON_MODELS_ROOT}"
}

# 校验本地镜像，避免静默使用漂移的同名镜像。
check_image() {
  local actual_image_id

  docker image inspect "${J6_DOCKER_IMAGE}" >/dev/null 2>&1 ||
    die "本地不存在镜像 ${J6_DOCKER_IMAGE}，请先通过官方渠道获取。"

  actual_image_id="$(docker image inspect "${J6_DOCKER_IMAGE}" --format '{{.Id}}')"
  [[ "${actual_image_id}" == "${J6_EXPECTED_IMAGE_ID}" ]] ||
    die "镜像 ID 不匹配，期望 ${J6_EXPECTED_IMAGE_ID}，实际 ${actual_image_id}。"
}

# 创建使用独立项目挂载的 J6P 工具链容器。
create_container() {
  if docker container inspect "${J6_CONTAINER_NAME}" >/dev/null 2>&1; then
    die "容器已存在: ${J6_CONTAINER_NAME}"
  fi

  docker run \
    --detach \
    --interactive \
    --tty \
    --name "${J6_CONTAINER_NAME}" \
    --hostname "${J6_CONTAINER_NAME}" \
    --network bridge \
    --runtime nvidia \
    --gpus all \
    --privileged \
    --shm-size 64m \
    --pull never \
    --mount "type=bind,src=${HORIZON_MODELS_ROOT},dst=${HORIZON_MODELS_ROOT}" \
    --workdir "${HORIZON_MODELS_ROOT}" \
    --label "com.horizon-models.platform=j6p" \
    "${J6_DOCKER_IMAGE}"

  docker ps \
    --filter "name=^/${J6_CONTAINER_NAME}$" \
    --format '容器={{.Names}} 镜像={{.Image}} 状态={{.Status}}'
}

readonly J6_DOCKER_IMAGE="${J6_DOCKER_IMAGE:-${DEFAULT_IMAGE}}"
readonly J6_EXPECTED_IMAGE_ID="${J6_EXPECTED_IMAGE_ID:-${DEFAULT_IMAGE_ID}}"
readonly J6_CONTAINER_NAME="${J6_CONTAINER_NAME:-horizon_models_j6_391}"

check_prerequisites
check_image
create_container
