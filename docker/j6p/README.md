# J6P Docker / Toolchain

本目录维护 J6P 独立工具链容器的创建方法。容器直接基于经过核对的官方镜像创建，不依赖其他开发者容器的可写层。

## 当前基线

| Field | Value | Evidence |
|---|---|---|
| Image | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` | 现有主机镜像标签。 |
| Image ID | `sha256:2fd28b8ba255fecac9670467af7f4453a64ee3e39fb1eec8243cd12068a1a7c6` | 现有主机 `docker image inspect`。 |
| Registry Digest | TBD | 当前本地镜像未记录 RepoDigest。 |
| Base OS | Ubuntu 22.04 | 镜像标签与容器内 `/etc/os-release`。 |
| Container Runtime | NVIDIA Runtime | 现有主机与源容器配置。 |
| J6 SDK | TBD | 尚未完成官方资料与真实板端核对。 |
| Toolchain UI | `hb_compile 3.5.16` | 新建干净容器实测。 |
| Compiler | HBDK 4.11.11 / HMCT 2.8.4 | 新建干净容器实测。 |
| Runtime Library | HBRT4 4.11.11 | HBM 加载日志。 |
| J6P BPU / march | Nash / `nash-p` | 官方映射文档与编译产物。 |
| Board Verification | TBD | 尚未在真实 J6P 开发板验证。 |

镜像 ID 用于防止同名标签发生静默漂移。由于当前没有 Registry Digest，跨主机获取镜像时仍需通过官方渠道确认来源。

## 依赖来源调查

镜像历史表明，环境通过镜像构建层统一安装，而不是在启动容器时临时安装。构建层包含 Ubuntu 22.04、CUDA 12.8、系统编译工具、AArch64 交叉编译器、Python 3.10 及 Horizon 工具链相关 Python 包。

原始镜像构建过程中使用了 `docker_install.sh`，但该脚本没有保留在最终镜像中。因此，本仓库当前不能从裸 Ubuntu 完整重建该官方环境，也不会根据镜像历史反向拼接一个未经验证的 Dockerfile。

以下关键依赖已在新建的干净容器中重新核对：

| Package | Observed Version |
|---|---|
| Python | 3.10.12 |
| `hbdk4-compiler` | 4.11.11 |
| `hbdk4-march` | 4.11.11 |
| `hmct-gpu` | 2.8.4+cu128 |
| `horizon-plugin-profiler` | 3.3.10 |
| `horizon-plugin-pytorch` | 3.3.10+cu128.torch280 |
| `horizon_tc_ui` | 3.5.16 |
| `onnx` | 1.15.0 |
| `onnxruntime-gpu` | 1.19.0 |
| `torch` | 2.8.0+cu128 |
| `torchvision` | 0.23.0+cu128 |

源容器的可写层包含个人工具、缓存、历史记录和额外环境变化。禁止使用 `docker commit` 将其固化为新镜像，也禁止复用其他开发者的私人宿主机挂载。

## 创建容器

创建脚本要求通过环境变量提供本机项目路径，避免在可提交文件中写入个人绝对路径。

```bash
export HORIZON_MODELS_ROOT=/path/to/horizon_models
export J6_CONTAINER_NAME=horizon_models_j6_391
bash docker/j6p/create_container.sh
```

可选变量：

| Variable | Default | Description |
|---|---|---|
| `J6_DOCKER_IMAGE` | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` | 官方工具链镜像标签。 |
| `J6_EXPECTED_IMAGE_ID` | 当前基线镜像 ID | 创建前必须匹配的镜像 ID。 |
| `J6_CONTAINER_NAME` | `horizon_models_j6_391` | 本地容器名称。 |

脚本不会自动下载镜像，不会删除或覆盖同名容器，也不会读取其他容器的文件系统。项目目录以宿主机原路径映射到容器内相同路径，并作为容器工作目录。

## 核对容器

```bash
docker ps --filter name=horizon_models_j6_391
docker inspect horizon_models_j6_391
docker exec -it horizon_models_j6_391 bash
```

容器创建成功只代表工具链环境能够启动，不代表 YOLO11 已导出、已编译或已在真实 J6P 开发板验证。

## 参考资料

- [Horizon OpenExplorer Toolchain - Key Concepts](https://doc.oe.horizon.auto/en/guide/oe_overview/key_concept.html)
- [Horizon OpenExplorer Toolchain - 模型量化编译](https://doc.oe.horizon.auto/3.8.1/guide/tools_guide/ptq_tools/hb_compile/convert.html)
