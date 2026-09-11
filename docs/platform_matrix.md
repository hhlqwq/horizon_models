# Platform Matrix

J5、J6P、RDK X5 和 RDK S100 分别维护独立的 Docker / Toolchain 环境。除非经过官方资料核对和实际验证，不假设不同平台共用 SDK、Compiler、Runtime 或 BPU 配置。

## V0.1 平台信息

| Platform | Docker Image | SDK | Toolchain | Compiler | Runtime | Python | PyTorch | BPU Architecture | march | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| J5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| J6P | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` | TBD | `hb_compile 3.5.16` | HBDK 4.11.11 | HBRT4 4.11.11 / libbpu 2.2.11~j6p | 3.10.12 | 2.8.0+cu128 | Nash | `nash-p` | In Progress |
| RDK X5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RDK S100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 每个平台必须记录的信息

- 官方 Docker Image 或经过验证的 Dockerfile 来源；
- SDK、Toolchain、Compiler 和 Runtime 的准确版本；
- Python、PyTorch 及关键依赖版本；
- BPU Architecture 和 `march`；
- Host 与 Board 的软硬件要求；
- 必要环境变量；
- 环境启动、模型编译和 Runtime 构建命令；
- 官方文档来源与真实验证记录。

如果官方仅提供现成 Docker Image，则记录镜像地址、Digest、License 和启动方式，不为目录完整性编造 Dockerfile。

## 目录映射

| Platform | Platform Files | Docker / Toolchain Files |
|---|---|---|
| J5 | `platforms/j5/` | `docker/j5/` |
| J6P | `platforms/j6p/` | `docker/j6p/` |
| RDK X5 | `platforms/x5/` | `docker/x5/` |
| RDK S100 | `platforms/s100/` | `docker/s100/` |
