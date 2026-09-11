# horizon_models

面向 Horizon Robotics / D-Robotics 平台的工业级 AI 模型部署工程仓库。

本仓库沉淀真实、可复现、可测试、可优化、可交付的模型部署能力。它不是学习笔记、教程、Demo、课程项目或博客仓库；仓库中的平台支持、精度和性能结论必须来自真实实现、真实编译和真实开发板验证。

## Repository Overview

`horizon_models` 以模型为中心组织从原始模型到目标平台交付物的完整生命周期，包括模型获取或训练、ONNX 导出、图与算子分析、模型修改、校准、PTQ/QAT、平台编译、BPU Runtime、Python/C++ 推理、精度评估、Benchmark、Profiling 和性能优化。

模型采用 `Capability / Task / Model` 三级目录结构。平台差异通过独立的平台配置、Docker/Toolchain 环境和 Runtime Adapter 管理，不为不同开发板复制整套模型工程。

当前目标平台：

- Horizon J5
- Horizon J6P
- RDK X5
- RDK S100

## Deployment Pipeline

```text
Original Model
  -> ONNX
  -> Validation
  -> Graph / Operator Analysis
  -> Calibration
  -> PTQ / QAT
  -> Compilation
  -> Platform Model
  -> Python / C++ Runtime
  -> Accuracy
  -> Benchmark
  -> Profiling
  -> Optimization
```

详细流程见 [docs/deployment_pipeline.md](docs/deployment_pipeline.md)。

## Model Support Matrix

状态仅允许使用 `TBD`、`Planned`、`In Progress`、`Verified` 和 `Unsupported`。各阶段状态按真实编译、运行和测试证据独立更新。

| Model | Task | Domain | J5 | J6P | X5 | S100 | ONNX | Quant | C++ | Benchmark |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s | 2D Object Detection | General | TBD | Verified | TBD | TBD | Verified | Verified | TBD | Verified |
| RT-DETR | 2D Object Detection | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RTMPose | Human Pose | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ViT Small | Vision Transformer | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Whisper | ASR | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen | LLM | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-VL | VLM | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

完整定义和更新规则见 [docs/model_support_matrix.md](docs/model_support_matrix.md)。

## Platform Matrix

V0.1 不预设不同平台共用 SDK、Compiler 或 Runtime。未知信息统一保留为 `TBD`。

| Platform | Docker Image | SDK | Toolchain | Compiler | Runtime | BPU Architecture | march | Status |
|---|---|---|---|---|---|---|---|---|
| J5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| J6P | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` | TBD | `hb_compile 3.5.16` | HBDK 4.11.11 | HBRT4 4.11.11 / libbpu 2.2.11~j6p | Nash | `nash-p` | In Progress |
| RDK X5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RDK S100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

平台环境记录要求见 [docs/platform_matrix.md](docs/platform_matrix.md)。

各平台的容器必须从可追溯镜像或构建定义独立创建，不依赖其他开发者容器的可写层。J6P 当前环境说明和容器创建入口见 [docker/j6p/README.md](docker/j6p/README.md)。

## Models

第一批正式适配模型：

- Perception / 2D Object Detection：YOLO11、RT-DETR
- Interaction / Human Pose：RTMPose
- Foundation AI / Vision Transformer：ViT Small
- Foundation AI / LLM：Qwen
- Foundation AI / VLM：Qwen-VL
- Audio / ASR：Whisper

其中 Qwen 和 Qwen-VL 的具体小型版本将在核对目标硬件能力后确定。YOLO11s 已使用 Raw6 部署边界完成 J6P 真实 COCO 校准、HBM 编译、5000 张板端精度评估和单核 Benchmark，INT8 mAP50-95 相对 FP32 下降 0.00813。掉点分析见 [YOLO11s J6P INT8 精度掉点分析与修复](docs/yolo11_j6p_int8_accuracy_recovery.md)。仓库自有 C++ Runtime 仍为 `TBD`。其他模型目录当前只包含精简元数据。仓库不提交权重、数据集或大型产物。

后续 Roadmap：

- 智驾：PointPillars、CenterPoint、BEVDet、BEVFormer、BEVFusion
- 机器人：Depth Anything V2 Small、FoundationPose、SuperPoint、LightGlue、Diffusion Policy、ACT
- 未来根据平台能力评估 VLA

Roadmap 仅表达研究方向，不代表支持承诺。

## Repository Structure

```text
horizon_models/
├── models/                 # 按 Capability / Task / Model 组织的模型工程。
├── platforms/              # J5、J6P、X5、S100 平台相关配置与适配。
├── docker/                 # 各平台相互独立的 Docker / Toolchain 环境。
├── common/                 # 可复用的预处理、后处理、评估和 Benchmark 组件。
├── tools/                  # 导出、校准、量化、编译和 Profiling 工具。
├── docs/                   # 部署规范、状态矩阵和排障文档。
└── tests/                  # 自动化测试。
```

模型真正进入部署阶段后，可按实际需要逐步增加 `configs/`、`source/`、`training/`、`finetuning/`、`export/`、`calibration/`、`quantization/`、`compile/`、`python/`、`cpp/`、`evaluation/`、`benchmark/` 和 `assets/`。不为目录完整性预先创建无意义占位工程。

## Verification Policy

平台状态只有同时满足以下条件才能标记为 `Verified`：

1. 完成目标平台模型编译；
2. 在真实目标开发板运行成功；
3. 输出正确性已经验证；
4. 提供可复现配置和运行方法。

仅 ONNX 导出成功、Compiler 编译成功、模拟器成功或官方文档声明支持，均不能标记为 `Verified`。任何 Latency、FPS、Accuracy、mAP、WER、TTFT、Token/s、Memory 或 BPU Usage 数据都必须附带真实测试条件和可追溯结果。

## License

本项目使用 [Apache License 2.0](LICENSE)。模型、数据集和第三方组件仍受各自许可证约束，纳入前必须单独核对并记录来源、版本和 License。
