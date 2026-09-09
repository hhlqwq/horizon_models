# AGENTS.md

## 项目定位

`horizon_models` 是面向 Horizon Robotics / D-Robotics 平台的工业级 AI 模型部署仓库。

目标是形成真实、可复现、可测试、可优化、可交付的完整部署能力：

模型 → ONNX → 算子分析 → PTQ/QAT → 编译 → BPU Runtime → C++ → 精度 → Benchmark → Profiling → 优化。

本仓库不是学习笔记、教程、Demo 或博客仓库。

---

## 核心规则

1. 所有支持状态、精度和性能数据必须来自真实实现或真实硬件测试。
2. 未确认的信息统一使用 `TBD`，禁止猜测 SDK、Toolchain、Runtime、算子支持和性能。
3. 未在真实目标开发板验证的模型不得标记 `Verified`。
4. 禁止伪造代码、测试结果、Benchmark 或平台能力。
5. 不提交公司/客户私有代码、数据、模型或敏感资产。
6. 不直接提交大型模型权重和数据集，应记录来源、版本和 License。
7. 优先定位并解决根因，不通过跳过检查、吞异常等方式掩盖问题。

---

## 模型组织

统一使用三级分类：

`Capability / Task / Model`

例如：

`models/perception/detection_2d/yolo11/`

智驾和机器人属于应用标签，不作为一级目录。

禁止按照开发板复制完整模型工程，例如：

`models/j5/yolo11/`
`models/j6p/yolo11/`

平台差异通过 configs、compile、runtime adapter 等方式管理。

---

## 当前优先模型

第一批：

- YOLO11
- RT-DETR
- RTMPose
- ViT Small
- Whisper
- Qwen LLM
- Qwen VLM

后续重点：

智驾：
PointPillars、CenterPoint、BEVDet、BEVFormer、BEVFusion。

机器人：
Depth Anything V2、FoundationPose、SuperPoint、LightGlue、Diffusion Policy、ACT。

未经明确要求，不要随意扩大模型范围。

---

## 平台规则

当前目标平台：

- J5
- J6P
- RDK X5
- RDK S100

每个平台必须维护独立 Docker / Toolchain 环境：

`docker/j5/`
`docker/j6p/`
`docker/x5/`
`docker/s100/`

不要默认不同平台共用 SDK、Compiler 或 Runtime。

---

## Verified 标准

只有同时满足以下条件才能标记 `Verified`：

1. 完成目标平台编译；
2. 在真实开发板成功运行；
3. 输出正确性已经验证；
4. 有可复现配置和运行方法。

仅 ONNX 导出成功、模型编译成功、模拟器成功或官方声称支持，都不能标记 `Verified`。

状态统一使用：

`TBD / Planned / In Progress / Verified / Unsupported`

---

## 工程原则

修改前先阅读与当前任务相关的 README、模型目录、平台配置和现有代码。

遵循：

- 公共代码放 `common/`
- 通用工具放 `tools/`
- 平台相关内容放 `platforms/`、`docker/` 或平台配置
- 避免复制重复实现
- 避免无必要的大规模重构
- 避免无理由增加大型依赖
- 不创建大量无意义占位代码
- 重点模型最终优先提供可构建的 C++ Runtime，而不是只提供 Python Demo

性能优化必须有优化前后实际数据，并检查精度是否退化。

模型真正开始部署后，只按实际需要逐步增加 `configs/`、`source/`、`training/`、`finetuning/`、`export/`、`calibration/`、`quantization/`、`compile/`、`python/`、`cpp/`、`evaluation/`、`benchmark/` 和 `assets/`，不为目录完整性创建空工程。

模型权重、数据集和大型平台产物不得直接提交。必须记录其官方来源、准确版本和 License，并提供可复现的获取或生成方法。

---

## Benchmark 规则

任何性能数据必须来自真实测试，并至少记录 Board、SDK/Runtime、模型版本、输入形状、精度、量化方式、Batch、Warmup、测试轮数、推理延迟和端到端延迟。

LLM/VLM 还需记录 Context Length、Prompt Tokens、Generated Tokens、TTFT、Prefill Tokens/s、Decode Tokens/s 和 Peak Memory。

没有真实结果时使用 `TBD`，禁止填入示例数字或推测值。

---

## 详细规范

仅在任务相关时读取：

- 部署流程：`docs/deployment_pipeline.md`
- 模型状态：`docs/model_support_matrix.md`
- 平台环境：`docs/platform_matrix.md`
- 量化：`docs/quantization.md`
- 算子问题：`docs/operator_compatibility.md`
- Benchmark / Profiling：`docs/performance_profiling.md`
- 排障：`docs/troubleshooting.md`

不要无理由读取与当前任务无关的全部文档。
