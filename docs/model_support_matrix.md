# Model Support Matrix

本文记录模型部署阶段与目标平台的真实支持状态。所有状态必须有可追溯证据，未知状态统一使用 `TBD`。

## 状态定义

| Status | Definition |
|---|---|
| `TBD` | 尚未评估或没有足够证据。 |
| `Planned` | 已纳入计划，但尚未开始实际适配。 |
| `In Progress` | 正在实施，尚未满足完整验证标准。 |
| `Verified` | 已满足目标平台完整验证标准。 |
| `Unsupported` | 已通过可复现实验确认不支持，并记录原因和环境。 |

## Verified 标准

平台状态只有同时满足以下条件才能标记为 `Verified`：

1. 完成目标平台模型编译；
2. 在真实目标开发板运行成功；
3. 输出结果正确；
4. 提供可复现配置和运行方法。

ONNX 导出成功、Compiler 编译成功、模拟器成功或官方文档声称支持，均不足以单独构成 `Verified`。

## V0.1 状态

| Model | Task | Domain | J5 | J6P | X5 | S100 | ONNX | Quant | C++ | Benchmark |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11 | 2D Object Detection | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RT-DETR | 2D Object Detection | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RTMPose | Human Pose | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ViT Small | Vision Transformer | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Whisper | ASR | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen | LLM | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-VL | VLM | General | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 更新要求

- 每次状态更新必须关联模型版本、平台环境、配置、命令、日志和验证结果。
- 平台状态与部署阶段状态分别维护，避免以某一阶段成功代替端到端验证。
- `Unsupported` 必须说明失败阶段、错误信息、复现条件和已核对的官方限制。
- 精度或性能结果不得直接写入状态单元格，应链接到对应评估或 Benchmark 记录。
