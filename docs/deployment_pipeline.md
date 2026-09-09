# Deployment Pipeline

本文定义模型从上游来源到 Horizon Robotics / D-Robotics 目标平台交付物的标准流程。各阶段只有在产生可追溯产物和验证记录后才能更新状态。

## 流程总览

```text
Original Model
  -> ONNX
  -> Validation
  -> Graph / Operator Analysis
  -> Model Modification
  -> Calibration
  -> PTQ / QAT
  -> Compilation
  -> Platform Model
  -> Python / C++ Runtime
  -> Accuracy Evaluation
  -> Benchmark
  -> Profiling
  -> Optimization
```

## 阶段要求

### 1. 模型来源

- 记录官方来源、模型名称、准确版本或提交、Framework 和 License。
- 不直接提交大型权重或数据集。
- 训练或微调模型必须记录配置、数据来源和可复现方法。

### 2. ONNX 导出与验证

- 固定输入输出名称、形状、动态维度和 Opset。
- 保存导出命令与环境版本。
- 使用代表性输入对比上游模型和 ONNX 输出，并记录误差标准。

### 3. Graph / Operator Analysis

- 记录图结构、关键算子、动态形状和潜在不兼容节点。
- 区分官方文档声明、工具静态分析、模拟器结果和真实平台结果。
- 模型修改必须说明原因、实现和正确性对比。

### 4. Calibration 与量化

- 记录校准数据来源、筛选规则、样本数量和预处理。
- PTQ/QAT 配置必须可追溯。
- 比较量化前后精度，不能只检查编译是否成功。

### 5. 平台编译

- 使用目标平台独立的 SDK、Toolchain、Compiler 和配置。
- 保存完整编译命令、日志、输入模型校验信息和输出产物信息。
- 编译成功不等于开发板运行成功，也不等于平台状态为 `Verified`。

### 6. Runtime

- Python Runtime 用于快速正确性验证和问题定位。
- 重点模型最终优先提供可构建的 C++ Runtime。
- 前处理、后处理和张量布局必须与导出及评估链路保持一致。

### 7. Accuracy、Benchmark 与 Profiling

- Accuracy 使用明确的数据集、版本、指标实现和评估配置。
- Benchmark 使用真实开发板并记录完整测试条件。
- Profiling 结果用于定位瓶颈，优化后必须重新检查正确性与精度。

## 状态边界

仅 ONNX 导出、Compiler 编译或模拟器执行成功时，不得将目标平台标记为 `Verified`。`Verified` 的完整判定标准见 [model_support_matrix.md](model_support_matrix.md)。
