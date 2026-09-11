# YOLO11s J6P INT8 精度掉点分析与修复

本文记录 Ultralytics YOLO11s 在 J6P 上从 FP32 ONNX 转换为 INT8 HBM 后出现严重 COCO 精度下降的定位、修复和真实开发板验证过程。

## 结论

本次严重掉点的主要原因是部署边界不合理：Ultralytics 标准 ONNX 输出为已经解码的 `1x84x8400` 张量，DFL Softmax、Sigmoid、距离到边框转换等数值敏感操作全部位于量化图内。工具链默认 PTQ 对完整图量化后，最终输出的单样本余弦相似度虽然达到 0.999257，但 COCO mAP50-95 从 0.46620 降至 0.25213，说明单样本输出余弦相似度不能代替任务精度验证。

修复方式是将 ONNX 截断为 P3、P4、P5 三个尺度的回归和分类六路原始输出，简称 Raw6。BPU 只执行主干、颈部和 Detect Head 卷积，CPU 使用 float32 完成 DFL、Sigmoid、anchor/stride 解码和 NMS。修复后的真实 J6P INT8 HBM 在 COCO 2017 val5000 上达到 0.45806 mAP50-95，相对 Raw6 FP32 ONNX 仅下降 0.00813。

100 张随机校准集没有覆盖全部 COCO 类别，是已确认的数据分布风险，但不是本次主要原因。受控 A/B 中，保持原随机校准集不变、只改用 Raw6 后，前 500 张 mAP50-95 从 0.29199 恢复到 0.49819；分层校准版本为 0.49575。

## 环境与固定条件

| 项目 | 配置 |
|---|---|
| 模型 | Ultralytics YOLO11s，Assets v8.3.0 |
| 输入 | RGB、NCHW、float32、`1x3x640x640`、除以 255 |
| ONNX Opset | 17 |
| 工具链镜像 | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` |
| Builder / HBDK / HMCT | 3.5.16 / 4.11.11 / 2.8.4 |
| BPU march | `nash-p` |
| 板端 Runtime | `hbm_infer 3.15.8`、`libbpu-runtime 2.2.11~j6p` |
| 精度数据 | COCO 2017 val，5000 张 |
| 后处理 | `conf=0.001`、NMS IoU 0.7、`max_det=300` |

## 官方资料与论坛结论

D-Robotics 官方 PTQ 指南把超过 4% 的明显掉点优先归因于转换配置、校准数据分布或流程一致性问题，并建议先检查这些因素，再使用敏感度分析、混合精度或 QAT。指南同时说明，输出余弦相似度只反映张量稳定性，与最终任务精度没有直接对应关系。

- [D-Robotics PTQ 原理、精度分析与调优](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/ptq_process)
- [OpenExplorer PTQ 部署一致性说明](https://doc.oe.horizon.auto/3.8.1/guide/model_compile/ptq/ptq_deployment_consistency.html)
- [OpenExplorer quant_config 配置说明](https://doc.oe.horizon.auto/3.8.1/guide/tools_guide/ptq_tools/hb_compile/quant_config.html)

地平线开发者社区中有与本项目相近的 J6M YOLO 案例。论坛答复建议先用全 INT16 判断是否属于位宽敏感，再使用 Debug 工具筛选敏感节点，只对少量节点使用 INT16。另一个社区实测案例表明，将完整 YOLO 图改为六路原始检测头输出，可以使严重掉点的 INT8 模型恢复到接近浮点精度。这些社区案例用于确定排查方向，最终结论仍以本项目自己的 J6P 实测为准。

- [地平线开发者社区：校准模型掉点非常厉害](https://developer.horizon.auto/forum/13303)
- [地平线开发者社区：J6M YOLOv8s INT8 精度下降排查记录](https://developer.horizon.auto/blog/14049)

## 失败现象

初始方案直接量化完整输出模型：

```text
YOLO11 backbone / neck / Detect Head
  -> DFL Softmax
  -> distance-to-box decode
  -> classification Sigmoid
  -> output [1, 84, 8400]
```

完整 5000 张 COCO 结果：

| Backend | mAP50-95 | mAP50 | mAP75 | AP Small | AP Medium | AP Large |
|---|---:|---:|---:|---:|---:|---:|
| FP32 ONNX | 0.46620 | 0.63476 | 0.50305 | 0.29270 | 0.51171 | 0.63846 |
| 完整图 J6P INT8 HBM | 0.25213 | 0.44068 | 0.26113 | 0.10036 | 0.25599 | 0.41786 |
| Absolute Delta | -0.21407 | -0.19407 | -0.24192 | -0.19234 | -0.25573 | -0.22060 |

量化报告中的低相似度节点主要集中在：

- `model.6/C3k2`，最低约 0.689926；
- `model.10/C2PSA`，最低约 0.661663；
- Detect 分类分支，最低约 0.869102；
- 图内 DFL Softmax 相关节点，最低约 0.962171。

最终输出相似度很高但 mAP 严重下降，是因为目标检测输出包含大量接近阈值的类别概率和连续边框坐标。局部误差经过 DFL、Sigmoid、解码和 NMS 后会表现为候选框消失、排序变化和 IoU 跨阈值，不能由一个整体余弦值完整描述。

## 校准集问题

初始校准集按固定随机种子从 val5000 抽取 100 张，统计结果如下：

- 只覆盖 71/80 个类别；
- 缺少 9 个类别；
- 2 张图片没有有效非 crowd 目标；
- 多个类别只有 1 到 2 个实例。

修复后的选择器先按类别稀有度完成全类别覆盖，再用固定随机种子补足样本。新校准集包含 100 张图片、897 个目标，覆盖 80/80 类，尺度分布为 small 421、medium 295、large 181。

校准集改善用于降低生产部署中的分布风险。由于 PTQ 校准针对激活范围而不是直接学习类别，本项目不把“覆盖 80 类”描述为充分条件，真实 mAP 仍是最终验收依据。

## 修复实现

### Raw6 模型边界

`export_raw6_onnx.py` 从固定版本完整 ONNX 中识别 Detect Head 末级卷积并提取以下输出：

| Scale | Regression | Classification |
|---|---|---|
| P3 / stride 8 | `1x64x80x80` | `1x80x80x80` |
| P4 / stride 16 | `1x64x40x40` | `1x80x40x40` |
| P5 / stride 32 | `1x64x20x20` | `1x80x20x20` |

评测端按每个位置 4 组、每组 16 个 bin 执行 float32 DFL Softmax 和期望计算，然后恢复三尺度 anchor center、stride、分类 Sigmoid 与 NMS。

### FP32 语义门禁

相同 50 张 COCO 图片的完整 ONNX 与 Raw6 ONNX：

- mAP50-95 差异：`4.85e-7`；
- mAP50 差异：`5.48e-7`。

完整 5000 张 Raw6 FP32 为 0.46618 mAP50-95，与完整输出 FP32 的 0.46620 一致。差异来自浮点计算顺序和极低置信度边界候选，不构成语义变化。

## 根因隔离

为避免把模型边界和校准集变化混为一个因素，额外执行以下受控实验：

| Model Boundary | Calibration | COCO Images | mAP50-95 | Delta vs FP32 |
|---|---|---:|---:|---:|
| Raw6 FP32 | 不适用 | 前 500 张 | 0.50445 | - |
| 完整 `1x84x8400` INT8 | 原随机 100 张 | 前 500 张 | 0.29199 | -0.21246 |
| Raw6 INT8 | 原随机 100 张 | 前 500 张 | 0.49819 | -0.00626 |
| Raw6 INT8 | 分层 100 张 | 前 500 张 | 0.49575 | -0.00870 |

在保持校准集、量化默认搜索、评测图片和后处理不变时，仅调整模型边界就恢复了 0.20620 mAP50-95，确认部署边界是本次严重掉点的主要原因。分层校准版本在这 500 张上比随机版本低 0.00244，因此不能声称分层选择提升了本次 COCO 精度；保留它是为了消除类别缺失和无目标样本风险，并为后续业务校准提供可审计清单。

## 最终精度

运行 ID：`j6p_raw6_ptq_coco100_20260911`。

| Backend | Images | mAP50-95 | mAP50 | mAP75 | AP Small | AP Medium | AP Large |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw6 FP32 ONNX | 5000 | 0.46618 | 0.63473 | 0.50305 | 0.29268 | 0.51171 | 0.63842 |
| Raw6 J6P INT8 HBM | 5000 | 0.45806 | 0.62555 | 0.49681 | 0.29139 | 0.50597 | 0.63363 |
| Absolute Delta | - | -0.00813 | -0.00918 | -0.00624 | -0.00129 | -0.00575 | -0.00479 |

验收门槛为 mAP50-95 绝对下降不超过 0.01，本次结果通过。

## 性能影响

真实 J6P、单 BPU Core 0、模型加载一次、float32 `1x3x640x640` 零输入、预热 100 次、正式运行 1000 次：

| Metric | 完整输出旧模型 | Raw6 修复模型 |
|---|---:|---:|
| Board Mean | 2.005 ms | 2.085 ms |
| Board P50 | 2.003 ms | 2.085 ms |
| Board P95 | 2.043 ms | 2.157 ms |
| Board P99 | 2.058 ms | 2.191 ms |
| Single-stream FPS | 498.71 | 479.51 |
| RPC Round-trip Mean | 118.462 ms | 103.104 ms |

Raw6 的纯板端平均推理增加约 0.080 ms。RPC 往返包含网络、输入和六路输出传输，不用于衡量纯模型性能，也不根据两次 RPC 数值变化推断优化收益。

## 证据与校验和

| Evidence | SHA-256 |
|---|---|
| Raw6 ONNX | `cba8bffae4480f20a3b11102bcd66bc66103fdd2168365ff05cf7af93881c20e` |
| Calibration Manifest | `b318f8903f8ebafad6fd28f9e5c3dc46d685fa0ff0193aaf10b70c899ffd27ea` |
| Raw6 HBM | `7f0a14445a58f65aff195d20e811b73bd8831bf0b776e0f5e64e314fb9f3e77b` |
| FP32 Metrics | `9f497a9d997046954e3b168e72cf7983245b434c8ac6412f91b7367b563af681` |
| J6P INT8 Metrics | `d3e93935963836dcb1b54f450b9a438695489d8396ca1c5cb6130a8558e7759a` |
| J6P Benchmark | `c059bab3dcc8d123cd6b0c336edddecbbf580e1bc889a49a2d0ac9efbc9fedd5` |
| A/B 完整图 500 张 Metrics | `54ac9e061b75bda6cdca21793af7aef985b81213c28c55709742847e2b897f6b` |
| A/B Raw6 随机校准 HBM | `af590e18a69399b09627d53296b595e22ab88959e36afba8607ec23b8f3f2fab` |
| A/B Raw6 随机校准 500 张 Metrics | `53e5e0d151e02ddbbf1c8e5d254eeae9f0642ad6c1647a8334ad0dfcd9d0606a` |

权重、ONNX、HBM、校准张量、预测 JSON 和日志位于被 Git 忽略的运行目录，不提交仓库。

## 后续策略

当前纯 INT8 已达到精度门槛，因此不启用全 INT16、局部 INT16 或 QAT。只有后续业务数据集重新出现超过验收门槛的掉点时，才按以下顺序处理：

1. 核对预处理、后处理和各阶段一致性；
2. 审核真实业务校准集分布；
3. 尝试官方建议的 max percentile、per-channel 或 layerwise search；
4. 使用敏感度分析，只把必要节点设为 INT16；
5. PTQ 仍不满足要求时再进入 QAT。

全 INT16 只适合作为敏感性诊断，不作为当前交付方案，因为它会扩大性能代价且无法说明具体敏感节点。
