# YOLO11

YOLO11 是本仓库首批 2D 目标检测模型之一，用于建立从 PyTorch、ONNX 到 Horizon J6P HBM 的基础部署链路。

## 当前范围

- 模型变体：YOLO11s。
- 上游发布：Ultralytics Assets `v8.3.0`。
- 固定输入：`1x3x640x640`。
- ONNX Opset：17。
- 当前目标平台：J6P，`march=nash-p`。
- 当前验证类型：无真实校准数据的编译冒烟测试。
- Accuracy、Benchmark 和真实开发板验证：TBD。

## 2026-09-10 J6P 编译冒烟结果

| Field | Result |
|---|---|
| Container Image | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` |
| Image ID | `sha256:2fd28b8ba255fecac9670467af7f4453a64ee3e39fb1eec8243cd12068a1a7c6` |
| Ultralytics | 8.3.0 |
| Weight SHA-256 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| ONNX SHA-256 | `ad6f27cb298ba27a7b94cbb1ef7112f530dd0c97514bf60e9bb42f65ba77cc23` |
| ONNX Checker | Passed |
| Builder / HBDK / HMCT | 3.5.16 / 4.11.11 / 2.8.4 |
| BPU march | `nash-p` |
| Calibration | Skip，工具链使用随机数据，仅用于编译冒烟。 |
| HBM Compile | Passed |
| HBM Size | 11,352,472 bytes |
| HBM SHA-256 | `e3a615f3c5da2b7f83abbeae654ac38aff9372162e9074d8e2038cc79fd00fdb` |
| HBM Load / Info | Passed |
| Board Runtime | TBD |
| Accuracy | TBD |
| Benchmark | TBD |

工具链生成的静态 Perf 报告不属于真实开发板 Benchmark，本仓库不将其中的 FPS 或 Latency 写入性能结论。

## 目录

```text
yolo11/
├── export/
│   └── export_onnx.py
├── compile/
│   └── j6p/
│       └── compile_smoke.sh
├── metadata.yaml
└── README.md
```

权重、ONNX、HBM、日志和隔离依赖均放在被 Git 忽略的 `artifacts/` 目录，不提交仓库。

## ONNX 导出

导出脚本要求调用方提供已经核对来源的权重，并使用固定 Batch 和固定分辨率导出静态 ONNX。

```bash
PYTHONPATH=/path/to/ultralytics/python_deps \
python3 export/export_onnx.py \
  --weights artifacts/j6p_compile_smoke/yolo11s.pt \
  --output artifacts/j6p_compile_smoke/yolo11s.onnx
```

## J6P 编译冒烟测试

```bash
bash compile/j6p/compile_smoke.sh \
  artifacts/j6p_compile_smoke/yolo11s.onnx \
  artifacts/j6p_compile_smoke/compile
```

该脚本使用 `featuremap` 输入，并且不配置真实校准数据。工具链会执行伪校准，仅用于检查 ONNX 图能否经过转换、量化和 `nash-p` 编译。生成 HBM 不代表量化精度正确，也不代表真实 J6P 开发板运行成功。

## 验证边界

只有在准备代表性校准集、完成浮点与量化精度对比、在真实 J6P 开发板正确运行并提供可复现方法后，平台状态才可标记为 `Verified`。

## 参考资料

- [Ultralytics Assets v8.3.0 - YOLO11 Release](https://github.com/ultralytics/assets/releases/tag/v8.3.0)
- [Ultralytics ONNX Export](https://docs.ultralytics.com/integrations/onnx/)
- [Horizon OpenExplorer Toolchain - J6P march 映射](https://doc.oe.horizon.auto/en/guide/oe_overview/key_concept.html)
