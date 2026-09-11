# YOLO11

YOLO11 是本仓库首批 2D 目标检测模型之一，用于建立从 PyTorch、ONNX 到 Horizon J6P HBM 的基础部署链路。

## 当前范围与状态

- 模型变体：YOLO11s。
- 上游发布：Ultralytics Assets `v8.3.0`。
- 固定输入：RGB、NCHW、float32、`1x3x640x640`，输入值除以 255。
- ONNX Opset：17。
- 当前目标平台：J6P，`march=nash-p`。
- Raw6 ONNX、真实校准编译、Python RPC Runtime、COCO 精度和单核 Benchmark 已完成真实验证。
- J6P INT8 mAP50-95 相对 FP32 下降 0.00813，满足不超过 0.01 的验收门槛。
- 仓库自有 J6P C++ Runtime：In Progress。源码和构建入口已经建立，真实编译与板端输出一致性尚待匹配版本的官方 OE UCP 开发依赖。

## 2026-09-11 Raw6 INT8 修复结果

初始完整输出 ONNX 把 DFL、Softmax、Sigmoid 和边框解码放在量化图内，导致 mAP50-95 从 0.46620 降至 0.25213。修复后，BPU 输出三尺度回归和分类共六路原始张量，CPU 使用 float32 完成 DFL 和解码。

| Backend | Images | mAP50-95 | mAP50 | mAP75 | AP Small | AP Medium | AP Large |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw6 FP32 ONNX | 5000 | 0.46618 | 0.63473 | 0.50305 | 0.29268 | 0.51171 | 0.63842 |
| Raw6 J6P INT8 HBM | 5000 | 0.45806 | 0.62555 | 0.49681 | 0.29139 | 0.50597 | 0.63363 |
| Absolute Delta | - | -0.00813 | -0.00918 | -0.00624 | -0.00129 | -0.00575 | -0.00479 |

新校准集固定种子为 `20260911`，包含 100 张 COCO 图片和 897 个有效目标，覆盖 80/80 类；small、medium、large 目标数分别为 421、295、181。详细根因、官方资料、论坛案例、受控实验和修复说明见 [YOLO11s J6P INT8 精度掉点分析与修复](../../../../docs/yolo11_j6p_int8_accuracy_recovery.md)。

Raw6 真实 J6P Benchmark 使用单 BPU Core 0、预热 100 次、正式运行 1000 次：

| Metric | Result |
|---|---:|
| Board Inference Mean | 2.085 ms |
| Board Inference P50 | 2.085 ms |
| Board Inference P95 | 2.157 ms |
| Board Inference P99 | 2.191 ms |
| Single-stream FPS | 479.51 |
| RPC Round-trip Mean | 103.104 ms |

运行 ID：`j6p_raw6_ptq_coco100_20260911`。核心证据 SHA-256：Raw6 ONNX `cba8bffae4480f20a3b11102bcd66bc66103fdd2168365ff05cf7af93881c20e`，HBM `7f0a14445a58f65aff195d20e811b73bd8831bf0b776e0f5e64e314fb9f3e77b`，板端精度报告 `d3e93935963836dcb1b54f450b9a438695489d8396ca1c5cb6130a8558e7759a`，Benchmark `c059bab3dcc8d123cd6b0c336edddecbbf580e1bc889a49a2d0ac9efbc9fedd5`。

## 2026-09-10 完整输出 PTQ 失败基线

### 模型与工具链

| Field | Result |
|---|---|
| Container Image | `openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.9.1` |
| Image ID | `sha256:2fd28b8ba255fecac9670467af7f4453a64ee3e39fb1eec8243cd12068a1a7c6` |
| Ultralytics | 8.3.0 |
| Weight SHA-256 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| ONNX SHA-256 | `ad6f27cb298ba27a7b94cbb1ef7112f530dd0c97514bf60e9bb42f65ba77cc23` |
| Builder / HBDK / HMCT | 3.5.16 / 4.11.11 / 2.8.4 |
| Board Runtime | `hbm_infer 3.15.8` / `libbpu-runtime 2.2.11~j6p` |
| BPU march | `nash-p` |

### 真实校准与编译

校准集从 COCO 2017 `val2017` 的 5000 张图片中以固定随机种子 `20260910` 抽取 100 张。预处理使用 RGB LetterBox 640、填充值 114、NCHW float32 和除以 255，与模型运行输入保持一致。

| Field | Result |
|---|---|
| Calibration Samples | 100 |
| Calibration Manifest SHA-256 | `3e86f1129863d7a93fcb06363a8846ad9d05bac7d88000e9e0e22b8e505bc820` |
| Selected Calibration Method | `max-percentile=0.99995` |
| Quantized Output Cosine | 0.999257，工具链单样本张量检查。 |
| HBM Compile / Load | Passed |
| HBM Size | 12,004,920 bytes |
| HBM SHA-256 | `ecc36604779f410b8eeacc80586a7fbb461da26fdfb417c5b88062d62f5c9fc9` |

### COCO 2017 精度

FP32 ONNX 和真实 J6P HBM 均评估完整的 5000 张 `val2017`。两者使用相同预处理、`conf=0.001`、NMS IoU 0.7、`max_det=300` 和 COCOeval。

| Backend | Images | mAP50-95 | mAP50 | mAP75 | AP Small | AP Medium | AP Large |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 ONNX | 5000 | 0.46620 | 0.63476 | 0.50305 | 0.29270 | 0.51171 | 0.63846 |
| J6P HBM | 5000 | 0.25213 | 0.44068 | 0.26113 | 0.10036 | 0.25599 | 0.41786 |
| Absolute Delta | - | -0.21407 | -0.19407 | -0.24192 | -0.19234 | -0.25573 | -0.22060 |

该完整输出 PTQ 精度不合格。相同 `conf=0.001` 下，FP32 ONNX 产生 675,644 个 NMS 后候选，J6P HBM 仅产生 359,185 个，说明量化后大量低置信度候选消失。量化报告同时显示多个骨干和注意力节点余弦相似度偏低，包括：

- `/model.10/m/m.0/ffn/ffn.0/act/Mul`：0.661663。
- `/model.10/m/m.0/ffn/ffn.1/conv/Conv`：0.664993。
- `/model.6/m.0/m/m.1/cv1/act/Mul`：0.689926。
- `/model.6/m.0/cv3/act/Mul`：0.712353。
- `/model.10/m/m.0/attn/proj/conv/Conv`：0.761098。

Raw6 修复已经满足精度门槛，因此当前不需要引入混合精度或 QAT。不得仅凭输出层单样本余弦相似度判断检测精度通过。

### 真实 J6P Benchmark

测试条件：J6P、单 BPU Core 0、模型加载一次、float32 `1x3x640x640` 零输入、预热 100 次、正式运行 1000 次。`board_inference` 来自板端 UCP profile；RPC 往返包含约 4.9 MB 输入、约 2.8 MB 输出和网络传输，不作为纯模型延迟。

| Metric | Result |
|---|---:|
| Board Inference Mean | 2.005 ms |
| Board Inference P50 | 2.003 ms |
| Board Inference P95 | 2.043 ms |
| Board Inference P99 | 2.058 ms |
| Board Inference Min / Max | 1.955 / 2.101 ms |
| Single-stream FPS | 498.71 |
| RPC Round-trip Mean | 118.462 ms |
| BPU Temperature Max Before / After | 43.528 / 46.285 C |

编译器静态 Perf 报告不属于真实开发板 Benchmark，不用于上述性能结论。

### 证据文件

运行 ID：`j6p_ptq_coco100_20260910`。以下文件位于被 Git 忽略的同名 artifact 目录中。

| Evidence | SHA-256 |
|---|---|
| Calibration Manifest | `3e86f1129863d7a93fcb06363a8846ad9d05bac7d88000e9e0e22b8e505bc820` |
| HBM | `ecc36604779f410b8eeacc80586a7fbb461da26fdfb417c5b88062d62f5c9fc9` |
| FP32 ONNX Metrics | `cf37fbabd77e2b917ed057ebd265aa899aae13915b3680b0406ebed117765a36` |
| J6P HBM Metrics | `25ba10d367dc280f055428be58223b613564ad01272a305eb5a292068f258de3` |
| J6P Benchmark | `9cf2fd694bb716eab98d439381c99d4140382c7721817d992f04b16d6acaa896` |

## 目录

```text
yolo11/
├── export/
│   ├── export_onnx.py
│   └── export_raw6_onnx.py
├── calibration/
│   └── prepare_coco_calibration.py
├── compile/
│   └── j6p/
│       ├── compile_smoke.sh
│       └── compile_ptq.sh
├── evaluation/
│   └── evaluate_coco.py
├── benchmark/
│   └── j6p/
│       └── benchmark_hbm.py
├── runtime/
│   └── j6p/
│       ├── CMakeLists.txt
│       ├── build.sh
│       └── src/
│           └── main.cc
├── metadata.yaml
└── README.md
```

权重、ONNX、HBM、校准张量、预测 JSON、日志和隔离依赖均放在被 Git 忽略的 `artifacts/` 目录，不提交仓库。

## 可复现流程

以下变量仅表示本地路径。开发板地址、用户、密码和工作目录必须由未提交的 `.env.local` 或进程环境提供，不得写入仓库。

```bash
export MODEL_ROOT=models/perception/detection_2d/yolo11
export ARTIFACT_ROOT="${MODEL_ROOT}/artifacts/j6p_ptq_coco100"
export COCO_ROOT=/path/to/coco
```

### ONNX 导出

```bash
PYTHONPATH=/path/to/ultralytics/python_deps \
python3 "${MODEL_ROOT}/export/export_onnx.py" \
  --weights "${ARTIFACT_ROOT}/yolo11s.pt" \
  --output "${ARTIFACT_ROOT}/yolo11s.onnx"

python3 "${MODEL_ROOT}/export/export_raw6_onnx.py" \
  --input "${ARTIFACT_ROOT}/yolo11s.onnx" \
  --output "${ARTIFACT_ROOT}/yolo11s_raw6.onnx"
```

### 真实校准数据

```bash
python3 "${MODEL_ROOT}/calibration/prepare_coco_calibration.py" \
  --images "${COCO_ROOT}/images/val2017" \
  --annotations "${COCO_ROOT}/annotations/instances_val2017.json" \
  --output "${ARTIFACT_ROOT}/calibration" \
  --count 100 \
  --seed 20260910 \
  --imgsz 640
```

输出目录必须为空。脚本将 NPY 放在 `calibration/inputs_f32/`，避免工具链把清单 JSON 误读为张量。

### J6P PTQ 编译

```bash
MODEL_PREFIX=yolo11s_640x640_nash_p_raw6_ptq \
bash "${MODEL_ROOT}/compile/j6p/compile_ptq.sh" \
  "${ARTIFACT_ROOT}/yolo11s_raw6.onnx" \
  "${ARTIFACT_ROOT}/calibration" \
  "${ARTIFACT_ROOT}/compile"
```

### COCO 精度

```bash
PYTHONPATH=/path/to/ultralytics/python_deps \
python3 "${MODEL_ROOT}/evaluation/evaluate_coco.py" \
  --backend onnx \
  --model "${ARTIFACT_ROOT}/yolo11s_raw6.onnx" \
  --images "${COCO_ROOT}/images/val2017" \
  --annotations "${COCO_ROOT}/annotations/instances_val2017.json" \
  --output "${ARTIFACT_ROOT}/evaluation/onnx_full5000"

PYTHONPATH=/path/to/ultralytics/python_deps \
python3 "${MODEL_ROOT}/evaluation/evaluate_coco.py" \
  --backend hbm-board \
  --model "${ARTIFACT_ROOT}/compile/model_output/yolo11s_640x640_nash_p_raw6_ptq.hbm" \
  --images "${COCO_ROOT}/images/val2017" \
  --annotations "${COCO_ROOT}/annotations/instances_val2017.json" \
  --output "${ARTIFACT_ROOT}/evaluation/hbm_board_full5000"
```

板端后端从 `J6P_HOST`、`J6P_USER`、`J6P_PASSWORD` 和 `J6P_WORKSPACE` 环境变量读取本地连接信息。评估每 50 张保存断点；再次使用同一输出目录会跳过已完成图片。

### J6P Benchmark

```bash
PYTHONPATH=/path/to/ultralytics/python_deps \
python3 "${MODEL_ROOT}/benchmark/j6p/benchmark_hbm.py" \
  --model "${ARTIFACT_ROOT}/compile/model_output/yolo11s_640x640_nash_p_raw6_ptq.hbm" \
  --output "${ARTIFACT_ROOT}/benchmark/j6p_core0_warmup100_iter1000.json" \
  --warmup 100 \
  --iterations 1000
```

### J6P C++ Runtime

C++ Runtime 使用官方 UCP/DNN API 加载 HBM，并在 CPU float32 中完成六路 Raw6 的 DFL、Sigmoid、三尺度解码和按类别 NMS。输入与 Python 精度评估保持一致：OpenCV `INTER_LINEAR`、居中 LetterBox、填充值 114、BGR 转 RGB、NCHW float32 和除以 255。

构建必须使用与目标 Runtime 匹配的官方 OpenExplorer 发布包依赖，不能从其他开发者容器的私人挂载复制。将 `J6P_DEPS_ROOT` 指向 OE 发布包的 `samples/ucp_tutorial/deps_aarch64`：

```bash
export J6P_DEPS_ROOT=/path/to/horizon_j6_open_explorer/samples/ucp_tutorial/deps_aarch64
bash "${MODEL_ROOT}/runtime/j6p/build.sh"
```

官方 `libdnn.so` 会声明 `libbpu`、`libhbmem` 等板端系统库依赖。交叉链接仅解析本程序直接使用的 DNN、UCP 和 OpenCV 接口，并忽略共享库尚未解析的板端符号；交付时不要从 `deps_aarch64/appsdk` 捆绑旧版系统库，避免覆盖开发板已安装且与固件配套的 BPU Runtime。

运行示例：

```bash
export LD_LIBRARY_PATH=/path/to/deployed/ucp/lib:${LD_LIBRARY_PATH:-}
"${MODEL_ROOT}/runtime/j6p/build/yolo11_j6p" \
  --model /path/to/yolo11s_640x640_nash_p_raw6_ptq.hbm \
  --image /path/to/image.jpg \
  --output /path/to/detections.json \
  --conf 0.25 \
  --iou 0.7 \
  --max-det 300 \
  --warmup 10 \
  --iterations 100 \
  --dump-dir /path/to/raw6_dump
```

`--warmup` 和 `--iterations` 用于在同一进程内复用 HBM，输出 JSON 同时记录平均 BPU 推理、CPU 后处理和二者合计耗时。`--dump-dir` 可选；启用后会把六路有效输出剔除物理 padding 后按连续 NCHW float32 保存，并生成 shape manifest，便于与 Python 后端逐元素比较。

程序按 HBM 返回的 byte stride 访问输入输出，避免把物理 padding 当成有效输出。当前只接受固定的 `1x3x640x640` float32 输入和六路 float32 Raw6 输出；接口属性不匹配时立即失败。正式更新为 `Verified` 前，仍需完成以下门槛：

1. 使用与当前 J6P Runtime 匹配的官方 `deps_aarch64` 完成 AArch64 交叉编译；
2. 在真实 J6P 上加载当前 HBM 并运行；
3. 固定输入下逐路比较 C++ 与 Python RPC 的有效输出；
4. 比较最终检测框、分数和类别，并完成端到端 Benchmark。

## 验证边界

Raw6 INT8 HBM 已完成真实 J6P 编译、运行、完整 COCO 输出正确性和 Benchmark 验证，并提供可复现脚本，因此 J6P 模型部署状态为 `Verified`。仓库自有 C++ Runtime 已进入开发阶段，但尚未完成官方依赖下的交叉编译、真实板端运行和输出一致性验证，独立保持 `In Progress`。

## 参考资料

- [Ultralytics Assets v8.3.0 - YOLO11 Release](https://github.com/ultralytics/assets/releases/tag/v8.3.0)
- [Ultralytics ONNX Export](https://docs.ultralytics.com/integrations/onnx/)
- [Horizon OpenExplorer Toolchain - J6P march 映射](https://doc.oe.horizon.auto/en/guide/oe_overview/key_concept.html)
- [D-Robotics PTQ 原理、精度分析与调优](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/ptq_process)
- [地平线开发者社区 - 校准模型掉点非常厉害](https://developer.horizon.auto/forum/13303)
