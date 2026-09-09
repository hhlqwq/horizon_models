# AGENTS.md

## 项目定位

`horizon_models` 是面向 Horizon Robotics / D-Robotics 平台的工业级 AI 模型部署仓库。

核心链路：

模型 → ONNX → 算子分析 → PTQ/QAT → 编译 → BPU Runtime → C++ → 精度/性能验证 → 优化。

不是学习笔记、教程、Demo 或博客仓库。


## 核心规则

- 所有平台支持、精度和性能结论必须来自真实实现或真实测试。
- 未确认的信息使用 `TBD`，禁止猜测 SDK、Toolchain、Runtime、算子支持或性能。
- 未经真实开发板验证不得标记 `Verified`。
- 禁止伪造代码、Benchmark、测试结果和平台能力。
- 不提交公司/客户私有资产、大型权重、数据集或敏感信息。
- 优先定位根因，不通过跳过检查、吞异常等方式掩盖问题。
- 修改前先阅读当前任务相关代码、README 和配置，避免无必要重构。


## 工程组织

模型按：

`Capability / Task / Model`

组织，例如：

`models/perception/detection_2d/yolo11/`

智驾、机器人作为应用标签，不作为一级目录。

公共代码放 `common/`，通用工具放 `tools/`，平台相关内容放 `platforms/`、`docker/` 或平台配置。

禁止按开发板复制完整模型工程。


## 平台与模型

目标平台：

- J5
- J6P
- RDK X5
- RDK S100

每个平台维护独立 Docker / Toolchain 环境。

第一批模型：

- YOLO11
- RT-DETR
- RTMPose
- ViT Small
- Whisper
- Qwen LLM
- Qwen VLM

未经明确要求，不随意扩大第一批范围。


## 完成标准

状态：

`TBD / Planned / In Progress / Verified / Unsupported`

只有完成：

1. 目标平台编译；
2. 真实开发板运行；
3. 输出正确性验证；
4. 可复现配置和运行方法；

才能标记 `Verified`。


## 本地环境与安全

如存在 `AGENTS.local.md`，执行 SSH、板端测试、数据集访问或远程编译前先读取。

`AGENTS.local.md` 和 `.env.local` 仅限本地使用，禁止提交 Git。

禁止把真实 IP、账号密码、Token、SSH 凭据、个人绝对路径等本地信息写入代码、README、docs、配置模板、日志或提交信息。

程序需要本地信息时使用环境变量或 `.env.local`。

执行 `git add / commit / push` 前必须检查 staged diff，发现本地环境或敏感信息时停止提交并处理。


## 其他

- 只按实际需要创建代码和目录。
- 重点模型最终优先提供可构建的 C++ Runtime。
- 权重和数据集记录官方来源、准确版本和 License，不直接提交。
- 详细部署、量化、Benchmark、算子和排障规范放在 `docs/`，任务相关时再读取。