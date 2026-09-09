# Troubleshooting

本文用于记录可复现的部署故障、根因与解决方法。V0.1 仅定义记录规范，不包含未经验证的平台排障结论。

## 排障顺序

1. 固定模型、输入、配置、SDK/Runtime 和目标平台版本；
2. 保存完整命令、标准输出、标准错误和返回码；
3. 确认问题发生在导出、图分析、校准、量化、编译、加载、推理或后处理阶段；
4. 使用最小可复现输入隔离问题；
5. 比较上游 Framework、ONNX、模拟器和真实开发板输出；
6. 定位根因并验证修复，不通过跳过检查或吞异常掩盖问题；
7. 补充回归验证和影响范围。

## 故障记录模板

| Field | Value |
|---|---|
| Date | TBD |
| Model / Version | TBD |
| Platform / Board | TBD |
| Environment | TBD |
| Failure Stage | TBD |
| Command | TBD |
| Error Message | TBD |
| Minimal Reproduction | TBD |
| Root Cause | TBD |
| Resolution | TBD |
| Verification | TBD |
| Remaining Risk | TBD |

## 信息边界

- 官方宣称与本仓库实测结论分开记录。
- 进程退出码为零不等于输出正确。
- Compiler 成功不等于 Runtime 成功。
- 模拟器成功不等于真实开发板成功。
- 无足够证据时保留 `TBD`。
