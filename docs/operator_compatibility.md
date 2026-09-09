# Operator Compatibility

本文记录模型图、算子兼容性、转换问题及模型修改。V0.1 不声明任何具体算子已经在目标平台验证。

## 证据层级

算子支持结论必须明确证据来源：

1. 官方文档描述；
2. 模型图静态分析；
3. Compiler 实际编译；
4. 模拟器执行；
5. 真实开发板执行与输出正确性验证。

这些层级不可互相替代。只有真实开发板执行成功且输出正确，才能作为完整平台验证的一部分。

## 问题记录模板

| Field | Value |
|---|---|
| Model / Version | TBD |
| Platform | TBD |
| SDK / Compiler | TBD |
| ONNX Opset | TBD |
| Node / Operator | TBD |
| Input Shape / DType | TBD |
| Failure Stage | TBD |
| Error Message | TBD |
| Root Cause | TBD |
| Resolution | TBD |
| Accuracy Impact | TBD |
| Runtime Impact | TBD |
| Reproduction | TBD |

## 模型修改要求

- 优先解决根因，不跳过检查或吞掉异常。
- 修改前后使用相同输入进行数值对比。
- 记录替换的子图、属性、形状和数据类型。
- 将通用图修改能力放入 `tools/`，模型特定配置保留在模型目录。
