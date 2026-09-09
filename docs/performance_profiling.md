# Performance Profiling

性能和资源数据只能来自真实测试。V0.1 不包含任何 Benchmark 数字或平台性能承诺。

## Benchmark 必备条件

- Board 与硬件版本；
- OS、SDK、Runtime、Compiler 和模型版本；
- 输入形状、精度、量化方式和 Batch；
- CPU/BPU 频率、线程、核心、功耗模式和环境温度等关键条件；
- Warmup 次数与测试轮数；
- 推理延迟与端到端延迟的统计口径；
- 可复现命令、原始日志和结果文件。

禁止编造 Latency、FPS、Accuracy、mAP、WER、TTFT、Token/s、Memory 或 BPU Usage。

## LLM / VLM 附加指标

- Context Length；
- Prompt Tokens；
- Generated Tokens；
- TTFT；
- Prefill Tokens/s；
- Decode Tokens/s；
- Peak Memory。

## Profiling 与优化

1. 建立正确性、精度和性能基线；
2. 通过 Profiling 确认瓶颈位置；
3. 每次实验尽量只改变一个主要变量；
4. 记录优化前后完整测试条件和数据；
5. 重新验证输出正确性和任务精度；
6. 只有真实收益稳定可复现时，才更新性能结论。
