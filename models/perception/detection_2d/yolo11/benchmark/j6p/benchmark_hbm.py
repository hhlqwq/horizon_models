"""使用 hbm_infer 在真实 J6P 上执行 YOLO11s Benchmark."""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="执行 J6P HBM Benchmark.")
    parser.add_argument("--model", required=True, type=Path, help="HBM 模型路径.")
    parser.add_argument("--output", required=True, type=Path, help="Benchmark JSON 路径.")
    parser.add_argument("--warmup", default=100, type=int, help="预热次数.")
    parser.add_argument("--iterations", default=1000, type=int, help="正式运行次数.")
    parser.add_argument("--progress-interval", default=100, type=int, help="进度输出间隔.")
    parser.add_argument("--host-env", default="J6P_HOST", help="板端地址环境变量名.")
    parser.add_argument("--user-env", default="J6P_USER", help="板端用户环境变量名.")
    parser.add_argument("--password-env", default="J6P_PASSWORD", help="板端密码环境变量名.")
    parser.add_argument("--root-env", default="J6P_WORKSPACE", help="板端工作目录环境变量名.")
    return parser.parse_args()


def required_env(name: str, allow_empty: bool = False) -> str:
    """读取必需环境变量且不输出其内容."""
    if name not in os.environ:
        raise ValueError(f"缺少环境变量: {name}")
    value = os.environ[name]
    if not value and not allow_empty:
        raise ValueError(f"环境变量不能为空: {name}")
    return value


def percentile(values: list[float], quantile: float) -> float:
    """计算线性插值百分位值."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summarize(values: list[float]) -> dict:
    """汇总毫秒耗时序列."""
    return {
        "min_ms": min(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def parse_temperatures(status: str) -> dict[str, float]:
    """从 hrut_somstatus 输出提取温度数据."""
    temperatures = {}
    in_temperature_section = False
    for line in status.splitlines():
        if line.strip() == "temperature-->":
            in_temperature_section = True
            continue
        if line.strip() == "voltage-->":
            break
        if not in_temperature_section:
            continue
        match = re.match(r"\s*([^:]+)\s*:\s*([0-9.]+)\s*\(C\)", line)
        if match:
            temperatures[match.group(1).strip()] = float(match.group(2))
    return temperatures


def main() -> None:
    """建立板端 RPC 会话并采集真实 BPU 与端到端耗时."""
    from hbm_infer.hbm_rpc_session_flexible import (
        HbmRpcSession,
        deinit_hbm,
        deinit_server,
        init_hbm,
        init_server,
    )
    from hbm_infer.utils import remote_execute

    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup 必须非负且 iterations 必须大于 0.")
    host = required_env(args.host_env)
    username = required_env(args.user_env)
    password = required_env(args.password_env, allow_empty=True)
    remote_root = required_env(args.root_env).rstrip("/") + "/hbm_infer"

    board_version = remote_execute(
        command="uname -a; dpkg-query -W libbpu-runtime bpu-fw-runtime",
        host=host,
        username=username,
        password=password,
        ssh_port=22,
    )
    status_before = remote_execute(
        command="hrut_somstatus",
        host=host,
        username=username,
        password=password,
        ssh_port=22,
    )
    print("[1/4] 初始化真实 J6P RPC Runtime.", flush=True)
    server = init_server(
        host=host,
        username=username,
        password=password,
        remote_root=remote_root,
    )
    handle = init_hbm(
        local_hbm_path=str(args.model),
        hbm_rpc_server=server,
    )
    session = HbmRpcSession(
        hbm_rpc_server=server,
        frame_timeout=300,
        hbm_handle=handle,
        core_id=[0],
        with_profile=True,
    )
    try:
        model_name = session.get_model_names()[0]
        input_info = session.get_input_info(model_name)
        inputs = {}
        for name, info in input_info.items():
            dtype = session.hbm_map_numpy[info["tensor_type"]][0]
            inputs[name] = np.zeros(tuple(info["valid_shape"]), dtype=dtype)

        print(f"[2/4] 预热 {args.warmup} 次.", flush=True)
        for index in range(args.warmup):
            session(inputs, model_name=model_name)
            if index == 0 or (index + 1) % args.progress_interval == 0:
                print(f"预热进度: {index + 1}/{args.warmup}", flush=True)

        print(f"[3/4] 正式运行 {args.iterations} 次.", flush=True)
        board_latency_ms = []
        rpc_latency_ms = []
        infer_meter = session.meters[model_name].infer_duration_meter
        for index in range(args.iterations):
            started = time.perf_counter()
            session(inputs, model_name=model_name)
            rpc_latency_ms.append((time.perf_counter() - started) * 1000.0)
            board_latency_ms.append(float(infer_meter.val))
            if index == 0 or (index + 1) % args.progress_interval == 0:
                print(f"Benchmark 进度: {index + 1}/{args.iterations}", flush=True)

        status_after = remote_execute(
            command="hrut_somstatus",
            host=host,
            username=username,
            password=password,
            ssh_port=22,
        )
        print("[4/4] 写入 Benchmark 报告.", flush=True)
        board_summary = summarize(board_latency_ms)
        report = {
            "runtime": "hbm_infer 3.15.8",
            "board_version": board_version.strip().splitlines(),
            "model": str(args.model.resolve()),
            "model_name": model_name,
            "core_id": 0,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "input_info": input_info,
            "board_inference": board_summary,
            "board_fps_single_stream": 1000.0 / board_summary["mean_ms"],
            "rpc_round_trip": summarize(rpc_latency_ms),
            "temperature_before_c": parse_temperatures(status_before),
            "temperature_after_c": parse_temperatures(status_after),
            "note": "board_inference 来自板端 UCP profile, rpc_round_trip 包含网络和张量传输.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        session.close_server()
        deinit_hbm(handle)
        deinit_server(server)


if __name__ == "__main__":
    main()
