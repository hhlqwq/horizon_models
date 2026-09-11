#!/usr/bin/env python3
"""导出 Python J6P RPC 后端的六路 Raw6 有效张量."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from evaluate_coco import HbmBoardBackend, preprocess, required_env


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="J6P HBM 路径.")
    parser.add_argument("--image", required=True, type=Path, help="输入图片路径.")
    parser.add_argument("--output", required=True, type=Path, help="输出目录.")
    parser.add_argument("--imgsz", default=640, type=int, help="模型输入尺寸.")
    return parser.parse_args()


def write_outputs(outputs: list[np.ndarray], output_dir: Path) -> None:
    """按连续 NCHW float32 保存输出并生成 shape manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"layout": "NCHW", "dtype": "float32", "outputs": []}
    for index, output in enumerate(outputs):
        tensor = np.ascontiguousarray(output, dtype=np.float32)
        filename = f"output_{index}.bin"
        tensor.tofile(output_dir / filename)
        manifest["outputs"].append(
            {"index": index, "file": filename, "shape": list(tensor.shape)}
        )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    """执行单图 RPC 推理并导出 Raw6 张量."""
    args = parse_args()
    print("[1/4] 按评估链路预处理图片.", flush=True)
    tensor, _, _, _, _ = preprocess(args.image, args.imgsz)
    print("[2/4] 建立真实 J6P RPC 会话.", flush=True)
    backend = HbmBoardBackend(
        args.model,
        required_env("J6P_HOST"),
        required_env("J6P_USER"),
        os.environ.get("J6P_PASSWORD"),
        required_env("J6P_WORKSPACE"),
    )
    try:
        print("[3/4] 执行一次 BPU 推理.", flush=True)
        outputs = backend.run(tensor)
        if len(outputs) != 6:
            raise ValueError(f"预期六路 Raw6 输出, 实际为 {len(outputs)}.")
        write_outputs(outputs, args.output)
    finally:
        backend.close()
    print(f"[4/4] Raw6 输出已写入: {args.output}", flush=True)


if __name__ == "__main__":
    main()
