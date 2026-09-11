#!/usr/bin/env python3
"""逐元素核对 C++ 与 Python RPC 导出的 Raw6 float32 张量."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", required=True, type=Path, help="C++ Raw6 目录.")
    parser.add_argument(
        "--python", required=True, type=Path, help="Python RPC Raw6 目录."
    )
    parser.add_argument("--output", type=Path, help="可选验证报告 JSON.")
    parser.add_argument("--atol", default=0.0, type=float, help="最大绝对误差阈值.")
    return parser.parse_args()


def load_manifest(directory: Path) -> dict[str, Any]:
    """读取并检查 Raw6 manifest."""
    with (directory / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("layout") != "NCHW" or manifest.get("dtype") != "float32":
        raise ValueError(f"不支持的 Raw6 manifest: {directory}.")
    return manifest


def compare_raw6(args: argparse.Namespace) -> dict[str, Any]:
    """逐路逐元素比较两个 Raw6 导出目录."""
    cpp_manifest = load_manifest(args.cpp)
    python_manifest = load_manifest(args.python)
    cpp_outputs = cpp_manifest["outputs"]
    python_outputs = python_manifest["outputs"]
    if len(cpp_outputs) != 6 or len(python_outputs) != 6:
        raise AssertionError(
            f"Raw6 路数不一致: C++={len(cpp_outputs)}, Python={len(python_outputs)}."
        )

    output_reports = []
    for index, (cpp_item, python_item) in enumerate(zip(cpp_outputs, python_outputs)):
        if cpp_item["shape"] != python_item["shape"]:
            raise AssertionError(
                f"第 {index} 路 shape 不一致: C++={cpp_item['shape']}, "
                f"Python={python_item['shape']}."
            )
        cpp_values = np.fromfile(args.cpp / cpp_item["file"], dtype=np.float32)
        python_values = np.fromfile(
            args.python / python_item["file"], dtype=np.float32
        )
        if cpp_values.size != python_values.size:
            raise AssertionError(
                f"第 {index} 路元素数不一致: C++={cpp_values.size}, "
                f"Python={python_values.size}."
            )
        difference = np.abs(cpp_values - python_values)
        max_error = float(difference.max(initial=0.0))
        mismatch_count = int(np.count_nonzero(difference > args.atol))
        output_reports.append(
            {
                "index": index,
                "shape": cpp_item["shape"],
                "element_count": int(cpp_values.size),
                "max_abs_error": max_error,
                "mismatch_count": mismatch_count,
            }
        )
        if mismatch_count:
            raise AssertionError(
                f"第 {index} 路存在 {mismatch_count} 个超限元素, "
                f"最大绝对误差={max_error}."
            )

    return {"status": "passed", "atol": args.atol, "outputs": output_reports}


def main() -> None:
    """执行 Raw6 一致性验证并可选保存报告."""
    args = parse_args()
    print("[1/3] 读取两端 Raw6 manifest.", flush=True)
    report = compare_raw6(args)
    total_elements = sum(item["element_count"] for item in report["outputs"])
    print(f"[2/3] 六路逐元素验证通过, 元素总数={total_elements}.", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"[3/3] 验证报告已写入: {args.output}", flush=True)
    else:
        print("[3/3] 未指定 --output, 不保存报告.", flush=True)


if __name__ == "__main__":
    main()
