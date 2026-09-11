#!/usr/bin/env python3
"""核对 J6P C++ 输出与 Python COCO 评估输出的一致性."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", required=True, type=Path, help="C++ 输出 JSON.")
    parser.add_argument(
        "--reference", required=True, type=Path, help="Python COCO predictions.json."
    )
    parser.add_argument(
        "--annotations", required=True, type=Path, help="COCO instances JSON."
    )
    parser.add_argument("--image-id", required=True, type=int, help="COCO image ID.")
    parser.add_argument("--output", type=Path, help="可选验证报告 JSON.")
    parser.add_argument("--score-atol", default=1e-6, type=float)
    parser.add_argument("--bbox-atol", default=1e-3, type=float)
    parser.add_argument(
        "--min-score",
        default=0.0,
        type=float,
        help="仅比较分数严格大于该值的检测框.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """读取一个 JSON 文件."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_outputs(args: argparse.Namespace) -> dict[str, Any]:
    """逐项比较检测数量、类别、分数和坐标."""
    cpp_result = load_json(args.cpp)
    reference = load_json(args.reference)
    annotations = load_json(args.annotations)
    category_ids = sorted(int(item["id"]) for item in annotations["categories"])
    if len(category_ids) != 80:
        raise ValueError(f"预期 80 个 COCO 类别, 实际为 {len(category_ids)}.")
    image_sizes = {
        int(item["id"]): (int(item["width"]), int(item["height"]))
        for item in annotations["images"]
    }
    if args.image_id not in image_sizes:
        raise ValueError(f"COCO 标注中不存在 image_id={args.image_id}.")
    image_width, image_height = image_sizes[args.image_id]

    actual = [
        item
        for item in cpp_result["detections"]
        if float(item["score"]) > args.min_score
    ]
    expected = [
        item
        for item in reference
        if int(item["image_id"]) == args.image_id
        and float(item["score"]) > args.min_score
    ]
    if len(actual) != len(expected):
        raise AssertionError(
            f"检测数量不一致: C++={len(actual)}, Python={len(expected)}."
        )

    max_score_error = 0.0
    max_bbox_error = 0.0
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        class_id = int(actual_item["class_id"])
        if class_id < 0 or class_id >= len(category_ids):
            raise AssertionError(f"第 {index} 项 class_id 越界: {class_id}.")
        actual_category = category_ids[class_id]
        expected_category = int(expected_item["category_id"])
        if actual_category != expected_category:
            raise AssertionError(
                f"第 {index} 项类别不一致: C++={actual_category}, "
                f"Python={expected_category}."
            )

        score_error = abs(float(actual_item["score"]) - float(expected_item["score"]))
        max_score_error = max(max_score_error, score_error)
        if score_error > args.score_atol:
            raise AssertionError(f"第 {index} 项分数误差超限: {score_error}.")

        x1, y1, x2, y2 = (float(value) for value in actual_item["xyxy"])
        expected_x, expected_y, expected_width, expected_height = (
            float(value) for value in expected_item["bbox"]
        )
        expected_x2 = min(max(expected_x + expected_width, 0.0), image_width)
        expected_y2 = min(max(expected_y + expected_height, 0.0), image_height)
        expected_x = min(max(expected_x, 0.0), image_width)
        expected_y = min(max(expected_y, 0.0), image_height)
        expected_bbox = [
            expected_x,
            expected_y,
            max(0.0, expected_x2 - expected_x),
            max(0.0, expected_y2 - expected_y),
        ]
        actual_bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        bbox_error = max(
            abs(actual_value - expected_value)
            for actual_value, expected_value in zip(actual_bbox, expected_bbox)
        )
        max_bbox_error = max(max_bbox_error, bbox_error)
        if bbox_error > args.bbox_atol:
            raise AssertionError(f"第 {index} 项坐标误差超限: {bbox_error}.")

    return {
        "status": "passed",
        "image_id": args.image_id,
        "detection_count": len(actual),
        "score_atol": args.score_atol,
        "bbox_atol": args.bbox_atol,
        "min_score": args.min_score,
        "max_score_error": max_score_error,
        "max_bbox_error": max_bbox_error,
    }


def main() -> None:
    """执行一致性验证并可选保存报告."""
    args = parse_args()
    print("[1/3] 读取 C++、Python 和 COCO 类别定义.", flush=True)
    report = compare_outputs(args)
    print(
        f"[2/3] 逐项验证通过, 检测数量={report['detection_count']}, "
        f"最大分数误差={report['max_score_error']:.9g}, "
        f"最大坐标误差={report['max_bbox_error']:.9g}.",
        flush=True,
    )
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
