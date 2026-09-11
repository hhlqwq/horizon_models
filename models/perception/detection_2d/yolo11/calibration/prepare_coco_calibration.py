"""从 COCO 图片生成 YOLO11 float32 校准数据."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="生成 YOLO11 COCO 校准数据.")
    parser.add_argument("--images", required=True, type=Path, help="COCO 图片目录.")
    parser.add_argument("--annotations", type=Path, help="COCO instances 标注 JSON.")
    parser.add_argument("--output", required=True, type=Path, help="校准数据输出目录.")
    parser.add_argument("--count", default=100, type=int, help="校准图片数量.")
    parser.add_argument("--seed", default=20260910, type=int, help="固定随机种子.")
    parser.add_argument("--imgsz", default=640, type=int, help="模型输入尺寸.")
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(images_dir: Path) -> list[Path]:
    """收集并排序支持的图片文件."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in suffixes
    )
    if not images:
        raise FileNotFoundError(f"图片目录中没有支持的文件: {images_dir}")
    return images


def select_annotated_images(
    images: list[Path],
    annotations_path: Path,
    count: int,
    seed: int,
) -> tuple[list[Path], dict]:
    """按类别覆盖优先策略选择有目标的 COCO 校准图片."""
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in annotations["images"]}
    path_by_name = {path.name: path for path in images}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    category_frequency: Counter[int] = Counter()
    for item in annotations["annotations"]:
        if item.get("iscrowd", 0):
            continue
        image_id = int(item["image_id"])
        annotations_by_image[image_id].append(item)
        category_frequency[int(item["category_id"])] += 1

    eligible: dict[int, Path] = {}
    for image_id, items in annotations_by_image.items():
        image_info = image_by_id.get(image_id)
        if not items or image_info is None:
            continue
        image_path = path_by_name.get(image_info["file_name"])
        if image_path is not None:
            eligible[image_id] = image_path
    if count > len(eligible):
        raise ValueError(f"有标注图片仅 {len(eligible)} 张, 无法抽取 {count} 张.")

    random_generator = random.Random(seed)
    tie_breaker = {image_id: random_generator.random() for image_id in eligible}
    uncovered = set(category_frequency)
    selected_ids: list[int] = []
    remaining = set(eligible)
    while uncovered and len(selected_ids) < count:
        def coverage_score(image_id: int) -> tuple[float, float]:
            """计算单张图片对未覆盖类别的稀有度加权贡献."""
            categories = {
                int(item["category_id"])
                for item in annotations_by_image[image_id]
                if int(item["category_id"]) in uncovered
            }
            score = sum(1.0 / category_frequency[category_id] for category_id in categories)
            return score, tie_breaker[image_id]

        selected_id = max(remaining, key=coverage_score)
        selected_categories = {
            int(item["category_id"]) for item in annotations_by_image[selected_id]
        }
        if not (selected_categories & uncovered):
            break
        selected_ids.append(selected_id)
        remaining.remove(selected_id)
        uncovered -= selected_categories

    candidates = sorted(remaining)
    random_generator.shuffle(candidates)
    selected_ids.extend(candidates[: count - len(selected_ids)])
    selected = sorted((eligible[image_id] for image_id in selected_ids), key=lambda path: path.name)

    selected_annotations = [
        item for image_id in selected_ids for item in annotations_by_image[image_id]
    ]
    category_counts = Counter(int(item["category_id"]) for item in selected_annotations)
    size_counts = Counter(
        "small"
        if float(item["area"]) < 32**2
        else "medium"
        if float(item["area"]) < 96**2
        else "large"
        for item in selected_annotations
    )
    category_names = {
        int(item["id"]): item["name"] for item in annotations["categories"]
    }
    statistics = {
        "selection": "coco_category_coverage_then_seeded_random",
        "annotation_sha256": calculate_sha256(annotations_path),
        "object_count": len(selected_annotations),
        "covered_category_count": len(category_counts),
        "missing_categories": [
            category_names[category_id]
            for category_id in sorted(category_names)
            if category_id not in category_counts
        ],
        "category_counts": {
            category_names[category_id]: category_counts[category_id]
            for category_id in sorted(category_counts)
        },
        "size_counts": dict(sorted(size_counts.items())),
    }
    if statistics["missing_categories"]:
        raise RuntimeError(
            f"{count} 张校准图片未覆盖全部类别: {statistics['missing_categories']}"
        )
    return selected, statistics


def letterbox(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """按 Ultralytics LetterBox 语义缩放并居中填充图片."""
    height, width = image.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    padding_width = imgsz - resized_width
    padding_height = imgsz - resized_height
    left = round(padding_width / 2 - 0.1)
    right = round(padding_width / 2 + 0.1)
    top = round(padding_height / 2 - 0.1)
    bottom = round(padding_height / 2 + 0.1)
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return image, ratio, left, top


def preprocess(image_path: Path, imgsz: int) -> tuple[np.ndarray, dict]:
    """读取图片并生成 RGB NCHW float32 输入."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片读取失败: {image_path}")
    original_height, original_width = image.shape[:2]
    image, ratio, pad_left, pad_top = letterbox(image, imgsz)
    tensor = image[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    metadata = {
        "source": image_path.name,
        "source_sha256": calculate_sha256(image_path),
        "original_shape": [original_height, original_width],
        "ratio": ratio,
        "padding": [pad_left, pad_top],
    }
    return tensor, metadata


def ensure_empty_output(output_dir: Path) -> None:
    """创建空输出目录并拒绝覆盖已有内容."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录不是空目录, 拒绝覆盖: {output_dir}")


def main() -> None:
    """抽样 COCO 图片并写入校准张量和清单."""
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count 必须大于 0.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0.")

    images = collect_images(args.images)
    if args.count > len(images):
        raise ValueError(f"请求 {args.count} 张, 但仅找到 {len(images)} 张图片.")
    selection_statistics = {"selection": "seeded_random"}
    if args.annotations is not None:
        if not args.annotations.is_file():
            raise FileNotFoundError(f"标注文件不存在: {args.annotations}")
        selected, selection_statistics = select_annotated_images(
            images,
            args.annotations,
            args.count,
            args.seed,
        )
    else:
        selected = sorted(random.Random(args.seed).sample(images, args.count))
    ensure_empty_output(args.output)
    tensor_dir = args.output / "inputs_f32"
    tensor_dir.mkdir()

    manifest = {
        "dataset": "COCO 2017 val",
        "sample_count": args.count,
        "seed": args.seed,
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "color_order": "RGB",
        "normalization": "value / 255.0",
        "letterbox_padding": 114,
        "selection_statistics": selection_statistics,
        "samples": [],
    }
    print(f"[1/2] 开始生成 {args.count} 份真实校准数据.", flush=True)
    for index, image_path in enumerate(selected, start=1):
        tensor, metadata = preprocess(image_path, args.imgsz)
        output_path = tensor_dir / f"{image_path.stem}.npy"
        np.save(output_path, tensor)
        metadata["calibration_file"] = str(output_path.relative_to(args.output))
        metadata["calibration_sha256"] = calculate_sha256(output_path)
        manifest["samples"].append(metadata)
        if index == 1 or index % 10 == 0 or index == args.count:
            print(f"校准数据进度: {index}/{args.count}", flush=True)

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[2/2] 校准数据完成: {tensor_dir}", flush=True)
    print(f"清单: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
