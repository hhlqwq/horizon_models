"""将 YOLO11 完整输出 ONNX 截断为六路原始检测头输出."""

import argparse
import hashlib
import re
from pathlib import Path

import onnx
from onnx import shape_inference, utils


RAW_OUTPUT_PATTERN = re.compile(
    r"^/model\.23/cv(?P<branch>[23])\.(?P<scale>[0-2])/"
    r"cv[23]\.(?P=scale)\.2/Conv$"
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="导出 YOLO11 Raw6 ONNX 模型.")
    parser.add_argument("--input", required=True, type=Path, help="完整输出 ONNX 路径.")
    parser.add_argument("--output", required=True, type=Path, help="Raw6 ONNX 输出路径.")
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_raw_outputs(model: onnx.ModelProto) -> list[str]:
    """按 Detect 末级卷积节点寻找三路回归和三路分类输出."""
    outputs: dict[tuple[int, int], str] = {}
    for node in model.graph.node:
        match = RAW_OUTPUT_PATTERN.fullmatch(node.name)
        if match is None:
            continue
        key = (int(match.group("scale")), int(match.group("branch")))
        if len(node.output) != 1:
            raise ValueError(f"Raw6 候选节点输出数量异常: {node.name}")
        outputs[key] = node.output[0]

    expected = {(scale, branch) for scale in range(3) for branch in (2, 3)}
    if set(outputs) != expected:
        missing = sorted(expected - set(outputs))
        raise ValueError(f"未找到完整 Raw6 输出, 缺少 scale/branch: {missing}")
    return [outputs[(scale, branch)] for scale in range(3) for branch in (2, 3)]


def value_shape(model: onnx.ModelProto, name: str) -> list[int | str]:
    """读取推断后的张量形状."""
    values = list(model.graph.value_info) + list(model.graph.output)
    value = next((item for item in values if item.name == name), None)
    if value is None:
        return []
    dimensions = value.type.tensor_type.shape.dim
    return [dimension.dim_value or dimension.dim_param for dimension in dimensions]


def export_raw6(input_path: Path, output_path: Path) -> None:
    """截取六路原始检测头输出并校验模型."""
    if not input_path.is_file():
        raise FileNotFoundError(f"输入 ONNX 不存在: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Raw6 输出路径不得覆盖输入 ONNX.")

    print(f"[1/3] 分析完整输出 ONNX: {input_path}", flush=True)
    inferred_model = shape_inference.infer_shapes(onnx.load(str(input_path)))
    input_names = [value.name for value in inferred_model.graph.input]
    if len(input_names) != 1:
        raise ValueError(f"预期一个模型输入, 实际为: {input_names}")
    output_names = find_raw_outputs(inferred_model)
    for name in output_names:
        print(f"Raw6 输出: {name}, shape={value_shape(inferred_model, name)}", flush=True)

    print("[2/3] 提取 Raw6 子图.", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utils.extract_model(
        str(input_path),
        str(output_path),
        input_names,
        output_names,
        check_model=True,
    )

    print("[3/3] 校验 Raw6 ONNX.", flush=True)
    raw_model = onnx.load(str(output_path))
    onnx.checker.check_model(raw_model)
    if len(raw_model.graph.output) != 6:
        raise ValueError(f"Raw6 输出数量错误: {len(raw_model.graph.output)}")
    print(f"Raw6 ONNX SHA-256: {calculate_sha256(output_path)}", flush=True)
    print(f"导出完成: {output_path}", flush=True)


def main() -> None:
    """执行 Raw6 ONNX 导出."""
    args = parse_args()
    export_raw6(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
