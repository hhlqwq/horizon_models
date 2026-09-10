"""导出固定输入尺寸的 YOLO11 ONNX 模型."""

import argparse
import hashlib
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="导出 YOLO11 静态 ONNX 模型.")
    parser.add_argument("--weights", required=True, type=Path, help="PyTorch 权重路径.")
    parser.add_argument("--output", required=True, type=Path, help="ONNX 输出路径.")
    parser.add_argument("--imgsz", default=640, type=int, help="正方形输入尺寸.")
    parser.add_argument("--opset", default=17, type=int, help="ONNX Opset 版本.")
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(weights: Path, output: Path, imgsz: int, opset: int) -> Path:
    """加载 YOLO11 权重并导出静态 ONNX."""
    if not weights.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] 加载权重: {weights}", flush=True)
    model = YOLO(str(weights))

    print(f"[2/3] 导出静态 ONNX: batch=1, imgsz={imgsz}, opset={opset}", flush=True)
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=opset,
            device="cpu",
        )
    ).resolve()

    target_path = output.resolve()
    if exported_path != target_path:
        shutil.copy2(exported_path, target_path)
    return target_path


def validate_onnx(model_path: Path) -> None:
    """检查 ONNX 结构并输出输入信息."""
    print("[3/3] 执行 ONNX 结构检查.", flush=True)
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    print(f"输入节点: {input_names}")
    print(f"输出节点: {output_names}")
    print(f"ONNX SHA-256: {calculate_sha256(model_path)}")


def main() -> None:
    """执行 YOLO11 ONNX 导出和结构验证."""
    args = parse_args()
    model_path = export_onnx(args.weights, args.output, args.imgsz, args.opset)
    validate_onnx(model_path)
    print(f"导出完成: {model_path}")


if __name__ == "__main__":
    main()
