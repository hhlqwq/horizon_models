"""评估 YOLO11 ONNX、HBM 模拟器或真实 J6P 的 COCO 精度."""

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils.ops import non_max_suppression


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="评估 YOLO11 COCO 精度.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=("onnx", "hbm-sim", "hbm-board"),
        help="推理后端.",
    )
    parser.add_argument("--model", required=True, type=Path, help="ONNX 或 HBM 路径.")
    parser.add_argument("--images", required=True, type=Path, help="COCO 图片目录.")
    parser.add_argument("--annotations", required=True, type=Path, help="COCO 标注 JSON.")
    parser.add_argument("--output", required=True, type=Path, help="结果输出目录.")
    parser.add_argument("--imgsz", default=640, type=int, help="模型输入尺寸.")
    parser.add_argument("--conf", default=0.001, type=float, help="候选框置信度阈值.")
    parser.add_argument("--iou", default=0.7, type=float, help="NMS IoU 阈值.")
    parser.add_argument("--max-det", default=300, type=int, help="单图最大检测数.")
    parser.add_argument("--limit", default=0, type=int, help="仅处理前 N 张, 0 表示全部.")
    parser.add_argument("--progress-interval", default=50, type=int, help="进度输出间隔.")
    parser.add_argument("--board-host-env", default="J6P_HOST", help="板端地址环境变量名.")
    parser.add_argument("--board-user-env", default="J6P_USER", help="板端用户环境变量名.")
    parser.add_argument(
        "--board-password-env",
        default="J6P_PASSWORD",
        help="板端密码环境变量名.",
    )
    parser.add_argument(
        "--board-root-env",
        default="J6P_WORKSPACE",
        help="板端工作目录环境变量名.",
    )
    return parser.parse_args()


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


def preprocess(image_path: Path, imgsz: int) -> tuple[np.ndarray, tuple[int, int], float, int, int]:
    """读取图片并返回模型输入和几何变换参数."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片读取失败: {image_path}")
    original_shape = image.shape[:2]
    image, ratio, pad_left, pad_top = letterbox(image, imgsz)
    tensor = image[:, :, ::-1].transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    return tensor, original_shape, ratio, pad_left, pad_top


def normalize_prediction(output: np.ndarray) -> np.ndarray:
    """将后端输出规范为 1x84x8400."""
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"不支持的输出形状: {output.shape}")
    if output.shape[1] == 84:
        return output.astype(np.float32, copy=False)
    if output.shape[2] == 84:
        return output.transpose(0, 2, 1).astype(np.float32, copy=False)
    raise ValueError(f"输出中没有 84 通道维度: {output.shape}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    """稳定计算 Sigmoid."""
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    """稳定计算 Softmax."""
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def as_nchw(output: np.ndarray, channels: int) -> np.ndarray:
    """将 Raw6 输出规范为 NCHW."""
    output = np.asarray(output)
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError(f"Raw6 输出形状错误: {output.shape}")
    if output.shape[1] == channels:
        return output.astype(np.float32, copy=False)
    if output.shape[-1] == channels and output.shape[1] != channels:
        return output.transpose(0, 3, 1, 2).astype(np.float32, copy=False)
    raise ValueError(f"Raw6 输出中没有唯一的 {channels} 通道维度: {output.shape}")


def decode_raw6(outputs: list[np.ndarray], imgsz: int) -> np.ndarray:
    """在 CPU 上完成三尺度 DFL、Sigmoid 和边框解码."""
    if len(outputs) != 6:
        raise ValueError(f"Raw6 必须包含 6 个输出, 实际为 {len(outputs)}.")

    scale_outputs: dict[int, dict[int, np.ndarray]] = {}
    for output in outputs:
        array = np.asarray(output)
        if array.ndim != 4:
            raise ValueError(f"Raw6 输出必须为四维张量: {array.shape}")
        channel_candidates = [axis for axis in (1, 3) if array.shape[axis] in (64, 80)]
        if not channel_candidates:
            raise ValueError(f"无法判断 Raw6 通道维度: {array.shape}")
        channel_axis = channel_candidates[0]
        channels = int(array.shape[channel_axis])
        nchw = as_nchw(array, channels)
        height, width = nchw.shape[2:]
        if height != width or imgsz % height != 0:
            raise ValueError(f"Raw6 特征图尺寸错误: {nchw.shape}")
        scale_outputs.setdefault(height, {})[channels] = nchw

    expected_sizes = {imgsz // stride for stride in (8, 16, 32)}
    if set(scale_outputs) != expected_sizes:
        raise ValueError(f"Raw6 特征尺度错误: {sorted(scale_outputs)}")

    boxes_by_scale = []
    scores_by_scale = []
    bins = np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)
    for size in sorted(scale_outputs, reverse=True):
        tensors = scale_outputs[size]
        if set(tensors) != {64, 80}:
            raise ValueError(f"尺度 {size} 缺少回归或分类输出: {sorted(tensors)}")
        regression = tensors[64].reshape(1, 4, 16, size * size)
        distances = np.sum(softmax(regression, axis=2) * bins, axis=2)
        classification = sigmoid(tensors[80]).reshape(1, 80, size * size)

        grid_y, grid_x = np.meshgrid(
            np.arange(size, dtype=np.float32),
            np.arange(size, dtype=np.float32),
            indexing="ij",
        )
        anchor_x = grid_x.reshape(1, -1) + 0.5
        anchor_y = grid_y.reshape(1, -1) + 0.5
        stride = float(imgsz // size)
        x1 = (anchor_x - distances[:, 0]) * stride
        y1 = (anchor_y - distances[:, 1]) * stride
        x2 = (anchor_x + distances[:, 2]) * stride
        y2 = (anchor_y + distances[:, 3]) * stride
        boxes = np.stack(
            ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
            axis=1,
        )
        boxes_by_scale.append(boxes)
        scores_by_scale.append(classification)

    return np.concatenate(
        (
            np.concatenate(boxes_by_scale, axis=2),
            np.concatenate(scores_by_scale, axis=2),
        ),
        axis=1,
    )


def decode_outputs(outputs: list[np.ndarray], imgsz: int) -> np.ndarray:
    """自动识别完整输出或 Raw6 输出并规范为 1x84x8400."""
    if len(outputs) == 1:
        return normalize_prediction(outputs[0])
    return decode_raw6(outputs, imgsz)


class OnnxBackend:
    """ONNX Runtime 推理后端."""

    def __init__(self, model_path: Path) -> None:
        """加载 ONNX 模型."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        selected = [provider for provider in providers if provider in available]
        self.session = ort.InferenceSession(str(model_path), providers=selected)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        """执行一次 ONNX 推理."""
        return self.session.run(self.output_names, {self.input_name: tensor})

    def close(self) -> None:
        """释放后端资源."""


class HbmSimBackend:
    """HBDK HBM 模拟器推理后端."""

    def __init__(self, model_path: Path) -> None:
        """加载 HBM 模型."""
        from horizon_tc_ui.hb_hbmruntime import HB_HBMRuntime

        self.runtime = HB_HBMRuntime(str(model_path))
        self.input_name = self.runtime.input_names[0]
        self.output_names = list(self.runtime.output_names)

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        """执行一次 HBM 模拟推理."""
        return list(self.runtime.run_sim(
            output_names=self.output_names,
            input_info={self.input_name: tensor},
        ))

    def close(self) -> None:
        """释放后端资源."""


class HbmBoardBackend:
    """真实 J6P RPC 推理后端."""

    def __init__(
        self,
        model_path: Path,
        host: str,
        username: str,
        password: str,
        remote_root: str,
    ) -> None:
        """创建板端服务、上传 HBM 并建立持久会话."""
        from hbm_infer.hbm_rpc_session_flexible import (
            HbmRpcSession,
            init_hbm,
            init_server,
        )

        self._deinit_hbm = None
        self._deinit_server = None
        self.server = init_server(
            host=host,
            username=username,
            password=password,
            remote_root=f"{remote_root.rstrip('/')}/hbm_infer",
        )
        self.handle = init_hbm(
            local_hbm_path=str(model_path),
            hbm_rpc_server=self.server,
        )
        self.session = HbmRpcSession(
            hbm_rpc_server=self.server,
            frame_timeout=300,
            hbm_handle=self.handle,
            core_id=[0],
        )
        model_names = self.session.get_model_names()
        if len(model_names) != 1:
            raise ValueError(f"预期一个 HBM 图, 实际为: {model_names}")
        self.model_name = model_names[0]
        self.input_name = next(iter(self.session.get_input_info(self.model_name)))
        self.output_names = list(self.session.get_output_info(self.model_name))

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        """在真实 J6P 上执行一次推理."""
        result = self.session({self.input_name: tensor}, model_name=self.model_name)
        return [result[name] for name in self.output_names]

    def close(self) -> None:
        """关闭会话并清理工具链上传的临时文件."""
        from hbm_infer.hbm_rpc_session_flexible import deinit_hbm, deinit_server

        if getattr(self, "session", None) is not None:
            self.session.close_server()
            self.session = None
        if getattr(self, "handle", None) is not None:
            deinit_hbm(self.handle)
            self.handle = None
        if getattr(self, "server", None) is not None:
            deinit_server(self.server)
            self.server = None


def required_env(name: str, allow_empty: bool = False) -> str:
    """读取必需环境变量且不输出其内容."""
    if name not in os.environ:
        raise ValueError(f"缺少环境变量: {name}")
    value = os.environ[name]
    if not value and not allow_empty:
        raise ValueError(f"环境变量不能为空: {name}")
    return value


def create_backend(args: argparse.Namespace) -> Any:
    """按参数创建推理后端."""
    if args.backend == "onnx":
        return OnnxBackend(args.model)
    if args.backend == "hbm-sim":
        return HbmSimBackend(args.model)
    return HbmBoardBackend(
        args.model,
        required_env(args.board_host_env),
        required_env(args.board_user_env),
        required_env(args.board_password_env, allow_empty=True),
        required_env(args.board_root_env),
    )


def restore_boxes(
    boxes: torch.Tensor,
    original_shape: tuple[int, int],
    ratio: float,
    pad_left: int,
    pad_top: int,
) -> torch.Tensor:
    """将 LetterBox 坐标还原到原图坐标."""
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, :4] /= ratio
    height, width = original_shape
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)
    return boxes


def build_detections(
    outputs: list[np.ndarray],
    image_id: int,
    category_ids: list[int],
    original_shape: tuple[int, int],
    ratio: float,
    pad_left: int,
    pad_top: int,
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int,
) -> list[dict]:
    """执行 YOLO11 NMS 并转换为 COCO 检测格式."""
    prediction = torch.from_numpy(decode_outputs(outputs, imgsz).copy())
    detections = non_max_suppression(
        prediction,
        conf_thres=conf,
        iou_thres=iou,
        multi_label=True,
        agnostic=False,
        max_det=max_det,
        nc=80,
    )[0]
    if not len(detections):
        return []
    restore_boxes(detections[:, :4], original_shape, ratio, pad_left, pad_top)

    results = []
    for x1, y1, x2, y2, score, class_index in detections.cpu().tolist():
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        results.append(
            {
                "image_id": image_id,
                "category_id": category_ids[int(class_index)],
                "bbox": [round(x1, 3), round(y1, 3), round(width, 3), round(height, 3)],
                "score": round(score, 7),
            }
        )
    return results


def save_checkpoint(
    output_dir: Path,
    predictions: list[dict],
    completed_image_ids: list[int],
) -> None:
    """保存可续跑的检测结果和已完成图片清单."""
    (output_dir / "predictions.partial.json").write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "progress.json").write_text(
        json.dumps({"completed_image_ids": completed_image_ids}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(output_dir: Path) -> tuple[list[dict], list[int]]:
    """读取已有断点, 文件不完整时拒绝静默续跑."""
    predictions_path = output_dir / "predictions.partial.json"
    progress_path = output_dir / "progress.json"
    if not predictions_path.exists() and not progress_path.exists():
        return [], []
    if not predictions_path.exists() or not progress_path.exists():
        raise RuntimeError(f"断点文件不完整, 请检查: {output_dir}")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    return predictions, [int(value) for value in progress["completed_image_ids"]]


def evaluate_predictions(
    coco: COCO,
    predictions: list[dict],
    image_ids: list[int],
) -> tuple[dict, str]:
    """运行 COCOeval 并返回核心指标和原始摘要."""
    if not predictions:
        raise RuntimeError("没有生成任何检测结果, 无法执行 COCOeval.")
    result_set = coco.loadRes(predictions)
    evaluator = COCOeval(coco, result_set, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluator.summarize()
    summary = buffer.getvalue()
    print(summary, end="", flush=True)
    metrics = {
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "map_75": float(evaluator.stats[2]),
        "map_small": float(evaluator.stats[3]),
        "map_medium": float(evaluator.stats[4]),
        "map_large": float(evaluator.stats[5]),
    }
    return metrics, summary


def main() -> None:
    """运行推理、保存检测结果并执行 COCO 精度评估."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(args.annotations))
    image_ids = sorted(coco.getImgIds())
    if args.limit > 0:
        image_ids = image_ids[: args.limit]
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda item: item["id"])
    category_ids = [int(item["id"]) for item in categories]
    if len(category_ids) != 80:
        raise ValueError(f"预期 80 个 COCO 类别, 实际为 {len(category_ids)}.")

    print(f"[1/3] 加载 {args.backend} 后端: {args.model}", flush=True)
    predictions, completed_image_ids = load_checkpoint(args.output)
    completed_set = set(completed_image_ids)
    pending_image_ids = [image_id for image_id in image_ids if image_id not in completed_set]
    if completed_image_ids:
        print(
            f"检测到断点: 已完成 {len(completed_image_ids)}, 待完成 {len(pending_image_ids)}.",
            flush=True,
        )
    backend = create_backend(args)
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        print(f"[2/3] 开始评估 {len(pending_image_ids)} 张 COCO 图片.", flush=True)
        for index, image_id in enumerate(pending_image_ids, start=1):
            image_info = coco.loadImgs([image_id])[0]
            image_path = args.images / image_info["file_name"]
            tensor, shape, ratio, pad_left, pad_top = preprocess(image_path, args.imgsz)
            infer_started = time.perf_counter()
            outputs = backend.run(tensor)
            inference_seconds += time.perf_counter() - infer_started
            predictions.extend(
                build_detections(
                    outputs,
                    image_id,
                    category_ids,
                    shape,
                    ratio,
                    pad_left,
                    pad_top,
                    args.conf,
                    args.iou,
                    args.max_det,
                    args.imgsz,
                )
            )
            completed_image_ids.append(image_id)
            total_completed = len(completed_image_ids)
            if (
                index == 1
                or index % args.progress_interval == 0
                or index == len(pending_image_ids)
            ):
                save_checkpoint(args.output, predictions, completed_image_ids)
                print(f"精度评估进度: {total_completed}/{len(image_ids)}", flush=True)
    finally:
        backend.close()

    predictions_path = args.output / "predictions.json"
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[3/3] 执行 COCOeval.", flush=True)
    metrics, summary = evaluate_predictions(coco, predictions, image_ids)
    report = {
        "backend": args.backend,
        "model": str(args.model.resolve()),
        "image_count": len(image_ids),
        "prediction_count": len(predictions),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "max_det": args.max_det,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "cocoeval_summary": summary,
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"评估报告: {report_path}", flush=True)


if __name__ == "__main__":
    main()
