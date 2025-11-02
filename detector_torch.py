"""PyTorch-based detector wrappers used when TensorRT is unavailable.

The classes defined here expose the same interface as the TensorRT-backed
`Detector*` implementations so the rest of the GLAD pipeline can run without a
GPU. They rely on the `yolov5` PyPI package, which bundles the Ultralytics
YOLOv5 inference helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from yolov5 import YOLOv5


class _TorchDetector:
    """Common utilities for running YOLOv5 detectors on CPU or CUDA."""

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float,
        iou_threshold: float,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Missing detector weights: {self.weights_path}")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # The YOLOv5 helper handles preprocessing and post-processing internally.
        self.model = YOLOv5(str(self.weights_path), device=self.device)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

        self.conf_threshold = conf_threshold

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Run the underlying YOLO model and return detections as numpy arrays."""
        results = self.model.predict(image, size=640, augment=False)
        predictions = results.xyxy[0]
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        else:
            predictions = np.asarray(predictions)

        if predictions.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        predictions = predictions.astype(np.float32, copy=False)
        return predictions

    @staticmethod
    def _to_xywh(box: Iterable[float]) -> np.ndarray:
        x1, y1, x2, y2 = box[:4]
        return np.array(
            [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))],
            dtype=np.int32,
        )


class Detector1(_TorchDetector):
    """Global YOLO detector backed by PyTorch."""

    def __init__(self, weights_path: str) -> None:
        super().__init__(weights_path, conf_threshold=0.5, iou_threshold=0.4)

    def detect(self, image: np.ndarray):  # type: ignore[override]
        predictions = self._predict(image)
        if predictions.shape[0] == 0:
            return []

        keep = predictions[:, 4] >= self.conf_threshold
        predictions = predictions[keep]
        if predictions.shape[0] == 0:
            return []

        best_index = int(np.argmax(predictions[:, 4]))
        return self._to_xywh(predictions[best_index])


class _LocalDetector(_TorchDetector):
    """Shared logic for the local YOLO detector variants."""

    def detect(self, image: np.ndarray, x_prev: float, y_prev: float):  # type: ignore[override]
        predictions = self._predict(image)
        if predictions.shape[0] == 0:
            return []

        keep = predictions[:, 4] >= self.conf_threshold
        predictions = predictions[keep]
        if predictions.shape[0] == 0:
            return []

        centers_x = (predictions[:, 0] + predictions[:, 2]) * 0.5
        centers_y = (predictions[:, 1] + predictions[:, 3]) * 0.5
        distances = np.sqrt((centers_x - x_prev) ** 2 + (centers_y - y_prev) ** 2)
        best_index = int(np.argmin(distances))
        return self._to_xywh(predictions[best_index])


class Detector2(_LocalDetector):
    """Local YOLO detector used on the search window."""

    def __init__(self, weights_path: str) -> None:
        super().__init__(weights_path, conf_threshold=0.1, iou_threshold=0.4)


class Detector3(_LocalDetector):
    """Local YOLO refinement detector triggered after motion detection."""

    def __init__(self, weights_path: str) -> None:
        super().__init__(weights_path, conf_threshold=0.5, iou_threshold=0.4)
