"""Helpers to construct detector instances for the GLAD demos."""
from __future__ import annotations

import ctypes
import importlib
from pathlib import Path
from typing import NamedTuple, Sequence

_AVAILABLE_BACKENDS = ("auto", "tensorrt", "torch")


class DetectorConfigurationError(RuntimeError):
    """Raised when the requested detector backend cannot be initialized."""


class DetectorBundle(NamedTuple):
    detector1: object
    detector2: object
    detector3: object
    backend: str


def available_backends() -> Sequence[str]:
    """Return the list of supported backend selectors."""

    return _AVAILABLE_BACKENDS


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_tensorrt_stack() -> bool:
    return _module_available("tensorrt") and _module_available("pycuda.driver")


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise DetectorConfigurationError(
            "Missing detector weights: " + ", ".join(missing)
        )


def build_detectors(
    backend: str = "auto",
    *,
    global_model: str = "yolov5s_GLAD",
    local_model: str = "yolov5s_GLAD-crop",
    weights_root: str | Path | None = None,
) -> DetectorBundle:
    """Instantiate detector objects for the selected backend.

    Args:
        backend: One of ``auto``, ``tensorrt``, or ``torch``.
        global_model: Stem of the weight files for the global detector.
        local_model: Stem of the weight files for the local detectors.
        weights_root: Directory containing the weight files. Defaults to the
            repository ``weights`` folder.
    """

    backend = backend.lower()
    if backend not in _AVAILABLE_BACKENDS:
        raise DetectorConfigurationError(
            f"Unsupported backend '{backend}'. Choose from: {', '.join(_AVAILABLE_BACKENDS)}."
        )

    root = Path(weights_root) if weights_root is not None else Path(__file__).resolve().parent / "weights"

    if backend == "auto":
        backend = "tensorrt" if _has_tensorrt_stack() else "torch"

    if backend == "tensorrt":
        if not _has_tensorrt_stack():
            raise DetectorConfigurationError(
                "TensorRT backend requested but TensorRT/PyCUDA are not available."
            )

        engine_global = root / f"{global_model}.engine"
        engine_local = root / f"{local_model}.engine"
        _require_files((engine_global, engine_local))

        plugin_path = root / "libmyplugins.so"
        if plugin_path.exists():
            ctypes.CDLL(str(plugin_path))

        from detector1_trt import Detector1 as GlobalDetector  # noqa: WPS433
        from detector2_trt import Detector2 as LocalDetector  # noqa: WPS433
        from detector3_trt import Detector3 as RefineDetector  # noqa: WPS433

        return DetectorBundle(
            GlobalDetector(str(engine_global)),
            LocalDetector(str(engine_local)),
            RefineDetector(str(engine_local)),
            "tensorrt",
        )

    if backend == "torch":
        if not _module_available("torch"):
            raise DetectorConfigurationError(
                "PyTorch is required for the torch backend. Install torch before running with --backend torch."
            )
        if not _module_available("yolov5"):
            raise DetectorConfigurationError(
                "The torch backend relies on the 'yolov5' package. Install it with 'pip install yolov5'."
            )

        weights_global = root / f"{global_model}.pt"
        weights_local = root / f"{local_model}.pt"
        _require_files((weights_global, weights_local))

        from detector_torch import Detector1 as GlobalDetector  # noqa: WPS433
        from detector_torch import Detector2 as LocalDetector  # noqa: WPS433
        from detector_torch import Detector3 as RefineDetector  # noqa: WPS433

        return DetectorBundle(
            GlobalDetector(str(weights_global)),
            LocalDetector(str(weights_local)),
            RefineDetector(str(weights_local)),
            "torch",
        )

    raise DetectorConfigurationError(f"Unhandled backend '{backend}'.")
