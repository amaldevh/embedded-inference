"""Low-latency PyTorch-to-TensorRT deployment helpers for NVIDIA Jetson."""

from typing import Any

__all__ = ["TensorRTRunner"]


def __getattr__(name: str) -> Any:
    # Keep ONNX export usable on a training workstation without TensorRT installed.
    if name == "TensorRTRunner":
        from .runtime import TensorRTRunner

        return TensorRTRunner
    raise AttributeError(name)
