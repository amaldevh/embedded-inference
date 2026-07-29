"""Low-latency PyTorch-to-TensorRT deployment helpers for NVIDIA Jetson."""

__all__ = ["TensorRTRunner"]


def __getattr__(name):
    # Keep ONNX export usable on a training workstation without TensorRT installed.
    if name == "TensorRTRunner":
        from .runtime import TensorRTRunner

        return TensorRTRunner
    raise AttributeError(name)
