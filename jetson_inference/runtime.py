"""Persistent-buffer TensorRT runtime backed by PyTorch CUDA tensors."""

from collections import namedtuple
from pathlib import Path
import numpy as np
import tensorrt as trt
import torch


LOGGER = trt.Logger(trt.Logger.WARNING)


TensorInfo = namedtuple(
    "TensorInfo", ("name", "shape", "dtype", "is_input", "binding_index")
)


def _torch_dtype(trt_dtype):
    numpy_dtype = np.dtype(trt.nptype(trt_dtype))
    mapping = {
        np.dtype(np.bool_): torch.bool,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }
    try:
        return mapping[numpy_dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc


class TensorRTRunner:
    """Load a fixed-shape TensorRT engine and reuse every runtime allocation.

    The returned output tensors/buffers are views into persistent storage and are
    overwritten by the next call. Copy them only when the caller must retain a result.
    """

    def __init__(
        self,
        engine_path,
        device="cuda:0",
        stream=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; TensorRTRunner requires an NVIDIA GPU"
            )
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        trt.init_libnvinfer_plugins(LOGGER, "")
        serialized = Path(engine_path).read_bytes()
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self.stream = stream or torch.cuda.Stream(device=self.device)
        self._uses_tensor_api = hasattr(self.engine, "num_io_tensors")
        self.infos = self._inspect_tensors()
        self.inputs = {}
        self.outputs = {}
        self.host_outputs = {}
        self._bindings = [0] * self._binding_count()
        self._allocate()
        self._cuda_graph = None

    def _binding_count(self):
        if self._uses_tensor_api:
            # execute_async_v3 does not consume this list, but keeping it populated
            # simplifies the common allocation path.
            return self.engine.num_io_tensors
        return self.engine.num_bindings

    def _inspect_tensors(self):
        infos = {}
        if self._uses_tensor_api:
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(self.engine.get_tensor_shape(name))
                mode = self.engine.get_tensor_mode(name)
                dtype = self.engine.get_tensor_dtype(name)
                infos[name] = TensorInfo(
                    name=name,
                    shape=shape,
                    dtype=_torch_dtype(dtype),
                    is_input=mode == trt.TensorIOMode.INPUT,
                    binding_index=index,
                )
        else:
            for index in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(index)
                infos[name] = TensorInfo(
                    name=name,
                    shape=tuple(self.engine.get_binding_shape(index)),
                    dtype=_torch_dtype(self.engine.get_binding_dtype(index)),
                    is_input=bool(self.engine.binding_is_input(index)),
                    binding_index=index,
                )
        dynamic = [
            info.name for info in infos.values() if any(d < 0 for d in info.shape)
        ]
        if dynamic:
            raise ValueError(
                "TensorRTRunner currently requires a fixed-shape engine; dynamic tensors: "
                + ", ".join(dynamic)
            )
        return infos

    def _allocate(self):
        for info in self.infos.values():
            tensor = torch.empty(info.shape, dtype=info.dtype, device=self.device)
            if info.is_input:
                self.inputs[info.name] = tensor
            else:
                self.outputs[info.name] = tensor
                self.host_outputs[info.name] = torch.empty(
                    info.shape, dtype=info.dtype, pin_memory=True
                )
            pointer = int(tensor.data_ptr())
            self._bindings[info.binding_index] = pointer
            if self._uses_tensor_api:
                if not self.context.set_tensor_address(info.name, pointer):
                    raise RuntimeError(f"Could not bind TensorRT tensor {info.name!r}")

    @property
    def input_info(self):
        return {name: info for name, info in self.infos.items() if info.is_input}

    @property
    def output_info(self):
        return {name: info for name, info in self.infos.items() if not info.is_input}

    def _enqueue(self):
        stream_pointer = self.stream.cuda_stream
        if self._uses_tensor_api:
            ok = self.context.execute_async_v3(stream_pointer)
        else:
            ok = self.context.execute_async_v2(
                bindings=self._bindings, stream_handle=stream_pointer
            )
        if not ok:
            raise RuntimeError("TensorRT enqueue failed")

    def set_inputs(self, values):
        missing = set(self.inputs) - set(values)
        extra = set(values) - set(self.inputs)
        if missing or extra:
            raise KeyError(
                f"Input names do not match engine; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        with torch.cuda.stream(self.stream):
            for name, value in values.items():
                destination = self.inputs[name]
                source = (
                    value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                )
                if tuple(source.shape) != tuple(destination.shape):
                    raise ValueError(
                        f"{name!r} expects shape {tuple(destination.shape)}, "
                        f"got {tuple(source.shape)}"
                    )
                if source.dtype != destination.dtype:
                    raise TypeError(
                        f"{name!r} expects {destination.dtype}, got {source.dtype}"
                    )
                destination.copy_(source, non_blocking=True)

    def enqueue(self, use_cuda_graph=False):
        with torch.cuda.stream(self.stream):
            if use_cuda_graph:
                if self._cuda_graph is None:
                    raise RuntimeError("Call capture_cuda_graph() before graph replay")
                self._cuda_graph.replay()
            else:
                self._enqueue()

    def synchronize(self):
        self.stream.synchronize()

    def copy_outputs_to_host(self, synchronize=True):
        with torch.cuda.stream(self.stream):
            for name, output in self.outputs.items():
                self.host_outputs[name].copy_(output, non_blocking=True)
        if synchronize:
            self.synchronize()
        return {name: value.numpy() for name, value in self.host_outputs.items()}

    def infer(
        self,
        values,
        return_cpu=True,
        synchronize=True,
        use_cuda_graph=False,
    ):
        self.set_inputs(values)
        self.enqueue(use_cuda_graph=use_cuda_graph)
        if return_cpu:
            return self.copy_outputs_to_host(synchronize=synchronize)
        if synchronize:
            self.synchronize()
        return self.outputs

    def capture_cuda_graph(self, warmup=5):
        """Capture TensorRT enqueue for fixed addresses and shapes.

        Input copies and output copies intentionally remain outside the graph. This
        targets enqueue-bound networks while letting sensors update persistent inputs.
        """
        for _ in range(warmup):
            self.enqueue()
        self.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.stream):
            graph.capture_begin()
            self._enqueue()
            graph.capture_end()
        self._cuda_graph = graph
        self.synchronize()

    def describe(self):
        lines = [f"TensorRT {trt.__version__} engine: {self.engine.name}"]
        for info in self.infos.values():
            mode = "input " if info.is_input else "output"
            lines.append(
                f"  {mode:6s} {info.name}: shape={info.shape}, dtype={info.dtype}"
            )
        return "\n".join(lines)
