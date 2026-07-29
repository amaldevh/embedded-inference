"""Build a device-specific TensorRT engine from an ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.INFO)


def _network_flag() -> int:
    return 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _set_workspace(config: trt.IBuilderConfig, workspace_mib: int) -> None:
    workspace_bytes = workspace_mib * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def _load_timing_cache(
    config: trt.IBuilderConfig, cache_path: Path | None
) -> None:
    if cache_path is None or not hasattr(config, "create_timing_cache"):
        return
    cache_bytes = cache_path.read_bytes() if cache_path.exists() else b""
    cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(cache, ignore_mismatch=False)


def _save_timing_cache(
    config: trt.IBuilderConfig, cache_path: Path | None
) -> None:
    if cache_path is None or not hasattr(config, "get_timing_cache"):
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))


def _engine_metadata(engine: trt.ICudaEngine) -> Dict[str, Any]:
    bindings: List[Dict[str, Any]] = []
    if hasattr(engine, "num_io_tensors"):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            bindings.append(
                {
                    "name": name,
                    "mode": str(engine.get_tensor_mode(name)).split(".")[-1].lower(),
                    "shape": list(engine.get_tensor_shape(name)),
                    "dtype": str(engine.get_tensor_dtype(name)),
                }
            )
    else:
        for index in range(engine.num_bindings):
            bindings.append(
                {
                    "name": engine.get_binding_name(index),
                    "mode": "input" if engine.binding_is_input(index) else "output",
                    "shape": list(engine.get_binding_shape(index)),
                    "dtype": str(engine.get_binding_dtype(index)),
                }
            )
    return {
        "tensorrt_version": trt.__version__,
        "engine_name": engine.name,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "bindings": bindings,
        "warning": (
            "TensorRT engines are tied to the TensorRT/CUDA/GPU environment. "
            "Rebuild this engine on the deployment Jetson."
        ),
    }


def build_engine(args: argparse.Namespace) -> Path:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    trt.init_libnvinfer_plugins(LOGGER, "")
    try:
        builder = trt.Builder(LOGGER)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "TensorRT could not initialize its builder. Build engines on an NVIDIA "
            "system with a working CUDA driver, preferably the deployment Jetson."
        ) from exc
    network = builder.create_network(_network_flag())
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            f"[{index}] {parser.get_error(index)}"
            for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    dynamic_inputs = [
        network.get_input(index).name
        for index in range(network.num_inputs)
        if any(dim < 0 for dim in network.get_input(index).shape)
    ]
    if dynamic_inputs:
        raise ValueError(
            "This latency-first builder requires fixed input shapes. Dynamic inputs: "
            + ", ".join(dynamic_inputs)
        )

    config = builder.create_builder_config()
    _set_workspace(config, args.workspace_mib)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = args.optimization_level

    precision = args.precision.lower()
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This TensorRT platform does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("This TensorRT platform does not report fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        print(
            "INT8 selected: the ONNX graph must contain explicit Q/DQ quantization "
            "nodes; no post-training calibrator is supplied by this generic builder."
        )
    elif precision != "fp32":
        raise ValueError("--precision must be fp32, fp16, or int8")

    timing_cache_path = Path(args.timing_cache) if args.timing_cache else None
    _load_timing_cache(config, timing_cache_path)

    print(
        f"Building {precision.upper()} engine with TensorRT {trt.__version__}; "
        "this can take several minutes..."
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; inspect the log above")
    destination.write_bytes(bytes(serialized))
    _save_timing_cache(config, timing_cache_path)

    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Built engine could not be deserialized")
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(
        json.dumps(_engine_metadata(engine), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {destination} ({destination.stat().st_size / (1024 * 1024):.2f} MiB)"
    )
    print(f"Saved binding metadata to {metadata_path}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Fixed-shape ONNX model")
    parser.add_argument("--output", required=True, help="Destination .engine file")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        help="FP16 is the recommended Xavier NX starting point",
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=2048,
        help="Maximum temporary workspace available while choosing tactics",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(0, 6),
        default=5,
        help="Builder optimization effort when supported",
    )
    parser.add_argument(
        "--timing-cache",
        help="Optional persistent tactic timing-cache path",
    )
    return parser


def main() -> None:
    build_engine(build_parser().parse_args())


if __name__ == "__main__":
    main()
