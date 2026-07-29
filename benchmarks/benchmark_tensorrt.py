"""Benchmark TensorRT GPU-only and host-to-host serial inference latency."""

import argparse
import json
import time
from pathlib import Path
import torch

from jetson_inference.common import percentile_summary, print_summary
from jetson_inference.runtime import TensorRTRunner


def _host_inputs(runner):
    inputs = {}
    for name, info in runner.input_info.items():
        tensor = torch.empty(info.shape, dtype=info.dtype, pin_memory=True)
        if info.dtype.is_floating_point:
            tensor.normal_()
        elif info.dtype == torch.bool:
            tensor.random_(0, 2)
        else:
            tensor.zero_()
        inputs[name] = tensor
    return inputs


def gpu_latency(runner, warmup, iterations, use_cuda_graph):
    for tensor in runner.inputs.values():
        if tensor.dtype.is_floating_point:
            tensor.normal_()
        else:
            tensor.zero_()
    for _ in range(warmup):
        runner.enqueue()
    runner.synchronize()
    if use_cuda_graph:
        runner.capture_cuda_graph()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(runner.stream)
        runner.enqueue(use_cuda_graph=use_cuda_graph)
        end.record(runner.stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return percentile_summary(samples)


def end_to_end_latency(
    runner,
    inputs,
    warmup,
    iterations,
    use_cuda_graph,
):
    for _ in range(warmup):
        runner.infer(
            inputs,
            return_cpu=True,
            synchronize=True,
            use_cuda_graph=use_cuda_graph,
        )

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        runner.infer(
            inputs,
            return_cpu=True,
            synchronize=True,
            use_cuda_graph=use_cuda_graph,
        )
        samples.append((time.perf_counter() - start) *1e3)
    return percentile_summary(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use CUDA Graph replay for the TensorRT enqueue",
    )
    parser.add_argument("--json", help="Optional result JSON path")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")

    runner = TensorRTRunner(args.engine)
    print(runner.describe())
    host_inputs = _host_inputs(runner)
    runner.set_inputs(host_inputs)
    results = {
        "engine_gpu": gpu_latency(
            runner, args.warmup, args.iterations, args.cuda_graph
        ),
        "host_to_host": end_to_end_latency(
            runner, host_inputs, args.warmup, args.iterations, args.cuda_graph
        ),
        "cuda_graph": args.cuda_graph,
        "engine": str(Path(args.engine).resolve()),
    }
    print_summary("TensorRT engine GPU latency", results["engine_gpu"])
    print_summary("TensorRT host-to-host latency", results["host_to_host"])
    print(
        "\nGPU latency excludes H2D/D2H transfers. Host-to-host includes pinned-host "
        "copies, enqueue, output copy, synchronization, and Python overhead."
    )
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
