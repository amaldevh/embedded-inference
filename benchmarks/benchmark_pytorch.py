"""Benchmark the original PyTorch module for an apples-to-apples baseline."""

#from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch

from jetson_inference.common import (
    parse_input_spec,
    parse_json_object,
    percentile_summary,
    print_summary,
)
from jetson_inference.model_loader import load_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="MODULE:SYMBOL")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-key")
    parser.add_argument("--factory-kwargs", default="{}")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME:D0,D1,...[:DTYPE]",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16"), default="fp32"
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    model = load_module(
        args.model,
        args.checkpoint,
        parse_json_object(args.factory_kwargs),
        args.checkpoint_key,
        device=args.device,
    )
    parsed_inputs = [parse_input_spec(value) for value in args.input]
    inputs = tuple(
        torch.randn(shape, dtype=getattr(torch, dtype), device=args.device)
        if dtype.startswith("float")
        else torch.zeros(shape, dtype=getattr(torch, dtype), device=args.device)
        for _, shape, dtype in parsed_inputs
    )
    if args.precision == "fp16":
        if args.device != "cuda":
            parser.error("FP16 benchmark is intended for CUDA")
        model = model.half()
        inputs = tuple(
            value.half() if value.dtype.is_floating_point else value for value in inputs
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()

        samples: List[float] = []
        if args.device == "cuda":
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(*inputs)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            label = "PyTorch CUDA compute latency"
        else:
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                model(*inputs)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            label = "PyTorch CPU forward latency"

    summary = percentile_summary(samples)
    print_summary(label, summary)
    print(
        "\nThis baseline uses preallocated inputs and excludes sensor copies and "
        "post-processing; compare it with TensorRT's engine_gpu result."
    )
    if args.json:
        result = {
            "backend": "pytorch",
            "device": args.device,
            "precision": args.precision,
            "latency": summary,
        }
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
