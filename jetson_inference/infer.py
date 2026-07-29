"""Run one or more inferences with a serialized TensorRT engine."""

import argparse
import numpy as np
import torch

from .runtime import TensorRTRunner


def _random_input(shape, dtype):
    value = torch.empty(shape, dtype=dtype, pin_memory=True)
    if dtype.is_floating_point:
        return value.normal_()
    if dtype == torch.bool:
        return value.random_(0, 2)
    return value.zero_()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument(
        "--input-npz",
        help="Optional .npz whose array names match the engine input names",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--cuda-graph", action="store_true", help="Capture and replay TensorRT enqueue"
    )
    parser.add_argument(
        "--device-output",
        action="store_true",
        help="Keep outputs on the GPU instead of copying them to NumPy",
    )
    args = parser.parse_args()

    runner = TensorRTRunner(args.engine)
    print(runner.describe())
    if args.input_npz:
        with np.load(args.input_npz) as archive:
            inputs = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
    else:
        inputs = {
            name: _random_input(info.shape, info.dtype)
            for name, info in runner.input_info.items()
        }

    runner.set_inputs(inputs)
    if args.cuda_graph:
        runner.capture_cuda_graph()

    outputs = None
    for _ in range(args.iterations):
        # set_inputs is intentionally called on every iteration: a flight application
        # normally receives fresh sensor/state data each cycle.
        outputs = runner.infer(
            inputs,
            return_cpu=not args.device_output,
            synchronize=True,
            use_cuda_graph=args.cuda_graph,
        )
    assert outputs is not None
    print("Outputs:")
    for name, value in outputs.items():
        print(f"  {name}: shape={tuple(value.shape)}, dtype={value.dtype}")
        print(f"    {value}")


if __name__ == "__main__":
    main()
