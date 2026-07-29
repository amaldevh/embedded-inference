"""Export an arbitrary torch.nn.Module to a fixed-shape ONNX graph."""

import argparse
import inspect
from pathlib import Path
import torch

from .common import parse_input_spec, parse_json_object
from .model_loader import load_module


def _flatten_outputs(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_outputs(item))
        return flattened
    if isinstance(value, (tuple, list)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_outputs(item))
        return flattened
    raise TypeError(
        "The model output must be a Tensor or a nested tuple/list/dict of Tensors, "
        f"got {type(value).__name__}"
    )


def export_model(args):
    parsed_inputs = [parse_input_spec(value) for value in args.input]
    input_names = [name for name, _, _ in parsed_inputs]
    if len(set(input_names)) != len(input_names):
        raise ValueError("Every ONNX input name must be unique")

    module = load_module(
        model_spec=args.model,
        checkpoint=args.checkpoint,
        factory_kwargs=parse_json_object(args.factory_kwargs),
        checkpoint_key=args.checkpoint_key,
        strict=not args.non_strict,
        device="cpu",
    )

    examples = tuple(
        torch.randn(shape, dtype=getattr(torch, dtype))
        if dtype.startswith("float")
        else torch.zeros(shape, dtype=getattr(torch, dtype))
        for _, shape, dtype in parsed_inputs
    )
    with torch.inference_mode():
        sample_outputs = _flatten_outputs(module(*examples))
    output_names = (
        args.output_names.split(",")
        if args.output_names
        else [f"output_{index}" for index in range(len(sample_outputs))]
    )
    if len(output_names) != len(sample_outputs):
        raise ValueError(
            f"Model produced {len(sample_outputs)} tensor outputs, but "
            f"{len(output_names)} --output-names were supplied"
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": args.opset,
        "export_params": True,
        "do_constant_folding": True,
    }
    # JetPack 5 commonly uses PyTorch versions that predate the dynamo exporter.
    # Opt in only when requested and supported by the installed PyTorch.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = args.dynamo

    with torch.inference_mode():
        torch.onnx.export(module, examples, str(destination), **export_kwargs)

    try:
        import onnx

        graph = onnx.load(str(destination))
        onnx.checker.check_model(graph)
    except ImportError:
        print("warning: onnx is not installed; skipped onnx.checker validation")

    size_mib = destination.stat().st_size / (1024 * 1024)
    print(f"Exported {destination} ({size_mib:.2f} MiB)")
    print(f"Inputs: {[(name, shape, dtype) for name, shape, dtype in parsed_inputs]}")
    print(f"Outputs: {output_names}")
    return destination


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Import path MODULE:SYMBOL for a Module, class, or factory",
    )
    parser.add_argument("--checkpoint", help="Module or state_dict checkpoint")
    parser.add_argument(
        "--checkpoint-key", help="Key containing weights inside the checkpoint"
    )
    parser.add_argument(
        "--factory-kwargs",
        default="{}",
        help="JSON object passed to the model factory",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow missing/unexpected state_dict keys",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME:D0,D1,...[:DTYPE]",
        help="Fixed input specification; repeat for multi-input models",
    )
    parser.add_argument(
        "--output-names",
        help="Comma-separated output names (default: output_0, output_1, ...)",
    )
    parser.add_argument("--output", required=True, help="Destination .onnx file")
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset (default: 13, compatible with JetPack 4.6/TensorRT 8.2)",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the newer torch.export ONNX exporter when available",
    )
    return parser


def main():
    export_model(build_parser().parse_args())


if __name__ == "__main__":
    main()
