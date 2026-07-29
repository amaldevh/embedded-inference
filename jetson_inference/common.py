"""Shared command-line and reporting utilities."""

import importlib
import json
from pathlib import Path
import numpy as np


TORCH_DTYPES = {
    "bool": "bool",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
}


def import_symbol(spec):
    """Import ``module.submodule:symbol``."""
    if ":" not in spec:
        raise ValueError(f"Expected MODULE:SYMBOL, got {spec!r}")
    module_name, symbol_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_name!r} has no symbol {symbol_name!r}") from exc


def parse_json_object(value):
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def parse_input_spec(value):
    """Parse ``NAME:D0,D1,...[:DTYPE]``."""
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Invalid input spec {value!r}; expected NAME:D0,D1,...[:DTYPE]"
        )
    name, shape_text = parts[:2]
    dtype = parts[2].lower() if len(parts) == 3 else "float32"
    if not name:
        raise ValueError(f"Input name is empty in {value!r}")
    try:
        shape = tuple(int(dim) for dim in shape_text.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid shape in input spec {value!r}") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Input shapes must be fixed and positive, got {shape}")
    if dtype not in TORCH_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; choose one of {sorted(TORCH_DTYPES)}"
        )
    return name, shape, dtype


def percentile_summary(samples_ms):
    if not samples_ms:
        raise ValueError("No latency samples were collected")
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "iterations": int(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "mean_hz": float(1000.0 / values.mean()),
    }


def print_summary(label, summary):
    print(f"\n{label}")
    print(
        "  mean={mean_ms:.4f} ms ({mean_hz:.1f} Hz), "
        "std={std_ms:.4f} ms, min={min_ms:.4f} ms".format(**summary)
    )
    print(
        "  p50={p50_ms:.4f} ms, p90={p90_ms:.4f} ms, "
        "p95={p95_ms:.4f} ms, p99={p99_ms:.4f} ms, "
        "max={max_ms:.4f} ms".format(**summary)
    )


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
