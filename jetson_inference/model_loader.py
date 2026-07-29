"""Load a user-supplied torch.nn.Module without coupling to its source tree."""

#from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .common import import_symbol


def _torch_load(path: str | Path, map_location: str) -> Any:
    # ``weights_only`` did not exist in the PyTorch releases commonly used with
    # JetPack 5. Newer versions default to weights-only loading, which cannot load
    # a serialized Module. A checkpoint is pickle data: load only trusted files.
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def load_module(
    model_spec: Optional[str],
    checkpoint: Optional[str],
    factory_kwargs: Optional[Mapping[str, Any]] = None,
    checkpoint_key: Optional[str] = None,
    strict: bool = True,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a module factory and/or checkpoint.

    ``model_spec`` points to a module instance, Module class, or zero-argument
    factory (additional keyword arguments are supported). A checkpoint may be a
    serialized Module, a raw state_dict, or a dict containing a state_dict.
    """
    factory_kwargs = dict(factory_kwargs or {})
    loaded: Any = None
    if checkpoint:
        loaded = _torch_load(checkpoint, map_location=device)

    if model_spec:
        target = import_symbol(model_spec)
        if isinstance(target, torch.nn.Module):
            model = target
        elif callable(target):
            model = target(**factory_kwargs)
        else:
            raise TypeError(f"{model_spec!r} is not a Module, class, or factory")
    elif isinstance(loaded, torch.nn.Module):
        model = loaded
        loaded = None
    else:
        raise ValueError(
            "--model is required unless --checkpoint contains a serialized Module"
        )

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Model factory returned {type(model).__name__}, not nn.Module")

    if loaded is not None:
        state = loaded
        if checkpoint_key:
            if not isinstance(state, Mapping) or checkpoint_key not in state:
                raise KeyError(f"Checkpoint has no key {checkpoint_key!r}")
            state = state[checkpoint_key]
        elif isinstance(state, Mapping):
            for common_key in ("state_dict", "model_state_dict", "model"):
                candidate = state.get(common_key)
                if isinstance(candidate, Mapping):
                    state = candidate
                    break
        if not isinstance(state, Mapping):
            raise TypeError(
                "Checkpoint is not a Module or state_dict; provide --checkpoint-key "
                "if the weights are nested"
            )
        # Handle checkpoints saved through DistributedDataParallel.
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=strict)

    return model.to(device).eval()
