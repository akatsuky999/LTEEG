"""Models: one package per model, selected by its folder name.

::

    lteeg/models/
    ├── base.py                 the contract every model satisfies (read this first)
    ├── contract.py             executable check of that contract
    ├── _template/              copy this folder to start a new model
    ├── seizure_transformer/    __init__.py, model.py, layers/, README.md
    ├── tcn/
    └── dcrnn/

``model.name: dcrnn`` imports ``lteeg.models.dcrnn`` and instantiates its ``Model``;
``model.name: "my_pkg.my_module:MyNet"`` instantiates a model kept outside the
framework. Only the selected package is imported, so one model's optional
dependencies or errors never affect the others. There is no registration step:
adding a folder is enough.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping

import torch

from ..config import Config
from ..registry import RegistryError, call_with_checked_kwargs
from ..utils.logging import get_logger
from .base import ModelOutput, as_output, count_parameters

log = get_logger("models")

PACKAGE_DIR = Path(__file__).resolve().parent
REQUIRED_ARGS = ("in_channels", "in_samples", "num_outputs")  # always passed by the framework
OPTIONAL_ARGS = ("fs", "channel_names")  # passed only if the constructor declares them

__all__ = ["ModelOutput", "as_output", "available_models", "build_model", "count_parameters",
           "describe_available_models", "describe_model", "get_model_class", "instantiate_model", "load_weights",
           "model_context", "model_module", "smoke_params"]


def available_models() -> List[str]:
    """Names of the built-in model packages: sub-folders of ``lteeg/models`` holding an
    ``__init__.py`` and a ``model.py``. Folders starting with ``_`` (e.g. ``_template``)
    are not models."""
    return sorted(p.name for p in PACKAGE_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "__init__.py").is_file()
                  and (p / "model.py").is_file())


def model_module(name: str) -> ModuleType:
    """Import the package of a built-in model (``lteeg.models.<name>``)."""
    names = available_models()
    if name not in names:
        close = difflib.get_close_matches(name, names, n=1)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        raise RegistryError(f"Unknown model '{name}'{hint}. Built-in models: {names}; a model outside the "
                            f"framework is selected with 'package.module:Class'.")
    return importlib.import_module(f"{__name__}.{name}")


def get_model_class(name: str) -> Any:
    """Resolve ``model.name`` to the model class (or factory function)."""
    if ":" in name:
        module_name, attr = name.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise RegistryError(f"model.name={name!r}: importing {module_name!r} failed ({e}). "
                                f"Is its package on PYTHONPATH?") from e
        obj = getattr(module, attr, None)
        if obj is None:
            raise RegistryError(f"model.name={name!r}: module {module_name!r} has no attribute {attr!r}")
        return obj
    module = model_module(name)
    obj = getattr(module, "Model", None)
    if obj is None:
        raise RegistryError(f"lteeg/models/{name}/__init__.py must define `Model` (the model class), "
                            f"e.g. `from .model import MyNet as Model`")
    return obj


def smoke_params(name: str) -> Dict[str, Any]:
    """Small hyper-parameters a built-in model declares for fast tests (``SMOKE_PARAMS``)."""
    return dict(getattr(model_module(name), "SMOKE_PARAMS", {}))


def describe_available_models() -> List[Dict[str, Any]]:
    """Name, class, one-line summary, framework-provided context and tunable parameters
    (with defaults) of every built-in model, read from the constructor signatures."""
    rows = []
    for name in available_models():
        module = model_module(name)
        cls = getattr(module, "Model", None)
        if cls is None:
            continue
        params: Dict[str, Any] = {}
        context: List[str] = []
        for key, p in inspect.signature(cls).parameters.items():
            if key in OPTIONAL_ARGS:
                context.append(key)
            elif key not in REQUIRED_ARGS and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                params[key] = p.default if p.default is not p.empty else "<required>"
        summary = (module.__doc__ or "").strip().splitlines()
        rows.append({"name": name, "class": getattr(cls, "__name__", str(cls)),
                     "summary": summary[0] if summary else "", "context": context, "params": params})
    return rows


def model_context(cfg: Config, num_outputs: int) -> Dict[str, Any]:
    """Everything the framework can tell a model about its input."""
    return {"in_channels": cfg.n_channels, "in_samples": cfg.window_samples, "num_outputs": num_outputs,
            "fs": cfg.fs, "channel_names": list(cfg.data.channels)}


def instantiate_model(factory: Any, name: str, context: Mapping[str, Any],
                      params: Mapping[str, Any]) -> torch.nn.Module:
    """Construct a model following the contract in :mod:`lteeg.models.base`."""
    reserved = sorted(set(params) & set(REQUIRED_ARGS + OPTIONAL_ARGS))
    if reserved:
        raise RegistryError(f"model.params must not set {reserved}: the framework provides them "
                            f"(from data.channels, windows.length_sec, task and the sampling rate)")
    kwargs = {k: context[k] for k in REQUIRED_ARGS}
    try:
        sig_params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        sig_params = None
    if sig_params is not None:
        var_kw = any(p.kind == p.VAR_KEYWORD for p in sig_params.values())
        missing = [a for a in REQUIRED_ARGS if a not in sig_params and not var_kw]
        if missing:
            raise RegistryError(f"model '{name}' must accept {list(REQUIRED_ARGS)} (missing {missing}); "
                                f"see lteeg/models/base.py")
        kwargs.update({k: context[k] for k in OPTIONAL_ARGS if k in sig_params})
    model = call_with_checked_kwargs(factory, "model", name, **kwargs, **dict(params))
    if not isinstance(model, torch.nn.Module):
        raise RegistryError(f"model '{name}' returned {type(model).__name__}, not a torch.nn.Module")
    return model


def build_model(cfg: Config, num_outputs: int) -> torch.nn.Module:
    factory = get_model_class(cfg.model.name)
    model = instantiate_model(factory, cfg.model.name, model_context(cfg, num_outputs), cfg.model.params)
    if cfg.model.init_checkpoint:
        load_weights(model, cfg.resolve_path(cfg.model.init_checkpoint))
    return model


def load_weights(model: torch.nn.Module, path: Path, strict: bool = True) -> None:
    """Load weights from an LTEEG checkpoint (``{'model': state_dict, ...}``) or a bare
    state dict. If the model defines ``convert_state_dict``, it is applied first, so
    weights saved by a model's original implementation load directly."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    convert = getattr(type(model), "convert_state_dict", None)
    if callable(convert):
        state = convert(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    log.info(f"Loaded weights from {path}"
             + (f" (missing={list(missing)}, unexpected={list(unexpected)})" if (missing or unexpected) else ""))


def describe_model(model: torch.nn.Module) -> str:
    return f"{type(model).__name__}: {count_parameters(model) / 1e6:.2f}M trainable parameters"
