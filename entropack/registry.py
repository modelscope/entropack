import logging
import threading
from typing import Any

import torch

from .backends import BACKENDS, Backend

logger = logging.getLogger("entropack.registry")

_backends: dict[str, Backend] = {backend.name: backend for backend in BACKENDS}
_schemes: dict[str, Any] = {}
_reasons: dict[str, str | None] = {}
_lock = threading.Lock()


_CONTAINER_DTYPES = (
    torch.float32, torch.float16, torch.bfloat16,
    torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint64, torch.uint32, torch.uint16, torch.uint8, torch.bool,
)


class DispatchError(RuntimeError):
    """``reasons`` maps every backend considered to why it was rejected, so no fallback is silent."""

    def __init__(self, scheme: str, reasons: dict[str, str]):
        self.scheme = scheme
        self.reasons = dict(reasons)
        detail = "; ".join(f"{name}: {reason}" for name, reason in self.reasons.items())
        super().__init__(f"No backend can handle '{scheme}'" + (f": {detail}" if detail else ""))


def register_scheme(scheme) -> Any:
    with _lock:
        _schemes[scheme.name] = scheme
    return scheme


def get_scheme(name: str) -> Any:
    try:
        return _schemes[name]
    except KeyError:
        raise KeyError(f"Unknown scheme '{name}'; available: {sorted(_schemes)}") from None


def all_schemes() -> list:
    return sorted(_schemes.values(), key=lambda scheme: (-scheme.priority, scheme.name))


def name_for_config(config) -> str:
    for cls in type(config).__mro__:
        for scheme in _schemes.values():
            if scheme.config_cls is cls:
                return scheme.name
    raise TypeError(f"{type(config).__name__} is not the config of any registered scheme")


def require_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _CONTAINER_DTYPES:
        raise ValueError(f"Unsupported tensor dtype {dtype}; supported: {sorted(_CONTAINER_DTYPES, key=str)}")
    return dtype


def default_config(dtype, **overrides):
    dtype = require_dtype(dtype)
    scheme = next(scheme for scheme in all_schemes() if scheme.supports(dtype))
    return scheme.make_config({**scheme.options_for(dtype), **overrides})


def _ordered() -> list[Backend]:
    return sorted(_backends.values(), key=lambda backend: backend.priority, reverse=True)


def reason(backend: str) -> str | None:
    spec = _backends.get(backend)
    if spec is None:
        return "not registered"
    if backend not in _reasons:
        with _lock:
            if backend not in _reasons:
                _reasons[backend] = spec.probe() if spec.probe is not None else None
    return _reasons[backend]


def _rejection(name: str, scheme, dtype: torch.dtype | None) -> str | None:
    if name not in _backends:
        return "not registered"
    if name not in scheme.lanes:
        return f"scheme '{scheme.name}' declares no '{name}' lane"
    unavailable = reason(name)
    if unavailable is not None:
        return unavailable
    if dtype is not None and not scheme.supports(dtype):
        return f"dtype {dtype} is not supported"
    return None


def select(scheme, tensor: torch.Tensor | None = None, backend: str | None = None,
           *, gate_dtype: bool = True) -> str:
    dtype = tensor.dtype if (gate_dtype and tensor is not None) else None
    if backend is not None:
        rejected = _rejection(backend, scheme, dtype)
        if rejected is not None:
            raise DispatchError(scheme.name, {backend: rejected})
        logger.debug("Backend %s selected for %s", backend, scheme.name)
        return backend

    failures: dict[str, str] = {}
    for spec in _ordered():
        rejected = _rejection(spec.name, scheme, dtype)
        if rejected is None:
            logger.debug("Backend %s selected for %s", spec.name, scheme.name)
            return spec.name
        failures[spec.name] = rejected
    raise DispatchError(scheme.name, failures)


def backend_device(backend: str | None) -> torch.device | None:
    spec = _backends.get(backend) if backend is not None else None
    return spec.device() if spec is not None and spec.device is not None else None
