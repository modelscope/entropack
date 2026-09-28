import importlib
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, get_type_hints

import torch

from .config import CompressionConfig, RawConfig, validate_config
from ..registry import DispatchError, backend_device, register_scheme, select

__all__ = ["RawScheme", "Scheme", "buffers_fingerprint", "cached_parse", "packed_buffers", "register_scheme"]

_lane_modules: dict[tuple[type, str], Any] = {}
_config_classes: dict[type, type] = {}


def buffers_fingerprint(buffers: dict, shape, dtype) -> Any:
    return (
        tuple(shape) if shape is not None else None,
        dtype,
        tuple(
            (name, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)
            for name, tensor in sorted(buffers.items())
        ),
    )


def packed_buffers(packed: dict, kind: type) -> Any:
    return kind(**{name: packed[name] for name in kind._fields})


def cached_parse(layout: torch.Tensor, parse, attribute: str):
    cached = getattr(layout, attribute, None)
    if cached is None:
        cached = parse(layout)
        setattr(layout, attribute, cached)
    return cached


class Scheme(ABC):
    name: str
    buffer_names: tuple[str, ...]
    lossless: bool = True
    priority: int = 0
    dtypes: tuple[torch.dtype, ...] | None = None
    lanes: dict[str, str] = {}

    def supports(self, dtype: torch.dtype) -> bool:
        return self.dtypes is None or dtype in self.dtypes

    def options_for(self, dtype: torch.dtype) -> dict[str, Any]:
        return {}

    def lane(self, backend: str) -> Any:
        key = (type(self), backend)
        module = _lane_modules.get(key)
        if module is None:
            if backend not in self.lanes:
                raise DispatchError(self.name, {backend: f"scheme '{self.name}' declares no '{backend}' lane"})
            try:
                module = importlib.import_module(f".{self.lanes[backend]}", package=type(self).__module__)
            except Exception as error:
                raise DispatchError(self.name, {backend: f"lane module failed to import: {error!r}"}) from error
            _lane_modules[key] = module
        return module

    def lane_for(self, tensor: torch.Tensor | None = None, backend: str | None = None, *, gate_dtype: bool = True):
        name = select(self, tensor, None if backend in (None, "auto") else backend, gate_dtype=gate_dtype)
        return self.lane(name), backend_device(name)

    @abstractmethod
    def encode(self, weight: Any, config: CompressionConfig) -> dict:
        ...

    @abstractmethod
    def decode(self, packed: dict, *, shape: tuple[int, ...], dtype: Any, config: CompressionConfig) -> Any:
        ...

    @property
    def config_cls(self) -> type:
        """The config class this scheme's ``encode`` annotation promises."""
        cached = _config_classes.get(type(self))
        if cached is None:
            cached = get_type_hints(type(self).encode)["config"]
            _config_classes[type(self)] = cached
        return cached

    def make_config(self, options: dict) -> Any:
        """Build this scheme's config from option names and values, rejecting both unknown and bad ones."""
        unknown = set(options) - {field.name for field in fields(self.config_cls)}
        if unknown:
            raise TypeError(f"Unknown {self.name} options: {sorted(unknown)}")
        config = self.config_cls(**options)
        validate_config(config)
        return config

    def validate_buffers(self, buffers: dict, shape: tuple[int, ...], dtype: Any) -> None:
        ...


class RawScheme(Scheme):
    name = "raw"
    buffer_names = ("data",)
    priority = -1
    dtypes = None
    lanes = {}

    def encode(self, weight, config: RawConfig) -> dict:
        data = weight.detach().contiguous().clone()
        return {"data": data}

    def validate_buffers(self, buffers, shape, dtype) -> None:
        data = buffers["data"]
        if data.dtype != dtype or tuple(data.shape) != tuple(shape):
            raise ValueError(f"raw buffer must be {tuple(shape)} {dtype}, got {tuple(data.shape)} {data.dtype}")

    def decode(self, packed: dict, *, shape, dtype, config: RawConfig) -> torch.Tensor:
        return packed["data"]


register_scheme(RawScheme())
