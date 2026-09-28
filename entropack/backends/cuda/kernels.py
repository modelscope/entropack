from collections.abc import Callable, Hashable, Sequence
from pathlib import Path

import cupy
import numpy as np
import torch

_modules: dict[tuple, object] = {}
_kernels: dict[tuple, object] = {}
_streams: dict[tuple, object] = {}

__all__ = ["KernelLibrary", "device_index", "ensure_dynamic_shared", "external_stream", "pointer"]


def ensure_dynamic_shared(kernel, shared_bytes: int) -> None:
    if shared_bytes > kernel.max_dynamic_shared_size_bytes:
        kernel.max_dynamic_shared_size_bytes = shared_bytes


class KernelLibrary:
    def __init__(
        self, key: str, source: Path, defines: Callable[[int, Hashable], Sequence[str]],
        includes: Sequence[Path] = (), kernel_names: Sequence[str] | None = None,
    ):
        self.key = key
        self.source = source
        self.defines = defines
        self.includes = tuple(includes)
        self.kernel_names = None if kernel_names is None else frozenset(kernel_names)

    def kernel(self, device_index: int, variant: Hashable, name: str):
        if self.kernel_names is not None and name not in self.kernel_names:
            raise KeyError(name)
        module_key = (self.key, device_index, variant)
        module = _modules.get(module_key)
        if module is None:
            options = ["--std=c++17"]
            options += [f"-D{definition}" for definition in self.defines(device_index, variant)]
            options += [f"-I{include}" for include in self.includes]
            with cupy.cuda.Device(device_index):
                module = cupy.RawModule(code=self.source.read_text(), options=tuple(options))
            _modules[module_key] = module
        kernel_key = (self.key, device_index, variant, name)
        kernel = _kernels.get(kernel_key)
        if kernel is None:
            kernel = module.get_function(name)
            _kernels[kernel_key] = kernel
        return kernel


def external_stream(stream: torch.cuda.Stream):
    key = (device_index(stream.device), int(stream.cuda_stream))
    wrapper = _streams.get(key)
    if wrapper is None:
        factory = getattr(cupy.cuda.Stream, "from_external", None)
        wrapper = factory(stream) if factory else cupy.cuda.ExternalStream(key[1])
        _streams[key] = wrapper
    return wrapper


def pointer(tensor: torch.Tensor) -> np.uint64:
    return np.uint64(tensor.data_ptr())


def device_index(target) -> int:
    if isinstance(target, int):
        return target
    device = target.device if isinstance(target, torch.Tensor) else target
    index = getattr(device, "index", None)
    return torch.cuda.current_device() if index is None else index
