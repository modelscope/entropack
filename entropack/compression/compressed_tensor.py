import copy
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from ..registry import get_scheme
from ..schemes import Scheme


def _parse_dtype(name: str) -> torch.dtype:
    if not name:
        raise TypeError("CompressedTensor serialized dtype must be a non-empty string")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported serialized torch dtype '{name}'")
    return dtype


class CompressedTensor(torch.Tensor):
    """Packed tensor storage exposed through a frozen PyTorch Tensor interface.

    Create containers with :func:`entropack.compress` or :meth:`from_state_dict`.
    The wrapper exposes the uncompressed shape, while ``buffers`` hold the stored
    data and codec metadata. Use :func:`entropack.decompress` to obtain a dense
    tensor before numerical operations.

    ``dtype`` selects the output type of ``decompress``; ``encoded_dtype`` records
    the codec's input type. With the default ``copy=False``, ``to(dtype=...)``
    changes only the output type and shares packed storage. Device transfers move
    buffers without changing their dtypes. Use ``clone()`` or ``to(..., copy=True)``
    for an independent storage copy.

    :meth:`state_dict` exports a tensor-only representation. Loading restores the
    encoded dtype; temporary output-dtype changes are not saved. The wrapper
    cannot require gradients. Treat packed buffers as read-only.

    Args:
        header: codec metadata containing the ``compress_method`` scheme name.
        buffers: named tensors holding stored data and codec metadata on one device.
        shape: shape of the uncompressed tensor.
        dtype: logical dtype returned by ``decompress``.
        encoded_dtype: dtype expected by the codec. Defaults to ``dtype``.
    """

    header: dict[str, Any]
    buffers: dict[str, torch.Tensor]

    @staticmethod
    def __new__(cls, header, buffers, shape, dtype, *, encoded_dtype=None):
        device = next(iter(buffers.values())).device
        return torch.Tensor._make_wrapper_subclass(cls, tuple(shape), dtype=dtype, device=device, requires_grad=False)

    def __init__(self, header, buffers, shape, dtype, *, encoded_dtype=None):
        self.header = copy.deepcopy(header)
        self.buffers = dict(buffers)
        self.encoded_dtype = dtype if encoded_dtype is None else encoded_dtype
        self.validate()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, scheme={self.compress_method!r})"

    def __tensor_flatten__(self):
        names = tuple(self.buffers)
        for name, value in self.buffers.items():
            setattr(self, "_packed_" + name, value)
        metadata = (names, copy.deepcopy(self.header), tuple(self.shape), self.dtype, self.encoded_dtype)
        return ["_packed_" + name for name in names], metadata

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        names, header, shape, dtype, encoded_dtype = metadata
        return cls(
            header=header, buffers={name: inner_tensors["_packed_" + name] for name in names},
            shape=shape, dtype=dtype, encoded_dtype=encoded_dtype,
        )

    def _map_buffers(self, fn, *, dtype=None) -> "CompressedTensor":
        return type(self)(
            header=self.header, buffers={name: fn(value) for name, value in self.buffers.items()},
            shape=self.shape, dtype=self.dtype if dtype is None else dtype, encoded_dtype=self.encoded_dtype,
        )

    def to(self, *args, copy: bool = False, **kwargs) -> "CompressedTensor":
        """Move buffers or change the logical dtype, with an optional keyword-only copy."""
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        result = super().to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, memory_format=memory_format)
        return result.clone() if copy and result.device == self.device else result

    def __copy__(self) -> "CompressedTensor":
        result = self._map_buffers(lambda value: value)
        if getattr(self, "_is_param", False):
            result._is_param = True
        return result

    def __deepcopy__(self, memo) -> "CompressedTensor":
        if id(self) in memo:
            return memo[id(self)]
        result = self._map_buffers(lambda value: copy.deepcopy(value, memo))
        if getattr(self, "_is_param", False):
            result._is_param = True
        memo[id(self)] = result
        return result

    def requires_grad_(self, requires_grad: bool = False):
        """Compressed values are frozen; gradients may flow through their consumers."""
        if requires_grad:
            raise RuntimeError("CompressedTensor cannot require gradients; decompress it to train dense values")
        return torch.Tensor.requires_grad_(self, False)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        source = args[0] if args else None
        aten = torch.ops.aten

        if func in (aten.detach.default, aten.alias.default):
            result = source._map_buffers(torch.detach)
            return return_and_correct_aliasing(func, args, kwargs, result)

        if func in (aten.clone.default, aten._to_copy.default):
            if kwargs.get("memory_format") not in (None, torch.preserve_format, torch.contiguous_format):
                raise ValueError("CompressedTensor does not support that memory format")
            if func == aten.clone.default:
                return source._map_buffers(torch.clone)
            if kwargs.get("layout", torch.strided) not in (None, torch.strided):
                raise ValueError("CompressedTensor only supports strided storage")
            options = {name: value for name, value in kwargs.items() if name in ("device", "non_blocking")}
            return source._map_buffers(lambda value: value.to(**options), dtype=kwargs.get("dtype"))

        result = func.decompose(*args, **kwargs)
        if result is not NotImplemented:
            return result
        raise NotImplementedError(f"CompressedTensor does not implement {func}; decompress it before numerical operations")

    @property
    def compress_method(self) -> str:
        """The scheme name the header carries."""
        return self.header["compress_method"]

    @property
    def scheme(self) -> Scheme:
        """The codec named by the header."""
        return get_scheme(self.compress_method)

    @property
    def lossless(self) -> bool:
        """Whether the encoding scheme is lossless, before any output dtype conversion."""
        return self.scheme.lossless

    @property
    def actual_bpp(self) -> float:
        """Bits per element of :attr:`shape`, serialized header included."""
        return self.storage_nbytes() * 8 / math.prod(self.shape)

    def validate(self) -> None:
        """Check the scheme, dtype, and buffer names without inspecting contents."""
        if any(isinstance(value, CompressedTensor) for value in self.buffers.values()):
            raise TypeError("CompressedTensor buffers cannot contain another CompressedTensor")
        scheme = self.scheme
        if not scheme.supports(self.encoded_dtype):
            raise ValueError(f"'{scheme.name}' does not support format {self.encoded_dtype}")
        if set(self.buffers) != set(scheme.buffer_names):
            raise ValueError(
                f"CompressedTensor buffers for {self.compress_method} must be "
                f"{list(scheme.buffer_names)}, got {sorted(self.buffers)}"
            )

    def _serialized_header_tensor(self) -> torch.Tensor:
        metadata = {
            "compress_method": self.compress_method, "dtype": str(self.encoded_dtype).removeprefix("torch."),
            "shape": list(self.shape), "buffer_names": list(self.buffers), "header": self.header,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return torch.frombuffer(bytearray(encoded), dtype=torch.uint8)

    def storage_nbytes(self, include_header: bool = True) -> int:
        """Bytes the buffers occupy, plus the serialized header unless ``include_header`` is false."""
        total = sum(value.numel() * value.element_size() for value in self.buffers.values())
        if include_header:
            total += self._serialized_header_tensor().numel()
        return total

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """The container as flat tensors under ``prefix``: one 1D uint8 header, then one entry per buffer."""
        self.validate()
        state = {f"{prefix}header": self._serialized_header_tensor()}
        state.update({f"{prefix}buffers.{name}": value.detach() for name, value in self.buffers.items()})
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor], prefix: str = "") -> "CompressedTensor":
        """Restore a container in its encoded dtype from a tensor-only checkpoint."""
        header_key = f"{prefix}header"
        if header_key not in state:
            raise ValueError(f"CompressedTensor state is missing '{header_key}'")
        header_tensor = state[header_key]
        if header_tensor.dtype != torch.uint8 or header_tensor.ndim != 1:
            raise TypeError("CompressedTensor serialized header must be a 1D uint8 tensor")
        try:
            metadata = json.loads(bytes(header_tensor.detach().cpu().tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("CompressedTensor serialized header is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("CompressedTensor serialized header must contain a JSON object")

        required = {"compress_method", "dtype", "shape", "buffer_names", "header"}
        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"CompressedTensor serialized header is missing fields: {sorted(missing)}")
        buffer_names = metadata["buffer_names"]
        if not isinstance(buffer_names, list) or any(not isinstance(name, str) or not name for name in buffer_names):
            raise TypeError("CompressedTensor buffer_names must be a list of strings")
        buffers = {}
        for name in buffer_names:
            key = f"{prefix}buffers.{name}"
            if key not in state:
                raise ValueError(f"CompressedTensor state is missing '{key}'")
            buffers[name] = state[key]

        return cls(header=metadata["header"], buffers=buffers, shape=tuple(metadata["shape"]), dtype=_parse_dtype(metadata["dtype"]))
