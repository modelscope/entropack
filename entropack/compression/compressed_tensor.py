import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ..registry import get_scheme
from ..schemes import Scheme


def _parse_dtype(name: str) -> torch.dtype:
    if not name:
        raise TypeError("CompressedTensor serialized dtype must be a non-empty string")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported serialized torch dtype '{name}'")
    return dtype


@dataclass
class CompressedTensor:
    """Compressed data and metadata for one tensor.

    The container records the scheme, input shape, and dtype. Use
    :func:`entropack.decompress` with the matching scheme's configuration to restore values.
    Use :meth:`state_dict` and :meth:`from_state_dict` for tensor-only checkpoint entries
    compatible with ``torch.load(..., weights_only=True)``.

    Construction checks the scheme, supported dtype, and buffer names."""

    #: ``{"compress_method": <scheme name>}`` for a coded container, plus ``requested`` and ``reason``
    #: when a codec refused the tensor and it was stored verbatim.
    header: dict[str, Any]
    #: The scheme's buffers, named exactly as its ``buffer_names`` declares.
    buffers: dict[str, torch.Tensor]
    #: Shape of the tensor that comes back, before any padding the encode applied.
    shape: tuple[int, ...]
    #: Its dtype. No scheme changes it: a container holds the format it was given.
    dtype: torch.dtype

    def __post_init__(self):
        self.header = copy.deepcopy(self.header)
        self.buffers = dict(self.buffers)
        self.shape = tuple(self.shape)
        self.validate()

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
        """Whether the scheme that wrote this container reconstructs its input exactly."""
        return self.scheme.lossless

    @property
    def actual_bpp(self) -> float:
        """Bits per element of :attr:`shape`, serialized header included."""
        return self.storage_nbytes() * 8 / math.prod(self.shape)

    def validate(self) -> None:
        """Check the scheme, supported dtype, and required buffer names.

        This check does not inspect buffer contents."""
        scheme = self.scheme
        if not scheme.supports(self.dtype):
            raise ValueError(f"'{scheme.name}' does not support format {self.dtype}")
        if set(self.buffers) != set(scheme.buffer_names):
            raise ValueError(
                f"CompressedTensor buffers for {self.compress_method} must be "
                f"{list(scheme.buffer_names)}, got {sorted(self.buffers)}"
            )

    def to(self, device: str | torch.device) -> "CompressedTensor":
        """A copy of this container with every buffer on ``device``."""
        target = torch.device(device)
        return type(self)(
            header=self.header, buffers={name: value.to(device=target) for name, value in self.buffers.items()},
            shape=self.shape, dtype=self.dtype,
        )

    def to_dict(self) -> dict[str, Any]:
        """A plain-dict view of the four fields, for a caller that serializes them itself."""
        self.validate()
        return {"header": copy.deepcopy(self.header), "buffers": dict(self.buffers), "shape": self.shape, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompressedTensor":
        """Rebuild the container :meth:`to_dict` produced."""
        required = {"header", "buffers", "shape", "dtype"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"CompressedTensor data is missing fields: {sorted(missing)}")
        return cls(header=data["header"], buffers=data["buffers"], shape=data["shape"], dtype=data["dtype"])

    def _serialized_header_tensor(self) -> torch.Tensor:
        metadata = {
            "compress_method": self.compress_method, "dtype": str(self.dtype).removeprefix("torch."),
            "shape": list(self.shape), "buffer_names": list(self.buffers), "header": self.header,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return torch.frombuffer(bytearray(encoded), dtype=torch.uint8)

    def storage_nbytes(self, include_header: bool = True) -> int:
        """Bytes the buffers occupy, plus the serialized header unless ``include_header`` is false."""
        self.validate()
        total = sum(value.numel() * value.element_size() for value in self.buffers.values())
        if include_header:
            header = self._serialized_header_tensor()
            total += header.numel() * header.element_size()
        return total

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """The container as flat tensors under ``prefix``: one 1D uint8 header, then one entry per buffer."""
        self.validate()
        state = {f"{prefix}header": self._serialized_header_tensor()}
        state.update({f"{prefix}buffers.{name}": value for name, value in self.buffers.items()})
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor], prefix: str = "") -> "CompressedTensor":
        """Restore a container from the tensor entries produced by :meth:`state_dict`.

        Metadata is stored as JSON bytes in a uint8 tensor."""
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

        return cls(
            header=metadata["header"], buffers=buffers, shape=tuple(metadata["shape"]),
            dtype=_parse_dtype(metadata["dtype"]),
        )
