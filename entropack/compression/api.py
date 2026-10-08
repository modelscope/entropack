import logging
import warnings

import torch

from ..registry import DispatchError, get_scheme, name_for_config
from ..schemes import CompressionConfig, RawConfig
from ..schemes.config import validate_config
from .compressed_tensor import CompressedTensor

logger = logging.getLogger("entropack")


class CompressionFallbackWarning(RuntimeWarning):
    """Warning emitted when compression falls back to storing the tensor uncompressed."""


def compress(tensor: torch.Tensor, config: CompressionConfig) -> CompressedTensor:
    """Compress a tensor using the selected scheme.

    Args:
        tensor: input values. The container preserves the input shape, dtype, and device.
        config: configuration selecting the compression scheme and execution backend.

    Returns:
        A :class:`CompressedTensor` containing the encoded data and metadata.

    Encoding failures can return an uncompressed ``raw`` container with a
    :class:`CompressionFallbackWarning`. Its header records the requested scheme and
    failure reason. Invalid configurations and backend dispatch failures raise errors.
    """
    if isinstance(tensor, CompressedTensor):
        raise TypeError("compress expects an uncompressed tensor; decompress the container before recompressing")
    scheme = get_scheme(name_for_config(config))
    validate_config(config)
    try:
        packed = scheme.encode(tensor, config)
    except DispatchError:
        raise
    except Exception as error:
        logger.warning(
            "%s could not encode a %s %s tensor, storing it uncompressed: %s",
            scheme.name, tuple(tensor.shape), tensor.dtype, error,
        )
        return _compress_raw(tensor, scheme.name, error)
    return CompressedTensor(header={"compress_method": scheme.name}, buffers=packed, shape=tuple(tensor.shape), dtype=tensor.dtype)


def _compress_raw(tensor: torch.Tensor, requested: str, error: Exception) -> CompressedTensor:
    reason = f"{type(error).__name__}: {error}"
    warnings.warn(
        f"entropack stored a {tuple(tensor.shape)} {tensor.dtype} tensor uncompressed, at "
        f"{tensor.element_size() * 8} bits per parameter: compress_method={requested!r} failed with {reason}",
        CompressionFallbackWarning, stacklevel=3,
    )
    return CompressedTensor(
        header={"compress_method": "raw", "requested": requested, "reason": reason},
        buffers=get_scheme("raw").encode(tensor, RawConfig()), shape=tuple(tensor.shape), dtype=tensor.dtype,
    )


def decompress(compressed: CompressedTensor, config: CompressionConfig) -> torch.Tensor:
    """Restore a tensor from its compressed representation.

    Args:
        compressed: container to decode. Its header identifies the compression scheme.
        config: configuration for the same scheme. Decode settings select the backend and
            execution options. Encode settings such as the target bitrate do not recompress
            the stored data.

    Returns:
        A tensor with the container's shape and dtype, on the device holding its buffers.
    """
    validate_config(config)
    restored = compressed.scheme.decode(compressed.buffers, shape=compressed.shape, dtype=compressed.encoded_dtype, config=config)
    return restored.to(compressed.dtype)
