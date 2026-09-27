import math
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import numpy as np
import torch

from ..config import CompressionConfig, Range
from ..base import cached_parse

BLOCK_SIZE = 256
MAX_RESIDENT_BLOCKS_PER_SM = 32


@dataclass
class DFloat11Config(CompressionConfig):
    """Lossless BF16 compression settings.

    The coding-region settings are stored with the compressed tensor. Decoding reads them
    from the representation rather than from the configuration."""

    #: Bitstream bytes per coding region. Controls metadata overhead and decode parallelism.
    bytes_per_thread: Annotated[int | None, Range(1, None)] = 16
    #: Threads per block in the encoded layout.
    threads_per_block: Annotated[int | None, Range(1, None)] = 128


META_COUNT_BITS = 11
META_COUNT_MASK = (1 << META_COUNT_BITS) - 1


class DFloat11Buffers(NamedTuple):
    encoded_exponent: torch.Tensor
    sign_mantissa: torch.Tensor
    luts: torch.Tensor
    output_positions: torch.Tensor
    thread_meta: torch.Tensor
    layout: torch.Tensor


PACKED_KEYS = DFloat11Buffers._fields


def reconstruct_bf16_bits(exponent: np.ndarray, sign_mantissa: np.ndarray) -> np.ndarray:
    exp = exponent.astype(np.uint16)
    sm = sign_mantissa.astype(np.uint16)
    return (((sm & 0x80) << 8) | (exp << 7) | (sm & 0x7F)).astype(np.uint16)


def pack_thread_meta(gaps, counts) -> torch.Tensor:
    g = np.asarray(gaps, dtype=np.int64)
    c = np.asarray(counts, dtype=np.int64)
    if g.size != c.size:
        raise ValueError(f"gaps/counts length mismatch: {g.size} vs {c.size}")
    if g.size and int(g.max()) > 31:
        raise ValueError(f"gap {int(g.max())} exceeds 5 bits; max Huffman code length must be < 32")
    if c.size and int(c.max()) > META_COUNT_MASK:
        raise ValueError(f"per-thread symbol count {int(c.max())} exceeds {META_COUNT_BITS} bits; lower BYTES_PER_THREAD")
    return torch.from_numpy(((g << META_COUNT_BITS) | c).astype(np.uint16))


def make_layout(bytes_per_thread: int, threads_per_block: int, max_block_elems: int):
    return torch.tensor([bytes_per_thread, threads_per_block, max_block_elems], dtype=torch.int32)


def parse_layout(layout) -> tuple[int, int, int]:
    if layout.numel() != 3:
        raise ValueError(f"dfloat11 layout must contain 3 int32 values, got {layout.numel()}")
    bpt, tpb, max_block_elems = (int(v) for v in layout.detach().cpu().tolist())
    if bpt <= 0 or tpb <= 0 or max_block_elems < 0:
        raise ValueError(f"invalid dfloat11 layout values: bpt={bpt}, tpb={tpb}, max_block_elems={max_block_elems}")
    return bpt, tpb, max_block_elems


def parse_layout_cached(layout) -> tuple[int, int, int]:
    return cached_parse(layout, parse_layout, "_dfloat11_layout")


def max_block_elems(output_positions) -> int:
    op = np.asarray(output_positions, dtype=np.int64)
    if op.size < 2:
        return int(op[0]) if op.size else 0
    return int(np.diff(op).max())


def validate_packed(buffers: dict, shape=None) -> None:
    missing = [key for key in PACKED_KEYS if key not in buffers]
    if missing:
        raise ValueError(f"dfloat11 packed data is missing buffers: {missing}")
    if not all(isinstance(buffers[key], torch.Tensor) for key in PACKED_KEYS):
        raise TypeError("dfloat11 packed buffers must be torch.Tensor values")

    expected = {
        "encoded_exponent": (torch.uint8, 1), "sign_mantissa": (torch.uint8, 1),
        "luts": (torch.uint8, 2), "output_positions": (torch.uint32, 1),
        "thread_meta": (torch.uint16, 1), "layout": (torch.int32, 1),
    }
    for key, (dtype, ndim) in expected.items():
        tensor = buffers[key]
        if tensor.dtype != dtype or tensor.ndim != ndim:
            raise ValueError(
                f"dfloat11 buffer '{key}' must be {ndim}D {dtype}, got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )

    devices = {buffers[key].device for key in PACKED_KEYS}
    if len(devices) != 1:
        raise ValueError(f"dfloat11 packed buffers must share one device, got {devices}")
    if buffers["luts"].shape[0] < 2 or buffers["luts"].shape[1] != 256:
        raise ValueError(f"dfloat11 luts must have shape (num_levels + 1, 256), got {tuple(buffers['luts'].shape)}")

    bpt, tpb, max_elems = parse_layout_cached(buffers["layout"])
    n_elements = buffers["sign_mantissa"].numel()
    if n_elements == 0:
        raise ValueError("dfloat11 does not support empty tensors")
    if shape is not None:
        normalized_shape = tuple(shape)
        if any(not isinstance(dim, int) or dim < 0 for dim in normalized_shape):
            raise ValueError(f"invalid dfloat11 tensor shape: {normalized_shape}")
        if math.prod(normalized_shape) != n_elements:
            raise ValueError(
                f"dfloat11 shape {normalized_shape} has {math.prod(normalized_shape)} "
                f"elements but sign_mantissa has {n_elements}"
            )

    n_bytes = buffers["encoded_exponent"].numel()
    blocks = (n_bytes + bpt * tpb - 1) // (bpt * tpb)
    if buffers["thread_meta"].numel() != blocks * tpb:
        raise ValueError("dfloat11 thread_meta length does not match layout/bitstream")
    if buffers["output_positions"].numel() != blocks + 1:
        raise ValueError("dfloat11 output_positions length does not match layout/bitstream")

    positions = buffers["output_positions"].detach().cpu().to(torch.int64)
    if positions[0] != 0 or positions[-1] != n_elements:
        raise ValueError("dfloat11 output_positions endpoints are invalid")
    differences = positions[1:] - positions[:-1]
    if (differences < 0).any() or (differences.max() if differences.numel() else 0) != max_elems:
        raise ValueError("dfloat11 output_positions are not monotone or mismatch layout")

    metadata = buffers["thread_meta"].detach().cpu().to(torch.int64)
    gaps = metadata >> META_COUNT_BITS
    counts = metadata & META_COUNT_MASK
    if (gaps > 31).any() or counts.sum() != n_elements:
        raise ValueError("dfloat11 thread_meta fields are invalid")
    lengths = buffers["luts"][-1].detach().cpu()
    if int(lengths.max()) > 32:
        raise ValueError("dfloat11 Huffman code length exceeds 32 bits")
