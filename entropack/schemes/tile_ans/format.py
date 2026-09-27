import math
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import torch

from ..config import CompressionConfig, OneOf, Range
from ..base import cached_parse

BLOCK_SIZE = 256
HISTOGRAM_WARP_BUDGET = 16
HISTOGRAM_MIN_WARPS = 2
HISTOGRAM_MAX_WARPS = 8
ENCODE_TABLE_SHARED_BYTES = 2 * 256 * 2

AUTO_PROB_BITS = 0
AUTO_TILE_ELEMENTS = 0
SUPPORTED_PROB_BITS = (9, 10, 11, 12)
NUM_STATES = 32
STATE_MIN = 1 << 15
LANE_ANS = 0
LANE_RAW = 1

TILE_ELEMENTS_LIMIT = 1 << 31
RAW_LANE_THRESHOLD_LIMIT = 8.0
TILE_ELEMENTS_MESSAGE = (
    "tile_elements must be positive, or 0 for automatic selection, and fit the CUDA int32 launch ABI"
)


@dataclass
class TileANSConfig(CompressionConfig):
    """Lossless compression settings for the supported tensor dtypes.

    Tile size and probability precision are stored with the compressed tensor. A zero value
    requests automatic selection. Block width controls GPU execution."""

    #: Elements per independent tile. Zero selects 4096, or 8192 for tensors larger than 32 MiB.
    tile_elements: Annotated[int, Range(0, TILE_ELEMENTS_LIMIT - 1, message=TILE_ELEMENTS_MESSAGE)] = AUTO_TILE_ELEMENTS
    #: Probability-table precision. Zero selects using the tensor symbol histogram.
    probability_bits: Annotated[int, OneOf(SUPPORTED_PROB_BITS, silent=(AUTO_PROB_BITS,))] = AUTO_PROB_BITS
    #: Byte streams at or above this estimated cost in bits per symbol are stored directly.
    raw_lane_threshold: Annotated[float, Range(0.0, RAW_LANE_THRESHOLD_LIMIT)] = 7.9
    #: GPU block width for encoding and decoding. None selects a device-dependent value.
    threads_per_block: Annotated[int | None, Range(1, None)] = None


OPTIONS_BY_DTYPE = {
    torch.float32: (8192, 10), torch.float16: (8192, 11), torch.bfloat16: (0, 11),
    torch.float8_e4m3fn: (8192, 0), torch.float8_e4m3fnuz: (8192, 0),
    torch.float8_e5m2: (8192, 0), torch.float8_e5m2fnuz: (8192, 0),
    torch.int64: (8192, 10), torch.int32: (16384, 10), torch.int16: (8192, 9),
    torch.int8: (8192, 0), torch.uint64: (8192, 9), torch.uint32: (8192, 10),
    torch.uint16: (8192, 9), torch.uint8: (8192, 0), torch.bool: (8192, 9),
}


class TileBuffers(NamedTuple):
    payload: torch.Tensor
    offsets: torch.Tensor
    states: torch.Tensor
    decode_tables: torch.Tensor
    lane_modes: torch.Tensor
    layout: torch.Tensor


PACKED_KEYS = TileBuffers._fields


def num_tiles(num_elements: int, tile_elements: int) -> int:
    return -(-num_elements // tile_elements)


def num_streams(num_elements: int, num_lanes: int, tile_elements: int) -> int:
    return num_tiles(num_elements, tile_elements) * num_lanes


def make_layout(num_elements: int, num_lanes: int, tile_elements: int, probability_bits: int):
    if probability_bits not in SUPPORTED_PROB_BITS:
        raise ValueError(f"probability_bits must be one of {SUPPORTED_PROB_BITS}")
    return torch.tensor([tile_elements, probability_bits, num_lanes, num_elements], dtype=torch.int64)


def parse_layout(layout: torch.Tensor) -> tuple[int, int, int, int]:
    if layout.dtype != torch.int64 or layout.ndim != 1 or layout.numel() != 4:
        raise ValueError("tile_ans layout must be int64[4]")
    tile_elements, prob_bits, num_lanes, num_elements = (int(value) for value in layout.detach().cpu().tolist())
    if tile_elements <= 0 or prob_bits not in SUPPORTED_PROB_BITS or num_lanes <= 0 or num_elements <= 0:
        raise ValueError(
            "invalid tile_ans layout values: " f"tile_elements={tile_elements}, prob_bits={prob_bits}, "
            f"num_lanes={num_lanes}, num_elements={num_elements}"
        )
    return tile_elements, prob_bits, num_lanes, num_elements


def parse_layout_cached(layout: torch.Tensor) -> tuple[int, int, int, int]:
    return cached_parse(layout, parse_layout, "_tile_ans_layout")


def validate_packed(buffers: dict[str, torch.Tensor], shape, dtype: torch.dtype) -> None:
    missing = [key for key in PACKED_KEYS if key not in buffers]
    if missing:
        raise ValueError(f"tile_ans packed data is missing buffers: {missing}")
    if not all(isinstance(buffers[key], torch.Tensor) for key in PACKED_KEYS):
        raise TypeError("tile_ans packed buffers must be torch.Tensor values")

    expected_layout = {
        "payload": (torch.uint16, 1), "offsets": (torch.uint32, 1), "states": (torch.uint32, 2),
        "decode_tables": (torch.uint32, 2), "lane_modes": (torch.uint8, 1), "layout": (torch.int64, 1),
    }
    for key, (expected_dtype, ndim) in expected_layout.items():
        tensor = buffers[key]
        if not tensor.is_contiguous():
            raise ValueError(f"tile_ans buffer '{key}' must be contiguous")
        if tensor.dtype != expected_dtype or tensor.ndim != ndim:
            raise ValueError(
                f"tile_ans buffer '{key}' must be {ndim}D {expected_dtype}, "
                f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )

    devices = {buffers[key].device for key in PACKED_KEYS}
    if len(devices) != 1:
        raise ValueError(f"tile_ans packed buffers must share one device, got {devices}")

    tile_elements, prob_bits, num_lanes, num_elements = parse_layout(buffers["layout"])
    element_size = torch.empty((), dtype=dtype).element_size()
    if num_lanes != element_size:
        raise ValueError(f"tile_ans layout has {num_lanes} byte lanes but dtype {dtype} uses {element_size} bytes")
    normalized_shape = tuple(shape)
    if any(not isinstance(dim, int) or dim < 0 for dim in normalized_shape):
        raise ValueError(f"invalid tile_ans tensor shape: {normalized_shape}")
    if math.prod(normalized_shape) != num_elements:
        raise ValueError(f"tile_ans shape {normalized_shape} does not match {num_elements} elements")

    streams = num_streams(num_elements, num_lanes, tile_elements)
    tiles = num_tiles(num_elements, tile_elements)
    if tuple(buffers["decode_tables"].shape) != (num_lanes, 1 << prob_bits):
        raise ValueError("tile_ans decode_tables shape does not match layout")
    if buffers["lane_modes"].numel() != num_lanes:
        raise ValueError("tile_ans lane_modes length does not match layout")
    lane_modes = buffers["lane_modes"].detach().cpu()
    if ((lane_modes != LANE_ANS) & (lane_modes != LANE_RAW)).any():
        raise ValueError("tile_ans lane_modes contains an unknown codec mode")
    ans_lanes = int((lane_modes == LANE_ANS).sum())
    if tuple(buffers["states"].shape) != (tiles * ans_lanes, NUM_STATES):
        raise ValueError("tile_ans states shape does not match ANS lanes/layout")
    if buffers["offsets"].numel() != streams + 1:
        raise ValueError("tile_ans offsets length does not match layout")

    offsets = buffers["offsets"].detach().cpu().to(torch.int64)
    if offsets[0] != 0 or offsets[-1] != buffers["payload"].numel():
        raise ValueError("tile_ans offsets endpoints are invalid")
    if (offsets[1:] < offsets[:-1]).any():
        raise ValueError("tile_ans offsets must be monotone")

    for byte_lane, mode in enumerate(lane_modes.tolist()):
        if mode != LANE_RAW:
            continue
        first_stream = byte_lane * tiles
        sizes = offsets[first_stream + 1 : first_stream + tiles + 1] - offsets[first_stream : first_stream + tiles]
        raw_words = torch.full_like(sizes, (tile_elements + 1) // 2)
        raw_words[-1] = (num_elements - (tiles - 1) * tile_elements + 1) // 2
        if not torch.equal(sizes, raw_words):
            raise ValueError("tile_ans raw lane payload length is invalid")
