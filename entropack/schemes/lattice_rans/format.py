import logging
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import torch

from ..config import CompressionConfig, OneOf, Range
from ..base import buffers_fingerprint, cached_parse
from ..tile_ans.format import NUM_STATES

logger = logging.getLogger("entropack.lattice_rans")

BLOCK_SIZE = 256
SHARED_STAGING_HEADROOM = 8 * 1024
STATIC_SHARED = 256
MIN_SHARED_BLOCKS_PER_SM = 2

SUPPORTED_PROB_BITS = (9, 10, 11, 12, 13, 14, 15)
#: The margin only saves a re-measure and cannot change the outcome, because every step is re-checked against the real
#: histogram. Both lanes read this one definition so they settle on the same scale.
ALPHABET_COARSEN_MARGIN = 1.05
SUBSAMPLE_SEED_MULTIPLIER = 1000003


@dataclass
class LatticeRANSConfig(CompressionConfig):
    """Target-rate compression settings for finite, two-dimensional tensors.

    The encoder selects a quantization scale using sampled storage estimates, fits row
    reconstruction scales, and optionally applies per-row rate–distortion refinement.
    Decoding restores the input shape and dtype. Tile size and probability precision are
    stored with the compressed representation."""

    #: Requested bits per input element, from 1 to 11 including non-integer values. Read actual_bpp for the stored rate.
    target_bpp: Annotated[float, Range(1.0, 11.0)] = 4.0
    #: Probability-table precision. None or zero selects automatically.
    prob_bits: Annotated[int | None, OneOf(SUPPORTED_PROB_BITS, silent=(0,))] = None
    #: Elements per tile. Larger tiles reduce per-tile metadata and the number of independent decode tasks. None selects by target rate.
    tile_elements: Annotated[int | None, Range(1, None)] = None
    #: Per-row rate–distortion allocation sweeps using estimated coding costs. Zero disables this refinement.
    row_rdo_iterations: Annotated[int, Range(0, 8)] = 0
    #: Number of candidate quantization scales per row for rate–distortion refinement.
    row_rdo_candidates: Annotated[int, Range(1, None)] = 5
    #: Number of bisection steps in quantization-scale selection.
    scale_search_iterations: Annotated[int, Range(1, None)] = 12
    #: Sampling budget for scale search, measured in eight-value vectors.
    scale_search_max_vectors: Annotated[int, Range(1, None)] = 262144
    #: GPU decode block width. None selects a device-dependent value.
    threads_per_block: Annotated[int | None, Range(1, None)] = None
    #: Prefetch encoded payload into the GPU L2 cache during decoding.
    l2_prefetch: bool = True


LATTICE_DIM = 8
NUM_COORD_FIELDS = 8
NUM_COORD_STREAMS = 2 * NUM_COORD_FIELDS
NUM_STREAMS_FULL = 1 + NUM_COORD_STREAMS

SUPPORTED_DTYPES = (
    torch.float32, torch.float16, torch.bfloat16,
    torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint64, torch.uint32, torch.uint16, torch.uint8, torch.bool,
)
_INTEGER_DTYPES = frozenset({
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint64, torch.uint32, torch.uint16, torch.uint8,
})
_EXPECTED_DTYPES = {
    "payload": torch.uint16, "offsets": torch.uint32, "states": torch.uint32, "stream_meta": torch.int32,
    "freq_tables": torch.uint16, "scales": torch.float32, "layout": torch.int64,
}

STREAM_META_WIDTH = 4
META_N_SYMBOLS = 0
META_SYM_MIN = 1
META_FREQ_OFFSET = 2
META_ALPHABET = 3

LAYOUT_LEN = 3

BITS_PER_BYTE = 8
#: Not ``LAYOUT_LEN * 8``: tying it to the format would let a layout change move the scale the bisection converges on, so
#: encoded bytes would shift for reasons unrelated to the rate.
MODEL_LAYOUT_BYTES = 64
#: Pinned the same way and for the same reason, so narrowing ``stream_meta`` changed no encoded byte.
MODEL_STREAM_META_BYTES = 408


class LatticeBuffers(NamedTuple):
    payload: torch.Tensor
    offsets: torch.Tensor
    states: torch.Tensor
    stream_meta: torch.Tensor
    freq_tables: torch.Tensor
    scales: torch.Tensor
    layout: torch.Tensor


PACKED_KEYS = LatticeBuffers._fields


def recommended_tile_elements(target_bpp: float) -> int:
    if target_bpp <= 2.0:
        return 32768
    if target_bpp <= 4.0:
        return 16384
    if target_bpp <= 7.0:
        return 8192
    return 4096


def resolve_prob_bits(prob_bits: int | None, target_bpp: float) -> tuple[int, bool]:
    if prob_bits in (None, 0):
        return (11 if float(target_bpp) <= 7.0 else 12), True
    return int(prob_bits), False


def report_alphabet_clamp(scale: float, alphabet: int, table_size: int) -> None:
    logger.debug(
        "lattice_rans coarsened the lattice scale to %.6g so a coordinate alphabet of %d fits the "
        "%d-entry rANS table; actual_bpp for this tensor will fall below target_bpp", scale, alphabet, table_size,
    )


def subsample_index(rows: int, cols: int, sub_rows: int, device: torch.device) -> torch.Tensor | None:
    """A seeded permutation rather than a prefix or a fixed stride: neither is unbiased against the row order a weight happens
    to come in.
    """
    if sub_rows >= rows:
        return None
    generator = torch.Generator().manual_seed(rows * SUBSAMPLE_SEED_MULTIPLIER + cols)
    return torch.randperm(rows, generator=generator)[:sub_rows].to(device)


def _integer_high_bound(dtype: torch.dtype, like: torch.Tensor) -> float:
    edge = torch.tensor(float(torch.iinfo(dtype).max) + 1.0, dtype=like.dtype)
    below = torch.nextafter(edge, torch.full_like(edge, float("-inf")))
    return float(below)


def check_row_scales_finite(rms: torch.Tensor, dtype: torch.dtype) -> None:
    if not bool(torch.isfinite(rms).all()):
        raise ValueError(
            f"lattice_rans normalizes rows in fp32, and the row RMS of this {dtype} tensor overflowed. "
            "The limit is sum(weight**2) per row below ~3.4e38, i.e. |values| below "
            "sqrt(3.4e38 / cols) -- about 3e17 for a 4k-wide row, 1.6e18 for a 128-wide one. "
            "Rescale the source, or use tile_ans, which codes storage bytes and has no numeric " "range limit."
        )


def snap_to_container(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Saturation is required: torch's float->``float8_e4m3fn`` cast yields NaN above the format's largest finite value and
    float->``float16`` yields inf above its own, and the lattice overshoots the source range regularly. A NaN in a decoded
    weight destroys inference.

    Integers round half-to-even via ``torch.round``; the CUDA lane matches that with ``rintf``, since ``llroundf`` rounds
    half away from zero and would disagree on exact ties.
    """
    if dtype == torch.float32:
        return values
    if dtype == torch.bool:
        return values.round() != 0
    if dtype in _INTEGER_DTYPES:
        return values.round().clamp(float(torch.iinfo(dtype).min), _integer_high_bound(dtype, values)).to(dtype)
    limit = float(torch.finfo(dtype).max)
    return values.clamp(-limit, limit).to(dtype)


def vector_tile_elements(symbol_tile_elements: int) -> int:
    return max(32, symbol_tile_elements // 9)


def make_layout(cols: int, prob_bits: int, tile_elements: int) -> torch.Tensor:
    return torch.tensor([cols, prob_bits, tile_elements], dtype=torch.int64)


def parse_layout(layout: torch.Tensor) -> dict:
    if layout.dtype != torch.int64 or layout.ndim != 1 or layout.numel() != LAYOUT_LEN:
        raise ValueError(f"lattice_rans layout must be int64[{LAYOUT_LEN}]")
    cols, prob_bits, tile_elements = (int(x) for x in layout.detach().cpu().tolist())
    if cols <= 0 or cols % LATTICE_DIM != 0:
        raise ValueError("lattice_rans vector-plane format requires positive cols divisible by 8")
    if prob_bits not in SUPPORTED_PROB_BITS:
        raise ValueError(f"lattice_rans prob_bits {prob_bits} unsupported")
    if tile_elements <= 0:
        raise ValueError("lattice_rans tile_elements must be positive")
    return {"cols": cols, "prob_bits": prob_bits, "tile_elements": tile_elements}


def _check_container(buffers: dict, shape: tuple, dtype: torch.dtype) -> None:
    missing = [k for k in PACKED_KEYS if k not in buffers]
    if missing:
        raise ValueError(f"lattice_rans packed data is missing buffers: {missing}")
    if set(buffers) != set(PACKED_KEYS):
        extra = sorted(set(buffers) - set(PACKED_KEYS))
        raise ValueError(f"lattice_rans packed data has unexpected buffers: {extra}")
    if not all(isinstance(buffers[k], torch.Tensor) for k in PACKED_KEYS):
        raise TypeError("lattice_rans packed buffers must be torch.Tensor values")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"lattice_rans does not support output dtype {dtype}")
    if len(shape) != 2 or any(not isinstance(d, int) or d <= 0 for d in shape):
        raise ValueError(f"lattice_rans shape must be non-empty 2D, got {shape}")


def _check_buffers(buffers: dict) -> None:
    devices = {buffers[k].device for k in PACKED_KEYS}
    if len(devices) != 1:
        raise ValueError(f"lattice_rans packed buffers must share one device, got {devices}")
    for k in PACKED_KEYS:
        if not buffers[k].is_contiguous():
            raise ValueError(f"lattice_rans buffer '{k}' must be contiguous")
    for k, dt in _EXPECTED_DTYPES.items():
        if buffers[k].dtype != dt:
            raise ValueError(f"lattice_rans buffer '{k}' must be {dt}, got {buffers[k].dtype}")


def _check_scales(buffers: dict, info: dict) -> None:
    scales = buffers["scales"]
    if scales.ndim != 1 or scales.numel() != info["rows"]:
        raise ValueError(f"lattice_rans scales must be 1D with one entry per row, got {tuple(scales.shape)}")
    if not bool((torch.isfinite(scales) & (scales > 0)).all().item()):
        raise ValueError("lattice_rans scales must be finite and positive")


def _check_tile_geometry(buffers: dict, info: dict) -> None:
    n_streams, total_tiles = info["n_streams"], info["total_tiles"]
    meta = buffers["stream_meta"]
    if meta.ndim != 2 or tuple(meta.shape) != (n_streams, STREAM_META_WIDTH):
        raise ValueError(f"lattice_rans stream_meta must be ({n_streams},{STREAM_META_WIDTH}), got {tuple(meta.shape)}")
    if tuple(buffers["states"].shape) != (total_tiles, NUM_STATES):
        raise ValueError("lattice_rans states shape does not match total_tiles")
    if buffers["offsets"].numel() != total_tiles + 1:
        raise ValueError("lattice_rans offsets length must be total_tiles+1")
    offsets = buffers["offsets"].to(torch.int64)
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != buffers["payload"].numel():
        raise ValueError("lattice_rans payload offsets endpoints are invalid")
    if total_tiles > 0 and bool((offsets[1:] < offsets[:-1]).any().item()):
        raise ValueError("lattice_rans payload offsets must be monotone")


def _check_stream_metadata(buffers: dict, info: dict) -> None:
    n_streams = info["n_streams"]
    rows, cols = info["rows"], info["cols"]
    table_size = 1 << info["prob_bits"]
    meta_cpu = buffers["stream_meta"].detach().cpu().to(torch.int64)
    freq_cpu = buffers["freq_tables"].detach().cpu().to(torch.int64)
    freq_total = freq_cpu.numel()
    running_freq = 0
    symbol_counts = []
    for table in range(n_streams):
        row = meta_cpu[table]
        n_sym = int(row[META_N_SYMBOLS])
        freq_off = int(row[META_FREQ_OFFSET])
        alphabet = int(row[META_ALPHABET])
        if n_sym < 0 or alphabet < 0 or freq_off < 0:
            raise ValueError("lattice_rans table metadata has a negative field")
        if freq_off != running_freq or freq_off + alphabet > freq_total:
            raise ValueError("lattice_rans frequency-table ranges must be contiguous")
        if (n_sym == 0) != (alphabet == 0):
            raise ValueError("lattice_rans empty table metadata is inconsistent")
        if alphabet:
            if alphabet > table_size:
                raise ValueError("lattice_rans table alphabet exceeds probability precision")
            if int(freq_cpu[freq_off : freq_off + alphabet].sum().item()) != table_size:
                raise ValueError("lattice_rans normalized frequencies have an invalid sum")
        running_freq += alphabet
        symbol_counts.append(n_sym)
    if running_freq != freq_total:
        raise ValueError("lattice_rans frequency table has trailing entries")
    vectors = rows * (cols // LATTICE_DIM)
    if symbol_counts[0] != vectors or int(meta_cpu[0, META_SYM_MIN]) != 0:
        raise ValueError("lattice_rans coset table metadata is invalid")
    n0, n1 = symbol_counts[1], symbol_counts[2]
    if n0 + n1 != vectors:
        raise ValueError("lattice_rans conditioned table counts do not cover all vectors")
    for field in range(NUM_COORD_FIELDS):
        if symbol_counts[1 + 2 * field] != n0 or symbol_counts[2 + 2 * field] != n1:
            raise ValueError("lattice_rans conditioned table counts disagree across fields")


def decode_geometry(buffers: dict, shape: tuple) -> dict:
    stored = cached_parse(buffers["layout"], parse_layout, "_lattice_rans_layout")
    rows = int(shape[0])
    tile_vectors = vector_tile_elements(stored["tile_elements"])
    vectors = rows * (stored["cols"] // LATTICE_DIM)
    return {
        **stored, "rows": rows, "n_streams": NUM_STREAMS_FULL,
        "total_tiles": (vectors + tile_vectors - 1) // tile_vectors,
    }


def validate_packed(buffers: dict, shape: tuple, dtype: torch.dtype) -> dict:
    _check_container(buffers, shape, dtype)

    cached = getattr(buffers["layout"], "_lattice_rans_validation_cache", None)
    fp = buffers_fingerprint(buffers, shape, dtype)
    if cached is not None and cached[0] == fp:
        return cached[1]

    _check_buffers(buffers)
    info = decode_geometry(buffers, shape)
    rows, cols = tuple(shape)
    if not 0 <= info["cols"] - cols < LATTICE_DIM:
        raise ValueError(f"lattice_rans layout columns {info['cols']} do not match {(rows, cols)}")
    _check_scales(buffers, info)
    _check_tile_geometry(buffers, info)
    _check_stream_metadata(buffers, info)

    buffers["layout"]._lattice_rans_validation_cache = (fp, info)
    return info
