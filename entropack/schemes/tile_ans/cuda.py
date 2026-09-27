from pathlib import Path

import cupy
import numpy as np
import torch

from ...backends.cuda import device as _device_caps
from ...backends.cuda.kernels import KernelLibrary
from ...backends.cuda.kernels import device_index as _device_index
from ...backends.cuda.kernels import external_stream as _external_stream
from ...backends.cuda.kernels import pointer as _pointer
from .eager import build_tables_from_counts, lane_modes_from_counts, select_coding_options
from .format import (
    AUTO_PROB_BITS, BLOCK_SIZE, ENCODE_TABLE_SHARED_BYTES, HISTOGRAM_MAX_WARPS, HISTOGRAM_MIN_WARPS,
    HISTOGRAM_WARP_BUDGET, LANE_ANS, LANE_RAW, NUM_STATES, TileBuffers, make_layout, num_streams,
    num_tiles, parse_layout_cached,
)

_CUDA_PATH = Path(__file__).parent / "tile_ans.cu"
_KERNEL_NAMES = (
    "tile_ans_histogram_kernel", "tile_ans_encode_kernel", "tile_ans_compact_kernel", "tile_ans_decode_raw0_ans1_kernel",
    "tile_ans_decode_raw3_ans1_kernel", "tile_ans_decode_all_raw_kernel", "tile_ans_decode_kernel",
)
_LIBRARY = KernelLibrary(
    key="tile_ans", source=_CUDA_PATH, defines=lambda _device, probability_bits: (f"TILE_ANS_PROB_BITS={probability_bits}",),
    includes=(_CUDA_PATH.parent,), kernel_names=_KERNEL_NAMES,
)
_kernel = _LIBRARY.kernel


def _lane_histogram(contiguous, caps, device_index, torch_stream, probability_bits):
    num_elements = contiguous.numel()
    num_lanes = contiguous.element_size()
    histograms = torch.zeros((num_lanes, 256), dtype=torch.int64, device=contiguous.device)
    histogram_warps = min(
        HISTOGRAM_MAX_WARPS, max(HISTOGRAM_MIN_WARPS, HISTOGRAM_WARP_BUDGET // num_lanes),
    )
    histogram_threads = histogram_warps * caps.warp_size
    histogram_blocks = caps.grid(-(-num_elements // histogram_threads), histogram_threads)
    histogram_shared = histogram_warps * num_lanes * 256 * 4
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, probability_bits, "tile_ans_histogram_kernel")(
            (histogram_blocks,), (histogram_threads,),
            (_pointer(contiguous), _pointer(histograms), np.int64(num_elements), np.int32(num_lanes)),
            shared_mem=histogram_shared,
        )
    return histograms.cpu().numpy()


def _lane_tables(counts, probability_bits, raw_lane_threshold, tile_elements, device):
    if probability_bits == AUTO_PROB_BITS:
        probability_bits, frequencies, cdfs, decode_tables, lane_modes = select_coding_options(
            counts, raw_lane_threshold, tile_elements
        )
    else:
        frequencies, cdfs, decode_tables = build_tables_from_counts(counts, probability_bits)
        lane_modes = lane_modes_from_counts(counts, raw_lane_threshold, tile_elements)
    return (
        probability_bits, int(np.count_nonzero(lane_modes == LANE_ANS)), torch.from_numpy(frequencies).to(device),
        torch.from_numpy(cdfs).to(device), torch.from_numpy(decode_tables).to(device), torch.from_numpy(lane_modes).to(device),
    )


def encode(
    *, weight: torch.Tensor, tile_elements: int, probability_bits: int, raw_lane_threshold: float,
    threads_per_block: int | None,
) -> TileBuffers:
    contiguous = weight.contiguous()
    num_elements = contiguous.numel()
    num_lanes = contiguous.element_size()
    streams = num_streams(num_elements, num_lanes, tile_elements)
    device = contiguous.device
    device_index = _device_index(contiguous)
    torch_stream = torch.cuda.current_stream(device)
    caps = _device_caps.caps(device)

    counts = _lane_histogram(contiguous, caps, device_index, torch_stream, probability_bits)
    (
        probability_bits, num_ans_lanes, frequencies_gpu, cdfs_gpu, decode_tables_gpu, lane_modes_gpu,
    ) = _lane_tables(counts, probability_bits, raw_lane_threshold, tile_elements, device)

    tiles = num_tiles(num_elements, tile_elements)
    states = torch.empty((tiles * num_ans_lanes, NUM_STATES), dtype=torch.uint32, device=device)
    word_counts = torch.empty(streams, dtype=torch.uint32, device=device)
    scratch = torch.empty(streams * tile_elements, dtype=torch.uint16, device=device)
    threads = _device_caps.resolve_threads(caps, threads_per_block, BLOCK_SIZE)
    warps = threads // caps.warp_size
    blocks = -(-tiles // warps)
    args = (
        _pointer(contiguous), _pointer(frequencies_gpu), _pointer(cdfs_gpu), _pointer(lane_modes_gpu), _pointer(scratch),
        _pointer(word_counts), _pointer(states), np.int64(num_elements), np.int32(tile_elements), np.int32(num_lanes),
        np.int32(tiles),
    )

    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, probability_bits, "tile_ans_encode_kernel")(
            (blocks, num_lanes), (threads,), args, shared_mem=ENCODE_TABLE_SHARED_BYTES,
        )
        counts_cp = cupy.from_dlpack(word_counts)
        offsets64 = torch.empty(streams + 1, dtype=torch.int64, device=device)
        offsets_cp = cupy.from_dlpack(offsets64)
        offsets_cp[0] = 0
        cupy.cumsum(counts_cp, dtype=cupy.int64, out=offsets_cp[1:])
        total_words = int(offsets64[-1].item())
    if total_words >= 1 << 32:
        raise ValueError("tile_ans payload exceeds uint32 offset capacity")

    offsets = offsets64.to(torch.uint32)
    payload = torch.empty(total_words, dtype=torch.uint16, device=device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, probability_bits, "tile_ans_compact_kernel")(
            (streams,), (threads,),
            (_pointer(scratch), _pointer(offsets), _pointer(payload), np.int32(tile_elements), np.int32(streams)),
        )
    layout = make_layout(num_elements, num_lanes, tile_elements, probability_bits).to(device)
    return TileBuffers(
        payload=payload, offsets=offsets, states=states, decode_tables=decode_tables_gpu, lane_modes=lane_modes_gpu,
        layout=layout,
    )


def decode(buffers: TileBuffers, *, dtype: torch.dtype, threads_per_block: int | None) -> torch.Tensor:
    payload, offsets, states = buffers.payload, buffers.offsets, buffers.states
    decode_tables, lane_modes, layout = buffers.decode_tables, buffers.lane_modes, buffers.layout
    tile_elements, probability_bits, num_lanes, num_elements = parse_layout_cached(layout)
    output = torch.empty(num_elements * num_lanes, dtype=torch.uint8, device=payload.device)
    device_index = _device_index(payload)
    torch_stream = torch.cuda.current_stream(payload.device)
    caps = _device_caps.caps(payload.device)

    tiles = num_tiles(num_elements, tile_elements)
    lane_mode_values = getattr(layout, "_tile_ans_lane_modes", None)
    if lane_mode_values is None:
        lane_mode_values = tuple(int(value) for value in lane_modes.detach().cpu().tolist())
        layout._tile_ans_lane_modes = lane_mode_values

    all_raw_writer = all(mode == LANE_RAW for mode in lane_mode_values)
    paired_writer = num_lanes == 2 and lane_mode_values == (LANE_RAW, LANE_ANS)
    quad_writer = num_lanes == 4 and lane_mode_values == (LANE_RAW, LANE_RAW, LANE_RAW, LANE_ANS)
    table_shared = (1 << probability_bits) * 4
    grid_lanes = 1
    block_owns_tile = False
    if all_raw_writer:
        kernel_name, shared_bytes = "tile_ans_decode_all_raw_kernel", 0
        block_owns_tile = True
    elif paired_writer:
        # bf16 splits into a raw low byte and an ANS-coded high byte, so the coded lane's table is small enough to stage in
        # shared memory: renormalization then drops its bounds check and the lane mask is hoisted out of the symbol loop.
        kernel_name = "tile_ans_decode_raw0_ans1_kernel"
        shared_bytes = table_shared
    elif quad_writer:
        kernel_name, shared_bytes = "tile_ans_decode_raw3_ans1_kernel", 0
    else:
        kernel_name, shared_bytes = "tile_ans_decode_kernel", table_shared
        grid_lanes = num_lanes

    kernel = _kernel(device_index, probability_bits, kernel_name)
    args = (
        _pointer(payload), _pointer(offsets), _pointer(states), _pointer(decode_tables), _pointer(lane_modes), _pointer(output),
        np.int64(num_elements), np.int32(tile_elements), np.int32(num_lanes), np.int32(tiles),
    )

    def launch_decode(threads: int) -> None:
        blocks = tiles if block_owns_tile else -(-tiles // (threads // caps.warp_size))
        grid = (blocks, grid_lanes) if grid_lanes > 1 else (blocks,)
        with cupy.cuda.Device(device_index), _external_stream(torch_stream):
            kernel(grid, (threads,), args, shared_mem=shared_bytes)

    launch_decode(_device_caps.resolve_threads(caps, threads_per_block, BLOCK_SIZE))
    return output.view(dtype)
