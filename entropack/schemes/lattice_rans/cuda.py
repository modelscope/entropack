from dataclasses import dataclass, replace
from pathlib import Path

import cupy
import numpy as np
import torch

from ...backends.cuda import device as _device_caps
from ...backends.cuda.kernels import KernelLibrary
from ...backends.cuda.kernels import device_index as _device_index
from ...backends.cuda.kernels import ensure_dynamic_shared as _ensure_dynamic_shared
from ...backends.cuda.kernels import external_stream as _external_stream
from ...backends.cuda.kernels import pointer as _pointer
from ..tile_ans.format import NUM_STATES
from . import rans, rdo
from . import eager as _eager
from .format import (
    ALPHABET_COARSEN_MARGIN, BITS_PER_BYTE, BLOCK_SIZE,
    LATTICE_DIM, META_ALPHABET, META_FREQ_OFFSET, META_N_SYMBOLS, META_SYM_MIN, MIN_SHARED_BLOCKS_PER_SM,
    MODEL_LAYOUT_BYTES, NUM_COORD_STREAMS, NUM_STREAMS_FULL, SHARED_STAGING_HEADROOM, STATIC_SHARED, STREAM_META_WIDTH,
    LatticeBuffers, check_row_scales_finite, decode_geometry, make_layout, report_alphabet_clamp, resolve_prob_bits,
    snap_to_container, subsample_index, vector_tile_elements,
)

_CUDA_PATH = Path(__file__).parent / "lattice_rans.cu"
_TILE_INCLUDE_PATH = Path(__file__).parent.parent / "tile_ans"
_KERNEL_NAMES = (
    "e8_quantize_fields_kernel",
    "e8_quantize_fields_f32_kernel",
    "e8_refit_scales_kernel",
    "e8_refit_scales_f32_kernel",
    "e8_minmax_fields_kernel",
    "e8_histogram_kernel",
    "e8_rans_encode_vector_kernel",
    "e8_compact_kernel",
    "e8_decode_vector_shlut8pf_kernel",
    "e8_decode_vector_shlut8pf_g_kernel",
    "e8_decode_vector_packed32pf_kernel",
    "e8_decode_vector_packed32pf_g_kernel",
    "e8_decode_vector_packed32_kernel",
    "e8_decode_vector_packed32_g_kernel",
    "e8_decode_vector_fused_kernel",
    "e8_decode_vector_fused_g_kernel",
)
_LIBRARY = KernelLibrary(
    key="lattice_rans", source=_CUDA_PATH,
    defines=lambda _device, probability_bits: (f"TILE_ANS_PROB_BITS={probability_bits}",), includes=(_TILE_INCLUDE_PATH,),
    kernel_names=_KERNEL_NAMES,
)
_kernel = _LIBRARY.kernel

_MIN_RMS = _eager.MIN_RMS
_INT_MAX = 0x7FFFFFFF
_INT_MIN = -0x7FFFFFFF - 1
_MM_CMAX = 2 * (NUM_STREAMS_FULL - 1)
_MM_LEN = _MM_CMAX + 1
_MINMAX_TEMPLATE = np.array(
    [(_INT_MAX if (t & 1) == 0 and t != _MM_CMAX else _INT_MIN) for t in range(_MM_LEN)], dtype=np.int32,
)

_ENCODE_KERNELS = {
    torch.bfloat16: ("e8_quantize_fields_kernel", "e8_refit_scales_kernel"),
    torch.float32: ("e8_quantize_fields_f32_kernel", "e8_refit_scales_f32_kernel"),
}
_STORE_KINDS = {
    torch.float32: 0, torch.float16: 1, torch.float8_e4m3fn: 2, torch.float8_e5m2: 3, torch.int8: 4, torch.int16: 5,
    torch.int32: 6, torch.int64: 7, torch.uint8: 8, torch.uint16: 9, torch.uint32: 10, torch.uint64: 11, torch.bool: 12,
}
_SCRATCH_DTYPES = frozenset({torch.float8_e4m3fnuz, torch.float8_e5m2fnuz})
_GENERIC_DECODE_KERNEL = {
    "e8_decode_vector_shlut8pf_kernel": "e8_decode_vector_shlut8pf_g_kernel",
    "e8_decode_vector_packed32pf_kernel": "e8_decode_vector_packed32pf_g_kernel",
    "e8_decode_vector_packed32_kernel": "e8_decode_vector_packed32_g_kernel",
    "e8_decode_vector_fused_kernel": "e8_decode_vector_fused_g_kernel",
}


def _grid(caps, total: int, threads: int = BLOCK_SIZE) -> int:
    return caps.grid(-(-total // threads), threads)


@dataclass(frozen=True)
class _QuantizeRequest:
    weight: torch.Tensor
    rms: torch.Tensor
    scale: float
    prob_bits: int
    rows: int
    cols: int
    device: torch.device
    device_index: int

    @classmethod
    def of(cls, weight: torch.Tensor, rms: torch.Tensor, scale: float, prob_bits: int):
        rows, cols = weight.shape
        return cls(
            weight=weight, rms=rms, scale=scale, prob_bits=prob_bits, rows=rows, cols=cols, device=weight.device,
            device_index=_device_index(weight),
        )


@dataclass(frozen=True)
class _QuantizeResult:
    counts: list
    sizes: list
    sym_min: list
    alphabets: list
    fields: torch.Tensor
    c_arr: torch.Tensor
    minmax: torch.Tensor
    scales: torch.Tensor | None = None
    row_sse: torch.Tensor | None = None


def _quantize_pass(request: _QuantizeRequest) -> _QuantizeResult:
    weight, rms = request.weight, request.rms
    rows, cols = request.rows, request.cols
    device, device_index, prob_bits = request.device, request.device_index, request.prob_bits
    V = rows * (cols // LATTICE_DIM)
    fields = torch.empty(V * LATTICE_DIM, dtype=torch.int32, device=device)
    c_arr = torch.empty(V, dtype=torch.int32, device=device)
    minmax = torch.from_numpy(_MINMAX_TEMPLATE.copy()).to(device)
    torch_stream = torch.cuda.current_stream(device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, prob_bits, _ENCODE_KERNELS[weight.dtype][0])(
            (_grid(_device_caps.caps(device_index), V),),
            (BLOCK_SIZE,),
            (
                _pointer(weight), _pointer(rms), np.float32(request.scale), np.int32(rows), np.int32(cols),
                np.int32(cols // LATTICE_DIM), _pointer(fields), _pointer(c_arr), _pointer(minmax),
            ),
        )
    minmax_np = minmax.cpu().numpy()
    counts, sizes, sym_min, alphabets = _counts_from_minmax(
        fields, c_arr, minmax, minmax_np, V, device, device_index, prob_bits
    )
    return _QuantizeResult(
        counts=counts, sizes=sizes, sym_min=sym_min, alphabets=alphabets, fields=fields, c_arr=c_arr, minmax=minmax,
    )


def _refit_row_scales(request: _QuantizeRequest, result: _QuantizeResult):
    weight, rms = request.weight, request.rms
    rows, cols = request.rows, request.cols
    device_index, prob_bits = request.device_index, request.prob_bits
    scales = torch.empty(rows, dtype=torch.float32, device=weight.device)
    row_sse = torch.empty(rows, dtype=torch.float32, device=weight.device)
    torch_stream = torch.cuda.current_stream(weight.device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, prob_bits, _ENCODE_KERNELS[weight.dtype][1])(
            (rows,),
            (BLOCK_SIZE,),
            (
                _pointer(weight), _pointer(result.fields), _pointer(result.c_arr), _pointer(rms), np.float32(request.scale),
                _pointer(scales), _pointer(row_sse), np.int32(rows), np.int32(cols), np.int32(cols // LATTICE_DIM),
            ),
        )
    return scales, row_sse


def _summarize_fields(request: _QuantizeRequest, fields, c_arr) -> _QuantizeResult:
    rows, cols = request.rows, request.cols
    device, device_index, prob_bits = request.device, request.device_index, request.prob_bits
    V = rows * (cols // LATTICE_DIM)
    minmax = torch.from_numpy(_MINMAX_TEMPLATE.copy()).to(device)
    torch_stream = torch.cuda.current_stream(device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, prob_bits, "e8_minmax_fields_kernel")(
            (_grid(_device_caps.caps(device_index), V),), (BLOCK_SIZE,),
            (_pointer(fields), _pointer(c_arr), np.int64(V), _pointer(minmax)),
        )
    minmax_np = minmax.cpu().numpy()
    counts, sizes, sym_min, alphabets = _counts_from_minmax(
        fields, c_arr, minmax, minmax_np, V, device, device_index, prob_bits
    )
    return _QuantizeResult(
        counts=counts, sizes=sizes, sym_min=sym_min, alphabets=alphabets, fields=fields, c_arr=c_arr, minmax=minmax,
    )


def _counts_from_minmax(fields, c_arr, minmax, minmax_np, V, device, device_index, prob_bits):
    alphabets = [int(minmax_np[_MM_CMAX]) + 1]
    sym_min = [0]
    for stream in range(NUM_COORD_STREAMS):
        lo, hi = int(minmax_np[stream * 2]), int(minmax_np[stream * 2 + 1])
        live = lo != _INT_MAX
        alphabets.append(hi - lo + 1 if live else 0)
        sym_min.append(lo if live else 0)

    bin_off = np.zeros(NUM_STREAMS_FULL, dtype=np.int32)
    cursor = 2
    for stream in range(1, NUM_STREAMS_FULL):
        bin_off[stream] = cursor
        cursor += alphabets[stream]
    total_bins = cursor

    bins = torch.zeros(total_bins, dtype=torch.int32, device=device)
    bin_off_gpu = torch.from_numpy(bin_off).to(device)
    caps = _device_caps.caps(device_index)
    staging_bins = (caps.shared_per_block - SHARED_STAGING_HEADROOM) // 4
    shared_bins = total_bins if total_bins <= staging_bins else 0
    torch_stream = torch.cuda.current_stream(device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, prob_bits, "e8_histogram_kernel")(
            (_grid(caps, V),),
            (BLOCK_SIZE,),
            (
                _pointer(fields), _pointer(c_arr), _pointer(minmax), _pointer(bin_off_gpu), _pointer(bins), np.int64(V),
                np.int32(shared_bins),
            ),
            shared_mem=shared_bins * 4,
        )
    bins_np = bins.cpu().numpy().astype(np.int64)
    n0, n1 = int(bins_np[0]), int(bins_np[1])
    sizes = [V] + [n0 if stream % 2 == 0 else n1 for stream in range(NUM_COORD_STREAMS)]
    counts = [bins_np[: alphabets[0]].copy()]
    for stream in range(1, NUM_STREAMS_FULL):
        offset = int(bin_off[stream])
        counts.append(bins_np[offset : offset + alphabets[stream]].copy())
    return counts, sizes, sym_min, alphabets


def _analytic_total_bytes(counts, sizes, rows, prob_bits, tile_elements):
    return rans.coded_bytes(counts, sizes, prob_bits, tile_elements) + rows * 4 + MODEL_LAYOUT_BYTES


def _optimize_rows(
    request: _QuantizeRequest, baseline: _QuantizeResult, iterations: int, candidate_count: int, tile_elements: int,
) -> _QuantizeResult:
    rows, device, prob_bits = request.rows, request.device, request.prob_bits
    ratios = rdo.ratio_ladder(candidate_count)
    baseline_index = ratios.index(1.0)
    candidates = []
    for index, ratio in enumerate(ratios):
        if index == baseline_index:
            candidate = baseline
        else:
            scaled = replace(request, scale=request.scale * ratio)
            summary = _quantize_pass(scaled)
            scales, row_sse = _refit_row_scales(scaled, summary)
            candidate = replace(summary, scales=scales, row_sse=row_sse)
        candidates.append(rdo.compact_candidate(candidate))

    summary, fitted_scales = rdo.optimize_rows(
        candidates, rows=rows, cols=request.cols, baseline_index=baseline_index, iterations=iterations, device=device,
        summarize=lambda fields, c_arr: _summarize_fields(request, fields, c_arr),
        total_bytes=lambda counts, sizes: _analytic_total_bytes(counts, sizes, rows, prob_bits, tile_elements),
    )
    return replace(summary, scales=fitted_scales)


def _bisect_scale_cuda(work, rms, prob_bits, target_bpp, tile_elements, n_iter=34, max_vectors=262144):
    rows, cols = work.shape
    device = work.device
    vecs_per_row = cols // LATTICE_DIM
    sub_rows = max(1, min(rows, max_vectors // vecs_per_row))
    index = subsample_index(rows, cols, sub_rows, device)
    w_sub = work if index is None else work.index_select(0, index)
    rms_sub = rms if index is None else rms.index_select(0, index)
    N_sub = sub_rows * cols

    lo, hi = 0.001, 8.0
    for _ in range(n_iter):
        mid = (lo * hi) ** 0.5
        summary = _quantize_pass(_QuantizeRequest.of(w_sub, rms_sub, mid, prob_bits))
        total = _analytic_total_bytes(summary.counts, summary.sizes, sub_rows, prob_bits, tile_elements)
        if BITS_PER_BYTE * total / N_sub > target_bpp:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


@dataclass(frozen=True)
class _EncodeOptions:
    target_bpp: float
    prob_bits: int
    auto_prob_bits: bool
    tile_elements: int
    row_rdo_iterations: int
    row_rdo_candidates: int
    scale_search_iterations: int
    scale_search_max_vectors: int
    table_size: int


def _resolve_options(
    target_bpp, prob_bits, tile_elements, row_rdo_iterations, row_rdo_candidates, scale_search_iterations,
    scale_search_max_vectors,
) -> _EncodeOptions:
    resolved_prob_bits, auto_prob_bits = resolve_prob_bits(prob_bits, target_bpp)
    return _EncodeOptions(
        target_bpp=float(target_bpp),
        prob_bits=resolved_prob_bits,
        auto_prob_bits=auto_prob_bits,
        tile_elements=int(tile_elements),
        row_rdo_iterations=int(row_rdo_iterations),
        row_rdo_candidates=int(row_rdo_candidates),
        scale_search_iterations=int(scale_search_iterations),
        scale_search_max_vectors=int(scale_search_max_vectors),
        table_size=1 << resolved_prob_bits,
    )


def _resolve_scale(work, rms, options: _EncodeOptions):
    """The table never shrinks below its starting precision: a coarser grid normalizes the distributions less exactly, so
    shrinking costs rate, while the decode it would buy is not certain -- the decode is priced per tile and per symbol, and
    the table size only decides which representation still stages in shared memory.
    """
    prob_bits = options.prob_bits
    if float(work.abs().amax().item()) == 0.0:
        request = _QuantizeRequest.of(work, rms, 1.0, prob_bits)
        summary = _quantize_pass(request)
        scales, row_sse = _refit_row_scales(request, summary)
        return options, replace(summary, scales=scales, row_sse=row_sse)
    clamped = False
    while True:
        escalate = False
        scale = _bisect_scale_cuda(
            work, rms, prob_bits, options.target_bpp, options.tile_elements, n_iter=options.scale_search_iterations,
            max_vectors=options.scale_search_max_vectors,
        )
        while True:
            request = _QuantizeRequest.of(work, rms, scale, prob_bits)
            summary = _quantize_pass(request)
            alpha = max(summary.alphabets) if summary.alphabets else 0
            if alpha <= options.table_size:
                scales, row_sse = _refit_row_scales(request, summary)
                summary = replace(summary, scales=scales, row_sse=row_sse)
                if options.row_rdo_iterations > 0:
                    summary = _optimize_rows(request, summary, options.row_rdo_iterations,
                                             options.row_rdo_candidates, options.tile_elements)
                    # RDO can widen the final alphabet beyond the baseline's capacity.
                    alpha = max(summary.alphabets) if summary.alphabets else 0
            if alpha <= options.table_size:
                break
            if options.auto_prob_bits and prob_bits < 15:
                prob_bits += 1
                options = replace(options, prob_bits=prob_bits, table_size=1 << prob_bits)
                escalate = True
                break
            clamped = True
            scale *= alpha / options.table_size * ALPHABET_COARSEN_MARGIN
        if not escalate:
            if clamped:
                report_alphabet_clamp(scale, alpha, options.table_size)
            return options, summary


def _build_codec_tables(summary: _QuantizeResult, options: _EncodeOptions, device):
    """Only alphabet-sized frequencies are stored, never a full ``table_size`` LUT; the decoder rebuilds its slot->symbol
    table from them, so a per-layer table stays small.
    """
    n_streams = NUM_STREAMS_FULL
    table_size = options.table_size
    freq_parts = []
    cdf_parts = []
    freq_cursor = 0
    meta = np.zeros((n_streams, STREAM_META_WIDTH), dtype=np.int64)
    for table in range(n_streams):
        n_s = int(summary.sizes[table])
        alphabet = int(summary.alphabets[table]) if n_s else 0
        if alphabet > table_size:
            raise ValueError(
                f"E8 coordinate alphabet {alphabet} exceeds rANS table_size {table_size}; "
                "raise prob_bits or coarsen the lattice scale"
            )
        if alphabet:
            frequency = rans.normalize_freq(summary.counts[table], table_size).astype(np.uint16)
            cdf = np.zeros(alphabet, dtype=np.uint16)
            if alphabet > 1:
                cdf[1:] = np.cumsum(frequency.astype(np.int64))[:-1].astype(np.uint16)
            freq_parts.append(frequency)
            cdf_parts.append(cdf)
        meta[table, META_N_SYMBOLS] = n_s
        meta[table, META_SYM_MIN] = summary.sym_min[table]
        meta[table, META_FREQ_OFFSET] = freq_cursor
        meta[table, META_ALPHABET] = alphabet
        freq_cursor += alphabet

    freq_tables_np = np.concatenate(freq_parts) if freq_parts else np.empty(0, dtype=np.uint16)
    cdfs_np = np.concatenate(cdf_parts) if cdf_parts else np.empty(0, dtype=np.uint16)
    return (
        torch.from_numpy(freq_tables_np).to(device), torch.from_numpy(cdfs_np).to(device),
        torch.from_numpy(meta.astype(np.int32)).to(device),
    )


def _encode_streams(
    summary: _QuantizeResult, freq_tables, cdfs_gpu, stream_meta, total_vectors: int, options: _EncodeOptions, device,
    device_index: int,
):
    tile_vectors = vector_tile_elements(options.tile_elements)
    total_tiles = max(1, (total_vectors + tile_vectors - 1) // tile_vectors)
    states = torch.empty((total_tiles, NUM_STATES), dtype=torch.uint32, device=device)
    word_counts = torch.empty(total_tiles, dtype=torch.uint32, device=device)
    scratch_stride = tile_vectors * 9
    scratch = torch.empty(total_tiles * scratch_stride, dtype=torch.uint16, device=device)
    warps_per_block = BLOCK_SIZE // _device_caps.caps(device_index).warp_size
    blocks_x = max(1, -(-total_tiles // warps_per_block))
    torch_stream = torch.cuda.current_stream(device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, options.prob_bits, "e8_rans_encode_vector_kernel")(
            (blocks_x,),
            (BLOCK_SIZE,),
            (
                _pointer(summary.fields), _pointer(summary.c_arr), _pointer(freq_tables), _pointer(cdfs_gpu),
                _pointer(stream_meta), _pointer(scratch), _pointer(word_counts), _pointer(states), np.int64(total_vectors),
                np.int32(tile_vectors), np.int32(total_tiles),
            ),
        )
        counts_cp = cupy.from_dlpack(word_counts)
        offsets64 = torch.empty(total_tiles + 1, dtype=torch.int64, device=device)
        offsets_cp = cupy.from_dlpack(offsets64)
        offsets_cp[0] = 0
        cupy.cumsum(counts_cp, dtype=cupy.int64, out=offsets_cp[1:])
        total_words = int(offsets64[-1].item())
    if total_words >= 1 << 32:
        raise ValueError("lattice_rans payload exceeds uint32 offset capacity")
    offsets = offsets64.to(torch.uint32)
    payload = torch.empty(total_words, dtype=torch.uint16, device=device)
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        _kernel(device_index, options.prob_bits, "e8_compact_kernel")(
            (total_tiles,), (BLOCK_SIZE,),
            (_pointer(scratch), _pointer(offsets), _pointer(payload), np.int32(scratch_stride), np.int32(total_tiles)),
        )
    return payload, offsets, states


def encode(
    weight, *, target_bpp, prob_bits, tile_elements, row_rdo_iterations, row_rdo_candidates, scale_search_iterations,
    scale_search_max_vectors,
):
    options = _resolve_options(target_bpp, prob_bits, tile_elements, row_rdo_iterations,
                               row_rdo_candidates, scale_search_iterations, scale_search_max_vectors)

    weight = weight.contiguous()
    device = weight.device
    device_index = _device_index(weight)
    rows, cols = weight.shape
    V = rows * (cols // LATTICE_DIM)

    if weight.dtype == torch.bfloat16:
        work = weight
        xf = weight.float()
    else:
        work = weight.float().contiguous()
        xf = work
    rms = xf.square().mean(dim=1, keepdim=True).sqrt().clamp_min(_MIN_RMS)
    check_row_scales_finite(rms, weight.dtype)
    # For a bf16 container this fp32 view only served the row RMS above. Releasing it before the scale search keeps two
    # full-tensor copies from being alive at once.
    del xf

    options, summary = _resolve_scale(work, rms, options)
    freq_tables, cdfs_gpu, stream_meta = _build_codec_tables(summary, options, device)
    payload, offsets, states = _encode_streams(
        summary, freq_tables, cdfs_gpu, stream_meta, V, options, device, device_index)
    fitted_scales = summary.scales
    del summary

    layout = make_layout(cols, options.prob_bits, options.tile_elements).to(device)
    return LatticeBuffers(
        payload=payload, offsets=offsets, states=states, stream_meta=stream_meta, freq_tables=freq_tables,
        scales=fitted_scales.contiguous(), layout=layout,
    )


def _shared_lut_usable(shared_bytes: int | None, caps, threads: int) -> bool:
    """A table that consumes most of an SM's shared memory leaves one CTA resident, and this decode hides renormalization
    latency with warp count, so it would run slower reading the table from shared memory than from global. The staged table
    has to leave room for ``MIN_SHARED_BLOCKS_PER_SM`` resident blocks.
    """
    if shared_bytes is None:
        return False
    staged = shared_bytes + STATIC_SHARED
    if staged > caps.shared_limit(STATIC_SHARED):
        return False
    return caps.blocks_per_sm(threads, staged) >= MIN_SHARED_BLOCKS_PER_SM


def _stream_slices(meta_np) -> list[tuple[int, int, int]]:
    return [
        (stream, int(row[META_FREQ_OFFSET]), int(row[META_ALPHABET])) for stream, row in enumerate(meta_np)
        if int(row[META_ALPHABET])
    ]


def _cdf_and_frequency(freq_np, offset: int, alphabet: int):
    frequency = freq_np[offset : offset + alphabet].astype(np.int64)
    cdf = np.zeros(alphabet, dtype=np.int64)
    cdf[1:] = np.cumsum(frequency)[:-1]
    return cdf, frequency


def _pack_bits(freq_np, slices, table_size: int, n_streams: int) -> np.ndarray | None:
    pack_bits = np.zeros(n_streams, dtype=np.int32)
    for stream, offset, alphabet in slices:
        _cdf, frequency = _cdf_and_frequency(freq_np, offset, alphabet)
        stored = np.where(frequency == table_size, 0, frequency)
        symbol_bits = max(1, (alphabet - 1).bit_length())
        frequency_bits = max(1, int(stored.max()).bit_length())
        delta_bits = max(1, (int(frequency.max()) - 1).bit_length())
        if symbol_bits + frequency_bits + delta_bits > 32:
            return None
        pack_bits[stream] = symbol_bits | (frequency_bits << 8)
    return pack_bits


def _slot_fields(freq_np, offset: int, alphabet: int, table_size: int):
    cdf, frequency = _cdf_and_frequency(freq_np, offset, alphabet)
    stored = np.where(frequency == table_size, 0, frequency)
    return (np.repeat(np.arange(alphabet), frequency), np.repeat(stored, frequency), np.repeat(cdf, frequency))


def _shared_tables(freq_np, slices, table_size: int, n_streams: int):
    symbols = np.zeros(n_streams * table_size, dtype=np.uint8)
    begin_frequency = np.zeros(int(freq_np.size), dtype=np.uint32)
    for stream, offset, alphabet in slices:
        cdf, frequency = _cdf_and_frequency(freq_np, offset, alphabet)
        begin_frequency[offset : offset + alphabet] = (cdf | (frequency << 16)).astype(np.uint32)
        base = stream * table_size
        symbols[base : base + table_size] = np.repeat(np.arange(alphabet, dtype=np.uint8), frequency)
    return symbols, begin_frequency


def _packed_tables(freq_np, slices, table_size: int, n_streams: int, pack_bits):
    packed = np.zeros((n_streams, table_size), dtype=np.uint32)
    for stream, offset, alphabet in slices:
        symbol, frequency, begin = _slot_fields(freq_np, offset, alphabet, table_size)
        symbol_bits = int(pack_bits[stream] & 0xFF)
        frequency_bits = int((pack_bits[stream] >> 8) & 0xFF)
        packed[stream] = (
            symbol.astype(np.uint32) | (frequency.astype(np.uint32) << np.uint32(symbol_bits))
            | ((np.arange(table_size, dtype=np.int64) - begin).astype(np.uint32) << np.uint32(symbol_bits + frequency_bits))
        )
    return packed


def _wide_tables(freq_np, slices, table_size: int, n_streams: int):
    luts = np.zeros((n_streams, table_size), dtype=np.uint64)
    for stream, offset, alphabet in slices:
        symbol, frequency, begin = _slot_fields(freq_np, offset, alphabet, table_size)
        luts[stream] = (
            (begin.astype(np.uint64) << np.uint64(32)) | (frequency.astype(np.uint64) << np.uint64(16))
            | symbol.astype(np.uint64)
        )
    return luts


@dataclass(frozen=True)
class _DecodePlan:
    representation: str
    coset_frequency0: int
    error: torch.Tensor
    vector_tiles: int
    tile_vectors: int
    sym_u8: torch.Tensor | None = None
    fb_lut: torch.Tensor | None = None
    shared_bytes: int = 0
    packed_luts: torch.Tensor | None = None
    pack_bits: torch.Tensor | None = None
    decode_luts: torch.Tensor | None = None


def _decode_plan(layout, stream_meta, freq_tables, info, caps, threads: int) -> _DecodePlan:
    device = freq_tables.device
    fingerprint = (stream_meta.data_ptr(), freq_tables.data_ptr(), device, threads)
    cached = getattr(layout, "_lattice_rans_decode_plan", None)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    meta_np = stream_meta.detach().cpu().numpy().astype(np.int64)
    freq_np = freq_tables.detach().cpu().numpy().astype(np.uint16)
    prob_bits = info["prob_bits"]
    table_size = 1 << prob_bits
    n_streams = info["n_streams"]
    slices = _stream_slices(meta_np)
    alphabets = meta_np[:, META_ALPHABET]
    max_alpha = int(alphabets.max()) if alphabets.size else 0
    alphabet_sum = int(alphabets.sum())

    shared_bytes = n_streams * table_size + alphabet_sum * 4 if max_alpha < 256 else None
    fields: dict = {}
    if _shared_lut_usable(shared_bytes, caps, threads):
        representation = "shared"
        symbols, begin_frequency = _shared_tables(freq_np, slices, table_size, n_streams)
        fields = {
            "sym_u8": torch.from_numpy(symbols).to(device), "fb_lut": torch.from_numpy(begin_frequency).to(device),
            "shared_bytes": int(symbols.nbytes + begin_frequency.nbytes),
        }
    else:
        pack_bits = _pack_bits(freq_np, slices, table_size, n_streams)
        if pack_bits is not None:
            representation = "packed"
            packed = _packed_tables(freq_np, slices, table_size, n_streams, pack_bits)
            fields = {"packed_luts": torch.from_numpy(packed).to(device), "pack_bits": torch.from_numpy(pack_bits).to(device)}
        else:
            representation = "wide"
            luts = _wide_tables(freq_np, slices, table_size, n_streams)
            fields = {"decode_luts": torch.from_numpy(luts).to(device)}

    vectors = info["rows"] * (info["cols"] // LATTICE_DIM)
    tile_vectors = vector_tile_elements(info["tile_elements"])
    plan = _DecodePlan(
        representation=representation, coset_frequency0=int(freq_np[0]), error=torch.zeros(1, dtype=torch.int32, device=device),
        vector_tiles=max(1, (vectors + tile_vectors - 1) // tile_vectors), tile_vectors=tile_vectors, **fields,
    )
    layout._lattice_rans_decode_plan = (fingerprint, plan)
    return plan


def _decode_launch(plan, buffers: LatticeBuffers, output, rows, cols, l2_prefetch):
    vecs_per_row = cols // LATTICE_DIM
    total_vectors = rows * vecs_per_row
    native_bf16 = output.dtype == torch.bfloat16
    store_kind = 0 if native_bf16 else _STORE_KINDS[output.dtype]
    if plan.representation == "shared":
        kernel_name = "e8_decode_vector_shlut8pf_kernel"
        shared = plan.shared_bytes
        args = (
            _pointer(buffers.payload), _pointer(buffers.offsets), _pointer(buffers.states), _pointer(plan.sym_u8),
            _pointer(plan.fb_lut), np.uint32(plan.coset_frequency0), _pointer(buffers.stream_meta),
            _pointer(buffers.scales), _pointer(output), np.int32(vecs_per_row), np.int64(total_vectors),
            np.int32(plan.tile_vectors), np.int32(plan.vector_tiles), np.int32(plan.fb_lut.numel()), _pointer(plan.error),
        )
    elif plan.representation == "packed":
        kernel_name = "e8_decode_vector_packed32pf_kernel" if l2_prefetch else "e8_decode_vector_packed32_kernel"
        shared = 0
        args = (
            _pointer(buffers.payload), _pointer(buffers.offsets), _pointer(buffers.states), _pointer(plan.packed_luts),
            _pointer(plan.pack_bits), np.uint32(plan.coset_frequency0), _pointer(buffers.stream_meta),
            _pointer(buffers.scales), _pointer(output), np.int32(vecs_per_row), np.int64(total_vectors),
            np.int32(plan.tile_vectors), np.int32(plan.vector_tiles), _pointer(plan.error),
        )
    else:
        kernel_name = "e8_decode_vector_fused_kernel"
        shared = 0
        args = (
            _pointer(buffers.payload), _pointer(buffers.offsets), _pointer(buffers.states), _pointer(plan.decode_luts),
            _pointer(buffers.stream_meta), _pointer(buffers.scales), _pointer(output), np.int32(vecs_per_row),
            np.int64(total_vectors), np.int32(plan.tile_vectors), np.int32(plan.vector_tiles), _pointer(plan.error),
        )
    if not native_bf16:
        kernel_name = _GENERIC_DECODE_KERNEL[kernel_name]
        args += (np.int32(store_kind),)
    return kernel_name, shared, args


def decode(buffers: LatticeBuffers, *, shape, dtype, threads_per_block, l2_prefetch):
    layout = buffers.layout
    info = decode_geometry(buffers._asdict(), shape)
    rows, cols = info["rows"], info["cols"]
    prob_bits = info["prob_bits"]

    payload = buffers.payload
    device_index = _device_index(payload)
    caps = _device_caps.caps(device_index)
    threads = _device_caps.resolve_threads(caps, threads_per_block, BLOCK_SIZE)
    plan = _decode_plan(layout, buffers.stream_meta, buffers.freq_tables, info, caps, threads)
    scratch = dtype in _SCRATCH_DTYPES
    out_dtype = torch.float32 if scratch else dtype
    output = torch.empty((rows, cols), dtype=out_dtype, device=payload.device)
    torch_stream = torch.cuda.current_stream(payload.device)

    kernel_name, shared, args = _decode_launch(plan, buffers, output, rows, cols, l2_prefetch)
    launch_blocks = max(1, -(-plan.vector_tiles // (threads // caps.warp_size)))
    with cupy.cuda.Device(device_index), _external_stream(torch_stream):
        kernel = _kernel(device_index, prob_bits, kernel_name)
        _ensure_dynamic_shared(kernel, shared)
        kernel((launch_blocks,), (threads,), args, shared_mem=shared)
    if scratch:
        output = snap_to_container(output, dtype)
    return output
