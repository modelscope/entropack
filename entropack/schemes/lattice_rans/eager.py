import math
from dataclasses import dataclass

import numpy as np
import torch

from . import rans, rdo
from .format import (
    ALPHABET_COARSEN_MARGIN, BITS_PER_BYTE, LATTICE_DIM,
    META_ALPHABET, META_FREQ_OFFSET, META_N_SYMBOLS, META_SYM_MIN, MODEL_LAYOUT_BYTES, MODEL_STREAM_META_BYTES,
    NUM_COORD_FIELDS, STREAM_META_WIDTH, LatticeBuffers, check_row_scales_finite, decode_geometry, make_layout,
    report_alphabet_clamp, resolve_prob_bits, snap_to_container, subsample_index, vector_tile_elements,
)

MIN_RMS = 1.0e-12


def _nearest_dn(Y: torch.Tensor) -> torch.Tensor:
    f = torch.round(Y)
    resid = Y - f
    par = torch.remainder(f.sum(1), 2.0)
    idx = resid.abs().argmax(1, keepdim=True)
    sgn = torch.sign(resid.gather(1, idx))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    onehot = torch.zeros_like(f)
    onehot.scatter_(1, idx, 1.0)
    return f + onehot * sgn * (par != 0).float().unsqueeze(1)


def nearest_e8(Y: torch.Tensor) -> torch.Tensor:
    """The coset comparison runs in float64 so exact ties, common on bf16 and half-integer grids, resolve as the CUDA kernel
    does rather than following fp32 summation order.
    """
    c0 = _nearest_dn(Y)
    c1 = _nearest_dn(Y - 0.5) + 0.5
    yd = Y.to(torch.float64)
    d0 = (yd - c0.to(torch.float64)).square().sum(1, keepdim=True)
    d1 = (yd - c1.to(torch.float64)).square().sum(1, keepdim=True)
    # ``+ 0.0`` folds -0.0 to +0.0: torch.round of a small negative coordinate is -0.0, and the CUDA kernel rounds in integer
    # coordinates so it can never emit one.
    return torch.where(d0 <= d1, c0, c1) + 0.0


def point_to_fields(p: torch.Tensor):
    doubled = torch.round(2 * p)
    c = (doubled.remainder(2.0) != 0).any(dim=1).to(torch.int64)
    z = torch.round(p - 0.5 * c.unsqueeze(1)).to(torch.int64)
    z0_6 = z[:, :7]
    par = z0_6.sum(1).remainder(2)
    m = torch.div(z[:, 7] - par, 2, rounding_mode="floor")
    return c, z0_6, m


def fields_to_point(c: torch.Tensor, z0_6: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    par = z0_6.sum(1).remainder(2)
    z7 = 2 * m + par
    z = torch.cat([z0_6, z7.unsqueeze(1)], dim=1)
    return z.to(torch.float32) + 0.5 * c.to(torch.float32).unsqueeze(1)


def _coord_streams(c_np, z0_6_np, m_np):
    idx = [np.flatnonzero(c_np == 0), np.flatnonzero(c_np == 1)]
    fields = [z0_6_np[:, f] for f in range(NUM_COORD_FIELDS - 1)] + [m_np]
    streams = [(c_np.astype(np.int64), 0)]
    for f in range(NUM_COORD_FIELDS):
        for k in (0, 1):
            vals = fields[f][idx[k]].astype(np.int64)
            if vals.size == 0:
                streams.append((vals, 0))
            else:
                mn = int(vals.min())
                streams.append((vals - mn, mn))
    return streams


def _field_matrix(z0_6_np, m_np):
    return np.column_stack([z0_6_np[:, field] for field in range(NUM_COORD_FIELDS - 1)] + [m_np])


def _stream_stats(streams):
    counts, sizes, sym_min, alphabets = [], [], [], []
    for symbols, minimum in streams:
        alphabet = int(symbols.max()) + 1 if symbols.size else 0
        counts.append(np.bincount(symbols, minlength=alphabet).astype(np.int64))
        sizes.append(int(symbols.size))
        sym_min.append(int(minimum))
        alphabets.append(alphabet)
    return counts, sizes, sym_min, alphabets


def _rms_and_X(x: torch.Tensor):
    rms = x.square().mean(dim=1, keepdim=True).sqrt().clamp_min(MIN_RMS)
    X = (x / rms).reshape(-1, LATTICE_DIM).contiguous()
    return rms, X


def _quantize_at(X, scale):
    c, z0_6, m = point_to_fields(nearest_e8(X / scale))
    return c.cpu().numpy(), z0_6.cpu().numpy(), m.cpu().numpy()


def _analytic_bytes_from_counts(counts, sizes, rows, prob_bits, tile_elements):
    """The cuda model leaves the stream metadata and the fixed slack out, which moves the scale the bisection settles on, so the
    two byte models cannot be unified.
    """
    coded = rans.coded_bytes(counts, sizes, prob_bits, tile_elements)
    return coded + rows * 4 + MODEL_LAYOUT_BYTES + MODEL_STREAM_META_BYTES + 384


def _analytic_total_bytes(c_np, z0_6_np, m_np, rows, prob_bits, tile_elements):
    counts, sizes, _sym_min, _alphabets = _stream_stats(_coord_streams(c_np, z0_6_np, m_np))
    return _analytic_bytes_from_counts(counts, sizes, rows, prob_bits, tile_elements)


def _bisect_scale(X, rows, cols, target_bpp, prob_bits, tile_elements, n_iter=34, max_vectors=262144):
    vecs_per_row = cols // LATTICE_DIM
    sub_rows = max(1, min(rows, max_vectors // vecs_per_row))
    index = subsample_index(rows, cols, sub_rows, X.device)
    X_sub = X if index is None else X.reshape(rows, cols).index_select(0, index).reshape(-1, LATTICE_DIM)
    N_sub = sub_rows * cols

    def total_at(s: float) -> float:
        p = nearest_e8(X_sub / s)
        c, z0_6, m = point_to_fields(p)
        return _analytic_total_bytes(
            c.cpu().numpy(), z0_6.cpu().numpy(), m.cpu().numpy(), sub_rows, prob_bits, tile_elements
        )

    lo, hi = 0.001, 8.0
    for _ in range(n_iter):
        mid = math.sqrt(lo * hi)
        if BITS_PER_BYTE * total_at(mid) / N_sub > target_bpp:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _max_stream_alpha(quantized) -> int:
    alpha = 0
    for symbols, _minimum in _coord_streams(*quantized):
        if symbols.size:
            alpha = max(alpha, int(symbols.max()) + 1)
    return alpha


def _resolve_scale(x, X, rows, cols, target_bpp, prob_bits, tile_elements, iterations, max_vectors):
    """Auto prob_bits grows a bit at a time, only when the alphabet the chosen scale produces overflows the table, up to the
    ceiling; that growth is what lets the real rate track ``target_bpp`` at the high end. It never shrinks below its start:
    a coarser grid normalizes the distributions less exactly, so shrinking costs rate.

    Past the ceiling, growing further is self-defeating: a wider table prices the same scale cheaper, so the bisection
    refines the scale and widens the alphabet again. The scale is coarsened by the actual overflow instead, and only the
    alphabet is re-measured; re-bisecting would undo it.
    """
    prob_bits, auto_prob_bits = resolve_prob_bits(prob_bits, target_bpp)
    if float(x.abs().amax().item()) == 0.0:
        return 1.0, _quantize_at(X, 1.0), prob_bits, True

    table_size = 1 << prob_bits
    clamped = False
    while True:
        escalate = False
        scale = _bisect_scale(
            X, rows, cols, target_bpp, prob_bits, tile_elements, n_iter=iterations, max_vectors=max_vectors
        )
        while True:
            quantized = _quantize_at(X, scale)
            alpha = _max_stream_alpha(quantized)
            if alpha <= table_size:
                break
            if auto_prob_bits and prob_bits < 15:
                prob_bits += 1
                table_size = 1 << prob_bits
                escalate = True
                break
            clamped = True
            scale *= alpha / table_size * ALPHABET_COARSEN_MARGIN
        if not escalate:
            if clamped:
                report_alphabet_clamp(scale, alpha, table_size)
            return scale, quantized, prob_bits, False


@dataclass(frozen=True)
class _Candidate:
    counts: list
    sizes: list
    sym_min: list
    alphabets: list
    fields: torch.Tensor
    c_arr: torch.Tensor
    scales: torch.Tensor | None = None
    row_sse: torch.Tensor | None = None


def _refit_row_scales(x, rms, quantized, scale, device):
    """Fitting the scale to the points the quantizer actually chose lowers distortion at no rate cost, so it runs on every
    encode, not only under the row-RDO pass.
    """
    rows, cols = x.shape
    c_np, z0_6_np, m_np = quantized
    c = torch.from_numpy(c_np).to(device=device, dtype=torch.int64)
    z0_6 = torch.from_numpy(z0_6_np).to(device=device, dtype=torch.int64)
    m = torch.from_numpy(m_np).to(device=device, dtype=torch.int64)
    levels = fields_to_point(c, z0_6, m).reshape(rows, cols)
    numerator = (x * levels).sum(1)
    denominator = levels.square().sum(1)
    scales = torch.where(denominator > 0, (numerator / denominator).clamp_min(MIN_RMS), scale * rms.squeeze(1))
    energy = x.square().sum(1)
    row_sse = torch.where(denominator > 0, (energy - numerator * numerator / denominator).clamp_min(0.0), energy)
    return scales, row_sse


def _candidate_of(quantized, scales, row_sse, device):
    c_np, z_np, m_np = quantized
    counts, sizes, sym_min, alphabets = _stream_stats(_coord_streams(c_np, z_np, m_np))
    return _Candidate(
        counts, sizes, sym_min, alphabets, torch.from_numpy(_field_matrix(z_np, m_np).reshape(-1)).to(device),
        torch.from_numpy(c_np).to(device), scales, row_sse,
    )


def _candidate_at(x, rms, X, scale, device):
    quantized = _quantize_at(X, scale)
    scales, row_sse = _refit_row_scales(x, rms, quantized, scale, device)
    return _candidate_of(quantized, scales, row_sse, device)


def _summarize_fields(fields, c_arr):
    field_matrix = fields.cpu().numpy().reshape(-1, LATTICE_DIM)
    c_np = c_arr.cpu().numpy()
    counts, sizes, sym_min, alphabets = _stream_stats(
        _coord_streams(c_np, field_matrix[:, : NUM_COORD_FIELDS - 1], field_matrix[:, NUM_COORD_FIELDS - 1])
    )
    return _Candidate(counts, sizes, sym_min, alphabets, fields, c_arr)


def _optimize_rows(x, rms, X, baseline, scale, prob_bits, tile_elements, iterations, candidate_count):
    rows, cols = x.shape
    device = x.device
    ratios = rdo.ratio_ladder(candidate_count)
    baseline_index = ratios.index(1.0)
    candidates = []
    for index, ratio in enumerate(ratios):
        candidate = baseline if index == baseline_index else _candidate_at(x, rms, X, scale * ratio, device)
        candidates.append(rdo.compact_candidate(candidate))
    summary, scales = rdo.optimize_rows(
        candidates, rows=rows, cols=cols, baseline_index=baseline_index, iterations=iterations, device=device,
        summarize=_summarize_fields,
        total_bytes=lambda counts, sizes: _analytic_bytes_from_counts(counts, sizes, rows, prob_bits, tile_elements),
    )
    chosen = summary.fields.cpu().numpy().reshape(-1, LATTICE_DIM)
    return summary.c_arr.cpu().numpy(), chosen[:, : NUM_COORD_FIELDS - 1], chosen[:, NUM_COORD_FIELDS - 1], scales


def _encode_streams(c_np, z_np, m_np, prob_bits, tile_elements):
    streams = _coord_streams(c_np, z_np, m_np)
    n_streams = len(streams)

    table_size = 1 << prob_bits
    meta = np.zeros((n_streams, STREAM_META_WIDTH), dtype=np.int64)
    frequencies = []
    freq_parts = []
    freq_cursor = 0
    for table, (symbols, sym_min) in enumerate(streams):
        n = symbols.size
        if n == 0:
            frequency = np.empty(0, dtype=np.uint16)
            alphabet = 0
        else:
            alphabet = int(symbols.max()) + 1
            counts = np.bincount(symbols, minlength=alphabet).astype(np.int64)
            frequency = rans.normalize_freq(counts, table_size).astype(np.uint16)
        meta[table, META_N_SYMBOLS] = n
        meta[table, META_SYM_MIN] = sym_min
        meta[table, META_FREQ_OFFSET] = freq_cursor
        meta[table, META_ALPHABET] = alphabet
        frequencies.append(frequency)
        if alphabet:
            freq_parts.append(frequency)
            freq_cursor += alphabet

    field_symbols = _field_matrix(z_np, m_np).astype(np.int64)
    for field in range(NUM_COORD_FIELDS):
        table0 = 1 + 2 * field
        table1 = table0 + 1
        field_symbols[c_np == 0, field] -= int(meta[table0, META_SYM_MIN])
        field_symbols[c_np == 1, field] -= int(meta[table1, META_SYM_MIN])
    tile_vectors = vector_tile_elements(tile_elements)
    payload, states, offsets = rans.encode_vector_stream(c_np, field_symbols, frequencies, prob_bits, tile_vectors)
    freq_tables = np.concatenate(freq_parts) if freq_parts else np.empty(0, dtype=np.uint16)
    return payload, states, offsets, meta, freq_tables


def _encode_impl(
    weight, target_bpp, prob_bits, tile_elements, row_rdo_iterations, row_rdo_candidates, scale_search_iterations,
    scale_search_max_vectors,
):
    if weight.shape[1] % LATTICE_DIM != 0:
        raise ValueError(
            "lattice_rans quantizes whole E8 vectors, so the column count must be a multiple of "
            f"{LATTICE_DIM}; got cols={weight.shape[1]}"
        )

    device = weight.device
    x = weight.float().contiguous()
    rows, cols = x.shape

    rms, X = _rms_and_X(x)
    check_row_scales_finite(rms, weight.dtype)

    s, quantized, prob_bits, all_zero = _resolve_scale(
        x, X, rows, cols, target_bpp, prob_bits, tile_elements, scale_search_iterations, scale_search_max_vectors,
    )
    c_np, z_np, m_np = quantized

    scales, row_sse = _refit_row_scales(x, rms, quantized, s, device)
    if row_rdo_iterations > 0 and not all_zero:
        baseline = _candidate_of(quantized, scales, row_sse, device)
        c_np, z_np, m_np, scales = _optimize_rows(
            x, rms, X, baseline, s, prob_bits, tile_elements, row_rdo_iterations, row_rdo_candidates
        )

    payload, states, offsets, meta, freq_tables = _encode_streams(c_np, z_np, m_np, prob_bits, tile_elements)

    layout = make_layout(cols, prob_bits, tile_elements)
    return LatticeBuffers(
        payload=torch.from_numpy(payload.astype(np.uint16)).to(device),
        offsets=torch.from_numpy(offsets.astype(np.uint32)).to(device),
        states=torch.from_numpy(states.astype(np.uint32)).to(device),
        stream_meta=torch.from_numpy(meta.astype(np.int32)).to(device),
        freq_tables=torch.from_numpy(freq_tables.astype(np.uint16)).to(device), scales=scales.contiguous(),
        layout=layout.to(device),
    )


def _host_views(buffers: LatticeBuffers):
    return (
        buffers.payload.detach().cpu().numpy().astype(np.uint16),
        buffers.offsets.detach().cpu().numpy().astype(np.int64),
        buffers.states.detach().cpu().numpy().astype(np.uint32),
        buffers.stream_meta.detach().cpu().numpy().astype(np.int64),
        buffers.freq_tables.detach().cpu().numpy().astype(np.uint16),
        buffers.scales.detach().cpu(),
    )


def _stream_frequencies(freq_tables: np.ndarray, meta: np.ndarray) -> list:
    return [
        freq_tables[int(row[META_FREQ_OFFSET]) : int(row[META_FREQ_OFFSET] + row[META_ALPHABET])] for row in meta
    ]


def _restore_field_minima(c_np: np.ndarray, shifted_fields: np.ndarray, meta: np.ndarray) -> np.ndarray:
    field_values = shifted_fields.copy()
    for field in range(NUM_COORD_FIELDS):
        table0 = 1 + 2 * field
        table1 = table0 + 1
        field_values[:, field] += np.where(c_np == 0, int(meta[table0, META_SYM_MIN]), int(meta[table1, META_SYM_MIN]))
    return field_values


def _reconstruct(c_np: np.ndarray, field_values: np.ndarray, scales: torch.Tensor, rows: int, cols: int):
    c = torch.from_numpy(c_np).to(torch.int64)
    z0_6 = torch.from_numpy(field_values[:, :7]).to(torch.int64)
    m = torch.from_numpy(field_values[:, 7]).to(torch.int64)
    points = fields_to_point(c, z0_6, m)
    scale_vec = scales.repeat_interleave(cols // LATTICE_DIM)
    return (points * scale_vec.unsqueeze(1)).reshape(rows, cols)


def _decode_impl(buffers: LatticeBuffers, shape, dtype):
    info = decode_geometry(buffers._asdict(), shape)
    rows, cols = info["rows"], info["cols"]
    prob_bits = info["prob_bits"]
    tile_vectors = vector_tile_elements(info["tile_elements"])
    payload, offsets, states, meta, freq_tables, scales = _host_views(buffers)

    frequencies = _stream_frequencies(freq_tables, meta)
    V = rows * (cols // LATTICE_DIM)
    c_np, shifted_fields = rans.decode_vector_stream(payload, offsets, states, frequencies, prob_bits, tile_vectors, V)

    field_values = _restore_field_minima(c_np, shifted_fields, meta)
    xhat = _reconstruct(c_np, field_values, scales, rows, cols)

    result = snap_to_container(xhat, dtype)
    if buffers.layout.device.type != "cpu":
        result = result.to(buffers.layout.device)
    return result


def encode(
    weight, *, target_bpp, prob_bits, tile_elements, row_rdo_iterations, row_rdo_candidates, scale_search_iterations,
    scale_search_max_vectors,
):
    pb = None if prob_bits in (None, 0) else int(prob_bits)
    return _encode_impl(
        weight, float(target_bpp), pb, int(tile_elements), int(row_rdo_iterations), int(row_rdo_candidates),
        int(scale_search_iterations), int(scale_search_max_vectors),
    )


def decode(buffers: LatticeBuffers, *, shape, dtype, threads_per_block, l2_prefetch):
    return _decode_impl(buffers, shape, dtype)
