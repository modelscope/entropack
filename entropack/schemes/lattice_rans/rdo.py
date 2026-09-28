from dataclasses import replace
from typing import NamedTuple

import numpy as np
import torch

from .format import BITS_PER_BYTE, LATTICE_DIM, NUM_COORD_FIELDS

RATIO_SPAN = (0.70, 1.45)


def ratio_ladder(count):
    """The scale ratios one refinement pass prices: 1.0, then the geometric midpoint of the widest
    log-gap inside RATIO_SPAN, one per further candidate. A K-point ladder is therefore a subset of
    every wider one, so widening can only help a row."""
    pts = [1.0]
    while len(pts) < count:
        bounds = [RATIO_SPAN[0], *pts, RATIO_SPAN[1]]
        i = max(range(1, len(bounds)), key=lambda j: bounds[j] / bounds[j - 1])
        pts.insert(i - 1, (bounds[i] * bounds[i - 1]) ** 0.5)
    return tuple(pts)


class RateTable(NamedTuple):
    """Both index tensors are int32, which keeps the per-iteration index temporaries :func:`row_rates` builds small against the
    table they index.
    """

    values: torch.Tensor
    offsets: torch.Tensor
    minima: torch.Tensor


def stream_ranges(candidates, n_streams):
    minima = [0] * n_streams
    maxima = [0] * n_streams
    for stream in range(n_streams):
        alive = [
            (candidate.sym_min[stream], candidate.sym_min[stream] + candidate.alphabets[stream] - 1) for candidate in candidates
            if candidate.alphabets[stream] > 0
        ]
        if alive:
            minima[stream] = min(item[0] for item in alive)
            maxima[stream] = max(item[1] for item in alive)
    return minima, maxima


def rate_costs(counts, sizes, sym_min, minima, maxima, device, alpha=0.5):
    """The streams are concatenated on the host and uploaded as one table; uploading them separately costs one host-to-device
    copy per stream and dominates this function. The arithmetic itself is small and runs slower on the device than here.
    """
    tables = []
    offsets = [0]
    for stream, count in enumerate(counts):
        width = maxima[stream] - minima[stream] + 1
        expanded = np.zeros(width, dtype=np.float64)
        if count.size:
            offset = sym_min[stream] - minima[stream]
            expanded[offset : offset + count.size] = count
        probability = (expanded + alpha) / (sizes[stream] + alpha * width)
        tables.append(-np.log2(probability).astype(np.float32))
        offsets.append(offsets[-1] + width)
    return RateTable(
        torch.from_numpy(np.concatenate(tables)).to(device), torch.tensor(offsets, dtype=torch.int32, device=device),
        torch.tensor(minima, dtype=torch.int32, device=device),
    )


def row_rates(rows, cols, candidate, table):
    """Gather rather than mask: a boolean mask per stream needs ``nonzero`` to bring the hit count back to the host, and those
    per-stream synchronizations leave small layers host-bound. Transposing the field matrix first turns the coordinate reads
    from strided into contiguous ones, which matters more the less of the matrix the device cache holds. The accumulation
    order is untouched, so the sum is bit-identical.
    """
    vectors_per_row = cols // LATTICE_DIM
    fields = candidate.fields.reshape(-1, LATTICE_DIM).t().contiguous().to(torch.int32)
    coset = candidate.c_arr.to(torch.int32)
    rate = table.values[table.offsets[0] + coset]
    for field in range(NUM_COORD_FIELDS):
        stream = 1 + 2 * field + coset
        rate = rate + table.values[table.offsets[stream] + (fields[field] - table.minima[stream])]
    return rate.reshape(rows, vectors_per_row).sum(1)


def compact_candidate(candidate):
    live = [index for index in range(1, len(candidate.sym_min)) if candidate.alphabets[index] > 0]
    lows = [candidate.sym_min[index] for index in live]
    highs = [candidate.sym_min[index] + candidate.alphabets[index] - 1 for index in live]
    low = min(lows, default=0)
    high = max(highs, default=0)
    if low >= -128 and high <= 127:
        field_dtype = torch.int8
    elif low >= -32768 and high <= 32767:
        field_dtype = torch.int16
    else:
        field_dtype = torch.int32
    return replace(candidate, fields=candidate.fields.to(field_dtype), c_arr=candidate.c_arr.to(torch.uint8))


def select_candidate_rows(rows, cols, candidates, choice):
    vectors_per_row = cols // LATTICE_DIM
    device = candidates[0].fields.device
    fields = torch.empty(candidates[0].fields.numel(), dtype=torch.int32, device=device)
    c_arr = torch.empty(candidates[0].c_arr.numel(), dtype=torch.int32, device=device)
    fields_rows = fields.reshape(rows, vectors_per_row, LATTICE_DIM)
    c_rows = c_arr.reshape(rows, vectors_per_row)
    for index, candidate in enumerate(candidates):
        selected_rows = torch.nonzero(choice == index, as_tuple=False).flatten()
        if selected_rows.numel() == 0:
            continue
        source = candidate.fields.reshape(rows, vectors_per_row, LATTICE_DIM)[selected_rows]
        fields_rows[selected_rows] = source.to(torch.int32)
        c_rows[selected_rows] = candidate.c_arr.reshape(rows, vectors_per_row)[selected_rows].to(torch.int32)
    scales = torch.stack([candidate.scales for candidate in candidates], dim=1)
    row = torch.arange(rows, device=choice.device)
    return fields, c_arr, scales[row, choice]


def choose_rate_tradeoff(distortion, rates, row, desired_rate):
    zero_choice = distortion.argmin(1)
    if rates[row, zero_choice].sum() <= desired_rate:
        return zero_choice
    choice = zero_choice
    low = 0.0
    high = max(float(distortion.mean().item()) * 1.0e-4, 1.0e-12)
    for _ in range(50):
        trial = (distortion + high * rates).argmin(1)
        if rates[row, trial].sum() <= desired_rate:
            break
        high *= 2.0
    for _ in range(20):
        middle = high * 0.5 if low == 0.0 else (low * high) ** 0.5
        trial = (distortion + middle * rates).argmin(1)
        if rates[row, trial].sum() > desired_rate:
            low = middle
        else:
            high = middle
            choice = trial
    return choice


def optimize_rows(candidates, *, rows, cols, baseline_index, iterations, device, summarize, total_bytes):
    minima, maxima = stream_ranges(candidates, len(candidates[0].counts))
    distortion = torch.stack([candidate.row_sse for candidate in candidates], dim=1)
    choice = torch.full((rows,), baseline_index, dtype=torch.int64, device=device)
    row = torch.arange(rows, device=device)
    fields, c_arr, fitted_scales = select_candidate_rows(rows, cols, candidates, choice)
    summary = summarize(fields, c_arr)
    baseline = candidates[baseline_index]
    target_bytes = total_bytes(baseline.counts, baseline.sizes)

    for _ in range(iterations):
        table = rate_costs(summary.counts, summary.sizes, summary.sym_min, minima, maxima, device)
        rates = torch.stack([row_rates(rows, cols, candidate, table) for candidate in candidates], dim=1)
        current_rate = rates[row, choice].sum()
        current_bytes = total_bytes(summary.counts, summary.sizes)
        desired_rate = current_rate + BITS_PER_BYTE * (target_bytes - current_bytes)
        choice = choose_rate_tradeoff(distortion, rates, row, desired_rate)
        fields, c_arr, fitted_scales = select_candidate_rows(rows, cols, candidates, choice)
        summary = summarize(fields, c_arr)

    return summary, fitted_scales
