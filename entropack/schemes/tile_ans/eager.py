import heapq

import numpy as np
import torch

from .format import (
    AUTO_PROB_BITS, LANE_ANS, LANE_RAW, NUM_STATES, STATE_MIN, SUPPORTED_PROB_BITS, TileBuffers,
    make_layout, num_tiles, parse_layout_cached,
)

_STATE_BITS = 31
_RENORM_BITS = 16
_AUTO_RATIO_TOLERANCE = 0.0015


def normalize_counts(counts: np.ndarray, table_size: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    present = np.flatnonzero(counts)
    if present.size == 0:
        raise ValueError("cannot build an rANS codebook from empty input")

    target = counts.astype(np.float64) * (table_size / int(counts.sum()))
    frequencies = np.floor(target).astype(np.int64)
    frequencies[present] = np.maximum(frequencies[present], 1)

    difference = table_size - int(frequencies.sum())
    if difference > 0:
        residual = target - frequencies
        queue = [(-float(residual[symbol]), int(symbol)) for symbol in present]
        heapq.heapify(queue)
        for _ in range(difference):
            negative_residual, symbol = heapq.heappop(queue)
            frequencies[symbol] += 1
            heapq.heappush(queue, (negative_residual + 1.0, symbol))
    elif difference < 0:
        residual = target - frequencies
        queue = [(float(residual[symbol]), int(symbol)) for symbol in present if frequencies[symbol] > 1]
        if not queue:
            raise ValueError("unable to normalize rANS frequencies")
        heapq.heapify(queue)
        for step in range(-difference):
            symbol_residual, symbol = heapq.heappop(queue)
            frequencies[symbol] -= 1
            if frequencies[symbol] > 1:
                heapq.heappush(queue, (symbol_residual + 1.0, symbol))
            elif not queue and step + 1 < -difference:
                raise ValueError("unable to normalize rANS frequencies")
    return frequencies.astype(np.uint16)


def _build_tables_from_frequencies(frequencies: np.ndarray, probability_bits: int):
    frequencies = np.asarray(frequencies, dtype=np.uint16)
    table_size = 1 << probability_bits
    num_lanes = frequencies.shape[0]
    cdfs = np.zeros((num_lanes, 256), dtype=np.uint16)
    decode_tables = np.empty((num_lanes, table_size), dtype=np.uint32)

    for lane, freq in enumerate(frequencies):
        cdf = np.zeros(256, dtype=np.uint16)
        running = 0
        for symbol in range(256):
            cdf[symbol] = running
            value = int(freq[symbol])
            if value:
                packed_frequency = 0 if value == 4096 else value
                decode_tables[lane, running : running + value] = (running << 20) | (packed_frequency << 8) | symbol
                running += value
        if running != table_size:
            raise AssertionError(f"normalized frequency sum is {running}, expected {table_size}")
        cdfs[lane] = cdf
    return frequencies, cdfs, decode_tables


def build_tables_from_counts(counts_by_lane: np.ndarray, probability_bits: int):
    counts_by_lane = np.asarray(counts_by_lane, dtype=np.int64)
    table_size = 1 << probability_bits
    frequencies = np.stack([normalize_counts(counts, table_size) for counts in counts_by_lane])
    return _build_tables_from_frequencies(frequencies, probability_bits)


def quantized_cross_entropy(counts: np.ndarray, frequencies: np.ndarray, probability_bits: int) -> float:
    present = counts > 0
    return float(
        np.sum(counts[present].astype(np.float64) * (probability_bits - np.log2(frequencies[present].astype(np.float64))))
    )


def lane_modes_from_counts(
    counts_by_lane: np.ndarray, raw_lane_threshold: float, tile_elements: int, frequencies: np.ndarray | None = None,
    probability_bits: int | None = None,
) -> np.ndarray:
    if not 0.0 <= raw_lane_threshold <= 8.0:
        raise ValueError("raw_lane_threshold must be in [0, 8]")
    effective_threshold = min(raw_lane_threshold, 8.0 - (NUM_STATES * 32) / tile_elements)
    counts_by_lane = np.asarray(counts_by_lane, dtype=np.int64)
    modes = np.empty(counts_by_lane.shape[0], dtype=np.uint8)
    for lane, counts in enumerate(counts_by_lane):
        if frequencies is None:
            probabilities = counts[counts > 0].astype(np.float64) / int(counts.sum())
            bits_per_symbol = -np.sum(probabilities * np.log2(probabilities))
        else:
            if probability_bits is None:
                raise ValueError("probability_bits is required with normalized frequencies")
            bits_per_symbol = quantized_cross_entropy(counts, frequencies[lane], probability_bits) / int(counts.sum())
        modes[lane] = LANE_RAW if bits_per_symbol >= effective_threshold else LANE_ANS
    return modes


def select_coding_options(counts_by_lane: np.ndarray, raw_lane_threshold: float, tile_elements: int):
    counts_by_lane = np.asarray(counts_by_lane, dtype=np.int64)
    num_elements = int(counts_by_lane[0].sum())
    num_lanes = counts_by_lane.shape[0]
    tiles = num_tiles(num_elements, tile_elements)
    full_tiles, tail = divmod(num_elements, tile_elements)
    raw_words = full_tiles * ((tile_elements + 1) // 2) + (tail + 1) // 2
    original_bytes = num_elements * num_lanes
    # Quantizing a table can only add cost, so a lane above the threshold stays above it at every precision; when all of them
    # are, only the coarsest table needs pricing.
    plain = lane_modes_from_counts(counts_by_lane, raw_lane_threshold, tile_elements)
    probability_variants = SUPPORTED_PROB_BITS[:1] if np.all(plain == LANE_RAW) else SUPPORTED_PROB_BITS

    candidates = []
    for probability_bits in probability_variants:
        table_size = 1 << probability_bits
        frequencies = np.stack([normalize_counts(counts, table_size) for counts in counts_by_lane])
        modes = lane_modes_from_counts(counts_by_lane, raw_lane_threshold, tile_elements, frequencies, probability_bits)
        estimated_bytes = (tiles * num_lanes + 1) * 4 + num_lanes * table_size * 4 + num_lanes + 5 * 8
        for lane, mode in enumerate(modes):
            if mode == LANE_RAW:
                estimated_bytes += raw_words * 2
            else:
                cross_entropy_bits = quantized_cross_entropy(counts_by_lane[lane], frequencies[lane], probability_bits)
                estimated_bytes += cross_entropy_bits / 8
                estimated_bytes += tiles * NUM_STATES * 4
        candidates.append((estimated_bytes, probability_bits, frequencies, modes))

    # Among the tables within tolerance of the cheapest, the smallest: a coarser grid costs a little rate and saves a
    # proportionally larger decode table.
    best_bytes = min(value for value, *_ in candidates)
    tolerance = original_bytes * _AUTO_RATIO_TOLERANCE
    _, probability_bits, frequencies, modes = min(
        (candidate for candidate in candidates if candidate[0] <= best_bytes + tolerance), key=lambda candidate: candidate[1],
    )
    frequencies, cdfs, decode_tables = _build_tables_from_frequencies(frequencies, probability_bits)
    return probability_bits, frequencies, cdfs, decode_tables, modes


def _encode_raw_stream(symbols: np.ndarray) -> np.ndarray:
    words = np.zeros((symbols.size + 1) // 2, dtype=np.uint16)
    words |= symbols[0::2].astype(np.uint16)
    if symbols.size > 1:
        words[: symbols[1::2].size] |= symbols[1::2].astype(np.uint16) << 8
    return words


def _encode_stream(symbols: np.ndarray, frequencies: np.ndarray, cdfs: np.ndarray, probability_bits: int):
    states = np.full(NUM_STATES, STATE_MIN, dtype=np.uint32)
    words: list[int] = []
    table_size = 1 << probability_bits
    state_check_shift = _STATE_BITS - probability_bits

    for base in range(0, symbols.size, NUM_STATES):
        limit = min(NUM_STATES, symbols.size - base)
        for lane in range(limit):
            symbol = int(symbols[base + lane])
            frequency = int(frequencies[symbol])
            state = int(states[lane])
            if state >= (frequency << state_check_shift):
                words.append(state & 0xFFFF)
                state >>= _RENORM_BITS
            state = (state // frequency) * table_size + (state % frequency) + int(cdfs[symbol])
            states[lane] = state
    return states, np.asarray(words, dtype=np.uint16)


def encode(
    *, weight: torch.Tensor, tile_elements: int, probability_bits: int, raw_lane_threshold: float, **_ignored,
) -> TileBuffers:
    if weight.numel() == 0:
        raise ValueError("tile_ans does not support empty tensors")
    contiguous = weight.detach().contiguous()
    num_elements = contiguous.numel()
    num_lanes = contiguous.element_size()
    raw = contiguous.reshape(-1).view(torch.uint8).cpu().numpy().copy()
    lane_bytes = raw.reshape(num_elements, num_lanes)
    counts = np.stack([np.bincount(lane_bytes[:, lane], minlength=256) for lane in range(num_lanes)])
    if probability_bits == AUTO_PROB_BITS:
        probability_bits, frequencies, cdfs, decode_tables, lane_modes = select_coding_options(
            counts, raw_lane_threshold, tile_elements
        )
    else:
        frequencies, cdfs, decode_tables = build_tables_from_counts(counts, probability_bits)
        lane_modes = lane_modes_from_counts(counts, raw_lane_threshold, tile_elements)

    tiles = num_tiles(num_elements, tile_elements)
    num_streams = tiles * num_lanes
    num_ans_lanes = int((lane_modes == LANE_ANS).sum())
    states = np.empty((tiles * num_ans_lanes, NUM_STATES), dtype=np.uint32)
    offsets = np.empty(num_streams + 1, dtype=np.uint32)
    offsets[0] = 0
    payload_parts = []

    stream = 0
    ans_stream = 0
    payload_words = 0
    for lane in range(num_lanes):
        for tile in range(tiles):
            begin = tile * tile_elements
            end = min(begin + tile_elements, num_elements)
            if lane_modes[lane] == LANE_RAW:
                words = _encode_raw_stream(lane_bytes[begin:end, lane])
            else:
                stream_states, words = _encode_stream(
                    lane_bytes[begin:end, lane], frequencies[lane], cdfs[lane], probability_bits,
                )
                states[ans_stream] = stream_states
                ans_stream += 1
            if payload_words + words.size >= 1 << 32:
                raise ValueError("tile_ans payload exceeds uint32 offset capacity")
            payload_parts.append(words)
            payload_words += words.size
            offsets[stream + 1] = payload_words
            stream += 1

    payload = np.concatenate(payload_parts) if payload_parts else np.empty(0, dtype=np.uint16)
    return TileBuffers(
        payload=torch.from_numpy(payload), offsets=torch.from_numpy(offsets), states=torch.from_numpy(states),
        decode_tables=torch.from_numpy(decode_tables), lane_modes=torch.from_numpy(lane_modes),
        layout=make_layout(num_elements, num_lanes, tile_elements, probability_bits),
    )


def _decode_group(states, table, words, pointer, output, base, valid_lanes, probability_bits):
    table_size = 1 << probability_bits
    reads = []
    for lane in range(valid_lanes):
        state = int(states[lane])
        slot = state & (table_size - 1)
        entry = int(table[slot])
        symbol = entry & 0xFF
        frequency = (entry >> 8) & 0xFFF
        if frequency == 0:
            frequency = table_size
        cdf = entry >> 20
        output[base + lane] = symbol
        state = frequency * (state >> probability_bits) + (slot - cdf)
        states[lane] = state
        if state < STATE_MIN:
            reads.append(lane)

    first_word = pointer - len(reads)
    if first_word < 0:
        raise ValueError("tile_ans payload is truncated")
    for index, lane in enumerate(reads):
        states[lane] = (int(states[lane]) << _RENORM_BITS) | int(words[first_word + index])
    return first_word


def decode(buffers: TileBuffers, *, dtype: torch.dtype, **_ignored) -> torch.Tensor:
    payload = buffers.payload.detach().cpu().numpy()
    offsets = buffers.offsets.detach().cpu().numpy().astype(np.int64)
    states = buffers.states.detach().cpu().numpy().astype(np.uint32)
    tables = buffers.decode_tables.detach().cpu().numpy().astype(np.uint32)
    lane_modes = buffers.lane_modes.detach().cpu().numpy().astype(np.uint8)
    tile_elements, probability_bits, num_lanes, num_elements = parse_layout_cached(buffers.layout)

    output = np.empty(num_elements * num_lanes, dtype=np.uint8)
    tiles = num_tiles(num_elements, tile_elements)
    stream = 0
    ans_stream = 0
    for byte_lane in range(num_lanes):
        for tile in range(tiles):
            tile_begin = tile * tile_elements
            tile_count = min(tile_elements, num_elements - tile_begin)
            begin = int(offsets[stream])
            pointer = int(offsets[stream + 1])
            words = payload[begin:pointer]
            if lane_modes[byte_lane] == LANE_RAW:
                expected_words = (tile_count + 1) // 2
                if words.size != expected_words:
                    raise ValueError("tile_ans raw lane payload length is invalid")
                lane_output = np.empty(tile_count, dtype=np.uint8)
                lane_output[0::2] = (words & 0xFF).astype(np.uint8)
                if tile_count > 1:
                    lane_output[1::2] = (words[: tile_count // 2] >> 8).astype(np.uint8)
            else:
                remainder = tile_count % NUM_STATES
                stream_states = states[ans_stream].copy()
                ans_stream += 1
                pointer -= begin
                lane_output = np.empty(tile_count, dtype=np.uint8)
                offset = tile_count - remainder
                if remainder:
                    pointer = _decode_group(
                        stream_states, tables[byte_lane], words, pointer, lane_output, offset, remainder, probability_bits
                    )
                while offset > 0:
                    offset -= NUM_STATES
                    pointer = _decode_group(
                        stream_states, tables[byte_lane], words, pointer, lane_output, offset, NUM_STATES, probability_bits
                    )
                if pointer != 0:
                    raise ValueError("tile_ans payload contains unread words")
            output[tile_begin * num_lanes + byte_lane : (tile_begin + tile_count) * num_lanes : num_lanes] = lane_output
            stream += 1

    return torch.from_numpy(output).view(dtype).to(buffers.payload.device)
