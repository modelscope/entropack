import numpy as np

from ..tile_ans.eager import normalize_counts, quantized_cross_entropy
from ..tile_ans.format import NUM_STATES, STATE_MIN
from .format import BITS_PER_BYTE, NUM_COORD_FIELDS, vector_tile_elements

__all__ = [
    "build_codec_tables", "coded_bytes", "decode_vector_stream", "encode_vector_stream", "normalize_freq",
    "vector_stream_analytic_bytes",
]


def normalize_freq(counts: np.ndarray, table_size: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0 or int(counts.sum()) == 0:
        raise ValueError("cannot build an rANS table from an empty symbol stream")
    if counts.size > table_size:
        raise ValueError(
            f"E8 coordinate alphabet {counts.size} exceeds rANS table_size {table_size}; "
            "raise prob_bits or coarsen the lattice scale"
        )
    return normalize_counts(counts, table_size)


def build_codec_tables(freq: np.ndarray, probability_bits: int):
    freq = np.asarray(freq, dtype=np.int64)
    table_size = 1 << probability_bits
    if freq.size > table_size:
        raise ValueError(f"alphabet {freq.size} exceeds table_size {table_size}")
    if int(freq.sum()) != table_size:
        raise ValueError("normalized frequencies must sum to table_size")
    cdf = np.zeros(freq.size, dtype=np.int64)
    cdf[1:] = np.cumsum(freq)[:-1]
    lut = np.zeros(table_size, dtype=np.uint64)
    running = 0
    for symbol in range(freq.size):
        value = int(freq[symbol])
        if value == 0:
            continue
        packed_freq = 0 if value == table_size else value
        entry = (np.uint64(running) << np.uint64(32)) | (np.uint64(packed_freq) << np.uint64(16)) | np.uint64(symbol)
        lut[running : running + value] = entry
        running += value
    if running != table_size:
        raise AssertionError(f"frequency sum is {running}, expected {table_size}")
    return cdf, lut


def _decode_group(states, luts, table_ids, words, pointer, output, valid_lanes, probability_bits):
    table_size = 1 << probability_bits
    reads = []
    for lane in range(valid_lanes):
        state = int(states[lane])
        slot = state & (table_size - 1)
        entry = int(luts[0 if table_ids is None else int(table_ids[lane])][slot])
        symbol = entry & 0xFFFF
        frequency = (entry >> 16) & 0xFFFF
        if frequency == 0:
            frequency = table_size
        cdf = entry >> 32
        output[lane] = symbol
        state = frequency * (state >> probability_bits) + (slot - cdf)
        states[lane] = state
        if state < STATE_MIN:
            reads.append(lane)
    first_word = pointer - len(reads)
    if first_word < 0:
        raise ValueError("lattice_rans rANS payload is truncated")
    for index, lane in enumerate(reads):
        states[lane] = (int(states[lane]) << 16) | int(words[first_word + index])
    return first_word


def _decode_tile_group(luts, states, words, pointer, coset_out, field_out, valid_lanes, probability_bits):
    pointer = _decode_group(states, luts, None, words, pointer, coset_out, valid_lanes, probability_bits)
    for field in range(NUM_COORD_FIELDS):
        pointer = _decode_group(
            states, [luts[1 + 2 * field], luts[2 + 2 * field]], coset_out, words, pointer, field_out[:, field], valid_lanes,
            probability_bits,
        )
    return pointer


def encode_vector_stream(
    cosets: np.ndarray, field_symbols: np.ndarray, frequencies: list[np.ndarray], probability_bits: int, tile_vectors: int,
):
    cosets = np.asarray(cosets, dtype=np.int64)
    field_symbols = np.asarray(field_symbols, dtype=np.int64)
    if field_symbols.shape != (cosets.size, NUM_COORD_FIELDS):
        raise ValueError(f"field_symbols must have shape [num_vectors, {NUM_COORD_FIELDS}]")
    cdfs = []
    for frequency in frequencies:
        frequency = np.asarray(frequency, dtype=np.int64)
        if frequency.size:
            cdfs.append(build_codec_tables(frequency, probability_bits)[0])
        else:
            cdfs.append(np.empty(0, dtype=np.int64))
    table_size = 1 << probability_bits
    state_check_shift = 31 - probability_bits
    num_tiles = max(1, (cosets.size + tile_vectors - 1) // tile_vectors)
    states = np.empty((num_tiles, NUM_STATES), dtype=np.uint32)
    parts = []
    offsets = np.zeros(num_tiles + 1, dtype=np.int64)
    total = 0
    for tile in range(num_tiles):
        begin = tile * tile_vectors
        end = min(begin + tile_vectors, cosets.size)
        tile_states = np.full(NUM_STATES, STATE_MIN, dtype=np.uint32)
        words = []
        for base in range(begin, end, NUM_STATES):
            limit = min(NUM_STATES, end - base)
            for field in range(7, -1, -1):
                for lane in range(limit):
                    index = base + lane
                    table = 1 + 2 * field + int(cosets[index])
                    symbol = int(field_symbols[index, field])
                    frequency = int(frequencies[table][symbol])
                    state = int(tile_states[lane])
                    if state >= (frequency << state_check_shift):
                        words.append(state & 0xFFFF)
                        state >>= 16
                    tile_states[lane] = (state // frequency) * table_size + (state % frequency) + int(cdfs[table][symbol])
            for lane in range(limit):
                symbol = int(cosets[base + lane])
                frequency = int(frequencies[0][symbol])
                state = int(tile_states[lane])
                if state >= (frequency << state_check_shift):
                    words.append(state & 0xFFFF)
                    state >>= 16
                tile_states[lane] = (state // frequency) * table_size + (state % frequency) + int(cdfs[0][symbol])
        tile_words = np.asarray(words, dtype=np.uint16)
        states[tile] = tile_states
        parts.append(tile_words)
        total += tile_words.size
        offsets[tile + 1] = total
    payload = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)
    return payload, states, offsets


def decode_vector_stream(
    words: np.ndarray, offsets: np.ndarray, states: np.ndarray, frequencies: list[np.ndarray], probability_bits: int,
    tile_vectors: int, num_vectors: int,
):
    luts = [
        np.empty(0, dtype=np.uint64) if np.asarray(frequency).size == 0
        else build_codec_tables(np.asarray(frequency, dtype=np.int64), probability_bits)[1] for frequency in frequencies
    ]
    cosets = np.empty(num_vectors, dtype=np.int64)
    fields = np.empty((num_vectors, NUM_COORD_FIELDS), dtype=np.int64)
    num_tiles = max(1, (num_vectors + tile_vectors - 1) // tile_vectors)
    for tile in range(num_tiles):
        begin = tile * tile_vectors
        tile_count = min(tile_vectors, num_vectors - begin)
        tile_words = np.asarray(words[offsets[tile] : offsets[tile + 1]], dtype=np.uint16)
        tile_states = np.asarray(states[tile], dtype=np.uint32).copy()
        pointer = tile_words.size
        remainder = tile_count % NUM_STATES
        offset = tile_count - remainder
        if remainder:
            pointer = _decode_tile_group(
                luts, tile_states, tile_words, pointer, cosets[begin + offset : begin + tile_count],
                fields[begin + offset : begin + tile_count], remainder, probability_bits,
            )
        while offset > 0:
            offset -= NUM_STATES
            pointer = _decode_tile_group(
                luts, tile_states, tile_words, pointer, cosets[begin + offset : begin + offset + NUM_STATES],
                fields[begin + offset : begin + offset + NUM_STATES], NUM_STATES, probability_bits,
            )
        if pointer != 0:
            raise ValueError("lattice_rans vector rANS stream contains unread payload words")
    return cosets, fields


def coded_bytes(counts, sizes, prob_bits: int, tile_elements: int) -> float:
    table_size = 1 << prob_bits
    counts_by_table = []
    frequencies = []
    overflow_bits = 0.0
    empty_counts = np.empty(0, dtype=np.int64)
    empty_freqs = np.empty(0, dtype=np.uint16)
    for n_symbols, count in zip(sizes, counts, strict=True):
        if n_symbols and count.size <= table_size:
            counts_by_table.append(count)
            frequencies.append(normalize_freq(count, table_size))
        else:
            overflow_bits += n_symbols * prob_bits
            counts_by_table.append(empty_counts)
            frequencies.append(empty_freqs)
    total = vector_stream_analytic_bytes(
        counts_by_table, frequencies, prob_bits, vector_tile_elements(tile_elements), sizes[0]
    )
    return total + overflow_bits / BITS_PER_BYTE


def vector_stream_analytic_bytes(
    counts_by_table: list[np.ndarray], frequencies: list[np.ndarray], probability_bits: int, tile_vectors: int,
    num_vectors: int,
) -> float:
    bits = 0.0
    frequency_bytes = 0
    for counts, frequency in zip(counts_by_table, frequencies, strict=True):
        if counts.size:
            bits += quantized_cross_entropy(counts.astype(np.int64), frequency.astype(np.int64), probability_bits)
            frequency_bytes += frequency.size * 2
    num_tiles = max(1, (num_vectors + tile_vectors - 1) // tile_vectors)
    state_residual_bits = num_tiles * NUM_STATES * 16
    payload_words = int(np.ceil(max(0.0, bits - state_residual_bits) / 16.0))
    return payload_words * 2 + num_tiles * NUM_STATES * 4 + (num_tiles + 1) * 4 + frequency_bytes
