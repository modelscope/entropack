# Copyright 2025 Tianyi Zhang
# Copyright 2026 ModelScope Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from DFloat11's dfloat11_utils.py and modified for EntroPack.

from copy import copy

import numpy as np
import torch

from .format import (
    DFloat11Buffers, make_layout, max_block_elems, pack_thread_meta,
    reconstruct_bf16_bits,
)


def _huffman_codec():
    from dahuffman import HuffmanCodec

    return HuffmanCodec


def exponent_counter(weight: torch.Tensor, threads_per_block: int | None = None) -> dict:
    W = weight.reshape(-1).view(torch.int16)
    exponent_8bits = ((W >> 7) & 0xFF).to(torch.int64)
    counts = torch.bincount(exponent_8bits, minlength=256).cpu().tolist()
    return {i: int(c) for i, c in enumerate(counts) if c > 0}


def get_32bit_codec(counter: dict):
    HuffmanCodec = _huffman_codec()
    codec = HuffmanCodec.from_frequencies(counter)
    table = codec.get_code_table()
    max_len = 0
    for _, (length, _) in table.items():
        max_len = max(max_len, length)

    compressed_codec = codec
    compressed_counter = counter

    min_k = 2
    freq = np.array(list(counter.values()))
    while max_len > 32:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        min_k += 1
        min_keys = np.array(list(counter.keys()))[min_indices]

        compressed_counter = copy(counter)
        for k in min_keys:
            compressed_counter[k] = 1
        compressed_codec = HuffmanCodec.from_frequencies(compressed_counter)
        table = compressed_codec.get_code_table()
        max_len = 0
        for _, (length, _) in table.items():
            max_len = max(max_len, length)

    return compressed_codec, compressed_counter, table


def get_luts(table: dict) -> torch.Tensor:
    prefixes = [""]

    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, "0")[: ((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)

    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)

    for pi, p in enumerate(prefixes):
        bytes_dict = {}
        pl = len(p) // 8
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                bin_val = bin(val)[2:].rjust(bits, "0")

                if bin_val.startswith(p):
                    if (bits - 1) // 8 == pl:
                        dict_key = int(bin_val[(pl * 8) :].ljust(8, "0"), 2)
                        dict_value = key
                    else:
                        dict_key = int(bin_val[(pl * 8) : (pl * 8 + 8)], 2)
                        dict_value = 256 - prefixes.index(bin_val[: (pl * 8 + 8)])

                    if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                        raise ValueError(f"Key {dict_key} already exists in {bytes_dict}")
                    else:
                        bytes_dict[dict_key] = dict_value

        curr_val = 0
        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[pi, i] = curr_val

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, _val) in table.items():
        if isinstance(key, int):
            lens[-1, key] = bits

    return torch.from_numpy(np.concatenate((luts, lens), axis=0))


def _encode_bitstream(data, codec, bytes_per_thread: int, threads_per_block: int):
    encoded = []

    gaps = []
    counts = []
    output_positions = []

    region_bits = 8 * bytes_per_thread
    block_bits = region_bits * threads_per_block

    buffer = 0
    size = 0
    total_size = 0
    element_count = 0
    for s in data:
        if total_size // region_bits + 1 > len(gaps):
            gaps.append(total_size - total_size // region_bits * region_bits)
            counts.append(0)

        if total_size // block_bits + 1 > len(output_positions):
            output_positions.append(element_count)

        counts[-1] += 1

        b, v = codec._table[s]
        buffer = (buffer << b) + v
        size += b
        total_size += b
        element_count += 1
        while size >= 8:
            byte = buffer >> (size - 8)
            encoded.append(byte)
            buffer = buffer - (byte << (size - 8))
            size -= 8

    if size > 0:
        if total_size // region_bits + 1 > len(gaps):
            gaps.append(0)
            counts.append(0)

        if total_size // block_bits + 1 > len(output_positions):
            output_positions.append(element_count)

        b, v = codec._table[codec._eof]
        buffer = (buffer << b) + v
        size += b
        if size >= 8:
            byte = buffer >> (size - 8)
        else:
            byte = buffer << (8 - size)
        encoded.append(byte)

    output_positions.append(len(data))

    blocks_per_grid = int(np.ceil(len(encoded) / (threads_per_block * bytes_per_thread)))
    n_regions = threads_per_block * blocks_per_grid
    gaps.extend([0] * (n_regions - len(gaps)))
    counts.extend([0] * (n_regions - len(counts)))

    return (
        np.frombuffer(bytes(encoded), dtype=np.uint8).copy(), np.array(gaps, dtype=np.int64), np.array(counts, dtype=np.int64),
        np.array(output_positions, dtype=np.uint32),
    )


def encode_weights(weights, codec, bytes_per_thread: int, threads_per_block: int):
    W_combined = torch.cat(weights).view(torch.int16)

    exponent_8bits = ((W_combined >> 7) & 0xFF).to(torch.uint8)
    other_8bits = ((W_combined >> 8) & 0x80 | (W_combined & 0x7F)).to(torch.uint8)

    encoded, gaps, counts, output_positions = _encode_bitstream(
        exponent_8bits.tolist(), codec, bytes_per_thread, threads_per_block
    )

    return (
        torch.from_numpy(encoded), other_8bits, torch.from_numpy(output_positions), pack_thread_meta(gaps, counts),
        make_layout(bytes_per_thread, threads_per_block, max_block_elems(output_positions)),
    )


def _decode_exponents(luts: np.ndarray, encoded: np.ndarray, n_elements: int) -> np.ndarray:
    lut = luts.astype(np.int64)
    lens_row = lut[-1]
    num_levels = lut.shape[0] - 1
    ptr_min = 256 - (num_levels - 1) if num_levels > 1 else 256

    bits = np.unpackbits(encoded.astype(np.uint8))
    n_bits = bits.size

    def read_byte(offset: int) -> int:
        if offset + 8 <= n_bits:
            seg = bits[offset : offset + 8]
        else:
            seg = np.zeros(8, np.uint8)
            avail = n_bits - offset
            if avail > 0:
                seg[:avail] = bits[offset:]
        return int(np.packbits(seg)[0])

    out = np.empty(n_elements, np.int64)
    cursor = 0
    for i in range(n_elements):
        level = 0
        hop = 0
        while True:
            entry = lut[level][read_byte(cursor + hop * 8)]
            if num_levels > 1 and entry >= ptr_min:
                level = 256 - entry
                hop += 1
            else:
                out[i] = entry
                cursor += int(lens_row[entry])
                break
    return out


def decode(buffers: DFloat11Buffers) -> torch.Tensor:
    encoded_exponent, sign_mantissa, luts = buffers.encoded_exponent, buffers.sign_mantissa, buffers.luts
    n_elements = sign_mantissa.numel()
    exponents = _decode_exponents(luts.detach().cpu().numpy(), encoded_exponent.detach().cpu().numpy(), n_elements)
    sm = sign_mantissa.detach().cpu().numpy().astype(np.uint8)
    bf16_bits = reconstruct_bf16_bits(exponents, sm)
    flat = torch.from_numpy(bf16_bits.view(np.int16)).view(torch.bfloat16)
    return flat.to(sign_mantissa.device)


def encode(
    *, weight: torch.Tensor, codec, luts: torch.Tensor, bytes_per_thread: int, threads_per_block: int,
) -> DFloat11Buffers:
    flat = weight.reshape(-1).cpu()
    encoded_exponent, sign_mantissa, output_positions, thread_meta, layout = encode_weights(
        [flat], codec, bytes_per_thread, threads_per_block
    )
    return DFloat11Buffers(
        encoded_exponent=encoded_exponent, sign_mantissa=sign_mantissa, luts=luts,
        output_positions=output_positions, thread_meta=thread_meta, layout=layout,
    )
