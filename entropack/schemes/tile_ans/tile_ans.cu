#include "device.cuh"

// Encode kernels:
//   tile_ans_histogram_kernel       per-byte-lane 256-bin symbol counts. Every warp keeps a private histogram in shared
//                                   memory and the block folds them, so a block contributes at most 256 global atomics per
//                                   lane.
//   tile_ans_encode_kernel          one warp per (tile, byte lane) stream: 32 interleaved rANS states, 16-bit
//                                   renormalization, words written to that stream's scratch region. A lane marked raw is
//                                   packed two bytes per word instead of being coded.
//   tile_ans_compact_kernel         gathers the per-stream scratch words into one payload.
//
// Decode kernels, selected by the byte-lane pattern stored in the checkpoint. All produce identical output and differ only
// in how many lanes one warp reassembles at once:
//   tile_ans_decode_raw0_ans1_kernel  two lanes, raw then coded: the bf16 case, where one warp rebuilds both bytes of every
//                                     element and stages the coded lane's table in shared memory.
//   tile_ans_decode_raw3_ans1_kernel  four lanes, the first three raw: the fp32 case.
//   tile_ans_decode_all_raw_kernel    every lane raw, i.e. nothing was compressible: one block per tile, a plain unpack.
//   tile_ans_decode_kernel            the general case: the grid carries the byte lane, coded lanes are decoded by rANS
//                                     through a shared-memory table and raw lanes are unpacked.
// All four share one argument list, so the host builds one tuple and picks a name; the raw-only kernels ignore the state and
// table arguments.
//
// The rANS state machine, the renormalization variants and the shared decode helpers live in device.cuh, which the
// lattice_rans lane compiles against as well.


extern "C" __global__ void tile_ans_histogram_kernel(
    const uint8_t* __restrict__ input,
    uint64_t* __restrict__ histograms,
    int64_t num_elements,
    int num_lanes) {
  extern __shared__ uint32_t warp_bins[];
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int bins_per_warp = num_lanes * 256;
  const int total_bins = warps_per_block * bins_per_warp;
  for (int index = threadIdx.x; index < total_bins; index += blockDim.x) {
    warp_bins[index] = 0;
  }
  __syncthreads();

  uint32_t* bins = warp_bins + warp * bins_per_warp;
  int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; element < num_elements; element += stride) {
    const uint8_t* value = input + element * num_lanes;
#pragma unroll
    for (int byte_lane = 0; byte_lane < num_lanes; ++byte_lane) {
      atomicAdd(&bins[byte_lane * 256 + value[byte_lane]], 1u);
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < bins_per_warp; index += blockDim.x) {
    uint32_t sum = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < warps_per_block; ++source_warp) {
      sum += warp_bins[source_warp * bins_per_warp + index];
    }
    if (sum) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(histograms + index),
          static_cast<unsigned long long>(sum));
    }
  }
}

extern "C" __global__ void tile_ans_encode_kernel(
    const uint8_t* __restrict__ input,
    const uint16_t* __restrict__ frequencies,
    const uint16_t* __restrict__ cdfs,
    const uint8_t* __restrict__ lane_modes,
    uint16_t* __restrict__ scratch,
    uint32_t* __restrict__ word_counts,
    uint32_t* __restrict__ final_states,
    int64_t num_elements,
    int tile_elements,
    int num_lanes,
    int num_tiles) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  const int byte_lane = blockIdx.y;

  extern __shared__ uint16_t shared_encode_tables[];
  uint16_t* frequency = shared_encode_tables;
  uint16_t* cdf = frequency + 256;
  const uint16_t* source_frequency = frequencies + byte_lane * 256;
  const uint16_t* source_cdf = cdfs + byte_lane * 256;
  for (int index = threadIdx.x; index < 256; index += blockDim.x) {
    frequency[index] = source_frequency[index];
    cdf[index] = source_cdf[index];
  }
  __syncthreads();

  if (tile >= num_tiles || byte_lane >= num_lanes) {
    return;
  }
  const int stream = byte_lane * num_tiles + tile;
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_elements;
  const int tile_count = static_cast<int>(
      (num_elements - tile_begin < tile_elements)
          ? (num_elements - tile_begin)
          : tile_elements);

  uint16_t* stream_scratch = scratch + static_cast<int64_t>(stream) * tile_elements;
  if (lane_modes[byte_lane] != 0) {
    const int raw_words = (tile_count + 1) >> 1;
    for (int word = lane; word < raw_words; word += kNumStates) {
      const int first = word << 1;
      const uint32_t low = input[(tile_begin + first) * num_lanes + byte_lane];
      const uint32_t high = (first + 1 < tile_count)
          ? input[(tile_begin + first + 1) * num_lanes + byte_lane]
          : 0u;
      stream_scratch[word] = static_cast<uint16_t>(low | (high << 8));
    }
    if (lane == 0) {
      word_counts[stream] = raw_words;
    }
    return;
  }

  int ans_lane = 0;
  for (int prior_lane = 0; prior_lane < byte_lane; ++prior_lane) {
    ans_lane += lane_modes[prior_lane] == 0;
  }
  const int ans_stream = ans_lane * num_tiles + tile;
  uint32_t state = kStateMin;
  uint32_t word_count = 0;
  constexpr uint32_t state_check_mul = 1u << (31 - kProbBits);

  for (int base = 0; base < tile_count; base += kNumStates) {
    const bool valid = base + lane < tile_count;
    const uint32_t symbol = valid
        ? input[(tile_begin + base + lane) * num_lanes + byte_lane]
        : 0u;
    const uint32_t freq = valid ? frequency[symbol] : 1u;
    const bool emit = valid && state >= freq * state_check_mul;
    const uint32_t vote = __ballot_sync(0xFFFFFFFFu, emit);
    const uint32_t prefix = __popc(vote & lane_mask_lt());
    if (emit) {
      stream_scratch[word_count + prefix] = static_cast<uint16_t>(state);
      state >>= 16;
    }
    word_count += __popc(vote);
    if (valid) {
      state = (state / freq) * kTableSize + (state % freq) + cdf[symbol];
    }
  }

  final_states[ans_stream * kNumStates + lane] = state;
  if (lane == 0) {
    word_counts[stream] = word_count;
  }
}

extern "C" __global__ void tile_ans_compact_kernel(
    const uint16_t* __restrict__ scratch,
    const uint32_t* __restrict__ offsets,
    uint16_t* __restrict__ payload,
    int tile_elements,
    int num_streams) {
  const int stream = blockIdx.x;
  if (stream >= num_streams) {
    return;
  }
  const uint32_t begin = offsets[stream];
  const uint32_t count = offsets[stream + 1] - begin;
  const uint16_t* source = scratch + static_cast<int64_t>(stream) * tile_elements;
  for (uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
    payload[begin + index] = source[index];
  }
}

extern "C" __global__ void tile_ans_decode_raw0_ans1_kernel(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint32_t* __restrict__ decode_tables,
    const uint8_t* __restrict__ lane_modes,
    uint8_t* __restrict__ output,
    int64_t num_elements,
    int tile_elements,
    int num_lanes,
    int num_tiles) {
  extern __shared__ uint32_t shared_table[];
  const uint4* src4 = reinterpret_cast<const uint4*>(decode_tables + kTableSize);
  uint4* dst4 = reinterpret_cast<uint4*>(shared_table);
  for (int i = threadIdx.x; i < kTableSize / 4; i += blockDim.x) dst4[i] = src4[i];
  __syncthreads();
  const uint32_t* __restrict__ table = shared_table;

  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  if (tile >= num_tiles || num_lanes != 2) {
    return;
  }

  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_elements;
  const int tile_count = static_cast<int>(
      (num_elements - tile_begin < tile_elements)
          ? (num_elements - tile_begin)
          : tile_elements);
  const int raw_stream = tile;
  const int ans_stream = num_tiles + tile;
  const uint32_t expected_raw_words = (tile_count + 1) >> 1;
  if (offsets[raw_stream + 1] - offsets[raw_stream] != expected_raw_words) {
    return;
  }
  const uint16_t* raw_words = payload + offsets[raw_stream];
  const uint16_t* input_begin = payload + offsets[ans_stream];
  const uint16_t* input = payload + offsets[ans_stream + 1];
  uint32_t state = states[tile * kNumStates + lane];
  uint16_t* output16 = reinterpret_cast<uint16_t*>(output);
  const uint32_t mask_ge = lane_mask_ge();

  const int remainder = tile_count & (kNumStates - 1);
  int output_offset = tile_count - remainder;
  if (remainder) {
    const bool valid = lane < remainder;
    const int local_index = output_offset + lane;
    if (valid) {
      const uint32_t high = rans_decode_symbol(state, table);
      const uint32_t packed_low = raw_words[local_index >> 1];
      const uint32_t low = (packed_low >> ((local_index & 1) * 8)) & 0xFFu;
      output16[tile_begin + local_index] = static_cast<uint16_t>(low | (high << 8));
    }
    rans_renormalize_unchecked(valid, state, input, mask_ge);
  }

  while (output_offset > 0) {
    output_offset -= kNumStates;
    const int local_index = output_offset + lane;
    const uint32_t high = rans_decode_symbol(state, table);
    const uint32_t packed_low = raw_words[local_index >> 1];
    const uint32_t low = (packed_low >> ((local_index & 1) * 8)) & 0xFFu;
    output16[tile_begin + local_index] = static_cast<uint16_t>(low | (high << 8));
    rans_renormalize_unchecked(true, state, input, mask_ge);
  }
}

__device__ __forceinline__ void decode_group_raw3_ans1(
    bool valid,
    uint32_t& state,
    const uint32_t* __restrict__ table,
    const uint16_t*& input,
    const uint16_t* __restrict__ input_begin,
    const uint16_t* __restrict__ raw0,
    const uint16_t* __restrict__ raw1,
    const uint16_t* __restrict__ raw2,
    uint32_t* __restrict__ output,
    int64_t output_index,
    int local_index) {
  if (valid) {
    const uint32_t high = rans_decode_symbol(state, table);
    const int word = local_index >> 1;
    const int shift = (local_index & 1) * 8;
    const uint32_t b0 = (raw0[word] >> shift) & 0xFFu;
    const uint32_t b1 = (raw1[word] >> shift) & 0xFFu;
    const uint32_t b2 = (raw2[word] >> shift) & 0xFFu;
    output[output_index] = b0 | (b1 << 8) | (b2 << 16) | (high << 24);
  }
  (void)rans_renormalize_checked(valid, state, input, input_begin);
}

extern "C" __global__ void tile_ans_decode_raw3_ans1_kernel(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint32_t* __restrict__ decode_tables,
    const uint8_t* __restrict__ lane_modes,
    uint8_t* __restrict__ output,
    int64_t num_elements,
    int tile_elements,
    int num_lanes,
    int num_tiles) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;

  const uint32_t* table = decode_tables + 3 * kTableSize;
  if (tile >= num_tiles || num_lanes != 4) {
    return;
  }
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_elements;
  const int tile_count = static_cast<int>(
      (num_elements - tile_begin < tile_elements)
          ? (num_elements - tile_begin)
          : tile_elements);

  const uint32_t expected_raw_words = (tile_count + 1) >> 1;
  if (offsets[tile + 1] - offsets[tile] != expected_raw_words ||
      offsets[num_tiles + tile + 1] - offsets[num_tiles + tile] != expected_raw_words ||
      offsets[2 * num_tiles + tile + 1] - offsets[2 * num_tiles + tile] != expected_raw_words) {
    return;
  }
  const uint16_t* raw0 = payload + offsets[tile];
  const uint16_t* raw1 = payload + offsets[num_tiles + tile];
  const uint16_t* raw2 = payload + offsets[2 * num_tiles + tile];
  const int ans_stream = 3 * num_tiles + tile;
  const uint16_t* input_begin = payload + offsets[ans_stream];
  const uint16_t* input = payload + offsets[ans_stream + 1];
  uint32_t state = states[tile * kNumStates + lane];
  uint32_t* output32 = reinterpret_cast<uint32_t*>(output);

  const int remainder = tile_count & (kNumStates - 1);
  int output_offset = tile_count - remainder;
  if (remainder) {
    decode_group_raw3_ans1(
        lane < remainder, state, table, input, input_begin,
        raw0, raw1, raw2, output32,
        tile_begin + output_offset + lane, output_offset + lane);
  }
  while (output_offset > 0) {
    output_offset -= kNumStates;
    decode_group_raw3_ans1(
        true, state, table, input, input_begin,
        raw0, raw1, raw2, output32,
        tile_begin + output_offset + lane, output_offset + lane);
  }
}

extern "C" __global__ void tile_ans_decode_all_raw_kernel(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint32_t* __restrict__ decode_tables,
    const uint8_t* __restrict__ lane_modes,
    uint8_t* __restrict__ output,
    int64_t num_elements,
    int tile_elements,
    int num_lanes,
    int num_tiles) {
  const int tile = blockIdx.x;
  if (tile >= num_tiles) {
    return;
  }
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_elements;
  const int tile_count = static_cast<int>(
      (num_elements - tile_begin < tile_elements)
          ? (num_elements - tile_begin)
          : tile_elements);

  for (int element = threadIdx.x; element < tile_count; element += blockDim.x) {
    uint64_t value = 0;
#pragma unroll
    for (int byte_lane = 0; byte_lane < num_lanes; ++byte_lane) {
      const int stream = byte_lane * num_tiles + tile;
      const uint32_t expected_words = (tile_count + 1) >> 1;
      if (offsets[stream + 1] - offsets[stream] != expected_words) {
        return;
      }
      const uint16_t packed = payload[offsets[stream] + (element >> 1)];
      const uint32_t byte = (packed >> ((element & 1) * 8)) & 0xFFu;
      value |= static_cast<uint64_t>(byte) << (byte_lane * 8);
    }
    const int64_t output_index = tile_begin + element;
    if (num_lanes == 8) {
      reinterpret_cast<uint64_t*>(output)[output_index] = value;
    } else if (num_lanes == 4) {
      reinterpret_cast<uint32_t*>(output)[output_index] = static_cast<uint32_t>(value);
    } else if (num_lanes == 2) {
      reinterpret_cast<uint16_t*>(output)[output_index] = static_cast<uint16_t>(value);
    } else {
      output[output_index] = static_cast<uint8_t>(value);
    }
  }
}

extern "C" __global__ void tile_ans_decode_kernel(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint32_t* __restrict__ decode_tables,
    const uint8_t* __restrict__ lane_modes,
    uint8_t* __restrict__ output,
    int64_t num_elements,
    int tile_elements,
    int num_lanes,
    int num_tiles) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  const int byte_lane = blockIdx.y;

  const bool raw_lane = lane_modes[byte_lane] != 0;
  extern __shared__ uint32_t shared_decode_tables[];
  uint32_t* table = shared_decode_tables;
  const uint32_t* source_table = decode_tables + byte_lane * kTableSize;
  if (!raw_lane) {
    for (int index = threadIdx.x; index < static_cast<int>(kTableSize); index += blockDim.x) {
      table[index] = source_table[index];
    }
  }
  __syncthreads();

  if (tile >= num_tiles || byte_lane >= num_lanes) {
    return;
  }
  const int stream = byte_lane * num_tiles + tile;
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_elements;
  const int tile_count = static_cast<int>(
      (num_elements - tile_begin < tile_elements)
          ? (num_elements - tile_begin)
          : tile_elements);

  const uint32_t begin_word = offsets[stream];
  const uint32_t end_word = offsets[stream + 1];
  const uint16_t* input_begin = payload + begin_word;
  const uint16_t* input = payload + end_word;

  if (raw_lane) {
    const int raw_words = (tile_count + 1) >> 1;
    if (end_word - begin_word != static_cast<uint32_t>(raw_words)) {
      return;
    }
    for (int word = lane; word < raw_words; word += kNumStates) {
      const uint32_t packed = input_begin[word];
      const int first = word << 1;
      output[(tile_begin + first) * num_lanes + byte_lane] =
          static_cast<uint8_t>(packed);
      if (first + 1 < tile_count) {
        output[(tile_begin + first + 1) * num_lanes + byte_lane] =
            static_cast<uint8_t>(packed >> 8);
      }
    }
    return;
  }

  int ans_lane = 0;
  for (int prior_lane = 0; prior_lane < byte_lane; ++prior_lane) {
    ans_lane += lane_modes[prior_lane] == 0;
  }
  const int ans_stream = ans_lane * num_tiles + tile;
  uint32_t state = states[ans_stream * kNumStates + lane];
  const int remainder = tile_count & (kNumStates - 1);
  int output_offset = tile_count - remainder;
  if (remainder) {
    const bool valid = lane < remainder;
    decode_group(
        valid,
        state,
        table,
        input,
        input_begin,
        output,
        tile_begin + output_offset + lane,
        num_lanes,
        byte_lane);
  }

  while (output_offset > 0) {
    output_offset -= kNumStates;
    decode_group(
        true,
        state,
        table,
        input,
        input_begin,
        output,
        tile_begin + output_offset + lane,
        num_lanes,
        byte_lane);
  }
}
