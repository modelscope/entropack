#pragma once

#include <cuda/std/cstdint>

#ifndef TILE_ANS_PROB_BITS
#define TILE_ANS_PROB_BITS 12
#endif

constexpr int kProbBits = TILE_ANS_PROB_BITS;
constexpr uint32_t kTableSize = 1u << kProbBits;
constexpr uint32_t kStateMask = kTableSize - 1u;
constexpr uint32_t kStateMin = 1u << 15;
constexpr int kNumStates = 32;

__device__ __forceinline__ uint32_t lane_mask_ge() {
  uint32_t mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ uint32_t lane_mask_lt() {
  uint32_t mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ uint32_t rans_decode_symbol(
    uint32_t& state,
    const uint32_t* __restrict__ table) {
  const uint32_t slot = state & kStateMask;
  const uint32_t entry = table[slot];
  const uint32_t symbol = entry & 0xFFu;
  uint32_t frequency = (entry >> 8) & 0xFFFu;
  if (frequency == 0) {
    frequency = kTableSize;
  }
  const uint32_t cdf = entry >> 20;
  state = frequency * (state >> kProbBits) + (slot - cdf);
  return symbol;
}

__device__ __forceinline__ bool rans_renormalize_checked(
    bool valid,
    uint32_t& state,
    const uint16_t*& input,
    const uint16_t* __restrict__ input_begin) {
  const bool read = valid && state < kStateMin;
  const uint32_t vote = __ballot_sync(0xFFFFFFFFu, read);
  const uint32_t prefix = __popc(vote & lane_mask_ge());
  bool valid_read = true;
  if (read) {
    const uint16_t* address = input - prefix;
    valid_read = address >= input_begin;
    const uint32_t word = valid_read ? *address : 0u;
    state = (state << 16) | word;
  }
  input -= __popc(vote);
  return valid_read;
}

__device__ __forceinline__ void rans_renormalize_unchecked(
    bool valid,
    uint32_t& state,
    const uint16_t*& input,
    uint32_t mask_ge) {
  const bool read = valid && state < kStateMin;
  const uint32_t vote = __ballot_sync(0xFFFFFFFFu, read);
  if (read) {
    const uint32_t prefix = __popc(vote & mask_ge);
    state = (state << 16) | input[-static_cast<int>(prefix)];
  }
  input -= __popc(vote);
}

__device__ __forceinline__ void decode_group(
    bool valid,
    uint32_t& state,
    const uint32_t* __restrict__ table,
    const uint16_t*& input,
    const uint16_t* __restrict__ input_begin,
    uint8_t* __restrict__ output,
    int64_t output_index,
    int64_t num_lanes,
    int byte_lane) {
  if (valid) {
    output[output_index * num_lanes + byte_lane] =
        static_cast<uint8_t>(rans_decode_symbol(state, table));
  }
  (void)rans_renormalize_checked(valid, state, input, input_begin);
}
