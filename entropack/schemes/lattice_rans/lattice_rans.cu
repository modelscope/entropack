// EntroPack lattice_rans CUDA codec: E8-lattice vector quantization + coset-conditioned rANS.
//
// Encode kernels:
//   e8_quantize_fields_kernel/_f32  nearest-E8 (Conway-Sloane: the two cosets D8 and D8+g, each reduced by a parity fix on
//                                   the max-residual coordinate) fused with the point->fields split (coset c, parity-reduced
//                                   coordinates z0..z6, m) and a block-reduced per-stream min/max pass. Bit-exact with
//                                   eager.nearest_e8 / point_to_fields: rintf == torch.round (half-to-even), the squared
//                                   distances use torch's sum(dim=1) tree order, __fmul_rn/__fadd_rn block FMA contraction,
//                                   argmax keeps the lowest index on ties, and the coset tie rule is d0 <= d1. The two
//                                   variants differ only in how the weight is read: bf16 native, fp32 for every other
//                                   container.
//   e8_refit_scales_kernel/_f32     least-squares refit of each row scale against the quantized points.
//   e8_minmax_fields_kernel         per-stream symbol min/max; sizes the alphabets.
//   e8_histogram_kernel             exact per-stream symbol counts. Bins are staged in shared memory when they fit the
//                                   device budget, and fall back to global atomics otherwise.
//   e8_rans_encode_vector_kernel    tiled 32-state interleaved rANS encode, one warp per tile, bit-exact with
//                                   tile_ans/encode_cpu._encode_stream: the state machine of device.cuh, 16-bit words,
//                                   ballot-coalesced renormalization emission.
//   e8_compact_kernel               gathers the per-tile scratch words into one payload.
//
// Decode kernels: one warp decodes one tile and reconstructs straight into the container, with no symbol scratch and no
// prefix sum. The variants differ only in how the slot->symbol table is held, and all produce identical output, so the host
// picks one from the stored alphabet widths and the queried device limits:
//   e8_decode_vector_shlut8pf_*     slot->symbol (uint8) and begin|freq tables staged in shared memory. Used when every
//                                   alphabet is below 256 and the staged table still leaves the SM its resident-block
//                                   target.
//   e8_decode_vector_packed32pf_*   symbol|freq|delta packed into 32 bits in global memory, so no shared memory is needed
//                                   and occupancy is unrestricted, plus an L2 prefetch of the descending renorm stream.
//   e8_decode_vector_packed32_*     the same without the prefetch, reached by decoding with l2_prefetch=False.
//   e8_decode_vector_fused_*        64-bit table in global memory; the fallback for alphabets too wide to pack into 32 bits.
// A "_g" suffix marks the generic-container twin, which takes a store kind and writes any supported container instead of
// bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>
#include "device.cuh"

namespace {

constexpr int kCMax = 32;
constexpr int kMinMaxLen = kCMax + 1;
constexpr int kIntMax = 0x7FFFFFFF;
constexpr int kIntMin = -0x7FFFFFFF - 1;

constexpr int kMetaStride = 4;
constexpr int kMetaSymMin = 1;
constexpr int kMetaFreqOff = 2;

__device__ __forceinline__ void set_error(int* error, int code) {
  atomicCAS(error, 0, code);
}

__device__ __forceinline__ int div_i64_i32(int64_t a, int b) {
  return static_cast<int>(a / b);
}

__device__ __forceinline__ void nearest_dn_int(
    const float* __restrict__ y, int* __restrict__ out) {
  float f[8];
  float r[8];
  int parity = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    f[j] = rintf(y[j]);                 // torch.round == round-half-to-even == rintf
    r[j] = __fsub_rn(y[j], f[j]);
    parity += static_cast<int>(f[j]);   // exact: f is an integer float within int32 range
  }
  int idx = 0;
  float best = -1.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float a = fabsf(r[j]);
    if (a > best) {                     // strict > keeps the lowest index on ties (torch argmax)
      best = a;
      idx = j;
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[j] = static_cast<int>(f[j]);
  }
  if ((parity & 1) != 0) {
    out[idx] += (r[idx] < 0.0f) ? -1 : 1;   // torch.sign(0) is replaced by +1 in eager
  }
}

// Accumulated in float64, in torch's sum(dim=1) tree order, so exact ties resolve identically to eager.nearest_e8:
// the parenthesization is the point and must not be flattened.
__device__ __forceinline__ double dist2_f64(
    const float* __restrict__ y, const int* __restrict__ z, float offset) {
  double s[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const double c = static_cast<double>(z[j]) + static_cast<double>(offset);
    const double r = static_cast<double>(y[j]) - c;
    s[j] = r * r;
  }
  return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
}

__device__ __forceinline__ void block_minmax_init(int* shared) {
  for (int t = threadIdx.x; t < kMinMaxLen; t += blockDim.x) {
    shared[t] = ((t & 1) == 0 && t != kCMax) ? kIntMax : kIntMin;
  }
  __syncthreads();
}

__device__ __forceinline__ void block_minmax_flush(int* shared, int* __restrict__ minmax) {
  __syncthreads();
  for (int t = threadIdx.x; t < kMinMaxLen; t += blockDim.x) {
    const int v = shared[t];
    if ((t & 1) == 0 && t != kCMax) {
      if (v != kIntMax) atomicMin(&minmax[t], v);
    } else {
      if (v != kIntMin) atomicMax(&minmax[t], v);
    }
  }
}

}  // namespace

__device__ __forceinline__ void e8_load8(const __nv_bfloat16* w, float* xf) {
  const uint4 packed = *reinterpret_cast<const uint4*>(w);
  const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned bits = (words[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu;
    xf[j] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
  }
}

__device__ __forceinline__ void e8_load8(const float* w, float* xf) {
  const float4 lo = *reinterpret_cast<const float4*>(w);
  const float4 hi = *reinterpret_cast<const float4*>(w + 4);
  xf[0] = lo.x; xf[1] = lo.y; xf[2] = lo.z; xf[3] = lo.w;
  xf[4] = hi.x; xf[5] = hi.y; xf[6] = hi.z; xf[7] = hi.w;
}

__device__ __forceinline__ float e8_load_scalar(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

__device__ __forceinline__ float e8_load_scalar(const float* p) {
  return *p;
}

template <typename InT>
__device__ __forceinline__ void e8_quantize_fields_body(
    const InT* __restrict__ weight,
    const float* __restrict__ rms,
    float s,
    int rows,
    int cols,
    int vecs_per_row,
    int* __restrict__ fields,   // [V,8] int32: z0..z6, m
    int* __restrict__ c_arr,    // [V] int32
    int* __restrict__ minmax) { // kMinMaxLen int32, pre-initialized to kIntMax/kIntMin
  __shared__ int sm[kMinMaxLen];
  block_minmax_init(sm);

  const int64_t V = static_cast<int64_t>(rows) * vecs_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < V; idx += stride) {
    const int row = div_i64_i32(idx, vecs_per_row);
    const int col8 = static_cast<int>(idx - static_cast<int64_t>(row) * vecs_per_row);
    const float rmsv = rms[row];
    const InT* w = weight + static_cast<int64_t>(row) * cols + col8 * 8;
    float xf[8];
    e8_load8(w, xf);
    float y[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __fdiv_rn(__fdiv_rn(xf[j], rmsv), s);
    }

    int z0[8];
    int z1[8];
    nearest_dn_int(y, z0);
    float ym[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ym[j] = __fsub_rn(y[j], 0.5f);
    }
    nearest_dn_int(ym, z1);
    const double d0 = dist2_f64(y, z0, 0.0f);
    const double d1 = dist2_f64(y, z1, 0.5f);
    const bool pick0 = d0 <= d1;       // tie picks the D8 coset, exactly as eager
    const int* z = pick0 ? z0 : z1;
    const int c = pick0 ? 0 : 1;

    int zv[8];
    int par = 0;
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      zv[j] = z[j];
      par += zv[j];
    }
    par &= 1;                                // two's-complement &1 == torch.remainder(., 2)
    zv[7] = (z[7] - par) >> 1;               // m = floor((z7 - par)/2)

    int* fout = fields + idx * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      fout[j] = zv[j];
    }
    c_arr[idx] = c;

    atomicMax(&sm[kCMax], c);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int st = j * 2 + c;              // coord stream index (field f, coset k)
      atomicMin(&sm[st * 2], zv[j]);
      atomicMax(&sm[st * 2 + 1], zv[j]);
    }
  }
  block_minmax_flush(sm, minmax);
}

#define E8_QUANTIZE_WRAPPER(NAME, INT)                                           \
  extern "C" __global__ void NAME(                                               \
      const INT* __restrict__ weight,                                            \
      const float* __restrict__ rms,                                             \
      float s,                                                                   \
      int rows,                                                                  \
      int cols,                                                                  \
      int vecs_per_row,                                                          \
      int* __restrict__ fields,                                                  \
      int* __restrict__ c_arr,                                                   \
      int* __restrict__ minmax) {                                                \
    e8_quantize_fields_body<INT>(                                                \
        weight, rms, s, rows, cols, vecs_per_row, fields, c_arr, minmax);        \
  }

E8_QUANTIZE_WRAPPER(e8_quantize_fields_kernel, __nv_bfloat16)
E8_QUANTIZE_WRAPPER(e8_quantize_fields_f32_kernel, float)

#undef E8_QUANTIZE_WRAPPER

template <typename InT>
__device__ __forceinline__ void e8_refit_scales_body(
    const InT* __restrict__ weight,
    const int* __restrict__ fields,       // [V,8]: z0..z6, m
    const int* __restrict__ c_arr,        // [V]
    const float* __restrict__ rms,
    float initial_s,
    float* __restrict__ scales,
    float* __restrict__ row_sse,
    int rows,
    int cols,
    int vecs_per_row) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  float numerator = 0.0f;
  float denominator = 0.0f;
  float energy = 0.0f;
  for (int element = threadIdx.x; element < cols; element += blockDim.x) {
    const int vector_in_row = element >> 3;
    const int coordinate = element & 7;
    const int64_t vector = static_cast<int64_t>(row) * vecs_per_row + vector_in_row;
    const int* vector_fields = fields + vector * 8;
    const int c = c_arr[vector];
    int z;
    if (coordinate < 7) {
      z = vector_fields[coordinate];
    } else {
      int parity = 0;
#pragma unroll
      for (int j = 0; j < 7; ++j) parity += vector_fields[j];
      z = 2 * vector_fields[7] + (parity & 1);
    }
    const float point = __fadd_rn(static_cast<float>(z), c ? 0.5f : 0.0f);
    const float value = e8_load_scalar(
        weight + static_cast<int64_t>(row) * cols + element);
    numerator = __fadd_rn(numerator, __fmul_rn(value, point));
    denominator = __fadd_rn(denominator, __fmul_rn(point, point));
    energy = __fadd_rn(energy, __fmul_rn(value, value));
  }

  __shared__ float numerator_shared[256];
  __shared__ float denominator_shared[256];
  __shared__ float energy_shared[256];
  numerator_shared[threadIdx.x] = numerator;
  denominator_shared[threadIdx.x] = denominator;
  energy_shared[threadIdx.x] = energy;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      numerator_shared[threadIdx.x] += numerator_shared[threadIdx.x + offset];
      denominator_shared[threadIdx.x] += denominator_shared[threadIdx.x + offset];
      energy_shared[threadIdx.x] += energy_shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float fallback = __fmul_rn(initial_s, rms[row]);
    const float fitted = denominator_shared[0] > 0.0f
        ? __fdiv_rn(numerator_shared[0], denominator_shared[0])
        : fallback;
    scales[row] = (isfinite(fitted) && fitted > 0.0f) ? fitted : fallback;
    row_sse[row] = denominator_shared[0] > 0.0f
        ? fmaxf(0.0f, energy_shared[0] - numerator_shared[0] * numerator_shared[0] /
            denominator_shared[0])
        : energy_shared[0];
  }
}

#define E8_REFIT_WRAPPER(NAME, INT)                                              \
  extern "C" __global__ void NAME(                                               \
      const INT* __restrict__ weight,                                            \
      const int* __restrict__ fields,                                            \
      const int* __restrict__ c_arr,                                             \
      const float* __restrict__ rms,                                             \
      float initial_s,                                                           \
      float* __restrict__ scales,                                                \
      float* __restrict__ row_sse,                                               \
      int rows,                                                                  \
      int cols,                                                                  \
      int vecs_per_row) {                                                        \
    e8_refit_scales_body<INT>(                                                   \
        weight, fields, c_arr, rms, initial_s, scales, row_sse,                  \
        rows, cols, vecs_per_row);                                               \
  }

E8_REFIT_WRAPPER(e8_refit_scales_kernel, __nv_bfloat16)
E8_REFIT_WRAPPER(e8_refit_scales_f32_kernel, float)

#undef E8_REFIT_WRAPPER

extern "C" __global__ void e8_minmax_fields_kernel(
    const int* __restrict__ fields,
    const int* __restrict__ c_arr,
    int64_t V,
    int* __restrict__ minmax) {
  __shared__ int sm[kMinMaxLen];
  block_minmax_init(sm);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < V; idx += stride) {
    const int c = c_arr[idx];
    atomicMax(&sm[kCMax], c);
    const int* vector_fields = fields + idx * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int stream = j * 2 + c;
      const int value = vector_fields[j];
      atomicMin(&sm[stream * 2], value);
      atomicMax(&sm[stream * 2 + 1], value);
    }
  }
  block_minmax_flush(sm, minmax);
}

extern "C" __global__ void e8_histogram_kernel(
    const int* __restrict__ fields,
    const int* __restrict__ c_arr,
    const int* __restrict__ minmax,
    const int* __restrict__ bin_off,   // [17]: coset (2 bins), then the 16 coord streams
    int* __restrict__ bins,
    int64_t V,
    int shared_bins) {                 // >0: stage in dynamic shared memory of this many ints
  extern __shared__ int sbins[];
  int* target;
  if (shared_bins > 0) {
    for (int t = threadIdx.x; t < shared_bins; t += blockDim.x) sbins[t] = 0;
    __syncthreads();
    target = sbins;
  } else {
    target = bins;
  }

  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < V; idx += stride) {
    const int c = c_arr[idx];
    atomicAdd(&target[bin_off[0] + c], 1);
    const int* f = fields + idx * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int st = j * 2 + c;
      atomicAdd(&target[bin_off[1 + st] + (f[j] - minmax[st * 2])], 1);
    }
  }
  if (shared_bins > 0) {
    __syncthreads();
    for (int t = threadIdx.x; t < shared_bins; t += blockDim.x) {
      if (sbins[t]) atomicAdd(&bins[t], sbins[t]);
    }
  }
}

extern "C" __global__ void e8_rans_encode_vector_kernel(
    const int* __restrict__ fields,
    const int* __restrict__ c_arr,
    const uint16_t* __restrict__ freq_tables,
    const uint16_t* __restrict__ cdfs,
    const int* __restrict__ table_meta,
    uint16_t* __restrict__ scratch,
    uint32_t* __restrict__ word_counts,
    uint32_t* __restrict__ final_states,
    int64_t num_vectors,
    int tile_vectors,
    int num_tiles) {
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  if (tile >= num_tiles) return;
  const int lane = threadIdx.x & 31;
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_vectors;
  const int tile_count = static_cast<int>(
      num_vectors - tile_begin < tile_vectors ? num_vectors - tile_begin : tile_vectors);
  uint16_t* tile_scratch = scratch + static_cast<int64_t>(tile) * tile_vectors * 9;
  uint32_t state = kStateMin;
  uint32_t word_count = 0;
  constexpr uint32_t state_check_mul = 1u << (31 - kProbBits);

  for (int base = 0; base < tile_count; base += kNumStates) {
    const bool valid = base + lane < tile_count;
    const int64_t vector = tile_begin + base + lane;
    const int c = valid ? c_arr[vector] : 0;
#pragma unroll
    for (int field = 7; field >= 0; --field) {
      const int table = 1 + 2 * field + c;
      const uint32_t symbol = valid ? static_cast<uint32_t>(
          fields[vector * 8 + field] - table_meta[table * kMetaStride + kMetaSymMin]) : 0u;
      const int freq_off = valid ? table_meta[table * kMetaStride + kMetaFreqOff] : 0;
      const uint32_t frequency = valid ? freq_tables[freq_off + symbol] : 1u;
      const bool emit = valid && state >= frequency * state_check_mul;
      const uint32_t vote = __ballot_sync(0xffffffffu, emit);
      const uint32_t prefix = __popc(vote & lane_mask_lt());
      if (emit) {
        tile_scratch[word_count + prefix] = static_cast<uint16_t>(state);
        state >>= 16;
      }
      word_count += __popc(vote);
      if (valid) {
        state = (state / frequency) * kTableSize + (state % frequency) +
                cdfs[freq_off + symbol];
      }
    }
    const uint32_t symbol = static_cast<uint32_t>(c);
    const uint32_t frequency = valid ? freq_tables[symbol] : 1u;
    const bool emit = valid && state >= frequency * state_check_mul;
    const uint32_t vote = __ballot_sync(0xffffffffu, emit);
    const uint32_t prefix = __popc(vote & lane_mask_lt());
    if (emit) {
      tile_scratch[word_count + prefix] = static_cast<uint16_t>(state);
      state >>= 16;
    }
    word_count += __popc(vote);
    if (valid) {
      state = (state / frequency) * kTableSize + (state % frequency) + cdfs[symbol];
    }
  }
  final_states[tile * kNumStates + lane] = state;
  if (lane == 0) word_counts[tile] = word_count;
}

extern "C" __global__ void e8_compact_kernel(
    const uint16_t* __restrict__ scratch,
    const uint32_t* __restrict__ offsets,
    uint16_t* __restrict__ payload,
    int tile_elements,
    int num_tiles) {
  const int tile = blockIdx.x;
  if (tile >= num_tiles) return;
  const uint32_t begin = offsets[tile];
  const uint32_t count = offsets[tile + 1] - begin;
  const uint16_t* source = scratch + static_cast<int64_t>(tile) * tile_elements;
  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    payload[begin + i] = source[i];
  }
}

__device__ __forceinline__ uint32_t e8_rans_decode_symbol(
    uint32_t& state, const uint64_t* __restrict__ table) {
  const uint32_t slot = state & kStateMask;
  const uint64_t entry = __ldg(table + slot);
  const uint32_t symbol = static_cast<uint32_t>(entry & 0xffffu);
  uint32_t frequency = static_cast<uint32_t>((entry >> 16) & 0xffffu);
  if (frequency == 0) frequency = kTableSize;
  const uint32_t cdf = static_cast<uint32_t>(entry >> 32);
  state = frequency * (state >> kProbBits) + (slot - cdf);
  return symbol;
}

__device__ __forceinline__ uint32_t e8_decode_coset(
    uint32_t& state, uint32_t frequency0) {
  const uint32_t slot = state & kStateMask;
  const uint32_t symbol = slot >= frequency0;
  const uint32_t begin = symbol ? frequency0 : 0u;
  const uint32_t frequency = symbol ? kTableSize - frequency0 : frequency0;
  state = frequency * (state >> kProbBits) + (slot - begin);
  return symbol;
}

__device__ __forceinline__ uint32_t e8_decode_packed32(
    uint32_t& state,
    const uint32_t* __restrict__ tables,
    const int* __restrict__ pack_bits,
    int table) {
  const uint32_t slot = state & kStateMask;
  const uint32_t entry = __ldg(tables + static_cast<int64_t>(table) * kTableSize + slot);
  const int bits = pack_bits[table];
  const int symbol_bits = bits & 0xff;
  const int frequency_bits = (bits >> 8) & 0xff;
  const uint32_t symbol_mask = (1u << symbol_bits) - 1u;
  const uint32_t frequency_mask = (1u << frequency_bits) - 1u;
  const uint32_t symbol = entry & symbol_mask;
  uint32_t frequency = (entry >> symbol_bits) & frequency_mask;
  if (frequency == 0) frequency = kTableSize;
  const uint32_t delta = entry >> (symbol_bits + frequency_bits);
  state = frequency * (state >> kProbBits) + delta;
  return symbol;
}

enum : int {
  kStoreF32 = 0,
  kStoreF16 = 1,
  kStoreF8E4M3 = 2,
  kStoreF8E5M2 = 3,
  kStoreI8 = 4,
  kStoreI16 = 5,
  kStoreI32 = 6,
  kStoreI64 = 7,
  kStoreU8 = 8,
  kStoreU16 = 9,
  kStoreU32 = 10,
  kStoreU64 = 11,
  kStoreBool = 12,
};

template <typename T> struct e8_int_range;

#define E8_INT_RANGE(TYPE, LOW, HIGH_EXCLUSIVE)          \
  template <> struct e8_int_range<TYPE> {                \
    static constexpr float kLow = LOW;                   \
    static constexpr float kHighExclusive = HIGH_EXCLUSIVE; \
  }

E8_INT_RANGE(int8_t, -128.0f, 128.0f);
E8_INT_RANGE(int16_t, -32768.0f, 32768.0f);
E8_INT_RANGE(int32_t, -2147483648.0f, 2147483648.0f);
E8_INT_RANGE(int64_t, -9223372036854775808.0f, 9223372036854775808.0f);
E8_INT_RANGE(uint8_t, 0.0f, 256.0f);
E8_INT_RANGE(uint16_t, 0.0f, 65536.0f);
E8_INT_RANGE(uint32_t, 0.0f, 4294967296.0f);
E8_INT_RANGE(uint64_t, 0.0f, 18446744073709551616.0f);

#undef E8_INT_RANGE

template <typename T>
__device__ __forceinline__ T e8_snap_integer(float value) {
  const float high = nextafterf(e8_int_range<T>::kHighExclusive, 0.0f);
  const float clamped = fminf(fmaxf(rintf(value), e8_int_range<T>::kLow), high);
  return static_cast<T>(clamped);
}

struct BF16Store {
  using pointer = __nv_bfloat16*;

  __device__ static void put8_words(
      pointer out, int64_t vector, const int* value, float chalf, float scale, int) {
    uint4 packed;
    unsigned* words = reinterpret_cast<unsigned*>(&packed);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float p0 = __fadd_rn(static_cast<float>(value[2 * pair]), chalf);
      const float p1 = __fadd_rn(static_cast<float>(value[2 * pair + 1]), chalf);
      const __nv_bfloat16 lo = __float2bfloat16_rn(__fmul_rn(p0, scale));
      const __nv_bfloat16 hi = __float2bfloat16_rn(__fmul_rn(p1, scale));
      const unsigned lo_bits = reinterpret_cast<const unsigned short*>(&lo)[0];
      const unsigned hi_bits = reinterpret_cast<const unsigned short*>(&hi)[0];
      words[pair] = lo_bits | (hi_bits << 16);
    }
    *reinterpret_cast<uint4*>(out + vector * 8) = packed;
  }

  __device__ static void put8_pair(
      pointer out, int64_t vector, const int* value, float chalf, float scale, int) {
    uint4 packed;
    unsigned* words = reinterpret_cast<unsigned*>(&packed);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float p0 = __fadd_rn(static_cast<float>(value[2 * pair]), chalf);
      const float p1 = __fadd_rn(static_cast<float>(value[2 * pair + 1]), chalf);
      const __nv_bfloat162 pair_bf = __floats2bfloat162_rn(
          __fmul_rn(p0, scale), __fmul_rn(p1, scale));
      words[pair] = *reinterpret_cast<const unsigned*>(&pair_bf);
    }
    *reinterpret_cast<uint4*>(out + vector * 8) = packed;
  }
};

template <int KIND> struct e8_conv;

template <> struct e8_conv<kStoreF32> {
  using type = uint32_t;
  __device__ static uint32_t from(float v) { return __float_as_uint(v); }
};

template <> struct e8_conv<kStoreF16> {
  using type = uint16_t;
  __device__ static uint16_t from(float v) {
    return __half_as_ushort(__float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f)));
  }
};

#define E8_CONV_FP8(K, LIMIT, INTERP)                             \
  template <> struct e8_conv<K> {                                 \
    using type = __nv_fp8_storage_t;                              \
    __device__ static __nv_fp8_storage_t from(float v) {          \
      return __nv_cvt_float_to_fp8(                               \
          fminf(fmaxf(v, -LIMIT), LIMIT), __NV_SATFINITE, INTERP); \
    }                                                             \
  }

E8_CONV_FP8(kStoreF8E4M3, 448.0f, __NV_E4M3);
E8_CONV_FP8(kStoreF8E5M2, 57344.0f, __NV_E5M2);

#undef E8_CONV_FP8

#define E8_CONV_INT(K, TYPE)                                    \
  template <> struct e8_conv<K> {                               \
    using type = TYPE;                                          \
    __device__ static TYPE from(float v) {                      \
      return e8_snap_integer<TYPE>(v);                          \
    }                                                           \
  }

E8_CONV_INT(kStoreI8, int8_t);
E8_CONV_INT(kStoreI16, int16_t);
E8_CONV_INT(kStoreI32, int32_t);
E8_CONV_INT(kStoreI64, int64_t);
E8_CONV_INT(kStoreU8, uint8_t);
E8_CONV_INT(kStoreU16, uint16_t);
E8_CONV_INT(kStoreU32, uint32_t);
E8_CONV_INT(kStoreU64, uint64_t);

#undef E8_CONV_INT

template <> struct e8_conv<kStoreBool> {
  using type = bool;
  __device__ static bool from(float v) { return rintf(v) != 0.0f; }
};

// Store count, not conversion cost, dominates here, so the eight converted values are packed into words and written with as
// few wide stores as the container allows. Only 8-byte containers store scalar: one vector of them is wider than a uint4.
template <typename Conv>
__device__ __forceinline__ void e8_store8(void* out, int64_t vector, const float* v) {
  using T = typename Conv::type;
  constexpr int kBytes = 8 * static_cast<int>(sizeof(T));
  char* base = static_cast<char*>(out) + vector * kBytes;
  if constexpr (sizeof(T) == 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) reinterpret_cast<T*>(base)[j] = Conv::from(v[j]);
  } else {
    constexpr int kWords = kBytes / 4;
    constexpr int kPerWord = 4 / static_cast<int>(sizeof(T));
    unsigned words[kWords];
#pragma unroll
    for (int word = 0; word < kWords; ++word) {
      unsigned packed = 0;
#pragma unroll
      for (int slot = 0; slot < kPerWord; ++slot) {
        const unsigned bits = static_cast<unsigned>(
            sizeof(T) == 1 ? static_cast<uint8_t>(Conv::from(v[word * kPerWord + slot]))
            : sizeof(T) == 2 ? static_cast<uint16_t>(Conv::from(v[word * kPerWord + slot]))
                             : static_cast<uint32_t>(Conv::from(v[word * kPerWord + slot])));
        packed |= bits << (8 * static_cast<int>(sizeof(T)) * slot);
      }
      words[word] = packed;
    }
    if constexpr (kWords == 2) {
      *reinterpret_cast<uint2*>(base) = make_uint2(words[0], words[1]);
    } else {
      *reinterpret_cast<uint4*>(base) = make_uint4(words[0], words[1], words[2], words[3]);
      if constexpr (kWords == 8) {
        *reinterpret_cast<uint4*>(base + 16) =
            make_uint4(words[4], words[5], words[6], words[7]);
      }
    }
  }
}

struct GenericStore {
  using pointer = void*;

  __device__ static void put8(
      pointer out, int64_t vector, const int* value, float chalf, float scale, int kind) {
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = __fmul_rn(__fadd_rn(static_cast<float>(value[j]), chalf), scale);
    }
    switch (kind) {
      case kStoreF32: e8_store8<e8_conv<kStoreF32>>(out, vector, v); break;
      case kStoreF16: e8_store8<e8_conv<kStoreF16>>(out, vector, v); break;
      case kStoreF8E4M3: e8_store8<e8_conv<kStoreF8E4M3>>(out, vector, v); break;
      case kStoreF8E5M2: e8_store8<e8_conv<kStoreF8E5M2>>(out, vector, v); break;
      case kStoreI8: e8_store8<e8_conv<kStoreI8>>(out, vector, v); break;
      case kStoreI16: e8_store8<e8_conv<kStoreI16>>(out, vector, v); break;
      case kStoreI32: e8_store8<e8_conv<kStoreI32>>(out, vector, v); break;
      case kStoreI64: e8_store8<e8_conv<kStoreI64>>(out, vector, v); break;
      case kStoreU8: e8_store8<e8_conv<kStoreU8>>(out, vector, v); break;
      case kStoreU16: e8_store8<e8_conv<kStoreU16>>(out, vector, v); break;
      case kStoreU32: e8_store8<e8_conv<kStoreU32>>(out, vector, v); break;
      case kStoreU64: e8_store8<e8_conv<kStoreU64>>(out, vector, v); break;
      default: e8_store8<e8_conv<kStoreBool>>(out, vector, v); break;
    }
  }

  __device__ static void put8_words(
      pointer out, int64_t vector, const int* value, float chalf, float scale, int kind) {
    put8(out, vector, value, chalf, scale, kind);
  }

  __device__ static void put8_pair(
      pointer out, int64_t vector, const int* value, float chalf, float scale, int kind) {
    put8(out, vector, value, chalf, scale, kind);
  }
};

template <typename Store>
__device__ __forceinline__ void e8_decode_vector_fused_body(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint64_t* __restrict__ decode_luts,
    const int* __restrict__ table_meta,
    const float* __restrict__ scales,
    typename Store::pointer __restrict__ output,
    int store_kind,
    int vecs_per_row,
    int64_t num_vectors,
    int tile_elements,
    int num_tiles,
    int* __restrict__ error) {
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  if (tile >= num_tiles) return;
  const int lane = threadIdx.x & 31;
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_elements;
  const int tile_count = static_cast<int>(
      num_vectors - tile_begin < tile_elements ? num_vectors - tile_begin : tile_elements);

  const uint16_t* input_begin = payload + offsets[tile];
  const uint16_t* input = payload + offsets[tile + 1];
  uint32_t state = states[tile * kNumStates + lane];

  const int remainder = tile_count & (kNumStates - 1);
  int output_offset = tile_count - remainder;
  bool first = true;
  while (first || output_offset > 0) {
    int valid_lanes;
    if (first && remainder) {
      valid_lanes = remainder;
    } else {
      if (first) output_offset = tile_count;
      output_offset -= kNumStates;
      valid_lanes = kNumStates;
    }
    first = false;
    const bool valid = lane < valid_lanes;
    int c = 0;
    int value[8];
    if (valid) c = static_cast<int>(e8_rans_decode_symbol(state, decode_luts));
    if (!rans_renormalize_checked(valid, state, input, input_begin)) set_error(error, 2);
#pragma unroll
    for (int field = 0; field < 8; ++field) {
      if (valid) {
        const int table = 1 + 2 * field + c;
        const uint64_t* lut = decode_luts + static_cast<int64_t>(table) * kTableSize;
        value[field] = static_cast<int>(e8_rans_decode_symbol(state, lut)) +
                       table_meta[table * kMetaStride + kMetaSymMin];
      }
      if (!rans_renormalize_checked(valid, state, input, input_begin)) set_error(error, 2);
    }
    if (valid) {
      int parity = 0;
#pragma unroll
      for (int field = 0; field < 7; ++field) parity += value[field];
      value[7] = 2 * value[7] + (parity & 1);
      const int64_t vector = tile_begin + output_offset + lane;
      const int row = div_i64_i32(vector, vecs_per_row);
      Store::put8_words(output, vector, value, c ? 0.5f : 0.0f, scales[row], store_kind);
    }
    if (output_offset == 0) break;
  }
  if (input != input_begin || state != kStateMin) set_error(error, 3);
}

#define E8_FUSED_WRAPPER(NAME)                                                   \
  extern "C" __global__ void NAME(                                               \
      const uint16_t* __restrict__ payload,                                      \
      const uint32_t* __restrict__ offsets,                                      \
      const uint32_t* __restrict__ states,                                       \
      const uint64_t* __restrict__ decode_luts,                                  \
      const int* __restrict__ table_meta,                                        \
      const float* __restrict__ scales,                                          \
      __nv_bfloat16* __restrict__ output,                                        \
      int vecs_per_row,                                                          \
      int64_t num_vectors,                                                       \
      int tile_elements,                                                         \
      int num_tiles,                                                             \
      int* __restrict__ error) {                                                 \
    e8_decode_vector_fused_body<BF16Store>(                                      \
        payload, offsets, states, decode_luts, table_meta, scales,               \
        output, 0, vecs_per_row, num_vectors, tile_elements, num_tiles,          \
        error);                                                                  \
  }

#define E8_FUSED_GENERIC_WRAPPER(NAME)                                           \
  extern "C" __global__ void NAME(                                               \
      const uint16_t* __restrict__ payload,                                      \
      const uint32_t* __restrict__ offsets,                                      \
      const uint32_t* __restrict__ states,                                       \
      const uint64_t* __restrict__ decode_luts,                                  \
      const int* __restrict__ table_meta,                                        \
      const float* __restrict__ scales,                                          \
      void* __restrict__ output,                                                 \
      int vecs_per_row,                                                          \
      int64_t num_vectors,                                                       \
      int tile_elements,                                                         \
      int num_tiles,                                                             \
      int* __restrict__ error,                                                   \
      int store_kind) {                                                          \
    e8_decode_vector_fused_body<GenericStore>(                                   \
        payload, offsets, states, decode_luts, table_meta, scales,               \
        output, store_kind, vecs_per_row, num_vectors, tile_elements,            \
        num_tiles, error);                                                       \
  }

E8_FUSED_WRAPPER(e8_decode_vector_fused_kernel)
E8_FUSED_GENERIC_WRAPPER(e8_decode_vector_fused_g_kernel)

#undef E8_FUSED_WRAPPER
#undef E8_FUSED_GENERIC_WRAPPER

template <bool PREF, typename Store>
__device__ __forceinline__ void e8_decode_packed32_body(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint32_t* __restrict__ decode_luts,
    const int* __restrict__ pack_bits,
    uint32_t coset_frequency0,
    const int* __restrict__ table_meta,
    const float* __restrict__ scales,
    typename Store::pointer __restrict__ output,
    int store_kind,
    int vecs_per_row,
    int64_t num_vectors,
    int tile_vectors,
    int num_tiles,
    int* __restrict__ error) {
  __shared__ int pack_bits_shared[17];
  __shared__ int sym_min_shared[17];
  if (threadIdx.x < 17) {
    pack_bits_shared[threadIdx.x] = pack_bits[threadIdx.x];
    sym_min_shared[threadIdx.x] = table_meta[threadIdx.x * kMetaStride + kMetaSymMin];
  }
  __syncthreads();
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  if (tile >= num_tiles) return;
  const int lane = threadIdx.x & 31;
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_vectors;
  const int tile_count = static_cast<int>(
      num_vectors - tile_begin < tile_vectors ? num_vectors - tile_begin : tile_vectors);
  const uint16_t* input_begin = payload + offsets[tile];
  const uint16_t* input = payload + offsets[tile + 1];
  uint32_t state = states[tile * kNumStates + lane];
  const int remainder = tile_count & (kNumStates - 1);
  int output_offset = tile_count - remainder;
  bool first = true;
  while (first || output_offset > 0) {
    int valid_lanes;
    if (first && remainder) {
      valid_lanes = remainder;
    } else {
      if (first) output_offset = tile_count;
      output_offset -= kNumStates;
      valid_lanes = kNumStates;
    }
    first = false;
    const bool valid = lane < valid_lanes;
    if constexpr (PREF) {
      // The renorm rate grows with the coded rate, so at high coded rates the cold-payload latency matters more.
      if (lane < 4) {
        const char* p = reinterpret_cast<const char*>(input) - 128 * (lane + 1);
        if (p >= reinterpret_cast<const char*>(payload)) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
      }
    }
    int c = 0;
    int value[8];
    if (valid) c = static_cast<int>(e8_decode_coset(state, coset_frequency0));
    if (!rans_renormalize_checked(valid, state, input, input_begin)) set_error(error, 2);
#pragma unroll
    for (int field = 0; field < 8; ++field) {
      if (valid) {
        const int table = 1 + 2 * field + c;
        value[field] = static_cast<int>(
            e8_decode_packed32(state, decode_luts, pack_bits_shared, table)) +
            sym_min_shared[table];
      }
      if (!rans_renormalize_checked(valid, state, input, input_begin)) set_error(error, 2);
    }
    if (valid) {
      int parity = 0;
#pragma unroll
      for (int field = 0; field < 7; ++field) parity += value[field];
      value[7] = 2 * value[7] + (parity & 1);
      const int64_t vector = tile_begin + output_offset + lane;
      const int c_row = div_i64_i32(vector, vecs_per_row);
      Store::put8_words(output, vector, value, c ? 0.5f : 0.0f, scales[c_row], store_kind);
    }
    if (output_offset == 0) break;
  }
  if (input != input_begin || state != kStateMin) set_error(error, 3);
}

#define E8_PACKED32_WRAPPER(NAME, PREF)                                          \
  extern "C" __global__ void NAME(                                               \
      const uint16_t* __restrict__ payload,                                      \
      const uint32_t* __restrict__ offsets,                                      \
      const uint32_t* __restrict__ states,                                       \
      const uint32_t* __restrict__ decode_luts,                                  \
      const int* __restrict__ pack_bits,                                         \
      uint32_t coset_frequency0,                                                 \
      const int* __restrict__ table_meta,                                        \
      const float* __restrict__ scales,                                          \
      __nv_bfloat16* __restrict__ output,                                        \
      int vecs_per_row,                                                          \
      int64_t num_vectors,                                                       \
      int tile_vectors,                                                          \
      int num_tiles,                                                             \
      int* __restrict__ error) {                                                 \
    e8_decode_packed32_body<PREF, BF16Store>(                                    \
        payload, offsets, states, decode_luts, pack_bits, coset_frequency0,      \
        table_meta, scales, output, 0, vecs_per_row, num_vectors,                \
        tile_vectors, num_tiles, error);                                         \
  }

#define E8_PACKED32_GENERIC_WRAPPER(NAME, PREF)                                  \
  extern "C" __global__ void NAME(                                               \
      const uint16_t* __restrict__ payload,                                      \
      const uint32_t* __restrict__ offsets,                                      \
      const uint32_t* __restrict__ states,                                       \
      const uint32_t* __restrict__ decode_luts,                                  \
      const int* __restrict__ pack_bits,                                         \
      uint32_t coset_frequency0,                                                 \
      const int* __restrict__ table_meta,                                        \
      const float* __restrict__ scales,                                          \
      void* __restrict__ output,                                                 \
      int vecs_per_row,                                                          \
      int64_t num_vectors,                                                       \
      int tile_vectors,                                                          \
      int num_tiles,                                                             \
      int* __restrict__ error,                                                   \
      int store_kind) {                                                          \
    e8_decode_packed32_body<PREF, GenericStore>(                                 \
        payload, offsets, states, decode_luts, pack_bits, coset_frequency0,      \
        table_meta, scales, output, store_kind, vecs_per_row,                    \
        num_vectors, tile_vectors, num_tiles, error);                            \
  }

E8_PACKED32_WRAPPER(e8_decode_vector_packed32_kernel, false)
E8_PACKED32_WRAPPER(e8_decode_vector_packed32pf_kernel, true)
E8_PACKED32_GENERIC_WRAPPER(e8_decode_vector_packed32_g_kernel, false)
E8_PACKED32_GENERIC_WRAPPER(e8_decode_vector_packed32pf_g_kernel, true)

#undef E8_PACKED32_WRAPPER
#undef E8_PACKED32_GENERIC_WRAPPER

// The global-LUT path gathers a 32-lane random table through L1, which a table larger than the cache does not stay in; staging
// two compact tables in dynamic shared memory is faster while the SM still holds enough resident CTAs.
template <typename Store>
__device__ __forceinline__ void e8_decode_shlut_body(
    const uint16_t* __restrict__ payload,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ states,
    const uint8_t* __restrict__ sym_lut,     // [17 * kTableSize]
    const uint32_t* __restrict__ fb_lut,     // [fb_entries] begin | freq<<16
    uint32_t coset_frequency0,
    const int* __restrict__ table_meta,      // [17, kMetaStride]: see kMetaFreqOff, kMetaSymMin
    const float* __restrict__ scales,
    typename Store::pointer __restrict__ output,
    int store_kind,
    int vecs_per_row,
    int64_t num_vectors,
    int tile_vectors,
    int num_tiles,
    int fb_entries,
    int* __restrict__ error) {
  extern __shared__ unsigned char shmem_raw[];
  uint8_t* sym_sh = reinterpret_cast<uint8_t*>(shmem_raw);
  uint32_t* fb_sh = reinterpret_cast<uint32_t*>(sym_sh + 17 * kTableSize);
  __shared__ int meta_sh[34];  // [0,17): freq_off, [17,34): sym_min
  {
    const int n4 = 17 * kTableSize / 16;  // uint8 slots, staged as uint4
    const uint4* src4 = reinterpret_cast<const uint4*>(sym_lut);
    uint4* dst4 = reinterpret_cast<uint4*>(sym_sh);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst4[i] = src4[i];
    for (int i = threadIdx.x; i < fb_entries; i += blockDim.x) fb_sh[i] = fb_lut[i];
    if (threadIdx.x < 17) {
      meta_sh[threadIdx.x] = table_meta[threadIdx.x * kMetaStride + kMetaFreqOff];
      meta_sh[threadIdx.x + 17] = table_meta[threadIdx.x * kMetaStride + kMetaSymMin];
    }
  }
  __syncthreads();

  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int tile = blockIdx.x * warps_per_block + warp_in_block;
  if (tile >= num_tiles) return;
  const int lane = threadIdx.x & 31;
  const uint32_t mask_ge = lane_mask_ge();
  const int64_t tile_begin = static_cast<int64_t>(tile) * tile_vectors;
  const int tile_count = static_cast<int>(
      num_vectors - tile_begin < tile_vectors ? num_vectors - tile_begin : tile_vectors);
  const uint16_t* input_begin = payload + offsets[tile];
  const uint16_t* input = payload + offsets[tile + 1];
  uint32_t state = states[tile * kNumStates + lane];
  const int remainder = tile_count & (kNumStates - 1);
  int output_offset = tile_count - remainder;
  bool first = true;
  while (first || output_offset > 0) {
    int valid_lanes;
    if (first && remainder) {
      valid_lanes = remainder;
    } else {
      if (first) output_offset = tile_count;
      output_offset -= kNumStates;
      valid_lanes = kNumStates;
    }
    first = false;
    const bool valid = lane < valid_lanes;
    if (lane < 4) {
      const char* p = reinterpret_cast<const char*>(input) - 128 * (lane + 1);
      if (p >= reinterpret_cast<const char*>(payload)) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
      }
    }
    int c = 0;
    int value[8];
    if (valid) c = static_cast<int>(e8_decode_coset(state, coset_frequency0));
    rans_renormalize_unchecked(valid, state, input, mask_ge);
    const int cTS = c * kTableSize;
#pragma unroll
    for (int field = 0; field < 8; ++field) {
      if (valid) {
        const uint32_t slot = state & kStateMask;
        const uint32_t sym = static_cast<uint32_t>(
            sym_sh[(2 * field + 1) * kTableSize + cTS + slot]);
        const int table = (2 * field + 1) + c;
        const uint32_t fb = fb_sh[meta_sh[table] + sym];
        const uint32_t begin = fb & 0xFFFFu;
        const uint32_t freq = fb >> 16;
        state = freq * (state >> kProbBits) + (slot - begin);
        value[field] = static_cast<int>(sym) + meta_sh[table + 17];
      }
      rans_renormalize_unchecked(valid, state, input, mask_ge);
    }
    if (valid) {
      int parity = 0;
#pragma unroll
      for (int field = 0; field < 7; ++field) parity += value[field];
      value[7] = 2 * value[7] + (parity & 1);
      const int64_t vector = tile_begin + output_offset + lane;
      const int c_row = div_i64_i32(vector, vecs_per_row);
      const float chalf = c ? 0.5f : 0.0f;
      Store::put8_pair(output, vector, value, chalf, scales[c_row], store_kind);
    }
    if (output_offset == 0) break;
  }
  if (input != input_begin || state != kStateMin) set_error(error, 3);
}

#define E8_SHLUT_WRAPPER(NAME)                                             \
  extern "C" __global__ void NAME(                                               \
      const uint16_t* __restrict__ payload,                                      \
      const uint32_t* __restrict__ offsets,                                      \
      const uint32_t* __restrict__ states,                                       \
      const uint8_t* __restrict__ sym_lut,                                       \
      const uint32_t* __restrict__ fb_lut,                                       \
      uint32_t coset_frequency0,                                                 \
      const int* __restrict__ table_meta,                                        \
      const float* __restrict__ scales,                                          \
      __nv_bfloat16* __restrict__ output,                                        \
      int vecs_per_row,                                                          \
      int64_t num_vectors,                                                       \
      int tile_vectors,                                                          \
      int num_tiles,                                                             \
      int fb_entries,                                                            \
      int* __restrict__ error) {                                                 \
    e8_decode_shlut_body<BF16Store>(                                             \
        payload, offsets, states, sym_lut, fb_lut, coset_frequency0, table_meta, \
        scales, output, 0, vecs_per_row, num_vectors, tile_vectors,              \
        num_tiles, fb_entries, error);                                           \
  }

#define E8_SHLUT_GENERIC_WRAPPER(NAME)                                     \
  extern "C" __global__ void NAME(                                               \
      const uint16_t* __restrict__ payload,                                      \
      const uint32_t* __restrict__ offsets,                                      \
      const uint32_t* __restrict__ states,                                       \
      const uint8_t* __restrict__ sym_lut,                                       \
      const uint32_t* __restrict__ fb_lut,                                       \
      uint32_t coset_frequency0,                                                 \
      const int* __restrict__ table_meta,                                        \
      const float* __restrict__ scales,                                          \
      void* __restrict__ output,                                                 \
      int vecs_per_row,                                                          \
      int64_t num_vectors,                                                       \
      int tile_vectors,                                                          \
      int num_tiles,                                                             \
      int fb_entries,                                                            \
      int* __restrict__ error,                                                   \
      int store_kind) {                                                          \
    e8_decode_shlut_body<GenericStore>(                                          \
        payload, offsets, states, sym_lut, fb_lut, coset_frequency0, table_meta, \
        scales, output, store_kind, vecs_per_row, num_vectors,                   \
        tile_vectors, num_tiles, fb_entries, error);                             \
  }

E8_SHLUT_WRAPPER(e8_decode_vector_shlut8pf_kernel)
E8_SHLUT_GENERIC_WRAPPER(e8_decode_vector_shlut8pf_g_kernel)

#undef E8_SHLUT_WRAPPER
#undef E8_SHLUT_GENERIC_WRAPPER

