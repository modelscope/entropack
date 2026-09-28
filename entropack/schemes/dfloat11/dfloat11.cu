// EntroPack -- DFloat11 lossless bf16 compression: CUDA kernels.
//
// Decode kernel:
//   dfloat11_decode_kernel              one thread per bitstream region: walks the multi-level Huffman LUTs to recover the
//                                       exponents, stages them in shared memory, and writes the reconstructed bf16 back
//                                       coalesced. __launch_bounds__ come from -D macros the host derives from the queried
//                                       device.
//
// Encode kernels, in pipeline order:
//   dfloat11_exponent_histogram_kernel  256-bin bf16 exponent counts, accumulated in shared memory per block.
//   dfloat11_split_len_kernel           splits bf16 into exponent and sign/mantissa bytes and looks up each exponent's
//                                       Huffman code length in the same pass.
//   dfloat11_pack_kernel                writes the MSB-first bitstream, one thread per short output chunk.
//   dfloat11_thread_meta_kernel         per-region gap and symbol count, as the checkpoint's uint16 thread_meta.
//   dfloat11_output_positions_kernel    per-block starting element index.
//
// The device helpers below serve both directions: MSB-first bit windows, the multi-level LUT walk, and a block-wide
// exclusive scan. Python orchestration lives in cuda.py; format.py is authoritative for the buffer layout and for the
// decode granularity recorded in each checkpoint. NVRTC has no system include path, so libcudacxx supplies the
// uint8_t/uint32_t/int64_t typedefs that <cstdint> would.
#include <cuda/std/cstdint>

constexpr int kMetaCountBits = 11;
constexpr uint32_t kMetaCountMask = (1u << kMetaCountBits) - 1u;

#ifndef DFLOAT11_THREADS_PER_BLOCK
#define DFLOAT11_THREADS_PER_BLOCK 128
#endif
#ifndef DFLOAT11_MIN_BLOCKS_PER_SM
#define DFLOAT11_MIN_BLOCKS_PER_SM 1
#endif


__device__ __forceinline__ uint32_t read_byte_msb(
    const uint8_t* __restrict__ data, int64_t n_bytes, int64_t bit_pos) {
  const int64_t byte_idx = bit_pos >> 3;
  const uint32_t shift = static_cast<uint32_t>(bit_pos & 7);
  const uint32_t hi = (byte_idx < n_bytes) ? data[byte_idx] : 0u;
  if (shift == 0u) {
    return hi;
  }
  const uint32_t lo = (byte_idx + 1 < n_bytes) ? data[byte_idx + 1] : 0u;
  return ((hi << shift) | (lo >> (8u - shift))) & 0xFFu;
}

__device__ __forceinline__ uint32_t decode_symbol(
    const uint8_t* __restrict__ luts,
    int num_luts,  // == num_levels + 1
    const uint8_t* __restrict__ encoded,
    int64_t n_bytes,
    int64_t bit_pos,
    uint32_t* code_len) {
  const int num_levels = num_luts - 1;
  const uint32_t ptr_min =
      (num_levels > 1) ? static_cast<uint32_t>(256 - (num_levels - 1)) : 256u;
  const uint8_t* lens_row = luts + static_cast<int64_t>(num_levels) * 256;

  int level = 0;
  int hop = 0;
  for (;;) {
    const uint32_t byte = read_byte_msb(encoded, n_bytes, bit_pos + hop * 8);
    const uint32_t entry = luts[static_cast<int64_t>(level) * 256 + byte];
    if (num_levels > 1 && entry >= ptr_min) {
      level = 256 - static_cast<int>(entry);  // pointer -> child level
      ++hop;
    } else {
      *code_len = lens_row[entry];  // leaf symbol
      return entry;
    }
  }
}

__device__ __forceinline__ uint16_t make_bf16_bits(uint32_t exponent, uint32_t sign_mantissa) {
  return static_cast<uint16_t>(
      ((sign_mantissa & 0x80u) << 8) | (exponent << 7) | (sign_mantissa & 0x7Fu));
}

__device__ __forceinline__ uint64_t pack_bf16x4(uint32_t e32, uint32_t s32) {
  const uint32_t lo = ((e32 & 0x01010101u) << 7) | (s32 & 0x7F7F7F7Fu);
  const uint32_t hi = (s32 & 0x80808080u) | ((e32 >> 1) & 0x7F7F7F7Fu);
  uint32_t w0, w1;
  asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(w0) : "r"(lo), "r"(hi));
  asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(w1) : "r"(lo), "r"(hi));
  return static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32);
}

// A sliding MSB-first bit window held in registers. `decode_symbol` re-reads the stream for every symbol, and again for each
// LUT hop because a codeword is not byte-aligned. Walking a contiguous run instead keeps the next bits in `buf` and refills a
// byte at a time, so the hot loop touches the stream once per byte consumed.
//
// The `BitWindowLocal` variant stages the block's slice in shared memory, zero-padded and sized for the worst-case lookahead,
// so the generic refill's bounds check is never taken and is dropped, and addressing shrinks to 32 bit.
struct BitWindow {
  const uint8_t* __restrict__ data;
  int64_t n_bytes;
  int64_t byte_pos;  // next byte to pull into `buf`
  uint64_t buf;
  int n_valid;
};

__device__ __forceinline__ void bit_window_refill(BitWindow* w) {
  while (w->n_valid <= 32) {
    const uint32_t byte = (w->byte_pos < w->n_bytes) ? w->data[w->byte_pos] : 0u;
    w->buf |= static_cast<uint64_t>(byte) << (56 - w->n_valid);
    w->n_valid += 8;
    ++w->byte_pos;
  }
}

__device__ __forceinline__ BitWindow bit_window_open(
    const uint8_t* __restrict__ data, int64_t n_bytes, int64_t bit_pos) {
  BitWindow w;
  w.data = data;
  w.n_bytes = n_bytes;
  w.byte_pos = bit_pos >> 3;
  w.buf = 0;
  w.n_valid = 0;
  bit_window_refill(&w);
  const int skip = static_cast<int>(bit_pos & 7);  // land on the first bit of the codeword
  w.buf <<= skip;
  w.n_valid -= skip;
  bit_window_refill(&w);
  return w;
}

struct LutWalk {
  const uint8_t* luts;     // staged decode rows
  const uint8_t* lens_row; // per-symbol code length row
  uint32_t ptr_min;        // entries >= ptr_min are pointers, below are leaf symbols
  int num_levels;
};

__device__ __forceinline__ uint32_t bit_window_decode(
    BitWindow* w,
    const uint8_t* __restrict__ luts,
    int num_luts) {  // == num_levels + 1
  const int num_levels = num_luts - 1;
  const uint32_t ptr_min =
      (num_levels > 1) ? static_cast<uint32_t>(256 - (num_levels - 1)) : 256u;
  const uint8_t* lens_row = luts + static_cast<int64_t>(num_levels) * 256;

  int level = 0;
  int shift = 56;  // peek the byte `hop` bytes into the window without consuming it
  for (;;) {
    const uint32_t byte = static_cast<uint32_t>((w->buf >> shift) & 0xFFu);
    const uint32_t entry = luts[static_cast<int64_t>(level) * 256 + byte];
    if (num_levels > 1 && entry >= ptr_min) {
      level = 256 - static_cast<int>(entry);  // pointer -> child level
      shift -= 8;
    } else {
      const int code_len = lens_row[entry];  // length of the whole codeword
      w->buf <<= code_len;
      w->n_valid -= code_len;
      bit_window_refill(w);
      return entry;
    }
  }
}

struct BitWindowLocal {
  const uint8_t* data;
  int byte_pos;
  uint64_t buf;
  int n_valid;
};

__device__ __forceinline__ void bwl_refill(BitWindowLocal* w) {
  while (w->n_valid <= 32) {
    w->buf |= static_cast<uint64_t>(w->data[w->byte_pos]) << (56 - w->n_valid);
    w->n_valid += 8;
    ++w->byte_pos;
  }
}

__device__ __forceinline__ BitWindowLocal bwl_open(const uint8_t* data, int bit_pos) {
  BitWindowLocal w;
  w.data = data;
  w.byte_pos = bit_pos >> 3;
  w.buf = 0;
  w.n_valid = 0;
  bwl_refill(&w);
  const int skip = bit_pos & 7;
  w.buf <<= skip;
  w.n_valid -= skip;
  bwl_refill(&w);
  return w;
}

__device__ __forceinline__ uint32_t bwl_decode(BitWindowLocal* w, const LutWalk* ctx) {
  int level = 0;
  int shift = 56;
  for (;;) {
    const uint32_t byte = static_cast<uint32_t>((w->buf >> shift) & 0xFFu);
    const uint32_t entry = ctx->luts[level * 256 + byte];
    if (ctx->num_levels > 1 && entry >= ctx->ptr_min) {
      level = 256 - static_cast<int>(entry);
      shift -= 8;
    } else {
      const int code_len = ctx->lens_row[entry];
      w->buf <<= code_len;
      w->n_valid -= code_len;
      bwl_refill(w);
      return entry;
    }
  }
}

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
  uint32_t r;
  asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(r) : "r"(x));
  return r;
}

// Register-resident bitstream for bytes_per_thread == 16: a thread's whole span, its region plus the worst-case lookahead, is
// preloaded into three big-endian u64 words, so the walk issues no stream loads, only funnel shifts.
struct RegStream {
  uint64_t w0, w1, w2;
  int bit;
};

__device__ __forceinline__ RegStream reg_stream_open(const uint8_t* span16, int gap_bits) {
  const uint4 q = *reinterpret_cast<const uint4*>(span16);
  const uint2 r = *reinterpret_cast<const uint2*>(span16 + 16);
  RegStream s;
  s.w0 = (static_cast<uint64_t>(bswap32(q.x)) << 32) | bswap32(q.y);
  s.w1 = (static_cast<uint64_t>(bswap32(q.z)) << 32) | bswap32(q.w);
  s.w2 = (static_cast<uint64_t>(bswap32(r.x)) << 32) | bswap32(r.y);
  s.bit = gap_bits;
  return s;
}

__device__ __forceinline__ uint32_t rs_peek32(const RegStream* s) {
  const int b = s->bit;
  const uint64_t v = (s->w0 << b) | (b ? (s->w1 >> (64 - b)) : 0ull);
  return static_cast<uint32_t>(v >> 32);
}

__device__ __forceinline__ uint32_t rs_decode(RegStream* s, const LutWalk* ctx) {
  uint32_t peek = rs_peek32(s);
  int lvl_off = 0;
  for (;;) {
    const uint32_t byte = peek >> 24;
    const uint32_t entry = ctx->luts[lvl_off + byte];
    if (entry >= ctx->ptr_min) {  // pointer -> child level (ptr_min == 256: never true)
      lvl_off = (256u - entry) << 8;
      peek <<= 8;
    } else {
      const int code_len = ctx->lens_row[entry];
      s->bit += code_len;
      if (s->bit >= 64) {  // rotate the word window; bit + code_len < 95 < 128, once is enough
        s->w0 = s->w1;
        s->w1 = s->w2;
        s->w2 = 0ull;
        s->bit -= 64;
      }
      return entry;
    }
  }
}

// Warp-shuffle scan per warp, then a scan of the warp totals: two barriers instead of a full Hillis-Steele sweep over the
// block.
__device__ __forceinline__ int32_t block_exclusive_scan(int32_t* s_scan, int32_t value) {
  if ((blockDim.x & 31u) == 0u && blockDim.x >= 32u) {
    const unsigned full = 0xFFFFFFFFu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int nwarps = static_cast<int>(blockDim.x) >> 5;

    int32_t v = value;  // inclusive scan within the warp
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int32_t n = __shfl_up_sync(full, v, off);
      if (lane >= off) v += n;
    }
    if (lane == 31) {
      s_scan[warp_id] = v;
    }
    __syncthreads();

    if (warp_id == 0) {
      const unsigned wmask = (nwarps == 32) ? full : ((1u << nwarps) - 1u);
      int32_t w = (lane < nwarps) ? s_scan[lane] : 0;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int32_t n = __shfl_up_sync(wmask, w, off);
        if (lane >= off && lane < nwarps) w += n;
      }
      if (lane < nwarps) {
        s_scan[lane] = w;
      }
    }
    __syncthreads();

    const int32_t prefix = (warp_id > 0) ? s_scan[warp_id - 1] : 0;
    return prefix + (v - value);
  }

  s_scan[threadIdx.x] = value;
  __syncthreads();
  for (unsigned offset = 1; offset < blockDim.x; offset <<= 1) {
    const int32_t addend = (threadIdx.x >= offset) ? s_scan[threadIdx.x - offset] : 0;
    __syncthreads();  // every read of round `offset` completes before any write
    if (threadIdx.x >= offset) {
      s_scan[threadIdx.x] += addend;
    }
    __syncthreads();
  }
  return s_scan[threadIdx.x] - value;  // inclusive -> exclusive
}


extern "C" __global__ void __launch_bounds__(DFLOAT11_THREADS_PER_BLOCK, DFLOAT11_MIN_BLOCKS_PER_SM) dfloat11_decode_kernel(
    const uint8_t* __restrict__ luts,
    const uint8_t* __restrict__ encoded,
    const uint8_t* __restrict__ sign_mantissa,
    const uint32_t* __restrict__ output_positions,
    const uint16_t* __restrict__ thread_meta,
    uint16_t* __restrict__ out_bits,  // bf16 bit patterns
    int num_luts,
    int64_t n_bytes,
    int64_t n_elements,
    int bytes_per_thread,
    int stage_enc_bytes,   // 0 = read the bitstream straight from global memory
    int stage_elems,       // 0 = scatter to out_bits directly instead of staging
    int sm_u32_aligned) {  // 1 = sign_mantissa pointer is 4-byte aligned (vector write-back)
  extern __shared__ __align__(16) uint8_t s_raw[];
  int32_t* const s_scan = reinterpret_cast<int32_t*>(s_raw);
  uint8_t* const s_luts = s_raw + blockDim.x * sizeof(int32_t);
  uint8_t* const s_enc = s_luts + static_cast<int64_t>(num_luts) * 256;

  const int64_t out_base = static_cast<int64_t>(output_positions[blockIdx.x]);
  uint8_t* const s_exp = s_enc + stage_enc_bytes + (out_base & 3);

  const int lut_bytes = num_luts * 256;
  for (int i = threadIdx.x; i < lut_bytes; i += blockDim.x) {
    s_luts[i] = luts[i];
  }

  const int64_t block_byte_base =
      static_cast<int64_t>(blockIdx.x) * blockDim.x * bytes_per_thread;
  const bool direct_ok = (bytes_per_thread == 16) && (stage_enc_bytes > 0) &&
                         (block_byte_base + stage_enc_bytes <= n_bytes) &&
                         ((reinterpret_cast<uint64_t>(encoded + block_byte_base) & 15u) == 0u);
  if (stage_enc_bytes > 0 && !direct_ok &&
      ((reinterpret_cast<uint64_t>(encoded + block_byte_base) & 15u) == 0u) &&
      blockDim.x * 16 >= stage_enc_bytes) {
    const uint4* src4 = reinterpret_cast<const uint4*>(encoded + block_byte_base);
    uint4* dst4 = reinterpret_cast<uint4*>(s_enc);
    const int n4 = stage_enc_bytes >> 4;  // floor; the tail is handled below
    if (static_cast<int>(threadIdx.x) < n4) {
      const int64_t src = block_byte_base + (static_cast<int64_t>(threadIdx.x) << 4);
      uint4 v;
      if (src + 16 <= n_bytes) {
        v = src4[threadIdx.x];
      } else {  // partial sector: rebuild byte-exactly like the scalar path (zero padding)
        uint8_t tmp[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          tmp[k] = (src + k < n_bytes) ? encoded[src + k] : 0u;
        }
        v.x = tmp[0] | (tmp[1] << 8) | (tmp[2] << 16) | (static_cast<uint32_t>(tmp[3]) << 24);
        v.y = tmp[4] | (tmp[5] << 8) | (tmp[6] << 16) | (static_cast<uint32_t>(tmp[7]) << 24);
        v.z = tmp[8] | (tmp[9] << 8) | (tmp[10] << 16) | (static_cast<uint32_t>(tmp[11]) << 24);
        v.w = tmp[12] | (tmp[13] << 8) | (tmp[14] << 16) | (static_cast<uint32_t>(tmp[15]) << 24);
      }
      dst4[threadIdx.x] = v;
    }
    for (int i = (n4 << 4) + threadIdx.x; i < stage_enc_bytes; i += blockDim.x) {
      const int64_t src = block_byte_base + i;
      s_enc[i] = (src < n_bytes) ? encoded[src] : 0u;
    }
  } else if (stage_enc_bytes > 0 && !direct_ok) {
    for (int i = threadIdx.x; i < stage_enc_bytes; i += blockDim.x) {
      const int64_t src = block_byte_base + i;
      s_enc[i] = (src < n_bytes) ? encoded[src] : 0u;
    }
  }

  const int64_t gt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t meta = thread_meta[gt];
  const int64_t start_bit =
      gt * static_cast<int64_t>(bytes_per_thread) * 8 + (meta >> kMetaCountBits);
  int32_t count = static_cast<int32_t>(meta & kMetaCountMask);

  __syncthreads();  // s_luts / s_enc are now visible to the whole block

  const int32_t local_base = block_exclusive_scan(s_scan, count);

  if (stage_enc_bytes > 0 && stage_elems > 0) {
    if (local_base + count > stage_elems) {  // only reachable on a corrupt buffer
      count = (local_base < stage_elems) ? (stage_elems - local_base) : 0;
    }

    LutWalk ctx;
    ctx.luts = s_luts;
    ctx.num_levels = num_luts - 1;
    ctx.ptr_min =
        (ctx.num_levels > 1) ? static_cast<uint32_t>(256 - (ctx.num_levels - 1)) : 256u;
    ctx.lens_row = s_luts + ctx.num_levels * 256;

    const int cursor = static_cast<int>(start_bit - block_byte_base * 8);
    if (bytes_per_thread == 16) {
      const uint8_t* const span =
          direct_ok ? (encoded + block_byte_base) : s_enc;
      RegStream rs = reg_stream_open(span + (static_cast<int64_t>(threadIdx.x) << 4),
                                     static_cast<int>(meta >> kMetaCountBits));
      for (int32_t i = 0; i < count; ++i) {
        s_exp[local_base + i] = static_cast<uint8_t>(rs_decode(&rs, &ctx));
      }
    } else {
      BitWindowLocal window = bwl_open(s_enc, cursor);
      for (int32_t i = 0; i < count; ++i) {
        s_exp[local_base + i] = static_cast<uint8_t>(bwl_decode(&window, &ctx));
      }
    }
    __syncthreads();

    const int64_t block_elems =
        static_cast<int64_t>(output_positions[blockIdx.x + 1]) - out_base;

    if (sm_u32_aligned && block_elems >= 8) {
      const int64_t p = (8 - (out_base & 7)) & 7;  // first 8-aligned element index
      int64_t j = threadIdx.x;
      for (; j < block_elems && j < p; j += blockDim.x) {
        const int64_t gidx = out_base + j;
        if (gidx < n_elements) {
          out_bits[gidx] = make_bf16_bits(s_exp[j], sign_mantissa[gidx]);
        }
      }
      const int64_t nvec = (block_elems - p) >> 3;
      const bool sm_u64_aligned = ((reinterpret_cast<uint64_t>(sign_mantissa) & 7u) == 0u);
      const bool in_bounds = out_base + block_elems <= n_elements;
      for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
        const int64_t v = p + (c << 3);
        const int64_t gidx = out_base + v;
        const uint32_t e0123 = *reinterpret_cast<const uint32_t*>(s_exp + v);
        const uint32_t e4567 = *reinterpret_cast<const uint32_t*>(s_exp + v + 4);
        uint32_t s0123, s4567;
        if (sm_u64_aligned) {
          const uint2 s8 = *reinterpret_cast<const uint2*>(sign_mantissa + gidx);
          s0123 = s8.x;
          s4567 = s8.y;
        } else {
          s0123 = *reinterpret_cast<const uint32_t*>(sign_mantissa + gidx);
          s4567 = *reinterpret_cast<const uint32_t*>(sign_mantissa + gidx + 4);
        }
        const uint64_t lo4 = pack_bf16x4(e0123, s0123);
        const uint64_t hi4 = pack_bf16x4(e4567, s4567);
        if (in_bounds) {
          uint4 o;
          o.x = static_cast<uint32_t>(lo4);
          o.y = static_cast<uint32_t>(lo4 >> 32);
          o.z = static_cast<uint32_t>(hi4);
          o.w = static_cast<uint32_t>(hi4 >> 32);
          *reinterpret_cast<uint4*>(out_bits + gidx) = o;
        } else {  // ragged final chunk of the final block
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            if (gidx + k < n_elements) {
              out_bits[gidx + k] = make_bf16_bits(s_exp[v + k], sign_mantissa[gidx + k]);
            }
          }
        }
      }
      const int64_t tail = (block_elems - p) & 7;
      const int64_t vtail = p + (nvec << 3);
      if (threadIdx.x < tail) {
        const int64_t gidx = out_base + vtail + threadIdx.x;
        if (gidx < n_elements) {
          out_bits[gidx] = make_bf16_bits(s_exp[vtail + threadIdx.x], sign_mantissa[gidx]);
        }
      }
      return;
    }

    for (int64_t j = threadIdx.x; j < block_elems; j += blockDim.x) {
      const int64_t gidx = out_base + j;
      if (gidx < n_elements) {
        out_bits[gidx] = make_bf16_bits(s_exp[j], sign_mantissa[gidx]);
      }
    }
    return;
  }

  const uint8_t* const stream = (stage_enc_bytes > 0) ? s_enc : encoded;
  const int64_t stream_bytes = (stage_enc_bytes > 0) ? stage_enc_bytes : n_bytes;
  const int64_t cursor = (stage_enc_bytes > 0) ? (start_bit - block_byte_base * 8) : start_bit;

  if (stage_elems > 0) {
    if (local_base + count > stage_elems) {  // only reachable on a corrupt buffer
      count = (local_base < stage_elems) ? (stage_elems - local_base) : 0;
    }
    BitWindow window = bit_window_open(stream, stream_bytes, cursor);
    for (int32_t i = 0; i < count; ++i) {
      s_exp[local_base + i] =
          static_cast<uint8_t>(bit_window_decode(&window, s_luts, num_luts));
    }
    __syncthreads();

    const int64_t block_elems =
        static_cast<int64_t>(output_positions[blockIdx.x + 1]) - out_base;
    for (int64_t j = threadIdx.x; j < block_elems; j += blockDim.x) {
      const int64_t gidx = out_base + j;
      if (gidx < n_elements) {
        out_bits[gidx] = make_bf16_bits(s_exp[j], sign_mantissa[gidx]);
      }
    }
    return;
  }

  const int64_t base = out_base + local_base;
  BitWindow window = bit_window_open(stream, stream_bytes, cursor);
  for (int32_t i = 0; i < count; ++i) {
    const uint32_t exponent = bit_window_decode(&window, s_luts, num_luts);
    const int64_t gidx = base + i;
    if (gidx < n_elements) {
      out_bits[gidx] = make_bf16_bits(exponent, sign_mantissa[gidx]);
    }
  }
}


// Each block accumulates in shared uint32 bins and then contributes at most one global uint64 atomic per bin, so the encode
// never materializes a per-element exponent array, which would be larger than the weight being encoded.
extern "C" __global__ void dfloat11_exponent_histogram_kernel(
    const uint16_t* __restrict__ bf16_bits,
    uint64_t* __restrict__ histogram,
    int64_t n_elements) {
  __shared__ uint32_t bins[256];
  if (threadIdx.x < 256) {
    bins[threadIdx.x] = 0;
  }
  __syncthreads();

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n_elements; idx += stride) {
    const uint32_t exponent = (bf16_bits[idx] >> 7) & 0xFFu;
    atomicAdd(&bins[exponent], 1u);
  }
  __syncthreads();

  if (threadIdx.x < 256 && bins[threadIdx.x] != 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(histogram + threadIdx.x),
              static_cast<unsigned long long>(bins[threadIdx.x]));
  }
}


extern "C" __global__ void dfloat11_split_len_kernel(
    const uint16_t* __restrict__ bf16_bits,
    const int32_t* __restrict__ code_len,  // 256 entries
    uint8_t* __restrict__ exponent,
    uint8_t* __restrict__ sign_mantissa,
    uint8_t* __restrict__ len_out,
    int64_t n_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_elements) {
    return;
  }
  const uint32_t bits = bf16_bits[idx];
  const uint32_t exp = (bits >> 7) & 0xFFu;
  exponent[idx] = static_cast<uint8_t>(exp);
  sign_mantissa[idx] =
      static_cast<uint8_t>(((bits >> 8) & 0x80u) | (bits & 0x7Fu));
  len_out[idx] = static_cast<uint8_t>(code_len[exp]);
}


__device__ __forceinline__ int64_t lower_bound_i64(
    const int64_t* __restrict__ a, int64_t m, int64_t key) {
  int64_t lo = 0;
  int64_t hi = m;
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (a[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}


__device__ __forceinline__ uint32_t byte_contrib(uint64_t vL, int64_t byte_start, int64_t p) {
  const int shift = static_cast<int>(byte_start - p);  // in [-7, 31] for overlapping codes
  const int sh = 24 - shift;
  if (sh >= 0) {
    return static_cast<uint32_t>((vL >> sh) & 0xFFu);
  }
  return static_cast<uint32_t>((vL << (-sh)) & 0xFFu);
}


// Consecutive output bytes walk the same monotone prefix array, so a thread pays one lower_bound per chunk and then advances
// the symbol cursor linearly. The chunk is wide enough that binary searches amortize over several output bytes and narrow
// enough that the cursor advance stays short and the thread count stays high.
constexpr int kPackBytesPerThread = 4;

extern "C" __global__ void dfloat11_pack_kernel(
    const int64_t* __restrict__ pref,      // n_elements + 1 exclusive prefix sums of code lengths
    const uint8_t* __restrict__ exponent,
    const int32_t* __restrict__ code_len,  // 256
    const int32_t* __restrict__ code_val,  // 256
    uint8_t* __restrict__ encoded,
    int64_t n_elements,
    int64_t n_bytes,
    int64_t total_bits,
    int eof_len,
    uint32_t eof_val) {
  const int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t first_byte = chunk * kPackBytesPerThread;
  if (first_byte >= n_bytes) {
    return;
  }
  const int byte_count = static_cast<int>(
      (n_bytes - first_byte < kPackBytesPerThread) ? (n_bytes - first_byte)
                                                    : kPackBytesPerThread);

  const int64_t first_bit = first_byte * 8;
  int64_t i_start = lower_bound_i64(pref, n_elements + 1, first_bit + 1) - 1;
  if (i_start < 0) {
    i_start = 0;
  }

  uint64_t packed = 0;
#pragma unroll
  for (int k = 0; k < kPackBytesPerThread; ++k) {
    if (k >= byte_count) {
      break;
    }
    const int64_t j = first_byte + k;
    const int64_t byte_start = j * 8;
    const int64_t byte_end = byte_start + 8;

    while (i_start < n_elements) {
      const uint32_t sym = exponent[i_start];
      const int64_t end = pref[i_start] + code_len[sym];
      if (end > byte_start) {
        break;
      }
      ++i_start;
    }

    uint32_t acc = 0;
    for (int64_t i = i_start; i < n_elements && pref[i] < byte_end; ++i) {
      const int64_t p = pref[i];
      const uint32_t sym = exponent[i];
      const int b = code_len[sym];
      const int64_t end = p + b;
      if (end <= byte_start) {
        continue;
      }
      const uint64_t vL =
          static_cast<uint64_t>(static_cast<uint32_t>(code_val[sym])) << (32 - b);
      acc |= byte_contrib(vL, byte_start, p);
    }

    if (j == n_bytes - 1) {
      const int64_t r = total_bits - byte_start;
      if (r > 0 && r < 8) {
        const uint64_t eL = static_cast<uint64_t>(eof_val) << (32 - eof_len);
        acc |= byte_contrib(eL, byte_start, total_bits);
      }
    }
    packed |= static_cast<uint64_t>(acc) << (k * 8);
  }

  if (byte_count == kPackBytesPerThread) {
    if constexpr (kPackBytesPerThread == 8) {
      *reinterpret_cast<uint64_t*>(encoded + first_byte) = packed;
    } else if constexpr (kPackBytesPerThread == 4) {
      *reinterpret_cast<uint32_t*>(encoded + first_byte) = static_cast<uint32_t>(packed);
    } else {
      *reinterpret_cast<uint16_t*>(encoded + first_byte) = static_cast<uint16_t>(packed);
    }
  } else {
#pragma unroll
    for (int k = 0; k < kPackBytesPerThread; ++k) {
      if (k < byte_count) {
        encoded[first_byte + k] = static_cast<uint8_t>(packed >> (k * 8));
      }
    }
  }
}


extern "C" __global__ void dfloat11_thread_meta_kernel(
    const int64_t* __restrict__ pref,
    uint16_t* __restrict__ thread_meta,
    int64_t n_elements,
    int64_t total_bits,
    int64_t region_bits,
    int64_t n_regions) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_regions) {
    return;
  }
  const int64_t key_lo = t * region_bits;
  if (key_lo > total_bits) {
    thread_meta[t] = 0;
    return;
  }
  const int64_t key_hi = key_lo + region_bits;
  int64_t lb_lo = lower_bound_i64(pref, n_elements + 1, key_lo);
  if (lb_lo > n_elements) {
    lb_lo = n_elements;
  }
  int64_t lb_hi = (key_hi > total_bits)
                      ? n_elements
                      : lower_bound_i64(pref, n_elements + 1, key_hi);
  if (lb_hi > n_elements) {
    lb_hi = n_elements;
  }
  const uint32_t count = static_cast<uint32_t>(lb_hi - lb_lo);
  const uint32_t gap = (count > 0) ? static_cast<uint32_t>(pref[lb_lo] - key_lo) : 0u;
  thread_meta[t] = static_cast<uint16_t>((gap << kMetaCountBits) | count);
}


extern "C" __global__ void dfloat11_output_positions_kernel(
    const int64_t* __restrict__ pref,
    uint32_t* __restrict__ output_positions,
    int64_t n_elements,
    int64_t block_bits,
    int64_t num_blocks) {
  const int64_t blk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (blk > num_blocks) {
    return;
  }
  if (blk == num_blocks) {
    output_positions[blk] = static_cast<uint32_t>(n_elements);
    return;
  }
  const int64_t key = blk * block_bits;
  const int64_t lb = lower_bound_i64(pref, n_elements + 1, key);
  output_positions[blk] = static_cast<uint32_t>((lb <= n_elements) ? lb : n_elements);
}
