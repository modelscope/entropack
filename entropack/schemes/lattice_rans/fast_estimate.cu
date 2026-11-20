// Device-resident storage-cost estimate for the lattice_rans rate search: one bisection iteration
// is four kernel launches and one double returning, instead of copying the 17 stream histograms to
// the host and pricing them with rans._normalize_counts, rans.quantized_cross_entropy and the
// container arithmetic.
//
// ep_histogram_padded  counts like e8_histogram_kernel, but derives the compact bin layout from
//                      minmax on device, so no per-iteration D2H stall. Streams wider than
//                      table_size are skipped; rans.coded_bytes prices those as raw prob_bits.
// ep_fused_estimate    one block per stream: largest-remainder normalization plus per-symbol
//                      cross-entropy terms, reduced to one bit count per stream.
// ep_total_bytes       folds the per-stream bits into container bytes, mirroring
//                      rans.vector_stream_analytic_bytes + rans.coded_bytes.

#define TPB 256
#define NS 17
#define F64_INF __longlong_as_double(0x7FF0000000000000LL)

__device__ __forceinline__ long long d2key(double x) {
  const long long b = __double_as_longlong(x);
  return (b >= 0) ? b : (b ^ 0x7FFFFFFFFFFFFFFFLL);
}

__device__ __forceinline__ double key2d(long long k) {
  const long long b = (k >= 0) ? k : (k ^ 0x7FFFFFFFFFFFFFFFLL);
  return __longlong_as_double(b);
}

// Compact bin layout: stream 0 is the coset (2 bins), streams 1..16 are the coordinate fields
// alternating between the c=0 and c=1 sub-populations. off[s] < 0 marks a stream whose counts are
// not needed (empty, or wider than the rANS table).
__device__ __forceinline__ int stream_layout(const int* __restrict__ minmax, int table_size, int mm_cmax,
                                             int int_max, int* off, int* alpha) {
  const int a0 = minmax[mm_cmax] + 1;
  alpha[0] = a0;
  off[0] = 0;
  int cursor = a0;
  for (int st = 0; st < NS - 1; ++st) {
    const int lo = minmax[st * 2], hi = minmax[st * 2 + 1];
    const int a = (lo != int_max) ? (hi - lo + 1) : 0;
    alpha[1 + st] = a;
    if (a > 0 && a <= table_size) {
      off[1 + st] = cursor;
      cursor += a;
    } else {
      off[1 + st] = -1;
    }
  }
  return cursor;
}

extern "C" __global__ void ep_histogram_padded(
    const int* __restrict__ fields, const int* __restrict__ c_arr, const int* __restrict__ minmax,
    int* __restrict__ bins, long long V, int shared_budget, int table_size, int mm_cmax, int int_max) {
  __shared__ int s_off[NS];
  __shared__ int s_alpha[NS];
  __shared__ int s_total;
  extern __shared__ int sbins[];
  if (threadIdx.x == 0) s_total = stream_layout(minmax, table_size, mm_cmax, int_max, s_off, s_alpha);
  __syncthreads();
  const int total_bins = s_total;
  const int use_shared = (total_bins * 4) <= shared_budget;

  const long long stride = (long long)gridDim.x * blockDim.x;
  if (use_shared) {
    for (int t = threadIdx.x; t < total_bins; t += blockDim.x) sbins[t] = 0;
    __syncthreads();
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < V; idx += stride) {
      const int c = c_arr[idx];
      atomicAdd(&sbins[s_off[0] + c], 1);
      const int* f = fields + idx * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int st = j * 2 + c;
        const int off = s_off[1 + st];
        if (off >= 0) atomicAdd(&sbins[off + (f[j] - minmax[st * 2])], 1);
      }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < total_bins; t += blockDim.x) {
      if (sbins[t]) atomicAdd(&bins[t], sbins[t]);
    }
  } else {
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < V; idx += stride) {
      const int c = c_arr[idx];
      atomicAdd(&bins[s_off[0] + c], 1);
      const int* f = fields + idx * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int st = j * 2 + c;
        const int off = s_off[1 + st];
        if (off >= 0) atomicAdd(&bins[off + (f[j] - minmax[st * 2])], 1);
      }
    }
  }
}

template <typename T>
__device__ __forceinline__ T block_sum(T v, T* scratch) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    T acc = (T)0;
    for (int i = 0; i < (TPB + 31) / 32; ++i) acc += scratch[i];
    scratch[0] = acc;
  }
  __syncthreads();
  const T r = scratch[0];
  __syncthreads();
  return r;
}

template <typename T>
__device__ __forceinline__ T block_minmax(T v, T* scratch, int want_max) {
  for (int o = 16; o > 0; o >>= 1) {
    const T other = __shfl_down_sync(0xffffffffu, v, o);
    v = want_max ? (other > v ? other : v) : (other < v ? other : v);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    T acc = scratch[0];
    for (int i = 1; i < (TPB + 31) / 32; ++i) {
      acc = want_max ? (scratch[i] > acc ? scratch[i] : acc) : (scratch[i] < acc ? scratch[i] : acc);
    }
    scratch[0] = acc;
  }
  __syncthreads();
  const T r = scratch[0];
  __syncthreads();
  return r;
}

// Exclusive prefix sum of mask[0..A) in index order, in place.
__device__ __forceinline__ void excl_scan(int* mask, int A, int* scan_buf) {
  const int per = (A + TPB - 1) / TPB;
  const int lo = threadIdx.x * per;
  const int hi = min(lo + per, A);
  int local = 0;
  for (int i = lo; i < hi; ++i) local += mask[i];
  scan_buf[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int t = 0; t < TPB; ++t) {
      const int v = scan_buf[t];
      scan_buf[t] = acc;
      acc += v;
    }
  }
  __syncthreads();
  int base = scan_buf[threadIdx.x];
  for (int i = lo; i < hi; ++i) {
    const int v = mask[i];
    mask[i] = base;
    base += v;
  }
  __syncthreads();
}

struct Ctx {
  const double* val;
  const int* mask;
  int A;
};

__device__ __forceinline__ long long count_strict(const Ctx& c, double threshold, int above,
                                                  double* scratch) {
  long long local = 0;
  for (int i = threadIdx.x; i < c.A; i += TPB) {
    if (!c.mask[i]) continue;
    if (above ? (c.val[i] > threshold) : (c.val[i] < threshold)) ++local;
  }
  long long* s = (long long*)scratch;
  return block_sum<long long>(local, s);
}

// k-th largest (above=1) or smallest (above=0) masked value, ties by smaller index. Bisects the
// total-order bit pattern, so no sort and no stability assumption. Needs 1 <= k <= masked count.
__device__ double kth_value(const Ctx& c, int k, int above, double* scratch, int* flag) {
  long long local_min = 0x7FFFFFFFFFFFFFFFLL, local_max = -0x7FFFFFFFFFFFFFFFLL - 1;
  for (int i = threadIdx.x; i < c.A; i += TPB) {
    if (!c.mask[i]) continue;
    const long long key = d2key(c.val[i]);
    if (key < local_min) local_min = key;
    if (key > local_max) local_max = key;
  }
  long long* s = (long long*)scratch;
  const long long kmin = block_minmax<long long>(local_min, s, 0);
  const long long kmax = block_minmax<long long>(local_max, s, 1);
  if (kmin > kmax) {
    if (threadIdx.x == 0) *flag = 1;
    __syncthreads();
    return 0.0;
  }
  // count_above is non-increasing and count_below non-decreasing, so the two directions need
  // opposite invariants: for the k-th largest the answer is the first key whose count_above drops
  // below k, for the k-th smallest the last key whose count_below is still below k.
  long long lo = kmin - 1, hi = kmax + 1;
  while (hi - lo > 1) {
    const long long mid = lo + ((hi - lo) >> 1);
    const long long cnt = count_strict(c, key2d(mid), above, scratch);
    if (above) {
      if (cnt >= k) {
        lo = mid;
      } else {
        hi = mid;
      }
    } else {
      if (cnt >= k) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
  }
  return above ? key2d(hi) : key2d(lo);
}

extern "C" __global__ void ep_fused_estimate(
    const int* __restrict__ bins, const int* __restrict__ minmax, const double* __restrict__ log2_lut,
    double* __restrict__ partial_bits, int* __restrict__ alpha_out, int* __restrict__ nsym_out,
    int* __restrict__ fallback, long long table_size, int prob_bits, long long V, int mm_cmax, int int_max,
    int max_a) {
  __shared__ int s_off[NS];
  __shared__ int s_alpha[NS];
  const int s = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid == 0) stream_layout(minmax, (int)table_size, mm_cmax, int_max, s_off, s_alpha);
  __syncthreads();

  extern __shared__ char smraw[];
  int* s_counts = (int*)smraw;
  const int a_pad = (max_a + 1) & ~1;
  double* s_val = (double*)(s_counts + a_pad);
  int* s_freq = (int*)(s_val + a_pad);
  int* s_mask = s_freq + a_pad;
  double* s_scratch = (double*)(s_mask + a_pad);
  int* s_scan = (int*)(s_scratch + 64);

  const int alpha = s_alpha[s];
  const int nsym = (s == 0) ? (int)V : bins[s_off[0] + ((s - 1) & 1)];
  if (tid == 0) {
    alpha_out[s] = alpha;
    nsym_out[s] = nsym;
    partial_bits[s] = 0.0;
  }
  if (alpha == 0 || nsym == 0) return;
  if (alpha > table_size) return;
  if (alpha > max_a) {
    if (tid == 0) *fallback = 1;
    return;
  }
  __syncthreads();

  const int base = s_off[s];
  for (int i = tid; i < alpha; i += TPB) s_counts[i] = bins[base + i];
  __syncthreads();

  long long local_total = 0;
  for (int i = tid; i < alpha; i += TPB) local_total += s_counts[i];
  const long long total = block_sum<long long>(local_total, (long long*)s_scratch);
  if (total == 0) {
    if (tid == 0) *fallback = 1;
    return;
  }
  const double ratio = (double)table_size / (double)total;

  int local_present = 0;
  for (int i = tid; i < alpha; i += TPB) {
    const int c = s_counts[i];
    const double target = (double)c * ratio;
    int f = (int)floor(target);
    if (c > 0) {
      if (f < 1) f = 1;
      ++local_present;
    }
    s_freq[i] = f;
    s_val[i] = target - (double)f;
  }
  const int n_present = block_sum<int>(local_present, (int*)s_scratch);

  long long local_sum = 0;
  for (int i = tid; i < alpha; i += TPB) local_sum += s_freq[i];
  const long long diff = table_size - block_sum<long long>(local_sum, (long long*)s_scratch);

  if (diff > 0) {
    if ((long long)n_present < diff) {
      if (tid == 0) *fallback = 1;
      return;
    }
    for (int i = tid; i < alpha; i += TPB) s_mask[i] = (s_counts[i] > 0);
    __syncthreads();
    const Ctx ctx{s_val, s_mask, alpha};
    const double v = kth_value(ctx, (int)diff, 1, s_scratch, fallback);
    if (*fallback) return;
    long long local_gt = 0;
    for (int i = tid; i < alpha; i += TPB) {
      if (s_mask[i] && s_val[i] > v) ++local_gt;
      s_mask[i] = (s_mask[i] && s_val[i] == v) ? 1 : 0;
    }
    const long long n_gt = block_sum<long long>(local_gt, (long long*)s_scratch);
    const int m = (int)(diff - n_gt);
    if (m > 0) excl_scan(s_mask, alpha, s_scan);
    __syncthreads();
    for (int i = tid; i < alpha; i += TPB) {
      if (s_counts[i] <= 0) continue;
      if (s_val[i] > v || (s_val[i] == v && m > 0 && s_mask[i] < m)) s_freq[i] += 1;
    }
    __syncthreads();
  } else if (diff < 0) {
    // Largest-remainder reclaim: a served symbol's residual grows by 1, landing behind every
    // not-yet-served one, so every eligible symbol is decremented once (ascending residual, ties by
    // index) before any is touched twice: R full rounds over the eligible set, then a partial round
    // over those with capacity left.
    for (int i = tid; i < alpha; i += TPB) s_mask[i] = (s_counts[i] > 0 && s_freq[i] > 1) ? 1 : 0;
    __syncthreads();
    long long local_cap = 0;
    int local_maxc = 0;
    for (int i = tid; i < alpha; i += TPB) {
      if (!s_mask[i]) continue;
      const int c = s_freq[i] - 1;
      local_cap += c;
      if (c > local_maxc) local_maxc = c;
    }
    const long long capacity = block_sum<long long>(local_cap, (long long*)s_scratch);
    const int max_c = block_minmax<int>(local_maxc, (int*)s_scratch, 1);
    const long long need = -diff;
    if (capacity < need || max_c <= 0) {
      if (tid == 0) *fallback = 1;
      return;
    }
    int r_lo = 0, r_hi = max_c;
    while (r_lo < r_hi) {
      const int mid = (r_lo + r_hi + 1) >> 1;
      long long local_s = 0;
      for (int i = tid; i < alpha; i += TPB) {
        if (!s_mask[i]) continue;
        const int c = s_freq[i] - 1;
        local_s += (c < mid) ? c : mid;
      }
      const long long sm = block_sum<long long>(local_s, (long long*)s_scratch);
      if (sm <= need) {
        r_lo = mid;
      } else {
        r_hi = mid - 1;
      }
    }
    const int R = r_lo;
    long long local_used = 0;
    for (int i = tid; i < alpha; i += TPB) {
      if (!s_mask[i]) continue;
      const int c = s_freq[i] - 1;
      const int d = (c < R) ? c : R;
      s_freq[i] -= d;
      local_used += d;
    }
    const long long used = block_sum<long long>(local_used, (long long*)s_scratch);
    const int rem = (int)(need - used);
    if (rem > 0) {
      for (int i = tid; i < alpha; i += TPB) {
        const int live = s_mask[i] && s_freq[i] > 1;
        s_mask[i] = live ? 1 : 0;
        if (!live) s_val[i] = F64_INF;
      }
      __syncthreads();
      const Ctx ctx{s_val, s_mask, alpha};
      const double v = kth_value(ctx, rem, 0, s_scratch, fallback);
      if (*fallback) return;
      long long local_lt = 0;
      for (int i = tid; i < alpha; i += TPB) {
        if (s_mask[i] && s_val[i] < v) ++local_lt;
        s_mask[i] = (s_mask[i] && s_val[i] == v) ? 1 : 0;
      }
      const long long n_lt = block_sum<long long>(local_lt, (long long*)s_scratch);
      const int m = rem - (int)n_lt;
      if (m > 0) excl_scan(s_mask, alpha, s_scan);
      __syncthreads();
      for (int i = tid; i < alpha; i += TPB) {
        if (s_counts[i] <= 0 || s_freq[i] <= 1) continue;
        if (s_val[i] < v || (s_val[i] == v && m > 0 && s_mask[i] < m)) s_freq[i] -= 1;
      }
      __syncthreads();
    }
  }

  const double pb = (double)prob_bits;
  double local_bits = 0.0;
  for (int i = tid; i < alpha; i += TPB) {
    const int c = s_counts[i];
    if (c > 0) local_bits += (double)c * (pb - log2_lut[s_freq[i]]);
  }
  const double bits = block_sum<double>(local_bits, s_scratch);
  if (tid == 0) partial_bits[s] = bits;
}

extern "C" __global__ void ep_total_bytes(
    const double* __restrict__ partial_bits, const int* __restrict__ alpha, const int* __restrict__ nsym,
    const int* __restrict__ fallback, double* __restrict__ out, long long table_size, int prob_bits,
    long long V, int tile_vectors, int num_states) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  double bits = 0.0, overflow_bits = 0.0;
  long long freq_bytes = 0;
  for (int s = 0; s < NS; ++s) {
    if (nsym[s] && alpha[s] <= table_size) {
      bits += partial_bits[s];
      freq_bytes += (long long)alpha[s] * 2;
    } else {
      overflow_bits += (double)nsym[s] * (double)prob_bits;
    }
  }
  long long num_tiles = (V + tile_vectors - 1) / tile_vectors;
  if (num_tiles < 1) num_tiles = 1;
  const double state_residual_bits = (double)num_tiles * (double)num_states * 16.0;
  double payload = bits - state_residual_bits;
  if (payload < 0.0) payload = 0.0;
  const long long payload_words = (long long)ceil(payload / 16.0);
  out[0] = (double)(payload_words * 2 + num_tiles * num_states * 4 + (num_tiles + 1) * 4 + freq_bytes)
           + overflow_bits / 8.0;
  out[1] = (double)(*fallback);
}
