# Compression configuration

A Config selects the compression scheme and its encoding and decoding parameters.
`execution_backend` defaults to `"auto"`, which selects the execution backend automatically.

## Choose a configuration

| Config | Compression | Input requirements |
| --- | --- | --- |
| `DFloat11Config()` | Lossless BF16 compression | BF16 tensors |
| `TileANSConfig()` | Lossless compression for multiple dtypes | Supported tensor dtypes |
| `LatticeRANSConfig(target_bpp=...)` | Lossy compression at a target bitrate | Nonempty, finite, two-dimensional tensors |

Both lossless schemes reproduce every input bit. Their compressed size depends on the tensor's
data distribution. `TileANSConfig` also supports BF16, so either lossless scheme can be used for
that dtype. `LatticeRANSConfig` accepts finite targets with `0.001 <= target_bpp <= 11` bits per
element, including non-integer values. Its actual stored rate is available through
`CompressedTensor.actual_bpp`.

## Supported tensor dtypes

| Tensor dtype | `DFloat11Config` | `TileANSConfig` | `LatticeRANSConfig` |
|---|---|---|---|
| `float32` | — | lossless | lossy |
| `float16` | — | lossless | lossy |
| `bfloat16` | lossless | lossless | lossy |
| `float8_e4m3fn` | — | lossless | lossy |
| `float8_e4m3fnuz` | — | lossless | lossy |
| `float8_e5m2` | — | lossless | lossy |
| `float8_e5m2fnuz` | — | lossless | lossy |
| `int64` | — | lossless | lossy |
| `int32` | — | lossless | lossy |
| `int16` | — | lossless | lossy |
| `int8` | — | lossless | lossy |
| `uint64` | — | lossless | lossy |
| `uint32` | — | lossless | lossy |
| `uint16` | — | lossless | lossy |
| `uint8` | — | lossless | lossy |
| `bool` | — | lossless | lossy |

The lossless schemes accept tensors of different shapes, while lattice quantization requires
two-dimensional input. BF16, FP16, and FP8 refer to the corresponding PyTorch dtypes above.
Packed four-bit formats, FP64, and complex dtypes are not supported.

## Parameter conventions

Changes to encode settings affect subsequent compression, not existing compressed data.
Decode settings take effect when restoring a tensor. Most settings can retain their defaults. For lossy compression, `target_bpp` sets the
target rate and a positive `row_rdo_iterations` enables per-row rate–distortion optimization (RDO).

## CompressionConfig

These execution settings apply to all three configuration classes.

| Field | Type | Default | Stage | Meaning |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` or `None` | `"auto"` | Both | `"auto"` or `None` prefers CUDA and falls back to the PyTorch implementation if unavailable. `"cuda"` requires CUDA. `"eager"` selects the PyTorch fallback for tensor encoding and decoding. |

## DFloat11Config

Lossless BF16 compression.

| Field | Type | Default | Stage | Meaning |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` or `None` | `"auto"` | Both | Backend selection as above |
| `bytes_per_thread` | Positive `int` or `None` | `16` | Encode | Encoded bytes processed per thread. Affects compression ratio and decoding parallelism. |
| `threads_per_block` | Positive `int` or `None` | `128` | Encode | Threads per block during encoding |

## TileANSConfig

Lossless compression of the supported tensor dtypes.

| Field | Type | Default | Stage | Meaning |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` or `None` | `"auto"` | Both | Backend selection as above |
| `tile_elements` | `int` in [0, 2^31 − 1] | `0` | Encode | Elements per compressed tile. Affects compression ratio and decoding parallelism. `0` selects automatically. |
| `probability_bits` | `0`, `9`, `10`, `11`, `12` | `0` | Encode | Probability-table precision. `0` selects automatically. |
| `raw_lane_threshold` | `float` in [0, 8] | `7.9` | Encode | Threshold for storing hard-to-compress data directly, measured in estimated encoded bits per input byte. |
| `threads_per_block` | Positive `int` or `None` | `None` | Both | GPU block width. `None` selects automatically. |

## LatticeRANSConfig

Lossy compression of finite, two-dimensional tensors.

At encoding time, the effective target is the smaller of `target_bpp` and the dtype limit:
1 bpp for `bool`, 8 bpp for `int8`, `uint8`, and all supported FP8 dtypes, and 11 bpp for
other supported dtypes. The config retains the requested value. These limits apply to the
encoding target; `actual_bpp` includes metadata and may exceed them. Tensor distribution and
metadata overhead can make low targets unattainable. Compression of `bool` remains lossy.

| Field | Type | Default | Stage | Meaning |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` or `None` | `"auto"` | Both | Backend selection as above |
| `target_bpp` | Finite `float` in [0.001, 11] | `4.0` | Encode | Target bits per input element, capped by dtype at encoding time. Non-integer targets are supported. Inspect `actual_bpp` for the stored rate. |
| `prob_bits` | `int` in [9, 15], `0`, or `None` | `None` | Encode | Probability-table precision. `None` or `0` selects automatically. |
| `tile_elements` | Positive `int` or `None` | `None` | Encode | Elements per compressed tile. Affects compression ratio and decoding parallelism. `None` selects automatically. |
| `row_rdo_iterations` | `int` in [0, 8] | `0` | Encode | Per-row rate–distortion refinement sweeps. `0` disables refinement. More sweeps increase compression time. |
| `row_rdo_candidates` | Positive `int` | `5` | Encode | Number of candidate quantizations per row for RDO. More candidates increase compression time. |
| `scale_search_iterations` | Positive `int` | `12` | Encode | Number of quantization-scale search iterations |
| `scale_search_max_vectors` | Positive `int` | `262144` | Encode | Sample limit for rate search, in vectors of eight elements |
| `threads_per_block` | Positive `int` or `None` | `None` | Decode | GPU block width. `None` selects automatically. |
| `l2_prefetch` | `bool` | `True` | Decode | Enable GPU L2 cache prefetching during decoding |

## Compression principles

The encoding and decoding processes are described in [DFloat11](../Principles/DFloat11.md),
[Tile-ANS](../Principles/Tile-ANS.md), and [Lattice-rANS](../Principles/Lattice-rANS.md).
