# API reference

The main functions and classes are available through `import entropack as ep`.
For complete examples, see [General tensor compression](../Usage/Tensor-compression.md) and
[Compressed Linear usage](../Usage/Linear-layers.md). Configuration parameters are listed in
[Compression configuration](../Usage/Configuration.md).

## Tensor encoding and decoding

### compress

```text
compress(tensor: torch.Tensor, config: CompressionConfig) -> CompressedTensor
```

Compresses `tensor` using the scheme selected by `config`. Returns a `CompressedTensor`
that decompresses to the same shape and dtype as the input by default.

Input requirements depend on the scheme:
DFloat11 accepts BF16 tensors, Tile-ANS supports multiple dtypes, and lattice quantization requires
a nonempty two-dimensional tensor with finite values.

| Parameter | Meaning |
| --- | --- |
| `tensor` | PyTorch tensor to compress |
| `config` | Required. Selects a scheme through `DFloat11Config`, `TileANSConfig`, or `LatticeRANSConfig` |

Some encoding failures emit a warning explaining the failure and return an uncompressed
container with `compress_method == "raw"`. Invalid configurations and backend dispatch
failures raise errors.

### decompress

```text
decompress(compressed: CompressedTensor, config: CompressionConfig) -> torch.Tensor
```

Returns a tensor with `compressed.shape`, `compressed.dtype`, and `compressed.device`.
Without an output dtype conversion, lossless schemes restore input values bit for bit;
lossy schemes return an approximate reconstruction.

| Parameter | Meaning |
| --- | --- |
| `compressed` | Container returned by `compress` or restored from a checkpoint |
| `config` | Configuration for the corresponding scheme; decode settings control reconstruction |

Changing encoding parameters such as `target_bpp` at decode time does not alter the stored data
or requantize the tensor.

## CompressedTensor

A `torch.Tensor` subclass holding the compressed representation of one tensor. Usually returned
by `compress` or restored from a checkpoint with `from_state_dict`. Use `decompress` before
performing numerical operations.

### Common properties

| Property | Type | Meaning |
| --- | --- | --- |
| `shape` | `torch.Size` | Original tensor shape |
| `dtype` | `torch.dtype` | Reconstructed tensor dtype |
| `encoded_dtype` | `torch.dtype` | Dtype used for encoding |
| `compress_method` | `str` | Scheme actually used by the container |
| `lossless` | `bool` | Whether the scheme is lossless |
| `actual_bpp` | `float` | Stored bits per element, including metadata |

`actual_bpp = 8 * storage_nbytes() / math.prod(shape)`.
This measures the compressed representation, not checkpoint file size or runtime memory use.

### Common methods

| Method | Returns | Meaning |
| --- | --- | --- |
| `to(...)` | `CompressedTensor` | Changes device or output dtype without recompression; `copy=True` copies storage |
| `storage_nbytes(include_header=True)` | `int` | Total compressed size in bytes; `include_header=False` excludes the container header |
| `state_dict(prefix="")` | `dict[str, torch.Tensor]` | Exports the compressed tensor for saving |
| `CompressedTensor.from_state_dict(state, prefix="")` | `CompressedTensor` | Restores the container from that dictionary without recompression |

Use the same `prefix` when saving and restoring. The dictionary can be saved with `torch.save`
and loaded with `torch.load(..., weights_only=True)`. Set `map_location` to choose the device
on which it will be restored.
Loading restores the encoded dtype. Call `.to(dtype=...)` afterwards if a different output dtype is needed.

## CompressedLinear

A linear layer that uses reconstructed weights for each forward call. Weights are stored in
compressed form; the bias is not compressed.
Requires a CUDA GPU and the matching CuPy package.

### Creating a layer

```text
CompressedLinear(in_features, out_features, bias=True, *,
                 config=None, device=None, dtype=torch.bfloat16)
CompressedLinear.from_linear(linear, **kwargs) -> CompressedLinear
```

| Constructor parameter | Meaning |
| --- | --- |
| `in_features` / `out_features` | Input and output feature counts |
| `bias` | Whether to include a bias |
| `config` | Weight compression configuration. The default `None` selects DFloat11 for BF16 and Tile-ANS for other supported dtypes |
| `device` | Bias device when constructing a layer directly |
| `dtype` | Dtype used when compressing weights and initializing the bias |

`from_linear` returns a new layer with compressed source weights and a copy of the bias.
The source weights must already be loaded and cannot be on the `meta` device.
A typical call is `ep.CompressedLinear.from_linear(linear, config=config)`.
Pass `config` or `dtype` through `kwargs`; `dtype` defaults to the source weight dtype,
and the device is taken from the source layer.

Calling the constructor directly creates a layer without weight data. Call `compress_weight`
or load a checkpoint before running inference.

### Common methods and properties

| Interface | Returns | Meaning |
| --- | --- | --- |
| `compress_weight(weight)` | `None` | Initializes compressed weights with shape `(out_features, in_features)` |
| `dequantize(device=None)` | `torch.Tensor` | Returns dense weights with `weight.dtype`, on the layer's device unless `device` is specified |
| `forward(x)` | `torch.Tensor` | Applies the layer to `x` of shape `(..., in_features)` and returns shape `(..., out_features)` |
| `weight` | `CompressedTensor` | Frozen compressed weight parameter held by the layer |
| `container_dtype` | `torch.dtype` | Dtype of the weights or quantized codes in the compressed container |
| `stored_nbytes` | `int` | Weight storage bytes, including metadata and low-precision quantization scales, excluding bias |
| `compressed_bits` | `float` | `8 * stored_nbytes / (in_features * out_features)` |

Invoke the forward operation as `layer(x)`. `.weight` is a compressed tensor; use `dequantize()`
when numerical weights are needed.

Use standard `state_dict()` / `load_state_dict()` calls to save and restore layer state.
Before loading, construct layers with matching classes, shapes, container dtypes, and compression
schemes. The Config object itself is not stored in the checkpoint. `.to(device)` moves the layer;
model dtype conversion does not re-encode its compressed weights.

## CompressedFP8Linear and CompressedINT8Linear

Linear layers with FP8 or INT8 weights and activations. They share the constructor arguments,
`from_linear`, storage properties, and checkpoint interfaces of `CompressedLinear`.
For input shape `(..., in_features)`, the output has shape `(..., out_features)` and the input's
dtype and device.

| Class | Weight and activation format | CUDA GPU requirement |
| --- | --- | --- |
| `CompressedFP8Linear` | FP8 E4M3FN | SM8.9 or later |
| `CompressedINT8Linear` | INT8 | SM8.0 or later |

The layer class determines the code format. The constructor's `dtype` argument does not change
the FP8 or INT8 format.

With `config=None`, quantized codes are stored directly. Passing `LatticeRANSConfig` applies
additional lossy compression with `0.001 <= target_bpp < 8`. `stored_nbytes` includes the
per-row quantization scales needed to reconstruct weights.

| Method | Returns | Meaning |
| --- | --- | --- |
| `codes(device=None)` | FP8 or INT8 tensor | Restores quantized codes without applying row scales |
| `dequantize(device=None)` | `torch.Tensor` | Returns the layer's initialization dtype, which `from_linear` defaults to the source weight dtype |

Both methods return tensors on the layer's device unless `device` is specified.
Lossy compression may change the codes from their initial quantized values.

## Config classes

Configs control tensor encoding and decoding as well as weight storage in Compressed Linear.
The following classes inherit from `CompressionConfig`:

| Class | Purpose |
| --- | --- |
| `DFloat11Config` | Lossless BF16 compression |
| `TileANSConfig` | Lossless tiled ANS compression for multiple dtypes |
| `LatticeRANSConfig` | Lossy lattice quantization with bitrate controlled by `target_bpp` |

See [Compression configuration](../Usage/Configuration.md) for scheme selection, defaults,
and parameter ranges.
