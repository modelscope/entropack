# Tensor compression

Use `compress(tensor, config)` to compress a weight or other tensor into a
`CompressedTensor`, then `decompress(compressed, config)` to restore it.
A lossless scheme preserves every input bit; a lossy scheme returns an approximation.

## Compress and restore

By default, the decompressed tensor has the same shape, `dtype`, and `device` as the input.
To select a different dtype or device, convert the input with `tensor.to(dtype=..., device=...)`
before compression.

This example compresses a tensor at a 4 bpp target and measures the relative L2 error
after decompression. Encoding and decoding use a configuration from the same scheme:

```python
import torch
import entropack as ep

tensor = (torch.randn(128, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.LatticeRANSConfig(target_bpp=4.0)
compressed = ep.compress(tensor, config)
restored = ep.decompress(compressed, config)
relative_error = (restored.float() - tensor.float()).norm() / tensor.float().norm()

print(f"Stored: {compressed.actual_bpp:.2f} bits per element")
print(f"Relative L2 error: {100 * relative_error:.2f}%")
```

To change the bitrate, compress the source tensor again with a new `target_bpp`.

## Inspect stored size

```python
import torch
import entropack as ep

tensor = (torch.randn(128, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.LatticeRANSConfig(target_bpp=4.0)
compressed = ep.compress(tensor, config)

print(compressed.shape, compressed.dtype, compressed.compress_method)
print(f"Stored: {compressed.storage_nbytes()} bytes")
print(f"Rate: {compressed.actual_bpp:.2f} bits per element")
```

`storage_nbytes()` reports the compressed size in bytes, including metadata needed for decompression.
`actual_bpp` is `8 * storage_nbytes() / tensor.numel()`. These measure the compressed
result, not the size of a checkpoint file or peak runtime memory.

The achieved bitrate can differ from `target_bpp`, particularly for small tensors.
Compare the stored rate and reconstruction error when selecting a target.

`CompressedTensor` also exposes `shape`, `dtype`, `compress_method`, and `lossless`.
The [API reference](../API_Reference/index.md) describes its remaining properties.

## Move a compressed tensor

`compressed.to(device)` returns a compressed tensor on the requested device. This example
moves it to CPU for storage, then back to the GPU for decompression:

```python
import torch
import entropack as ep

tensor = (torch.randn(128, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.DFloat11Config()
compressed = ep.compress(tensor, config)
cpu_copy = compressed.to("cpu")
gpu_copy = cpu_copy.to("cuda")
restored = ep.decompress(gpu_copy, config)
print(restored.device, restored.dtype)
```

Decompressing the object returned by `compressed.to(device)` restores the tensor on that device.

## Save and load

Save a compressed tensor's `state_dict()`, load it with
`torch.load(..., weights_only=True)`, and restore the `CompressedTensor` with
`CompressedTensor.from_state_dict()`:

```python
from pathlib import Path

import torch
import entropack as ep

tensor = (torch.randn(128, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.DFloat11Config()
compressed = ep.compress(tensor, config)

path = Path("compressed_tensor.pt")
torch.save(compressed.state_dict(), path)
state = torch.load(path, map_location="cuda", weights_only=True)
loaded = ep.CompressedTensor.from_state_dict(state)

restored = ep.decompress(loaded, config)
assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
```

The example saves `compressed_tensor.pt` in the current directory. Change the path as needed.
`map_location` selects the device for loading. Keep the scheme configuration and library version
with the checkpoint.

## Input requirements

The lattice scheme requires a nonempty, finite, two-dimensional tensor. DFloat11 accepts
BF16, and Tile-ANS accepts the dtypes listed in [Config](Configuration.md). Lossless
schemes accept higher-dimensional tensors directly. For lattice compression, reshape
higher-dimensional data to 2D before compression and restore its outer shape after
decompression.

If compression emits a warning, check `compress_method`: a value of `"raw"` means the
tensor was stored without compression. Invalid configurations and unsupported backend
selections raise errors.
