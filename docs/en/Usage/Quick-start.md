# Quick start

This guide covers installation, tensor compression and decompression, and basic Compressed Linear usage.

## Installation

Python 3.10 or later is required. First install a CUDA-enabled build of PyTorch 2.10 or later
for your environment.

### Install from source (recommended)

```bash
git clone https://github.com/modelscope/entropack.git
cd entropack
pip install -e ".[cuda13]"
```

### Install from PyPI

PyPI releases may lag behind source updates. Install from source for the latest features.

```bash
pip install "entropack[cuda13]"
```

Both installation methods above use CUDA 13 and include the matching CuPy package.
For CUDA 12, replace `cuda13` with `cuda12` in either command. If a compatible CuPy is
already installed, use `pip install -e .` for source installation or `pip install entropack` for PyPI.

Select PyTorch's CUDA variant when installing PyTorch.
Optional Triton kernels for INT8 computation can be enabled with `pip install triton`.
See [Compressed Linear usage](Linear-layers.md) for additional FP8 and INT8 hardware requirements.
The first call may be slower while CUDA kernels compile. Measure performance after warm-up.

## Direct tensor compression

This example selects lossy compression with `LatticeRANSConfig(target_bpp=3.5)`, compresses
a 2D BF16 tensor at a target of 3.5 bits per element, and restores its original shape and dtype.

```python
import torch
import entropack as ep

tensor = (torch.randn(256, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.LatticeRANSConfig(target_bpp=3.5)

compressed = ep.compress(tensor, config)
restored = ep.decompress(compressed, config)

print(f"Target: {config.target_bpp:.2f} bits per element")
print(f"Stored: {compressed.actual_bpp:.2f} bits per element")
print(restored.shape, restored.dtype)
```

`target_bpp` is measured in bits per element (bpp) and accepts finite integer or non-integer
values with `0.001 <= target_bpp <= 11`. Encoding caps the target by dtype as described in
[Compression configuration](Configuration.md). `actual_bpp` includes metadata and can differ from the target,
especially for small tensors. This scheme requires a nonempty 2D input without NaN or infinite values.

See [Tensor compression](Tensor-compression.md) for reconstruction error,
device transfers, and saving and loading. [Compression configuration](Configuration.md)
covers lossless schemes and the complete parameter reference.

## Use Compressed Linear

`CompressedLinear.from_linear` compresses an existing layer's weights and returns a new
layer with a copy of the original bias:

```python
import torch
import entropack as ep

linear = torch.nn.Linear(256, 256, dtype=torch.bfloat16, device="cuda")
config = ep.LatticeRANSConfig(target_bpp=4.0)
layer = ep.CompressedLinear.from_linear(linear, config=config)
x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")

with torch.inference_mode():
    output = layer(x)
print(output.shape, f"{layer.compressed_bits:.2f} bits per weight")
```

For pretrained models, load the checkpoint before replacing the corresponding layers.
[Compressed Linear usage](Linear-layers.md)
covers replacing multiple layers, low-precision computation, and saving and loading compressed checkpoints.
