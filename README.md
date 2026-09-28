# EntroPack

### General-purpose tensor compression for PyTorch

EntroPack is a general-purpose tensor compression library for PyTorch. It supports lossless
compression for exact recovery and lossy compression with a target bitrate to balance storage
and reconstruction accuracy. GPU encoding and decoding compress tensors and restore them
in their original shape and dtype.

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)

[Documentation](https://entropackdoc.readthedocs.io/en/latest/) · [中文](README_zh.md)

- **Flexible bitrates.** Compress each weight matrix at any non-integer target bitrate,
  or preserve every input bit with a lossless scheme.
- **Multiple dtypes.** Compress tensors in 16 floating-point, integer, and Boolean dtypes,
  such as FP32, BF16, FP16, FP8, and INT8.
- **Easy to use.** Compress and restore tensors through a common API, and save
  them with `state_dict`. Compressed linear layers provide an integration for model weights.

## Installation

Use Python 3.10 or later and install a CUDA-enabled build of PyTorch 2.10 or later
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

See [Quick start](https://entropackdoc.readthedocs.io/en/latest/Usage/Quick-start.html) for environment requirements and usage examples.

## Get started

### Direct tensor compression

This example compresses a 2D BF16 tensor at a target of 3.5 bits per element,
then decompresses it to a tensor with the original shape and dtype:

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

`target_bpp` is the requested number of bits per element (bpp). `actual_bpp` reports the stored
rate, including metadata. Finite targets with `0.001 <= target_bpp <= 11` are supported, including
non-integer values. See [Compression configuration](https://entropackdoc.readthedocs.io/en/latest/Usage/Configuration.html) for target limits by dtype.

### Compressed Linear

`CompressedLinear.from_linear` compresses an existing `torch.nn.Linear`'s weights using
the supplied Config and returns a new Compressed Linear. Call it as `layer(x)` to compute
the output, just as with an ordinary linear layer:

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

See [Compressed Linear usage](https://entropackdoc.readthedocs.io/en/latest/Usage/Linear-layers.html) for model replacement and checkpoint examples.

## Configuration

Config defines the compression scheme and its settings. It is passed to tensor compression
and decompression functions or to a compressed Linear layer's constructor.

| Config | Compression and use cases |
| --- | --- |
| `DFloat11Config()` | Specialized lossless compression for BF16 tensors. Decompression restores every input bit, for applications requiring exact recovery. |
| `TileANSConfig()` | Lossless compression for BF16, FP16, FP32, FP8, INT8, and other supported dtypes. The compression ratio depends on the input data distribution. |
| `LatticeRANSConfig(target_bpp=...)` | Lossy compression of 2D floating-point and integer tensors. `target_bpp` specifies the target bits per element in [0.001, 11], including non-integer values, to balance storage size and reconstruction accuracy. |

See [Compression configuration](https://entropackdoc.readthedocs.io/en/latest/Usage/Configuration.html) for scheme selection
and the complete parameter reference.

## Performance

On one NVIDIA H20, EntroPack compresses the weights of all 276 linear layers in
Z-Image-Turbo's diffusion transformer at a 4 bpp target in **3.5 seconds**. The compressed
weights occupy **4.02 bpp**, with **7.18%** relative L2 reconstruction error. Inference with
these weights takes **544.7 ms** per denoising step, only **7.7%** above the original BF16
model's 505.7 ms.

## Documentation

| Guide | Contents |
| --- | --- |
| [Quick start](https://entropackdoc.readthedocs.io/en/latest/Usage/Quick-start.html) | Install and run tensor compression and Compressed Linear examples |
| [Compression configuration](https://entropackdoc.readthedocs.io/en/latest/Usage/Configuration.html) | Choose a scheme and look up supported dtypes and parameters |
| [Tensor compression](https://entropackdoc.readthedocs.io/en/latest/Usage/Tensor-compression.html) | Encode, decode, inspect storage, move data, and save or load tensors |
| [Compressed Linear usage](https://entropackdoc.readthedocs.io/en/latest/Usage/Linear-layers.html) | Replace model layers, use low-precision computation, and manage checkpoints |
| [API reference](https://entropackdoc.readthedocs.io/en/latest/API_Reference/index.html) | Look up functions, classes, and properties |

Compression principles: [DFloat11](https://entropackdoc.readthedocs.io/en/latest/Principles/DFloat11.html), [tile-ANS](https://entropackdoc.readthedocs.io/en/latest/Principles/Tile-ANS.html),
and [EntroPack lattice quantization](https://entropackdoc.readthedocs.io/en/latest/Principles/Lattice-rANS.html).

## Acknowledgements

EntroPack's design is inspired by [DFloat11](https://github.com/LeanModels/DFloat11),
[dahuffman](https://github.com/soxofaan/dahuffman),
[DietGPU](https://github.com/facebookresearch/dietgpu), and
[tile-ANS](https://arxiv.org/abs/2606.15789).

## License

[Apache License 2.0](LICENSE).
