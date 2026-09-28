# Compressed Linear usage

Compressed Linear replaces a PyTorch linear layer with one that uses compressed weights.
Call `layer(x)` as usual: the input's last dimension changes from `in_features` to
`out_features`, and the other dimensions stay the same. A [Config](Configuration.md)
selects the compression scheme and its parameters.

Compressed Linear requires a CUDA GPU and the matching CuPy package. See
[Quick start](Quick-start.md) for installation.

`CompressedLinear` accepts `DFloat11Config`, `TileANSConfig`, or `LatticeRANSConfig`.
The selected scheme must support the weight dtype.

## Replace an existing layer

`from_linear` compresses an existing layer's weights and returns a new layer with a copy
of the original bias. For pretrained models, load the checkpoint before calling this method:

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

Assign the returned layer to the corresponding module attribute to use it in the model.

## Replace several layers in a model

This example replaces ordinary linear layers recursively and keeps the output layer in its
original format. Names in `skip` are module paths, as reported by `named_modules()`.

```python
import torch
import entropack as ep

model = torch.nn.Sequential(
    torch.nn.Linear(256, 256),
    torch.nn.GELU(),
    torch.nn.Linear(256, 64),
).to(device="cuda", dtype=torch.bfloat16).eval()
config = ep.LatticeRANSConfig(target_bpp=4.0)


def compress_linears(module, config, skip=(), prefix=""):
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if path in skip:
            continue
        if type(child) is torch.nn.Linear:
            replacement = ep.CompressedLinear.from_linear(child, config=config)
            setattr(module, name, replacement.train(child.training))
        else:
            compress_linears(child, config, skip, path)


compress_linears(model, config, skip={"2"})
x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    output = model(x)
print(output.shape, type(model[0]).__name__, type(model[2]).__name__)
```

The example selects standard `torch.nn.Linear` layers. Custom linear classes or shared
weights may need model-specific handling. Omit `skip` to compress every ordinary linear
layer. If `CompressedLinear` cannot compress a layer's weights, replacement raises an
error; use `skip` to keep that layer in its original form.

## Combine compression with FP8 or INT8 computation

| Layer | Weight format | Computation |
| --- | --- | --- |
| `CompressedLinear` | Input weight dtype | Standard linear operation, using the activation dtype |
| `CompressedFP8Linear` | FP8 E4M3FN codes | FP8 weights and activations, requires a CUDA GPU with SM8.9 or later |
| `CompressedINT8Linear` | INT8 codes | INT8 weights and activations, requires a CUDA GPU with SM8.0 or later |

For `CompressedFP8Linear` and `CompressedINT8Linear`, use `config=None` for FP8 or INT8
quantization alone, or pass `LatticeRANSConfig(target_bpp=...)` to apply further lossy
compression to the quantized weights. The target must be at least 1 bpp and less than 8 bpp.

```python
import torch
import entropack as ep

linear = torch.nn.Linear(256, 256, dtype=torch.bfloat16, device="cuda")
layer = ep.CompressedINT8Linear.from_linear(
    linear, config=ep.LatticeRANSConfig(target_bpp=4.0)
)
x = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")
with torch.inference_mode():
    output = layer(x)
print(output.shape, layer.container_dtype, f"{layer.compressed_bits:.2f} bits per weight")
```

Use `CompressedFP8Linear` in the same pattern for FP8, on supported hardware.
To inspect the weights, `codes()` returns their FP8 or INT8 values, and `dequantize()`
returns their floating-point values after dequantization.

## Measure storage

`stored_nbytes` reports the compressed weight size, including metadata and FP8 or INT8 quantization scales.
`compressed_bits` is `8 * stored_nbytes / (in_features * out_features)`.
For multiple layers, sum stored bytes and weight elements before computing the ratio.
Biases are separate from this weight-storage measure.

This measures weight storage, not peak inference memory.

## Save and load a model

Save the model's `state_dict`, then construct a model with the same architecture and
Compressed Linear classes before loading it. The following example compresses two layers
and restores their saved weights into a fresh model:

```python
from pathlib import Path

import torch
import entropack as ep

config = ep.LatticeRANSConfig(target_bpp=4.0)
model = torch.nn.Sequential(
    torch.nn.Linear(256, 256),
    torch.nn.GELU(),
    torch.nn.Linear(256, 64),
).to(device="cuda", dtype=torch.bfloat16)
for index in (0, 2):
    model[index] = ep.CompressedLinear.from_linear(model[index], config=config)
model.eval()

x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    expected = model(x)

path = Path("compressed_model.pt")
torch.save(model.state_dict(), path)

restored = torch.nn.Sequential(
    ep.CompressedLinear(256, 256, config=config, device="cuda", dtype=torch.bfloat16),
    torch.nn.GELU(),
    ep.CompressedLinear(256, 64, config=config, device="cuda", dtype=torch.bfloat16),
).eval()
state = torch.load(path, map_location="cuda", weights_only=True)
restored.load_state_dict(state)
with torch.inference_mode():
    actual = restored(x)

assert torch.allclose(actual, expected)
print(actual.shape)
```

The example saves `compressed_model.pt` in the current directory. Change the path as needed.
To load an existing checkpoint, construct the `restored` model, then call `torch.load` and
`load_state_dict`.
Keep the model architecture, layer names and classes, compression configurations, weight
dtypes, and library version with the checkpoint. Ordinary `torch.nn.Linear` layers
cannot load Compressed Linear checkpoints directly.
