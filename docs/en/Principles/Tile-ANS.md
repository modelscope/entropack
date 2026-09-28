# Tile-ANS

Tile-ANS compresses tensors losslessly by encoding their storage bytes. It supports
floating-point and integer tensors, including BF16, FP16, FP32, FP8, and INT8. Because it
works on bit representations rather than numerical approximations, decompression restores
the original values exactly.

## Byte streams and probability tables

Different byte positions within a numerical format often have different distributions.
Tile-ANS therefore groups bytes by their position within each element. A two-byte format
produces two streams, while a four-byte format produces four. Each stream collects the
corresponding byte from every tensor element.

The encoder counts byte frequencies separately for these streams and builds a probability
table for each. A skewed distribution can be encoded compactly because common bytes receive
shorter representations on average. When a stream offers little benefit after accounting
for coding overhead, it is stored directly. A single tensor can therefore contain both
entropy-coded streams and directly stored streams.

## Tiled encoding and decoding

Each stream is divided into independently decodable tiles. Entropy-coded tiles use range
asymmetric numeral systems (rANS), which encode symbols through reversible integer-state
updates. Multiple interleaved states allow symbols within a tile to be decoded in parallel,
while separate tiles provide additional parallel work. All tiles of a stream share its
probability table, avoiding a separate table for every tile.

The decoder uses the same probability tables to reverse the state updates and recover
each coded byte stream. It then combines the decoded and directly stored streams, placing
their bytes back into the original positions within the tensor elements.

No quantization is performed. Storage depends on the byte distributions and metadata,
so lossless compression does not provide a chosen target bitrate or guarantee a smaller
representation for every input. Larger tiles reduce metadata per element, while smaller
tiles expose more independent decoding tasks.

## Usage

The example compresses an FP16 tensor and checks its original bits after decompression.
Tile and probability-table settings are described in [Config](../Usage/Configuration.md).

```python
import torch
import entropack as ep

tensor = (torch.randn(256, 256, device="cuda") * 0.02).to(torch.float16)
config = ep.TileANSConfig()

compressed = ep.compress(tensor, config)
restored = ep.decompress(compressed, config)

assert restored.shape == tensor.shape
assert restored.dtype == tensor.dtype
assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
print(f"Stored: {compressed.actual_bpp:.2f} bits per element")
```
