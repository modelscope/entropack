# DFloat11

DFloat11 compresses BF16 tensors without changing their bits. It exploits the fact that
the exponent values in many tensors are concentrated in a small part of the available
range. Frequent exponents can then be represented with fewer bits, while the sign and
fraction remain unchanged.

## Encoding

A BF16 value contains one sign bit, eight exponent bits, and seven fraction bits.
The encoder separates each value into an exponent and a byte containing its sign and
fraction. These bytes are stored directly. The exponent stream is compressed using a
Huffman code constructed from the tensor's exponent frequencies: common exponents receive
short codes and uncommon exponents receive longer ones.

Huffman codes have variable lengths, so a decoder cannot start at an arbitrary bit and
immediately identify the next symbol. EntroPack records entry positions and symbol counts
for coding regions, allowing different regions to be decoded in parallel. The entry
information and Huffman tables add metadata to the compressed representation.

## Decoding and storage

Decoding recovers the exponent sequence through the Huffman tables and combines each
exponent with its stored sign and fraction. Reassembling these fields restores the
original BF16 bits, and the stored shape determines how they form the output tensor.
No numerical quantization or rounding is involved.

The achieved size depends on the exponent distribution and decoding metadata. A concentrated
distribution offers more compression than a broad one, and metadata has a larger relative
cost for small tensors. `DFloat11Config` does not specify a target bitrate. The name
DFloat11 does not imply that every tensor is stored at exactly 11 bits per element.

## Usage

The example compresses a BF16 tensor and verifies that decompression preserves its bits.
See [Config](../Usage/Configuration.md) for coding-region parameters.

```python
import torch
import entropack as ep

tensor = (torch.randn(256, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.DFloat11Config()

compressed = ep.compress(tensor, config)
restored = ep.decompress(compressed, config)

assert restored.shape == tensor.shape
assert restored.dtype == tensor.dtype
assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
print(f"Stored: {compressed.actual_bpp:.2f} bits per element")
```
