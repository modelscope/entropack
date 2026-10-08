# DFloat11

DFloat11 对 BF16 张量进行无损压缩，完整保留原始位表示。许多张量的指数值集中在较小的范围内，
因此可以用较短的编码表示常见指数，同时原样保存符号位和尾数部分。

## 编码

一个 BF16 数值包含 1 位符号、8 位指数和 7 位尾数。编码器将它拆成两部分：
指数单独组成符号序列，符号位和尾数则合并成一个字节直接存储。
编码器统计当前张量的指数频率，再构建 Huffman 编码表。常见指数使用较短的编码，
不常见指数使用较长的编码，从而减少整个指数序列占用的空间。

Huffman 编码长度不固定，因此解码器无法从任意一位直接识别下一个符号。
EntroPack 为编码区域保存起始位置和符号数量，使不同区域可以并行解码。
这些入口信息和 Huffman 表构成压缩表示中的元数据开销。

## 解码与存储大小

解码器通过 Huffman 表恢复指数序列，再将每个指数与对应的符号位、尾数重新组合，
得到原始 BF16 位表示，并按照保存的形状组织为输出张量。整个过程不涉及数值量化或舍入。

实际存储大小取决于指数分布和解码所需的元数据。指数越集中，通常越容易压缩。
对于较小的张量，元数据占比也会更高。`DFloat11Config` 不设置目标码率，
DFloat11 这一名称也不意味着所有张量都恰好以每参数 11 bit 存储。

## 使用示例

以下示例压缩一个 BF16 张量，并检查解压后的位表示是否与输入一致。
编码区域相关参数见 [Config](../Usage/Configuration.md)。

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
print(f"Stored: {compressed.actual_bpp:.2f} bits per parameter")
```
