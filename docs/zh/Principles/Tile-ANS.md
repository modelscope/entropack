# Tile-ANS

Tile-ANS 通过编码张量的存储字节实现无损压缩，支持 BF16、FP16、FP32、FP8、INT8 等
浮点和整数类型。它处理的是数值的位表示，不对数值进行近似，因此解压后能够完整恢复原始数据。

## 字节流与概率表

同一种数值格式中，不同字节位置的分布往往不同。Tile-ANS 按字节在元素内部的位置，
将张量拆成多个字节流。例如，两字节格式对应两个流，四字节格式对应四个流。
每个流收集所有元素在对应位置上的字节。

编码器分别统计这些流的字节频率，为每个流构建概率表。
当分布较集中时，常见字节平均使用更短的表示，从而减少存储空间。
对于计入编码开销后压缩收益仍较小的流，编码器直接存储原始字节。
因此，同一个张量中可以同时存在熵编码流和直接存储的流。

## 分块编码与解码

每个流进一步划分为可独立解码的 tile。需要熵编码的 tile 使用范围非对称数字系统 rANS，
通过可逆的整数状态更新编码符号。一个 tile 内部交错使用多个编码状态，支持并行恢复符号。
不同 tile 之间也可以独立解码。同一字节流的所有 tile 共享概率表，无需逐 tile 存储一份表。

解码器使用相同的概率表逆转状态更新，恢复各个经过熵编码的字节流，
再将恢复出的字节流与直接存储的字节流合并，把字节放回元素内的原始位置，重建张量。

该过程不涉及量化。实际大小由字节分布和元数据共同决定，因此不能指定一个有损压缩式的目标码率，
也不保证每个输入都能缩小。较大的 tile 可以降低每元素的元数据开销，较小的 tile 则提供更多独立解码任务。

## 使用示例

以下示例压缩一个 FP16 张量，并检查解压后的位表示是否与输入一致。
分块大小和概率表参数见 [Config](../Usage/Configuration.md)。

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
