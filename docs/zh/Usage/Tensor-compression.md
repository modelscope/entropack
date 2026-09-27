# 通用张量压缩

使用 `compress(tensor, config)` 将权重或其他张量压缩为 `CompressedTensor`，
再通过 `decompress(compressed, config)` 恢复张量。
无损方案逐位还原输入，有损方案返回近似结果。

## 压缩与恢复

解压后的张量默认与输入张量具有相同的形状、`dtype` 和 `device`。
如需指定数据类型或设备，在压缩前用 `tensor.to(dtype=..., device=...)` 转换输入即可。

以下示例使用 4 bpp 的目标码率压缩张量，并计算解压后的相对 L2 误差。
压缩与解压使用同一方案的配置：

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

需要改变码率时，应使用新的 `target_bpp` 重新压缩源张量。

## 查看实际存储

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

`storage_nbytes()` 统计压缩结果的总字节数，包含恢复张量所需的元数据，
`actual_bpp` 等于 `8 * storage_nbytes() / tensor.numel()`。
它们衡量压缩结果的大小，不代表检查点文件大小或运行时峰值显存。

实际码率可能与 `target_bpp` 有所不同，小张量尤其如此。
选择目标码率时，应同时比较实际存储和重建误差。

`CompressedTensor` 还提供 `shape`、`dtype`、`compress_method` 和 `lossless`。
其他属性见 [API 参考](../API_Reference/index.md)。

## 移动压缩张量

`compressed.to(device)` 返回位于指定设备的压缩张量。以下示例先将其转存到 CPU，
再移回 GPU 解压：

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

对 `compressed.to(device)` 返回的压缩张量解压，得到的张量也位于该设备上。

## 保存与加载

保存压缩张量的 `state_dict()`，用 `torch.load(..., weights_only=True)` 加载后，
通过 `CompressedTensor.from_state_dict()` 恢复 `CompressedTensor`：

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

示例将检查点保存到当前目录的 `compressed_tensor.pt`，可按需修改路径。
`map_location` 指定加载设备。建议随检查点保留对应的配置和库版本。

## 输入要求

格量化方案要求非空、有限值组成的二维张量。DFloat11 接受 BF16，
Tile-ANS 接受 [Config](Configuration.md) 中列出的数据类型。无损方案可直接接受高维张量。
使用格量化压缩高维数据时，需先转换为二维布局，并在解压后还原原始形状。

压缩时若出现警告，可检查 `compress_method`：值为 `"raw"` 表示张量未经压缩就被保存。
无效配置或不支持的后端选择会直接报错。
