# 快速上手

本页介绍安装、张量编解码和 Compressed Linear 的基本用法。

## 安装

需要 Python 3.10 及以上，并先安装与环境匹配的 CUDA 版 PyTorch 2.10 及以上。

### 源码安装（推荐）

```bash
git clone https://github.com/modelscope/entropack.git
cd entropack
pip install -e ".[cuda13]"
```

### 从 PyPI 安装

PyPI 版本更新可能有所延迟，如需最新功能，推荐从源码安装。

```bash
pip install "entropack[cuda13]"
```

上述两种安装方式均以 CUDA 13 为例，并包含对应版本的 CuPy。使用 CUDA 12 时，
将命令中的 `cuda13` 改为 `cuda12`。如果已安装匹配的 CuPy，源码安装和 PyPI 安装
可分别使用 `pip install -e .` 和 `pip install entropack`。

PyTorch 的 CUDA 版本需在安装 PyTorch 时选定。
INT8 计算可通过 `pip install triton` 启用可选的 Triton 内核。
FP8 和 INT8 的额外硬件要求见 [Compressed Linear 使用指南](Linear-layers.md)。
首次调用需要编译 CUDA 内核，可能比后续调用更慢，性能计时应在预热后进行。

## 直接压缩张量

以下示例使用 `LatticeRANSConfig(target_bpp=3.5)` 选择有损压缩，将二维 BF16 张量
压缩到每元素 3.5 bit 的目标码率，再恢复为原来的形状和数据类型。

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

`target_bpp` 的单位为每元素比特数（bpp），可设为 1–11 范围内的整数或非整数值。
`actual_bpp` 返回包含元数据的实际存储码率，可能与目标不同，尤其在张量较小时。
该方案要求输入为非空的二维张量，且不含 NaN 或无穷值。

重建误差、设备迁移和保存加载见[通用张量压缩](Tensor-compression.md)。
无损方案及完整参数见[压缩配置](Configuration.md)。

## 使用 Compressed Linear

`CompressedLinear.from_linear` 压缩现有层的权重并返回一个新层，偏置保持原样复制：

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

对于预训练模型，应先加载检查点，再将模型中的对应层替换为新层。
[Compressed Linear 使用指南](Linear-layers.md)介绍多个层的替换、
低精度计算和压缩检查点的保存加载。
