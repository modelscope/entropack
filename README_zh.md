# EntroPack

### 面向 PyTorch 的通用张量压缩

EntroPack 是一个面向 PyTorch 的通用张量压缩库，支持完整保留原始数据的无损压缩，
以及通过目标码率控制存储大小与重建精度的有损压缩。EntroPack 提供 GPU 编解码，
将压缩后的张量恢复为原来的形状和数据类型。

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg) [![arXiv](https://img.shields.io/badge/arXiv-2609.34185-b31b1b.svg)](https://arxiv.org/abs/2609.34185)

[文档](https://entropackdoc.readthedocs.io/zh-cn/latest/) · [English](README.md)

- **灵活设置码率。** 支持每个权重矩阵以任意非整数目标码率压缩，也可以选择逐位保留输入的无损方案。
- **多种数据类型。** 支持 16 种浮点、整数和布尔类型，例如 FP32、BF16、FP16、FP8、INT8 等。
- **简单易用。** 通过统一接口压缩和恢复张量，使用 `state_dict` 保存；模型权重还可以通过 Compressed Linear 接入。

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

环境要求与使用示例见[快速上手](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Quick-start.html)。

## 快速开始

### 直接压缩张量

以下示例将一个二维 BF16 张量以每参数 3.5 bit 为目标压缩，
再解压为相同形状和数据类型的张量：

```python
import torch
import entropack as ep

tensor = (torch.randn(256, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.LatticeRANSConfig(target_bpp=3.5)

compressed = ep.compress(tensor, config)
restored = ep.decompress(compressed, config)

print(f"Target: {config.target_bpp:.2f} bits per parameter")
print(f"Stored: {compressed.actual_bpp:.2f} bits per parameter")
print(restored.shape, restored.dtype)
```

`target_bpp` 表示目标码率，单位为 bpp（bits per parameter）。对于通用张量，每个张量元素按一个参数计数。
`actual_bpp` 返回包含元数据的实际存储码率。
支持满足 `0.001 <= target_bpp <= 11` 的有限目标值，包括非整数值。
按数据类型限制的目标上限见[压缩配置](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Configuration.html)。

### 使用 Compressed Linear

`CompressedLinear.from_linear` 按传入的 Config 压缩现有 `torch.nn.Linear` 的权重，
返回一个新的 Compressed Linear。仍可像普通线性层一样，通过 `layer(x)` 计算输出：

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

模型中的层替换和检查点操作见 [Compressed Linear 使用指南](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Linear-layers.html)。

### 预量化模型

我们在 [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/EntroPackPreQuants) 提供由 EntroPack 生成的预量化权重包，
覆盖以下代表性扩散模型，可通过 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 的量化功能加载使用：

- **Z-Image-Turbo**：DiT 与文本编码器。
- **Qwen-Image-2.1**：DiT 与文本编码器。
- **MiniMax-H3**：FL2VA 和 Ref2VA 两套 DiT，共享文本编码器与视频 VAE。

各模型的 DiT 和文本编码器均提供 **4/5/6/7/8 bpp** 版本，另有两类特殊权重包：

- **FP8@5**（`dit_fp8_5bpp`）：将 FP8 W8A8 量化后的权重码值经 EntroPack 压缩至 5 bpp，推理使用 FP8 GEMM。
- **极限混合码率**：按层敏感度分配码率，Z-Image-Turbo 达到 **2.3 bpp**，Qwen-Image-2.1 和 MiniMax-H3 达到 **3.0 bpp**。

## Config：压缩配置

Config 定义压缩方案及其参数，在调用张量编解码函数或构造 Compressed Linear 时传入。

| Config | 压缩方式与适用场景 |
| --- | --- |
| `DFloat11Config()` | 专用于 BF16 张量的无损压缩，解压后逐位恢复输入，适合要求精确恢复的场景。 |
| `TileANSConfig()` | 支持 BF16、FP16、FP32、FP8、INT8 等多种数据类型的无损压缩。压缩比取决于输入的数据分布。 |
| `LatticeRANSConfig(target_bpp=...)` | 支持浮点和整数二维张量的有损压缩。`target_bpp` 指定每参数的目标比特数，范围为 [0.001, 11]，支持非整数值，用于调整存储大小与重建精度之间的取舍。 |

方案选择和完整参数见[压缩配置](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Configuration.html)。

## 性能

在单张 NVIDIA H20 上，EntroPack 以 4 bpp 为目标压缩 Z-Image-Turbo 扩散 Transformer
的 276 个线性层权重，耗时 **3.5 秒**。压缩后的实际存储为 **4.02 bpp**，
权重相对 L2 重建误差为 **7.18%**。使用压缩权重推理时，去噪单步耗时为 **544.7 ms**，
相对原始 BF16 模型的 505.7 ms 仅增加 **7.7%**。

## 文档

| 指南 | 内容 |
| --- | --- |
| [快速上手](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Quick-start.html) | 安装并运行张量压缩与 Compressed Linear 示例 |
| [压缩配置](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Configuration.html) | 选择方案、查看支持类型与完整参数 |
| [通用张量压缩](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Tensor-compression.html) | 编解码、存储统计、设备迁移和保存加载 |
| [Compressed Linear 使用指南](https://entropackdoc.readthedocs.io/zh-cn/latest/Usage/Linear-layers.html) | 模型替换、低精度计算和检查点使用 |
| [API 参考](https://entropackdoc.readthedocs.io/zh-cn/latest/API_Reference/index.html) | 查询函数、类与属性 |

压缩原理：[DFloat11](https://entropackdoc.readthedocs.io/zh-cn/latest/Principles/DFloat11.html)、[tile-ANS](https://entropackdoc.readthedocs.io/zh-cn/latest/Principles/Tile-ANS.html)、
[EntroPack 格量化](https://entropackdoc.readthedocs.io/zh-cn/latest/Principles/Lattice-rANS.html)。

## 致谢

EntroPack 的设计受到 [DFloat11](https://github.com/LeanModels/DFloat11)、
[dahuffman](https://github.com/soxofaan/dahuffman)、
[DietGPU](https://github.com/facebookresearch/dietgpu) 和
[tile-ANS](https://arxiv.org/abs/2606.15789) 的启发。

## 许可证

[Apache License 2.0](LICENSE)。

## 引用

```bibtex
@misc{zhang2026entropackfastaccurateentropycoded,
      title={EntroPack: Fast and Accurate Entropy-Coded Weight Compression at Arbitrary Bitrates},
      author={Hong Zhang and Zhongjie Duan and Yingda Chen},
      year={2026},
      eprint={2609.34185},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.34185},
}
```
