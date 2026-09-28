# Compressed Linear 使用指南

Compressed Linear 将 PyTorch 线性层替换为使用压缩权重的层。
仍通过 `layer(x)` 调用：输入的最后一维从 `in_features` 变为 `out_features`，
其余维度不变。[Config](Configuration.md) 指定压缩方案和参数。

Compressed Linear 需要 CUDA GPU 和对应版本的 CuPy，安装方式见[快速上手](Quick-start.md)。

`CompressedLinear` 可使用 `DFloat11Config`、`TileANSConfig` 或 `LatticeRANSConfig`，
所选方案需支持权重的数据类型。

## 替换一个已有层

`from_linear` 压缩现有层的权重并返回一个新层，偏置保持原样复制。
对于预训练模型，应先加载检查点，再调用该方法：

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

将返回的层赋给模型中的对应属性后，即可使用压缩版本。

## 替换模型中的多个层

下面的例子递归替换普通线性层，并让输出层保留原来的格式。
`skip` 使用 `named_modules()` 中的模块路径。

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

示例仅选择标准 `torch.nn.Linear`，自定义线性层或共享权重需要结合模型处理。
省略 `skip` 即可压缩所有普通线性层。如果 `CompressedLinear` 无法压缩某层的权重，
替换时会报错；可通过 `skip` 让该层保留原始格式。

## 结合 FP8 或 INT8 计算

| 层 | 权重格式 | 计算方式 |
| --- | --- | --- |
| `CompressedLinear` | 输入权重的数据类型 | 使用激活数据类型进行普通线性运算 |
| `CompressedFP8Linear` | FP8 E4M3FN 量化码 | FP8 权重和激活，需 CUDA GPU（SM8.9 及以上） |
| `CompressedINT8Linear` | INT8 量化码 | INT8 权重和激活，需 CUDA GPU（SM8.0 及以上） |

对于 `CompressedFP8Linear` 和 `CompressedINT8Linear`，`config=None` 仅做 FP8 或 INT8 量化，
传入 `LatticeRANSConfig(target_bpp=...)` 则会对量化后的权重进一步进行有损压缩。
目标码率需大于等于 0.001 bpp 且低于 8 bpp。

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

在支持的硬件上，FP8 可以按同样方式使用 `CompressedFP8Linear`。
需要查看权重时，`codes()` 返回 FP8 或 INT8 数值，`dequantize()` 返回反量化后的浮点数值。

## 统计存储

`stored_nbytes` 统计压缩权重的总字节数，包含元数据及 FP8、INT8 的量化尺度。
`compressed_bits` 等于 `8 * stored_nbytes / (in_features * out_features)`。
统计多个层时，应先分别累加字节数与权重元素数，再计算比例。偏置不计入这项权重存储指标。

这项指标衡量权重存储大小，不代表推理时的峰值显存。

## 保存与加载模型

保存模型的 `state_dict` 后，先构造具有相同结构、使用相同 Compressed Linear 类的模型，再加载状态。
以下示例压缩两个层，并将保存的权重加载到一个新模型中：

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

示例将检查点保存到当前目录的 `compressed_model.pt`，可按需修改路径。
加载已有检查点时，构造 `restored` 模型，再调用 `torch.load` 和 `load_state_dict`。
应随检查点保留模型结构、层名及类型、压缩配置、权重数据类型和库版本。
普通 `torch.nn.Linear` 无法直接加载 Compressed Linear 的检查点。
