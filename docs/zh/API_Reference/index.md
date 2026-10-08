# API 参考

常用函数与类均可通过 `import entropack as ep` 访问。
完整用例见[通用张量压缩](../Usage/Tensor-compression.md)和
[Compressed Linear 使用指南](../Usage/Linear-layers.md)，配置参数见[压缩配置](../Usage/Configuration.md)。

## 张量编解码

### compress

```text
compress(tensor: torch.Tensor, config: CompressionConfig) -> CompressedTensor
```

按 `config` 指定的方案压缩 `tensor`，返回 `CompressedTensor`。
解压后的张量默认与输入具有相同的形状和数据类型。

输入要求由方案决定：DFloat11 接受 BF16 张量，Tile-ANS 支持多种数据类型，格量化要求非空、有限值组成的二维张量。

| 参数 | 含义 |
| --- | --- |
| `tensor` | 待压缩的 PyTorch 张量 |
| `config` | 必传，使用 `DFloat11Config`、`TileANSConfig` 或 `LatticeRANSConfig` 选择方案 |

部分编码失败会发出说明原因的警告，并返回 `compress_method == "raw"` 的未压缩容器。
无效配置或后端选择失败会直接报错。

### decompress

```text
decompress(compressed: CompressedTensor, config: CompressionConfig) -> torch.Tensor
```

返回形状为 `compressed.shape`、数据类型为 `compressed.dtype`、设备为 `compressed.device` 的张量。
未转换输出类型时，无损方案逐位恢复输入值，有损方案返回近似重建。

| 参数 | 含义 |
| --- | --- |
| `compressed` | `compress` 生成或从检查点加载的容器 |
| `config` | 对应压缩方案的配置，解码参数控制恢复过程 |

解码时修改 `target_bpp` 等编码参数不会改变已保存的数据或重新量化张量。

## CompressedTensor

保存一个张量压缩表示的 `torch.Tensor` 子类。通常由 `compress` 返回，或由 `from_state_dict` 从检查点恢复。
进行数值计算前，需先使用 `decompress` 解压。

### 常用属性

| 属性 | 类型 | 含义 |
| --- | --- | --- |
| `shape` | `torch.Size` | 原始张量的形状 |
| `dtype` | `torch.dtype` | 解压后的数据类型 |
| `encoded_dtype` | `torch.dtype` | 编码时的数据类型 |
| `compress_method` | `str` | 容器实际使用的压缩方案 |
| `lossless` | `bool` | 该方案是否无损 |
| `actual_bpp` | `float` | 每参数实际存储比特数，包含元数据 |

`actual_bpp = 8 * storage_nbytes() / math.prod(shape)`。
这项指标衡量压缩表示的大小，不等于检查点文件大小或运行时显存占用。

### 常用方法

| 方法 | 返回值 | 含义 |
| --- | --- | --- |
| `to(...)` | `CompressedTensor` | 改变设备或解压输出类型，不重新压缩；`copy=True` 可复制存储 |
| `storage_nbytes(include_header=True)` | `int` | 压缩结果的总字节数，`include_header=False` 时不计容器头部 |
| `state_dict(prefix="")` | `dict[str, torch.Tensor]` | 将压缩张量导出为可保存的字典 |
| `CompressedTensor.from_state_dict(state, prefix="")` | `CompressedTensor` | 从上述字典恢复容器，不重新压缩 |

保存与加载的 `prefix` 必须一致。可用 `torch.save` 保存字典，并用
`torch.load(..., weights_only=True)` 加载，通过 `map_location` 指定恢复后的设备。
加载后恢复编码时的数据类型；需要其他输出类型时，再调用 `.to(dtype=...)`。

## CompressedLinear

每次前向调用使用重建权重执行线性运算的层。权重以压缩形式保存，偏置不压缩。
运行需要 CUDA GPU 和对应版本的 CuPy。

### 创建层

```text
CompressedLinear(in_features, out_features, bias=True, *,
                 config=None, device=None, dtype=torch.bfloat16)
CompressedLinear.from_linear(linear, **kwargs) -> CompressedLinear
```

| 构造参数 | 含义 |
| --- | --- |
| `in_features` / `out_features` | 输入与输出特征数 |
| `bias` | 是否包含偏置 |
| `config` | 权重压缩配置。默认 `None` 为 BF16 选择 DFloat11，为其他支持的数据类型选择 Tile-ANS |
| `device` | 直接构造时偏置所在的设备 |
| `dtype` | 压缩权重和初始化偏置时使用的数据类型 |

`from_linear` 返回一个新层，压缩源层的权重并复制偏置。源权重必须已加载，不能位于 `meta` 设备。
常用调用为 `ep.CompressedLinear.from_linear(linear, config=config)`。
`kwargs` 可指定 `config` 或 `dtype`，其中 `dtype` 默认沿用源权重类型，设备自动沿用源层。

直接调用构造函数会创建尚无权重数据的层，需要再调用 `compress_weight` 或加载检查点后才能推理。

### 常用方法与属性

| 接口 | 返回值 | 含义 |
| --- | --- | --- |
| `compress_weight(weight)` | `None` | 初始化层内压缩权重，形状应为 `(out_features, in_features)` |
| `dequantize(device=None)` | `torch.Tensor` | 返回数据类型为 `weight.dtype` 的稠密权重；未指定 `device` 时位于层所在设备 |
| `forward(x)` | `torch.Tensor` | 对形状为 `(..., in_features)` 的输入 `x` 执行线性运算，返回形状为 `(..., out_features)` 的张量 |
| `weight` | `CompressedTensor` | 层持有的冻结压缩权重参数 |
| `container_dtype` | `torch.dtype` | 压缩容器中权重或量化码的数据类型 |
| `stored_nbytes` | `int` | 权重存储字节数，含元数据和低精度层的量化尺度，不含偏置 |
| `compressed_bits` | `float` | `8 * stored_nbytes / (in_features * out_features)` |

通过 `layer(x)` 调用前向运算。`.weight` 为压缩张量，需要数值权重时使用 `dequantize()`。

使用标准 `state_dict()` / `load_state_dict()` 保存与恢复层状态。
加载前须创建相同层类型、形状、容器数据类型和压缩方案的层，Config 对象本身不会保存在检查点中。
`.to(device)` 可迁移层，模型的数据类型转换不会重新编码已压缩的权重。

## CompressedFP8Linear 与 CompressedINT8Linear

权重和激活均使用 FP8 或 INT8 的线性层，沿用 `CompressedLinear` 的构造参数、
`from_linear`、存储属性和检查点接口。输入形状为 `(..., in_features)` 时，
输出形状为 `(..., out_features)`，数据类型和设备与输入一致。

| 类 | 权重与激活格式 | CUDA GPU 要求 |
| --- | --- | --- |
| `CompressedFP8Linear` | FP8 E4M3FN | SM8.9 及以上 |
| `CompressedINT8Linear` | INT8 | SM8.0 及以上 |

量化码格式由层类决定，构造参数 `dtype` 不改变 FP8 或 INT8 格式。

`config=None` 时直接保存量化码。指定 `LatticeRANSConfig` 时进一步进行有损压缩，
目标码率须满足 `0.001 <= target_bpp < 8`。`stored_nbytes` 包含重建权重所需的逐行量化尺度。

| 方法 | 返回值 | 含义 |
| --- | --- | --- |
| `codes(device=None)` | FP8 或 INT8 张量 | 恢复量化码，尚未乘回行尺度 |
| `dequantize(device=None)` | `torch.Tensor` | 返回层初始化时的数据类型，`from_linear` 默认沿用原始权重类型 |

两种方法未指定 `device` 时，返回张量均位于层所在设备。有损压缩后的量化码可能与初始量化结果不同。

## Config 类

Config 同时用于张量编解码和 Compressed Linear 的权重存储。以下三个类继承自 `CompressionConfig`：

| 类 | 用途 |
| --- | --- |
| `DFloat11Config` | BF16 无损压缩 |
| `TileANSConfig` | 多种数据类型的分块 ANS 无损压缩 |
| `LatticeRANSConfig` | 以 `target_bpp` 控制码率的格量化有损压缩 |

方案选择、默认值和参数范围见[压缩配置](../Usage/Configuration.md)。
