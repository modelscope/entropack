# 压缩配置

Config 选择压缩方案并设置编解码参数。`execution_backend` 默认为 `"auto"`，自动选择计算后端。

## 选择配置

| Config | 压缩方式 | 输入要求 |
| --- | --- | --- |
| `DFloat11Config()` | BF16 无损压缩 | BF16 张量 |
| `TileANSConfig()` | 多种数据类型的无损压缩 | 支持的数据类型 |
| `LatticeRANSConfig(target_bpp=...)` | 按目标码率进行有损压缩 | 非空、有限值组成的二维张量 |

两个无损方案均逐位恢复输入，压缩后的大小取决于张量的数据分布。`TileANSConfig` 也支持 BF16，
因此 BF16 张量可以选择其中任一无损方案。`LatticeRANSConfig` 接受每元素 1–11 bit 的目标码率，
支持非整数值，实际存储码率可通过 `CompressedTensor.actual_bpp` 查看。

## 支持的张量数据类型

| 数据类型 | `DFloat11Config` | `TileANSConfig` | `LatticeRANSConfig` |
|---|---|---|---|
| `float32` | — | 无损 | 有损 |
| `float16` | — | 无损 | 有损 |
| `bfloat16` | 无损 | 无损 | 有损 |
| `float8_e4m3fn` | — | 无损 | 有损 |
| `float8_e4m3fnuz` | — | 无损 | 有损 |
| `float8_e5m2` | — | 无损 | 有损 |
| `float8_e5m2fnuz` | — | 无损 | 有损 |
| `int64` | — | 无损 | 有损 |
| `int32` | — | 无损 | 有损 |
| `int16` | — | 无损 | 有损 |
| `int8` | — | 无损 | 有损 |
| `uint64` | — | 无损 | 有损 |
| `uint32` | — | 无损 | 有损 |
| `uint16` | — | 无损 | 有损 |
| `uint8` | — | 无损 | 有损 |
| `bool` | — | 无损 | 有损 |

无损方案接受多种形状的张量，格量化要求二维输入。文中的 BF16、FP16、FP8 对应上表列出的
PyTorch 数据类型。打包的四比特格式、FP64 和复数类型不在支持范围内。

## 参数说明

修改编码参数只影响后续压缩，不会改变已有压缩结果。解码参数在恢复张量时生效。
多数设置可保留默认值。
有损压缩的存储码率由 `target_bpp` 控制，`row_rdo_iterations` 设为正数时启用逐行率失真优化（RDO）。

## CompressionConfig

以下执行参数适用于三个配置类。

| 字段 | 类型 | 默认值 | 阶段 | 含义 |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` 或 `None` | `"auto"` | 编解码 | `"auto"` 或 `None` 优先选择 CUDA，不可用时回退到 PyTorch 实现；`"cuda"` 强制使用 CUDA。`"eager"` 为张量编解码的 PyTorch 后备实现。 |

## DFloat11Config

用于 BF16 无损压缩。

| 字段 | 类型 | 默认值 | 阶段 | 含义 |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` 或 `None` | `"auto"` | 编解码 | 后端选择，含义同上 |
| `bytes_per_thread` | 正整数或 `None` | `16` | 编码 | 每个线程处理的编码字节数，影响压缩率和解码并行度 |
| `threads_per_block` | 正整数或 `None` | `128` | 编码 | 编码时每个线程块的线程数 |

## TileANSConfig

用于支持的数据类型的无损压缩。

| 字段 | 类型 | 默认值 | 阶段 | 含义 |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` 或 `None` | `"auto"` | 编解码 | 后端选择，含义同上 |
| `tile_elements` | [0, 2^31 − 1] 内整数 | `0` | 编码 | 每个压缩块的元素数，影响压缩率和解码并行度。`0` 自动选择。 |
| `probability_bits` | `0`、`9`、`10`、`11`、`12` | `0` | 编码 | 概率表精度，`0` 自动选择 |
| `raw_lane_threshold` | [0, 8] 内浮点数 | `7.9` | 编码 | 决定何时直接存储难以压缩的数据，单位为每字节的预计编码比特数 |
| `threads_per_block` | 正整数或 `None` | `None` | 编解码 | GPU 线程块宽度，`None` 自动选择 |

## LatticeRANSConfig

用于有限值组成的二维张量的有损压缩。

| 字段 | 类型 | 默认值 | 阶段 | 含义 |
| --- | --- | --- | --- | --- |
| `execution_backend` | `str` 或 `None` | `"auto"` | 编解码 | 后端选择，含义同上 |
| `target_bpp` | [1, 11] 内浮点数 | `4.0` | 编码 | 每个输入元素的目标比特数，支持非整数。实际码率通过 `actual_bpp` 查看。 |
| `prob_bits` | [9, 15] 内整数、`0` 或 `None` | `None` | 编码 | 概率表精度，`None` 或 `0` 自动选择 |
| `tile_elements` | 正整数或 `None` | `None` | 编码 | 每个压缩块的元素数，影响压缩率和解码并行度。`None` 自动选择。 |
| `row_rdo_iterations` | [0, 8] 内整数 | `0` | 编码 | 逐行率失真优化的轮数，`0` 关闭。更多轮次会增加压缩耗时。 |
| `row_rdo_candidates` | 正整数 | `5` | 编码 | RDO 为每行比较的候选量化结果数，更多候选会增加压缩耗时 |
| `scale_search_iterations` | 正整数 | `12` | 编码 | 量化尺度搜索的迭代次数 |
| `scale_search_max_vectors` | 正整数 | `262144` | 编码 | 码率搜索的采样上限，每个向量包含八个元素 |
| `threads_per_block` | 正整数或 `None` | `None` | 解码 | GPU 线程块宽度，`None` 自动选择 |
| `l2_prefetch` | `bool` | `True` | 解码 | 解码时启用 GPU L2 缓存预取 |

## 压缩原理

各方案的编解码过程分别见 [DFloat11](../Principles/DFloat11.md)、
[Tile-ANS](../Principles/Tile-ANS.md) 和 [Lattice-rANS](../Principles/Lattice-rANS.md)。
