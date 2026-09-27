from importlib import metadata

try:
    __version__ = metadata.version("entropack")
except metadata.PackageNotFoundError:
    __version__ = "0+unknown"

from .compression import CompressedTensor, compress, decompress
from .linear import CompressedFP8Linear, CompressedINT8Linear, CompressedLinear
from .schemes import CompressionConfig, DFloat11Config, LatticeRANSConfig, RawConfig, TileANSConfig

__all__ = [
    "CompressedFP8Linear", "CompressedINT8Linear", "CompressedLinear", "CompressedTensor", "CompressionConfig",
    "DFloat11Config", "LatticeRANSConfig", "RawConfig", "TileANSConfig", "__version__", "compress", "decompress",
]
