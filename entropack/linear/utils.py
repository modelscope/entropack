import torch

_kernels = None
_looked = False
_device_capabilities: dict[int, tuple[int, int]] = {}


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def pad(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    shortfall = size - tensor.shape[dim]
    if shortfall <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = shortfall
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


def load_quant_kernels():
    global _kernels, _looked
    if not _looked:
        _looked = True
        try:
            from . import quant_kernels
            _kernels = quant_kernels
        except ImportError:
            _kernels = None
    return _kernels


def capability_of(device: torch.device) -> tuple[int, int]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = _device_capabilities.get(index)
    if capability is None:
        capability = torch.cuda.get_device_capability(index)
        _device_capabilities[index] = capability
    return capability

__all__ = ["capability_of", "load_quant_kernels", "pad", "round_up"]
