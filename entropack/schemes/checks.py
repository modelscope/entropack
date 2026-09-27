from collections.abc import Iterable
from numbers import Real

import torch

__all__ = ["prepare_weight"]

_NO_MINMAX_KERNEL = frozenset({
    torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz,
})


def _all_finite(weight: torch.Tensor) -> bool:
    if not weight.dtype.is_floating_point:
        return True
    probe = weight.float() if weight.dtype in _NO_MINMAX_KERNEL else weight
    limit = float(torch.finfo(weight.dtype).max)
    return float(probe.amin()) >= -limit and float(probe.amax()) <= limit


def prepare_weight(
    weight: torch.Tensor, *, scheme: str, ndim: int | None = None, dtypes: Iterable[torch.dtype] | None = None,
    require_finite: bool = False,
) -> torch.Tensor:
    if dtypes is not None and weight.dtype not in frozenset(dtypes):
        raise ValueError(f"{scheme} does not support dtype {weight.dtype}")
    if ndim is not None and weight.ndim != ndim:
        raise ValueError(f"{scheme} encode requires exactly {ndim}D input, got {weight.ndim}D")
    if weight.numel() == 0:
        raise ValueError(f"{scheme} does not support empty tensors")
    if require_finite and not _all_finite(weight):
        raise ValueError(f"{scheme} input must contain only finite values")
    return weight.contiguous()
