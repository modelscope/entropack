from collections.abc import Callable
from dataclasses import dataclass

import torch

from . import cuda


@dataclass(frozen=True)
class Backend:
    name: str
    priority: int = 0
    probe: Callable[[], str | None] | None = None
    device: Callable[[], torch.device] | None = None


BACKENDS: tuple[Backend, ...] = (
    Backend("cuda", priority=100, probe=cuda.probe, device=cuda.runs_on), Backend("eager", priority=0),
)
