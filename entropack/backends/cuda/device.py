from dataclasses import dataclass

import cupy
import torch

from .kernels import device_index

_FALLBACK_WARP_SIZE = 32

_DEFAULT_GRID_WAVES = 8

__all__ = ["DeviceCaps", "caps", "resolve_threads", "validate_threads_per_block"]


def _shared_optin(index: int, properties) -> int:
    optin = getattr(properties, "shared_memory_per_block_optin", 0)
    if optin:
        return int(optin)
    with cupy.cuda.Device(index):
        attributes = cupy.cuda.Device(index).attributes
    return int(attributes["MaxSharedMemoryPerBlockOptin"])


@dataclass(frozen=True)
class DeviceCaps:
    index: int
    name: str
    compute_capability: tuple[int, int]
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    threads_per_sm: int
    regs_per_sm: int
    shared_per_block: int
    shared_optin: int
    shared_per_sm: int
    l2_bytes: int

    def blocks_per_sm(self, threads_per_block: int, shared_per_block: int = 0, regs_per_thread: int = 0) -> int:
        blocks = self.threads_per_sm // threads_per_block
        if shared_per_block > 0:
            blocks = min(blocks, self.shared_per_sm // shared_per_block)
        if regs_per_thread > 0:
            blocks = min(blocks, self.regs_per_sm // (regs_per_thread * threads_per_block))
        return max(1, blocks)

    def resident_blocks(self, threads_per_block: int, shared_per_block: int = 0, regs_per_thread: int = 0) -> int:
        return self.sm_count * self.blocks_per_sm(threads_per_block, shared_per_block, regs_per_thread)

    def grid(self, wanted: int, threads_per_block: int, shared_per_block: int = 0, waves: int = _DEFAULT_GRID_WAVES) -> int:
        limit = self.resident_blocks(threads_per_block, shared_per_block) * waves
        return max(1, min(wanted, limit))

    def shared_limit(self, static_slack: int = 0) -> int:
        return max(0, min(self.shared_optin, self.shared_per_sm - static_slack))

    def threads_per_block(self, wanted: int) -> int:
        usable = min(wanted, self.max_threads_per_block)
        usable -= usable % self.warp_size
        return max(self.warp_size, usable)


_caps_cache: dict[int, DeviceCaps] = {}


def caps(device=None) -> DeviceCaps:
    index = device_index(device)
    cached = _caps_cache.get(index)
    if cached is not None:
        return cached
    properties = torch.cuda.get_device_properties(index)
    queried = DeviceCaps(
        index=index, name=properties.name, compute_capability=(properties.major, properties.minor),
        sm_count=properties.multi_processor_count, warp_size=getattr(properties, "warp_size", 0) or _FALLBACK_WARP_SIZE,
        max_threads_per_block=properties.max_threads_per_block, threads_per_sm=properties.max_threads_per_multi_processor,
        regs_per_sm=properties.regs_per_multiprocessor, shared_per_block=properties.shared_memory_per_block,
        shared_optin=_shared_optin(index, properties), shared_per_sm=properties.shared_memory_per_multiprocessor,
        l2_bytes=getattr(properties, "L2_cache_size", 0),
    )
    _caps_cache[index] = queried
    return queried


def validate_threads_per_block(caps: DeviceCaps, threads_per_block: int) -> None:
    if threads_per_block % caps.warp_size or threads_per_block > caps.max_threads_per_block:
        raise ValueError(
            f"threads_per_block={threads_per_block} is not launchable on {caps.name}: it must be "
            f"a multiple of the {caps.warp_size}-thread warp size and at most {caps.max_threads_per_block}"
        )


def resolve_threads(caps: DeviceCaps, requested: int | None, default: int) -> int:
    if requested is None:
        return caps.threads_per_block(default)
    validate_threads_per_block(caps, requested)
    return requested
