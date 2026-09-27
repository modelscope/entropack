import torch

from ..base import Scheme, packed_buffers, register_scheme
from ..checks import prepare_weight
from .format import (
    LATTICE_DIM, PACKED_KEYS, SUPPORTED_DTYPES, LatticeBuffers, LatticeRANSConfig, recommended_tile_elements,
    validate_packed,
)

__all__ = ["LatticeRANSScheme", "LatticeRANSConfig"]


class LatticeRANSScheme(Scheme):
    name = "lattice_rans"
    buffer_names = PACKED_KEYS
    lossless = False
    priority = 0
    dtypes = SUPPORTED_DTYPES
    lanes = {"eager": "eager", "cuda": "cuda"}

    def encode(self, weight: torch.Tensor, config: LatticeRANSConfig) -> dict:
        weight = prepare_weight(weight, scheme=self.name, ndim=2, dtypes=SUPPORTED_DTYPES, require_finite=True)
        shape, device = tuple(weight.shape), weight.device
        pad = -shape[1] % LATTICE_DIM
        if pad:
            weight = torch.cat([weight, weight.new_zeros(shape[0], pad)], dim=1)
        lane, run_on = self.lane_for(weight, config.execution_backend)
        if run_on is not None and device != run_on:
            weight = weight.to(run_on)
        tile_elements = (
            recommended_tile_elements(float(config.target_bpp)) if config.tile_elements is None
            else int(config.tile_elements)
        )
        packed = lane.encode(
            weight=weight, target_bpp=float(config.target_bpp),
            prob_bits=None if config.prob_bits in (None, 0) else int(config.prob_bits),
            tile_elements=tile_elements, row_rdo_iterations=config.row_rdo_iterations,
            row_rdo_candidates=config.row_rdo_candidates, scale_search_iterations=config.scale_search_iterations,
            scale_search_max_vectors=config.scale_search_max_vectors,
        )
        return {key: value.to(device) for key, value in packed._asdict().items()}

    def validate_buffers(self, buffers, shape, dtype):
        validate_packed(buffers, tuple(shape), dtype)

    def decode(self, packed: dict, *, shape: tuple[int, ...], dtype: torch.dtype,
               config: LatticeRANSConfig) -> torch.Tensor:
        buffers = packed_buffers(packed, LatticeBuffers)
        source = buffers.layout.device
        lane, lane_device = self.lane_for(buffers.layout, config.execution_backend, gate_dtype=False)
        if lane_device is not None and source != lane_device:
            buffers = LatticeBuffers._make(value.to(lane_device) for value in buffers)
        out = lane.decode(buffers, shape=shape, dtype=dtype, threads_per_block=config.threads_per_block,
                          l2_prefetch=config.l2_prefetch)
        if shape and tuple(out.shape) != shape:
            out = out[: shape[0], : shape[1]]
        return out if out.device == source else out.to(source)


register_scheme(LatticeRANSScheme())
