import torch

from ..base import Scheme, packed_buffers, register_scheme
from ..checks import prepare_weight
from .format import OPTIONS_BY_DTYPE, PACKED_KEYS, TileBuffers, TileANSConfig, validate_packed


class TileANSScheme(Scheme):
    name = "tile_ans"
    buffer_names = PACKED_KEYS
    priority = 100
    dtypes = tuple(OPTIONS_BY_DTYPE)
    lanes = {"eager": "eager", "cuda": "cuda"}

    def options_for(self, dtype):
        tile_elements, probability_bits = OPTIONS_BY_DTYPE[dtype]
        return {
            "tile_elements": tile_elements, "probability_bits": probability_bits,
            "raw_lane_threshold": TileANSConfig().raw_lane_threshold,
        }

    def encode(self, weight: torch.Tensor, config: TileANSConfig) -> dict:
        weight = prepare_weight(weight, scheme=self.name)
        tile_elements = config.tile_elements
        if tile_elements == 0:
            storage_bytes = weight.numel() * weight.element_size()
            tile_elements = 4096 if storage_bytes <= 32 * 1024 * 1024 else 8192
        device = weight.device
        lane, run_on = self.lane_for(weight, config.execution_backend)
        if run_on is not None and device != run_on:
            weight = weight.to(run_on)
        buffers = lane.encode(
            weight=weight, tile_elements=tile_elements, probability_bits=config.probability_bits,
            raw_lane_threshold=config.raw_lane_threshold, threads_per_block=config.threads_per_block,
        )
        return {key: value.to(device) for key, value in buffers._asdict().items()}

    def validate_buffers(self, buffers, shape, dtype):
        validate_packed(buffers, shape, dtype)

    def decode(self, packed: dict, *, shape: tuple[int, ...], dtype: torch.dtype,
               config: TileANSConfig) -> torch.Tensor:
        buffers = packed_buffers(packed, TileBuffers)
        source = buffers.layout.device
        lane, lane_device = self.lane_for(buffers.layout, config.execution_backend, gate_dtype=False)
        if lane_device is not None and source != lane_device:
            buffers = TileBuffers._make(value.to(lane_device) for value in buffers)
        flat = lane.decode(buffers, dtype=dtype, threads_per_block=config.threads_per_block)
        out = flat.reshape(shape)
        return out if out.device == source else out.to(source)


register_scheme(TileANSScheme())

__all__ = ["TileANSConfig", "TileANSScheme"]
