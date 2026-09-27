import torch

from ..base import Scheme, packed_buffers, register_scheme
from ..checks import prepare_weight
from .eager import get_32bit_codec, get_luts
from .format import PACKED_KEYS, DFloat11Buffers, DFloat11Config, validate_packed


class DFloat11Scheme(Scheme):
    name = "dfloat11"
    buffer_names = PACKED_KEYS
    priority = 200
    dtypes = (torch.bfloat16,)
    lanes = {"eager": "eager", "cuda": "cuda"}

    def encode(self, weight: torch.Tensor, config: DFloat11Config) -> dict:
        weight = prepare_weight(weight, scheme=self.name, dtypes=self.dtypes)
        device = weight.device
        flat = weight.reshape(-1)

        lane, run_on = self.lane_for(flat, config.execution_backend)
        if run_on is not None and device != run_on:
            flat = flat.to(run_on)
        counter = lane.exponent_counter(flat, config.threads_per_block)
        codec, _counter, table = get_32bit_codec(counter)
        luts = get_luts(table)

        buffers = lane.encode(
            weight=flat, codec=codec, luts=luts, bytes_per_thread=config.bytes_per_thread,
            threads_per_block=config.threads_per_block,
        )

        return {key: value.to(device) for key, value in buffers._asdict().items()}

    def validate_buffers(self, buffers, shape, dtype):
        validate_packed(buffers, shape)

    def decode(self, packed: dict, *, shape: tuple[int, ...], dtype: torch.dtype,
               config: DFloat11Config) -> torch.Tensor:
        buffers = packed_buffers(packed, DFloat11Buffers)
        source = buffers.layout.device
        lane, lane_device = self.lane_for(buffers.layout, config.execution_backend, gate_dtype=False)
        if lane_device is not None and source != lane_device:
            buffers = DFloat11Buffers._make(value.to(lane_device) for value in buffers)
        flat = lane.decode(buffers)
        out = flat.reshape(shape)
        return out if out.device == source else out.to(source)


register_scheme(DFloat11Scheme())

__all__ = ["DFloat11Config", "DFloat11Scheme"]
