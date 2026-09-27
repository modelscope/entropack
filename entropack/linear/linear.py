import copy
import dataclasses
from numbers import Real
from typing import Any, ClassVar

import torch
from torch.nn import functional as F

from ..compression import CompressedTensor, compress, decompress
from ..registry import default_config as _default_config, get_scheme, name_for_config, require_dtype
from ..schemes import LatticeRANSConfig, RawConfig
from .utils import capability_of, load_quant_kernels, pad, round_up

_EPS = torch.finfo(torch.float32).eps
_BACKEND = "cuda"


class CompressedLinear(torch.nn.Linear):
    """A linear layer with compressed weights, reconstructed during each forward call.

    The layer registers compressed buffers for ``state_dict``, device transfers, and
    ``deepcopy``. Its dense ``weight`` parameter is ``None``. Each forward reconstructs a
    temporary weight, casts it to the activation dtype, and applies ``F.linear``.
    CUDA and CuPy are required.

    Args:
        in_features: number of input features.
        out_features: number of output features.
        bias: whether to keep a bias. The bias is not compressed.
        config: compression configuration. ``None`` selects a lossless scheme by dtype.
        device: device for the bias. Compressed buffers retain the source weight's device.
        dtype: container dtype, which must be supported by the selected scheme."""

    state_buffer_names: ClassVar[tuple[str, ...]] = ()
    state_prefix: ClassVar[str] = "_entropack."
    accepts_raw_container: ClassVar[bool] = False

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, *, config=None,
        device: str | torch.device | None = None, dtype: torch.dtype = torch.bfloat16,
    ):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=False, dtype=dtype)
        self.weight = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False)
        self.config = config
        self._container_dtype = require_dtype(dtype)
        if config is not None:
            scheme = get_scheme(name_for_config(config))
            if not scheme.supports(self.container_dtype):
                raise ValueError(f"'{scheme.name}' does not support format {self.container_dtype}")
        self._compressed = None
        for name in self.state_buffer_names:
            self.register_buffer(name, None, persistent=False)

    @property
    def scheme_name(self) -> str:
        """The scheme the stored container uses, else the one the config names, else ``"auto"``."""
        if self._compressed is not None:
            return self._compressed.compress_method
        return "auto" if self.config is None else name_for_config(self.config)

    @property
    def _encode_config(self):
        if self.config is None:
            return _default_config(self.container_dtype, execution_backend=_BACKEND)
        return dataclasses.replace(self.config, execution_backend=_BACKEND)

    @property
    def _decode_config(self):
        if self.config is None:
            return self._compressed.scheme.make_config({"execution_backend": _BACKEND})
        return dataclasses.replace(self.config, execution_backend=_BACKEND)

    @property
    def container_dtype(self) -> torch.dtype:
        """The format the weight is stored in, which the scheme has to serve."""
        return self._container_dtype

    @property
    def buffer_names(self) -> tuple[str, ...]:
        """The stored container's buffer names; empty while the layer holds no weight."""
        return () if self._compressed is None else tuple(self._compressed.scheme.buffer_names)

    @property
    def qweight(self) -> torch.Tensor:
        """One tensor standing in for the weight, for a caller that must move or measure it."""
        for name in self.buffer_names:
            buffer = self._buffers.get(name)
            if buffer is not None:
                return buffer
        return self.bias

    @property
    def compressed_weight(self) -> CompressedTensor:
        """The stored container."""
        if self._compressed is None:
            raise RuntimeError(f"{type(self).__name__} has no compressed weight; load one or call compress_weight")
        return self._compressed

    @property
    def stored_nbytes(self) -> int:
        """Bytes this layer's weight occupies: the container, its serialized header, and any W8A8 scale."""
        total = self.compressed_weight.storage_nbytes()
        for name in self.state_buffer_names:
            buffer = self._buffers.get(name)
            if buffer is not None:
                total += buffer.numel() * buffer.element_size()
        return total

    @property
    def compressed_bits(self) -> float:
        """Bits per element of the source weight's shape, which is what a network rate is built from."""
        rows, cols = self.compressed_weight.shape
        return self.stored_nbytes * 8 / (rows * cols)

    @property
    def _held_buffers(self) -> tuple[str, ...]:
        return self.buffer_names + self.state_buffer_names

    def _holds(self, method: str) -> bool:
        return self.config is None or method == self.scheme_name or (
            self.accepts_raw_container and method == "raw"
        )

    def compress_weight(self, weight: torch.Tensor) -> None:
        """Compress ``weight`` and store it, replacing whatever the layer held."""
        self._prepare(weight.detach())

    def _prepare(self, weight: torch.Tensor) -> None:
        self.set_compressed(self._container_for(weight))

    def _container_for(self, tensor: torch.Tensor) -> CompressedTensor:
        compressed = compress(tensor.to(self.container_dtype), self._encode_config)
        if compressed.compress_method == "raw" and not self.accepts_raw_container:
            raise RuntimeError(
                f"{self.scheme_name} cannot store a {tuple(tensor.shape)} {self.container_dtype} weight for "
                f"{type(self).__name__}: {compressed.header.get('reason', 'asked to store the weight verbatim')}"
            )
        return compressed

    def set_compressed(self, compressed: CompressedTensor) -> None:
        """Adopt a container built elsewhere, such as one read from a checkpoint.

        It has to hold this layer's shape and container format and use the scheme this layer's config
        names, so a container written for another layer cannot be loaded into this one by accident.
        """
        if not isinstance(compressed, CompressedTensor):
            raise TypeError(f"expected an entropack CompressedTensor, got {type(compressed).__name__}")
        expected = (self.out_features, self.in_features)
        if compressed.shape != expected:
            raise ValueError(f"compressed weight shape {compressed.shape} does not match this Linear's {expected}")
        if not self._holds(compressed.compress_method):
            raise ValueError(
                f"compressed weight uses '{compressed.compress_method}', this Linear is '{self.scheme_name}'"
            )
        if compressed.dtype != self.container_dtype:
            raise ValueError(
                f"compressed weight holds {compressed.dtype}, this Linear is configured for {self.container_dtype}"
            )
        names = tuple(compressed.scheme.buffer_names)
        self._compressed = CompressedTensor(
            header=compressed.header, buffers={name: compressed.buffers[name] for name in names},
            shape=compressed.shape, dtype=compressed.dtype,
        )
        for name in names:
            self.register_buffer(name, self._compressed.buffers[name], persistent=False)

    def _container_on(self, device: str | torch.device | None) -> CompressedTensor:
        compressed = self.compressed_weight
        if device is None or compressed.buffers[self.buffer_names[0]].device == torch.device(device):
            return compressed
        return compressed.to(device)

    def _reconstruct(self, device: str | torch.device | None = None) -> torch.Tensor:
        return decompress(self._container_on(device), self._decode_config)

    def dequantize(self, device: str | torch.device | None = None) -> torch.Tensor:
        """The dense weight, reconstructed on ``device``, or on the container's device by default."""
        return self._reconstruct(device)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, **kwargs) -> "CompressedLinear":
        """Compress an existing layer's weight into a new layer of this class.

        Args:
            linear: the source layer, with its weight materialized -- so this runs after a
                ``load_state_dict``, not on a meta-device skeleton.
            **kwargs: passed to the constructor; ``dtype`` defaults to the source weight's.

        Returns:
            A layer of this class holding the compressed weight and, when the source had one, a copy
            of its bias.
        """
        weight = linear.weight
        if weight is None or weight.device.type == "meta":
            raise ValueError("cannot compress a Linear whose weight is not materialized")
        kwargs.setdefault("dtype", weight.dtype)
        out = cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None, device=weight.device, **kwargs,
        )
        out.compress_weight(weight.data)
        if linear.bias is not None:
            out.bias = torch.nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        return out

    @property
    def device(self) -> torch.device:
        """The device the layer's buffers are on, or ``meta`` while it holds none."""
        for name in self._held_buffers:
            buffer = self._buffers.get(name)
            if buffer is not None:
                return buffer.device
        return self.bias.device if self.bias is not None else torch.device("meta")

    def _bias_on(self, device: torch.device) -> torch.Tensor | None:
        return None if self.bias is None else self.bias.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the weight on ``x``'s device and apply the layer to ``x``."""
        return F.linear(x, self.dequantize(x.device).to(x.dtype), self._bias_on(x.device))

    def extra_repr(self) -> str:
        parts = [f"scheme={self.scheme_name}", f"container={str(self.container_dtype).removeprefix('torch.')}"]
        if self._compressed is not None:
            parts.append(f"bits={self.compressed_bits:.3f}")
        if self.config is not None:
            parts.append(f"config={type(self.config).__name__}")
        return ", ".join(parts)

    def _sync_compressed(self) -> None:
        if self._compressed is not None:
            self._compressed.buffers = {name: self._buffers[name] for name in self.buffer_names}

    def _apply(self, fn, recurse=True):
        held = {name: self._buffers.pop(name) for name in self._held_buffers if name in self._buffers}
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            for name, buffer in held.items():
                if buffer is None:
                    self._buffers[name] = None
                    continue
                moved = fn(buffer)
                self._buffers[name] = moved if moved.dtype == buffer.dtype else buffer.to(device=moved.device)
        self._sync_compressed()
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "CompressedLinear":
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            clone.__dict__[key] = copy.deepcopy(value, memo)
        clone._sync_compressed()
        return clone

    def _save_to_state_dict(self, destination: dict, prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._compressed is None:
            return
        container_prefix = prefix + self.state_prefix
        written = self._compressed.state_dict(container_prefix)
        for name in self.state_buffer_names:
            buffer = self._buffers[name]
            if buffer is not None:
                written[container_prefix + name] = buffer
        for key, value in written.items():
            destination[key] = value if keep_vars else value.detach()

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        container_prefix = prefix + self.state_prefix
        header_key = container_prefix + "header"
        if header_key in state_dict:
            try:
                self.set_compressed(CompressedTensor.from_state_dict(state_dict, prefix=container_prefix))
                for name in self.state_buffer_names:
                    key = container_prefix + name
                    if key not in state_dict:
                        raise ValueError(f"compressed state is missing '{key}'")
                    self._buffers[name] = state_dict[key]
            except (TypeError, ValueError) as error:
                error_msgs.append(f"{prefix[:-1]}: {error}")
            for key in [key for key in state_dict if key.startswith(container_prefix)]:
                state_dict.pop(key)
        elif strict:
            missing_keys.append(header_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )


class _W8A8LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, layer):
        ctx.layer = layer
        ctx.x_shape = x.shape
        return layer._forward8(x.detach())

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        grad = grad_output.reshape(-1, grad_output.shape[-1])
        weight = ctx.layer.dequantize(grad.device).to(grad.dtype)
        return torch.mm(grad, weight).reshape(ctx.x_shape), None


class QuantizedLinear(CompressedLinear):

    code_dtype: ClassVar[torch.dtype]
    code_max: ClassVar[float]
    rounds_to_integer: ClassVar[bool]
    code_alignment: ClassVar[int]
    min_tokens: ClassVar[int]
    min_capability: ClassVar[tuple[int, int]]
    state_buffer_names = ("weight_scale",)
    accepts_raw_container = True

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, *, config=None,
        device: str | torch.device | None = None, dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(in_features, out_features, bias=bias, config=self._coded(config), device=device, dtype=dtype)

    @property
    def container_dtype(self) -> torch.dtype:
        return self.code_dtype

    @classmethod
    def _coded(cls, config):
        if config is None:
            return RawConfig()
        if isinstance(config, RawConfig):
            return config
        if not isinstance(config, LatticeRANSConfig):
            raise TypeError(
                f"{cls.__name__} stores {cls.code_dtype} codes: either uncoded (RawConfig) or lattice-coded "
                f"(LatticeRANSConfig), got {type(config).__name__}"
            )
        rate = config.target_bpp
        if isinstance(rate, bool) or not isinstance(rate, Real) or not 0.0 < float(rate) < 8.0:
            raise ValueError(
                f"{cls.__name__} stores {cls.code_dtype} codes, 8 bits wide, so a target_bpp "
                f"only pays below 8; got {rate!r}. Pass RawConfig to store the codes uncoded."
            )
        return config

    def _quantize_rows(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = tensor.reshape(-1, tensor.shape[-1]).float()
        scale = (flat.abs().amax(dim=1) / self.code_max).clamp(min=_EPS)
        scaled = flat / scale.unsqueeze(1)
        if self.rounds_to_integer:
            scaled = scaled.round()
        return scaled.clamp(-self.code_max, self.code_max).to(self.code_dtype), scale

    def _prepare(self, weight: torch.Tensor) -> None:
        codes, scale = self._quantize_rows(weight)
        self.set_compressed(self._container_for(codes))
        self.weight_scale = scale

    def codes(self, device: str | torch.device | None = None) -> torch.Tensor:
        """The stored codes as a dense tensor. The weight is these times their per-row scale."""
        return self._reconstruct(device)

    def dequantize(self, device: str | torch.device | None = None) -> torch.Tensor:
        """The dense weight: :meth:`codes` times the per-row scale the quantization fitted."""
        codes = self.codes(device)
        return codes.float() * self.weight_scale.to(codes.device).unsqueeze(1)

    def _require_8bit_gemm(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(
                f"{type(self).__name__} needs a CUDA device with an 8-bit tensor core, compute capability "
                f"{self.min_capability} or later; got {device}."
            )
        capability = capability_of(device)
        if capability < self.min_capability:
            raise RuntimeError(
                f"{type(self).__name__} needs compute capability {self.min_capability} or later and "
                f"{torch.cuda.get_device_name(device)} reports {capability}. This format has no path on "
                "older hardware."
            )

    def _quantize_activation(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kernels = load_quant_kernels()
        if kernels is None:
            return self._quantize_rows(flat)
        return kernels.quantize_rows(flat, self.code_dtype, self.code_max, self.rounds_to_integer)

    def _gemm_shapes(self, tokens: int) -> tuple[int, int, int]:
        return (max(tokens, self.min_tokens), round_up(self.out_features, self.code_alignment),
                round_up(self.in_features, self.code_alignment))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the low-precision linear operation.

        The layer recovers its FP8 or INT8 codes, quantizes activations per row, and runs the
        corresponding matrix multiplication. Input gradients use the reconstructed numerical
        weight. The compressed base weights remain frozen."""
        self._require_8bit_gemm(x.device)
        if x.requires_grad:
            return _W8A8LinearFunction.apply(x, self)
        return self._forward8(x)

    def _forward8(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        activation, scale = self._quantize_activation(flat.detach())
        codes = self.codes(x.device)
        weight_scale = self.weight_scale.to(codes.device)
        tokens = activation.shape[0]
        rows, outs, cols = self._gemm_shapes(tokens)
        padded_activation = pad(pad(activation, 0, rows), 1, cols)
        padded_codes = pad(pad(codes, 0, outs), 1, cols)
        out = self._gemm(padded_activation, padded_codes, pad(scale, 0, rows),
                         pad(weight_scale, 0, outs), x.dtype)
        return out[:tokens, : self.out_features].reshape(*x.shape[:-1], self.out_features)

    def _bias_padded(self, device: torch.device, columns: int) -> torch.Tensor | None:
        bias = self._bias_on(device)
        return None if bias is None else pad(bias, 0, columns)

    def _epilogue(
        self, product: torch.Tensor, activation_scale: torch.Tensor, weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        out = product.to(torch.float32)
        out.mul_(activation_scale.unsqueeze(1))
        bias = self._bias_padded(out.device, out.shape[1])
        if bias is None:
            return out.mul_(weight_scale).to(out_dtype)
        return torch.addcmul(bias.to(torch.float32), out, weight_scale).to(out_dtype)

    def _gemm(
        self, activation: torch.Tensor, codes: torch.Tensor, activation_scale: torch.Tensor,
        weight_scale: torch.Tensor, out_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError


class CompressedFP8Linear(QuantizedLinear):
    """A linear layer using FP8 E4M3FN weights and activations.

    Requires CUDA compute capability 8.9 or later and uses ``torch._scaled_mm``.
    With no compression configuration, quantized codes are stored directly.
    :class:`~entropack.LatticeRANSConfig` additionally compresses them at targets from 1 up
    to, but excluding, 8 bits per element. Inference decodes the FP8 codes before matrix
    multiplication. Stored size also includes metadata and per-row weight scales."""

    code_dtype = torch.float8_e4m3fn
    code_max = float(torch.finfo(torch.float8_e4m3fn).max)
    rounds_to_integer = False
    code_alignment = 16
    min_tokens = 1
    min_capability = (8, 9)

    def _gemm(self, activation, codes, activation_scale, weight_scale, out_dtype):
        bias = self._bias_padded(codes.device, weight_scale.shape[0])
        fused = bias is None or out_dtype is not torch.float32
        out = torch._scaled_mm(
            activation, codes.t(), scale_a=activation_scale.unsqueeze(1), scale_b=weight_scale.unsqueeze(0),
            bias=bias if fused else None, out_dtype=out_dtype,
        )
        return out if fused else out.add_(bias)


class CompressedINT8Linear(QuantizedLinear):
    """A linear layer using symmetric INT8 weights and activations.

    Requires CUDA compute capability 8.0 or later. Matrix multiplication uses Triton when
    available and ``torch._int_mm`` otherwise. With no compression configuration,
    quantized codes are stored directly. :class:`~entropack.LatticeRANSConfig` additionally compresses
    them at targets from 1 up to, but excluding, 8 bits per element. Stored size also
    includes metadata and per-row weight scales."""

    code_dtype = torch.int8
    code_max = 127.0
    rounds_to_integer = True
    code_alignment = 8
    min_tokens = 17
    min_capability = (8, 0)

    def _gemm_shapes(self, tokens):
        if load_quant_kernels() is not None:
            return tokens, self.out_features, self.in_features
        return super()._gemm_shapes(tokens)

    def _gemm(self, activation, codes, activation_scale, weight_scale, out_dtype):
        bias = self._bias_padded(codes.device, weight_scale.shape[0])
        kernels = load_quant_kernels()
        if kernels is not None:
            return kernels.int8_gemm(activation, codes, activation_scale, weight_scale, bias, out_dtype)
        return self._epilogue(torch._int_mm(activation, codes.t()), activation_scale, weight_scale, out_dtype)

__all__ = [
    "CompressedFP8Linear", "CompressedINT8Linear", "CompressedLinear", "QuantizedLinear",
]
