import dataclasses
from numbers import Real
from typing import ClassVar

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

    The frozen ``weight`` is a ``CompressedTensor`` parameter. Each forward
    reconstructs the stored weight, casts it to the activation dtype, and applies
    ``F.linear``. Checkpoints retain the original packed representation.
    CUDA and CuPy are required.

    Args:
        in_features: number of input features.
        out_features: number of output features.
        bias: whether to keep a bias. The bias is not compressed.
        config: compression configuration. ``None`` selects a lossless scheme by dtype.
        device: device for the bias. Compressed buffers retain the source weight's device.
        dtype: container dtype, which must be supported by the selected scheme.
    """

    state_buffer_names: ClassVar[tuple[str, ...]] = ()
    state_prefix: ClassVar[str] = "weight._entropack."

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
        self._init_dtype = require_dtype(dtype)
        if config is not None:
            scheme = get_scheme(name_for_config(config))
            if not scheme.supports(self.container_dtype):
                raise ValueError(f"'{scheme.name}' does not support format {self.container_dtype}")
        for name in self.state_buffer_names:
            self.register_buffer(name, None, persistent=False)

    @property
    def _encode_config(self):
        if self.config is None:
            return _default_config(self.container_dtype, execution_backend=_BACKEND)
        return dataclasses.replace(self.config, execution_backend=_BACKEND)

    @property
    def _decode_config(self):
        if self.config is None:
            return self.weight.scheme.make_config({"execution_backend": _BACKEND})
        return dataclasses.replace(self.config, execution_backend=_BACKEND)

    @property
    def container_dtype(self) -> torch.dtype:
        """The configured compression dtype, unchanged by layer dtype casts."""
        return self._init_dtype

    @property
    def qweight(self) -> torch.Tensor:
        """A packed weight buffer exposed for quantization-framework compatibility."""
        if self.weight is not None:
            for name in self.weight.scheme.buffer_names:
                return self.weight.buffers[name]
        return self.bias

    @property
    def stored_nbytes(self) -> int:
        """Stored weight bytes, including the serialized header and any W8A8 scales."""
        total = self.weight.storage_nbytes()
        for name in self.state_buffer_names:
            buffer = self._buffers.get(name)
            if buffer is not None:
                total += buffer.numel() * buffer.element_size()
        return total

    @property
    def compressed_bits(self) -> float:
        """Stored bits per weight element, including metadata."""
        rows, cols = self.weight.shape
        return self.stored_nbytes * 8 / (rows * cols)

    def compress_weight(self, weight: torch.Tensor) -> None:
        """Initialize the layer with compressed ``weight``."""
        self.weight = torch.nn.Parameter(self._compress_tensor(weight.detach()), requires_grad=False)

    def _compress_tensor(self, tensor: torch.Tensor) -> CompressedTensor:
        compressed = compress(tensor.to(self.container_dtype), self._encode_config)
        if compressed.compress_method == "raw" and "reason" in compressed.header:
            raise RuntimeError(
                f"{compressed.header['requested']} cannot store a {tuple(tensor.shape)} {self.container_dtype} weight for "
                f"{type(self).__name__}: {compressed.header['reason']}"
            )
        return compressed

    def _reconstruct(self, device: str | torch.device | None = None) -> torch.Tensor:
        return decompress(self.weight.to(device=device), self._decode_config)

    def dequantize(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Reconstruct the dense weight on ``device``, defaulting to the weight's device."""
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
        out = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, device=weight.device, **kwargs)
        out.compress_weight(weight.data)
        if linear.bias is not None:
            out.bias = torch.nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        return out

    @property
    def device(self) -> torch.device:
        """The weight device, or the bias device while the weight is not loaded."""
        return self.weight.device if self.weight is not None else self.bias.device if self.bias is not None else torch.device("meta")

    def _bias_on(self, device: torch.device) -> torch.Tensor | None:
        return None if self.bias is None else self.bias.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the weight on ``x``'s device and apply the layer to ``x``."""
        return F.linear(x, self.dequantize(x.device).to(x.dtype), self._bias_on(x.device))

    def extra_repr(self) -> str:
        scheme = "auto" if self.config is None else name_for_config(self.config)
        parts = [f"scheme={scheme}", f"container={str(self.container_dtype).removeprefix('torch.')}"]
        if self.weight is not None:
            parts.append(f"bits={self.compressed_bits:.3f}")
        if self.config is not None:
            parts.append(f"config={type(self.config).__name__}")
        return ", ".join(parts)

    def _apply(self, fn, recurse=True):
        # Preserve auxiliary scale dtypes when Module.to casts the layer.
        held = {name: self._buffers.pop(name) for name in self.state_buffer_names if name in self._buffers}
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            for name, buffer in held.items():
                if buffer is None:
                    self._buffers[name] = None
                    continue
                moved = fn(buffer)
                self._buffers[name] = moved if moved.dtype == buffer.dtype else buffer.to(device=moved.device)
        return self

    def _save_to_state_dict(self, destination: dict, prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Serialize the packed representation instead of the wrapper parameter.
        destination.pop(prefix + "weight", None)
        if self.weight is None:
            return
        container_prefix = prefix + self.state_prefix
        written = self.weight.state_dict(container_prefix)
        for name in self.state_buffer_names:
            buffer = self._buffers[name]
            if buffer is not None:
                written[container_prefix + name] = buffer
        for key, value in written.items():
            destination[key] = value if keep_vars else value.detach()

    def _load_from_state_dict(self, state_dict: dict, prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        container_prefix = prefix + self.state_prefix
        header_key = container_prefix + "header"
        if header_key in state_dict:
            consumed = [header_key]
            try:
                compressed = CompressedTensor.from_state_dict(state_dict, prefix=container_prefix)
                consumed.extend(container_prefix + "buffers." + name for name in compressed.buffers)
                assign = local_metadata.get("assign_to_params_buffers", False)

                # Copy loading keeps the target device and existing Parameter; assign=True adopts the checkpoint tensors.
                if self.weight is not None:
                    device = self.weight.device
                elif self.bias is not None and self.bias.device.type != "meta":
                    device = self.bias.device
                else:
                    device = compressed.device
                if not assign:
                    compressed = compressed.to(device=device, copy=True)
                loaded_weight = torch.nn.Parameter(compressed, requires_grad=False)
                if not assign and self.weight is not None:
                    torch.utils.swap_tensors(self.weight, loaded_weight)
                else:
                    self.weight = loaded_weight

                # Apply the same copy/assign behavior to auxiliary state, such as FP8/INT8 weight scales.
                for name in self.state_buffer_names:
                    key = container_prefix + name
                    if key not in state_dict:
                        raise ValueError(f"compressed state is missing '{key}'")
                    value = state_dict[key]
                    if not assign:
                        if self._buffers[name] is None:
                            value = value.to(device=device, copy=True)
                        else:
                            self._buffers[name].copy_(value)
                            value = self._buffers[name]
                    self._buffers[name] = value
                    consumed.append(key)
            except (TypeError, ValueError) as error:
                error_msgs.append(f"{prefix[:-1]}: {error}")

            # Leave unrecognized keys for the parent's strict checks.
            for key in consumed:
                state_dict.pop(key)
        elif strict:
            missing_keys.append(header_key)

        # Let the parent load bias without expecting a dense weight entry.
        weight = self._parameters.pop("weight")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._parameters["weight"] = weight


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

    def compress_weight(self, weight: torch.Tensor) -> None:
        """Quantize the source weight, compress its codes, and store the row scales."""
        codes, scale = self._quantize_rows(weight.detach())
        self.weight = torch.nn.Parameter(self._compress_tensor(codes), requires_grad=False)
        self.weight_scale = scale

    def codes(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Decode the weight codes in ``code_dtype`` for low-precision matrix multiplication."""
        return decompress(self.weight.to(device=device, dtype=self.code_dtype), self._decode_config)

    def dequantize(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Reconstruct dense weights in the dtype used to initialize the layer."""
        codes = self.codes(device)
        return (codes.float() * self.weight_scale.to(codes.device).unsqueeze(1)).to(self._init_dtype)

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
        return (max(tokens, self.min_tokens), round_up(self.out_features, self.code_alignment), round_up(self.in_features, self.code_alignment))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the low-precision linear operation.

        The layer recovers its FP8 or INT8 codes, quantizes activations per row, and runs the
        corresponding matrix multiplication. Input gradients use the reconstructed numerical
        weight. The compressed base weights remain frozen.
        """
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
        out = self._gemm(padded_activation, padded_codes, pad(scale, 0, rows), pad(weight_scale, 0, outs), x.dtype)
        return out[:tokens, : self.out_features].reshape(*x.shape[:-1], self.out_features)

    def _bias_padded(self, device: torch.device, columns: int) -> torch.Tensor | None:
        bias = self._bias_on(device)
        return None if bias is None else pad(bias, 0, columns)

    def _epilogue(self, product: torch.Tensor, activation_scale: torch.Tensor, weight_scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
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
    :class:`~entropack.LatticeRANSConfig` additionally compresses them at targets from 0.001 up
    to, but excluding, 8 bits per parameter. Inference decodes the FP8 codes before matrix
    multiplication. Stored size also includes metadata and per-row weight scales.
    """

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
    them at targets from 0.001 up to, but excluding, 8 bits per parameter. Stored size also
    includes metadata and per-row weight scales.
    """

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


__all__ = ["CompressedFP8Linear", "CompressedINT8Linear", "CompressedLinear", "QuantizedLinear"]
