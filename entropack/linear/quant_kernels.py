
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

_INT8_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=_INT8_GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _int8_gemm_kernel(
    A, B, A_SCALE, B_SCALE, BIAS, C, M, N, K, stride_am, stride_bn,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N

    a_ptr = A + rm[:, None] * stride_am + rk[None, :]
    b_ptr = B + rn[:, None] * stride_bn + rk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        mask_k = (k + rk) < K
        a = tl.load(a_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptr, mask=mask_n[:, None] & mask_k[None, :], other=0)
        acc += tl.dot(a, tl.trans(b))
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K

    out = acc.to(tl.float32)
    out = out * tl.load(A_SCALE + rm, mask=mask_m, other=0.0)[:, None]
    out = out * tl.load(B_SCALE + rn, mask=mask_n, other=0.0)[None, :]
    if HAS_BIAS:
        out += tl.load(BIAS + rn, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    tl.store(C + rm[:, None] * N + rn[None, :], out.to(C.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _quantize_rows_kernel(
    X, OUT, SCALE, K, stride_xm, CODE_MAX: tl.constexpr, ROUNDS: tl.constexpr, EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    base = X + row * stride_xm
    amax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        x = tl.load(base + rk, mask=rk < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    scale = tl.maximum(amax / CODE_MAX, EPS)

    out_base = OUT + row * K
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        x = tl.load(base + rk, mask=rk < K, other=0.0).to(tl.float32)
        scaled = x / scale
        if ROUNDS:
            scaled = libdevice.rint(scaled)
        scaled = tl.minimum(tl.maximum(scaled, -CODE_MAX), CODE_MAX)
        tl.store(out_base + rk, scaled.to(OUT.dtype.element_ty), mask=rk < K)
    tl.store(SCALE + row, scale)

def int8_gemm(
    activation: torch.Tensor, codes: torch.Tensor, activation_scale: torch.Tensor,
    weight_scale: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype,
) -> torch.Tensor:
    tokens, inner = activation.shape
    outer = codes.shape[0]
    out = torch.empty(tokens, outer, dtype=out_dtype, device=activation.device)
    grid = lambda meta: (triton.cdiv(tokens, meta['BLOCK_M']) * triton.cdiv(outer, meta['BLOCK_N']),)
    _int8_gemm_kernel[grid](
        activation, codes, activation_scale, weight_scale, bias, out, tokens, outer, inner,
        activation.stride(0), codes.stride(0), bias is not None,
    )
    return out

def quantize_rows(
    tensor: torch.Tensor, code_dtype: torch.dtype, code_max: float, rounds_to_integer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.reshape(-1, tensor.shape[-1])
    rows, inner = flat.shape
    codes = torch.empty(rows, inner, dtype=code_dtype, device=flat.device)
    scale = torch.empty(rows, dtype=torch.float32, device=flat.device)
    _quantize_rows_kernel[(rows,)](
        flat, codes, scale, inner, flat.stride(0), code_max, rounds_to_integer,
        torch.finfo(torch.float32).eps, BLOCK_K=1024, num_warps=8,
    )
    return codes, scale

__all__ = ["int8_gemm", "quantize_rows"]
