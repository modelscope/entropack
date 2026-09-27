from pathlib import Path

import cupy
import numpy as np
import torch

from ...backends.cuda import device as _device_caps
from ...backends.cuda.kernels import KernelLibrary
from ...backends.cuda.kernels import device_index as _device_index
from ...backends.cuda.kernels import ensure_dynamic_shared as _ensure_dynamic_shared
from ...backends.cuda.kernels import external_stream as _external_stream
from ...backends.cuda.kernels import pointer as _pointer
from .format import BLOCK_SIZE, MAX_RESIDENT_BLOCKS_PER_SM, DFloat11Buffers, make_layout, max_block_elems, parse_layout_cached

_CUDA_PATH = Path(__file__).parent / "dfloat11.cu"
_DECODE_KERNEL_NAME = "dfloat11_decode_kernel"
_ENCODE_KERNEL_NAMES = (
    "dfloat11_exponent_histogram_kernel", "dfloat11_split_len_kernel", "dfloat11_pack_kernel", "dfloat11_thread_meta_kernel",
    "dfloat11_output_positions_kernel",
)


def _compile_defines(device_index: int, threads_per_block: int) -> tuple[str, ...]:
    caps = _device_caps.caps(device_index)
    min_blocks = max(1, min(MAX_RESIDENT_BLOCKS_PER_SM, caps.threads_per_sm // threads_per_block))
    return (f"DFLOAT11_THREADS_PER_BLOCK={threads_per_block}", f"DFLOAT11_MIN_BLOCKS_PER_SM={min_blocks}")


_LIBRARY = KernelLibrary(
    key="dfloat11", source=_CUDA_PATH, defines=_compile_defines, kernel_names=(_DECODE_KERNEL_NAME, *_ENCODE_KERNEL_NAMES),
)
_kernel = _LIBRARY.kernel


def _encode_kernels(device_index: int, threads_per_block: int) -> dict:
    return {name: _kernel(device_index, threads_per_block, name) for name in _ENCODE_KERNEL_NAMES}


def _shared_budget(device: torch.device) -> int:
    return _device_caps.caps(device).shared_limit()


def _kernel_tensor(t: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
    contiguous = t.contiguous()
    if contiguous is not t:
        contiguous.record_stream(stream)
    return contiguous


def _cupy_view(t: torch.Tensor):
    """Not cached: the DLPack capsule keeps the source tensor alive, so a cache would pin every tensor ever viewed for the life
    of the process.
    """
    return cupy.from_dlpack(t)


def decode(buffers: DFloat11Buffers) -> torch.Tensor:
    encoded_exponent, sign_mantissa, luts = buffers.encoded_exponent, buffers.sign_mantissa, buffers.luts
    output_positions, thread_meta, layout = buffers.output_positions, buffers.thread_meta, buffers.layout
    bytes_per_thread, threads_per_block, max_block_elems = parse_layout_cached(layout)
    num_luts = int(luts.shape[0])
    n_bytes = int(encoded_exponent.numel())
    n_elements = int(sign_mantissa.numel())

    n_threads = (n_bytes + bytes_per_thread - 1) // bytes_per_thread
    blocks = (n_threads + threads_per_block - 1) // threads_per_block

    budget = _shared_budget(sign_mantissa.device)
    fixed_bytes = threads_per_block * 4 + num_luts * 256
    stage_enc_bytes = threads_per_block * bytes_per_thread + 8
    stage_elems = max_block_elems
    if fixed_bytes + stage_enc_bytes + stage_elems + 4 > budget:
        stage_elems = 0
        if fixed_bytes + stage_enc_bytes > budget:
            stage_enc_bytes = 0
    shared_bytes = fixed_bytes + stage_enc_bytes + stage_elems + (4 if stage_elems else 0)

    out = torch.empty(n_elements, dtype=torch.bfloat16, device=sign_mantissa.device)
    if blocks == 0:
        return out

    device_index = _device_index(sign_mantissa)
    with cupy.cuda.Device(device_index):
        kernel = _kernel(device_index, threads_per_block, _DECODE_KERNEL_NAME)
        _ensure_dynamic_shared(kernel, shared_bytes)

        torch_stream = torch.cuda.current_stream(sign_mantissa.device)
        luts_c = _kernel_tensor(luts, torch_stream)
        encoded_c = _kernel_tensor(encoded_exponent, torch_stream)
        sign_mantissa_c = _kernel_tensor(sign_mantissa, torch_stream)
        output_positions_c = _kernel_tensor(output_positions, torch_stream)
        thread_meta_c = _kernel_tensor(thread_meta, torch_stream)
        args = (
            _pointer(luts_c), _pointer(encoded_c), _pointer(sign_mantissa_c), _pointer(output_positions_c),
            _pointer(thread_meta_c), _pointer(out), np.int32(num_luts), np.int64(n_bytes), np.int64(n_elements),
            np.int32(bytes_per_thread), np.int32(stage_enc_bytes), np.int32(stage_elems),
            np.int32(1 if (sign_mantissa_c.data_ptr() & 3) == 0 else 0),
        )
        with _external_stream(torch_stream):
            kernel((blocks,), (threads_per_block,), args, shared_mem=shared_bytes)
    return out


def exponent_counter(weight: torch.Tensor, threads_per_block: int) -> dict[int, int]:
    flat = weight.reshape(-1)
    n_elements = int(flat.numel())
    histogram = torch.zeros(256, dtype=torch.int64, device=flat.device)
    if n_elements == 0:
        return {}

    caps = _device_caps.caps(flat.device)
    threads = caps.threads_per_block(BLOCK_SIZE)
    blocks = caps.grid((n_elements + threads - 1) // threads, threads)
    device_index = _device_index(flat)
    torch_stream = torch.cuda.current_stream(flat.device)
    stream_ctx = _external_stream(torch_stream)
    with cupy.cuda.Device(device_index), stream_ctx:
        _kernel(device_index, threads_per_block, "dfloat11_exponent_histogram_kernel")(
            (blocks,), (threads,), (_pointer(flat), _pointer(histogram), np.int64(n_elements)),
        )
    counts = histogram.cpu().tolist()
    return {i: int(count) for i, count in enumerate(counts) if count > 0}


def _code_table(codec, device):
    code_len = torch.zeros(256, dtype=torch.int32)
    code_val = torch.zeros(256, dtype=torch.int32)
    for k, (bits, val) in codec._table.items():
        if isinstance(k, int):
            code_len[k] = bits
            code_val[k] = val
    eof_len, eof_val = codec._table[codec._eof]
    return (code_len.to(device).contiguous(), code_val.to(device).contiguous(), int(eof_len), int(eof_val))


def encode(
    *, weight: torch.Tensor, codec, luts: torch.Tensor, bytes_per_thread: int, threads_per_block: int,
) -> DFloat11Buffers:
    device = weight.device
    device_index = _device_index(device)
    flat = weight.reshape(-1)
    n_elements = int(flat.numel())
    if not 5 <= bytes_per_thread <= 255:
        raise ValueError(
            "dfloat11 bytes_per_thread must be in [5, 255]: the region must exceed "
            "the 32-bit maximum code length and its worst-case symbol count must fit " "the 11-bit thread_meta field"
        )
    if n_elements >= 1 << 32:
        raise ValueError("dfloat11 requires fewer than 2^32 elements per tensor")
    code_len_gpu, code_val_gpu, eof_len, eof_val = _code_table(codec, device)

    kernels = _encode_kernels(device_index, threads_per_block)
    threads = _device_caps.caps(device).threads_per_block(BLOCK_SIZE)

    exponent = torch.empty(n_elements, dtype=torch.uint8, device=device)
    sign_mantissa = torch.empty(n_elements, dtype=torch.uint8, device=device)
    len_scratch = torch.empty(n_elements, dtype=torch.uint8, device=device)
    pref = torch.empty(n_elements + 1, dtype=torch.int64, device=device)

    blocks_split = (n_elements + threads - 1) // threads

    torch_stream = torch.cuda.current_stream(flat.device)
    stream_ctx = _external_stream(torch_stream)
    with cupy.cuda.Device(device_index), stream_ctx:
        kernels["dfloat11_split_len_kernel"](
            (blocks_split,),
            (threads,),
            (
                _pointer(flat), _pointer(code_len_gpu), _pointer(exponent), _pointer(sign_mantissa), _pointer(len_scratch),
                np.int64(n_elements),
            ),
        )

        pref_cp = _cupy_view(pref)
        pref_cp[0] = 0
        if n_elements > 0:
            cupy.cumsum(_cupy_view(len_scratch), dtype=cupy.int64, out=pref_cp[1:])

        total_bits = int(pref[n_elements].item())

    n_bytes = (total_bits + 7) // 8
    region_bits = bytes_per_thread * 8
    block_bits = region_bits * threads_per_block
    bytes_per_block = bytes_per_thread * threads_per_block
    num_blocks = (n_bytes + bytes_per_block - 1) // bytes_per_block
    n_regions = threads_per_block * num_blocks

    encoded = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    thread_meta = torch.empty(n_regions, dtype=torch.uint16, device=device)
    output_positions = torch.empty(num_blocks + 1, dtype=torch.uint32, device=device)

    with cupy.cuda.Device(device_index), stream_ctx:
        pack_chunks = (n_bytes + 3) // 4
        kernels["dfloat11_pack_kernel"](
            ((pack_chunks + threads - 1) // threads,),
            (threads,),
            (
                _pointer(pref), _pointer(exponent), _pointer(code_len_gpu), _pointer(code_val_gpu), _pointer(encoded),
                np.int64(n_elements), np.int64(n_bytes), np.int64(total_bits), np.int32(eof_len), np.uint32(eof_val),
            ),
        )
        kernels["dfloat11_thread_meta_kernel"](
            ((n_regions + threads - 1) // threads,),
            (threads,),
            (
                _pointer(pref), _pointer(thread_meta), np.int64(n_elements), np.int64(total_bits), np.int64(region_bits),
                np.int64(n_regions),
            ),
        )
        kernels["dfloat11_output_positions_kernel"](
            ((num_blocks + 1 + threads - 1) // threads,), (threads,),
            (_pointer(pref), _pointer(output_positions), np.int64(n_elements), np.int64(block_bits), np.int64(num_blocks)),
        )

    op_host = torch.empty(num_blocks + 1, dtype=torch.uint32, pin_memory=True)
    op_host.copy_(output_positions, non_blocking=True)
    torch_stream.synchronize()

    layout = make_layout(bytes_per_thread, threads_per_block, max_block_elems(op_host))
    return DFloat11Buffers(
        encoded_exponent=encoded, sign_mantissa=sign_mantissa, luts=luts, output_positions=output_positions,
        thread_meta=thread_meta, layout=layout,
    )
