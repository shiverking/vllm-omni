# SPDX-License-Identifier: Apache-2.0
"""Opt-in Triton-Ascend SnakeBeta. No eager fallback or import-time JIT."""

from functools import lru_cache

import torch


def reference(x, alpha, inv_beta):
    return x + inv_beta[None, :, None] * torch.pow(torch.sin(x * alpha[None, :, None]), 2)


def validate(x, alpha, inv_beta, out, block, programs):
    if x.ndim != 3 or x.stride(-1) != 1:
        raise ValueError("SnakeBeta requires [B,C,T] with contiguous time axis; no hidden layout copy")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("SnakeBeta supports BF16/FP16/FP32")
    for parameter in (alpha, inv_beta):
        if parameter.shape != (x.shape[1],) or parameter.dtype != x.dtype:
            raise ValueError("SnakeBeta parameters must be [channels] with the input dtype")
        if parameter.device != x.device or not parameter.is_contiguous():
            raise ValueError("SnakeBeta parameters must be contiguous on the input device")
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("Invalid SnakeBeta output buffer")
    if block < 128 or block > 8192 or block & (block - 1) or programs < 1:
        raise ValueError("block must be a power of two in [128,8192]; programs must be positive")


@lru_cache(maxsize=1)
def kernel():
    # Ascend JIT resolves source names from function globals, not closure cells.
    # Keep loading lazy so default/CPU paths do not require Triton.
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def snake(
        x_ptr,
        alpha_ptr,
        inv_ptr,
        out_ptr,
        stride_b: tl.constexpr,
        stride_c: tl.constexpr,
        channels: tl.constexpr,
        length: tl.constexpr,
        tiles: tl.constexpr,
        tasks: tl.constexpr,
        block: tl.constexpr,
    ):
        # A bounded number of programs traverse channel-local contiguous tiles.
        # Five intermediates stay on chip; explicit casts preserve eager rounding.
        for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
            channel = (task // tiles) % channels
            row = task // tiles
            t = (task % tiles) * block + tl.arange(0, block)
            offset = row * length + t
            input_offset = (row // channels) * stride_b + channel * stride_c + t
            x = tl.load(x_ptr + input_offset, t < length, other=0)
            a = tl.load(alpha_ptr + channel).to(tl.float32)
            ib = tl.load(inv_ptr + channel).to(tl.float32)
            z = (x.to(tl.float32) * a).to(x.dtype).to(tl.float32)
            s = tl.sin(z).to(x.dtype).to(tl.float32)
            sq = (s * s).to(x.dtype).to(tl.float32)
            term = (ib * sq).to(x.dtype).to(tl.float32)
            y = (x.to(tl.float32) + term).to(x.dtype)
            tl.store(out_ptr + offset, y, t < length)

    return snake


def fused_snake(x, alpha, inv_beta, *, out=None, block=2048, programs=32):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    validate(x, alpha, inv_beta, out, block, programs)
    if x.device.type != "npu":
        raise ValueError("Triton-Ascend SnakeBeta requires NPU")
    if x.numel() == 0:
        return out
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, alpha, inv_beta)):
        raise ValueError("Experimental SnakeBeta is inference-only")
    b, c, t = x.shape
    tiles = (t + block - 1) // block
    tasks = b * c * tiles
    kernel()[(min(programs, tasks),)](
        x, alpha, inv_beta, out, x.stride(0), x.stride(1), c, t, tiles, tasks, block, enable_fp_fusion=False
    )
    return out


def configure(model, execution="eager", block=2048, programs=32):
    from .snake_activation import SnakeBeta

    if execution not in ("eager", "triton_ascend"):
        raise ValueError(f"Unknown SnakeBeta execution: {execution}")
    count = 0
    for module in model.modules():
        if isinstance(module, SnakeBeta):
            module._npu_execution = execution
            module._npu_block = block
            module._npu_programs = programs
            count += 1
    return count
