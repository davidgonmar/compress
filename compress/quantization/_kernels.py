"""
Triton kernels for quantized linear and conv2d.
The conv2d one is slow in practice.
"""

import triton
import triton.language as tl
import torch
from typing import Optional


@triton.jit
def triton_quantized_int8_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    scale_a,
    scale_b,
    zero_point_a,
    zero_point_b,
    has_zp_a: tl.constexpr,
    has_zp_b: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    scale_a = tl.load(scale_a)
    scale_b = tl.load(scale_b)
    for k in range(0, K, BLOCK_K):
        a_row = rm + tl.arange(0, BLOCK_M)
        a_col = k + tl.arange(0, BLOCK_K)
        a_mask = (a_row < M)[:, None] & (a_col < K)[None, :]
        a_ptrs = A_ptr + (a_row[:, None] * stride_am) + (a_col[None, :] * stride_ak)
        b_row = k + tl.arange(0, BLOCK_K)
        b_col = rn + tl.arange(0, BLOCK_N)
        b_mask = (b_row < K)[:, None] & (b_col < N)[None, :]
        b_ptrs = B_ptr + (b_row[:, None] * stride_bk) + (b_col[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        if has_zp_a:
            a = a - tl.load(zero_point_a)
        if has_zp_b:
            b = b - tl.load(zero_point_b)
        acc += tl.dot(a, b)
    acc = acc.to(tl.float32) * scale_a * scale_b
    c_row = rm + tl.arange(0, BLOCK_M)
    c_col = rn + tl.arange(0, BLOCK_N)
    c_mask = (c_row < M)[:, None] & (c_col < N)[None, :]
    c_ptrs = C_ptr + (c_row[:, None] * stride_cm) + (c_col[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def triton_quantized_int8_matvec_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    K,
    N,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    scale_a,
    scale_b,
    zero_point_a,
    zero_point_b,
    has_zp_a: tl.constexpr,
    has_zp_b: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    rn = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    scale_a = tl.load(scale_a)
    scale_b = tl.load(scale_b)
    for k in range(0, K, BLOCK_K):
        a_col = k + tl.arange(0, BLOCK_K)
        a_mask = a_col < K
        a_ptrs = A_ptr + (a_col * stride_ak)
        b_row = k + tl.arange(0, BLOCK_K)
        b_col = rn + tl.arange(0, BLOCK_N)
        b_mask = (b_row < K)[:, None] & (b_col < N)[None, :]
        b_ptrs = B_ptr + (b_row[:, None] * stride_bk) + (b_col[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        if has_zp_a:
            a = a - tl.load(zero_point_a)
        if has_zp_b:
            b = b - tl.load(zero_point_b)
        acc += tl.sum(a[:, None] * b, axis=0)
    acc = acc.to(tl.float32) * scale_a * scale_b
    c_col = rn + tl.arange(0, BLOCK_N)
    c_mask = c_col < N
    c_ptrs = C_ptr + (c_col * stride_cn)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_quantized_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: float,
    scale_b: float,
    zero_point_a: int,
    zero_point_b: int,
) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "A and B must be on CUDA"

    # if A is not ndim 2, flatten the first dimensions (assume batch)
    if A.ndim > 2 or A.ndim == 1:
        reshape = A.shape[:-1]
        A = A.view(-1, A.shape[-1])
    else:
        reshape = None

    is_vec = A.shape[0] == 1

    if is_vec:
        K, N = B.shape
        C = torch.empty((1, N), device=A.device, dtype=torch.float32)
        stride_ak = A.stride()[1]
        stride_bk, stride_bn = B.stride()
        stride_cn = C.stride()[1]
        BLOCK_K = 32
        BLOCK_N = 64
        grid = ((N + BLOCK_N - 1) // BLOCK_N,)

        triton_quantized_int8_matvec_kernel[grid](
            A,
            B,
            C,
            K,
            N,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cn,
            scale_a,
            scale_b,
            zero_point_a,
            zero_point_b,
            zero_point_a is not None,
            zero_point_b is not None,
            BLOCK_K,
            BLOCK_N,
        )
    else:
        M, K = A.shape
        K2, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = C.stride()
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        triton_quantized_int8_matmul_kernel[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            scale_a,
            scale_b,
            zero_point_a,
            zero_point_b,
            zero_point_a is not None,
            zero_point_b is not None,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
    if reshape is not None:
        return C.view(*reshape, N)
    return C


@triton.jit
def fused_int8_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    O_H: tl.constexpr,
    O_W: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_dim: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h_in: tl.constexpr,
    stride_w_in: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_a_k: tl.constexpr,
    stride_c_batch: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_c_n: tl.constexpr,
    scale_a_ptr,
    scale_b_ptr,
    zp_a_ptr,
    zp_b_ptr,
    has_zp_a: tl.constexpr,
    has_zp_b: tl.constexpr,
    bias_ptr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K_dim, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = (
            weight_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        )
        a_tile = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0)
        a_tile = a_tile
        if has_zp_b:
            loaded_zp_b = tl.load(zp_b_ptr)
            a_tile = a_tile - loaded_zp_b
        c_indices = offs_k // (K_H * K_W)
        rem = offs_k % (K_H * K_W)
        k_h = rem // K_W
        k_w = rem % K_W
        o_h = offs_n // O_W
        o_w = offs_n % O_W
        k_h_tile = k_h[:, None]
        k_w_tile = k_w[:, None]
        o_h_tile = o_h[None, :]
        o_w_tile = o_w[None, :]
        i_h = o_h_tile * stride_h - pad_h + k_h_tile * dilation_h
        i_w = o_w_tile * stride_w - pad_w + k_w_tile * dilation_w
        c_tile = c_indices[:, None]
        input_offsets = (
            b * stride_b + c_tile * stride_c + i_h * stride_h_in + i_w * stride_w_in
        )
        valid = (i_h >= 0) & (i_h < H) & (i_w >= 0) & (i_w < W)
        b_tile = tl.load(input_ptr + input_offsets, mask=valid, other=0)
        b_tile = b_tile
        if has_zp_a:
            loaded_zp_a = tl.load(zp_a_ptr)
            b_tile = b_tile - loaded_zp_a
        acc += tl.dot(a_tile, b_tile)
    result = acc.to(tl.float32) * tl.load(scale_a_ptr) * tl.load(scale_b_ptr)
    if has_bias:
        bias_vals = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0)
        result = result + bias_vals[:, None]
    out_ptrs = (
        output_ptr
        + b * stride_c_batch
        + offs_m[:, None] * stride_c_m
        + offs_n[None, :] * stride_c_n
    )
    tl.store(out_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_quantized_int8_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    zero_point_a: Optional[torch.Tensor] = None,
    zero_point_b: Optional[torch.Tensor] = None,
    padding: tuple[int, int] = (0, 0),
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA."
    B, C, H, W = input.shape
    C_out, C_w, K_H, K_W = weight.shape
    assert C == C_w, "Mismatch in input and weight channels."
    assert groups == 1, "Only groups==1 is supported."
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    O_H = (H + 2 * pad_h - (dilation_h * (K_H - 1) + 1)) // stride_h + 1
    O_W = (W + 2 * pad_w - (dilation_w * (K_W - 1) + 1)) // stride_w + 1
    M = C_out
    K_dim = C * K_H * K_W
    N = O_H * O_W
    stride_b, stride_c, stride_h_in, stride_w_in = input.stride()
    weight_mat = weight.view(C_out, -1)
    stride_a_m, stride_a_k = weight_mat.stride()
    output = torch.empty((B, C_out, N), device=input.device, dtype=torch.float32)
    stride_c_batch, stride_c_m, stride_c_n = output.stride()
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    grid = (
        B,
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    fused_int8_conv_kernel[grid](
        input,
        weight_mat,
        output,
        B,
        C,
        H,
        W,
        K_H,
        K_W,
        O_H,
        O_W,
        M,
        N,
        K_dim,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        stride_b,
        stride_c,
        stride_h_in,
        stride_w_in,
        stride_a_m,
        stride_a_k,
        stride_c_batch,
        stride_c_m,
        stride_c_n,
        scale_a,
        scale_b,
        zero_point_a if zero_point_a is not None else 0,
        zero_point_b if zero_point_b is not None else 0,
        zero_point_a is not None,
        zero_point_b is not None,
        bias if bias is not None else 0,
        bias is not None,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    output = output.view(B, C_out, O_H, O_W)
    return output
