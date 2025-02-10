import triton
import triton.language as tl
import torch


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
    # params for scaling the computation
    scale_a,
    scale_b,
    zero_point_a,
    zero_point_b,
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
    zero_point_a = tl.load(zero_point_a)
    zero_point_b = tl.load(zero_point_b)
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
        a = a - zero_point_a
        b = b - zero_point_b
        acc += tl.dot(a, b)
    acc = acc.to(tl.float32) * scale_a * scale_b
    c_row = rm + tl.arange(0, BLOCK_M)
    c_col = rn + tl.arange(0, BLOCK_N)
    c_mask = (c_row < M)[:, None] & (c_col < N)[None, :]
    c_ptrs = C_ptr + (c_row[:, None] * stride_cm) + (c_col[None, :] * stride_cn)
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
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return C


@triton.jit
def triton_quantized_int8_conv2d_batched_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_out: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    stride_batch_in: tl.constexpr,
    stride_in_channel: tl.constexpr,
    stride_in_height: tl.constexpr,
    stride_in_width: tl.constexpr,
    stride_w_oc: tl.constexpr,
    stride_w_ic: tl.constexpr,
    stride_w_kh: tl.constexpr,
    stride_w_kw: tl.constexpr,
    stride_batch_out: tl.constexpr,
    stride_out_channel: tl.constexpr,
    stride_out_height: tl.constexpr,
    stride_out_width: tl.constexpr,
    scale_a,
    scale_b,
    zero_point_a,
    zero_point_b,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    oh_offset = pid_h * BLOCK_H
    ow_offset = pid_w * BLOCK_W
    scale_a_val = tl.load(scale_a)
    scale_b_val = tl.load(scale_b)
    zp_a = tl.load(zero_point_a)
    zp_b = tl.load(zero_point_b)

    for i in range(BLOCK_H):
        oh = oh_offset + i
        if oh < OH:
            for j in range(BLOCK_W):
                ow = ow_offset + j
                if ow < OW:
                    for oc in range(C_out):
                        acc = tl.zeros([], dtype=tl.int32)
                        for kh in range(KH):
                            ih = oh + kh - pad_h
                            if (ih >= 0) and (ih < H):
                                for kw in range(KW):
                                    iw = ow + kw - pad_w
                                    if (iw >= 0) and (iw < W):
                                        for ic in range(C):
                                            in_ptr = (
                                                input_ptr
                                                + b * stride_batch_in
                                                + ic * stride_in_channel
                                                + ih * stride_in_height
                                                + iw * stride_in_width
                                            )
                                            a = tl.load(in_ptr, mask=True, other=0)
                                            w_ptr = (
                                                weight_ptr
                                                + oc * stride_w_oc
                                                + ic * stride_w_ic
                                                + kh * stride_w_kh
                                                + kw * stride_w_kw
                                            )
                                            w_val = tl.load(w_ptr, mask=True, other=0)
                                            a_int = tl.cast(a, tl.int32)
                                            w_int = tl.cast(w_val, tl.int32)
                                            zp_a_int = tl.cast(zp_a, tl.int32)
                                            zp_b_int = tl.cast(zp_b, tl.int32)
                                            acc += (a_int - zp_a_int) * (
                                                w_int - zp_b_int
                                            )
                        res = acc.to(tl.float32) * scale_a_val * scale_b_val
                        out_ptr = (
                            output_ptr
                            + b * stride_batch_out
                            + oc * stride_out_channel
                            + oh * stride_out_height
                            + ow * stride_out_width
                        )
                        tl.store(out_ptr, res)


def triton_quantized_int8_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: float,
    scale_b: float,
    zero_point_a: int,
    zero_point_b: int,
    padding: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda, "Input and weight must be CUDA tensors"
    B, C, H, W = input.shape
    C_out, C_w, KH, KW = weight.shape
    assert C == C_w, "Input and weight channel mismatch!"
    pad_h, pad_w = padding
    assert groups == 1, "Only groups=1 is supported!"
    assert dilation == (1, 1), "Only dilation=(1, 1) is supported!"
    assert stride == (1, 1), "Only stride=(1, 1) is supported!"
    OH = H + 2 * pad_h - KH + 1
    OW = W + 2 * pad_w - KW + 1
    output = torch.empty((B, C_out, OH, OW), device=input.device, dtype=torch.float32)
    (
        stride_batch_in,
        stride_in_channel,
        stride_in_height,
        stride_in_width,
    ) = input.stride()
    (
        stride_batch_out,
        stride_out_channel,
        stride_out_height,
        stride_out_width,
    ) = output.stride()
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw = weight.stride()
    BLOCK_H = 8
    BLOCK_W = 8
    grid = (B, (OH + BLOCK_H - 1) // BLOCK_H, (OW + BLOCK_W - 1) // BLOCK_W)
    triton_quantized_int8_conv2d_batched_kernel[grid](
        input,
        weight,
        output,
        H,
        W,
        C,
        OH,
        OW,
        C_out,
        KH,
        KW,
        pad_h,
        pad_w,
        stride_batch_in,
        stride_in_channel,
        stride_in_height,
        stride_in_width,
        stride_w_oc,
        stride_w_ic,
        stride_w_kh,
        stride_w_kw,
        stride_batch_out,
        stride_out_channel,
        stride_out_height,
        stride_out_width,
        torch.tensor([scale_a], device=input.device, dtype=torch.float32),
        torch.tensor([scale_b], device=input.device, dtype=torch.float32),
        torch.tensor([zero_point_a], device=input.device, dtype=torch.int8),
        torch.tensor([zero_point_b], device=input.device, dtype=torch.int8),
        BLOCK_H,
        BLOCK_W,
    )
    return output
