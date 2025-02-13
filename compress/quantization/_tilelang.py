import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T


def convolution(
    N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads
):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        data: T.Buffer((N, H, W, C), dtype),
        kernel: T.Buffer((KH, KW, C, F), dtype),
        out: T.Buffer((N, OH, OW, F), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=threads
        ) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            kernel_flat = T.Buffer((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Buffer((N * OH * OW, F), accum_dtype, out.data)

            T.annotate_layout(
                {
                    out_shared: tilelang.layout.make_swizzled_layout(out_shared),
                    data_shared: tilelang.layout.make_swizzled_layout(data_shared),
                    kernel_shared: tilelang.layout.make_swizzled_layout(kernel_shared),
                }
            )

            T.clear(out_local)
            for k_iter in T.Pipelined(
                T.ceildiv(KH * KW * C, block_K), num_stages=num_stages
            ):
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                    access_w = m % OW * S + k // C % KW * D - P
                    in_bound = (
                        (access_h >= 0)
                        and (access_w >= 0)
                        and (access_h < H)
                        and (access_w < W)
                    )
                    data_shared[i, j] = T.if_then_else(
                        in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0
                    )
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


CACHED_FNS = {}

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
NUM_STAGES = 6
NUM_THREADS = 256


def tilelang_convolution(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
):
    N, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()  # NCWH -> NHWC
    assert groups == 1, "Group convolution is not supported."
    F, C, KH, KW = w.shape
    w = w.permute(2, 3, 1, 0).contiguous()  # FCHW -> HWCF
    S, D, P = stride, dilation, padding

    cache_key = (N, C, H, W, F, KH, KW, S, D, P)
    if cache_key in CACHED_FNS:
        fn = CACHED_FNS[cache_key]
    else:
        fn = convolution(
            N,
            C,
            H,
            W,
            F,
            KH,
            S,
            D,
            P,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_STAGES,
            NUM_THREADS,
        )
        jit_kernel = tilelang.JITKernel(fn, out_idx=[2], target="cuda")
        fn = jit_kernel
        CACHED_FNS[cache_key] = fn

    return fn(x, w).permute(0, 3, 1, 2)  # NHWC -> NCHW


if __name__ == "__main__":
    x = torch.randn(128, 3, 64, 64).cuda().to(torch.float16)
    w = torch.randn(32, 3, 3, 3).cuda().to(torch.float16)

    for _ in range(10):
        res = tilelang_convolution(x, w, 1, 1, 1, 1)
        ref = torch.nn.functional.conv2d(x.float(), w.float(), stride=1, padding=1)

    import time

    start_time = time.time()
    for _ in range(10):
        res = tilelang_convolution(x, w, 1, 1, 1, 1)
        torch.cuda.synchronize()
    end_time = time.time()
    tilelang_time = (end_time - start_time) / 10

    start_time = time.time()
    for _ in range(10):
        ref = torch.nn.functional.conv2d(x.float(), w.float(), stride=1, padding=1)
        torch.cuda.synchronize()
    end_time = time.time()
    torch_time = (end_time - start_time) / 10

    print(f"Tile-lang convolution time: {tilelang_time * 1000:.2f} ms")
    print(f"Torch convolution time: {torch_time * 1000:.2f} ms")

    print(torch.max(torch.abs(res - ref)))
    print(torch.mean(torch.abs(res - ref)))
    print(torch.max(res))
