import torch

try:
    from ._kernels import triton_quantized_int8_matmul, triton_quantized_int8_conv2d

    TRITON_AVAILABLE = True

except ImportError:
    TRITON_AVAILABLE = False

    def triton_quantized_int8_matmul(
        A: torch.Tensor,
        B: torch.Tensor,
        scale_a: float,
        scale_b: float,
        zero_point_a: int,
        zero_point_b: int,
    ) -> torch.Tensor:
        raise NotImplementedError("triton_quantized_int8_matmul is not available")

    def triton_quantized_int8_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: float,
        scale_b: float,
        zero_point_a: int,
        zero_point_b: int,
        padding,
        stride,
        dilation,
        groups,
    ) -> torch.Tensor:
        raise NotImplementedError("triton_quantized_int8_conv2d is not available")
