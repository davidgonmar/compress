import torch
from compress.quantization.util import (
    IntQuantizationInfo,
    ste_round,
)


def quantize(x: torch.Tensor, info: IntQuantizationInfo):
    return torch.clamp(
        ste_round(x / info.scale + info.zero_point) - info.zero_point,
        info.qmin,
        info.qmax,
    ).to(info.get_dtype())


def dequantize(x: torch.Tensor, info: IntQuantizationInfo):
    return info.scale * (x - info.zero_point)


def fake_quantize(x: torch.Tensor, info: IntQuantizationInfo):
    return dequantize(quantize(x, info), info)
