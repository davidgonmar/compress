import torch
from torch import nn
from enum import Enum


class IntAffineQuantizationMode(Enum):
    SYMMETRIC = "SYMMETRIC"
    ASYMMETRIC = "ASYMMETRIC"
    STATISTICS_AWARE_BINNING_SYMMETRIC = "STATISTICS_AWARE_BINNING_SYMMETRIC"  # from https://arxiv.org/abs/1807.06964, it's symmetric
    STATISTICS_AWARE_BINNING_ASYMMETRIC = (
        "STATISTICS_AWARE_BINNING_ASYMMETRIC"  # extension from the above
    )


class IntAffineQuantizationSpec:
    nbits: int
    signed: bool
    quant_mode: IntAffineQuantizationMode
    mode_args: dict = {}

    group_dims: list[int] = None

    # Example
    # We have a (O, I, H, W) tensor and we want to quantize the (output) channels, we can set group_dims=[1, 2, 3]

    def __post_init__(self):
        if self.group_dims == []:
            self.group_dims = None
        if self.quant_mode in [
            IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_SYMMETRIC,
            IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_ASYMMETRIC,
        ]:
            assert (
                self.signed
            ), "Statistics-aware binning only supports signed quantization"

        if self.quant_mode in [
            IntAffineQuantizationMode.SYMMETRIC,
            IntAffineQuantizationMode.ASYMMETRIC,
        ]:
            if "percentile" not in self.mode_args:
                self.mode_args["percentile"] = 1.0

    @property
    def qmin(self):
        return -(1 << (self.nbits - 1)) if self.signed else 0

    @property
    def qmax(self):
        return (1 << (self.nbits - 1)) - 1 if self.signed else (1 << self.nbits) - 1

    def get_dtype(self):
        return {
            (8, False): torch.uint8,
            (8, True): torch.int8,
            (16, False): torch.uint16,
            (16, True): torch.int16,
        }.get((self.nbits, self.signed), torch.float32)

    def from_dtype(dtype: torch.dtype | str):
        if dtype in ["int2, int4"]:
            return IntAffineQuantizationSpec(2 if dtype == "int2" else 4, True)
        if isinstance(dtype, str):
            dtype = torch.dtype(dtype)
        return {
            torch.uint8: IntAffineQuantizationSpec(8, False),
            torch.int8: IntAffineQuantizationSpec(8, True),
            torch.uint16: IntAffineQuantizationSpec(16, False),
            torch.int16: IntAffineQuantizationSpec(16, True),
        }[dtype]

    def __init__(
        self,
        nbits: int,
        signed: bool,
        quant_mode: IntAffineQuantizationMode,
        group_dims: list[int] = None,
        **kwargs,
    ):
        self.nbits = nbits
        self.signed = signed
        self.group_dims = group_dims
        self.quant_mode = quant_mode
        self.mode_args = kwargs

    def __repr__(self):
        return f"IntAffineQuantizationSpec(nbits={self.nbits}, signed={self.signed}, group_dims={self.group_dims}, quant_mode={self.quant_mode})"


class IntAffineQuantizationInfo(nn.Module):
    spec: IntAffineQuantizationSpec
    scale: torch.Tensor
    zero_point: torch.Tensor | None

    @property
    def nbits(self):
        return self.spec.nbits

    @property
    def signed(self):
        return self.spec.signed

    @property
    def qmin(self):
        return self.spec.qmin

    @property
    def qmax(self):
        return self.spec.qmax

    @property
    def group_dims(self):
        return self.spec.group_dims

    def get_dtype(self):
        return self.spec.get_dtype()

    def __init__(
        self,
        spec: IntAffineQuantizationSpec,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.scale = nn.Parameter(scale, requires_grad=False)
        # where scale is 0, make it one
        self.scale.data[self.scale == 0] = 1.0

        self.zero_point = (
            nn.Parameter(zero_point, requires_grad=False)
            if zero_point is not None
            else None
        )

    def __repr__(self):
        return f"IntAffineQuantizationInfo(spec={self.spec}, scale={self.scale}, zero_point={self.zero_point})"


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_round(x):
    return STERound.apply(x)


class STEFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_floor(x):
    return STEFloor.apply(x)


def quantize(x: torch.Tensor, info: IntAffineQuantizationInfo):
    if info.zero_point is None:
        return torch.clamp(ste_round(x / info.scale), info.qmin, info.qmax).to(
            info.get_dtype()
        )
    return torch.clamp(
        ste_round(x / info.scale + info.zero_point),
        info.qmin,
        info.qmax,
    ).to(info.get_dtype())


def dequantize(x: torch.Tensor, info: IntAffineQuantizationInfo):
    if info.zero_point is None:
        return info.scale * x
    return info.scale * (x - info.zero_point)


def fake_quantize(x: torch.Tensor, info: IntAffineQuantizationInfo):
    return dequantize(quantize(x, info), info)
