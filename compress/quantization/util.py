import torch
from torch import nn


class IntQuantizationSpec:
    nbits: int
    signed: bool

    group_dims: list[int] = None

    # Example
    # We have a (O, I, H, W) tensor and we want to quantize the (output) channels, we can set group_dims=[1, 2, 3]

    def __post_init__(self):
        if self.group_dims == []:
            self.group_dims = None

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
            return IntQuantizationSpec(2 if dtype == "int2" else 4, True)
        if isinstance(dtype, str):
            dtype = torch.dtype(dtype)
        return {
            torch.uint8: IntQuantizationSpec(8, False),
            torch.int8: IntQuantizationSpec(8, True),
            torch.uint16: IntQuantizationSpec(16, False),
            torch.int16: IntQuantizationSpec(16, True),
        }[dtype]

    def __init__(self, nbits: int, signed: bool, group_dims: list[int] = None):
        self.nbits = nbits
        self.signed = signed
        self.group_dims = group_dims


class IntQuantizationInfo(nn.Module):
    spec: IntQuantizationSpec
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
        spec: IntQuantizationSpec,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.zero_point = (
            nn.Parameter(zero_point, requires_grad=False)
            if zero_point is not None
            else None
        )


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
