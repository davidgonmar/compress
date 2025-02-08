import torch
from torch import nn
from dataclasses import dataclass
from tqdm import tqdm
from compress.common import gather_submodules, default_should_do

# q(X) = clamp(round(X / scale) + zero_point, qmin, qmax)
# X(q) = scale * (q - zero_point)

# rMM(qX, qW, b) = X(q) @ W(q).T + b = (X(q) - zero_point) @ (W(q) - zero_point).T * scaleX * scaleW + b


# rConv2d(qX, qW, b) = X(q) * W(q) + b = (X(q) - zero_point) * (W(q) - zero_point) * scaleX * scaleW + b
@dataclass
class IntQuantizationSpec:
    nbits: int
    signed: bool

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
        }[(self.nbits, self.signed)]

    def from_dtype(dtype: torch.dtype | str):
        if isinstance(dtype, str):
            dtype = torch.dtype(dtype)
        return {
            torch.uint8: IntQuantizationSpec(8, False),
            torch.int8: IntQuantizationSpec(8, True),
            torch.uint16: IntQuantizationSpec(16, False),
            torch.int16: IntQuantizationSpec(16, True),
        }[dtype]


@dataclass
class IntQuantizationInfo:
    spec: IntQuantizationSpec
    scale: float
    zero_point: int

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

    def get_dtype(self):
        return self.spec.get_dtype()


def calibrate(x: torch.Tensor, spec: IntQuantizationSpec, symmetric: bool = False):
    if not symmetric:
        xmin = x.min().item()
        xmax = x.max().item()
        if (xmin == 0 and xmax == 0) or (xmin == xmax):
            return IntQuantizationInfo(spec, 1.0, 0)  # avoid division by zero
        scale = (xmax - xmin) / (spec.qmax - spec.qmin)
        zero_point = round(spec.qmin - xmin / scale)
        return IntQuantizationInfo(spec, scale, zero_point)
    else:
        xmax = x.abs().max().item()
        scale = xmax / spec.qmax
        zero_point = 0
        return IntQuantizationInfo(spec, scale, zero_point)


def quantize(x: torch.Tensor, info: IntQuantizationInfo, fake=True):
    return (
        torch.clamp(torch.round(x / info.scale) + info.zero_point, info.qmin, info.qmax)
        .to(info.get_dtype())
        .to(x.dtype if fake else info.get_dtype())
    )


def dequantize(x: torch.Tensor, info: IntQuantizationInfo):
    return info.scale * (x - info.zero_point)


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_info_or_spec: IntQuantizationInfo | IntQuantizationSpec,
        linear: nn.Linear,
    ):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        assert isinstance(linear, nn.Linear), "Only nn.Linear is supported"
        super().__init__(in_features, out_features, bias)
        self.weight_spec = weight_spec
        self.weight_info = calibrate(linear.weight, weight_spec)
        if isinstance(input_info_or_spec, IntQuantizationInfo):
            self.input_info = input_info_or_spec
            self.online_quant = False
        else:
            self.input_spec = input_info_or_spec
            self.online_quant = True
        self.weight = nn.Parameter(
            quantize(linear.weight, self.weight_info), requires_grad=False
        )
        self.bias = nn.Parameter(linear.bias, requires_grad=False) if bias else None

    def quantize_input(self, x: torch.Tensor):
        if self.online_quant:
            input_info = calibrate(x, self.input_spec)
            return quantize(x, input_info), input_info
        return quantize(x, self.input_info), self.input_info

    def forward(self, x: torch.Tensor):
        x, input_info = self.quantize_input(x)
        x = (
            (
                (x - input_info.zero_point)
                @ (self.weight - self.weight_info.zero_point).T
            ).to(torch.float32)
            * input_info.scale
            * self.weight_info.scale
        )
        if self.bias is not None:
            x += self.bias

        return x

    def __repr__(self):
        return f"QuantizedLinear({self.in_features}, {self.out_features}, {self.bias})"


class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_info_or_spec: IntQuantizationInfo | IntQuantizationSpec,
        conv2d: nn.Conv2d,
    ):
        (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        ) = (
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            conv2d.bias is not None,
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        assert isinstance(conv2d, nn.Conv2d), "Only nn.Conv2d is supported"
        self.weight_spec = weight_spec
        self.weight_info = calibrate(conv2d.weight, weight_spec)
        if isinstance(input_info_or_spec, IntQuantizationInfo):
            self.input_info = input_info_or_spec
            self.online_quant = False
        else:
            self.input_spec = input_info_or_spec
            self.online_quant = True
        self.weight = nn.Parameter(
            quantize(conv2d.weight, self.weight_info), requires_grad=False
        )
        self.bias = nn.Parameter(conv2d.bias, requires_grad=False) if bias else None

    def quantize_input(self, x: torch.Tensor):
        if self.online_quant:
            input_info = calibrate(x, self.input_spec)
            return quantize(x, input_info), input_info
        return quantize(x, self.input_info), self.input_info

    def forward(self, x: torch.Tensor):
        x, input_info = self.quantize_input(x)
        conv2dres = (
            nn.functional.conv2d(
                (x - input_info.zero_point),
                (self.weight - self.weight_info.zero_point),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            ).to(torch.float32)
            * input_info.scale
            * self.weight_info.scale
        )
        if self.bias is not None:
            conv2dres += self.bias
        return conv2dres

    def __repr__(self):
        return f"QuantizedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding}, {self.dilation}, {self.groups}, {self.bias})"


def to_quantized_online(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
    inplace=True,
    should_do=default_should_do,
    **kwargs,
):
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            QuantizedLinear(weight_specs["linear"], input_specs["linear"], module)
            if isinstance(module, nn.Linear)
            else QuantizedConv2d(weight_specs["conv2d"], input_specs["conv2d"], module),
        )

    return model
