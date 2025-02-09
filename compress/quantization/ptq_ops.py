import torch
from torch import nn
from compress.quantization.util import (
    IntQuantizationInfo,
    IntQuantizationSpec,
    ste_round,
)
from compress.quantization.calibrate import calibrate


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
