from torch import nn
from compress.quantization.util import IntQuantizationSpec, IntQuantizationInfo
import torch
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq_ops import fake_quantize


class QATLinear(nn.Linear):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_spec: IntQuantizationInfo | IntQuantizationSpec,
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
        self.input_spec = input_spec
        self.weight = nn.Parameter(linear.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(linear.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )

    def forward(self, x: torch.Tensor):
        x = fake_quantize(x, calibrate(x, self.input_spec))
        w = fake_quantize(self.weight, calibrate(self.weight, self.weight_spec))
        return nn.functional.linear(x, w, self.bias)

    def __repr__(self):
        return (
            f"FakeQuantizedLinear({self.in_features}, {self.out_features}, {self.bias})"
        )

    def to_linear(self):
        ret = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        ret.weight = self.weight
        ret.bias = self.bias
        return ret


class QATConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_spec: IntQuantizationSpec,
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
        self.input_spec = input_spec
        self.weight = nn.Parameter(conv2d.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(conv2d.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )

    def forward(self, x: torch.Tensor):
        x = fake_quantize(x, calibrate(x, self.input_spec))
        w = fake_quantize(self.weight, calibrate(self.weight, self.weight_spec))
        return nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def __repr__(self):
        return f"FakeQuantizedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding}, {self.dilation}, {self.groups}, {self.bias})"

    def to_conv2d(self):
        ret = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
        )
        ret.weight = self.weight
        ret.bias = self.bias
        return ret
