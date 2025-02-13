from torch import nn
from compress.quantization.util import IntQuantizationSpec, IntQuantizationInfo
import torch
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq_ops import (
    fake_quantize,
    QuantizedConv2d,
    QuantizedLinear,
    quantize,
)
import math


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


class LSQQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, info):
        assert info.zero_point is None, "Zero point is not supported"
        ctx.save_for_backward(x, scale)
        ctx.qmin = info.qmin
        ctx.qmax = info.qmax
        return fake_quantize(x, info)

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        x_grad = torch.clamp(grad_output, qmin, qmax)
        v_s = x / scale
        s_grad = (
            (
                torch.where(
                    -qmin >= v_s,
                    -qmin,
                    torch.where(qmax <= v_s, qmax, -v_s + torch.round(v_s)),
                )
                * grad_output
            )
            .sum()
            .reshape(scale.shape)
        )
        # rescale as the paper mentions
        rescaling = 1 / math.sqrt((x.numel() * (qmax - qmin)))
        s_grad = s_grad * rescaling
        return x_grad, s_grad, None


class LSQLinear(nn.Linear):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_spec: IntQuantizationSpec,
        linear: nn.Linear,
        data_batch: torch.Tensor,
    ):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        assert isinstance(linear, nn.Linear), "Only nn.Linear is supported"
        super().__init__(in_features, out_features, bias)
        self.weight_info = calibrate(linear.weight, weight_spec)
        self.input_info = calibrate(data_batch, input_spec)
        self.weight_info.scale.requires_grad_(True)
        self.input_info.scale.requires_grad_(True)
        self.weight = nn.Parameter(linear.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(linear.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )

    def forward(self, x: torch.Tensor):
        x = LSQQuantize.apply(x, self.input_info.scale, self.input_info)
        w = LSQQuantize.apply(self.weight, self.weight_info.scale, self.weight_info)
        ret = nn.functional.linear(x, w, self.bias)
        return ret

    def __repr__(self):
        return (
            f"LSQQuantizedLinear({self.in_features}, {self.out_features}, {self.bias})"
        )

    def to_linear(self):
        ret = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        ret.weight = self.weight
        ret.bias = self.bias
        return ret

    def to_quant_linear(self):
        ret = QuantizedLinear(self.weight_info.spec, self.input_info, self.to_linear())
        ret.weight_info = self.weight_info
        ret.weight = torch.nn.Parameter(
            quantize(self.weight, self.weight_info), requires_grad=False
        )
        return ret


class LSQConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_spec: IntQuantizationSpec,
        conv2d: nn.Conv2d,
        data_batch: torch.Tensor,
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
        self.weight_info = calibrate(conv2d.weight, weight_spec)
        self.input_info = calibrate(data_batch, input_spec)
        self.weight_info.scale.requires_grad_(True)
        self.input_info.scale.requires_grad_(True)
        self.weight = nn.Parameter(conv2d.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(conv2d.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )

    def forward(self, x: torch.Tensor):
        x = LSQQuantize.apply(x, self.input_info.scale, self.input_info)
        w = LSQQuantize.apply(self.weight, self.weight_info.scale, self.weight_info)
        ret = nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return ret

    def __repr__(self):
        return f"LSQQuantizedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding}, {self.dilation}, {self.groups}, {self.bias})"

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

    def to_quant_conv2d(self):
        ret = QuantizedConv2d(self.weight_info.spec, self.input_info, self.to_conv2d())
        ret.weight_info = self.weight_info
        ret.weight = torch.nn.Parameter(
            quantize(self.weight, self.weight_info), requires_grad=False
        )
        return ret
