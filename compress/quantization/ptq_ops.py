import torch
from torch import nn
from compress.quantization.util import (
    IntQuantizationInfo,
    IntQuantizationSpec,
    ste_round,
)
from compress.quantization.calibrate import calibrate
from .kernels import triton_quantized_int8_matmul, triton_quantized_int8_conv2d


def quantize(x: torch.Tensor, info: IntQuantizationInfo):
    if info.zero_point is None:
        return torch.clamp(ste_round(x / info.scale), info.qmin, info.qmax).to(
            info.get_dtype()
        )

    return torch.clamp(
        ste_round(x / info.scale + info.zero_point) - info.zero_point,
        info.qmin,
        info.qmax,
    ).to(info.get_dtype())


def dequantize(x: torch.Tensor, info: IntQuantizationInfo):
    return info.scale * (x - info.zero_point)


def fake_quantize(x: torch.Tensor, info: IntQuantizationInfo):
    return dequantize(quantize(x, info), info)


def torch_simulated_int8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: float,
    scale_b: float,
    zero_point_a: int,
    zero_point_b: int,
):
    if zero_point_a is None:
        zero_point_a = torch.tensor(0).to(a.device)
    if zero_point_b is None:
        zero_point_b = torch.tensor(0).to(b.device)
    a, b, zero_point_a, zero_point_b = (
        a.to(torch.float32),
        b.to(torch.float32),
        zero_point_a.to(torch.float32),
        zero_point_b.to(torch.float32),
    )
    ret = (a - zero_point_a) @ (b - zero_point_b).T
    return ret * scale_a * scale_b


def torch_simulated_int8_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: float,
    scale_b: float,
    zero_point_a: int,
    zero_point_b: int,
    stride,
    padding,
    dilation,
    groups,
):
    if zero_point_a is None:
        zero_point_a = torch.tensor(0).to(input.device)
    if zero_point_b is None:
        zero_point_b = torch.tensor(0).to(input.device)
    input, weight, zero_point_a, zero_point_b = (
        input.to(torch.float32),
        weight.to(torch.float32),
        zero_point_a.to(torch.float32),
        zero_point_b.to(torch.float32),
    )
    ret = nn.functional.conv2d(
        (input - zero_point_a),
        (weight - zero_point_b),
        None,
        stride,
        padding,
        dilation,
        groups,
    )
    return ret.to(torch.float32) * scale_a * scale_b


def is_supported_linear(spec_a: IntQuantizationSpec, spec_b: IntQuantizationSpec):
    if isinstance(spec_a, IntQuantizationInfo):
        spec_a = spec_a.spec
    if isinstance(spec_b, IntQuantizationInfo):
        spec_b = spec_b.spec

    return spec_a.nbits == 8 and spec_b.nbits == 8 and spec_a.signed and spec_b.signed


def is_supported_conv2d(spec_a: IntQuantizationSpec, spec_b: IntQuantizationSpec, conv):
    if isinstance(spec_a, IntQuantizationInfo):
        spec_a = spec_a.spec
    if isinstance(spec_b, IntQuantizationInfo):
        spec_b = spec_b.spec
    return (
        spec_a.nbits == 8
        and spec_b.nbits == 8
        and spec_a.signed
        and spec_b.signed
        and conv.stride == (1, 1)
        and conv.padding == (0, 0)
        and conv.dilation == (1, 1)
        and conv.groups == 1
    )


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        weight_spec: IntQuantizationSpec,
        input_info_or_spec: IntQuantizationInfo | IntQuantizationSpec,
        linear: nn.Linear,
        simulated: bool = False,
    ):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        self.simulated = (
            simulated if is_supported_linear(input_info_or_spec, weight_spec) else True
        )
        assert isinstance(linear, nn.Linear), "Only nn.Linear is supported"
        super().__init__(in_features, out_features, bias)
        self.weight_spec = weight_spec
        self.weight_info = calibrate(
            linear.weight, weight_spec, return_z_as_int=not self.simulated
        )
        if isinstance(input_info_or_spec, IntQuantizationInfo):
            self.input_info = input_info_or_spec
            self.online_quant = False
        else:
            self.input_spec = input_info_or_spec
            self.online_quant = True
        self.weight = nn.Parameter(
            quantize(linear.weight.detach(), self.weight_info), requires_grad=False
        )
        self.bias = (
            nn.Parameter(linear.bias.detach(), requires_grad=False) if bias else None
        )

    def quantize_input(self, x: torch.Tensor):
        if self.online_quant:
            input_info = calibrate(
                x, self.input_spec, return_z_as_int=not self.simulated
            )
            return quantize(x, input_info), input_info
        return quantize(x, self.input_info), self.input_info

    def forward(self, x: torch.Tensor):
        x, input_info = self.quantize_input(x)
        if not self.simulated:
            x = triton_quantized_int8_matmul(
                x,
                self.weight.T,
                scale_a=input_info.scale,
                scale_b=self.weight_info.scale,
                zero_point_a=input_info.zero_point,
                zero_point_b=self.weight_info.zero_point,
            )
        else:
            x = torch_simulated_int8_matmul(
                x,
                self.weight,
                scale_a=input_info.scale,
                scale_b=self.weight_info.scale,
                zero_point_a=input_info.zero_point,
                zero_point_b=self.weight_info.zero_point,
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
        simulated: bool = False,
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
        self.simulated = (
            simulated
            if is_supported_conv2d(input_info_or_spec, weight_spec, conv2d)
            else True
        )
        assert isinstance(conv2d, nn.Conv2d), "Only nn.Conv2d is supported"
        self.weight_spec = weight_spec
        self.weight_info = calibrate(
            conv2d.weight, weight_spec, return_z_as_int=not simulated
        )
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
            input_info = calibrate(
                x, self.input_spec, return_z_as_int=not self.simulated
            )
            return quantize(x, input_info), input_info
        return quantize(x, self.input_info), self.input_info

    def forward(self, x: torch.Tensor):
        x, input_info = self.quantize_input(x)
        if self.simulated:
            conv2dres = torch_simulated_int8_conv2d(
                x,
                self.weight,
                scale_a=input_info.scale,
                scale_b=self.weight_info.scale,
                zero_point_a=input_info.zero_point,
                zero_point_b=self.weight_info.zero_point,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if self.bias is not None:
                conv2dres += self.bias.reshape(1, -1, 1, 1)
        else:
            conv2dres = triton_quantized_int8_conv2d(
                x,
                self.weight,
                scale_a=input_info.scale,
                scale_b=self.weight_info.scale,
                zero_point_a=input_info.zero_point,
                zero_point_b=self.weight_info.zero_point,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
        return conv2dres

    def __repr__(self):
        return f"QuantizedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding}, {self.dilation}, {self.groups}, {self.bias})"


if __name__ == "__main__":
    linear = nn.Linear(10, 10).to("cuda")
    qlinear = QuantizedLinear(
        IntQuantizationSpec(8, True), IntQuantizationSpec(8, True), linear
    )

    inp = torch.randn(10, 10).to("cuda")

    qlinearsim = QuantizedLinear(
        IntQuantizationSpec(8, True),
        IntQuantizationSpec(8, True),
        linear,
        simulated=True,
    )

    print(qlinear(inp))

    print(qlinearsim(inp))

    assert torch.allclose(qlinear(inp), qlinearsim(inp), atol=1e-3)

    conv2d = nn.Conv2d(3, 3, 3).to("cuda")
    qconv2d = QuantizedConv2d(
        IntQuantizationSpec(8, True), IntQuantizationSpec(8, True), conv2d
    )

    inp = torch.randn(1, 3, 10, 10).to("cuda")

    qconv2dsim = QuantizedConv2d(
        IntQuantizationSpec(8, True),
        IntQuantizationSpec(8, True),
        conv2d,
        simulated=True,
    )

    print(qconv2d(inp))

    print(qconv2dsim(inp))

    assert torch.allclose(
        qconv2d(inp), qconv2dsim(inp), atol=1e-3
    ), "Quantization tests failed. Diff is: " + str(qconv2d(inp) - qconv2dsim(inp))

    print("Quantization tests passed")
