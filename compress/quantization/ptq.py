import torch
from torch import nn
from compress.quantization.common import (
    IntAffineQuantizationInfo,
    IntAffineQuantizationSpec,
    quantize,
    dequantize,
)
from compress.quantization.calibrate import calibrate
from .kernels import (
    triton_quantized_int8_matmul,
    triton_quantized_int8_conv2d,
    TRITON_AVAILABLE,
)
import os
import logging


USE_TRITON_KERNELS = os.getenv("USE_TRITON_KERNELS", "0") == "1"

if USE_TRITON_KERNELS:
    logging.warning(
        "Using Triton kernels for quantized linear and conv2d. These are experimental and might not work as expected"
    )


def is_supported_linear(
    spec_a: IntAffineQuantizationSpec,
    spec_b: IntAffineQuantizationSpec,
):
    if not TRITON_AVAILABLE or not USE_TRITON_KERNELS:
        return False
    return spec_a.nbits == 8 and spec_b.nbits == 8 and spec_a.signed and spec_b.signed


def is_supported_conv2d(
    spec_a: IntAffineQuantizationSpec, spec_b: IntAffineQuantizationSpec, conv
):
    if not TRITON_AVAILABLE or not USE_TRITON_KERNELS:
        return False
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
        weight_spec: IntAffineQuantizationSpec,
        input_info_or_spec: IntAffineQuantizationInfo | IntAffineQuantizationSpec,
        linear: nn.Linear,
        simulated: bool = False,
    ):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        assert isinstance(linear, nn.Linear), "linear must be an nn.Linear module"
        super().__init__(in_features, out_features, bias)
        self.weight_spec = weight_spec
        self.weight_info = calibrate(linear.weight, weight_spec)
        if isinstance(input_info_or_spec, IntAffineQuantizationInfo):
            self.input_info = input_info_or_spec
            self.input_spec = input_info_or_spec.spec
            self.online_quant = False
        else:
            self.input_spec = input_info_or_spec
            self.online_quant = True
        self.weight = nn.Parameter(
            quantize(linear.weight.detach(), self.weight_info), requires_grad=False
        )
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
            if bias
            else None
        )
        self.simulated = (
            simulated
            if is_supported_linear(self.weight_spec, self.input_spec)
            else True
        )

    def quantize_input(self, x: torch.Tensor):
        if self.online_quant:
            input_info = calibrate(x, self.input_spec)
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
            x, w = dequantize(x, input_info), dequantize(self.weight, self.weight_info)
            x = torch.nn.functional.linear(
                x,
                w,
            )
        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self):
        return (
            f"QuantizedLinear({self.in_features}, {self.out_features} | "
            f"W{self.weight_spec.nbits}{'S' if self.weight_spec.signed else 'U'}:{self.weight_spec.mode_args} | "
            f"A{self.input_spec.nbits}{'S' if self.input_spec.signed else 'U'}:{self.input_spec.mode_args})"
        )


class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_info_or_spec: IntAffineQuantizationInfo | IntAffineQuantizationSpec,
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
        assert isinstance(conv2d, nn.Conv2d), "Only nn.Conv2d is supported"
        self.weight_spec = weight_spec
        self.weight_info = calibrate(conv2d.weight, weight_spec)
        if isinstance(input_info_or_spec, IntAffineQuantizationInfo):
            self.input_info = input_info_or_spec
            self.input_spec = input_info_or_spec.spec
            self.online_quant = False
        else:
            self.input_spec = input_info_or_spec
            self.online_quant = True
        self.weight = nn.Parameter(
            quantize(conv2d.weight.detach(), self.weight_info), requires_grad=False
        )
        self.bias = (
            nn.Parameter(conv2d.bias.detach().clone(), requires_grad=False)
            if bias
            else None
        )
        self.simulated = (
            simulated
            if is_supported_conv2d(self.weight_spec, self.input_spec, conv2d)
            else True
        )

    def quantize_input(self, x: torch.Tensor):
        if self.online_quant:
            input_info = calibrate(x, self.input_spec)
            return quantize(x, input_info), input_info
        return quantize(x, self.input_info), self.input_info

    def forward(self, x: torch.Tensor):
        x, input_info = self.quantize_input(x)
        if self.simulated:
            x = dequantize(x, input_info)
            w = dequantize(self.weight, self.weight_info)
            conv2dres = torch.nn.functional.conv2d(
                x,
                w,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
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
        return (
            f"QuantizedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups} | "
            f"W{self.weight_spec.nbits}{'S' if self.weight_spec.signed else 'U'}:{self.weight_spec.mode_args} | "
            f"A{self.input_spec.nbits}{'S' if self.input_spec.signed else 'U'}:{self.input_spec.mode_args})"
        )


# ================================== CODEBOOK QUANTIZATION ==================================

# Only online quantization is supported for codebook quantization atm


class KMeansQuantizer:
    def _get_quantized_codebook(self, x: torch.Tensor, nbits: int):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("KMeansQuantizer requires scikit-learn")

        if self.init_uniform:
            amin, amax = x.min(), x.max()
            centroids = (
                torch.linspace(amin, amax, 2**nbits)
                .cpu()
                .detach()
                .numpy()
                .reshape(-1, 1)
            )
            kmeans = KMeans(n_clusters=2**nbits, init=centroids, n_init=1)
        else:
            kmeans = KMeans(n_clusters=2**nbits, n_init="auto")
        kmeans.fit(x.cpu().detach().numpy().reshape(-1, 1))
        codebook = torch.tensor(kmeans.cluster_centers_).to(x.device).reshape(-1)
        return codebook

    def __init__(self, tensor: torch.Tensor, nbits: int, init_uniform: bool = False):
        super().__init__()
        self.init_uniform = init_uniform
        self.codebook = nn.Parameter(
            self._get_quantized_codebook(tensor, nbits), requires_grad=False
        )
        self.nbits = nbits

    def quantize(self, x: torch.Tensor):
        arange = torch.arange(2**self.nbits).to(x.device)
        best_idxs = torch.argmin(
            (x.reshape(-1, 1) - self.codebook.reshape(1, -1)) ** 2, dim=1
        )
        ret = arange[best_idxs]
        ret = ret.reshape(x.shape)
        return ret

    def dequantize(self, x: torch.Tensor):
        return self.codebook[x]


def get_bias_correction_conv(kernel: torch.Tensor, quant_kernel):
    return kernel.float().mean(dim=(1, 2, 3)) - (
        quant_kernel.float().mean(dim=(1, 2, 3))
    )


def get_bias_correction_linear(kernel: torch.Tensor, quant_kernel):
    return kernel.float().mean(dim=1) - quant_kernel.float().mean(dim=1)


class KMeansQuantizedLinear(nn.Linear):
    def __init__(
        self,
        linear: nn.Linear,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        correct_bias: bool = False,
    ):
        super().__init__(
            linear.in_features, linear.out_features, linear.bias is not None
        )
        self.weight_spec = weight_spec
        self.input_spec = input_spec
        self.weight_quantizer = KMeansQuantizer(linear.weight, weight_spec.nbits)
        self.weight = nn.Parameter(
            self.weight_quantizer.quantize(linear.weight), requires_grad=False
        )
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
            if linear.bias is not None
            else None
        )

        if self.bias is not None and correct_bias:
            self.bias.data += get_bias_correction_linear(
                linear.weight, self.weight_quantizer.dequantize(self.weight)
            )
        elif correct_bias:
            self.bias = nn.Parameter(
                get_bias_correction_linear(
                    linear.weight, self.weight_quantizer.dequantize(self.weight)
                ),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor):
        w = self.weight_quantizer.dequantize(self.weight)
        if self.input_spec is not None:
            info = calibrate(x, self.input_spec)
            x = dequantize(quantize(x, info), info)
        return nn.functional.linear(x, w, self.bias)

    def __repr__(self):
        return (
            f"KMeansQuantizedLinear({self.in_features}, {self.out_features}, "
            f"W{self.weight_spec.nbits}{'S' if self.weight_spec.signed else 'U'}:{self.weight_spec.mode_args} | "
            f"A{self.input_spec.nbits}{'S' if self.input_spec.signed else 'U'}:{self.input_spec.mode_args})"
        )


class KMeansQuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        conv2d: nn.Conv2d,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        correct_bias: bool = False,
    ):
        super().__init__(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            conv2d.bias is not None,
        )
        self.weight_spec = weight_spec
        self.input_spec = input_spec
        self.weight_quantizer = KMeansQuantizer(conv2d.weight, weight_spec.nbits)
        self.weight = nn.Parameter(
            self.weight_quantizer.quantize(conv2d.weight), requires_grad=False
        )
        self.bias = (
            nn.Parameter(conv2d.bias.detach().clone(), requires_grad=False)
            if conv2d.bias is not None
            else None
        )

        if self.bias is not None and correct_bias:
            self.bias.data += get_bias_correction_conv(
                conv2d.weight, self.weight_quantizer.dequantize(self.weight)
            )
        elif correct_bias:
            self.bias = nn.Parameter(
                get_bias_correction_conv(
                    conv2d.weight, self.weight_quantizer.dequantize(self.weight)
                ),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor):
        w = self.weight_quantizer.dequantize(self.weight)
        if self.input_spec is not None:
            info = calibrate(x, self.input_spec)
            x = dequantize(quantize(x, info), info)
        return nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def __repr__(self):
        return (
            f"KMeansQuantizedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups} | "
            f"W{self.weight_spec.nbits}{'S' if self.weight_spec.signed else 'U'}:{self.weight_spec.mode_args} | "
            f"A{self.input_spec.nbits}{'S' if self.input_spec.signed else 'U'}:{self.input_spec.mode_args})"
        )
