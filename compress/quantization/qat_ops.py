from torch import nn
from compress.quantization.util import IntQuantizationSpec, IntQuantizationInfo
import torch
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq_ops import (
    fake_quantize,
    QuantizedConv2d,
    QuantizedLinear,
    quantize,
    dequantize,
)
from collections import defaultdict
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
        return f"FakeQuantizedLinear({self.in_features}, {self.out_features}, act_bits={self.input_spec.nbits}, weight_bits={self.weight_spec.nbits})"

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
        return f"FakeQuantizedConv2d({self.in_channels}, {self.out_channels}, ..., act_bits={self.input_spec.nbits}, weight_bits={self.weight_spec.nbits})"

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


def mse(x, y):
    return (x - y).pow(2).mean()


def _snap_loss_layer_params(
    layer: QATConv2d | QATLinear,
):
    weight = layer.weight
    quanted = quantize(weight, calibrate(weight, layer.weight_spec))
    dequanted = dequantize(quanted, calibrate(weight, layer.weight_spec)).detach()
    return mse(weight, dequanted)


def snap_loss_model_params(
    model: nn.Module,
):
    loss = 0
    for _, layer in model.named_modules():
        if isinstance(layer, (QATConv2d, QATLinear)):
            loss += _snap_loss_layer_params(layer)
    return loss


def snap_loss_model_activations(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
):
    loss = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (QATConv2d, QATLinear)) and name in activations:
            loss += mse(
                activations[name],
                dequantize(
                    quantize(
                        activations[name],
                        calibrate(activations[name], layer.input_spec),
                    ),
                    calibrate(activations[name], layer.input_spec),
                ),
            )
    return loss


class ActivationCatcher:
    def __init__(self, layer_types=(nn.ReLU, nn.Linear)):
        self.layer_types = layer_types
        self._activations = defaultdict(dict)
        self._hooks = defaultdict(list)

    def initialize(self, model, model_id=None):
        model_id = model_id or id(model)

        def get_hook(name):
            def hook(module, input, output):
                self._activations[model_id][name] = output.detach()

            return hook

        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                h = module.register_forward_hook(get_hook(name))
                self._hooks[model_id].append(h)

    def get_last_activations(self, model, model_id=None):
        model_id = model_id or id(model)
        return self._activations.get(model_id, {})

    def clear(self, model=None):
        if model:
            model_id = id(model)
            self._activations.pop(model_id, None)
        else:
            self._activations.clear()

    def remove_hooks(self, model=None):
        if model:
            model_id = id(model)
            for h in self._hooks.get(model_id, []):
                h.remove()
            self._hooks.pop(model_id, None)
        else:
            for hooks in self._hooks.values():
                for h in hooks:
                    h.remove()
            self._hooks.clear()


class SnapRegularizer:
    def __init__(self, model, do_activations=True, do_params=True):
        self.model = model
        self.activations = ActivationCatcher(layer_types=(QATLinear, QATConv2d))
        if do_activations:
            self.activations.initialize(model)
        self.do_activations, self.do_params = do_activations, do_params

        assert (
            do_activations or do_params
        ), "At least one of activations or params must be True"

    def snap_loss(self):
        res = {}
        if self.do_activations:
            res["activations"] = snap_loss_model_activations(
                self.model,
                self.activations.get_last_activations(self.model),
            )
        if self.do_params:
            res["params"] = snap_loss_model_params(self.model)
        return res


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
