from torch import nn
from compress.quantization.common import (
    IntAffineQuantizationSpec,
    fake_quantize,
    quantize,
    dequantize,
    ste_round,
)
import torch
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq import (
    QuantizedConv2d,
    QuantizedLinear,
)

import copy
from collections import defaultdict
import math


# ============================================================
# ============= REGULAR QUANTIZATION AWARE TRAINING ==========
# ============================================================


class QATLinear(nn.Linear):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Linear,
    ):
        in_features, out_features, bias = (
            original_layer.in_features,
            original_layer.out_features,
            original_layer.bias is not None,
        )
        assert isinstance(original_layer, nn.Linear), "Only nn.Linear is supported"
        super().__init__(in_features, out_features, bias)
        self.weight_spec = weight_spec
        self.input_spec = input_spec
        self.weight = nn.Parameter(
            original_layer.weight, requires_grad=False
        ).requires_grad_(True)
        self.bias = (
            nn.Parameter(original_layer.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )

    def forward(self, x: torch.Tensor):
        x = fake_quantize(x, calibrate(x, self.input_spec))
        w = fake_quantize(self.weight, calibrate(self.weight, self.weight_spec))
        return nn.functional.linear(x, w, self.bias)

    def to_linear(self):
        ret = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        ret.weight = self.weight
        ret.bias = self.bias
        return ret

    def __repr__(self):
        return f"FakeQuantizedLinear({self.in_features}, {self.out_features}, act_bits={self.input_spec.nbits}, weight_bits={self.weight_spec.nbits})"


class QATConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Conv2d,
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
            original_layer.in_channels,
            original_layer.out_channels,
            original_layer.kernel_size,
            original_layer.stride,
            original_layer.padding,
            original_layer.dilation,
            original_layer.groups,
            original_layer.bias is not None,
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
        assert isinstance(original_layer, nn.Conv2d), "Only nn.Conv2d is supported"
        self.weight_spec = weight_spec
        self.input_spec = input_spec
        self.weight = nn.Parameter(
            original_layer.weight, requires_grad=False
        ).requires_grad_(True)
        self.bias = (
            nn.Parameter(original_layer.bias, requires_grad=False).requires_grad_(True)
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


# ============================================================
# ============= SNAP REGULARIZATION ==========================
# ============================================================


def _snap_loss_layer_params(
    layer: QATConv2d | QATLinear,
):
    weight = layer.weight
    quanted = quantize(weight, calibrate(weight, layer.weight_spec))
    dequanted = dequantize(quanted, calibrate(weight, layer.weight_spec)).detach()
    return (weight - dequanted).pow(2).mean()


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
            loss += (
                (
                    activations[name]
                    - fake_quantize(
                        activations[name],
                        calibrate(activations[name], layer.input_spec),
                    )
                )
                .pow(2)
                .mean()
            )
    return loss / len(activations)


import torch.nn as nn


class ActivationCatcher:
    def __init__(self, layer_types=(nn.ReLU, nn.Linear), include_inputs=False):
        self.layer_types = layer_types
        self.include_inputs = include_inputs
        self._activations = defaultdict(dict)
        self._hooks = defaultdict(list)

    def initialize(self, model, model_id=None):
        model_id = model_id or id(model)

        def get_hook(name):
            def hook(module, input, output):
                entry = {"output": output}
                if self.include_inputs:
                    entry["input"] = tuple(i for i in input)
                self._activations[model_id][name] = entry

            return hook

        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                h = module.register_forward_hook(get_hook(name))
                self._hooks[model_id].append(h)

    def get_last_activations(self, model, model_id=None, clear=True):
        model_id = model_id or id(model)
        ret = self._activations.get(model_id, {})
        if clear:
            self.clear(model_id)
        return ret

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


# ===========================================================================
# ============= MUTUAL INFORMATION REGULARIZATION ==========================
# ===========================================================================


def mutual_info_gaussian(Y, Y_tilde, eps=1e-4):
    Y = Y - Y.mean(dim=0, keepdim=True)
    Y_tilde = Y_tilde - Y_tilde.mean(dim=0, keepdim=True)

    cov = (Y * Y_tilde).mean(dim=0)
    var_Y = (Y**2).mean(dim=0)
    var_Y_tilde = (Y_tilde**2).mean(dim=0)

    var_Y = torch.clamp(var_Y, min=eps)
    var_Y_tilde = torch.clamp(var_Y_tilde, min=eps)

    rho_squared = cov**2 / (var_Y * var_Y_tilde + eps)
    rho_squared = torch.clamp(rho_squared, max=1 - eps)

    mi = -0.5 * torch.log(1 - rho_squared + eps)

    return mi.sum()


def mutual_info_loss(model, acts):
    loss = 0
    for name, layer in model.named_modules():
        if name in acts:
            act = acts[name]
            layer = (
                layer.to_linear()
                if hasattr(layer, "to_linear")
                else layer.to_conv2d() if hasattr(layer, "to_conv2d") else None
            )
            if layer is None:
                continue
            inp = act["input"]
            if isinstance(inp, (tuple, list)):
                inp = inp[0]
            out_unquant = layer(inp)
            out_unquant = out_unquant.reshape(out_unquant.shape[0], -1)
            out_quant = act["output"].reshape(act["output"].shape[0], -1)
            loss += mutual_info_gaussian(out_unquant, out_quant)
    return loss / len(acts)


class MutualInfoRegularizer:
    def __init__(self, model):
        self.model = model
        self.activations = ActivationCatcher(
            layer_types=(QATLinear, QATConv2d, LSQConv2d, LSQLinear),
            include_inputs=True,
        )
        self.activations.initialize(model)

    def mutual_info_quant_loss(self):
        res = {}
        acts = self.activations.get_last_activations(self.model)
        assert len(acts) > 0, "No activations found. Make sure to run the model first."
        if acts:
            res["mutual_info_loss"] = -mutual_info_loss(self.model, acts)
        return res


# ============================================================
# ============= LEARNED STEP QUANTIZATION (LSQ) ==============
# ============================================================


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
        v_s = x / scale
        mask = (v_s >= qmin) & (v_s <= qmax)
        x_grad = grad_output * mask.float()
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
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        linear: nn.Linear,
        data_batch: torch.Tensor = None,
        online: bool = True,
    ):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        assert isinstance(linear, nn.Linear), "Only nn.Linear is supported"
        super().__init__(in_features, out_features, bias)
        self.weight_info = calibrate(linear.weight, weight_spec)
        self.weight_info.scale.requires_grad_(True)
        self.weight = nn.Parameter(linear.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(linear.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )
        self.online = online
        if not online:
            assert data_batch is not None, "data_batch is required for offline LSQ"
            self.input_info = calibrate(data_batch, input_spec)
            self.input_info.scale.requires_grad_(True)
        else:
            self.input_spec = input_spec

    def forward(self, x: torch.Tensor):
        input_info = calibrate(x, self.input_spec) if self.online else self.input_info
        x = LSQQuantize.apply(x, input_info.scale, input_info)
        w = LSQQuantize.apply(self.weight, self.weight_info.scale, self.weight_info)
        return nn.functional.linear(x, w, self.bias)

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
        input_info = (
            self.input_info
            if not self.online
            else calibrate(torch.empty(1), self.input_spec)
        )
        ret = QuantizedLinear(self.weight_info.spec, input_info, self.to_linear())
        ret.weight_info = self.weight_info
        ret.weight = torch.nn.Parameter(
            quantize(self.weight, self.weight_info), requires_grad=False
        )
        return ret


class LSQConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        conv2d: nn.Conv2d,
        data_batch: torch.Tensor = None,
        online: bool = True,
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
        self.weight_info = calibrate(conv2d.weight, weight_spec)
        self.weight_info.scale.requires_grad_(True)
        self.weight = nn.Parameter(conv2d.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(conv2d.bias, requires_grad=False).requires_grad_(True)
            if bias
            else None
        )
        self.online = online
        if not online:
            assert data_batch is not None, "data_batch is required for offline LSQ"
            self.input_info = calibrate(data_batch, input_spec)
            self.input_info.scale.requires_grad_(True)
        else:
            self.input_spec = input_spec

    def forward(self, x: torch.Tensor):
        input_info = calibrate(x, self.input_spec) if self.online else self.input_info
        x = LSQQuantize.apply(x, input_info.scale, input_info)
        w = LSQQuantize.apply(self.weight, self.weight_info.scale, self.weight_info)
        return nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

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
        input_info = (
            self.input_info
            if not self.online
            else calibrate(torch.empty(1), self.input_spec)
        )
        ret = QuantizedConv2d(self.weight_info.spec, input_info, self.to_conv2d())
        ret.weight_info = self.weight_info
        ret.weight = torch.nn.Parameter(
            quantize(self.weight, self.weight_info), requires_grad=False
        )
        return ret


# ============================================================
# ============= PACT QUANTIZATION ============================
# ============================================================

# References https://arxiv.org/pdf/1805.06085


class PACTReLU(nn.ReLU):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__(inplace)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        # print(self.alpha.grad, self.alpha)
        return torch.clamp(x, torch.tensor(0).to(x.device), self.alpha)

    def __repr__(self):
        return f"PACTReLU(alpha={self.alpha}, inplace={self.inplace})"


def get_regularizer_for_pact(model):
    # finds PACT layers and returns a regularizer for them: L2 of the alpha parameter
    pact_layers = []
    for name, module in model.named_modules():
        if isinstance(module, PACTReLU):
            pact_layers.append(module)
    if not pact_layers:

        def pact_regularizer():
            return torch.tensor(0.0)

    def pact_regularizer():
        return sum((layer.alpha**2).sum() for layer in pact_layers)

    return pact_regularizer


# ============================================================
# ============= AUTOMATIC BIT ALLOCATION ==========================
# ============================================================

# Using gradient-informed optimizer, it optimizes the number of bits needed


def override_property(obj, name, value):
    obj.__class__ = type(
        f"Patched{obj.__class__.__name__}",
        (obj.__class__,),
        {name: property(lambda self: value)},
    )


class AutoBitAllocationLinear(nn.Module):
    def __init__(
        self,
        initial_weight_spec: IntAffineQuantizationSpec,
        initial_input_spec: IntAffineQuantizationSpec,
        linear: nn.Linear,
    ):
        super().__init__()
        self.b_w = torch.nn.Parameter(
            torch.tensor(initial_weight_spec.nbits).float(), requires_grad=True
        )
        self.b_a = torch.nn.Parameter(
            torch.tensor(initial_input_spec.nbits).float(), requires_grad=True
        )
        self.initial_weight_spec = initial_weight_spec
        self.initial_input_spec = initial_input_spec
        self.weight = nn.Parameter(linear.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(linear.bias, requires_grad=False).requires_grad_(True)
            if linear.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor):
        def _quantize_diff_b(x, b, spec):
            qmin = -(2 ** (b - 1))
            qmax = 2 ** (b - 1) - 1
            spec = copy.deepcopy(spec)
            spec.nbits = int(b.item())
            override_property(spec, "qmin", qmin)
            override_property(spec, "qmax", qmax)
            calib = calibrate(x, spec)
            scale = calib.scale
            # assert spec.zero_point is None, "Zero point is not supported"
            # manual quantization

            q = torch.clamp(ste_round(x / scale), min=qmin, max=qmax)
            # dequantization
            x_ = q * scale
            return x_

        x = _quantize_diff_b(x, self.b_a, self.initial_input_spec)
        w = _quantize_diff_b(self.weight, self.b_w, self.initial_weight_spec)
        return nn.functional.linear(x, w, self.bias)

    def __repr__(self):
        return f"AutoBitAllocationLinear({self.initial_weight_spec}, {self.initial_input_spec})"

    def to_linear(self):
        ret = nn.Linear(
            self.initial_weight_spec.nbits,
            self.initial_input_spec.nbits,
            self.bias is not None,
        )
        ret.weight = self.weight
        ret.bias = self.bias
        return ret


class AutoBitAllocationConv2d(nn.Module):
    def __init__(
        self,
        initial_weight_spec: IntAffineQuantizationSpec,
        initial_input_spec: IntAffineQuantizationSpec,
        conv2d: nn.Conv2d,
    ):
        super().__init__()
        self.b_w = torch.nn.Parameter(
            torch.tensor(initial_weight_spec.nbits).float(), requires_grad=True
        )
        self.b_a = torch.nn.Parameter(
            torch.tensor(initial_input_spec.nbits).float(), requires_grad=True
        )
        self.initial_weight_spec = initial_weight_spec
        self.initial_input_spec = initial_input_spec
        self.weight = nn.Parameter(conv2d.weight, requires_grad=False).requires_grad_(
            True
        )
        self.bias = (
            nn.Parameter(conv2d.bias, requires_grad=False).requires_grad_(True)
            if conv2d.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor):
        def _quantize_diff_b(x, b, spec):
            spec = copy.deepcopy(spec)
            spec.nbits = int(b.item())
            calib = calibrate(x, spec)
            scale = calib.scale
            assert spec.zero_point is None, "Zero point is not supported"
            # manual quantization
            qmin = -(2 ** (b - 1))
            qmax = 2 ** (b - 1) - 1
            q = torch.clamp(torch.round(x / scale), min=qmin, max=qmax)

            # dequantization
            x_ = q * scale
            return x_

        x = _quantize_diff_b(x, self.b_a, self.initial_input_spec)
        w = _quantize_diff_b(self.weight, self.b_w, self.initial_weight_spec)
        return nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def __repr__(self):
        return f"AutoBitAllocationConv2d({self.initial_weight_spec}, {self.initial_input_spec})"

    def to_conv2d(self):
        ret = nn.Conv2d(
            self.initial_weight_spec.nbits,
            self.initial_input_spec.nbits,
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


# ============================================================
