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
from functools import partial

# ============================================================
# ============= REGULAR QUANTIZATION AWARE TRAINING ==========
# ============================================================


def _qat_prepare(
    self,
    layer: nn.Module,
    weight_spec: IntAffineQuantizationSpec,
    input_spec: IntAffineQuantizationSpec,
):
    self.weight_spec = weight_spec
    self.input_spec = input_spec
    self.weight = nn.Parameter(layer.weight, requires_grad=False).requires_grad_(True)
    self.bias = (
        nn.Parameter(layer.bias, requires_grad=False).requires_grad_(True)
        if layer.bias is not None
        else None
    )


def _forward_qat(
    self,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    functional: callable,
    **kwargs,
):
    x = fake_quantize(x, calibrate(x, self.input_spec))
    w = fake_quantize(weight, calibrate(weight, self.weight_spec))
    return functional(x, w, bias, **kwargs)


def _extract_kwargs_from_orig_layer(self, keys_list, orig_layer):
    # Extract kwargs from the original layer
    for key in keys_list:
        setattr(self, key, getattr(orig_layer, key))


def _to_float(self, kwargs_to_extract, orig_layer_cls):
    # Extract kwargs from the original layer
    kwargs = {key: getattr(self, key) for key in kwargs_to_extract}
    rt = orig_layer_cls(**kwargs)
    rt.weight = self.weight
    rt.bias = self.bias
    return rt


class QATLinear(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Linear,
    ):
        super().__init__()
        _extract_kwargs_from_orig_layer(
            self,
            ["in_features", "out_features", "bias"],
            original_layer,
        )
        _qat_prepare(self, original_layer, weight_spec, input_spec)

    def forward(self, x: torch.Tensor):
        return _forward_qat(
            self,
            x,
            self.weight,
            self.bias,
            nn.functional.linear,
        )

    to_linear = partial(
        _to_float,
        ["in_features", "out_features", "bias"],
        nn.Linear,
    )


def __repr__(self):
    return (
        f"QATLinear(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
        f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
        f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
        f"{self.in_features}, {self.out_features}, {self.bias})"
    )


class QATConv2d(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Conv2d,
    ):
        super().__init__()
        _extract_kwargs_from_orig_layer(
            self,
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
            ],
            original_layer,
        )
        _qat_prepare(self, original_layer, weight_spec, input_spec)

    def forward(self, x: torch.Tensor):
        return _forward_qat(
            self,
            x,
            self.weight,
            self.bias,
            nn.functional.conv2d,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    to_conv2d = partial(
        _to_float,
        [
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
        ],
        nn.Conv2d,
    )

    def __repr__(self):
        return (
            f"QATConv2d(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups})"
        )


def _get_bn_and_conv_weight(conv, bn):
    # Get the weight and bias of the conv layer
    w = conv.weight
    b = conv.bias if conv.bias is not None else 0

    # Get the weight and bias of the bn layer
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    # returns the transformed weight and bias
    inv_std = gamma / torch.sqrt(running_var + eps)  # shape [out_channels]
    w = w * inv_std.reshape(-1, 1, 1, 1)
    b = beta + (b - running_mean) * inv_std
    return w, b


class FusedQATConv2dBatchNorm2d(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_conv_layer: nn.Conv2d,
        original_bn_layer: nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv = original_conv_layer
        self.bn = original_bn_layer
        self.weight_spec, self.input_spec = weight_spec, input_spec

        _extract_kwargs_from_orig_layer(
            self,
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
            ],
            original_conv_layer,
        )
        _extract_kwargs_from_orig_layer(
            self,
            ["num_features", "eps", "momentum"],
            original_bn_layer,
        )

    def forward(self, x: torch.Tensor):
        w = self.conv.weight
        b = self.conv.bias

        w, b = _get_bn_and_conv_weight(self.conv, self.bn)

        return _forward_qat(
            self,
            x,
            w,
            b,
            nn.functional.conv2d,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    to_conv2d = partial(
        _to_float,
        [
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
        ],
        nn.Conv2d,
    )

    @property
    def weight(self):
        return _get_bn_and_conv_weight(self.conv, self.bn)[0]

    def __repr__(self):
        return (
            f"FusedQATConv2dBatchNorm(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups})"
        )


# ============================================================
# ============= LEARNED STEP QUANTIZATION (LSQ) ==============
# ============================================================


class LSQQuantize(torch.autograd.Function):
    # zero_point might be None
    @staticmethod
    def forward(ctx, x, scale, zero_point, info):
        if zero_point is None:
            zero_point = info.zero_point
        assert zero_point is None, "Zero point is not supported"
        assert (
            info.scale is scale
        ), "Scale must be the same as the one used in calibration"

        if zero_point is not None:
            ctx.save_for_backward(x, scale, zero_point)
        else:
            ctx.save_for_backward(x, scale)
        ctx.qmin = info.qmin
        ctx.qmax = info.qmax
        ctx.spec = info.spec
        return fake_quantize(x, info)

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 3:
            x, scale, zero_point = ctx.saved_tensors
        else:
            x, scale = ctx.saved_tensors
            zero_point = None

        if zero_point is None:
            qmin, qmax = ctx.qmin, ctx.qmax
            spec = ctx.spec
            x_grouped = spec.grouper.group(x)
            v_s = x_grouped / scale
            mask = (v_s >= qmin) & (v_s <= qmax)
            x_grad = ctx.spec.grouper.group(grad_output) * mask.float()
            s_grad = (
                torch.where(
                    v_s <= qmin,
                    qmin,
                    torch.where(qmax <= v_s, qmax, -v_s + torch.round(v_s).detach()),
                )
                * ctx.spec.grouper.group(grad_output)
            ).sum(dim=0)

            """
            # rescale as the paper mentions
            rescaling = 1 / math.sqrt((x.numel() * (qmax)))
            """
            # the previous is for the per-tensor case
            # generalizing this, we need to rescale by the number of elements in the group
            numels_grp = x_grouped.shape[0]  # shape[1] is the number of groups

            s_grad = s_grad * (1 / math.sqrt(numels_grp * (qmax)))
            # print(s_grad)
            return ctx.spec.grouper.ungroup(x_grad, x), s_grad, None, None

        else:
            raise NotImplementedError("Zero point is not supported")


def _forward_lsq(
    self,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    functional: callable,
    **kwargs,
):
    if self.online:
        info = calibrate(x, self.input_spec)
        x = LSQQuantize.apply(x, info.scale, info.zero_point, info)
    else:
        x = LSQQuantize.apply(
            x, self.input_info.scale, self.input_info.zero_point, self.input_info
        )
    w = LSQQuantize.apply(
        weight, self.weight_info.scale, self.weight_info.zero_point, self.weight_info
    )
    return functional(x, w, bias, **kwargs)


def _lsq_prepare(
    self,
    layer: nn.Module,
    weight_spec: IntAffineQuantizationSpec,
    input_spec: IntAffineQuantizationSpec,
    data_batch: torch.Tensor = None,
    online: bool = False,
):
    self.weight_spec = weight_spec
    self.weight_info = calibrate(layer.weight, weight_spec)
    self.weight_info.scale.requires_grad_(True)

    self.weight = nn.Parameter(layer.weight, requires_grad=False).requires_grad_(True)
    self.bias = (
        nn.Parameter(layer.bias, requires_grad=False).requires_grad_(True)
        if layer.bias is not None
        else None
    )

    self.online = online
    if not online:
        assert data_batch is not None, "data_batch is required for offline LSQ"
        self.input_info = calibrate(data_batch, input_spec)
        self.input_info.scale.requires_grad_(True)
        self.input_spec = input_spec
    else:
        self.input_spec = input_spec


def _lsq_to_float(self, kwargs_to_extract, orig_layer_cls):
    # Extract kwargs from the original layer
    kwargs = {key: getattr(self, key) for key in kwargs_to_extract}
    rt = orig_layer_cls(**kwargs)
    rt.weight = self.weight
    rt.bias = self.bias
    return rt


def _lsq_to_quant(self, orig_layer_cls, quant_layer_cls):
    # Extract kwargs from the original layer
    kwargs = {
        key: getattr(self, key) for key in ["in_features", "out_features", "bias"]
    }
    input_info = (
        self.input_info
        if not self.online
        else calibrate(torch.empty(1), self.input_spec)
    )

    rt = quant_layer_cls(self.weight_info.spec, input_info, orig_layer_cls(**kwargs))
    rt.weight_info = self.weight_info
    rt.weight = torch.nn.Parameter(
        quantize(self.weight, self.weight_info), requires_grad=False
    )
    return rt


class LSQLinear(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Linear,
        data_batch: torch.Tensor = None,
        online: bool = False,
    ):
        super().__init__()
        _extract_kwargs_from_orig_layer(
            self,
            ["in_features", "out_features", "bias"],
            original_layer,
        )
        _lsq_prepare(self, original_layer, weight_spec, input_spec, data_batch, online)

    def forward(self, x: torch.Tensor):
        return _forward_lsq(
            self,
            x,
            self.weight,
            self.bias,
            nn.functional.linear,
        )

    def __repr__(self):
        return (
            f"LSQQuantizedLinear(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_features}, {self.out_features}, {self.bias})"
        )

    to_linear = partial(
        _lsq_to_float,
        ["in_features", "out_features", "bias"],
        nn.Linear,
    )

    to_quant_linear = partial(
        _lsq_to_quant,
        nn.Linear,
        QuantizedLinear,
    )


conv2d_kwargs = [
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "bias",
]


class LSQConv2d(nn.Module):

    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        conv2d: nn.Conv2d,
        data_batch: torch.Tensor = None,
        online: bool = False,
    ):
        super().__init__()
        _extract_kwargs_from_orig_layer(self, conv2d_kwargs, conv2d)
        _lsq_prepare(self, conv2d, weight_spec, input_spec, data_batch, online)

    def forward(self, x: torch.Tensor):
        return _forward_lsq(
            self,
            x,
            self.weight,
            self.bias,
            nn.functional.conv2d,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def __repr__(self):
        return (
            f"LSQConv2d(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups})"
        )

    to_conv2d = partial(
        _lsq_to_float,
        conv2d_kwargs,
        nn.Conv2d,
    )

    to_quant_conv2d = partial(
        _lsq_to_quant,
        nn.Conv2d,
        QuantizedConv2d,
    )


class FusedLSQConv2dBatchNorm2d(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        conv2d: nn.Conv2d,
        bn: nn.BatchNorm2d,
        data_batch: torch.Tensor = None,
        online: bool = False,
    ):
        super().__init__()
        self.conv = conv2d
        self.bn = bn
        self.weight_info = calibrate(
            _get_bn_and_conv_weight(conv2d, bn)[0],
            weight_spec,
        )
        self.weight_info.scale.requires_grad_(True)
        _extract_kwargs_from_orig_layer(
            self,
            conv2d_kwargs,
            conv2d,
        )
        _extract_kwargs_from_orig_layer(
            self,
            ["num_features", "eps", "momentum"],
            bn,
        )

        self.weight_spec = weight_spec
        self.input_spec = input_spec

        self.online = online
        if not online:
            assert data_batch is not None, "data_batch is required for offline LSQ"
            self.input_info = calibrate(data_batch, input_spec)
            self.input_info.scale.requires_grad_(True)
        else:
            self.input_info = calibrate(torch.empty(1), input_spec)
            self.input_spec = input_spec

    def forward(self, x: torch.Tensor):
        w, b = _get_bn_and_conv_weight(self.conv, self.bn)
        return _forward_lsq(
            self,
            x,
            w,
            b,
            nn.functional.conv2d,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    @property
    def weight(self):
        return _get_bn_and_conv_weight(self.conv, self.bn)[0]

    def __repr__(self):
        return (
            f"FusedLSQConv2dBatchNorm(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups})"
        )


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


def safe_logdet(cov):
    sign, logabsdet = torch.linalg.slogdet(cov)
    if (sign <= 0).any():
        # If the matrix is not positive-definite, logdet doesn't make sense
        return torch.tensor(float("nan"), device=cov.device)
    return logabsdet


def mutual_info_gaussian(Y, Y_tilde, eps=1e-4):
    B, D = Y.shape
    Y = Y - Y.mean(dim=0, keepdim=True)
    Y_tilde = Y_tilde - Y_tilde.mean(dim=0, keepdim=True)

    joint = torch.cat([Y, Y_tilde], dim=1)  # [B, 2D]
    cov = (joint.T @ joint) / (B - 1)
    cov += eps * torch.eye(2 * D, device=Y.device)

    cov_Y = cov[:D, :D]
    cov_Y_tilde = cov[D:, D:]

    logdet_Y = safe_logdet(cov_Y)
    logdet_Y_tilde = safe_logdet(cov_Y_tilde)
    logdet_joint = safe_logdet(cov)

    if (
        torch.isnan(logdet_Y)
        or torch.isnan(logdet_Y_tilde)
        or torch.isnan(logdet_joint)
    ):
        return torch.tensor(0.0, device=Y.device)

    mi = 0.5 * (logdet_Y + logdet_Y_tilde - logdet_joint)
    return mi


def project_features(feat, d_out=256):
    B, D = feat.shape
    proj = torch.randn(D, d_out, device=feat.device) / D**0.5
    return feat @ proj


def mutual_info_loss(student_acts, teacher_acts, proj_dim=256):
    losses = []

    for name, actstu in student_acts.items():
        if name not in teacher_acts:
            continue

        stu = actstu["output"]
        tea = teacher_acts[name]["output"]

        if stu.dim() == 4:
            stu = stu.reshape(stu.shape[0], -1)
            tea = tea.reshape(tea.shape[0], -1)

        if stu.shape[0] < 2:
            continue  # can't estimate covariance from 1 sample

        stu_proj = project_features(stu, proj_dim)
        tea_proj = project_features(tea, proj_dim)

        mi = mutual_info_gaussian(stu_proj, tea_proj)
        losses.append(mi)

    if not losses:
        return torch.tensor(0.0, device=stu.device)

    return -torch.stack(losses).mean()


class MutualInfoRegularizer:
    def __init__(self, student, teacher):
        self.teacher = teacher
        self.student = student
        self.activations = ActivationCatcher(
            layer_types=(
                QATLinear,
                QATConv2d,
                LSQConv2d,
                LSQLinear,
                nn.Linear,
                nn.Conv2d,
            ),
            include_inputs=True,
        )
        self.activations.initialize(self.student)
        self.activations.initialize(self.teacher)

    def mutual_info_quant_loss(self):
        res = {}
        actsstu = self.activations.get_last_activations(self.student)
        actstea = self.activations.get_last_activations(self.teacher)
        assert (
            len(actsstu) > 0
        ), "No activations found. Make sure to run the model first."
        res["mutual_info_loss"] = -mutual_info_loss(actsstu, actstea)
        return res


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
