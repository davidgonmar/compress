from torch import nn
from compress.quantization.common import (
    IntAffineQuantizationSpec,
    fake_quantize,
    quantize,
)
import torch
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq import (
    QuantizedConv2d,
    QuantizedLinear,
)

import math
from functools import partial
from typing import Callable

# ============================================================
# ============= REGULAR QUANTIZATION AWARE TRAINING ==========
# ============================================================


class EMAInfoObserver(nn.Module):
    def __init__(self, spec, averaging_constant: float = 0.01):
        super().__init__()
        self.spec = spec
        self.averaging_constant = averaging_constant
        self.initialized = False
        self.frozen = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.frozen or not self.training:
            assert self.initialized, "Observer not initialized"
            return self.info

        info = calibrate(x, self.spec)

        if not self.initialized:
            self.add_module("info", info)
            self.initialized = True
        else:
            self.info.scale.mul_(1 - self.averaging_constant).add_(
                info.scale * self.averaging_constant
            )
            if info.zero_point is not None:
                self.info.zero_point.mul_(1 - self.averaging_constant).add_(
                    info.zero_point * self.averaging_constant
                )
        return self.info

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False


def _qat_prepare(
    self,
    layer: nn.Module,
    weight_spec: IntAffineQuantizationSpec,
    input_spec: IntAffineQuantizationSpec,
):
    self.weight_spec = weight_spec
    self.input_spec = input_spec
    self.weight = nn.Parameter(layer.weight.detach().clone(), requires_grad=True)
    self.bias = (
        nn.Parameter(layer.bias.detach().clone(), requires_grad=True)
        if layer.bias is not None
        else None
    )
    if not self.online:
        self.input_observer = EMAInfoObserver(input_spec)


def _forward_qat(
    self,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    functional: Callable,
    **kwargs,
):
    if not self.online:
        x = fake_quantize(x, self.input_observer(x))
    else:
        x = fake_quantize(x, calibrate(x, self.input_spec))
    w = fake_quantize(weight, calibrate(weight, self.weight_spec))
    return functional(x, w, bias, **kwargs)


def _extract_kwargs_from_orig_layer(self, keys_list, orig_layer):
    for key in keys_list:
        setattr(self, key, getattr(orig_layer, key))


def _to_float(self, kwargs_to_extract, orig_layer_cls):
    kwargs = {key: getattr(self, key) for key in kwargs_to_extract}
    rt = orig_layer_cls(**kwargs, bias=self.bias is not None)
    rt.weight = self.weight.detach().clone()
    if self.bias is not None:
        rt.bias = self.bias.detach().clone()
    return rt


class QATLinear(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Linear,
        online: bool = False,
    ):
        super().__init__()
        self.online = online
        _extract_kwargs_from_orig_layer(
            self,
            ["in_features", "out_features"],
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
        ["in_features", "out_features"],
        nn.Linear,
    )

    def __repr__(self):
        return (
            f"QATLinear(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_features}, {self.out_features}, {self.bias is not None})"
        )


class QATConv2d(nn.Module):
    def __init__(
        self,
        weight_spec: IntAffineQuantizationSpec,
        input_spec: IntAffineQuantizationSpec,
        original_layer: nn.Conv2d,
        online: bool = False,
    ):
        super().__init__()
        self.online = online
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
        ],
        nn.Conv2d,
    )

    def __repr__(self):
        return (
            f"QATConv2d(W{'S' if self.weight_spec.signed else 'U'}{self.weight_spec.nbits}"
            f"A{'S' if self.input_spec.signed else 'U'}{self.input_spec.nbits}, "
            f"WGrouper={self.weight_spec.grouper}, AGrouper={self.input_spec.grouper}, "
            f"{self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"{self.stride}, {self.padding}, {self.dilation}, {self.groups}, {self.bias is not None})"
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
        online: bool = False,
        bn_track_running_stats: bool = True,
        use_fast_bn_path=False,
    ):
        super().__init__()
        self.online = online
        self.conv = original_conv_layer
        self.bn = original_bn_layer
        self.weight_spec, self.input_spec = weight_spec, input_spec
        if not self.online:
            self.input_observer = EMAInfoObserver(input_spec)
        self.bn_track_running_stats = bn_track_running_stats
        self.use_fast_bn_path = use_fast_bn_path

    def forward(self, x: torch.Tensor):
        # similar to the default path of https://github.com/pytorch/pytorch/blob/v2.7.0/torch/ao/nn/intrinsic/qat/modules/conv_fused.py
        if self.training and self.bn_track_running_stats:
            if self.use_fast_bn_path:
                x_quant = (
                    fake_quantize(x, self.input_observer(x))
                    if not self.online
                    else fake_quantize(x, calibrate(x, self.input_spec))
                )
                conv = self.conv
                bn = self.bn
                w = conv.weight
                eps = bn.eps
                w_scale = bn.weight / torch.sqrt(
                    bn.running_var + eps
                )  # shape [out_channels]
                # we simulate qat forward with running mean/var (as those will be used during inference)
                w_simulated = w * w_scale.reshape(-1, 1, 1, 1)
                w_quant = fake_quantize(
                    w_simulated, calibrate(w_simulated, self.weight_spec)
                )  # quantize the simulated weight
                conv_out = nn.functional.conv2d(
                    x_quant,
                    w_quant,
                    bias=None,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                )
                conv_orig_simulated = conv_out / w_scale.reshape(
                    1, -1, 1, 1
                )  # simulate the conv output with the original weight
                if conv.bias is not None:
                    conv_orig_simulated = conv_orig_simulated + conv.bias.reshape(
                        1, -1, 1, 1
                    )
                # now we apply the batch norm
                return self.bn(
                    conv_orig_simulated,
                )
            else:  # slow path
                # first, update bn
                x = (
                    fake_quantize(x, self.input_observer(x))
                    if not self.online
                    else fake_quantize(x, calibrate(x, self.input_spec))
                )
                conv_out_float_weight = torch.nn.functional.conv2d(
                    x,
                    self.conv.weight,
                    bias=None,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups,
                )
                with torch.no_grad():
                    if self.conv.bias is not None:
                        conv_out_for_bn = (
                            self.conv.bias.reshape(1, -1, 1, 1) + conv_out_float_weight
                        )
                    else:
                        conv_out_for_bn = conv_out_float_weight
                    # update batch norm stats
                    self.bn(conv_out_for_bn)
                # pass with running var
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                w_scale = self.bn.weight / running_std  # shape [out_channels]
                conv_out = nn.functional.conv2d(
                    x,
                    self.conv.weight * w_scale.reshape(-1, 1, 1, 1),
                    bias=None,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups,
                )
                # compute BATCH statistics
                batch_mean = conv_out_float_weight.mean(dim=(0, 2, 3))
                batch_var = conv_out_float_weight.var(dim=(0, 2, 3), unbiased=False)
                batch_std = torch.sqrt(batch_var + self.bn.eps)
                scale_to_batch = running_std / batch_std
                conv_out_rescaled = conv_out * scale_to_batch.reshape(
                    1, -1, 1, 1
                )  # scaled to batch stats
                # bias
                fused_bias = (
                    self.bn.bias - self.bn.weight * batch_mean / batch_std
                ).reshape(1, -1, 1, 1)
                # now we apply the batch norm
                return conv_out_rescaled + fused_bias
        else:
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
            f"{self.conv.in_channels}, {self.conv.out_channels}, {self.conv.kernel_size}, "
            f"{self.conv.stride}, {self.conv.padding}, {self.conv.dilation}, {self.conv.groups})"
        )


# ============================================================
# ============= LEARNED STEP QUANTIZATION (LSQ) ==============
# ============================================================


class LSQQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, info):
        assert (
            zero_point is info.zero_point
        ), "Zero point must be the same as the one used in calibration"
        assert zero_point is None, "Zero point is not supported"
        assert (
            info.scale is scale
        ), "Scale must be the same as the one used in calibration"

        ctx.save_for_backward(x, scale)
        ctx.qmin = info.qmin
        ctx.qmax = info.qmax
        ctx.spec = info.spec
        return fake_quantize(x, info)

    @staticmethod
    def backward(ctx, grad_output):
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

            # the original paper rescales with scale = 1 / math.sqrt((x.numel() * (qmax)))
            # which applies for per-tensor quantization
            # generalizing this, we need to rescale by the number of elements in the group
            numels_grp = x_grouped.shape[0]  # shape[1] is the number of groups

            s_grad = s_grad * (1 / math.sqrt(numels_grp * (qmax)))
            return ctx.spec.grouper.ungroup(x_grad, x), s_grad, None, None
        else:
            raise NotImplementedError("Zero point is not supported")


def _forward_lsq(
    self,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    functional: Callable,
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

    self.weight = nn.Parameter(layer.weight.detach().clone(), requires_grad=True)
    self.bias = (
        nn.Parameter(layer.bias.detach().clone(), requires_grad=True)
        if layer.bias is not None
        else None
    )

    self.online = online
    if not online:
        assert (
            data_batch is not None
        ), "data_batch is required to initialize offline LSQ"
        self.input_info = calibrate(data_batch, input_spec)
        self.input_info.scale.requires_grad_(True)
        self.input_spec = input_spec
    else:
        self.input_spec = input_spec


def _lsq_to_float(self, kwargs_to_extract, orig_layer_cls):
    # Extract kwargs from the original layer
    kwargs = {key: getattr(self, key) for key in kwargs_to_extract}
    rt = orig_layer_cls(**kwargs, bias=self.bias is not None)
    rt.weight = self.weight
    rt.bias = self.bias
    return rt


def _lsq_to_quant(self, orig_layer_cls, quant_layer_cls):
    # Extract kwargs from the original layer
    kwargs = {key: getattr(self, key) for key in ["in_features", "out_features"]}
    input_info = (
        self.input_info
        if not self.online
        else calibrate(torch.empty(1), self.input_spec)
    )
    rt = quant_layer_cls(
        self.weight_info.spec,
        input_info,
        orig_layer_cls(**kwargs, bias=self.bias is not None),
    )
    rt.weight_info = self.weight_info
    rt.weight = torch.nn.Parameter(
        quantize(self.weight, self.weight_info), requires_grad=False
    )
    if self.bias is not None:
        rt.bias = torch.nn.Parameter(self.bias.detach().clone(), requires_grad=False)
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
            ["in_features", "out_features"],
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
        ["in_features", "out_features"],
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
        bn_track_running_stats: bool = True,
        use_fast_bn_path=False,
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
        self.bn_track_running_stats = bn_track_running_stats
        self.use_fast_bn_path = use_fast_bn_path

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
        # similar to the default path of https://github.com/pytorch/pytorch/blob/v2.7.0/torch/ao/nn/intrinsic/qat/modules/conv_fused.py
        if self.training and self.bn_track_running_stats:

            if self.use_fast_bn_path:
                if self.online:
                    info = calibrate(x, self.input_spec)
                    x = LSQQuantize.apply(x, info.scale, info.zero_point, info)
                else:
                    x = LSQQuantize.apply(
                        x,
                        self.input_info.scale,
                        self.input_info.zero_point,
                        self.input_info,
                    )
                conv = self.conv
                bn = self.bn
                w = conv.weight
                eps = bn.eps
                w_scale = bn.weight / torch.sqrt(bn.running_var + eps)
                # shape [out_channels]
                # we simulate qat forward with running mean/var (as those will be used during inference)
                w_simulated = w * w_scale.reshape(-1, 1, 1, 1)
                w_quant = LSQQuantize.apply(
                    w_simulated,
                    self.weight_info.scale,
                    self.weight_info.zero_point,
                    self.weight_info,
                )  # quantize the simulated weight
                conv_out = nn.functional.conv2d(
                    x,
                    w_quant,
                    bias=None,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                )
                conv_orig_simulated = conv_out / w_scale.reshape(
                    1, -1, 1, 1
                )  # simulate the conv output with the original weight
                if conv.bias is not None:
                    conv_orig_simulated = conv_orig_simulated + conv.bias.reshape(
                        1, -1, 1, 1
                    )
                # now we apply the batch norm
                return self.bn(
                    conv_orig_simulated,
                )
            else:
                # first, update bn
                if self.online:
                    info = calibrate(x, self.input_spec)
                    x = LSQQuantize.apply(x, info.scale, info.zero_point, info)
                else:
                    x = LSQQuantize.apply(
                        x,
                        self.input_info.scale,
                        self.input_info.zero_point,
                        self.input_info,
                    )
                conv_out_float_weight = torch.nn.functional.conv2d(
                    x,
                    self.conv.weight,
                    bias=None,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups,
                )
                with torch.no_grad():
                    if self.conv.bias is not None:
                        conv_out_for_bn = (
                            self.conv.bias.reshape(1, -1, 1, 1) + conv_out_float_weight
                        )
                    else:
                        conv_out_for_bn = conv_out_float_weight
                    # update batch norm stats
                    self.bn(conv_out_for_bn)

                # pass with running var
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                w_scale = self.bn.weight / running_std  # shape [out_channels]
                conv_out = nn.functional.conv2d(
                    x,
                    self.conv.weight * w_scale.reshape(-1, 1, 1, 1),
                    bias=None,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups,
                )
                # compute BATCH statistics
                batch_mean = conv_out_float_weight.mean(dim=(0, 2, 3))
                batch_var = conv_out_float_weight.var(dim=(0, 2, 3), unbiased=False)
                batch_std = torch.sqrt(batch_var + self.bn.eps)
                scale_to_batch = running_std / batch_std
                conv_out_rescaled = conv_out * scale_to_batch.reshape(1, -1, 1, 1)
                # bias
                fused_bias = (
                    self.bn.bias - self.bn.weight * batch_mean / batch_std
                ).reshape(1, -1, 1, 1)
                # now we apply the batch norm
                return conv_out_rescaled + fused_bias
        else:
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
            f"{self.conv.in_channels}, {self.conv.out_channels}, {self.conv.kernel_size}, "
            f"{self.conv.stride}, {self.conv.padding}, {self.conv.dilation}, {self.conv.groups})"
        )


def qat_freeze_bn_running_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (FusedLSQConv2dBatchNorm2d, FusedQATConv2dBatchNorm2d)):
            module.bn_track_running_stats = False


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
            return torch.tensor(0.0).to(next(model.parameters()).device)

    def pact_regularizer():
        return sum(layer.alpha**2 for layer in pact_layers)

    return pact_regularizer
