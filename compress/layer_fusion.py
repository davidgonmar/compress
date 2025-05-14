from compress.common import (
    gather_submodules,
    keys_passlist_should_do,
)
import copy
import torch.nn as nn
from utils import unzip
from typing import List, Tuple, Callable
import torch

def fuse_conv_bn(
    model: nn.Module,
    fuse_bn_keys: List[Tuple[str, str]],
    fuse_impl: Callable[[nn.Module, nn.Module, str, str], nn.Module],
    inplace=True,
):
    if not inplace:
        model = copy.deepcopy(model)

    # each element in the key is a tuple of (conv_key, bn_key)
    conv_keys, bn_keys = unzip(fuse_bn_keys)

    # gather all conv and bn layers
    convs = gather_submodules(
        model,
        should_do=keys_passlist_should_do(conv_keys),
    )
    bns = gather_submodules(
        model,
        should_do=keys_passlist_should_do(bn_keys),
    )

    # create a dict with the layer type as key and the input and weight specs as values
    convs_dict = {name: module for name, module in convs}

    bns_dict = {name: module for name, module in bns}

    # fuse the conv and bn layers
    for conv_key, bn_key in fuse_bn_keys:
        if conv_key not in convs_dict:
            raise KeyError(f"Conv layer {conv_key} not found in model")
        if bn_key not in bns_dict:
            raise KeyError(f"BN layer {bn_key} not found in model")

        conv = convs_dict[conv_key]
        bn = bns_dict[bn_key]

        fused_conv = fuse_impl(conv, bn)

        parent_module = model
        *parent_path, attr_name = conv_key.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(parent_module, attr_name, fused_conv)

        # remove the bn layer
        parent_module = model
        *parent_path, attr_name = bn_key.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, attr_name, nn.Identity())

    return model

def get_new_params(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype)
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    inv_std = gamma / torch.sqrt(running_var + eps)  # shape [out_channels]
    w = w * inv_std.reshape(-1, 1, 1, 1)
    b = beta + (b - running_mean) * inv_std
    return w, b

def fuse_batch_norm_inference(conv: nn.Conv2d, bn: nn.BatchNorm2d, conv_name: str, bn_name: str) -> nn.Conv2d:
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"Expected nn.Conv2d, got {type(conv)}")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"Expected nn.BatchNorm2d, got {type(bn)}")
    
    # assert eval mode
    assert bn.training is False, "BatchNorm must be in eval mode"
    assert bn.track_running_stats is True, "BatchNorm must be tracking running stats"

    w, b = get_new_params(conv, bn)

    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        device=w.device,
        dtype=w.dtype,
    )

    fused_conv.weight.data.copy_(w.detach())
    fused_conv.bias.data.copy_(b.detach())
    return fused_conv
