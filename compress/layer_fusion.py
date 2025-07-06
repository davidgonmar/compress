from compress.utils import (
    gather_submodules,
    keys_passlist_should_do,
)
import copy
import torch.nn as nn
from compress.utils import unzip
from typing import List, Tuple, Callable
import torch
from compress.factorization.low_rank_ops import LowRankConv2d


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

        # if conv is a LowRankConv2d, then we need to fuse the internal conv and bn
        fused_conv = fuse_impl(conv, bn, conv_key, bn_key)

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
    b = (
        conv.bias
        if conv.bias is not None
        else torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype)
    )
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    inv_std = gamma / torch.sqrt(running_var + eps)  # shape [out_channels]
    w = w * inv_std.reshape(-1, 1, 1, 1)
    b = beta + (b - running_mean) * inv_std
    return w, b


def fuse_batch_norm_inference(
    conv: nn.Conv2d, bn: nn.BatchNorm2d, conv_name: str, bn_name: str
) -> nn.Conv2d:
    if not isinstance(conv, (nn.Conv2d, LowRankConv2d)):
        raise TypeError(f"Expected nn.Conv2d, got {type(conv)}")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"Expected nn.BatchNorm2d, got {type(bn)}")

    if isinstance(conv, LowRankConv2d):
        w_conv_fuse = conv.w1
        dummy_conv = nn.Conv2d(
            in_channels=conv.rank,
            out_channels=conv.out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            device=conv.w1.device,
            dtype=conv.w1.dtype,
        )

        dummy_conv.weight.data.copy_(w_conv_fuse.detach())
        if conv.bias is not None:
            dummy_conv.bias.data.copy_(conv.bias.detach())
        else:
            dummy_conv.bias.data.copy_(
                torch.zeros(
                    conv.out_channels, device=conv.w1.device, dtype=conv.w1.dtype
                )
            )
        w, b = get_new_params(dummy_conv, bn)

        if conv.bias is None:
            conv.bias = nn.Parameter(
                torch.empty(conv.out_channels, device=w.device, dtype=w.dtype)
            )

        conv.bias.data.copy_(b.detach())

        conv.w1.data.copy_(w.detach())

        return conv

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


class TrainableFusedConv2dBatchNorm2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ):
        super(TrainableFusedConv2dBatchNorm2d, self).__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"Expected nn.Conv2d, got {type(conv)}")
        if not isinstance(bn, nn.BatchNorm2d):
            raise TypeError(f"Expected nn.BatchNorm2d, got {type(bn)}")
        self.conv = conv
        self.bn = bn

    @property
    def weight(self):
        return get_new_params(self.conv, self.bn)[0]

    @property
    def bias(self):
        return get_new_params(self.conv, self.bn)[1]

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def fuse_batch_norm_training(
    conv: nn.Conv2d, bn: nn.BatchNorm2d, conv_name: str, bn_name: str
) -> TrainableFusedConv2dBatchNorm2d:
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"Expected nn.Conv2d, got {type(conv)}")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"Expected nn.BatchNorm2d, got {type(bn)}")

    return TrainableFusedConv2dBatchNorm2d(conv, bn)


resnet20_fuse_pairs = [
    # stem
    ("conv1", "bn1"),
    # layer1 – BasicBlock(0, 1, 2)
    ("layer1.0.conv1", "layer1.0.bn1"),
    ("layer1.0.conv2", "layer1.0.bn2"),
    ("layer1.1.conv1", "layer1.1.bn1"),
    ("layer1.1.conv2", "layer1.1.bn2"),
    ("layer1.2.conv1", "layer1.2.bn1"),
    ("layer1.2.conv2", "layer1.2.bn2"),
    # layer2 – BasicBlock(0, 1, 2)
    ("layer2.0.conv1", "layer2.0.bn1"),
    ("layer2.0.conv2", "layer2.0.bn2"),
    ("layer2.0.shortcut.0", "layer2.0.shortcut.1"),  # down‑sample path
    ("layer2.1.conv1", "layer2.1.bn1"),
    ("layer2.1.conv2", "layer2.1.bn2"),
    ("layer2.2.conv1", "layer2.2.bn1"),
    ("layer2.2.conv2", "layer2.2.bn2"),
    # layer3 – BasicBlock(0, 1, 2)
    ("layer3.0.conv1", "layer3.0.bn1"),
    ("layer3.0.conv2", "layer3.0.bn2"),
    ("layer3.0.shortcut.0", "layer3.0.shortcut.1"),  # down‑sample path
    ("layer3.1.conv1", "layer3.1.bn1"),
    ("layer3.1.conv2", "layer3.1.bn2"),
    ("layer3.2.conv1", "layer3.2.bn1"),
    ("layer3.2.conv2", "layer3.2.bn2"),
]


def get_fuse_bn_keys(model_name: str):
    if model_name == "resnet20":
        return resnet20_fuse_pairs
    else:
        print("Model not supported for BN fusion")
        return []
