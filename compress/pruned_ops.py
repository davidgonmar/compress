import torch
from torch import nn
import torch.nn.functional as F
from compress.pruning_strats import PruningGranularity


def _get_mask_from_ratio(
    weight: torch.Tensor, ratio_to_keep: float, granularity_cls: PruningGranularity
):
    assert 0.0 <= ratio_to_keep <= 1.0
    granularity = granularity_cls()
    wabs = weight.abs()
    w_reshaped = granularity.transform(wabs)  # shape (n_groups, m_elements_per_group)
    num_to_keep = int(
        ratio_to_keep * w_reshaped.shape[0]
    )  # keep only selected percentage of groups
    _, indices = torch.topk(
        w_reshaped.mean(1, keepdim=False), k=num_to_keep, largest=True, dim=0
    )  # shape (n_groups_to_keep), selected by mean
    mask = torch.zeros_like(w_reshaped)
    mask[indices] = 1
    return granularity.untransform(mask)


class PrunedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PrunedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, input):
        pruned_weight = self.weight * self.mask
        return F.linear(input, pruned_weight, self.bias)

    @staticmethod
    def from_linear(linear, granularity_cls: PruningGranularity, ratio_to_keep=1.0):
        pruned_linear = PrunedLinear(
            linear.in_features, linear.out_features, linear.bias is not None
        ).to(linear.weight.device)
        pruned_linear.weight.data = linear.weight.data.clone()
        pruned_linear.mask.data = _get_mask_from_ratio(
            pruned_linear.weight, ratio_to_keep, granularity_cls
        )
        if linear.bias is not None:
            pruned_linear.bias.data = linear.bias.data.clone()
        return pruned_linear


class PrunedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(PrunedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, input):
        pruned_weight = self.weight * self.mask
        return F.conv2d(
            input,
            pruned_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @staticmethod
    def from_conv2d(conv2d, granularity_cls: PruningGranularity, ratio_to_keep=1.0):
        pruned_conv2d = PrunedConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size[0],
            conv2d.stride[0],
            conv2d.padding[0],
            conv2d.dilation[0],
            conv2d.groups,
            conv2d.bias is not None,
        ).to(conv2d.weight.device)
        pruned_conv2d.weight.data = conv2d.weight.data.clone()
        pruned_conv2d.mask.data = _get_mask_from_ratio(
            pruned_conv2d.weight, ratio_to_keep, granularity_cls
        )
        if conv2d.bias is not None:
            pruned_conv2d.bias.data = conv2d.bias.data.clone()
        return pruned_conv2d
