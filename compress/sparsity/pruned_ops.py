import torch
from torch import nn
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured


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
    def from_linear(
        linear,
        mask,
    ):
        pruned_linear = PrunedLinear(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        ).to(linear.weight.device)
        pruned_linear.weight.data = linear.weight.data.clone()
        pruned_linear.mask.data = mask.to(linear.weight.dtype)
        if linear.bias is not None:
            pruned_linear.bias.data = linear.bias.data.clone()
        return pruned_linear

    def to_sparse_semi_structured(self):
        w = self.weight * self.mask
        linear = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )
        linear.weight = nn.Parameter(to_sparse_semi_structured(w))
        linear.bias = self.bias
        return linear

    def nonzero_params(self):
        multed = self.weight * self.mask
        return torch.count_nonzero(multed).item() + (
            self.bias.numel() if self.bias is not None else 0
        )

    def to_linear(self):
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None
        ).to(self.weight.device)
        pruned_weight = self.weight * self.mask
        linear.weight = nn.Parameter(pruned_weight.clone())
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.data.clone())
        return linear


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
    def from_conv2d(
        conv2d,
        mask,
    ):
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
        pruned_conv2d.mask.data = mask.to(conv2d.weight.dtype)
        if conv2d.bias is not None:
            pruned_conv2d.bias.data = conv2d.bias.data.clone()
        return pruned_conv2d

    def nonzero_params(self):
        multed = self.weight * self.mask
        return torch.count_nonzero(multed).item() + (
            self.bias.numel() if self.bias is not None else 0
        )

    def to_conv2d(self):
        conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None
        ).to(self.weight.device)
        pruned_weight = self.weight * self.mask
        conv.weight = nn.Parameter(pruned_weight.clone())
        if self.bias is not None:
            conv.bias = nn.Parameter(self.bias.data.clone())
        return conv

