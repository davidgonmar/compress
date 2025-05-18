import torch
from torch import nn
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured
from compress.layer_fusion import get_new_params


class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.ones_like(self.bias))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pruned_weight = self.weight * self.weight_mask
        pruned_bias = self.bias * self.bias_mask if self.bias is not None else None
        return F.linear(input, pruned_weight, pruned_bias)

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        weight_mask: torch.Tensor | None = None,
        bias_mask: torch.Tensor | None = None,
    ) -> "SparseLinear":
        pruned_linear = SparseLinear(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        ).to(linear.weight.device)
        pruned_linear.weight.data = linear.weight.data.clone()
        if weight_mask is not None:
            pruned_linear.weight_mask.data = weight_mask.to(linear.weight.dtype)
        if linear.bias is not None:
            pruned_linear.bias.data = linear.bias.data.clone()
            if bias_mask is not None:
                pruned_linear.bias_mask.data = bias_mask.to(
                    linear.bias.dtype
                )  # else, it will be all 1s
        return pruned_linear

    def to_sparse_semi_structured(self):
        w = self.weight * self.weight_mask
        linear = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )
        linear.weight = nn.Parameter(to_sparse_semi_structured(w))
        linear.bias = self.bias * self.bias_mask if self.bias is not None else None
        return linear

    def nonzero_params(self):
        weight_multed = self.weight * self.weight_mask
        bias_multed = (
            self.bias * self.bias_mask if self.bias is not None else torch.tensor(0)
        )
        return (
            torch.count_nonzero(weight_multed).item()
            + torch.count_nonzero(bias_multed).item()
        )

    def to_linear(self):
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        ).to(self.weight.device)
        pruned_weight = self.weight * self.weight_mask
        linear.weight = nn.Parameter(pruned_weight.clone())
        if self.bias is not None:
            pruned_bias = self.bias * self.bias_mask
            linear.bias = nn.Parameter(pruned_bias.data.clone())
        return linear

    def get_weight(self):
        return self.weight * self.weight_mask

    def get_bias(self):
        return self.bias * self.bias_mask if self.bias is not None else None

    def total_params(self):
        return self.in_features * self.out_features + (
            self.out_features if self.bias is not None else 0
        )
    
class SparseConv2d(nn.Module):
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
        super(SparseConv2d, self).__init__()
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
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.ones_like(self.bias))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pruned_weight = self.weight * self.weight_mask
        pruned_bias = self.bias * self.bias_mask if self.bias is not None else None
        return F.conv2d(
            input,
            pruned_weight,
            pruned_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @staticmethod
    def from_conv2d(
        conv2d,
        weight_mask: torch.Tensor | None = None,
        bias_mask: torch.Tensor | None = None,
    ) -> "SparseConv2d":
        pruned_conv = SparseConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            conv2d.bias is not None,
        ).to(conv2d.weight.device)
        pruned_conv.weight.data = conv2d.weight.data.clone()
        if weight_mask is not None:
            pruned_conv.weight_mask.data = weight_mask.to(conv2d.weight.dtype)
        if conv2d.bias is not None:
            pruned_conv.bias.data = conv2d.bias.data.clone()
            if bias_mask is not None:
                pruned_conv.bias_mask.data = bias_mask.to(conv2d.bias.dtype)
        return pruned_conv

    def nonzero_params(self):
        weight_multed = self.weight * self.weight_mask
        bias_multed = (
            self.bias * self.bias_mask if self.bias is not None else torch.tensor(0)
        )
        return (
            torch.count_nonzero(weight_multed).item()
            + torch.count_nonzero(bias_multed).item()
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
            bias=self.bias is not None,
        ).to(self.weight.device)
        pruned_weight = self.weight * self.weight_mask
        conv.weight = nn.Parameter(pruned_weight.clone())
        if self.bias is not None:
            conv.bias = nn.Parameter(self.bias.data.clone())
        return conv

    def get_weight(self):
        return self.weight * self.weight_mask

    def get_bias(self):
        return self.bias * self.bias_mask if self.bias is not None else None

    def total_params(self):
        return (
            self.in_channels
            * self.out_channels
            * self.kernel_size
            * self.kernel_size
            + (self.out_channels if self.bias is not None else 0)
        )

class SparseFusedConv2dBatchNorm2d(nn.Module):
    def __init__(
        self,
        conv_params: dict,
        bn_params: dict,
    ):
        super(SparseFusedConv2dBatchNorm2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=conv_params["in_channels"],
            out_channels=conv_params["out_channels"],
            kernel_size=conv_params["kernel_size"],
            stride=conv_params["stride"],
            padding=conv_params["padding"],
            dilation=conv_params["dilation"],
            groups=conv_params["groups"],
            bias=True,
        )

        self.bn = nn.BatchNorm2d(
            num_features=conv_params["out_channels"],
            eps=bn_params["eps"],
            momentum=bn_params["momentum"],
            affine=True,
            track_running_stats=True,
        )

        self.register_buffer("weight_mask", torch.ones_like(self.conv.weight))
        self.register_buffer(
            "bias_mask",
            torch.ones(self.conv.weight.shape[0])
            .to(self.conv.weight.device)
            .to(self.conv.weight.dtype),
        )
        self.conv_params = conv_params
        self.bn_params = bn_params

    @staticmethod
    def from_conv_bn(
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        weight_mask: torch.Tensor | None = None,
        bias_mask: torch.Tensor | None = None,
    ):
        pruned_conv = SparseFusedConv2dBatchNorm2d(
            {
                "in_channels": conv.in_channels,
                "out_channels": conv.out_channels,
                "kernel_size": conv.kernel_size,
                "stride": conv.stride,
                "padding": conv.padding,
                "dilation": conv.dilation,
                "groups": conv.groups,
            },
            {
                "eps": bn.eps,
                "momentum": bn.momentum,
            },
        )
        pruned_conv.conv.weight.data = conv.weight.data.clone()
        pruned_conv.bn.weight.data = bn.weight.data.clone()
        pruned_conv.bn.bias.data = bn.bias.data.clone()
        if weight_mask is not None:
            pruned_conv.weight_mask.data = weight_mask.to(conv.weight.dtype).to(
                conv.weight.device)
        if bias_mask is not None:
            pruned_conv.bias_mask.data = bias_mask.to(conv.weight.dtype).to(
                conv.weight.device
            )
        return pruned_conv

    def forward(self, x):
        w, b = get_new_params(self.conv, self.bn)
        pruned_weight = w * self.weight_mask
        pruned_bias = b * self.bias_mask

        x = F.conv2d(
            x,
            pruned_weight,
            pruned_bias,
            self.conv_params["stride"],
            self.conv_params["padding"],
            self.conv_params["dilation"],
            self.conv_params["groups"],
        )

        return x

    def nonzero_params(self):
        w, b = get_new_params(self.conv, self.bn)

        weight_multed = w * self.weight_mask
        bias_multed = b * self.bias_mask

        return (
            torch.count_nonzero(weight_multed).item()
            + torch.count_nonzero(bias_multed).item()
        )

    def to_conv2d(self):
        conv = nn.Conv2d(
            self.conv_params["in_channels"],
            self.conv_params["out_channels"],
            self.conv_params["kernel_size"],
            stride=self.conv_params["stride"],
            padding=self.conv_params["padding"],
            dilation=self.conv_params["dilation"],
            groups=self.conv_params["groups"],
            bias=True,
        ).to(self.conv.weight.device)
        w, b = get_new_params(self.conv, self.bn)
        pruned_weight = w * self.weight_mask
        pruned_bias = b * self.bias_mask
        conv.weight = nn.Parameter(pruned_weight.clone())
        conv.bias = nn.Parameter(pruned_bias.data.clone())
        return conv

    def get_weight(self):
        w, _ = get_new_params(self.conv, self.bn)
        return w * self.weight_mask

    def get_bias(self):
        _, b = get_new_params(self.conv, self.bn)
        return b * self.bias_mask

    def total_params(self):
        return self.conv.weight.numel() + self.bn.bias.numel()
    
    @property
    def weight(self):
        return self.get_weight()
    
    @property
    def bias(self):
        return self.get_bias()
    
