import torch
from torch import nn
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured
from torch.nn.modules.utils import _pair
from compress.layer_fusion import get_new_params


class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn((out_features, in_features), dtype=torch.float32)
        )
        self.bias = (
            nn.Parameter(torch.randn((out_features), dtype=torch.float32))
            if bias
            else None
        )
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
                pruned_linear.bias_mask.data = bias_mask.to(linear.bias.dtype)
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
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(
                (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
                dtype=torch.float32,
            )
        )
        self.bias = (
            nn.Parameter(torch.randn((out_channels), dtype=torch.float32))
            if bias
            else None
        )
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
        return self.in_channels * self.out_channels * self.kernel_size[
            0
        ] * self.kernel_size[1] + (self.out_channels if self.bias is not None else 0)


class SparseFusedConv2dBatchNorm2d(nn.Module):

    def __init__(self, conv_params: dict, bn_params: dict):
        super().__init__()

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
            torch.ones(self.conv.out_channels, dtype=self.conv.weight.dtype),
        )

        self.conv_params = {
            "stride": conv_params["stride"],
            "padding": conv_params["padding"],
            "dilation": conv_params["dilation"],
            "groups": conv_params["groups"],
        }

    @classmethod
    def from_conv_bn(
        cls,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        weight_mask: torch.Tensor | None = None,
        bias_mask: torch.Tensor | None = None,
    ) -> "SparseFusedConv2dBatchNorm2d":
        """Build a sparse‑aware block that behaves exactly like `conv+bn`."""
        obj = cls(
            conv_params={
                "in_channels": conv.in_channels,
                "out_channels": conv.out_channels,
                "kernel_size": conv.kernel_size,
                "stride": conv.stride,
                "padding": conv.padding,
                "dilation": conv.dilation,
                "groups": conv.groups,
            },
            bn_params={
                "eps": bn.eps,
                "momentum": bn.momentum,
            },
        )

        assert bn.track_running_stats, "BatchNorm must track running stats"
        assert bn.affine, "BatchNorm must be affine"

        obj.conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            obj.conv.bias.data.copy_(conv.bias.data)
        else:
            obj.conv.bias = None

        obj.bn.weight.data.copy_(bn.weight.data)
        obj.bn.bias.data.copy_(bn.bias.data)
        obj.bn.running_mean.data.copy_(bn.running_mean.data)
        obj.bn.running_var.data.copy_(bn.running_var.data)
        obj.bn.num_batches_tracked.data.copy_(bn.num_batches_tracked.data)

        if weight_mask is not None:
            obj.weight_mask.data.copy_(weight_mask.to(obj.weight_mask.dtype))
        if bias_mask is not None:
            obj.bias_mask.data.copy_(bias_mask.to(obj.bias_mask.dtype))

        return obj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # update running stats
            w_masked = self.conv.weight * self.weight_mask
            b_masked = (
                self.conv.bias * self.bias_mask if self.conv.bias is not None else None
            )
            _x = F.conv2d(
                x,
                w_masked,
                b_masked,
                stride=self.conv_params["stride"],
                padding=self.conv_params["padding"],
                dilation=self.conv_params["dilation"],
                groups=self.conv_params["groups"],
            )
            self.bn(_x)

        w_fused, b_fused = get_new_params(self.conv, self.bn)
        w_fused.retain_grad()
        b_fused.retain_grad()
        self._cached_weight = w_fused
        self._cached_bias = b_fused
        w_pruned = w_fused * self.weight_mask
        b_pruned = b_fused * self.bias_mask
        return F.conv2d(
            x,
            w_pruned,
            b_pruned,
            stride=self.conv_params["stride"],
            padding=self.conv_params["padding"],
            dilation=self.conv_params["dilation"],
            groups=self.conv_params["groups"],
        )

    def get_weight(self) -> torch.Tensor:
        w, _ = get_new_params(self.conv, self.bn)
        return w * self.weight_mask

    def get_bias(self) -> torch.Tensor:
        _, b = get_new_params(self.conv, self.bn)
        return b * self.bias_mask

    @property
    def weight(self) -> torch.Tensor:
        return self.get_weight()

    @property
    def bias(self) -> torch.Tensor:
        return self.get_bias()

    # ------------------------------------------------------------------------- #
    def nonzero_params(self) -> int:
        """Number of *stored* parameters that are still non‑zero after pruning."""
        return int(
            torch.count_nonzero(self.get_weight())
            + torch.count_nonzero(self.get_bias())
        )

    def total_params(self) -> int:
        """Total parameters before pruning (Conv + BN)."""
        return self.conv.weight.numel() + self.bn.bias.numel()

    # ------------------------------------------------------------------------- #
    def to_conv2d(self) -> nn.Conv2d:
        """Export a plain Conv2d containing the fused, pruned weights & bias."""
        fused_w, fused_b = self.get_weight(), self.get_bias()

        conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True,
            device=fused_w.device,
            dtype=fused_w.dtype,
        )
        conv.weight.data.copy_(fused_w)
        conv.bias.data.copy_(fused_b)
        return conv

    # conv
    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def groups(self):
        return self.conv.groups

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation

    # bn
    @property
    def eps(self):
        return self.bn.eps

    @property
    def momentum(self):
        return self.bn.momentum

    @property
    def affine(self):
        return self.bn.affine

    @property
    def track_running_stats(self):
        return self.bn.track_running_stats
