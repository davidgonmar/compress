import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super(LowRankLinear, self).__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, rank))
        self.w1 = nn.Parameter(torch.randn(rank, out_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    @staticmethod
    def from_linear(linear: nn.Linear, ratio_to_keep: float = 0.1):
        # Original linear -> O = X @ W
        # Low rank linear -> W = U @ S @ V_T -> O = X @ U @ S @ V_T
        W, b = linear.weight.T, linear.bias
        U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD
        orig_rank = min(S.shape)
        rank = max(int(orig_rank * ratio_to_keep), 1)
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
        # pad S to be {IN x OUT}
        in_f, out_f = W.shape
        assert S.shape == (rank, rank)
        assert U.shape == (in_f, in_f)
        assert V_T.shape == (out_f, out_f)
        W0 = U[:, :rank] @ S  # in R^{IN x RANK}
        W1 = V_T[:rank, :]  # in R^{RANK x OUT}
        low_rank_linear = LowRankLinear(
            linear.weight.shape[1],
            linear.weight.shape[0],
            rank,
            bias=linear.bias is not None,
        )
        low_rank_linear.w0.data = W0
        low_rank_linear.w1.data = W1
        if b is not None and linear.bias is not None:
            low_rank_linear.bias.data = b
        else:
            low_rank_linear.bias = None
        return low_rank_linear

    def forward(self, x: torch.Tensor):
        # X in R^{... x IN}
        # W0 in R^{IN x RANK} -> X @ W0 in R^{... x RANK}
        # W1 in R^{RANK x OUT} -> O = (X @ W0) @ W1 in R^{... x OUT}
        if self.bias is not None:
            return torch.matmul(torch.matmul(x, self.w0), self.w1) + self.bias
        else:
            return torch.matmul(torch.matmul(x, self.w0), self.w1)

    def __repr__(self):
        return f"LowRankLinear(in_features={self.w0.shape[0]}, out_features={self.w1.shape[1]}, rank={self.w0.shape[1]}, bias={self.bias is not None})"


class LowRankConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(LowRankConv2d, self).__init__()
        self.w0 = nn.Parameter(torch.randn(rank, in_channels, kernel_size, kernel_size))
        self.w1 = nn.Parameter(torch.randn(out_channels, rank, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    @staticmethod
    def from_conv2d(conv2d: nn.Conv2d, ratio_to_keep: float = 0.1):
        # Original conv2d -> O = conv2d(W, X)
        # Low rank conv2d -> O = conv2d(W1, conv2d(W0, X))
        W, b = conv2d.weight, conv2d.bias
        o, i, h, w = W.shape
        U, S, V_T = torch.linalg.svd(
            W.permute(1, 2, 3, 0).reshape(i * h * w, o), full_matrices=True
        )
        orig_rank = min(S.shape)
        rank = max(int(orig_rank * ratio_to_keep), 1)
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
        W0 = (
            (U[:, :rank] @ S).reshape(i, h, w, rank).permute(3, 0, 1, 2)
        )  # shape = (rank, i, h, w)
        W1 = (
            V_T[:rank, :].reshape(rank, o, 1, 1).permute(1, 0, 2, 3)
        )  # shape = (o, rank, 1, 1)
        low_rank_conv2d = LowRankConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size[0],
            rank,
            stride=conv2d.stride[0],
            padding=conv2d.padding[0],
            dilation=conv2d.dilation[0],
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
        )
        low_rank_conv2d.w0.data = W0
        low_rank_conv2d.w1.data = W1

        if b is not None and conv2d.bias is not None:
            low_rank_conv2d.bias.data = b
        else:
            low_rank_conv2d.bias = None

        return low_rank_conv2d

    def forward(self, x: torch.Tensor):
        conv_out = torch.nn.functional.conv2d(
            x,
            self.w0,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )  # shape (batch, rank, h_out, w_out)
        h_out, w_out = conv_out.shape[2], conv_out.shape[3]
        linear_out = torch.nn.functional.linear(
            conv_out.permute(0, 2, 3, 1).reshape(-1, self.rank),
            self.w1.reshape(self.out_channels, self.rank),
            bias=self.bias,
        )  # shape (batch * h_out * w_out, out_channels)
        return linear_out.reshape(-1, h_out, w_out, self.out_channels).permute(
            0, 3, 1, 2
        )
